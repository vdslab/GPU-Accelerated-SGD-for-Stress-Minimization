use crate::graph;
use anyhow::Result;
use metal::*;
use std::mem;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuEdgeInfo {
    pub u: u32,
    pub v: u32,
    pub dij: f32,
    pub wij: f32,
}

#[derive(Debug)]
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        
        println!("Using Metal device: {}", device.name());
        
        println!("device maxThreadsPerThreadgroup: {}", device.max_threads_per_threadgroup().width);
        let command_queue = device.new_command_queue();
        
        // Load and compile shader
        let shader_source = include_str!("shader.metal");
        let compile_options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &compile_options)
            .map_err(|e| anyhow::anyhow!("Failed to compile shader: {}", e))?;
        
        let kernel = library.get_function("sgd", None)
            .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;
        
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;
        
        Ok(MetalContext {
            device,
            command_queue,
            pipeline,
        })
    }
    
    pub fn execute_sgd(
        &self,
        params: graph::SgdParams,
    ) -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>)> {
        let gpu_etas: Vec<f32> = params.etas.into_iter().map(|e| e as f32).collect();
        let gpu_positions: Vec<[f32; 2]> = params
            .positions
            .into_iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let initial_positions = gpu_positions.clone();
        
        let gpu_pairs: Vec<GpuEdgeInfo> = params
            .pairs
            .into_iter()
            .map(|p| GpuEdgeInfo {
                u: p.u as u32,
                v: p.v as u32,
                dij: p.dij as f32,
                wij: p.wij as f32,
            })
            .collect();
        
        let node_size = gpu_positions.len();
        let num_iterations = gpu_etas.len();
        let num_pairs = gpu_pairs.len();
        
        println!("Setting up Metal buffers...");
        println!("  Nodes: {}, Pairs: {}, Iterations: {}", node_size, num_pairs, num_iterations);
        
        // Create buffers
        let etas_buffer = self.device.new_buffer_with_data(
            gpu_etas.as_ptr() as *const _,
            (gpu_etas.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Positions buffer - flattened to allow atomic operations
        let mut positions_flat: Vec<f32> = gpu_positions.iter()
            .flat_map(|p| vec![p[0], p[1]])
            .collect();
        
        let positions_buffer = self.device.new_buffer_with_data(
            positions_flat.as_ptr() as *const _,
            (positions_flat.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let pairs_buffer = self.device.new_buffer_with_data(
            gpu_pairs.as_ptr() as *const _,
            (gpu_pairs.len() * mem::size_of::<GpuEdgeInfo>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Lock buffer (initialized to 0)
        let locks: Vec<u32> = vec![0; node_size];
        let lock_buffer = self.device.new_buffer_with_data(
            locks.as_ptr() as *const _,
            (locks.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Updated pairs tracking
        let updated_pairs: Vec<u32> = vec![0; num_pairs];
        let updated_pairs_buffer = self.device.new_buffer_with_data(
            updated_pairs.as_ptr() as *const _,
            (updated_pairs.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let updated_count: Vec<u32> = vec![0];
        let updated_count_buffer = self.device.new_buffer_with_data(
            updated_count.as_ptr() as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Positions before buffer
        let positions_before: Vec<[f32; 4]> = vec![[0.0; 4]; num_pairs];
        let positions_before_buffer = self.device.new_buffer_with_data(
            positions_before.as_ptr() as *const _,
            (positions_before.len() * mem::size_of::<[f32; 4]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Num pairs buffer (for bounds checking in shader)
        let num_pairs_buffer = self.device.new_buffer_with_data(
            &(num_pairs as u32) as *const _ as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        println!("Executing SGD iterations...");
        
        let iteration_start = std::time::Instant::now();
        
        // Execute iterations
        for iteration in 0..num_iterations {
            // Create iteration buffer for this iteration
            let iteration_buffer = self.device.new_buffer_with_data(
                &(iteration as u32) as *const _ as *const _,
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            
            // Reset updated count
            unsafe {
                let count_ptr = updated_count_buffer.contents() as *mut u32;
                *count_ptr = 0;
            }
            
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&etas_buffer), 0);
            encoder.set_buffer(1, Some(&positions_buffer), 0);
            encoder.set_buffer(2, Some(&pairs_buffer), 0);
            encoder.set_buffer(3, Some(&iteration_buffer), 0);
            encoder.set_buffer(4, Some(&lock_buffer), 0);
            encoder.set_buffer(5, Some(&updated_pairs_buffer), 0);
            encoder.set_buffer(6, Some(&updated_count_buffer), 0);
            encoder.set_buffer(7, Some(&positions_before_buffer), 0);
            encoder.set_buffer(8, Some(&num_pairs_buffer), 0);
            
            // Dispatch workgroups matching WGSL implementation:
            // @workgroup_size(32,1,1): Each workgroup = 32 threads (= 1 warp)
            // Each workgroup processes one pair (only thread 0 does work)
            // Use 2D dispatch to handle more pairs (up to 65535 * 65535)
            let max_x = 65535u64;
            let workgroup_count_x = (num_pairs as u64).min(max_x);
            let workgroup_count_y = ((num_pairs as u64) + max_x - 1) / max_x;
            
            let threadgroups = MTLSize {
                width: workgroup_count_x,
                height: workgroup_count_y,
                depth: 1,
            };
            
            let threads_per_threadgroup = MTLSize {
                width: 32,  // Match WGSL @workgroup_size(32,1,1)
                height: 1,
                depth: 1,
            };
            
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            // Read back updated count for every iteration
            let updated_count_val = unsafe {
                let ptr = updated_count_buffer.contents() as *const u32;
                *ptr
            };
            
            println!("Iteration {} - Updated {} pairs", iteration, updated_count_val);
        }
        
        let iteration_duration = iteration_start.elapsed();
        println!("\nSGD execution completed!");
        let per_iteration = iteration_duration.as_secs_f64() / num_iterations as f64;
        println!("\n=== Performance Summary ===");
        println!("Iterations total: {:.3}s", iteration_duration.as_secs_f64());
        println!("Per iteration:    {:.3}s ({:.1}ms)", per_iteration, per_iteration * 1000.0);
        
        // Read back final positions
        unsafe {
            let ptr = positions_buffer.contents() as *const f32;
            positions_flat = std::slice::from_raw_parts(ptr, positions_flat.len()).to_vec();
        }
        
        let final_positions: Vec<[f32; 2]> = positions_flat
            .chunks(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        
        Ok((initial_positions, final_positions))
    }
}

