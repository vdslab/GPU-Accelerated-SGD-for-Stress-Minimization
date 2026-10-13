use crate::graph;
use anyhow::Result;
use experiment_common::OutputFormat;
use metal::*;
use std::mem;
use std::time::{Duration, Instant};

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
    pub device_name: String,
}

#[derive(Debug)]
pub struct MetalRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f32; 2]>,
    pub method_setup_time: Duration,
    pub upload_time: Duration,
    pub iteration_time: Duration,
    pub readback_time: Duration,
    pub attempted_updates: u64,
    pub completed_updates: u64,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;

        let device_name = device.name().to_owned();
        let command_queue = device.new_command_queue();

        // Load and compile shader
        let shader_source = include_str!("shader.metal");
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_source, &compile_options)
            .map_err(|e| anyhow::anyhow!("Failed to compile shader: {}", e))?;

        let kernel = library
            .get_function("sgd", None)
            .map_err(|e| anyhow::anyhow!("Failed to get kernel function: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;

        Ok(MetalContext {
            device,
            command_queue,
            pipeline,
            device_name,
        })
    }

    pub fn execute_sgd(
        &self,
        params: graph::SgdParams,
        verbose: bool,
        output_format: OutputFormat,
    ) -> Result<MetalRunResult> {
        let method_setup_started = Instant::now();
        let initial_positions = params.positions.clone();
        let gpu_etas: Vec<f32> = params.etas.into_iter().map(|e| e as f32).collect();
        let gpu_positions: Vec<[f32; 2]> = params
            .positions
            .into_iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
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
        let mut positions_flat: Vec<f32> = gpu_positions
            .iter()
            .flat_map(|p| vec![p[0], p[1]])
            .collect();
        let locks: Vec<u32> = vec![0; node_size];
        let updated_pairs: Vec<u32> = vec![0; num_pairs];
        let updated_count: Vec<u32> = vec![0];
        let positions_before: Vec<[f32; 4]> = vec![[0.0; 4]; num_pairs];
        let method_setup_time = method_setup_started.elapsed();

        let upload_started = Instant::now();
        let etas_buffer = self.device.new_buffer_with_data(
            gpu_etas.as_ptr() as *const _,
            (gpu_etas.len() * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
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
        let lock_buffer = self.device.new_buffer_with_data(
            locks.as_ptr() as *const _,
            (locks.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let updated_pairs_buffer = self.device.new_buffer_with_data(
            updated_pairs.as_ptr() as *const _,
            (updated_pairs.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let updated_count_buffer = self.device.new_buffer_with_data(
            updated_count.as_ptr() as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
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
        let upload_time = upload_started.elapsed();

        let iteration_start = Instant::now();
        let mut completed_updates = 0_u64;
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
                width: 32, // Match WGSL @workgroup_size(32,1,1)
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
            completed_updates += u64::from(updated_count_val);
            if verbose {
                let message = format!(
                    "Iteration {} - Updated {} pairs",
                    iteration + 1,
                    updated_count_val
                );
                match output_format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
        }
        let iteration_time = iteration_start.elapsed();

        let readback_started = Instant::now();
        unsafe {
            let ptr = positions_buffer.contents() as *const f32;
            positions_flat = std::slice::from_raw_parts(ptr, positions_flat.len()).to_vec();
        }

        let final_positions: Vec<[f32; 2]> = positions_flat
            .chunks(2)
            .map(|chunk| [chunk[0], chunk[1]])
            .collect();
        let readback_time = readback_started.elapsed();

        Ok(MetalRunResult {
            initial_positions,
            positions: final_positions,
            method_setup_time,
            upload_time,
            iteration_time,
            readback_time,
            attempted_updates: (num_pairs as u64) * (num_iterations as u64),
            completed_updates,
        })
    }
}
