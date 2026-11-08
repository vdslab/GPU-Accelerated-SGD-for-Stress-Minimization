//! Controlled tile-size diagnostics; production source is included unchanged.
#![allow(dead_code)]
#[path = "../src/graph.rs"]
mod graph;
mod gpu {
    include!("../src/gpu.rs");
    pub use timing::PassTimer;
    pub fn source(config: KernelConfig, capacity: u32) -> String {
        let source = shader_source(config);
        if config.memory_mode == MemoryMode::Shared {
            source.replace(
                &format!("array<vec2<f32>, {}>", config.tile_size),
                &format!("array<vec2<f32>, {}>", capacity),
            )
        } else {
            source
        }
    }
}
use anyhow::{ensure, Result};
use experiment_common::OutputFormat;
use rand::{seq::SliceRandom, SeedableRng};
use serde_json::json;
use std::path::Path;

fn copy_params(p: &graph::SgdParams) -> graph::SgdParams {
    graph::SgdParams {
        etas: p.etas.clone(),
        positions: p.positions.clone(),
        pairs: p.pairs.clone(),
        center: p.center,
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("micro") {
        return micro();
    }
    let input = args.get(1).expect("input.mtx [repetitions=3] [seed=10]");
    let reps: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let seed: u64 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(10);
    let graph = graph::Graph::from_mtx(Path::new(input))?;
    let params = graph.prepare_sgd_params(15, 0.1, true, seed);
    let mut context = gpu::GpuContext::new(gpu::KernelConfig::default())?;
    ensure!(context.timestamp_supported, "GPU timestamps required");
    let mut configs = Vec::new();
    for t in [64, 128, 256, 512, 1024] {
        for w in [32, 128, 256, 1024] {
            configs.push((t, w, t, gpu::MemoryMode::Shared));
        }
    }
    // Same update work/order, larger declared workgroup allocation only.
    for t in [128, 256] {
        configs.push((t, 1024, 1024, gpu::MemoryMode::Shared));
    }
    for t in [128, 256, 1024] {
        configs.push((t, 1024, t, gpu::MemoryMode::Global));
    }
    if std::env::var("TILE_CAUSE_FILTER").as_deref() == Ok("baseline") {
        configs.retain(|&(t, w, c, m)| w == 1024 && t == c && m == gpu::MemoryMode::Shared);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261107);
    let mut reference = std::collections::BTreeMap::new();
    for repetition in 0..=reps {
        configs.shuffle(&mut rng);
        for &(tile, workgroup, capacity, memory) in &configs {
            let config = gpu::KernelConfig {
                tile_size: tile,
                workgroup_size: workgroup,
                memory_mode: memory,
                ..gpu::KernelConfig::default()
            };
            context.kernel = config;
            context.module = context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("tile cause intervention"),
                    source: wgpu::ShaderSource::Wgsl(gpu::source(config, capacity).into()),
                });
            let blocks = (graph.node_size as u32).div_ceil(tile);
            let result = context.execute_sgd(
                copy_params(&params),
                &gpu::build_schedule(blocks),
                seed,
                false,
                OutputFormat::Json,
            )?;
            let positions64: Vec<_> = result
                .positions
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64])
                .collect();
            let hash = experiment_common::positions_sha256_f64(&positions64);
            if let Some(expected) = reference.get(&tile) {
                ensure!(
                    expected == &hash,
                    "coordinate mismatch for T{tile}/W{workgroup}/{memory:?}"
                );
            } else {
                reference.insert(tile, hash.clone());
            }
            println!(
                "{}",
                json!({"input": input, "nodes": graph.node_size,
                "seed": seed, "repetition": repetition, "warmup": repetition == 0,
                "tile": tile, "workgroup": workgroup, "capacity": capacity,
                "memory": memory.as_str(), "position_sha": hash,
                "wall_ms": result.iteration_time.as_secs_f64()*1000.0,
                "gpu_ms": result.gpu_device_time.map(|v| v.as_secs_f64()*1000.0),
                "dispatches": result.dispatches, "cross_workgroups": blocks/2,
                "adapter": context.adapter_name, "backend": context.backend})
            );
        }
    }
    Ok(())
}

// One cross round of the unchanged production arithmetic. Replicas have
// independent writable coordinates, and share a read-only all-ones distance
// matrix. Replication changes available work, not per-tile work or allocation.
// Matrix vs constant distance uses identical mathematical distances (d=1).
fn micro() -> Result<()> {
    use std::time::Instant;
    use wgpu::util::DeviceExt;
    let context = gpu::GpuContext::new(gpu::KernelConfig::default())?;
    ensure!(context.timestamp_supported, "GPU timestamps required");
    let device = &context.device;
    let queue = &context.queue;
    let n = 8192u32;
    let repeats = 8;
    let matrix = vec![1.0f32; (n * n) as usize];
    let distances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("shared read-only d=1 matrix"),
        contents: bytemuck::cast_slice(&matrix),
        usage: wgpu::BufferUsages::STORAGE,
    });
    drop(matrix);
    let initial: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            [
                (i as f32 * 0.01).sin() * 10.0,
                (i as f32 * 0.017).cos() * 10.0,
            ]
        })
        .collect();
    let mut configs = Vec::new();
    for tile in [128u32, 256, 1024] {
        for replicas in [1u32, 2, 4, 8, 16] {
            for mode in [0u32, 1, 2, 3] {
                configs.push((tile, replicas, mode));
            }
        }
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(72388);
    let mut references = std::collections::BTreeMap::new();
    for repetition in 0..=5 {
        configs.shuffle(&mut rng);
        for &(tile, replicas, mode) in &configs {
            let config = gpu::KernelConfig {
                tile_size: tile,
                ..gpu::KernelConfig::default()
            };
            let mut source = gpu::source(config, tile)
                .replace(
                    "let tile = tiles[wg.x];",
                    "let replica_base = wg.y * uniforms.n;\n    let tile = tiles[wg.x];",
                )
                .replace("positions[", "positions[replica_base + ");
            let distance_mode = match mode {
                0 => "matrix",
                1 => "constant",
                2 => "uniform",
                _ => "packed",
            };
            if mode == 3 {
                // Cross-only benchmark: same number of d=1 loads, contiguous
                // across active lanes instead of the production row stride.
                let split = source.find("// Circle method").unwrap();
                let (cross, self_part) = source.split_at(split);
                source = format!(
                    "{}{}",
                    cross.replace(
                        "dist_flat[gu * uniforms.n + gv]",
                        "dist_flat[(wg.x * BLOCK_T + offset) * BLOCK_T + local]"
                    ),
                    self_part
                );
            } else if mode != 0 {
                source = source.replace(
                    "dist_flat[gu * uniforms.n + gv]",
                    if mode == 1 {
                        "1.0"
                    } else {
                        "bitcast<f32>(uniforms.padding.x)"
                    },
                );
            }
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("independent replica cross tiles"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &(0..6)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if binding == 0 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: binding != 1,
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("sgd_rr"),
                compilation_options: Default::default(),
                cache: None,
            });
            let uniform = gpu::Uniforms {
                n,
                eta: 0.01,
                full_self_perm_offset: tile,
                partial_self_perm_offset: tile * 2 - 1,
                chunk: 0,
                chunks: 1,
                padding: [1.0f32.to_bits(), 0],
            };
            let positions_data = initial.repeat(replicas as usize);
            let make_buffer = |label, contents: &[u8], usage| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
            };
            let positions = make_buffer(
                "replica coordinates",
                bytemuck::cast_slice(&positions_data),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let uniforms = make_buffer(
                "uniform",
                bytemuck::bytes_of(&uniform),
                wgpu::BufferUsages::UNIFORM,
            );
            let lens = make_buffer(
                "lens",
                bytemuck::cast_slice(&vec![tile; (n / tile) as usize]),
                wgpu::BufferUsages::STORAGE,
            );
            let tile_data: Vec<[u32; 2]> = (0..n / tile / 2).map(|i| [i * 2, i * 2 + 1]).collect();
            let tiles = make_buffer(
                "tiles",
                bytemuck::cast_slice(&tile_data),
                wgpu::BufferUsages::STORAGE,
            );
            let mut permutation: Vec<u32> = (0..tile).collect();
            permutation.shuffle(&mut rand::rngs::StdRng::seed_from_u64(10));
            let perms = make_buffer(
                "permutation",
                bytemuck::cast_slice(&permutation),
                wgpu::BufferUsages::STORAGE,
            );
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[&uniforms, &positions, &distances, &lens, &tiles, &perms]
                    .iter()
                    .enumerate()
                    .map(|(binding, b)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut timer = gpu::PassTimer::new(device, repeats);
            let started = Instant::now();
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..repeats {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: timer.next(),
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(n / tile / 2, replicas, 1);
            }
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            timer.end_iteration();
            let gpu_ms =
                timer.finish(device, queue)?.expect("timestamps")[0].as_secs_f64() * 1000.0;
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: positions.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&positions, 0, &read, 0, positions.size());
            queue.submit([encoder.finish()]);
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let mapped = read.slice(..).get_mapped_range();
            let output: &[[f32; 2]] = bytemuck::cast_slice(&mapped);
            for replica in output.chunks(n as usize) {
                ensure!(
                    replica == &output[..n as usize],
                    "replica coordinate mismatch"
                );
            }
            let output64: Vec<_> = output[..n as usize]
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64])
                .collect();
            let hash = experiment_common::positions_sha256_f64(&output64);
            if let Some(expected) = references.get(&tile) {
                ensure!(expected == &hash, "matrix/constant coordinate mismatch");
            } else {
                references.insert(tile, hash.clone());
            }
            drop(mapped);
            read.unmap();
            println!(
                "{}",
                json!({"kind": "cross_round_micro", "nodes":n,
                "tile":tile, "replicas":replicas, "distance_mode":distance_mode,
                "workgroup":1024, "workgroups":n/tile/2*replicas,
                "repetition":repetition, "warmup":repetition==0, "dispatches":repeats,
                "updates":n as u64*tile as u64/2*replicas as u64*repeats as u64,
                "gpu_ms":gpu_ms, "wall_ms":wall_ms, "position_sha":hash,
                "adapter":context.adapter_name})
            );
        }
    }
    Ok(())
}
