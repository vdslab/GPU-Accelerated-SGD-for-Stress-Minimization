use crate::graph;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

const T: u32 = 1024; // ブロックサイズ（WGの1024スレッドをフル活用）

/// カーネルに渡す uniform バッファ（外側ループごとに更新）
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub r_outer: u32,
    pub n:       u32,
    pub big_b:   u32,
    pub eta:     f32,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    pub module: wgpu::ShaderModule,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("GPU アダプタが見つかりません: {:?}", e))?;

        println!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue): (wgpu::Device, wgpu::Queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label:                None,
                required_features:    wgpu::Features::empty(),
                required_limits:      adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints:         wgpu::MemoryHints::MemoryUsage,
                trace:                wgpu::Trace::Off,
            },
        ))
        .map_err(|e| anyhow::anyhow!("デバイス作成失敗: {}", e))?;

        let module: wgpu::ShaderModule = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        Ok(GpuContext { device, queue, module })
    }

    /// SGD を実行し、初期座標と最終座標を返す
    pub fn execute_sgd(
        &self,
        params: graph::SgdParams,
    ) -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>)> {
        let n = params.positions.len() as u32;
        let big_b = n.div_ceil(T); // ブロック数 B = ceil(n/T)

        println!("n={}, T={}, B={}", n, T, big_b);

        // ---- f64 → f32 変換 ----
        let etas: Vec<f32> = params.etas.iter().map(|&e| e as f32).collect();

        let positions_f32: Vec<[f32; 2]> = params
            .positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let initial_positions = positions_f32.clone();

        // ---- 距離行列 (n×n, f32, 到達不能=0.0) ----
        println!("距離行列を構築中... ({}×{})", n, n);
        let dist = params.pairs; // EdgeInfo には u,v,dij が入っている

        // 上三角だけ持つ EdgeInfo から n×n フラット配列を作る
        let mut dist_flat = vec![0.0f32; (n * n) as usize];
        for e in &dist {
            let u = e.u;
            let v = e.v;
            let d = e.dij as f32;
            dist_flat[(u * n as usize + v) as usize] = d;
            dist_flat[(v * n as usize + u) as usize] = d; // 対称
        }

        // ---- ブロック長配列 ----
        let block_lens: Vec<u32> = (0..big_b)
            .map(|bi| std::cmp::min(T, n - bi * T))
            .collect();

        // ================================================================
        // wgpu バッファ作成
        // ================================================================
        let positions_flat: Vec<f32> = positions_f32.iter().flat_map(|p| [p[0], p[1]]).collect();

        let positions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("positions"),
            contents: bytemuck::cast_slice(&positions_flat),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let dist_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("dist_flat"),
            contents: bytemuck::cast_slice(&dist_flat),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let block_lens_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("block_lens"),
            contents: bytemuck::cast_slice(&block_lens),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let uniforms_init = Uniforms { r_outer: 0, n, big_b, eta: etas[0] };
        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms_init),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("download"),
            size:               positions_buffer.size(),
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ================================================================
        // バインドグループレイアウト
        // ================================================================
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[
                // binding 0: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   Some(NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap()),
                    },
                    count: None,
                },
                // binding 1: positions (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(NonZeroU64::new(8).unwrap()),
                    },
                    count: None,
                },
                // binding 2: dist_flat (read)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                // binding 3: block_lens (read)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dist_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: block_lens_buffer.as_entire_binding() },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               None,
            layout:              Some(&pipeline_layout),
            module:              &self.module,
            entry_point:         None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache:               None,
        });

        // ================================================================
        // SGD 実行: iteration × B 回 dispatch
        // ================================================================
        let num_iterations = etas.len();
        println!("SGD 開始: iterations={}, B={}", num_iterations, big_b);

        let iter_start = std::time::Instant::now();

        for iter in 0..num_iterations {
            let eta = etas[iter];

            for r_outer in 0..big_b {
                // Uniforms を毎ラウンド更新
                let uni = Uniforms { r_outer, n, big_b, eta };
                self.queue.write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uni));

                let mut encoder = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: None },
                );

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label:             None,
                        timestamp_writes:  None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    // WG g → ブロック対 (i=g, j=(g+r_outer)%B)
                    // workgroup_size(32,32,1) なので各 WG は 32×32 スレッド
                    pass.dispatch_workgroups(big_b, 1, 1);
                }

                self.queue.submit([encoder.finish()]);
                // 1ラウンドごとに待機（WG間グローバル同期の代わり）
                self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }

            println!("iter {}/{} 完了 (eta={:.4})", iter + 1, num_iterations, eta);
        }

        let elapsed = iter_start.elapsed();
        println!(
            "\nSGD 完了: 合計 {:.3}s / iter平均 {:.1}ms",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / num_iterations as f64 * 1000.0
        );

        // ================================================================
        // 結果ダウンロード
        // ================================================================
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        encoder.copy_buffer_to_buffer(&positions_buffer, 0, &download_buffer, 0, positions_buffer.size());
        self.queue.submit([encoder.finish()]);

        let buf_slice = download_buffer.slice(..);
        buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let data = buf_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let final_positions: Vec<[f32; 2]> = floats
            .chunks(2)
            .map(|c| [c[0], c[1]])
            .collect();

        Ok((initial_positions, final_positions))
    }
}
