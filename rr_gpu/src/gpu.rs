use crate::graph;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

const T: u32 = 1024; // ブロックサイズ

/// カーネルに渡す uniform バッファ
/// r_outer / big_b は廃止。タイル割り当ては tiles バッファで渡す。
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub n:   u32,
    pub eta: f32,
    _pad:    [u32; 2],
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    pub module: wgpu::ShaderModule,
}

// ── スケジュール生成 ──────────────────────────────────────────────────────────

/// Berger 1-factorization に基づくラウンドスケジュールを生成する。
///
/// 各ラウンドは独立なタイル集合（各ブロックが高々1回のみ登場）で構成されるため、
/// 同一ラウンド内を並列 dispatch しても書き込み競合が発生しない。
///
/// 返値: Vec<Vec<(i, j)>>
///   外側: ラウンド列
///   内側: そのラウンドで処理するタイル (block_i, block_j)
///         i == j のとき自己タイル（ケースB）
///         i != j のとき異ブロックタイル（ケースA）
pub fn build_schedule(big_b: u32) -> Vec<Vec<(u32, u32)>> {
    let b = big_b as usize;
    let mut rounds: Vec<Vec<(u32, u32)>> = Vec::new();

    if b == 0 {
        return rounds;
    }

    if b == 1 {
        // 自己タイルのみ
        rounds.push(vec![(0, 0)]);
        return rounds;
    }

    // ── Cross-block タイル: Berger round-robin ──────────────────────────────
    // B が奇数のときダミーブロック B を追加して偶数化。
    // ダミーを含むペアは skip する。
    let b_eff = if b % 2 == 0 { b } else { b + 1 };
    let fixed = (b_eff - 1) as u32; // 固定ノード（実ブロック or ダミー）

    for r in 0..(b_eff - 1) {
        let mut round: Vec<(u32, u32)> = Vec::new();

        // 固定ブロックとの対戦
        let opp = r as u32;
        if (fixed as usize) < b && (opp as usize) < b && fixed != opp {
            round.push((fixed.min(opp), fixed.max(opp)));
        }

        // 回転ペア
        for k in 1..b_eff / 2 {
            let u = ((r + k) % (b_eff - 1)) as u32;
            let v = ((r + b_eff - 1 - k) % (b_eff - 1)) as u32;
            if (u as usize) < b && (v as usize) < b && u != v {
                round.push((u.min(v), u.max(v)));
            }
        }

        if !round.is_empty() {
            rounds.push(round);
        }
    }

    // ── Self-tile: 各ブロック (i, i) を chunk_size 個ずつ別ラウンドへ ────────
    // self-tile は1ブロックしか使わないため、同一ラウンドに複数詰め込める。
    // chunk_size = B/2（cross-block ラウンドと同程度の並列数）
    let chunk_size = (b / 2).max(1);
    let self_tiles: Vec<(u32, u32)> = (0..big_b).map(|i| (i, i)).collect();
    for chunk in self_tiles.chunks(chunk_size) {
        rounds.push(chunk.to_vec());
    }

    rounds
}

// ── GPU コンテキスト ──────────────────────────────────────────────────────────

impl GpuContext {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("GPU アダプタが見つかりません: {:?}", e))?;

        println!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
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

        let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        Ok(GpuContext { device, queue, module })
    }

    /// SGD を実行し、初期座標と最終座標を返す
    pub fn execute_sgd(
        &self,
        params: graph::SgdParams,
    ) -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>)> {
        let n      = params.positions.len() as u32;
        let big_b  = n.div_ceil(T);

        println!("n={}, T={}, B={}", n, T, big_b);

        // ── f64 → f32 変換 ────────────────────────────────────────────────────
        let etas: Vec<f32> = params.etas.iter().map(|&e| e as f32).collect();

        let positions_f32: Vec<[f32; 2]> = params.positions.iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let initial_positions = positions_f32.clone();

        // ── 距離行列 n×n (f32, 到達不能=0.0) ──────────────────────────────────
        println!("距離行列を構築中... ({}×{})", n, n);
        let mut dist_flat = vec![0.0f32; (n * n) as usize];
        for e in &params.pairs {
            let d = e.dij as f32;
            dist_flat[e.u * n as usize + e.v] = d;
            dist_flat[e.v * n as usize + e.u] = d;
        }

        // ── ブロック長配列 ─────────────────────────────────────────────────────
        let block_lens: Vec<u32> = (0..big_b)
            .map(|bi| T.min(n - bi * T))
            .collect();

        // ── スケジュール生成 ───────────────────────────────────────────────────
        let schedule = build_schedule(big_b);
        let max_tiles_per_round = schedule.iter().map(|r| r.len()).max().unwrap_or(1);

        println!("スケジュール: {} ラウンド, 最大 {} タイル/ラウンド",
            schedule.len(), max_tiles_per_round);

        // ── バッファ作成 ───────────────────────────────────────────────────────
        let positions_flat: Vec<f32> = positions_f32.iter()
            .flat_map(|p| [p[0], p[1]])
            .collect();

        let positions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("positions"),
            contents: bytemuck::cast_slice(&positions_flat),
            usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
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

        let uniforms_init = Uniforms { n, eta: etas[0], _pad: [0; 2] };
        let uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms_init),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // tiles バッファ: 各ラウンドのタイル割り当て [(i,j), ...]
        // 最大 max_tiles_per_round タイル × 8 bytes（vec2<u32>）
        let tiles_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("tiles"),
            size:               (max_tiles_per_round * 8).max(8) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("download"),
            size:               positions_buffer.size(),
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ── バインドグループレイアウト ──────────────────────────────────────────
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
                        min_binding_size:   Some(
                            NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap()
                        ),
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
                // binding 4: tiles (read) — ラウンドごとのタイル割り当て
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   Some(NonZeroU64::new(8).unwrap()),
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
                wgpu::BindGroupEntry { binding: 4, resource: tiles_buffer.as_entire_binding() },
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

        // ── SGD 実行 ───────────────────────────────────────────────────────────
        let num_iterations = etas.len();
        println!("SGD 開始: iterations={}, rounds/iter={}", num_iterations, schedule.len());

        let iter_start = std::time::Instant::now();

        for iter in 0..num_iterations {
            let eta = etas[iter];
            let uni = Uniforms { n, eta, _pad: [0; 2] };
            self.queue.write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uni));

            for round in &schedule {
                // タイルバッファを今ラウンドの割り当てで更新
                let tiles_flat: Vec<u32> = round.iter()
                    .flat_map(|&(ti, tj)| [ti, tj])
                    .collect();
                self.queue.write_buffer(&tiles_buffer, 0, bytemuck::cast_slice(&tiles_flat));

                let mut encoder = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: None },
                );
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label:            None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    // workgroup g → タイル tiles[g] = (i, j) を担当
                    pass.dispatch_workgroups(round.len() as u32, 1, 1);
                }
                self.queue.submit([encoder.finish()]);
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

        // ── 結果ダウンロード ───────────────────────────────────────────────────
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None },
        );
        encoder.copy_buffer_to_buffer(
            &positions_buffer, 0,
            &download_buffer,  0,
            positions_buffer.size(),
        );
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
