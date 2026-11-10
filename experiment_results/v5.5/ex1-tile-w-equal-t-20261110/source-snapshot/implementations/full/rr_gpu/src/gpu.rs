#[path = "../../../../shared/gpu-timing/wgpu.rs"]
mod timing;
use crate::graph;
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use experiment_common::seed::{rng_for_stream, FULL_UPDATE_STREAM};
use experiment_common::OutputFormat;
use rand::seq::SliceRandom;
use rand::Rng;
use std::num::NonZeroU64;
use std::time::{Duration, Instant};

pub const DEFAULT_TILE_SIZE: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    Shared,
    Global,
}

impl MemoryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Global => "global",
        }
    }
}

impl std::str::FromStr for MemoryMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "shared" => Ok(Self::Shared),
            "global" => Ok(Self::Global),
            _ => anyhow::bail!("--coordinate-memory (--memory-mode) は shared または global です"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    pub tile_size: u32,
    pub workgroup_size: u32,
    pub memory_mode: MemoryMode,
    pub outer_shuffle: bool,
    pub inner_shuffle: bool,
    pub phase_chunks: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            workgroup_size: 1024,
            memory_mode: MemoryMode::Shared,
            outer_shuffle: true,
            inner_shuffle: true,
            phase_chunks: 1,
        }
    }
}

impl KernelConfig {
    pub fn validate(self) -> Result<()> {
        anyhow::ensure!(
            (1..=1024).contains(&self.tile_size),
            "--tile-size は1〜1024です"
        );
        anyhow::ensure!(
            (1..=1024).contains(&self.workgroup_size),
            "--workgroup-size は1〜1024です"
        );
        anyhow::ensure!(self.phase_chunks > 0, "--phase-chunks must be positive");
        Ok(())
    }
}

fn self_round_count(length: u32) -> u32 {
    if length < 2 {
        0
    } else {
        length + length % 2 - 1
    }
}

// Each domain has its own exact permutation. Taking a shuffled prefix modulo
// a smaller domain repeats some rounds and omits others.
fn build_inner_permutation(
    tile_size: u32,
    last_block_len: u32,
    shuffle: bool,
    rng: &mut impl Rng,
) -> Vec<u32> {
    let mut permutation = Vec::new();
    for count in [
        tile_size,
        self_round_count(tile_size),
        self_round_count(last_block_len),
    ] {
        let start = permutation.len();
        permutation.extend(0..count);
        if shuffle {
            permutation[start..].shuffle(rng);
        }
    }
    permutation
}

fn shader_source(config: KernelConfig) -> String {
    let (helpers, init_cross, init_self, flush_cross, flush_self, barrier) = match config
        .memory_mode
    {
        MemoryMode::Shared => (
            r#"var<workgroup> wg_pos_i: array<vec2<f32>, __BLOCK_T__>;
var<workgroup> wg_pos_j: array<vec2<f32>, __BLOCK_T__>;
fn read_i(block: u32, index: u32) -> vec2<f32> { return wg_pos_i[index]; }
fn read_j(block: u32, index: u32) -> vec2<f32> { return wg_pos_j[index]; }
fn write_i(block: u32, index: u32, value: vec2<f32>) { wg_pos_i[index] = value; }
fn write_j(block: u32, index: u32, value: vec2<f32>) { wg_pos_j[index] = value; }"#,
            r#"for (var local = tx; local < BLOCK_T; local += WORKGROUP_SIZE) {
                if local < li { wg_pos_i[local] = positions[i * BLOCK_T + local]; }
                if local < lj { wg_pos_j[local] = positions[j * BLOCK_T + local]; }
            }
            workgroupBarrier();"#,
            r#"for (var local = tx; local < li; local += WORKGROUP_SIZE) {
                wg_pos_i[local] = positions[i * BLOCK_T + local];
            }
            workgroupBarrier();"#,
            r#"for (var local = tx; local < BLOCK_T; local += WORKGROUP_SIZE) {
                if local < li { positions[i * BLOCK_T + local] = wg_pos_i[local]; }
                if local < lj { positions[j * BLOCK_T + local] = wg_pos_j[local]; }
            }"#,
            r#"for (var local = tx; local < li; local += WORKGROUP_SIZE) {
                positions[i * BLOCK_T + local] = wg_pos_i[local];
            }"#,
            "workgroupBarrier();",
        ),
        MemoryMode::Global => (
            r#"fn read_i(block: u32, index: u32) -> vec2<f32> { return positions[block * BLOCK_T + index]; }
fn read_j(block: u32, index: u32) -> vec2<f32> { return positions[block * BLOCK_T + index]; }
fn write_i(block: u32, index: u32, value: vec2<f32>) { positions[block * BLOCK_T + index] = value; }
fn write_j(block: u32, index: u32, value: vec2<f32>) { positions[block * BLOCK_T + index] = value; }"#,
            "",
            "",
            "",
            "",
            "storageBarrier();",
        ),
    };
    include_str!("shader.wgsl")
        .replace("__MEMORY_HELPERS__", helpers)
        .replace("__INIT_CROSS__", init_cross)
        .replace("__INIT_SELF__", init_self)
        .replace("__FLUSH_CROSS__", flush_cross)
        .replace("__FLUSH_SELF__", flush_self)
        .replace("__MEMORY_BARRIER__", barrier)
        .replace("__BLOCK_T__", &config.tile_size.to_string())
        .replace("__WORKGROUP_SIZE__", &config.workgroup_size.to_string())
}

/// カーネルに渡す uniform バッファ
/// Self-tile permutation offsets select the exact full/partial domains.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub n: u32,
    pub eta: f32,
    pub full_self_perm_offset: u32,
    pub partial_self_perm_offset: u32,
    pub chunk: u32,
    pub chunks: u32,
    pub tiles_offset: u32,
    pub perm_offset: u32,
    pub subround_start: u32,
    pub subround_end: u32,
    pub padding: [u32; 2],
}

const MAX_DISPATCHES_PER_SUBMISSION: usize = 256;

fn submission_ranges(invocation_count: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::with_capacity(invocation_count.div_ceil(MAX_DISPATCHES_PER_SUBMISSION));
    let mut start = 0;
    while start < invocation_count {
        let end = start
            .saturating_add(MAX_DISPATCHES_PER_SUBMISSION)
            .min(invocation_count);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
    pub adapter_name: String,
    pub backend: String,
    pub timestamp_supported: bool,
    pub kernel: KernelConfig,
}

#[derive(Debug)]
pub struct GpuRunResult {
    pub initial_positions: Vec<[f64; 2]>,
    pub positions: Vec<[f32; 2]>,
    pub upload_time: Duration,
    pub preparation_time: Duration,
    pub data_setup_time: Duration,
    pub time_to_result: Duration,
    pub result_ready_monotonic_ms: f64,
    pub iteration_times: Option<Vec<Duration>>,
    pub iteration_time: Duration,
    pub gpu_device_time: Option<Duration>,
    pub readback_time: Duration,
    pub dispatches: u64,
}

// ── スケジュール生成 ──────────────────────────────────────────────────────────

/// サイクルアルゴリズムに基づくラウンドスケジュールを生成する。
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

    // ── サイクルアルゴリズム ──────────────────────────────
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
    pub fn new(kernel: KernelConfig) -> Result<Self> {
        kernel.validate()?;
        let (_instance, adapter, device, queue) =
            experiment_common::gpu::init_gpu_context(wgpu::PowerPreference::HighPerformance)?;

        let info = adapter.get_info();
        let timestamp_supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);

        anyhow::ensure!(
            kernel.workgroup_size <= device.limits().max_compute_invocations_per_workgroup,
            "workgroup size exceeds this GPU's invocation limit"
        );
        anyhow::ensure!(
            kernel.workgroup_size <= device.limits().max_compute_workgroup_size_x,
            "workgroup size exceeds this GPU's x-dimension limit"
        );
        if kernel.memory_mode == MemoryMode::Shared {
            anyhow::ensure!(
                kernel.tile_size * 16 <= device.limits().max_compute_workgroup_storage_size,
                "tile size exceeds this GPU's shared memory limit"
            );
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("race-free two-level RR SGD"),
            source: wgpu::ShaderSource::Wgsl(shader_source(kernel).into()),
        });

        Ok(GpuContext {
            device,
            queue,
            module,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            timestamp_supported,
            kernel,
        })
    }

    /// SGD を実行し、初期座標と最終座標を返す
    pub fn execute_sgd(
        &self,
        params: graph::SgdParams,
        schedule: &[Vec<(u32, u32)>],
        seed: u64,
        verbose: bool,
        output_format: OutputFormat,
    ) -> Result<GpuRunResult> {
        let execution_started = Instant::now();
        let n = params.positions.len() as u32;
        let tile_size = self.kernel.tile_size;
        let phase_chunks = self.kernel.phase_chunks.min(tile_size);

        // ── f64 → f32 変換 ────────────────────────────────────────────────────
        let etas: Vec<f32> = params.etas.iter().map(|&e| e as f32).collect();

        let initial_positions = params.positions.clone();
        let positions_f32: Vec<[f32; 2]> = params
            .positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();

        anyhow::ensure!(
            n <= 16384,
            "Full SGD supports up to N=16384 for 4-bank binding (got N={n})"
        );
        const BANK_WORDS: usize = 33_554_432; // 128 MiB = 2^25 words

        let dist_u16 = if !params.dist_u16.is_empty() {
            params.dist_u16
        } else {
            let total_distances = (n * n) as usize;
            let total_words = total_distances.div_ceil(2);
            let mut dist_u16 = vec![0u16; total_words * 2];
            for e in &params.pairs {
                let d = e.dij as u16;
                dist_u16[e.u * n as usize + e.v] = d;
                dist_u16[e.v * n as usize + e.u] = d;
            }
            dist_u16
        };
        let dist_words: &[u32] = bytemuck::cast_slice(&dist_u16);

        let get_bank_slice = |bank_idx: usize| -> &[u32] {
            let start = (bank_idx * BANK_WORDS).min(dist_words.len());
            let end = ((bank_idx + 1) * BANK_WORDS).min(dist_words.len());
            &dist_words[start..end]
        };

        let bank0_words = get_bank_slice(0);
        let bank1_words = get_bank_slice(1);
        let bank2_words = get_bank_slice(2);
        let bank3_words = get_bank_slice(3);

        // ── ブロック長配列 ─────────────────────────────────────────────────────
        let big_b = n.div_ceil(tile_size);
        let block_lens: Vec<u32> = (0..big_b)
            .map(|bi| tile_size.min(n - bi * tile_size))
            .collect();
        let last_block_len = *block_lens.last().context("no vertices")?;
        let full_self_perm_offset = tile_size;
        let partial_self_perm_offset = tile_size + self_round_count(tile_size);

        // Prepack all tiles across all schedule rounds statically
        let total_tiles: usize = schedule.iter().map(|r| r.len()).sum();
        let mut all_tiles = Vec::with_capacity(total_tiles * 2);
        let mut schedule_tile_offsets = Vec::with_capacity(schedule.len());
        for round in schedule {
            schedule_tile_offsets.push((all_tiles.len() / 2) as u32);
            for &(ti, tj) in round {
                all_tiles.push(ti);
                all_tiles.push(tj);
            }
        }

        let perm_len = tile_size + self_round_count(tile_size) + self_round_count(last_block_len);
        let total_perm_elements = schedule.len() * perm_len as usize;

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as usize;
        let slot_size = std::mem::size_of::<Uniforms>().div_ceil(alignment) * alignment;
        let invocation_count = (phase_chunks as usize) * schedule.len();
        let submission_ranges = submission_ranges(invocation_count);

        // ── バッファ作成 ───────────────────────────────────────────────────────
        let positions_flat: Vec<f32> = positions_f32.iter().flat_map(|p| [p[0], p[1]]).collect();

        let data_setup_time = execution_started.elapsed();
        let preparation_started = Instant::now();
        let positions_buffer = timing::buffer(
            &self.device,
            Some("positions"),
            (bytemuck::cast_slice::<_, u8>(&positions_flat)).len(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let dist_bank0_buffer = timing::buffer(
            &self.device,
            Some("dist_bank0"),
            bank0_words.len() * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let dist_bank1_buffer = timing::buffer(
            &self.device,
            Some("dist_bank1"),
            bank1_words.len() * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let dist_bank2_buffer = timing::buffer(
            &self.device,
            Some("dist_bank2"),
            bank2_words.len() * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let dist_bank3_buffer = timing::buffer(
            &self.device,
            Some("dist_bank3"),
            bank3_words.len() * 4,
            wgpu::BufferUsages::STORAGE,
        );

        let block_lens_buffer = timing::buffer(
            &self.device,
            Some("block_lens"),
            (bytemuck::cast_slice::<_, u8>(&block_lens)).len(),
            wgpu::BufferUsages::STORAGE,
        );

        let uniforms_init = Uniforms {
            n,
            eta: etas[0],
            full_self_perm_offset,
            partial_self_perm_offset,
            chunk: 0,
            chunks: phase_chunks,
            tiles_offset: 0,
            perm_offset: 0,
            subround_start: 0,
            subround_end: tile_size / phase_chunks,
            padding: [0; 2],
        };
        let uniforms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: (slot_size * invocation_count.max(1)) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tiles_buffer = timing::buffer(
            &self.device,
            Some("tiles"),
            all_tiles.len() * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        );

        let perm_buffer = timing::buffer(
            &self.device,
            Some("inner_perm"),
            total_perm_elements * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        );

        let download_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: positions_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ── バインドグループレイアウト ──────────────────────────────────────────
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // binding 0: Uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: Some(
                                NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap(),
                            ),
                        },
                        count: None,
                    },
                    // binding 1: positions (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(8).unwrap()),
                        },
                        count: None,
                    },
                    // binding 2: dist_bank0 (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // binding 3: block_lens (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // binding 4: tiles (read) — 全ラウンドのタイル割り当て
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(8).unwrap()),
                        },
                        count: None,
                    },
                    // binding 5: inner_perm (read) — ケースA内側ループのオフセット順列
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // binding 6: dist_bank1 (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // binding 7: dist_bank2 (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                    // binding 8: dist_bank3 (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniforms_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dist_bank0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: block_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tiles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: perm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dist_bank1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dist_bank2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dist_bank3_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &self.module,
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Warm up Mesa NIR->DXIL compilation before timed iteration loops
        if !all_tiles.is_empty() {
            self.queue
                .write_buffer(&tiles_buffer, 0, bytemuck::cast_slice(&all_tiles));
        }
        let initial_perms = vec![0u32; total_perm_elements.max(1)];
        self.queue
            .write_buffer(&perm_buffer, 0, bytemuck::cast_slice(&initial_perms));
        self.queue
            .write_buffer(&uniforms_buffer, 0, bytemuck::bytes_of(&uniforms_init));

        experiment_common::gpu::run_warmup_dispatch(
            &self.device,
            &self.queue,
            &pipeline,
            &bind_group,
            &[0],
        )?;
        self.queue
            .write_buffer(&positions_buffer, 0, bytemuck::cast_slice(&positions_flat));
        self.queue.submit([]);
        self.device.poll(wgpu::PollType::wait_indefinitely())?;

        // ── SGD 実行 ───────────────────────────────────────────────────────────
        let num_iterations = etas.len();
        let mut timer = timing::PassTimer::new(
            &self.device,
            num_iterations * submission_ranges.len(),
        );
        let preparation_time = preparation_started.elapsed();
        let upload_started = Instant::now();
        self.queue
            .write_buffer(&positions_buffer, 0, bytemuck::cast_slice(&positions_flat));
        self.queue
            .write_buffer(&dist_bank0_buffer, 0, bytemuck::cast_slice(bank0_words));
        if !bank1_words.is_empty() {
            self.queue
                .write_buffer(&dist_bank1_buffer, 0, bytemuck::cast_slice(bank1_words));
        }
        if !bank2_words.is_empty() {
            self.queue
                .write_buffer(&dist_bank2_buffer, 0, bytemuck::cast_slice(bank2_words));
        }
        if !bank3_words.is_empty() {
            self.queue
                .write_buffer(&dist_bank3_buffer, 0, bytemuck::cast_slice(bank3_words));
        }
        self.queue
            .write_buffer(&block_lens_buffer, 0, bytemuck::cast_slice(&block_lens));
        if !all_tiles.is_empty() {
            self.queue
                .write_buffer(&tiles_buffer, 0, bytemuck::cast_slice(&all_tiles));
        }
        self.queue.submit([]);
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let upload_time = upload_started.elapsed();
        let mut trace = experiment_common::trace::IterationTrace::from_env();
        if let Some(trace) = &mut trace {
            trace.record(0, Duration::ZERO, || {
                positions_f32
                    .iter()
                    .map(|p| [p[0] as f64, p[1] as f64])
                    .collect()
            });
        }
        let initial_trace_overhead = trace.as_ref().map(|t| t.overhead()).unwrap_or_default();
        let iteration_started = Instant::now();
        let mut rng = rng_for_stream(seed, FULL_UPDATE_STREAM);

        // 外側シャッフル用にスケジュールのインデックス列を用意
        let mut round_order: Vec<usize> = (0..schedule.len()).collect();

        let mut uniform_bytes = vec![0u8; slot_size * invocation_count];
        let mut all_perms = Vec::with_capacity(total_perm_elements);

        for iter in 0..num_iterations {
            let trace_started = trace.as_ref().map(|_| Instant::now());
            let eta = etas[iter];

            // 外側ランダム: ラウンドの処理順をイテレーションごとにシャッフル
            if self.kernel.outer_shuffle {
                round_order.shuffle(&mut rng);
            }

            let permutations: Vec<_> = round_order
                .iter()
                .map(|_| {
                    build_inner_permutation(
                        tile_size,
                        last_block_len,
                        self.kernel.inner_shuffle,
                        &mut rng,
                    )
                })
                .collect();

            // 1. Pack and upload all inner permutations in ONE call
            all_perms.clear();
            for perm in &permutations {
                all_perms.extend_from_slice(perm);
            }
            if !all_perms.is_empty() {
                self.queue
                    .write_buffer(&perm_buffer, 0, bytemuck::cast_slice(&all_perms));
            }

            // 2. Pre-populate all dispatch uniforms in host memory
            let mut slot = 0;
            for chunk in 0..phase_chunks {
                let subround_start = chunk * tile_size / phase_chunks;
                let subround_end = (chunk + 1) * tile_size / phase_chunks;
                for (round_position, &ri) in round_order.iter().enumerate() {
                    let uni = Uniforms {
                        n,
                        eta,
                        full_self_perm_offset,
                        partial_self_perm_offset,
                        chunk,
                        chunks: phase_chunks,
                        tiles_offset: schedule_tile_offsets[ri],
                        perm_offset: (round_position as u32) * perm_len,
                        subround_start,
                        subround_end,
                        padding: [0; 2],
                    };
                    let offset = slot * slot_size;
                    uniform_bytes[offset..offset + std::mem::size_of::<Uniforms>()]
                        .copy_from_slice(bytemuck::bytes_of(&uni));
                    slot += 1;
                }
            }
            if !uniform_bytes.is_empty() {
                self.queue.write_buffer(&uniforms_buffer, 0, &uniform_bytes);
            }

            // 3. Command recording: Bounded ComputePass submissions
            for batch_range in &submission_ranges {
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let timestamp_writes = timer.next();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes,
                    });
                    pass.set_pipeline(&pipeline);

                    for slot in batch_range.clone() {
                        let round_position = slot % schedule.len();
                        let ri = round_order[round_position];
                        let round = &schedule[ri];

                        pass.set_bind_group(0, &bind_group, &[(slot * slot_size) as u32]);
                        pass.dispatch_workgroups(round.len() as u32, 1, 1);
                    }
                }
                self.queue.submit([encoder.finish()]);
            }

            // Queue writes and submissions retain round order. Synchronize once
            // per epoch so iteration timing includes completed GPU work.
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            timer.end_iteration();

            if verbose {
                let message = format!("Iteration {}/{} (eta={eta:.4})", iter + 1, num_iterations);
                match output_format {
                    OutputFormat::Json => eprintln!("{message}"),
                    OutputFormat::Human => println!("{message}"),
                }
            }
            if let (Some(trace), Some(started)) = (&mut trace, trace_started) {
                trace.record(iter + 1, started.elapsed(), || {
                    trace_positions(
                        &self.device,
                        &self.queue,
                        &positions_buffer,
                        &download_buffer,
                        n as usize,
                    )
                });
            }
        }
        let iteration_time = iteration_started.elapsed().saturating_sub(
            trace
                .as_ref()
                .map(|t| t.overhead().saturating_sub(initial_trace_overhead))
                .unwrap_or_default(),
        );

        // ── 結果ダウンロード ───────────────────────────────────────────────────
        let readback_started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &positions_buffer,
            0,
            &download_buffer,
            0,
            positions_buffer.size(),
        );
        self.queue.submit([encoder.finish()]);

        let buf_slice = download_buffer.slice(..);
        buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let data = buf_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let final_positions: Vec<[f32; 2]> = floats.chunks(2).map(|c| [c[0], c[1]]).collect();
        drop(data);
        download_buffer.unmap();
        let readback_time = readback_started.elapsed();
        let time_to_result = execution_started.elapsed();
        let result_ready_monotonic_ms = experiment_common::monotonic_time_ms();
        let iteration_times = timer.finish(&self.device, &self.queue)?;
        let gpu_device_time = iteration_times
            .as_ref()
            .map(|times| times.iter().copied().sum());

        Ok(GpuRunResult {
            initial_positions,
            positions: final_positions,
            upload_time,
            preparation_time,
            data_setup_time,
            time_to_result,
            result_ready_monotonic_ms,
            iteration_times,
            iteration_time,
            gpu_device_time,
            readback_time,
            dispatches: (schedule.len() * num_iterations) as u64 * phase_chunks as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    // CPU image of the shader's matchings, including strided invocation work.
    fn tile_matchings(
        li: u32,
        lj: u32,
        same_block: bool,
        config: KernelConfig,
        permutation: &[u32],
    ) -> Vec<Vec<(u32, u32)>> {
        let mut matchings = Vec::new();
        let rounds = if same_block {
            self_round_count(li)
        } else {
            config.tile_size
        };
        for &r in &permutation[..rounds as usize] {
            let mut matching = Vec::new();
            for thread in 0..config.workgroup_size {
                let limit = if same_block { (li + li % 2) / 2 } else { li };
                let mut slot = thread;
                while slot < limit {
                    let (u, v) = if same_block {
                        if slot == 0 {
                            (rounds, r)
                        } else {
                            ((r + slot) % rounds, (r + rounds - slot) % rounds)
                        }
                    } else {
                        (slot, (slot + r) % config.tile_size)
                    };
                    if u < li && v < lj {
                        matching.push((u, v));
                    }
                    slot += config.workgroup_size;
                }
            }
            matchings.push(matching);
        }
        matchings
    }

    fn assert_matching_is_disjoint(matching: &[(u32, u32)], same_block: bool) {
        let mut occupied = BTreeSet::new();
        for &(u, v) in matching {
            assert!(occupied.insert((0, u)), "overlapping first endpoint");
            assert!(
                occupied.insert((if same_block { 0 } else { 1 }, v)),
                "overlapping second endpoint"
            );
        }
    }

    #[test]
    fn exact_permutations_cover_all_cross_and_self_pairs_without_races() {
        let mut rng = rng_for_stream(17, FULL_UPDATE_STREAM);
        for tile_size in [1, 2, 3, 7, 8, 32] {
            for workgroup_size in [1, 2, 4, 64] {
                let config = KernelConfig {
                    tile_size,
                    workgroup_size,
                    ..KernelConfig::default()
                };
                for li in 1..=tile_size {
                    for lj in 1..=tile_size {
                        let permutation = build_inner_permutation(tile_size, li, true, &mut rng);
                        let cross = tile_matchings(li, lj, false, config, &permutation);
                        let mut counts = BTreeMap::new();
                        for matching in &cross {
                            assert_matching_is_disjoint(matching, false);
                            for &pair in matching {
                                *counts.entry(pair).or_insert(0) += 1;
                            }
                        }
                        assert_eq!(counts.len(), (li * lj) as usize);
                        assert!(counts.values().all(|&count| count == 1));
                    }
                    let permutation = build_inner_permutation(tile_size, li, true, &mut rng);
                    let offset = if li == tile_size {
                        tile_size
                    } else {
                        tile_size + self_round_count(tile_size)
                    };
                    let self_tiles =
                        tile_matchings(li, li, true, config, &permutation[offset as usize..]);
                    let mut counts = BTreeMap::new();
                    for matching in &self_tiles {
                        assert_matching_is_disjoint(matching, true);
                        for &(u, v) in matching {
                            *counts.entry((u.min(v), u.max(v))).or_insert(0) += 1;
                        }
                    }
                    assert_eq!(counts.len(), (li * (li - 1) / 2) as usize);
                    assert!(counts.values().all(|&count| count == 1));
                }
            }
        }
    }

    #[test]
    fn outer_rounds_are_disjoint_and_cover_odd_and_even_block_counts() {
        for blocks in 0..=31 {
            let mut counts = BTreeMap::new();
            for round in build_schedule(blocks) {
                let mut occupied = BTreeSet::new();
                for (u, v) in round {
                    assert!(occupied.insert(u));
                    if u != v {
                        assert!(occupied.insert(v));
                    }
                    *counts.entry((u.min(v), u.max(v))).or_insert(0) += 1;
                }
            }
            assert_eq!(counts.len(), (blocks * (blocks + 1) / 2) as usize);
            assert!(counts.values().all(|&count| count == 1));
        }
    }

    #[test]
    fn global_shader_allocates_no_shared_memory_and_barriers_match_memory_mode() {
        let shared = shader_source(KernelConfig::default());
        assert!(shared.contains("var<workgroup>"));
        assert!(shared.contains("workgroupBarrier();"));
        let global = shader_source(KernelConfig {
            memory_mode: MemoryMode::Global,
            ..KernelConfig::default()
        });
        assert!(!global.contains("var<workgroup>"));
        assert!(!global.contains("workgroupBarrier();"));
        assert!(global.contains("storageBarrier();"));
        assert!(!global.contains("__"));
    }

    fn fixture(n: u32) -> graph::SgdParams {
        let pairs: Vec<_> = (0..n)
            .flat_map(|u| {
                ((u + 1)..n).map(move |v| {
                    let d = (1 + (u + v) % 5) as f64;
                    graph::EdgeInfo {
                        u: u as usize,
                        v: v as usize,
                        dij: d,
                        wij: 1.0 / (d * d),
                    }
                })
            })
            .collect();
        let constraint_count = pairs.len();
        graph::SgdParams {
            etas: vec![0.2, 0.05],
            positions: (0..n)
                .map(|u| [u as f64 * 0.3, ((u * u + 1) % 11) as f64 * 0.4])
                .collect(),
            dist_u16: Vec::new(),
            constraint_count,
            pairs,
            center: false,
        }
    }

    fn reference_sgd(params: &graph::SgdParams, config: KernelConfig, seed: u64) -> Vec<[f32; 2]> {
        let n = params.positions.len() as u32;
        let blocks = n.div_ceil(config.tile_size);
        let block_len = |b| config.tile_size.min(n - b * config.tile_size);
        let last_len = block_len(blocks - 1);
        let schedule = build_schedule(blocks);
        let mut round_order: Vec<_> = (0..schedule.len()).collect();
        let mut rng = rng_for_stream(seed, FULL_UPDATE_STREAM);
        let mut positions: Vec<[f32; 2]> = params
            .positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let mut distances = vec![0.0_f32; n as usize * n as usize];
        if !params.dist_u16.is_empty() {
            for u in 0..n as usize {
                for v in 0..n as usize {
                    distances[u * n as usize + v] = params.dist_u16[u * n as usize + v] as f32;
                }
            }
        } else {
            for pair in &params.pairs {
                distances[pair.u * n as usize + pair.v] = pair.dij as f32;
                distances[pair.v * n as usize + pair.u] = pair.dij as f32;
            }
        }
        for &eta in &params.etas {
            if config.outer_shuffle {
                round_order.shuffle(&mut rng);
            }
            let permutations: Vec<_> = round_order
                .iter()
                .map(|_| {
                    build_inner_permutation(
                        config.tile_size,
                        last_len,
                        config.inner_shuffle,
                        &mut rng,
                    )
                })
                .collect();
            let chunks = config.phase_chunks.min(config.tile_size) as usize;
            for chunk in 0..chunks {
                for (&round, permutation) in round_order.iter().zip(&permutations) {
                    for &(i, j) in &schedule[round] {
                        let offset = if i != j {
                            0
                        } else if block_len(i) == config.tile_size {
                            config.tile_size
                        } else {
                            config.tile_size + self_round_count(config.tile_size)
                        };
                        let matchings = tile_matchings(
                            block_len(i),
                            block_len(j),
                            i == j,
                            config,
                            &permutation[offset as usize..],
                        );
                        let count = matchings.len();
                        for matching in
                            &matchings[chunk * count / chunks..(chunk + 1) * count / chunks]
                        {
                            for &(u, v) in matching {
                                let gu = (i * config.tile_size + u) as usize;
                                let gv = (j * config.tile_size + v) as usize;
                                let d = distances[gu * n as usize + gv];
                                if d < 0.5 {
                                    continue;
                                }
                                let dx = positions[gv][0] - positions[gu][0];
                                let dy = positions[gv][1] - positions[gu][1];
                                let current = (dx * dx + dy * dy).sqrt().max(1e-12);
                                let mu = ((1.0 / (d * d)) * eta as f32).min(1.0);
                                let delta = [
                                    mu * ((current - d) / 2.0) * (dx / current),
                                    mu * ((current - d) / 2.0) * (dy / current),
                                ];
                                for axis in 0..2 {
                                    positions[gu][axis] += delta[axis];
                                    positions[gv][axis] -= delta[axis];
                                }
                            }
                        }
                    }
                }
            }
        }
        positions
    }

    #[test]
    fn tiny_gpu_shared_and_global_agree_with_exact_rr_reference() {
        // Odd and partial blocks, T<W and T>W exercise the corrected paths.
        for (tile_size, workgroup_size, n, phase_chunks) in [
            (1, 32, 5, 16),
            (7, 2, 19, 4),
            (8, 2, 19, 16),
            (32, 256, 35, 16),
            (32, 256, 35, 1),
        ] {
            let mut outcomes = Vec::new();
            for memory_mode in [MemoryMode::Shared, MemoryMode::Global] {
                let config = KernelConfig {
                    tile_size,
                    workgroup_size,
                    memory_mode,
                    phase_chunks,
                    ..KernelConfig::default()
                };
                let params = fixture(n);
                let expected = reference_sgd(&params, config, 13);
                let context =
                    GpuContext::new(config).expect("GPU required for RR kernel smoke test");
                let result = context
                    .execute_sgd(
                        params,
                        &build_schedule(n.div_ceil(tile_size)),
                        13,
                        false,
                        OutputFormat::Json,
                    )
                    .unwrap();
                for (index, (got, want)) in result.positions.iter().zip(&expected).enumerate() {
                    for axis in 0..2 {
                        assert!((got[axis] - want[axis]).abs() < 2e-5, "T={tile_size} W={workgroup_size} mode={memory_mode:?} vertex={index}: {got:?} vs {want:?}");
                    }
                }
                outcomes.push(result.positions);
            }
            for (a, b) in outcomes[0].iter().zip(&outcomes[1]) {
                for axis in 0..2 {
                    assert!((a[axis] - b[axis]).abs() < 2e-5);
                }
            }
        }
    }
}

/// Read a completed iteration only for an enabled convergence checkpoint.
/// This transfer is measured by IterationTrace as excluded overhead.
fn trace_positions(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    positions: &wgpu::Buffer,
    download: &wgpu::Buffer,
    node_count: usize,
) -> Vec<[f64; 2]> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("convergence checkpoint readback"),
    });
    encoder.copy_buffer_to_buffer(positions, 0, download, 0, positions.size());
    queue.submit([encoder.finish()]);
    let slice = download.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("convergence readback poll");
    let mapped = slice.get_mapped_range();
    let values: &[[f32; 2]] = bytemuck::cast_slice(&mapped);
    let result = values[..node_count]
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64])
        .collect();
    drop(mapped);
    download.unmap();
    result
}
