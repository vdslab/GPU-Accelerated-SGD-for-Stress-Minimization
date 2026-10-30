use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

pub const DEFAULT_NODE_RADIUS_PX: f32 = 1.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    node_r_x: f32,
    node_r_y: f32,
    _pad0: f32,
    _pad1: f32,
}

const QUAD_VERTS: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

pub struct RenderOutput {
    pub pixels: Vec<u8>,
    pub prepare_time: Duration,
    pub execute_readback_time: Duration,
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
}

impl GpuRenderer {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("GPU adapter not found: {e:?}"))?;
        println!(
            "GPU: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| anyhow::anyhow!("Device creation failed: {e}"))?;
        Ok(Self {
            device,
            queue,
            limits,
        })
    }

    pub fn render(
        &self,
        positions: &[[f32; 2]],
        edges: &[(usize, usize)],
        width: u32,
        height: u32,
        node_radius_px: f32,
        colors: &[f32],
        sunset: bool,
    ) -> Result<RenderOutput> {
        let prepare_started = Instant::now();
        let indices = build_edge_indices(edges)?;
        self.validate_sizes(positions.len(), indices.len(), width, height)?;
        let (min_x, max_x, min_y, max_y) = bounds(positions);
        let pad_x = (max_x - min_x).max(1e-6) * 0.05;
        let pad_y = (max_y - min_y).max(1e-6) * 0.05;
        let uniforms = Uniforms {
            min_x: min_x - pad_x,
            max_x: max_x + pad_x,
            min_y: min_y - pad_y,
            max_y: max_y + pad_y,
            node_r_x: node_radius_px * 2.0 / width as f32,
            node_r_y: node_radius_px * 2.0 / height as f32,
            _pad0: if sunset { 1.0 } else { 0.0 },
            _pad1: 0.0,
        };

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("graph_shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let texture_format = wgpu::TextureFormat::Rgba8Unorm;
        let attr0 = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }];
        let attr1 = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 1,
        }];
        let color_attr = wgpu::vertex_attr_array![2 => Float32];
        let target = || {
            Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let edge_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("edge_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_edge"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attr0,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_edge"),
                    targets: &[target()],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
        let node_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("node_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_node"),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &attr0,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &attr1,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &color_attr,
                        },
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_node"),
                    targets: &[target()],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        // This is the only graph-position buffer. Edges index into it and nodes reuse it as instances.
        let position_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shared_positions"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let color_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("topology_colors"),
                contents: bytemuck::cast_slice(colors),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_storage = if indices.is_empty() {
            vec![0_u32]
        } else {
            indices
        };
        let edge_index_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edge_indices"),
                contents: bytemuck::cast_slice(&index_storage),
                usage: wgpu::BufferUsages::INDEX,
            });
        let edge_index_count = edges
            .len()
            .checked_mul(2)
            .context("辺index数がオーバーフローしました")? as u32;
        let quad_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quad_verts"),
                contents: bytemuck::cast_slice(&QUAD_VERTS),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let tex_view = texture.create_view(&Default::default());
        let bytes_per_row = width
            .checked_mul(4)
            .context("画像の行サイズがオーバーフローしました")?
            .next_multiple_of(256);
        let staging_size = u64::from(bytes_per_row) * u64::from(height);
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let prepare_time = prepare_started.elapsed();

        let execute_started = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &tex_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            if edge_index_count > 0 {
                pass.set_pipeline(&edge_pipeline);
                pass.set_vertex_buffer(0, position_buf.slice(..));
                pass.set_index_buffer(edge_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..edge_index_count, 0, 0..1);
            }
            pass.set_pipeline(&node_pipeline);
            pass.set_vertex_buffer(0, quad_buf.slice(..));
            pass.set_vertex_buffer(1, position_buf.slice(..));
            pass.set_vertex_buffer(2, color_buf.slice(..));
            pass.draw(0..QUAD_VERTS.len() as u32, 0..positions.len() as u32);
        }
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU poll failed")?;
        let data = slice.get_mapped_range();
        let packed_row = width as usize * 4;
        let mut pixels = Vec::with_capacity(packed_row * height as usize);
        for row in 0..height as usize {
            let start = row * bytes_per_row as usize;
            pixels.extend_from_slice(&data[start..start + packed_row]);
        }
        let execute_readback_time = execute_started.elapsed();
        Ok(RenderOutput {
            pixels,
            prepare_time,
            execute_readback_time,
        })
    }

    fn validate_sizes(&self, nodes: usize, indices: usize, width: u32, height: u32) -> Result<()> {
        let position_bytes = nodes
            .checked_mul(size_of::<[f32; 2]>())
            .context("頂点bufferサイズがオーバーフローしました")?
            as u64;
        let index_bytes = indices
            .checked_mul(size_of::<u32>())
            .context("辺bufferサイズがオーバーフローしました")? as u64;
        let row = u64::from(
            width
                .checked_mul(4)
                .context("画像サイズがオーバーフローしました")?
                .next_multiple_of(256),
        );
        let staging_bytes = row
            .checked_mul(u64::from(height))
            .context("画像bufferサイズがオーバーフローしました")?;
        for (name, required) in [
            ("vertex", position_bytes),
            ("index", index_bytes),
            ("readback", staging_bytes),
        ] {
            anyhow::ensure!(required <= self.limits.max_buffer_size, "{name} bufferがGPU上限を超えます: required={required} bytes, max_buffer_size={} bytes", self.limits.max_buffer_size);
        }
        anyhow::ensure!(
            width <= self.limits.max_texture_dimension_2d
                && height <= self.limits.max_texture_dimension_2d,
            "出力画像がGPUの2D texture上限を超えます: requested={}x{}, max={}",
            width,
            height,
            self.limits.max_texture_dimension_2d
        );
        Ok(())
    }
}

pub fn build_edge_indices(edges: &[(usize, usize)]) -> Result<Vec<u32>> {
    let capacity = edges
        .len()
        .checked_mul(2)
        .context("辺index数がオーバーフローしました")?;
    let mut result = Vec::with_capacity(capacity);
    for &(u, v) in edges {
        result.push(u32::try_from(u).context("辺端点がu32上限を超えています")?);
        result.push(u32::try_from(v).context("辺端点がu32上限を超えています")?);
    }
    Ok(result)
}

pub fn auto_node_radius(node_count: usize, width: u32, height: u32) -> f32 {
    if node_count == 0 {
        return DEFAULT_NODE_RADIUS_PX;
    }
    let pixels_per_node = f64::from(width) * f64::from(height) / node_count as f64;
    (0.5 * pixels_per_node.sqrt()).clamp(0.75, f64::from(DEFAULT_NODE_RADIUS_PX)) as f32
}

pub fn select_edges(
    edges: &[(usize, usize)],
    max_edges: Option<usize>,
    seed: u64,
) -> Vec<(usize, usize)> {
    let Some(max) = max_edges else {
        return edges.to_vec();
    };
    if max >= edges.len() {
        return edges.to_vec();
    }
    let chosen = crate::stress::sample_indices(edges.len(), max, seed);
    chosen.into_iter().map(|i| edges[i]).collect()
}

fn bounds(positions: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    positions.iter().fold(
        (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), &[x, y]| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_buffer_is_two_u32_per_edge() {
        let edges = vec![(0, 2), (2, 1)];
        let indices = build_edge_indices(&edges).unwrap();
        assert_eq!(indices, vec![0, 2, 2, 1]);
        assert_eq!(
            indices.len() * size_of::<u32>(),
            edges.len() * 2 * size_of::<u32>()
        );
    }

    #[test]
    fn large_graph_uses_smaller_nodes() {
        assert!(auto_node_radius(250_000, 2048, 2048) < 4.0);
    }

    #[test]
    fn all_edges_are_retained_without_limit() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        assert_eq!(select_edges(&edges, None, 0), edges);
        assert_eq!(select_edges(&edges, Some(10), 0), edges);
    }

    #[test]
    fn limited_edges_are_repeatable() {
        let edges: Vec<_> = (0..100).map(|i| (i, i + 1)).collect();
        let a = select_edges(&edges, Some(10), 5);
        assert_eq!(a, select_edges(&edges, Some(10), 5));
        assert_eq!(a.len(), 10);
    }

    #[test]
    fn paper_scale_cpu_contract() {
        const NODES: usize = 250_000;
        const EDGES: usize = 1_941_926;
        assert_eq!(
            crate::stress::StressMode::Auto.resolve(NODES),
            crate::stress::StressMode::Sampled
        );
        let edges: Vec<_> = (0..EDGES)
            .map(|i| {
                let u = i % NODES;
                let v = (u + 1 + (i / NODES) * 7_919) % NODES;
                (u, v)
            })
            .collect();
        let indices = build_edge_indices(&edges).unwrap();
        assert_eq!(indices.len(), EDGES * 2);
        assert_eq!(indices.len() * size_of::<u32>(), EDGES * 8);
        assert_eq!(select_edges(&edges, None, 0).len(), EDGES);
    }
}

// One BFS per component, using CSR adjacency: O(V+E) time and memory.
// Independent of coordinates and edge order, so before/after retain node identity.
pub fn topology_colors(n: usize, edges: &[(usize, usize)], enabled: bool) -> Vec<f32> {
    let mut colors = vec![0.5; n];
    if !enabled {
        return colors;
    }
    let mut offsets = vec![0usize; n + 1];
    for &(a, b) in edges {
        offsets[a + 1] += 1;
        offsets[b + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets[..n].to_vec();
    let mut neighbors = vec![0usize; offsets[n]];
    for &(a, b) in edges {
        neighbors[cursor[a]] = b;
        cursor[a] += 1;
        neighbors[cursor[b]] = a;
        cursor[b] += 1;
    }
    let mut distance = vec![usize::MAX; n];
    let mut queue = Vec::new();
    for root in 0..n {
        if distance[root] != usize::MAX {
            continue;
        }
        queue.clear();
        queue.push(root);
        distance[root] = 0;
        let mut head = 0;
        let mut max_distance = 0;
        while head < queue.len() {
            let a = queue[head];
            head += 1;
            for &b in &neighbors[offsets[a]..offsets[a + 1]] {
                if distance[b] == usize::MAX {
                    distance[b] = distance[a] + 1;
                    max_distance = max_distance.max(distance[b]);
                    queue.push(b);
                }
            }
        }
        for &v in &queue {
            colors[v] = if max_distance == 0 {
                0.5
            } else {
                distance[v] as f32 / max_distance as f32
            };
        }
    }
    colors
}

#[cfg(test)]
mod palette_tests {
    use super::topology_colors;

    #[test]
    fn colors_follow_hops_and_handle_disconnected_and_isolated_nodes() {
        let edges = [(0, 1), (1, 2), (3, 4)];
        assert_eq!(
            topology_colors(6, &edges, true),
            vec![0.0, 0.5, 1.0, 0.0, 1.0, 0.5]
        );
        let reversed = [(4, 3), (2, 1), (1, 0)];
        assert_eq!(
            topology_colors(6, &edges, true),
            topology_colors(6, &reversed, true)
        );
        assert_eq!(topology_colors(6, &edges, false), vec![0.5; 6]);
    }
}

/// Layout-space density on a fixed normalized grid, independent of output resolution.
/// A separable Gaussian smooths the histogram; bilinear sampling avoids hard cell borders.
/// Only the densest ~20% of nodes receive full color. This is not graph degree or stress.
pub fn density_colors(positions: &[[f32; 2]]) -> Vec<f32> {
    if positions.is_empty() {
        return Vec::new();
    }
    const SIDE: usize = 512;
    let (min_x, max_x, min_y, max_y) = bounds(positions);
    let project = |p: &[f32; 2]| {
        [
            ((p[0] - min_x) / (max_x - min_x).max(1e-6) * (SIDE - 1) as f32)
                .clamp(0.0, (SIDE - 1) as f32),
            ((p[1] - min_y) / (max_y - min_y).max(1e-6) * (SIDE - 1) as f32)
                .clamp(0.0, (SIDE - 1) as f32),
        ]
    };
    let mut grid = vec![0.0f32; SIDE * SIDE];
    for p in positions {
        let [x, y] = project(p);
        grid[y.round() as usize * SIDE + x.round() as usize] += 1.0;
    }
    let weights = [1.0, 4.0, 6.0, 4.0, 1.0];
    let mut temp = vec![0.0f32; grid.len()];
    for horizontal in [true, false] {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let mut sum = 0.0;
                for (k, w) in weights.iter().enumerate() {
                    let delta = k as isize - 2;
                    let xx = x as isize + if horizontal { delta } else { 0 };
                    let yy = y as isize + if horizontal { 0 } else { delta };
                    if (0..SIDE as isize).contains(&xx) && (0..SIDE as isize).contains(&yy) {
                        sum += grid[yy as usize * SIDE + xx as usize] * w;
                    }
                }
                temp[y * SIDE + x] = sum / 16.0;
            }
        }
        std::mem::swap(&mut grid, &mut temp);
    }
    let density: Vec<f32> = positions
        .iter()
        .map(|p| {
            let [x, y] = project(p);
            let (ix, iy) = (x as usize, y as usize);
            let (jx, jy) = ((ix + 1).min(SIDE - 1), (iy + 1).min(SIDE - 1));
            let (fx, fy) = (x - ix as f32, y - iy as f32);
            let a = grid[iy * SIDE + ix] * (1.0 - fx) + grid[iy * SIDE + jx] * fx;
            let b = grid[jy * SIDE + ix] * (1.0 - fx) + grid[jy * SIDE + jx] * fx;
            (a * (1.0 - fy) + b * fy).ln_1p()
        })
        .collect();
    let mut sorted = density.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let low = sorted[(sorted.len() - 1) * 70 / 100];
    let threshold = sorted[(sorted.len() - 1) * 80 / 100];
    let high = sorted[(sorted.len() - 1) * 995 / 1000];
    if high - low < 1e-5 {
        return vec![-1.0; positions.len()];
    }
    density
        .iter()
        .map(|&d| {
            if d <= low {
                -1.0
            } else if d < threshold {
                -1.0 + (d - low) / (threshold - low).max(1e-5)
            } else {
                ((d - threshold) / (high - threshold).max(1e-5)).clamp(0.0, 1.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod density_tests {
    use super::density_colors;
    #[test]
    fn dense_cluster_is_colored_and_sparse_nodes_are_gray() {
        let mut points: Vec<_> = (0..900)
            .map(|i| [(i % 30) as f32, (i / 30) as f32])
            .collect();
        points.extend((0..100).map(|i| {
            [
                14.0 + (i % 10) as f32 * 0.001,
                14.0 + (i / 10) as f32 * 0.001,
            ]
        }));
        let colors = density_colors(&points);
        assert!(colors[..900].iter().filter(|&&c| c < 0.0).count() > 600);
        assert!(colors[950] > 0.5);
        assert!(colors
            .iter()
            .all(|c| c.is_finite() && *c >= -1.0 && *c <= 1.0));
        let scaled: Vec<_> = points.iter().map(|p| [p[0] * 2.0, p[1] * 2.0]).collect();
        assert_eq!(colors, density_colors(&scaled));
    }
    #[test]
    fn coincident_nodes_do_not_invent_density_contrast() {
        assert_eq!(density_colors(&[[1.0, 1.0]; 10]), vec![-1.0; 10]);
        assert!(density_colors(&[]).is_empty());
    }
}
