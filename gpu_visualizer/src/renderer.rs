use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Default node radius in pixels for the output image.
pub const DEFAULT_NODE_RADIUS_PX: f32 = 4.0;

/// Uniform buffer sent to both shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    min_x:    f32,
    max_x:    f32,
    min_y:    f32,
    max_y:    f32,
    node_r_x: f32, // radius in NDC x = radius_px * 2 / width
    node_r_y: f32, // radius in NDC y = radius_px * 2 / height
    _pad0:    f32,
    _pad1:    f32,
}

// Unit quad: two CCW triangles covering [-1,1]²
// Six vertices, each vec2<f32>
const QUAD_VERTS: [[f32; 2]; 6] = [
    [-1.0, -1.0], [1.0, -1.0], [1.0,  1.0],
    [-1.0, -1.0], [1.0,  1.0], [-1.0, 1.0],
];

pub struct GpuRenderer {
    device: wgpu::Device,
    queue:  wgpu::Queue,
}

impl GpuRenderer {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|e| anyhow::anyhow!("GPU adapter not found: {:?}", e))?;

        println!(
            "GPU: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label:                None,
                required_features:    wgpu::Features::empty(),
                required_limits:      wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints:         wgpu::MemoryHints::MemoryUsage,
                trace:                wgpu::Trace::Off,
            },
        ))
        .map_err(|e| anyhow::anyhow!("Device creation failed: {}", e))?;

        Ok(GpuRenderer { device, queue })
    }

    /// Render graph to a tightly-packed RGBA8 pixel buffer (top-to-bottom).
    pub fn render(
        &self,
        positions:      &[[f32; 2]],
        edges:          &[(usize, usize)],
        width:          u32,
        height:         u32,
        node_radius_px: f32,
    ) -> Result<Vec<u8>> {
        // ── Bounds + padding ──────────────────────────────────────────────────
        let (min_x, max_x, min_y, max_y) = bounds(positions);
        let pad_x = (max_x - min_x).max(1e-6) * 0.05;
        let pad_y = (max_y - min_y).max(1e-6) * 0.05;

        let uniforms = Uniforms {
            min_x:    min_x - pad_x,
            max_x:    max_x + pad_x,
            min_y:    min_y - pad_y,
            max_y:    max_y + pad_y,
            node_r_x: node_radius_px * 2.0 / width  as f32,
            node_r_y: node_radius_px * 2.0 / height as f32,
            _pad0:    0.0,
            _pad1:    0.0,
        };

        // ── Shader ────────────────────────────────────────────────────────────
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("graph_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // ── Uniform buffer + bind group ───────────────────────────────────────
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout =
            self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                None,
                bind_group_layouts:   &[&bgl],
                push_constant_ranges: &[],
            });

        let texture_format = wgpu::TextureFormat::Rgba8Unorm;

        // ── Shared attribute definition ────────────────────────────────────────
        let vec2_attr = [wgpu::VertexAttribute {
            format:           wgpu::VertexFormat::Float32x2,
            offset:           0,
            shader_location:  0,
        }];
        let vec2_attr_slot1 = [wgpu::VertexAttribute {
            format:           wgpu::VertexFormat::Float32x2,
            offset:           0,
            shader_location:  1,
        }];

        // ── Edge pipeline: LINE_LIST + alpha blending ─────────────────────────
        let edge_pipeline =
            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("edge_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module:      &module,
                    entry_point: Some("vs_edge"),
                    buffers:     &[wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode:    wgpu::VertexStepMode::Vertex,
                        attributes:   &vec2_attr,
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &module,
                    entry_point: Some("fs_edge"),
                    targets:     &[Some(wgpu::ColorTargetState {
                        format:     texture_format,
                        blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
                cache:         None,
            });

        // ── Node pipeline: instanced quads + alpha blending ───────────────────
        // slot 0 = quad corner offsets (Vertex step)
        // slot 1 = node center positions (Instance step)
        let node_pipeline =
            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("node_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module:      &module,
                    entry_point: Some("vs_node"),
                    buffers:     &[
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode:    wgpu::VertexStepMode::Vertex,
                            attributes:   &vec2_attr,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: 8,
                            step_mode:    wgpu::VertexStepMode::Instance,
                            attributes:   &vec2_attr_slot1,
                        },
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &module,
                    entry_point: Some("fs_node"),
                    targets:     &[Some(wgpu::ColorTargetState {
                        format:     texture_format,
                        blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
                cache:         None,
            });

        // ── Vertex buffers ─────────────────────────────────────────────────────
        // Edge: flatten each (u,v) pair into two consecutive positions
        let edge_verts: Vec<[f32; 2]> = edges
            .iter()
            .flat_map(|&(u, v)| [positions[u], positions[v]])
            .collect();

        let edge_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("edge_verts"),
            contents: bytemuck::cast_slice(&edge_verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        // Unit quad (shared by all node instances)
        let quad_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("quad_verts"),
            contents: bytemuck::cast_slice(&QUAD_VERTS),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        // Node instance buffer
        let node_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("node_instances"),
            contents: bytemuck::cast_slice(positions),
            usage:    wgpu::BufferUsages::VERTEX,
        });

        // ── Off-screen render texture ──────────────────────────────────────────
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label:             Some("render_texture"),
            size:              wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count:   1,
            sample_count:      1,
            dimension:         wgpu::TextureDimension::D2,
            format:            texture_format,
            usage:             wgpu::TextureUsages::RENDER_ATTACHMENT
                             | wgpu::TextureUsages::COPY_SRC,
            view_formats:      &[],
        });
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // bytes_per_row must be a multiple of 256 for COPY_SRC → buffer
        let bytes_per_row = (width * 4).next_multiple_of(256);
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               (bytes_per_row * height) as u64,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ── Render pass ────────────────────────────────────────────────────────
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &tex_view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        // White background
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0, g: 1.0, b: 1.0, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });

            pass.set_bind_group(0, &bind_group, &[]);

            // Pass 1: edges
            if !edge_verts.is_empty() {
                pass.set_pipeline(&edge_pipeline);
                pass.set_vertex_buffer(0, edge_buf.slice(..));
                pass.draw(0..edge_verts.len() as u32, 0..1);
            }

            // Pass 2: nodes (instanced quads)
            pass.set_pipeline(&node_pipeline);
            pass.set_vertex_buffer(0, quad_buf.slice(..));
            pass.set_vertex_buffer(1, node_buf.slice(..));
            pass.draw(0..QUAD_VERTS.len() as u32, 0..positions.len() as u32);
        }

        // ── Copy texture → staging ─────────────────────────────────────────────
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset:          0,
                    bytes_per_row:   Some(bytes_per_row),
                    rows_per_image:  Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        self.queue.submit([encoder.finish()]);

        // ── Readback ───────────────────────────────────────────────────────────
        let buf_slice = staging_buf.slice(..);
        buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let data = buf_slice.get_mapped_range();

        // Strip row padding → tightly-packed RGBA8
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            pixels.extend_from_slice(&data[start..start + (width * 4) as usize]);
        }

        Ok(pixels)
    }
}

fn bounds(positions: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &[x, y] in positions {
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }
    (min_x, max_x, min_y, max_y)
}
