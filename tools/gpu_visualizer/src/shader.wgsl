// GPU Graph Visualizer – Render Shaders
//
// Coordinate transform: graph-space → NDC via Uniforms bounding box.
// Y is negated so larger Y values appear at the top, matching matplotlib.
//
// Edge pass  : vs_edge / fs_edge   – LINE_LIST, semi-transparent neutral gray (sunset) or blue
// Node pass  : vs_node / fs_node   – instanced unit quads, circular clip

struct Uniforms {
    min_x:    f32,
    max_x:    f32,
    min_y:    f32,
    max_y:    f32,
    node_r_x: f32,   // node radius in NDC x  (radius_px * 2 / width)
    node_r_y: f32,   // node radius in NDC y  (radius_px * 2 / height)
    _pad0:    f32,
    _pad1:    f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

fn graph_to_ndc(pos: vec2<f32>) -> vec2<f32> {
    let nx =  2.0 * (pos.x - u.min_x) / (u.max_x - u.min_x) - 1.0;
    let ny = -(2.0 * (pos.y - u.min_y) / (u.max_y - u.min_y) - 1.0);
    return vec2<f32>(nx, ny);
}

// ── Edge pass ─────────────────────────────────────────────────────────────────

@vertex
fn vs_edge(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(graph_to_ndc(pos), 0.0, 1.0);
}

@fragment
fn fs_edge() -> @location(0) vec4<f32> {
    if u._pad0 > 0.5 {
        return vec4<f32>(0.48, 0.49, 0.53, 0.18);
    }
    // matplotlib default blue #1f77b4 at 70% opacity
    return vec4<f32>(0.122, 0.467, 0.706, 0.7);
}

// ── Node pass (instanced unit quad → circular clip) ───────────────────────────

struct NodeOut {
    @builtin(position) pos:    vec4<f32>,
    @location(0)       offset: vec2<f32>,
    @location(1)       color: vec3<f32>,  // quad corner in [-1,1]²
}

// @location(0) offset : per-vertex quad corner   (VertexStepMode::Vertex)
// @location(1) center : per-node graph position  (VertexStepMode::Instance)
@vertex
fn vs_node(
    @location(0) offset: vec2<f32>,
    @location(1) center: vec2<f32>,
    @location(2) shade: f32,
) -> NodeOut {
    let ndc_center = graph_to_ndc(center);
    let ndc_pos    = ndc_center + offset * vec2<f32>(u.node_r_x, u.node_r_y);
    var color = vec3<f32>(0.122, 0.467, 0.706);
    if u._pad0 > 0.5 {
        let t = clamp(shade, 0.0, 1.0) * 4.0;
        let palette = array<vec3<f32>, 5>(
            vec3<f32>(0.18, 0.08, 0.28),
            vec3<f32>(0.48, 0.18, 0.51),
            vec3<f32>(0.81, 0.28, 0.40),
            vec3<f32>(0.98, 0.57, 0.35),
            vec3<f32>(0.99, 0.93, 0.63)
        );
        let i = min(u32(t), 3u);
        color = mix(palette[i], palette[i + 1u], t - f32(i));
    }
    // Density mode encodes sparse nodes as -1, with a soft gray-to-color boundary.
    if shade < 0.0 {
        color = mix(vec3<f32>(0.72, 0.73, 0.75),
                    vec3<f32>(0.18, 0.08, 0.28), clamp(shade + 1.0, 0.0, 1.0));
    }
    return NodeOut(vec4<f32>(ndc_pos, 0.0, 1.0), offset, color);
}

@fragment
fn fs_node(in: NodeOut) -> @location(0) vec4<f32> {
    // Clip to circle
    if length(in.offset) > 1.0 {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}
