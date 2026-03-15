// GPU Graph Visualizer - Render Shaders
//
// Vertex shader: maps graph-space (x, y) → NDC via uniform bounds
// fs_edge: semi-transparent blue for edges
// fs_node: opaque orange-red for nodes

struct Uniforms {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    let range_x = u.max_x - u.min_x;
    let range_y = u.max_y - u.min_y;
    // Map to NDC [-1, 1]. Y is negated so higher Y values appear at the top,
    // matching matplotlib's default axis orientation.
    let nx =  2.0 * (pos.x - u.min_x) / range_x - 1.0;
    let ny = -(2.0 * (pos.y - u.min_y) / range_y - 1.0);
    return vec4<f32>(nx, ny, 0.0, 1.0);
}

@fragment
fn fs_edge() -> @location(0) vec4<f32> {
    return vec4<f32>(0.35, 0.55, 0.85, 0.55);
}

@fragment
fn fs_node() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.45, 0.1, 1.0);
}
