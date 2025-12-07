struct EdgeInfo {
    u: u32,
    v: u32,
    dij: f32,
    wij: f32,
}

struct Debug {
    val1: f32,
    val2: f32,
    val3: f32,
}

@group(0) @binding(0)
var<storage, read> etas: array<f32>;

@group(0) @binding(1)
var<storage, read_write> positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> pairs: array<EdgeInfo>;

@group(0) @binding(3)
var<storage, read_write> debug: Debug;

@compute @workgroup_size(32,32,1)
fn sgd(@builtin(global_invocation_id) global_id: vec3<u32>) {}