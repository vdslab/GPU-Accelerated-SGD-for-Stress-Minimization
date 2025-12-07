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

@group(0) @binding(4)
var<uniform> iteration: u32;

@group(0) @binding(5)
var<storage, read_write> locks: array<atomic<u32>>;

@group(0) @binding(6)
var<storage, read_write> debug_pairs: array<u32>;

@compute @workgroup_size(32,32,1)
fn sgd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = iteration;
    debug.val1 = f32(i);
    
    // LOG: Find pairs that involve node 2 (0-indexed) and record their partners
    let target_node = 2u;
    var node2_pair_count = 0u;
    
    for (var idx = 0u; idx < arrayLength(&pairs); idx++) {
        let pair = pairs[idx];
        var partner: u32 = 0xFFFFFFFFu; // Invalid value
        
        if (pair.u == target_node) {
            partner = pair.v;
        } else if (pair.v == target_node) {
            partner = pair.u;
        }
        
        if (partner != 0xFFFFFFFFu) {
            debug_pairs[node2_pair_count] = partner;
            node2_pair_count++;
        }
    }

}