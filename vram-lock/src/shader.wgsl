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

@group(0) @binding(7)
var<storage, read> pair_map: array<u32>;

// Debug helper function (called only by thread 0,0)
fn log_debug_info() {
    debug.val1 = f32(iteration);
    
    // Count pairs that involve node 2
    let target_node = 2u;
    var node2_pair_count = 0u;
    
    for (var idx = 0u; idx < arrayLength(&pairs); idx++) {
        let p = pairs[idx];
        var partner: u32 = 0xFFFFFFFFu;
        
        if (p.u == target_node) {
            partner = p.v;
        } else if (p.v == target_node) {
            partner = p.u;
        }
        
        if (partner != 0xFFFFFFFFu) {
            debug_pairs[node2_pair_count] = partner;
            node2_pair_count++;
        }
    }
    
    debug.val2 = f32(node2_pair_count);
}

@compute @workgroup_size(32,32,1)
fn sgd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_size = arrayLength(&positions);
    let i = global_id.y;
    let j = global_id.x;
    
    // Debug: Only thread (0,0) logs
    if (global_id.x == 0u && global_id.y == 0u) {
        log_debug_info();
    }
    
    // Early return if out of bounds or not upper triangle
    if (i >= node_size || j >= node_size || i >= j) {
        return;
    }
    
    // Look up pair using pair_map (O(1) instead of O(n) scan)
    let map_idx = i * node_size + j;
    let pair_idx = pair_map[map_idx];
    
    // If pair not found (0xFFFFFFFF), return
    if (pair_idx == 0xFFFFFFFFu) {
        return;
    }
    
    // Get pair information directly
    let pair = pairs[pair_idx];
    let dij = pair.dij;
    let wij = pair.wij;
    
    // Get learning rate for this iteration
    let eta = etas[iteration];
    
    // SGD update (matching Python implementation)
    let tiny = 1e-12;
    var diff = positions[j] - positions[i];
    var dist = length(diff);
    
    // Handle zero/tiny distance case
    if (dist < tiny) {
        diff = vec2<f32>(1e-6, 1e-6);
        dist = length(diff);
    }
    
    let r = ((dist - dij) / 2.0) * (diff / dist);
    let mu = min(wij * eta, 1.0);
    
    positions[i] += mu * r;
    positions[j] -= mu * r;
}