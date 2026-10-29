struct Uniforms {
    n: u32,
    start: u32,
    count: u32,
    iteration: u32,
    eta: f32,
    seed: u32,
    pivot_count: u32,
    pad: u32,
    round_mask: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

struct OneSidedEntry {
    dij: f32,
    weight: f32,
}

struct Pair {
    u: u32,
    v: u32,
    dij: f32,
    weight_u: f32,
    weight_v: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> one_sided: array<OneSidedEntry>;
@group(0) @binding(3) var<storage, read> pivot_permutation: array<u32>;
@group(0) @binding(4) var<storage, read> pairs: array<Pair>;

// The all-ones mask is supplied at runtime. Crossing the integer operation
// forces an f32 rounding boundary, preventing algebraic removal of residuals.
fn rounded(value: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(value) & uniforms.round_mask);
}
fn rounded2(value: vec2<f32>) -> vec2<f32> {
    return bitcast<vec2<f32>>(bitcast<vec2<u32>>(value) & vec2<u32>(uniforms.round_mask));
}

fn hash32(value: u32) -> u32 {
    var x = value;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    return (x >> 16u) ^ x;
}

// A scalar is (high, low); a point is (x_high, y_high, x_low, y_low).
fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = rounded(a.x + b.x);
    let v = rounded(s - a.x);
    let error = rounded(rounded(a.x - rounded(s - v)) + rounded(b.x - v));
    let e = rounded(rounded(error + a.y) + b.y);
    let h = rounded(s + e);
    return vec2<f32>(h, rounded(e - rounded(h - s)));
}
fn ds_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = rounded(a.x * b.x);
    let error = rounded(fma(a.x, b.x, -p));
    let cross = rounded(rounded(a.x * b.y) + rounded(a.y * b.x));
    let e = rounded(rounded(error + cross) + rounded(a.y * b.y));
    let h = rounded(p + e);
    return vec2<f32>(h, rounded(e - rounded(h - p)));
}
fn ds_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q = rounded(a.x / b.x);
    let r = ds_add(a, -ds_mul(b, vec2<f32>(q, 0.0)));
    return ds_add(vec2<f32>(q, 0.0), vec2<f32>(rounded(rounded(r.x + r.y) / b.x), 0.0));
}
fn ds_sqrt(a: vec2<f32>) -> vec2<f32> {
    let q = rounded(sqrt(a.x));
    if q == 0.0 { return vec2<f32>(0.0); }
    let r = ds_add(a, -ds_mul(vec2<f32>(q, 0.0), vec2<f32>(q, 0.0)));
    return ds_add(vec2<f32>(q, 0.0), vec2<f32>(rounded(rounded(r.x + r.y) / rounded(2.0 * q)), 0.0));
}
fn point_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let x = ds_add(a.xz, b.xz);
    let y = ds_add(a.yw, b.yw);
    return vec4<f32>(x.x, y.x, x.y, y.y);
}
fn point_scale(a: vec4<f32>, b: vec2<f32>) -> vec4<f32> {
    let x = ds_mul(a.xz, b);
    let y = ds_mul(a.yw, b);
    return vec4<f32>(x.x, y.x, x.y, y.y);
}
fn displacement(pos_u: vec4<f32>, pos_v: vec4<f32>, dij: f32, salt: u32) -> vec4<f32> {
    var difference = point_add(pos_v, -pos_u);
    var norm = ds_sqrt(ds_add(ds_mul(difference.xz, difference.xz), ds_mul(difference.yw, difference.yw)));
    if norm.x < 1e-12 {
        let angle = f32(hash32(salt) % 6283u) * 0.001;
        difference = vec4<f32>(vec2<f32>(cos(angle), sin(angle)) * 1e-6, 0.0, 0.0);
        norm = ds_sqrt(ds_add(ds_mul(difference.xz, difference.xz), ds_mul(difference.yw, difference.yw)));
    }
    let scale = ds_div(ds_add(norm, vec2<f32>(-dij, 0.0)), norm * 2.0);
    return point_scale(difference, scale);
}

// One invocation owns one non-pivot vertex. It is the only writer of that vertex;
// pivots are read-only throughout this dispatch.
@compute @workgroup_size(256)
fn one_sided_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex = gid.x;
    if vertex >= uniforms.n {
        return;
    }
    var local_position = positions[vertex];
    var changed = false;
    let rotation = hash32(vertex ^ uniforms.seed ^ uniforms.iteration) % uniforms.pivot_count;
    var stride = 1u;
    if uniforms.pad != 0u && uniforms.pivot_count > 1u {
        stride = 1u + hash32(vertex ^ uniforms.seed ^ (uniforms.iteration * 0x9e3779b9u)) % (uniforms.pivot_count - 1u);
        loop {
            var a = stride;
            var b = uniforms.pivot_count;
            loop {
                if b == 0u { break; }
                let t = a % b;
                a = b;
                b = t;
            }
            if a == 1u { break; }
            stride = stride % (uniforms.pivot_count - 1u) + 1u;
        }
    }
    for (var k = uniforms.start; k < uniforms.start + uniforms.count; k++) {
        let pivot_index = pivot_permutation[(k * stride + rotation) % uniforms.pivot_count];
        let entry = one_sided[vertex * uniforms.pivot_count + pivot_index];
        if entry.weight > 0.0 {
            let pivot = pivot_permutation[uniforms.pivot_count + pivot_index];
            let pivot_position = positions[pivot];
            changed = true;
            let delta = displacement(
                local_position,
                pivot_position,
                entry.dij,
                vertex ^ pivot ^ uniforms.seed ^ uniforms.iteration,
            );
            let mu = min(entry.weight * uniforms.eta, 1.0);
            local_position = point_add(local_position, point_scale(delta, vec2<f32>(mu, 0.0)));
        }
    }
    // Pivot rows have no positive one-sided entries: never write their coordinates.
    if changed {
        positions[vertex] = local_position;
    }
}

// The host guarantees that each dispatched slice is a matching, so both endpoint
// writes are race-free without atomics or locks.
@compute @workgroup_size(256)
fn two_sided_round(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= uniforms.count {
        return;
    }
    let pair_index = uniforms.start + gid.x;
    let pair = pairs[pair_index];
    let pos_u = positions[pair.u];
    let pos_v = positions[pair.v];
    let delta = displacement(
        pos_u,
        pos_v,
        pair.dij,
        pair.u ^ pair.v ^ uniforms.seed ^ uniforms.iteration,
    );
    let mu_u = min(pair.weight_u * uniforms.eta, 1.0);
    let mu_v = min(pair.weight_v * uniforms.eta, 1.0);
    positions[pair.u] = point_add(pos_u, point_scale(delta, vec2<f32>(mu_u, 0.0)));
    positions[pair.v] = point_add(pos_v, point_scale(delta, vec2<f32>(-mu_v, 0.0)));
}
