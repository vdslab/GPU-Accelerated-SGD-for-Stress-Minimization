// Appended only to the original f32 shader for the recentering diagnostic.
// The scratch vector follows the two guard vectors; it has a single writer.
var<workgroup> center_partials: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn reduce_center(@builtin(local_invocation_index) lane: u32) {
    var sum = vec2<f32>(0.0);
    for (var vertex = lane; vertex < uniforms.n; vertex += 256u) {
        sum += positions[vertex];
    }
    center_partials[lane] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if lane < stride { center_partials[lane] += center_partials[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u { positions[uniforms.n + 2u] = center_partials[0] / f32(uniforms.n); }
}

@compute @workgroup_size(256)
fn subtract_center(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < uniforms.n { positions[gid.x] -= positions[uniforms.n + 2u]; }
}
