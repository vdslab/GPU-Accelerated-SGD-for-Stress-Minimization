struct Params {
    width: u32,
}

struct Debug {
    val1:f32,
    val2:f32,
    val3:f32,
}
@group(0) @binding(0)
var<uniform> params: Params;
@group(0) @binding(1)
var<storage, read> input: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;
@group(0) @binding(3)
var<storage, read_write> debug: Debug;

@compute @workgroup_size(16,16,1)
fn copyToOutput(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.y * params.width + global_id.x;
    let array_length = arrayLength(&input);
    if (index >= array_length) {
        return;
    } 
    debug.val1 = input[1];
    debug.val2 = input[2];
    debug.val3 = input[3];
    output[index] = input[index];
}