from pathlib import Path
import difflib
root=Path(__file__).resolve().parents[3]
base=root/'implementations/full/rr_gpu'
out=base/'examples/wiki_full_cause'
out.mkdir(exist_ok=True)
source=(base/'src/gpu.rs').read_text()
gpu=source.replace('pub inner_shuffle: bool,','pub inner_shuffle: bool,\n    pub phase_chunks: u32,')
gpu=gpu.replace('inner_shuffle: true,','inner_shuffle: true,\n            phase_chunks: 1,')
gpu=gpu.replace('pub partial_self_perm_offset: u32,','pub partial_self_perm_offset: u32,\n    pub chunk: u32,\n    pub chunks: u32,\n    pub padding: [u32; 2],')
# Both initial and per-iteration uniform constructors.
gpu=gpu.replace('            partial_self_perm_offset,','            partial_self_perm_offset,\n            chunk: 0,\n            chunks: self.kernel.phase_chunks,\n            padding: [0; 2],')
needle='            for (round_position, &ri) in round_order.iter().enumerate() {'
replacement='''            let permutations: Vec<_> = round_order.iter().map(|_| {
                build_inner_permutation(tile_size, last_block_len,
                                        self.kernel.inner_shuffle, &mut rng)
            }).collect();
            for chunk in 0..self.kernel.phase_chunks {
                let chunk_uniforms = Uniforms { chunk, ..uni };
                self.queue.write_buffer(&uniforms_buffer, 0,
                                        bytemuck::bytes_of(&chunk_uniforms));
            for (round_position, &ri) in round_order.iter().enumerate() {'''
assert gpu.count(needle)==1
gpu=gpu.replace(needle,replacement)
needle='''                let inner_perm = build_inner_permutation(
                    tile_size,
                    last_block_len,
                    self.kernel.inner_shuffle,
                    &mut rng,
                );'''
assert gpu.count(needle)==1
gpu=gpu.replace(needle,'                let inner_perm = &permutations[round_position];')
gpu=gpu.replace('(round_position == 0).then_some(timestamp_index)', '(chunk == 0 && round_position == 0).then_some(timestamp_index)')
gpu=gpu.replace('(round_position + 1 == round_order.len())', '(chunk + 1 == self.kernel.phase_chunks && round_position + 1 == round_order.len())')
gpu=gpu.replace('''                self.queue.submit([encoder.finish()]);
            }
            // Queue writes''','''                self.queue.submit([encoder.finish()]);
            }
            }
            // Queue writes''')
gpu=gpu.replace('(schedule.len() * num_iterations) as u64','(schedule.len() * num_iterations) as u64 * self.kernel.phase_chunks as u64')
(out/'gpu.rs').write_text(gpu)
shader_source=(base/'src/shader.wgsl').read_text()
shader=shader_source.replace('    partial_self_perm_offset: u32,','    partial_self_perm_offset: u32,\n    chunk: u32,\n    chunks: u32,\n    padding: vec2<u32>,')
shader=shader.replace('for (var s = 0u; s < BLOCK_T; s++)', 'for (var s = uniforms.chunk * BLOCK_T / uniforms.chunks; s < (uniforms.chunk + 1u) * BLOCK_T / uniforms.chunks; s++)')
shader=shader.replace('for (var s = 0u; s < rounds; s++)','for (var s = uniforms.chunk * rounds / uniforms.chunks; s < (uniforms.chunk + 1u) * rounds / uniforms.chunks; s++)')
(out/'shader.wgsl').write_text(shader)
patch=''.join(difflib.unified_diff(source.splitlines(True),gpu.splitlines(True),fromfile='production/gpu.rs',tofile='diagnostic/gpu.rs'))
patch+=''.join(difflib.unified_diff(shader_source.splitlines(True),shader.splitlines(True),fromfile='production/shader.wgsl',tofile='diagnostic/shader.wgsl'))
(Path(__file__).parent/'gpu-chunks.patch').write_text(patch)
