use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const FULL_INITIAL_POSITION_STREAM: u64 = 0x4655_4c4c_5f49_4e49;
pub const FULL_UPDATE_STREAM: u64 = 0x4655_4c4c_5f55_5044;

pub fn rng_for_stream(seed: u64, stream: u64) -> StdRng {
    StdRng::seed_from_u64(splitmix64(seed ^ stream))
}

pub fn seeded_positions(n: usize, seed: u64, center: bool) -> Vec<[f64; 2]> {
    let mut rng = rng_for_stream(seed, FULL_INITIAL_POSITION_STREAM);
    let mut positions: Vec<[f64; 2]> = (0..n)
        .map(|_| [rng.random::<f64>(), rng.random::<f64>()])
        .collect();
    if center && !positions.is_empty() {
        let mean_x = positions.iter().map(|position| position[0]).sum::<f64>() / n as f64;
        let mean_y = positions.iter().map(|position| position[1]).sum::<f64>() / n as f64;
        for position in &mut positions {
            position[0] -= mean_x;
            position[1] -= mean_y;
        }
    }
    positions
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::positions_sha256_f64;

    #[test]
    fn full_initial_positions_are_reproducible() {
        let first = seeded_positions(20, 7, true);
        let second = seeded_positions(20, 7, true);
        assert_eq!(positions_sha256_f64(&first), positions_sha256_f64(&second));
        assert_ne!(
            positions_sha256_f64(&first),
            positions_sha256_f64(&seeded_positions(20, 8, true))
        );
    }
}
