use bytemuck::{Pod, Zeroable};
use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

pub const TILE_SIZE: u32 = 32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Pair {
    pub u: u32,
    pub v: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedPair {
    pub pair: Pair,
    pub weight: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DispatchRange {
    pub color: u32,
    pub round: u32,
    pub start: u32,
    pub len: u32,
}

#[derive(Debug, Clone)]
pub struct ScheduledPairs {
    pub packed_pairs: Vec<Pair>,
    pub packed_weights: Vec<f32>,
    pub dispatch_ranges: Vec<DispatchRange>,
}

pub fn calc_offdiag_round(u_local: u32, v_local: u32) -> u32 {
    (v_local + TILE_SIZE - u_local) % TILE_SIZE
}

pub fn calc_color(tile_x: u32, tile_y: u32, num_tiles: u32) -> u32 {
    assert_ne!(tile_x, tile_y);
    assert!(tile_x < num_tiles);
    assert!(tile_y < num_tiles);

    let a = tile_x.min(tile_y);
    let b = tile_x.max(tile_y);
    let b_eff = if num_tiles % 2 == 0 {
        num_tiles
    } else {
        num_tiles + 1
    };
    let fixed = b_eff - 1;

    for round in 0..(b_eff - 1) {
        let opp = round;
        if fixed < num_tiles && opp < num_tiles {
            let u = fixed.min(opp);
            let v = fixed.max(opp);
            if u == a && v == b {
                return round;
            }
        }

        for k in 1..(b_eff / 2) {
            let u = (round + k) % (b_eff - 1);
            let v = (round + b_eff - 1 - k) % (b_eff - 1);
            if u < num_tiles && v < num_tiles {
                let x = u.min(v);
                let y = u.max(v);
                if x == a && y == b {
                    return round;
                }
            }
        }
    }

    unreachable!("tile pair must appear in round-robin coloring")
}

pub fn offdiag_color_count(num_tiles: u32) -> u32 {
    if num_tiles <= 1 {
        0
    } else if num_tiles % 2 == 0 {
        num_tiles - 1
    } else {
        num_tiles
    }
}

pub fn calc_diag_round(u_local: u32, v_local: u32) -> Option<u32> {
    if u_local == v_local {
        return None;
    }
    assert!(u_local < TILE_SIZE);
    assert!(v_local < TILE_SIZE);

    let table = diag_round_table();
    let round = table[u_local as usize][v_local as usize];
    if round == u32::MAX {
        None
    } else {
        Some(round)
    }
}

pub fn classify_and_pack(pairs: &[WeightedPair], n: usize) -> ScheduledPairs {
    let num_tiles = (n as u32).div_ceil(TILE_SIZE);
    let diag_color = offdiag_color_count(num_tiles);
    let mut buffers: BTreeMap<(u32, u32), Vec<WeightedPair>> = BTreeMap::new();

    for pair in pairs {
        let u = pair.pair.u;
        let v = pair.pair.v;
        if u == v || u as usize >= n || v as usize >= n {
            continue;
        }

        let tile_x = u / TILE_SIZE;
        let tile_y = v / TILE_SIZE;
        let u_local = u % TILE_SIZE;
        let v_local = v % TILE_SIZE;

        let (color, round) = if tile_x == tile_y {
            match calc_diag_round(u_local, v_local) {
                Some(round) => (diag_color, round),
                None => continue,
            }
        } else {
            (
                calc_color(tile_x, tile_y, num_tiles),
                calc_offdiag_round(u_local, v_local),
            )
        };

        buffers.entry((color, round)).or_default().push(*pair);
    }

    let mut packed_pairs = Vec::with_capacity(pairs.len());
    let mut packed_weights = Vec::with_capacity(pairs.len());
    let mut dispatch_ranges = Vec::new();

    for ((color, round), bucket) in buffers {
        let start = packed_pairs.len() as u32;
        for pair in bucket {
            packed_pairs.push(pair.pair);
            packed_weights.push(pair.weight);
        }
        let len = packed_pairs.len() as u32 - start;
        if len > 0 {
            dispatch_ranges.push(DispatchRange {
                color,
                round,
                start,
                len,
            });
        }
    }

    ScheduledPairs {
        packed_pairs,
        packed_weights,
        dispatch_ranges,
    }
}

pub fn validate_dispatch_ranges_conflict_free(schedule: &ScheduledPairs) -> Result<(), String> {
    for range in &schedule.dispatch_ranges {
        let start = range.start as usize;
        let end = start + range.len as usize;
        let mut seen = HashSet::new();

        for pair in &schedule.packed_pairs[start..end] {
            if !seen.insert(pair.u) {
                return Err(format!(
                    "node {} appears more than once in color={}, round={}",
                    pair.u, range.color, range.round
                ));
            }
            if !seen.insert(pair.v) {
                return Err(format!(
                    "node {} appears more than once in color={}, round={}",
                    pair.v, range.color, range.round
                ));
            }
        }
    }

    Ok(())
}

fn diag_round_table() -> &'static [[u32; TILE_SIZE as usize]; TILE_SIZE as usize] {
    static TABLE: OnceLock<[[u32; TILE_SIZE as usize]; TILE_SIZE as usize]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[u32::MAX; TILE_SIZE as usize]; TILE_SIZE as usize];
        let fixed = TILE_SIZE - 1;

        for round in 0..(TILE_SIZE - 1) {
            fill_diag_pair(&mut table, fixed, round, round);
            for k in 1..(TILE_SIZE / 2) {
                let u = (round + k) % (TILE_SIZE - 1);
                let v = (round + TILE_SIZE - 1 - k) % (TILE_SIZE - 1);
                fill_diag_pair(&mut table, u, v, round);
            }
        }

        table
    })
}

fn fill_diag_pair(
    table: &mut [[u32; TILE_SIZE as usize]; TILE_SIZE as usize],
    u: u32,
    v: u32,
    round: u32,
) {
    table[u as usize][v as usize] = round;
    table[v as usize][u as usize] = round;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offdiag_rounds_do_not_conflict_locally() {
        let mut buckets = vec![Vec::new(); TILE_SIZE as usize];
        for u in 0..TILE_SIZE {
            for v in 0..TILE_SIZE {
                buckets[calc_offdiag_round(u, v) as usize].push((u, v));
            }
        }

        for bucket in buckets {
            let mut seen_u = HashSet::new();
            let mut seen_v = HashSet::new();
            for (u, v) in bucket {
                assert!(seen_u.insert(u));
                assert!(seen_v.insert(v));
            }
        }
    }

    #[test]
    fn diag_rounds_form_matchings() {
        let mut round_counts = [0usize; (TILE_SIZE - 1) as usize];
        let mut covered = HashSet::new();

        for u in 0..TILE_SIZE {
            assert_eq!(calc_diag_round(u, u), None);
            for v in (u + 1)..TILE_SIZE {
                let round = calc_diag_round(u, v).unwrap();
                assert!(round < TILE_SIZE - 1);
                round_counts[round as usize] += 1;
                covered.insert((u, v));
            }
        }

        assert_eq!(covered.len(), (TILE_SIZE * (TILE_SIZE - 1) / 2) as usize);
        for (round, &count) in round_counts.iter().enumerate() {
            assert_eq!(count, (TILE_SIZE / 2) as usize);
            let mut seen = HashSet::new();
            for u in 0..TILE_SIZE {
                for v in (u + 1)..TILE_SIZE {
                    if calc_diag_round(u, v) == Some(round as u32) {
                        assert!(seen.insert(u));
                        assert!(seen.insert(v));
                    }
                }
            }
        }
    }

    #[test]
    fn colors_do_not_share_tiles() {
        for num_tiles in 2..20 {
            let mut buckets: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
            for x in 0..num_tiles {
                for y in (x + 1)..num_tiles {
                    let color = calc_color(x, y, num_tiles);
                    buckets.entry(color).or_default().push((x, y));
                }
            }

            for bucket in buckets.values() {
                let mut seen = HashSet::new();
                for &(x, y) in bucket {
                    assert!(seen.insert(x));
                    assert!(seen.insert(y));
                }
            }
        }
    }

    #[test]
    fn packed_ranges_are_conflict_free_for_all_unordered_pairs() {
        let n = 70usize;
        let mut pairs = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                pairs.push(WeightedPair {
                    pair: Pair {
                        u: u as u32,
                        v: v as u32,
                    },
                    weight: 1.0,
                });
            }
        }

        let schedule = classify_and_pack(&pairs, n);
        validate_dispatch_ranges_conflict_free(&schedule).unwrap();
        assert_eq!(schedule.packed_pairs.len(), pairs.len());
    }
}
