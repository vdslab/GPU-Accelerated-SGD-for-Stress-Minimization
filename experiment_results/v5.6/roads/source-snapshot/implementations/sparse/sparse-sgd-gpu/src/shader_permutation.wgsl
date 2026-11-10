// Four-round balanced Feistel permutation on the smallest power-of-four domain.
// Cycle walking restricts that bijection to [0, count), with no omissions/repeats.
fn shuffle_pivot(index: u32, key: u32, count: u32) -> u32 {
    if count <= 1u { return 0u; }
    let bits = (33u - countLeadingZeros(count - 1u)) >> 1u;
    let mask = (1u << bits) - 1u;
    var value = index;
    loop {
        var left = value >> bits;
        var right = value & mask;
        for (var round = 0u; round < 4u; round++) {
            let next = left ^ (hash32(right ^ key ^ ((round + 1u) * 0x85ebca6bu)) & mask);
            left = right;
            right = next;
        }
        value = (left << bits) | right;
        if value < count { break; }
    }
    return value;
}

fn shuffle_pivot_precomputed(
    index: u32,
    count: u32,
    bits: u32,
    mask: u32,
    rk0: u32,
    rk1: u32,
    rk2: u32,
    rk3: u32,
) -> u32 {
    var value = index;
    loop {
        var left = value >> bits;
        var right = value & mask;
        var next = left ^ (hash32(right ^ rk0) & mask);
        left = right;
        right = next;
        next = left ^ (hash32(right ^ rk1) & mask);
        left = right;
        right = next;
        next = left ^ (hash32(right ^ rk2) & mask);
        left = right;
        right = next;
        next = left ^ (hash32(right ^ rk3) & mask);
        left = right;
        right = next;
        value = (left << bits) | right;
        if value < count { break; }
    }
    return value;
}
