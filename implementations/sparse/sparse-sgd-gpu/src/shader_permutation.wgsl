// Four-round balanced Feistel permutation on the smallest power-of-four domain.
// Cycle walking restricts that bijection to [0, count), with no omissions/repeats.
fn shuffle_pivot(index: u32, key: u32, count: u32) -> u32 {
    if count <= 1u { return 0u; }
    var remaining = count - 1u;
    var bits = 0u;
    loop {
        if remaining == 0u { break; }
        bits += 1u;
        remaining >>= 2u;
    }
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
