// CPU preprocessing is intentionally shared with the verified reference implementation.
// This keeps graph parsing, MaxMinRandomSP pivot selection, regions and weights identical.
#[path = "../../baseline-sparse-sgd-non-gpu/src/graph.rs"]
pub mod graph;

pub mod gpu;
pub mod schedule;

#[cfg(test)]
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
pub mod cpu_reference;
