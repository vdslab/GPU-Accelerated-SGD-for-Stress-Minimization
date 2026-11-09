// Share the CPU Sparse reference preprocessing, including pivots and directional weights.
pub mod gpu;
#[path = "../../baseline-sparse-sgd-non-gpu/src/graph.rs"]
pub mod graph;

#[cfg(test)]
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
pub mod cpu_reference;
