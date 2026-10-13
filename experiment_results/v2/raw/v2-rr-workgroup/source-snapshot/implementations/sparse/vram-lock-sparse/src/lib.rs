// Share the CPU Sparse reference preprocessing, including pivots and directional weights.
#[path = "../../baseline-sparse-sgd-non-gpu/src/graph.rs"]
pub mod graph;
pub mod metal;

#[cfg(test)]
#[path = "../../baseline-sparse-sgd-non-gpu/src/algorithm.rs"]
pub mod cpu_reference;
