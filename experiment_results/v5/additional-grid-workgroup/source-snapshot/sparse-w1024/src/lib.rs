// Graph parsing and the independent CPU reference are shared.
// preprocess streams equivalent constraints into the compact RR representation;
// tests compare its values, ordering and RNG state against this reference.
#[path = "graph_reference.rs"]
pub mod graph;

pub mod gpu;
pub mod schedule;

#[cfg(test)]
#[path = "cpu_reference.rs"]
pub mod cpu_reference;

pub mod preprocess;
