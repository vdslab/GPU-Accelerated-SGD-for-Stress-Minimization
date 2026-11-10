pub mod cli;
pub mod metadata;
pub mod paths;
pub mod record;
pub mod seed;
pub mod stress;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use cli::{CommonExperimentArgs, OutputFormat};
pub use metadata::{
    current_binary, dataset_name, environment_metadata, git_metadata, positions_sha256_f32,
    positions_sha256_f64, sha256_file, FingerprintBuilder, GitMetadata,
};
pub use paths::repository_path;
pub use record::{
    Artifacts, EnvironmentMetadata, ExperimentRecord, Family, GraphMetrics, Method, MethodStats,
    Parameters, RecordIdentity, RunMode, RunStatus, TimingBreakdown, SCHEMA_VERSION,
};
pub use stress::{
    evaluate_f32, evaluate_f64, measure_auto_f32, measure_auto_f64, StressKind, StressMeasurement,
    StressMode, StressResult, AUTO_EXACT_MAX_N, DEFAULT_SAMPLES, DEFAULT_SEED,
};

pub mod trace;

/// System-wide monotonic clock, comparable between the solver and renderer.
/// Unlike a wall clock this is unaffected by clock synchronization.
pub fn monotonic_time_ms() -> f64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
    assert_eq!(result, 0, "monotonic clock unavailable");
    value.tv_sec as f64 * 1000.0 + value.tv_nsec as f64 / 1_000_000.0
}
