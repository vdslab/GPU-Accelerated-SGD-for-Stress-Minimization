pub mod cli;
pub mod metadata;
pub mod record;
pub mod seed;
pub mod stress;

pub use cli::{CommonExperimentArgs, OutputFormat};
pub use metadata::{
    current_binary, dataset_name, environment_metadata, git_metadata, positions_sha256_f32,
    positions_sha256_f64, sha256_file, FingerprintBuilder, GitMetadata,
};
pub use record::{
    Artifacts, EnvironmentMetadata, ExperimentRecord, Family, GraphMetrics, Method, MethodStats,
    Parameters, RecordIdentity, RunMode, RunStatus, TimingBreakdown, SCHEMA_VERSION,
};
pub use stress::{
    evaluate_f32, evaluate_f64, measure_auto_f32, measure_auto_f64, StressKind, StressMeasurement,
    StressMode, StressResult, AUTO_EXACT_MAX_N, DEFAULT_SAMPLES, DEFAULT_SEED,
};
