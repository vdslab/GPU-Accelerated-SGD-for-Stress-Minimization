use crate::metadata::GitMetadata;
use crate::stress::StressMeasurement;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Full,
    Sparse,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Sgd,
    AtomicSgd,
    RrSgd,
    SparseSgd,
    AtomicSparseSgd,
    RrSparseSgd,
}

impl Method {
    pub fn family(self) -> Family {
        match self {
            Self::Sgd | Self::AtomicSgd | Self::RrSgd => Family::Full,
            Self::SparseSgd | Self::AtomicSparseSgd | Self::RrSparseSgd => Family::Sparse,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sgd => "sgd",
            Self::AtomicSgd => "atomic_sgd",
            Self::RrSgd => "rr_sgd",
            Self::SparseSgd => "sparse_sgd",
            Self::AtomicSparseSgd => "atomic_sparse_sgd",
            Self::RrSparseSgd => "rr_sparse_sgd",
        }
    }
}

impl FromStr for Method {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "sgd" => Ok(Self::Sgd),
            "atomic_sgd" => Ok(Self::AtomicSgd),
            "rr_sgd" => Ok(Self::RrSgd),
            "sparse_sgd" => Ok(Self::SparseSgd),
            "atomic_sparse_sgd" => Ok(Self::AtomicSparseSgd),
            "rr_sparse_sgd" => Ok(Self::RrSparseSgd),
            _ => bail!("未知のmethod IDです: {value}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    #[default]
    Benchmark,
    Diagnostic,
}

impl FromStr for RunMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "benchmark" => Ok(Self::Benchmark),
            "diagnostic" => Ok(Self::Diagnostic),
            _ => bail!("--run-mode は benchmark または diagnostic を指定してください"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Success,
    Failure,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecordIdentity {
    pub run_id: String,
    pub run_mode: RunMode,
    pub method: Method,
    pub binary: String,
    pub dataset: String,
    pub input_path: String,
    pub input_sha256: String,
    pub seed: u64,
    pub initial_positions_sha256: String,
    pub preprocess_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphMetrics {
    pub nodes: usize,
    pub edges: usize,
    pub constraints: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Parameters {
    pub pivots: Option<usize>,
    pub iterations: usize,
    pub epsilon: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EnvironmentMetadata {
    pub cpu_model: String,
    pub gpu_name: Option<String>,
    pub gpu_backend: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct TimingBreakdown {
    pub input_time_ms: Option<f64>,
    pub common_preprocess_time_ms: Option<f64>,
    pub method_setup_time_ms: Option<f64>,
    pub runtime_init_time_ms: Option<f64>,
    pub upload_time_ms: Option<f64>,
    pub iteration_time_ms: Option<f64>,
    pub gpu_device_time_ms: Option<f64>,
    pub readback_time_ms: Option<f64>,
    pub postprocess_time_ms: Option<f64>,
    pub algorithm_time_cold_ms: Option<f64>,
    pub algorithm_time_warm_ms: Option<f64>,
    pub cli_total_time_cold_ms: Option<f64>,
}

impl TimingBreakdown {
    pub fn derive_totals(&mut self) -> Result<()> {
        for (name, value) in self.stage_values() {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "{name} は有限な非負値である必要があります: {value}"
                );
            }
        }
        let common = self
            .common_preprocess_time_ms
            .ok_or_else(|| anyhow::anyhow!("common_preprocess_time_ms が必要です"))?;
        let iteration = self
            .iteration_time_ms
            .ok_or_else(|| anyhow::anyhow!("iteration_time_ms が必要です"))?;
        let postprocess = self
            .postprocess_time_ms
            .ok_or_else(|| anyhow::anyhow!("postprocess_time_ms が必要です"))?;
        let input = self
            .input_time_ms
            .ok_or_else(|| anyhow::anyhow!("input_time_ms が必要です"))?;
        let algorithm = common
            + self.method_setup_time_ms.unwrap_or(0.0)
            + self.runtime_init_time_ms.unwrap_or(0.0)
            + self.upload_time_ms.unwrap_or(0.0)
            + iteration
            + self.readback_time_ms.unwrap_or(0.0)
            + postprocess;
        self.algorithm_time_cold_ms = Some(algorithm);
        self.algorithm_time_warm_ms = Some(algorithm - self.runtime_init_time_ms.unwrap_or(0.0));
        self.cli_total_time_cold_ms = Some(input + algorithm);
        self.validate()
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in self.all_values() {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "{name} は有限な非負値である必要があります: {value}"
                );
            }
        }
        let expected_algorithm = self
            .common_preprocess_time_ms
            .ok_or_else(|| anyhow::anyhow!("common_preprocess_time_ms が必要です"))?
            + self.method_setup_time_ms.unwrap_or(0.0)
            + self.runtime_init_time_ms.unwrap_or(0.0)
            + self.upload_time_ms.unwrap_or(0.0)
            + self
                .iteration_time_ms
                .ok_or_else(|| anyhow::anyhow!("iteration_time_ms が必要です"))?
            + self.readback_time_ms.unwrap_or(0.0)
            + self
                .postprocess_time_ms
                .ok_or_else(|| anyhow::anyhow!("postprocess_time_ms が必要です"))?;
        let algorithm = self
            .algorithm_time_cold_ms
            .ok_or_else(|| anyhow::anyhow!("algorithm_time_cold_ms が必要です"))?;
        ensure_close("algorithm_time_cold_ms", algorithm, expected_algorithm)?;
        let expected_cli = self
            .input_time_ms
            .ok_or_else(|| anyhow::anyhow!("input_time_ms が必要です"))?
            + expected_algorithm;
        ensure_close(
            "cli_total_time_cold_ms",
            self.cli_total_time_cold_ms
                .ok_or_else(|| anyhow::anyhow!("cli_total_time_cold_ms が必要です"))?,
            expected_cli,
        )?;
        let expected_warm = expected_algorithm - self.runtime_init_time_ms.unwrap_or(0.0);
        ensure_close(
            "algorithm_time_warm_ms",
            self.algorithm_time_warm_ms
                .ok_or_else(|| anyhow::anyhow!("algorithm_time_warm_ms が必要です"))?,
            expected_warm,
        )
    }

    fn stage_values(&self) -> [(&'static str, Option<f64>); 9] {
        [
            ("input_time_ms", self.input_time_ms),
            ("common_preprocess_time_ms", self.common_preprocess_time_ms),
            ("method_setup_time_ms", self.method_setup_time_ms),
            ("runtime_init_time_ms", self.runtime_init_time_ms),
            ("upload_time_ms", self.upload_time_ms),
            ("iteration_time_ms", self.iteration_time_ms),
            ("gpu_device_time_ms", self.gpu_device_time_ms),
            ("readback_time_ms", self.readback_time_ms),
            ("postprocess_time_ms", self.postprocess_time_ms),
        ]
    }

    fn all_values(&self) -> [(&'static str, Option<f64>); 12] {
        [
            ("input_time_ms", self.input_time_ms),
            ("common_preprocess_time_ms", self.common_preprocess_time_ms),
            ("method_setup_time_ms", self.method_setup_time_ms),
            ("runtime_init_time_ms", self.runtime_init_time_ms),
            ("upload_time_ms", self.upload_time_ms),
            ("iteration_time_ms", self.iteration_time_ms),
            ("gpu_device_time_ms", self.gpu_device_time_ms),
            ("readback_time_ms", self.readback_time_ms),
            ("postprocess_time_ms", self.postprocess_time_ms),
            ("algorithm_time_cold_ms", self.algorithm_time_cold_ms),
            ("algorithm_time_warm_ms", self.algorithm_time_warm_ms),
            ("cli_total_time_cold_ms", self.cli_total_time_cold_ms),
        ]
    }
}

fn ensure_close(name: &str, actual: f64, expected: f64) -> Result<()> {
    let tolerance = (expected.abs() * 1e-9).max(1e-6);
    ensure!(
        (actual - expected).abs() <= tolerance,
        "{name} が内訳と一致しません: actual={actual}, expected={expected}"
    );
    Ok(())
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct MethodStats {
    pub attempted_updates: Option<u64>,
    pub completed_updates: Option<u64>,
    pub retry_failures: Option<u64>,
    pub rounds: Option<usize>,
    pub dispatches: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Artifacts {
    pub final_positions_path: Option<String>,
    pub vertex_map_path: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ExperimentRecord {
    pub schema_version: u32,
    #[serde(flatten)]
    pub identity: RecordIdentity,
    pub status: RunStatus,
    pub family: Family,
    pub git_commit: String,
    pub git_dirty: bool,
    #[serde(flatten)]
    pub graph: GraphMetrics,
    #[serde(flatten)]
    pub parameters: Parameters,
    #[serde(flatten)]
    pub environment: EnvironmentMetadata,
    #[serde(flatten)]
    pub timings: TimingBreakdown,
    #[serde(flatten)]
    pub stress: StressMeasurement,
    #[serde(flatten)]
    pub stats: MethodStats,
    #[serde(flatten)]
    pub artifacts: Artifacts,
    pub error_stage: Option<String>,
    pub error_message: Option<String>,
    pub exit_code: Option<i32>,
    pub stderr_log_path: Option<String>,
}

impl ExperimentRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        identity: RecordIdentity,
        git: GitMetadata,
        graph: GraphMetrics,
        parameters: Parameters,
        environment: EnvironmentMetadata,
        mut timings: TimingBreakdown,
        stress: StressMeasurement,
        stats: MethodStats,
        artifacts: Artifacts,
    ) -> Result<Self> {
        timings.derive_totals()?;
        let record = Self {
            schema_version: SCHEMA_VERSION,
            family: identity.method.family(),
            identity,
            status: RunStatus::Success,
            git_commit: git.commit,
            git_dirty: git.dirty,
            graph,
            parameters,
            environment,
            timings,
            stress,
            stats,
            artifacts,
            error_stage: None,
            error_message: None,
            exit_code: None,
            stderr_log_path: None,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION,
            "未対応schema versionです: {}",
            self.schema_version
        );
        ensure!(
            self.family == self.identity.method.family(),
            "methodとfamilyが一致しません"
        );
        ensure!(self.parameters.iterations > 0, "iterationsは1以上です");
        ensure!(
            self.parameters.epsilon.is_finite() && self.parameters.epsilon > 0.0,
            "epsilonは正の有限値です"
        );
        ensure!(
            !self.identity.run_id.is_empty()
                && !self.identity.input_sha256.is_empty()
                && !self.identity.initial_positions_sha256.is_empty(),
            "追跡用IDとhashは空にできません"
        );
        if self.family == Family::Full {
            ensure!(
                self.parameters.pivots.is_none(),
                "Full手法のpivotsはnullです"
            );
        } else {
            ensure!(
                self.parameters.pivots.is_some_and(|value| value > 0),
                "Sparse手法には正のpivotsが必要です"
            );
        }
        self.timings.validate()?;
        self.stress.validate()?;
        if self.identity.run_mode == RunMode::Benchmark {
            // 値の取得自体は許可するが、追加同期を行わないことは各実装の責務。
            ensure!(
                self.status == RunStatus::Success,
                "Rust method recordは成功runだけを出力します"
            );
        }
        Ok(())
    }

    pub fn to_json_line(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stress::{StressKind, StressMeasurement};

    fn timings() -> TimingBreakdown {
        let mut value = TimingBreakdown {
            input_time_ms: Some(1.0),
            common_preprocess_time_ms: Some(2.0),
            method_setup_time_ms: None,
            runtime_init_time_ms: None,
            upload_time_ms: None,
            iteration_time_ms: Some(3.0),
            gpu_device_time_ms: None,
            readback_time_ms: None,
            postprocess_time_ms: Some(4.0),
            ..TimingBreakdown::default()
        };
        value.derive_totals().unwrap();
        value
    }

    #[test]
    fn totals_are_derived_from_stages() {
        let value = timings();
        assert_eq!(value.algorithm_time_cold_ms, Some(9.0));
        assert_eq!(value.algorithm_time_warm_ms, Some(9.0));
        assert_eq!(value.cli_total_time_cold_ms, Some(10.0));
    }

    #[test]
    fn incompatible_method_family_is_rejected() {
        let record = ExperimentRecord {
            schema_version: SCHEMA_VERSION,
            identity: RecordIdentity {
                run_id: "r".into(),
                run_mode: RunMode::Benchmark,
                method: Method::Sgd,
                binary: "sgd".into(),
                dataset: "g".into(),
                input_path: "g.mtx".into(),
                input_sha256: "a".into(),
                seed: 0,
                initial_positions_sha256: "b".into(),
                preprocess_sha256: None,
            },
            status: RunStatus::Success,
            family: Family::Sparse,
            git_commit: "c".into(),
            git_dirty: false,
            graph: GraphMetrics {
                nodes: 2,
                edges: 1,
                constraints: Some(1),
            },
            parameters: Parameters {
                pivots: Some(1),
                iterations: 1,
                epsilon: 0.1,
            },
            environment: EnvironmentMetadata {
                cpu_model: "cpu".into(),
                gpu_name: None,
                gpu_backend: None,
            },
            timings: timings(),
            stress: StressMeasurement {
                stress_kind: Some(StressKind::Exact),
                stress_value: Some(0.0),
                stress_eval_time_ms: Some(0.0),
                stress_samples: None,
                stress_seed: None,
            },
            stats: MethodStats::default(),
            artifacts: Artifacts {
                final_positions_path: Some("result.txt".into()),
                vertex_map_path: None,
            },
            error_stage: None,
            error_message: None,
            exit_code: None,
            stderr_log_path: None,
        };
        assert!(record.validate().is_err());
    }

    #[test]
    fn atomic_sparse_method_is_schema_compatible() {
        assert_eq!(
            "atomic_sparse_sgd".parse::<Method>().unwrap().family(),
            Family::Sparse
        );
        let stats = MethodStats {
            attempted_updates: None,
            completed_updates: None,
            retry_failures: None,
            rounds: None,
            dispatches: None,
        };
        assert!(serde_json::to_value(stats).unwrap()["attempted_updates"].is_null());
    }
}
