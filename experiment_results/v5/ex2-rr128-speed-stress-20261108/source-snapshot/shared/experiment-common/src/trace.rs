//! Optional convergence telemetry. No RNG draws or stress evaluation are performed.
//!
//! `SGD_TRACE_DIR` enables this hook. `SGD_TRACE_CHECKPOINTS` is a comma-separated
//! set of iteration numbers (0 means the initial GPU/CPU coordinates).
//! Snapshots are interleaved little-endian f64 x/y coordinates; GPU values are
//! converted from f32 without changing the algorithm's working storage.
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

pub struct IterationTrace {
    directory: PathBuf,
    checkpoints: BTreeSet<usize>,
    output: BufWriter<File>,
    cumulative_algorithm: Duration,
    overhead: Duration,
    completed_iteration: usize,
}

impl IterationTrace {
    pub fn from_env() -> Option<Self> {
        let directory = PathBuf::from(std::env::var_os("SGD_TRACE_DIR")?);
        fs::create_dir_all(&directory).expect("create SGD_TRACE_DIR");
        let checkpoints = std::env::var("SGD_TRACE_CHECKPOINTS")
            .unwrap_or_else(|_| "0,1,2,3,5,8,10,15".into())
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim()
                    .parse()
                    .expect("SGD_TRACE_CHECKPOINTS must contain iteration integers")
            })
            .collect();
        let output = BufWriter::new(
            File::create(directory.join("trace.jsonl")).expect("create convergence trace"),
        );
        Some(Self {
            directory,
            checkpoints,
            output,
            cumulative_algorithm: Duration::ZERO,
            overhead: Duration::ZERO,
            completed_iteration: 0,
        })
    }

    pub fn overhead(&self) -> Duration {
        self.overhead
    }

    /// Call after the iteration's completion wait. The closure is evaluated only
    /// at requested checkpoints, after the algorithm duration has been measured.
    pub fn record<F>(&mut self, iteration: usize, algorithm: Duration, coordinates: F)
    where
        F: FnOnce() -> Vec<[f64; 2]>,
    {
        let overhead_start = Instant::now();
        self.completed_iteration = iteration;
        self.cumulative_algorithm += algorithm;
        let mut snapshot = None;
        let mut nodes = None;
        if self.checkpoints.contains(&iteration) {
            let positions = coordinates();
            nodes = Some(positions.len());
            let filename = format!("positions-{iteration:04}.f64le");
            let mut file = BufWriter::new(
                File::create(self.directory.join(&filename)).expect("create convergence snapshot"),
            );
            for position in &positions {
                for coordinate in position {
                    file.write_all(&coordinate.to_le_bytes())
                        .expect("write convergence snapshot");
                }
            }
            file.flush().expect("flush convergence snapshot");
            snapshot = Some(filename);
        }
        let record = json!({
            "iteration": iteration,
            "algorithm_seconds": algorithm.as_secs_f64(),
            "cumulative_algorithm_seconds": self.cumulative_algorithm.as_secs_f64(),
            "snapshot": snapshot, "node_count": nodes,
            "coordinate_format": "interleaved_xy_f64_le",
            "excluded_from_algorithm_time": "checkpoint coordinate readback/copy, snapshot serialization/file IO, trace JSONL IO; stress evaluated offline",
            "previous_trace_overhead_seconds": self.overhead.as_secs_f64(),
        });
        serde_json::to_writer(&mut self.output, &record).expect("write convergence trace");
        self.output.write_all(b"\n").expect("write trace newline");
        self.output.flush().expect("flush convergence trace");
        self.overhead += overhead_start.elapsed();
    }
}

impl Drop for IterationTrace {
    fn drop(&mut self) {
        // Written outside the algorithm timer. Avoid panicking during unwinding.
        let _ = fs::write(
            self.directory.join("overhead.json"),
            json!({
                "trace_overhead_seconds": self.overhead.as_secs_f64(),
                "cumulative_algorithm_seconds": self.cumulative_algorithm.as_secs_f64(),
                "completed_iteration": self.completed_iteration,
                "summary_write_outside_iteration_timer": true,
            })
            .to_string(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_bytes_and_cumulative_timing_are_independent_of_io() {
        let directory = std::env::temp_dir().join(format!("sgd-trace-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let mut trace = IterationTrace {
            output: BufWriter::new(File::create(directory.join("trace.jsonl")).unwrap()),
            directory: directory.clone(),
            checkpoints: [0, 2].into_iter().collect(),
            cumulative_algorithm: Duration::ZERO,
            overhead: Duration::ZERO,
            completed_iteration: 0,
        };
        trace.record(0, Duration::ZERO, || vec![[1.0, -2.0]]);
        trace.record(1, Duration::from_millis(7), || {
            panic!("unrequested checkpoint must not read coordinates")
        });
        trace.record(2, Duration::from_millis(11), || vec![[3.0, 4.0]]);
        assert_eq!(trace.cumulative_algorithm, Duration::from_millis(18));
        let bytes = fs::read(directory.join("positions-0002.f64le")).unwrap();
        assert_eq!(
            bytes,
            [3.0_f64.to_le_bytes(), 4.0_f64.to_le_bytes()].concat()
        );
        drop(trace);
        let records: Vec<serde_json::Value> = fs::read_to_string(directory.join("trace.jsonl"))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert!(records[1]["snapshot"].is_null());
        assert_eq!(records[2]["cumulative_algorithm_seconds"], 0.018);
        assert_eq!(records[2]["node_count"], 1);
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("overhead.json")).unwrap()).unwrap();
        assert_eq!(summary["completed_iteration"], 2);
        assert!(summary["trace_overhead_seconds"].as_f64().unwrap() > 0.0);
        fs::remove_dir_all(directory).unwrap();
    }
}
