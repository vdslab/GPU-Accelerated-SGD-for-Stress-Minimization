use std::process::Command;

#[test]
fn json_mode_writes_one_record_and_no_progress_to_stdout() {
    let directory = tempfile::tempdir().unwrap();
    let graph = directory.path().join("path3.mtx");
    std::fs::write(
        &graph,
        "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_baseline-sgd-non-gpu"))
        .args([
            "--run-id",
            "contract-sgd",
            "--input",
            graph.to_str().unwrap(),
            "--iterations",
            "2",
            "--epsilon",
            "0.1",
            "--seed",
            "7",
            "--output-format",
            "json",
            "--output-dir",
            directory.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(!stdout.contains("Iteration:"));
    let record: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["method"], "sgd");
    assert_eq!(record["run_id"], "contract-sgd");
    assert!(record["gpu_name"].is_null());
    assert!(record["pivots"].is_null());
}
