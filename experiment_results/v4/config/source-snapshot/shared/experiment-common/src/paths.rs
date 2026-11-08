//! Default paths are anchored to this checkout, independently of the CLI working directory.
use std::path::{Path, PathBuf};

pub fn repository_path(relative: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("experiment-common must live in shared/experiment-common")
        .join(relative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_paths_are_anchored_to_the_repository() {
        let root = repository_path("");
        assert!(root
            .join("implementations/full/baseline-sgd-non-gpu/Cargo.toml")
            .is_file());
        assert!(root.join("experiments/experiment_plan.py").is_file());
        assert_eq!(repository_path("output"), root.join("output"));
    }
}
