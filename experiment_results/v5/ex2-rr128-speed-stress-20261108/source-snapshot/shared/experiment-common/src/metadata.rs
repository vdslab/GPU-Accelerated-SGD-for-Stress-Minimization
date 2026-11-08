use crate::record::EnvironmentMetadata;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitMetadata {
    pub commit: String,
    pub dirty: bool,
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("SHA-256対象を読めません: {}", path.display()))?;
    Ok(hex(Sha256::digest(bytes)))
}

pub fn positions_sha256_f64(positions: &[[f64; 2]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"positions-f64-v1");
    hasher.update((positions.len() as u64).to_le_bytes());
    for position in positions {
        hasher.update(position[0].to_le_bytes());
        hasher.update(position[1].to_le_bytes());
    }
    hex(hasher.finalize())
}

pub fn positions_sha256_f32(positions: &[[f32; 2]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"positions-f32-v1");
    hasher.update((positions.len() as u64).to_le_bytes());
    for position in positions {
        hasher.update(position[0].to_le_bytes());
        hasher.update(position[1].to_le_bytes());
    }
    hex(hasher.finalize())
}

pub struct FingerprintBuilder(Sha256);

impl FingerprintBuilder {
    pub fn new(domain: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        Self(hasher)
    }

    pub fn usize(&mut self, value: usize) -> &mut Self {
        self.0.update((value as u64).to_le_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.0.update(value.to_le_bytes());
        self
    }

    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.0.update(value.to_le_bytes());
        self
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
        self
    }

    pub fn finish(self) -> String {
        hex(self.0.finalize())
    }
}

pub fn git_metadata(repo_hint: &Path) -> Result<GitMetadata> {
    let commit = git(repo_hint, &["rev-parse", "HEAD"])?;
    let dirty = !git(repo_hint, &["status", "--porcelain"])?.is_empty();
    Ok(GitMetadata { commit, dirty })
}

pub fn current_binary() -> String {
    std::env::current_exe()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "unknown".to_owned())
}

pub fn dataset_name(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

pub fn environment_metadata(
    gpu_name: Option<String>,
    gpu_backend: Option<String>,
) -> EnvironmentMetadata {
    EnvironmentMetadata {
        cpu_model: cpu_model(),
        gpu_name,
        gpu_backend,
    }
}

fn git(repo_hint: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_hint)
        .args(args)
        .output()
        .context("gitを起動できません")?;
    if !output.status.success() {
        anyhow::bail!(
            "git {} に失敗しました: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if output.status.success() && !value.is_empty() {
                return value;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(value) = contents.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|line| line.split_once(':'))
                    .map(|(_, value)| value.trim().to_owned())
            }) {
                return value;
            }
        }
    }
    std::env::consts::ARCH.to_owned()
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_hash_is_repeatable_and_order_sensitive() {
        let positions = [[0.0, 1.0], [2.0, 3.0]];
        assert_eq!(
            positions_sha256_f64(&positions),
            positions_sha256_f64(&positions)
        );
        assert_ne!(
            positions_sha256_f64(&positions),
            positions_sha256_f64(&[positions[1], positions[0]])
        );
    }
}
