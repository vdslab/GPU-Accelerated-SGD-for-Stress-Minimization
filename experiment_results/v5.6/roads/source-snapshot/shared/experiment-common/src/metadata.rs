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
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                positions.as_ptr() as *const u8,
                std::mem::size_of_val(positions),
            )
        };
        hasher.update(bytes);
    }
    #[cfg(not(target_endian = "little"))]
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
    #[cfg(target_endian = "little")]
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                positions.as_ptr() as *const u8,
                std::mem::size_of_val(positions),
            )
        };
        hasher.update(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for position in positions {
        hasher.update(position[0].to_le_bytes());
        hasher.update(position[1].to_le_bytes());
    }
    hex(hasher.finalize())
}

pub struct FingerprintBuilder {
    hasher: Sha256,
    buf: [u8; 4096],
    len: usize,
}

impl FingerprintBuilder {
    pub fn new(domain: &str) -> Self {
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 4096];
        let domain_bytes = domain.as_bytes();
        let len = if domain_bytes.len() <= buf.len() {
            buf[..domain_bytes.len()].copy_from_slice(domain_bytes);
            domain_bytes.len()
        } else {
            hasher.update(domain_bytes);
            0
        };
        Self { hasher, buf, len }
    }

    #[cold]
    #[inline(never)]
    fn flush_blocks(&mut self) {
        let aligned = self.len & !63;
        if aligned > 0 {
            self.hasher.update(&self.buf[..aligned]);
            let rem = self.len - aligned;
            if rem > 0 {
                self.buf.copy_within(aligned..self.len, 0);
            }
            self.len = rem;
        }
    }

    fn flush_all(&mut self) {
        if self.len > 0 {
            self.hasher.update(&self.buf[..self.len]);
            self.len = 0;
        }
    }

    #[inline(always)]
    fn push_bytes_8(&mut self, bytes: [u8; 8]) {
        if self.len + 8 > self.buf.len() {
            self.flush_blocks();
        }
        self.buf[self.len..self.len + 8].copy_from_slice(&bytes);
        self.len += 8;
    }

    #[inline(always)]
    pub fn usize(&mut self, value: usize) -> &mut Self {
        self.push_bytes_8((value as u64).to_le_bytes());
        self
    }

    #[inline(always)]
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.push_bytes_8(value.to_le_bytes());
        self
    }

    #[inline(always)]
    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.push_bytes_8(value.to_le_bytes());
        self
    }

    #[inline(always)]
    pub fn edge_info(
        &mut self,
        u: usize,
        v: usize,
        dij: f64,
        weight_u: f64,
        weight_v: f64,
    ) -> &mut Self {
        if self.len + 40 > self.buf.len() {
            self.flush_blocks();
        }
        let dst = &mut self.buf[self.len..self.len + 40];
        dst[0..8].copy_from_slice(&(u as u64).to_le_bytes());
        dst[8..16].copy_from_slice(&(v as u64).to_le_bytes());
        dst[16..24].copy_from_slice(&dij.to_le_bytes());
        dst[24..32].copy_from_slice(&weight_u.to_le_bytes());
        dst[32..40].copy_from_slice(&weight_v.to_le_bytes());
        self.len += 40;
        self
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.push_bytes_8((value.len() as u64).to_le_bytes());
        self.flush_all();
        self.hasher.update(value);
        self
    }

    pub fn finish(mut self) -> String {
        self.flush_all();
        hex(self.hasher.finalize())
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

    #[test]
    fn fingerprint_builder_matches_unbuffered_sha256_across_buffer_flushes() {
        let mut builder = FingerprintBuilder::new("test-tag");
        let mut raw = Sha256::new();
        raw.update(b"test-tag");
        for i in 0_u64..1500 {
            let u = (i * 17) as usize;
            let f = (i as f64) * 0.125;
            builder.usize(u).u64(i).f64(f);
            raw.update((u as u64).to_le_bytes());
            raw.update(i.to_le_bytes());
            raw.update(f.to_le_bytes());
        }
        assert_eq!(builder.finish(), hex(raw.finalize()));
    }
}
