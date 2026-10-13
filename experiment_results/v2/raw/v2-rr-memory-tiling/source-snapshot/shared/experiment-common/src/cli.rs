use crate::record::RunMode;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Human,
    Json,
}

impl FromStr for OutputFormat {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "human" => Ok(Self::Human),
            "json" => Ok(Self::Json),
            _ => bail!("--output-format は human または json を指定してください"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommonExperimentArgs {
    pub run_id: String,
    pub output_dir: PathBuf,
    pub output_format: OutputFormat,
    pub run_mode: RunMode,
    pub verbose: bool,
}

impl CommonExperimentArgs {
    pub fn human(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            run_id: "manual".to_owned(),
            output_dir: output_dir.into(),
            output_format: OutputFormat::Human,
            run_mode: RunMode::Benchmark,
            verbose: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.run_id.trim().is_empty() {
            bail!("--run-id は空にできません");
        }
        if self
            .run_id
            .chars()
            .any(|character| !(character.is_ascii_alphanumeric() || "-_.".contains(character)))
        {
            bail!("--run-id には英数字、-、_、.だけを使用してください");
        }
        Ok(())
    }

    pub fn log(&self, message: impl AsRef<str>) {
        if self.verbose {
            eprintln!("{}", message.as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_is_strict() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("JSON".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn run_id_rejects_path_characters() {
        let mut args = CommonExperimentArgs::human("out");
        args.run_id = "../bad".to_owned();
        assert!(args.validate().is_err());
    }
}
