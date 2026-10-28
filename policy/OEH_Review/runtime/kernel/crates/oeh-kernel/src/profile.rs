use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Bundle / Patch 里的一行：一个要激活的插件及其配置。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntryRow {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub config: Value,
    #[serde(default)]
    pub disabled: bool,
}

impl EntryRow {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            config: Value::Null,
            disabled: false,
        }
    }
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn with_config(mut self, config: Value) -> Self {
        self.config = config;
        self
    }
    /// Patch 定位与状态显示用的键：name 优先，否则 id。
    pub fn key(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PatchOp {
    Insert {
        insert: Vec<EntryRow>,
    },
    Replace {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        config: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disabled: Option<bool>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProfileFile {
    pub bundles: Vec<String>,
    #[serde(default)]
    pub patch: Vec<PatchOp>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("bundle `{0}` not found")]
    BundleNotFound(String),
    #[error("patch target `{0}` not found")]
    PatchTargetNotFound(String),
    #[error("patch target `{0}` matches more than one row; give the row a unique `name`")]
    PatchTargetAmbiguous(String),
    #[error("yaml error: {0}")]
    Yaml(String),
    #[error("io error: {0}")]
    Io(String),
}

/// bundles 按序拼接成行列表，再按序叠加每一层 patch。
/// Replace 匹配两阶段：(1) key()（名字或id）等于 id，或 (2) 原始 id 字段等于 id（仅在第1阶段无匹配时）。
/// 无论哪阶段产出匹配：恰好一个→应用；多于一个→PatchTargetAmbiguous；两阶段都无匹配→PatchTargetNotFound。
pub fn compose(
    bundles: &[(String, Vec<EntryRow>)],
    layers: &[Vec<PatchOp>],
) -> Result<Vec<EntryRow>, ProfileError> {
    let mut rows: Vec<EntryRow> = bundles
        .iter()
        .flat_map(|(_, r)| r.iter().cloned())
        .collect();
    for layer in layers {
        for op in layer {
            match op {
                PatchOp::Insert { insert } => rows.extend(insert.iter().cloned()),
                PatchOp::Replace {
                    id,
                    config,
                    disabled,
                } => {
                    // First try to match by key() (name or id)
                    let mut matches: Vec<usize> = rows
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.key() == id)
                        .map(|(i, _)| i)
                        .collect();
                    // If no matches by key(), try to match by id
                    if matches.is_empty() {
                        matches = rows
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| r.id == *id)
                            .map(|(i, _)| i)
                            .collect();
                    }
                    let idx = match matches.as_slice() {
                        [] => return Err(ProfileError::PatchTargetNotFound(id.clone())),
                        [i] => *i,
                        _ => return Err(ProfileError::PatchTargetAmbiguous(id.clone())),
                    };
                    if let Some(c) = config {
                        rows[idx].config = c.clone();
                    }
                    if let Some(d) = disabled {
                        rows[idx].disabled = *d;
                    }
                }
            }
        }
    }
    Ok(rows)
}

fn read_yaml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ProfileError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ProfileError::Io(format!("{}: {e}", path.display())))?;
    serde_yaml_ng::from_str(&text)
        .map_err(|e| ProfileError::Yaml(format!("{}: {e}", path.display())))
}

/// 叠加顺序：bundles → profile.patch → <oeh_home>/patch.yaml（存在才用）→ extra_patch_files。
pub fn load_profile_with_home(
    profile_path: &Path,
    bundles_dir: &Path,
    oeh_home: Option<&Path>,
    extra_patch_files: &[PathBuf],
) -> Result<Vec<EntryRow>, ProfileError> {
    let profile: ProfileFile = read_yaml(profile_path)?;
    let mut bundles = Vec::new();
    for b in &profile.bundles {
        let p = bundles_dir.join(format!("{b}.yaml"));
        if !p.exists() {
            return Err(ProfileError::BundleNotFound(p.display().to_string()));
        }
        bundles.push((b.clone(), read_yaml::<Vec<EntryRow>>(&p)?));
    }
    let mut layers = vec![profile.patch.clone()];
    if let Some(home) = oeh_home {
        let p = home.join("patch.yaml");
        if p.exists() {
            layers.push(read_yaml::<Vec<PatchOp>>(&p)?);
        }
    }
    for f in extra_patch_files {
        layers.push(read_yaml::<Vec<PatchOp>>(f)?);
    }
    compose(&bundles, &layers)
}

/// 读 $OEH_HOME 后委托给 load_profile_with_home。
pub fn load_profile(
    profile_path: &Path,
    bundles_dir: &Path,
    extra_patch_files: &[PathBuf],
) -> Result<Vec<EntryRow>, ProfileError> {
    let home = std::env::var_os("OEH_HOME").map(PathBuf::from);
    load_profile_with_home(
        profile_path,
        bundles_dir,
        home.as_deref(),
        extra_patch_files,
    )
}
