use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::frame::StreamSpec;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ServiceName(pub String);

impl From<&str> for ServiceName {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}
impl std::fmt::Display for ServiceName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 进程内与跨进程插件共用的声明。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    #[serde(default)]
    pub inject: Vec<ServiceName>,
    #[serde(default)]
    pub provide: Vec<ServiceName>,
    #[serde(default)]
    pub streams: Vec<StreamSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_schema: Option<Value>,
}

impl Manifest {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inject: vec![],
            provide: vec![],
            streams: vec![],
            config_schema: None,
        }
    }
    pub fn inject(mut self, names: &[&str]) -> Self {
        self.inject = names.iter().map(|n| ServiceName::from(*n)).collect();
        self
    }
    pub fn provide(mut self, names: &[&str]) -> Self {
        self.provide = names.iter().map(|n| ServiceName::from(*n)).collect();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_builder_and_defaults() {
        let m = Manifest::new("frame-counter").inject(&["frame-store", "world-log"]);
        assert_eq!(
            m.inject,
            vec![
                ServiceName::from("frame-store"),
                ServiceName::from("world-log")
            ]
        );
        let j = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
        let minimal: Manifest = serde_json::from_str(r#"{"name":"x"}"#).unwrap();
        assert!(
            minimal.inject.is_empty() && minimal.provide.is_empty() && minimal.streams.is_empty()
        );
    }
}
