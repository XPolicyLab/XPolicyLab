use std::sync::Arc;

use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::clock::ClockService;
use crate::factories::PluginFactory;
use crate::frame_store::FrameStore;
use crate::plugin::{Ctx, Plugin, PluginError};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "default_window")]
    window_s: f64,
    /// 预留：spec §4.2 的帧段文件持久化。本计划不实现，置 true 只 warn。
    #[serde(default)]
    record: bool,
}
fn default_window() -> f64 {
    2.0
}
impl Default for Config {
    fn default() -> Self {
        Self {
            window_s: default_window(),
            record: false,
        }
    }
}

pub struct FrameStorePlugin;

impl Plugin for FrameStorePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("frame-store")
            .inject(&["clock"])
            .provide(&["frame-store"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        if cfg.record {
            tracing::warn!("frame-store.record is reserved for a later plan; ignored");
        }
        let clock = ctx.get::<ClockService>("clock")?;
        ctx.provide(
            "frame-store",
            Arc::new(FrameStore::new(clock.0.clone(), cfg.window_s)),
        )
    }
}

inventory::submit! { PluginFactory { id: "frame-store", build: || Box::new(FrameStorePlugin) } }

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn manual_default_matches_spec() {
        let d = Config::default();
        assert_eq!((d.window_s, d.record), (2.0, false));
        let empty: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!((empty.window_s, empty.record), (2.0, false));
        let rec: Config = serde_json::from_value(serde_json::json!({"record": true})).unwrap();
        assert!(rec.record);
        // deny_unknown_fields 仍然生效
        assert!(serde_json::from_value::<Config>(serde_json::json!({"nope": 1})).is_err());
    }
}
