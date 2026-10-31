use std::path::PathBuf;
use std::sync::Arc;

use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::clock::ClockService;
use crate::factories::PluginFactory;
use crate::plugin::{Ctx, Plugin, PluginError};
use crate::world_log::{JsonlSegmentSink, WorldLog};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    dir: Option<PathBuf>,
    #[serde(default = "default_segment_mb")]
    segment_mb: u64,
    #[serde(default = "default_retain")]
    retain: usize,
}
fn default_segment_mb() -> u64 {
    64
}
fn default_retain() -> usize {
    4096
}
impl Default for Config {
    fn default() -> Self {
        Self {
            dir: None,
            segment_mb: default_segment_mb(),
            retain: default_retain(),
        }
    }
}

/// 一次 boot 一个 run 子目录：`<dir>/run-<boot_unix_ms>/`（unix 毫秒，补零到 13 位）。
/// 段文件里的 seq 从 1 起；若两次 boot 共用一个目录，追加进同一段会让回放看到 1,2,1,2 →
/// Gap。分 run 目录后重启只是新的 run，默认回放取最新 run。
pub struct WorldLogPlugin;

/// `run-<boot_unix_ms>`：会话日志在没有 world-log run 目录时也用它，保持同一套命名。
pub(crate) fn boot_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

impl Plugin for WorldLogPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("world-log")
            .inject(&["clock"])
            .provide(&["world-log"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let clock = ctx.get::<ClockService>("clock")?;
        let mut log = WorldLog::new(clock.0.clone()).with_retain(cfg.retain);
        if let Some(dir) = cfg.dir {
            let run_dir = dir.join(format!("run-{:013}", boot_unix_ms()));
            crate::fsutil::create_dir_private(&dir)
                .map_err(|e| PluginError::Other(format!("world-log dir {}: {e}", dir.display())))?;
            // create（非 create_all）：run 目录必须是新的，撞名是硬错误而不是静默追加。
            // 权限随后由 `JsonlSegmentSink::new` 落到 0700（见 [`crate::fsutil`]）。
            std::fs::create_dir(&run_dir).map_err(|e| {
                PluginError::Other(format!("world-log run dir {}: {e}", run_dir.display()))
            })?;
            let sink = JsonlSegmentSink::new(&run_dir, cfg.segment_mb.saturating_mul(1024 * 1024))
                .map_err(|e| {
                    PluginError::Other(format!("world-log run dir {}: {e}", run_dir.display()))
                })?;
            log = log.with_sink(Box::new(sink)).with_run_dir(run_dir);
        }
        let log = Arc::new(log);
        let l2 = log.clone();
        ctx.on_dispose(Box::new(move || {
            let _ = l2.flush();
        }));
        ctx.provide("world-log", log)
    }
}

inventory::submit! { PluginFactory { id: "world-log", build: || Box::new(WorldLogPlugin) } }

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn manual_default_matches_spec() {
        let c = Config::default();
        assert_eq!((c.segment_mb, c.retain), (64, 4096));
        assert!(c.dir.is_none());
        let empty: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!((empty.segment_mb, empty.retain), (64, 4096));
    }
}
