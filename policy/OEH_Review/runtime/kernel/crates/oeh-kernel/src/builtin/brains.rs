use std::path::PathBuf;
use std::sync::Arc;

use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::brains::{Brains, BrainsConfig};
use crate::clock::ClockService;
use crate::factories::PluginFactory;
use crate::observation::ObservationBudget;
use crate::plugin::{Ctx, Plugin, PluginError};
use crate::state::StateRegistry;
use crate::world_log::WorldLog;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "default_session_dir")]
    session_dir: PathBuf,
    #[serde(default)]
    observation: ObservationBudget,
    /// Running 的 turn 超过这么久没 `done` 就强制结束并记 `brain/error`。0 = 关闭。
    #[serde(default = "default_turn_timeout_ms")]
    turn_timeout_ms: u64,
}
fn default_session_dir() -> PathBuf {
    PathBuf::from("./logs/sessions")
}
fn default_turn_timeout_ms() -> u64 {
    30_000
}
impl Default for Config {
    fn default() -> Self {
        Self {
            session_dir: default_session_dir(),
            observation: ObservationBudget::default(),
            turn_timeout_ms: default_turn_timeout_ms(),
        }
    }
}

/// 提供 `brains` 服务：脑 Provider 用它 attach，世界日志的事件经它变成 turn。
pub struct BrainsPlugin;

impl Plugin for BrainsPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("brains")
            .inject(&["clock", "world-log", "state"])
            .provide(&["brains"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let clock = ctx.get::<ClockService>("clock")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        // 会话日志按 run 分目录，和世界日志同一个 run：turn 号每次 attach 从 1 起，
        // 两次运行写进同一个文件会让它读起来像 1,2,1,2。world-log 没配持久化目录时
        // （run_dir 为 None）自己造一个同名的 run 子目录。
        let session_dir = match log.run_dir() {
            Some(d) => d.join("sessions"),
            None => cfg
                .session_dir
                .join(format!("run-{:013}", super::world_log::boot_unix_ms())),
        };
        let brains = Brains::new(
            clock.0.clone(),
            log,
            state,
            session_dir,
            BrainsConfig {
                obs_budget: cfg.observation,
                turn_timeout_ms: cfg.turn_timeout_ms,
            },
        )
        .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.provide("brains", Arc::new(brains))
    }
}

inventory::submit! { PluginFactory { id: "brains", build: || Box::new(BrainsPlugin) } }

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn null_and_empty_config_share_the_defaults() {
        let manual = Config::default();
        let empty: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(manual.session_dir, empty.session_dir);
        assert_eq!(manual.observation, empty.observation);
        assert_eq!(manual.turn_timeout_ms, empty.turn_timeout_ms);
        assert_eq!(empty.turn_timeout_ms, 30_000);
        assert_eq!(empty.session_dir, std::path::Path::new("./logs/sessions"));
    }
}
