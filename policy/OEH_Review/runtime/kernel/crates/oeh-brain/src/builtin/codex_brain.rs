//! `brain-codex` 插件：起一个 `codex app-server` 子进程，把它挂到内核的 `brains` 服务上。
//!
//! 配置：
//!
//! ```yaml
//! - id: brain-codex
//!   config:
//!     brain_id: slow
//!     wake: wake/slow.yaml          # 相对进程 cwd，和 brain-native 一样
//!     command: [codex, app-server]  # argv，第一项是可执行文件
//!     cwd: ./workspace              # 可选：子进程的工作目录
//!     turn_timeout_ms: 120000
//! ```
//!
//! **`apply` 是同步的，起子进程不是。** 所以这一行做的是：把配置（含唤醒策略文件）在
//! 这里就校验干净——配错了要现在报，不是等到 attach 之后每次唤醒记一条错——然后
//! `Handle::try_current()` 拿到内核的 runtime，`spawn` 一个任务去
//! [`CodexBrain::spawn`] + `Brains::attach`。**绝不在插件线程上等子进程**：`settle` 的
//! 那一轮里所有插件是串行激活的，一个起得慢的 codex 会把整个 profile 的激活拖住。
//!
//! 起进程或者 attach 失败 → 一条 `brain/error{brain_id, reason}` 进世界日志，这一行
//! 就此什么也不挂（插件本身仍然是激活的：它已经把该做的做完了，失败的是子进程）。
//! **不自动重试**，和"子进程中途死掉不自动重启"是同一条纪律。
//!
//! **能力面按 turn 对账**（和 `brain-native` 一样）：每个 turn 开始时若
//! `CapabilityRegistry::generation()` 变了就重建 `additional_context` 里的工具面，一个 turn
//! 之内不变。行序仍是纪律：执行器那几行排在脑这几行前面，而 bundle 里 `brain-codex` 排在
//! `brain-native` 前面、两者都在最后。

use std::path::PathBuf;
use std::sync::Arc;

use oeh_admission::admission::Admission;
use oeh_kernel::factories::PluginFactory;
use oeh_kernel::plugin::Disposer;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{Brains, Ctx, Plugin, PluginError};
use oeh_proto::{EventDraft, EventKind, Manifest};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::codex::{CodexBrain, CodexConfig};

const ID: &str = "brain-codex";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    brain_id: String,
    /// YAML 唤醒策略文件，相对进程 cwd（和 `brain-native` 同一条规则）。
    wake: PathBuf,
    /// 子进程的 argv。第一项是可执行文件。
    command: Vec<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default = "d_turn_timeout_ms")]
    turn_timeout_ms: u64,
}

fn d_turn_timeout_ms() -> u64 {
    120_000
}

/// attach 的结果要在**两条时间线**之间交接：`apply` 同步注册 `on_dispose`，而真正的
/// `Disposer` 要等那个 spawn 出来的任务把脑挂上才有。`cancelled` 管的是反过来的次序
/// ——这一行在子进程起好之前就被撤销了。
#[derive(Default)]
struct AttachSlot {
    cancelled: bool,
    disposer: Option<Disposer>,
}

pub struct CodexBrainPlugin;

impl Plugin for CodexBrainPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["brains", "admission", "world-log"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let brains = ctx.get::<Brains>("brains")?;
        let adm = ctx.get::<Admission>("admission")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let bad = |reason: String| PluginError::Config {
            plugin: ID.into(),
            reason,
        };

        if cfg.command.is_empty() {
            return Err(bad("command must have at least the executable".into()));
        }
        let text = std::fs::read_to_string(&cfg.wake)
            .map_err(|e| bad(format!("wake policy {}: {e}", cfg.wake.display())))?;
        let policy = WakePolicy::from_yaml(&text)
            .map_err(|e| bad(format!("wake policy {}: {e}", cfg.wake.display())))?;
        // 子进程、JSON-RPC 客户端、看门狗全都活在 tokio 上。没有 runtime 就没得谈——
        // 现在说清楚，好过 attach 之后第一次唤醒才发现。
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| bad("brain-codex needs to be applied on a tokio runtime".into()))?;

        let slot: Arc<Mutex<AttachSlot>> = Arc::default();
        ctx.on_dispose(Box::new({
            let slot = slot.clone();
            move || {
                let d = {
                    let mut g = slot.lock();
                    g.cancelled = true;
                    g.disposer.take()
                };
                // 出锁之后才 detach：`Disposer` 会 drop 掉 sink（`CodexBrain::drop` 杀
                // 子进程），没有理由把那一下放进临界区。
                if let Some(d) = d {
                    d();
                }
            }
        }));

        let codex = CodexConfig {
            brain_id: cfg.brain_id.clone(),
            command: cfg.command,
            cwd: cfg.cwd,
            turn_timeout_ms: cfg.turn_timeout_ms,
        };
        let brain_id = cfg.brain_id;
        rt.spawn(async move {
            let sink = match CodexBrain::spawn(codex, adm, log.clone()).await {
                Ok(s) => s,
                Err(e) => return error_event(&log, &brain_id, &e.to_string()),
            };
            // 起进程要花时间，这中间这一行完全可能已经被撤销了。`sink` 在这里析构，
            // 子进程跟着被杀。
            if slot.lock().cancelled {
                return;
            }
            let disposer = match brains.attach(&brain_id, policy, sink) {
                Ok(d) => d,
                Err(e) => return error_event(&log, &brain_id, &e.to_string()),
            };
            let mut g = slot.lock();
            if g.cancelled {
                // 和上面那次检查之间被撤销了：`on_dispose` 已经跑完，没人会来取这个
                // disposer，这里自己收。
                drop(g);
                disposer();
            } else {
                g.disposer = Some(disposer);
            }
        });
        Ok(())
    }
}

/// 起不来 / 挂不上：世界日志里必须看得见，而这两件事都不属于任何一个 turn（没有 trace）。
fn error_event(log: &WorldLog, brain_id: &str, reason: &str) {
    log.append(EventDraft::new(
        EventKind::new("brain/error").expect("static kind"),
        "brain",
        json!({"brain_id": brain_id, "reason": reason}),
    ));
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(CodexBrainPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> Result<Config, serde_json::Error> {
        serde_json::from_value(v)
    }

    fn parse_err(v: Value) -> String {
        match parse(v) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("这份配置本该被拒"),
        }
    }

    #[test]
    fn the_documented_defaults_are_the_brief_s_numbers() {
        let c = parse(json!({
            "brain_id": "slow",
            "wake": "wake/slow.yaml",
            "command": ["codex", "app-server"]
        }))
        .unwrap();
        assert_eq!(c.turn_timeout_ms, 120_000);
        assert!(c.cwd.is_none());
        assert_eq!(c.command, vec!["codex", "app-server"]);
    }

    #[test]
    fn a_typo_in_the_config_is_rejected_rather_than_ignored() {
        let err = parse_err(json!({
            "brain_id": "slow",
            "wake": "w.yaml",
            "command": ["codex"],
            "turn_timeout": 10
        }));
        assert!(err.contains("turn_timeout"), "{err}");
    }

    #[test]
    fn the_command_is_required() {
        let err = parse_err(json!({"brain_id": "slow", "wake": "w.yaml"}));
        assert!(err.contains("command"), "{err}");
    }
}
