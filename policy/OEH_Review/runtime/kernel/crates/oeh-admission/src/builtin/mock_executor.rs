//! `mock-executor` 插件：给配置里列出的每个能力装一个"什么也不做，但做得很像"的执行器。
//!
//! 它按 `heartbeat_ms` 的节奏发心跳（喂租约），跑满 `duration_s`（或提案自带的
//! `params.duration_s`）就结束，中途被撤销则立刻返回——用来在没有真实机器人的情况下
//! 演练抢占、租约过期和 FSM 抢占。

use std::sync::Arc;
use std::time::{Duration, Instant};

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::proposal::{BoxFuture, ProposalId};
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::Admission;
use crate::capability::{CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    capabilities: Vec<CapabilitySpec>,
    #[serde(default = "d_duration_s")]
    duration_s: f64,
    #[serde(default = "d_heartbeat_ms")]
    heartbeat_ms: u64,
}
fn d_duration_s() -> f64 {
    0.2
}
fn d_heartbeat_ms() -> u64 {
    50
}

/// 非有限 / 负数的时长一律当 0：`Duration::from_secs_f64` 对它们会 panic，而一个配置
/// 错误不该把插件树炸穿。
fn sane_secs(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        0.0
    }
}

/// 公开是为了测试：Plan 2b 里没有真实适配器，想验"内核确实回头叫了停"就得拿到这个
/// 假执行器本身。生产路径仍然只经插件构造。
pub struct MockExecutor {
    duration_s: f64,
    heartbeat: Duration,
    /// 内核叫停过哪几次执行（`Executor::stop`）。假执行器没有适配器可发，就记下来。
    stopped: Mutex<Vec<ProposalId>>,
}

impl MockExecutor {
    pub fn new(duration_s: f64, heartbeat_ms: u64) -> Self {
        Self {
            duration_s: sane_secs(duration_s),
            heartbeat: Duration::from_millis(heartbeat_ms.max(1)),
            stopped: Mutex::new(Vec::new()),
        }
    }

    /// 内核叫停过的那些执行，按顺序。
    pub fn stopped(&self) -> Vec<ProposalId> {
        self.stopped.lock().clone()
    }
}

impl Executor for MockExecutor {
    fn execute(&self, call: ExecutionCall, ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let duration = call
            .params
            .get("duration_s")
            .and_then(|v| v.as_f64())
            .map(sane_secs)
            .unwrap_or(self.duration_s);
        let hb = self.heartbeat;
        let mut ctx = ctx;
        Box::pin(async move {
            let started = Instant::now();
            loop {
                // 起跑前就被撤销（抢占、FSM）也算数：先看取消，再发心跳。
                if ctx.is_cancelled() {
                    break;
                }
                ctx.heartbeat();
                let elapsed = started.elapsed().as_secs_f64();
                if elapsed >= duration {
                    break;
                }
                let left = Duration::from_secs_f64(duration - elapsed);
                tokio::select! {
                    _ = tokio::time::sleep(hb.min(left)) => {}
                    _ = ctx.cancelled() => break,
                }
            }
            ExecOutcome {
                ok: true,
                result: json!({"elapsed_s": started.elapsed().as_secs_f64()}),
                error: None,
            }
        })
    }

    /// 真适配器在这里会推零速度指令；假执行器只把"内核叫过停"记下来。
    fn stop(&self, proposal_id: &ProposalId) -> BoxFuture<()> {
        self.stopped.lock().push(proposal_id.clone());
        Box::pin(async {})
    }
}

pub struct MockExecutorPlugin;

impl Plugin for MockExecutorPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("mock-executor").inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let adm = ctx.get::<Admission>("admission")?;
        let exec: Arc<dyn Executor> = Arc::new(MockExecutor::new(cfg.duration_s, cfg.heartbeat_ms));
        for spec in cfg.capabilities {
            let d = adm
                .capabilities()
                .register(spec, exec.clone())
                .map_err(|e| PluginError::Other(e.to_string()))?;
            ctx.on_dispose(d);
        }
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: "mock-executor", build: || Box::new(MockExecutorPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_sanity() {
        let c: Config = serde_json::from_value(json!({"capabilities": []})).unwrap();
        assert_eq!(c.duration_s, 0.2);
        assert_eq!(c.heartbeat_ms, 50);
        assert_eq!(sane_secs(-1.0), 0.0);
        assert_eq!(sane_secs(f64::NAN), 0.0);
        assert_eq!(sane_secs(1.5), 1.5);
    }

    /// 叫停记的是**被叫停的那一次**，不是"当前在跑的那个 motion"。
    #[test]
    fn stop_records_the_proposal_it_was_called_for() {
        let e = MockExecutor::new(0.1, 10);
        assert!(e.stopped().is_empty());
        drop(e.stop(&ProposalId("t-p7".into())));
        assert_eq!(e.stopped(), vec![ProposalId("t-p7".into())]);
    }
}
