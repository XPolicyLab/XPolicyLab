use std::path::PathBuf;
use std::sync::Arc;

use oeh_proto::{EventDraft, EventKind, Manifest};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::brains::{BrainSink, Brains, TurnHandle};
use crate::factories::PluginFactory;
use crate::observation::Observation;
use crate::plugin::{Ctx, Plugin, PluginError};
use crate::wake::WakePolicy;
use crate::world_log::WorldLog;

/// 每次被唤醒就提一次的提案（`propose_on_wake`）。不配就是老行为：记一条事件、立刻
/// 结束 turn。`params` 必填——没有参数的能力写 `params: {}`，而不是让它悄悄变成 `null`
/// 然后栽在词汇校验上。
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct ProposeOnWake {
    capability: String,
    params: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    brain_id: String,
    /// YAML 唤醒策略文件，相对进程 cwd。
    wake: PathBuf,
    #[serde(default)]
    propose_on_wake: Option<ProposeOnWake>,
}

/// 不带模型的脑：把每个 turn / steer / interrupt 记成一条世界日志事件，立刻结束 turn。
/// 配了 `propose_on_wake` 的话，结束之前先把那一条提案走一遍准入服务。
struct MockBrain {
    log: Arc<WorldLog>,
    brain_id: String,
    propose_on_wake: Option<ProposeOnWake>,
}

impl MockBrain {
    fn emit(&self, kind: &str, payload: Value) {
        self.log.append(EventDraft::new(
            EventKind::new(kind).expect("static kind"),
            "mock-brain",
            payload,
        ));
    }
}

impl BrainSink for MockBrain {
    /// 回调必须非阻塞：`propose` 返回的是一个 future（准入要跑规则链、可能要等审批、
    /// 还要等执行器），在这里 await 会把 brains 的路由任务按住。所以提案连同 `TurnHandle`
    /// 一起交给一个 tokio 任务：结局回来之后记 `brain/mock_outcome`，再 `done`。
    fn on_turn(&self, obs: Observation, turn: TurnHandle) {
        self.emit(
            "brain/mock_turn",
            json!({
                "brain_id": self.brain_id,
                "turn": turn.turn,
                "n_events": obs.events.len(),
                "wake_kind": obs.wake_reason.kind,
            }),
        );
        let Some(p) = self.propose_on_wake.clone() else {
            turn.done("completed");
            return;
        };
        let (log, brain_id) = (self.log.clone(), self.brain_id.clone());
        tokio::spawn(async move {
            let outcome = turn.propose(&p.capability, p.params).await;
            // §12：每个 `act/*` 与 `brain/*` 事件都带 `trace_id`，一个 turn 的提案、
            // 裁决、执行、结果靠它串成一条链。Plan 4 的脑会照抄这个文件。
            log.append(
                EventDraft::new(
                    EventKind::new("brain/mock_outcome").expect("static kind"),
                    "mock-brain",
                    json!({
                        "brain_id": brain_id,
                        "turn": turn.turn,
                        "outcome": serde_json::to_value(&outcome).unwrap_or(Value::Null),
                    }),
                )
                .with_trace_id(turn.trace_id.as_str()),
            );
            turn.done("completed");
        });
    }
    fn steer(&self, obs: Observation) {
        self.emit(
            "brain/mock_steer",
            json!({
                "brain_id": self.brain_id,
                "n_events": obs.events.len(),
                "wake_kind": obs.wake_reason.kind,
            }),
        );
    }
    fn interrupt(&self) {
        self.emit("brain/mock_interrupt", json!({"brain_id": self.brain_id}));
    }
}

pub struct MockBrainPlugin;

impl Plugin for MockBrainPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("mock-brain").inject(&["brains", "world-log"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let brains = ctx.get::<Brains>("brains")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let bad = |reason: String| PluginError::Config {
            plugin: "mock-brain".into(),
            reason,
        };
        let text = std::fs::read_to_string(&cfg.wake)
            .map_err(|e| bad(format!("wake policy {}: {e}", cfg.wake.display())))?;
        let policy = WakePolicy::from_yaml(&text)
            .map_err(|e| bad(format!("wake policy {}: {e}", cfg.wake.display())))?;
        let sink = Arc::new(MockBrain {
            log,
            brain_id: cfg.brain_id.clone(),
            propose_on_wake: cfg.propose_on_wake,
        });
        let disposer = brains
            .attach(&cfg.brain_id, policy, sink)
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: "mock-brain", build: || Box::new(MockBrainPlugin) } }
