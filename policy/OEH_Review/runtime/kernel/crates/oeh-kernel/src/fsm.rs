use std::sync::Arc;

use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::state::StateRegistry;
use crate::world_log::WorldLog;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FsmState {
    Boot,
    Standing,
    Engaged,
    Acting,
    EmergencyStop,
    Fault,
    Recovering,
}

impl std::fmt::Display for FsmState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(self).expect("enum serializes");
        f.write_str(s.trim_matches('"'))
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FsmError {
    #[error("illegal transition {from} -> {to}")]
    Illegal { from: FsmState, to: FsmState },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub from: FsmState,
    pub to: FsmState,
}

fn legal(from: FsmState, to: FsmState) -> bool {
    use FsmState::*;
    match (from, to) {
        (Boot, Standing)
        | (Standing, Engaged)
        | (Engaged, Acting)
        | (Acting, Standing)
        | (Engaged, Standing) => true,
        (f, EmergencyStop) if f != EmergencyStop => true,
        (_, Fault) => true,
        (EmergencyStop, Recovering) | (Fault, Recovering) | (Recovering, Standing) => true,
        _ => false,
    }
}

/// 7 态安全状态机；迁移写 `safety/fsm` 事件和 `fsm` 状态寄存器。
pub struct RobotFsm {
    /// 串行化整个 `transition`（翻转 + 写寄存器 + append），见 [`RobotFsm::transition`]。
    /// 与 `state` 分开：`state()` 只读当前值，不该被一次进行中的 append 挡住。
    order: Mutex<()>,
    state: Mutex<FsmState>,
    log: Arc<WorldLog>,
    registry: Arc<StateRegistry>,
}

impl RobotFsm {
    /// `new` 不写 `safety/fsm`：BOOT 是初值而不是一次迁移，凭空造一条 `BOOT → BOOT`
    /// 会让事件链的第一条没有真实的 `from`。只有寄存器被初始化成 BOOT。
    pub fn new(log: Arc<WorldLog>, registry: Arc<StateRegistry>) -> Self {
        registry.set("fsm", json!(FsmState::Boot), "robot-fsm");
        Self {
            order: Mutex::new(()),
            state: Mutex::new(FsmState::Boot),
            log,
            registry,
        }
    }
    pub fn state(&self) -> FsmState {
        *self.state.lock()
    }
    pub fn allows_motion(&self) -> bool {
        matches!(
            self.state(),
            FsmState::Standing | FsmState::Engaged | FsmState::Acting
        )
    }
    /// 对并发调用者是一个原子步骤：状态翻转、`fsm` 寄存器写、`safety/fsm` append 全在
    /// `order` 锁内完成，否则两个线程会让 `state()`、寄存器和事件顺序三者互相矛盾。
    ///
    /// **不可重入**：`append` 会同步跑世界日志的观察者，所以 `transition` 不能从一个
    /// 能反过来摸到本 FSM 的 `WorldLog` 观察者里调用（`parking_lot::Mutex` 不可重入，
    /// 那会直接死锁）。需要在观察者里改状态的插件（`distill-fsm`）只对**别的** kind
    /// 反应，不对 `safety/fsm` 自己反应。
    pub fn transition(
        &self,
        to: FsmState,
        reason: &str,
        source: &str,
    ) -> Result<Transition, FsmError> {
        let _order = self.order.lock();
        let from = {
            let mut g = self.state.lock();
            let from = *g;
            if from == to {
                return Ok(Transition { from, to });
            }
            if !legal(from, to) {
                return Err(FsmError::Illegal { from, to });
            }
            *g = to;
            from
        };
        self.registry.set("fsm", json!(to), source);
        self.log.append(
            EventDraft::new(
                EventKind::new("safety/fsm").expect("static"),
                source,
                json!({"from": from, "to": to, "reason": reason}),
            )
            .with_salience(230)
            .with_collapse_key("fsm"),
        );
        Ok(Transition { from, to })
    }
}
