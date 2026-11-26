use oeh_proto::{EventKind, WorldEvent};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::brains::BrainStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WakeAction {
    Ignore,
    Accumulate,
    Wake,
    Steer,
    Interrupt,
}

/// 唤醒预算：**多久之内最多开几个 turn**。
///
/// 它管的是**开 turn 这件事**，不是事件上那个 `wake` / `steer` 的标签。一条判定要不要
/// 过这一关，只看它会不会（现在或者紧接着）让这个脑多跑一个 turn——所以同一个
/// [`WakeAction::Steer`] 在两种状态下待遇完全不同：
///
/// | 判定 | 脑的状态 | 会不会开 turn | 受不受预算约束 |
/// |---|---|---|---|
/// | [`WakeAction::Wake`] | Idle | 立刻开一个 | **受** |
/// | [`WakeAction::Wake`] | Running / Maintenance | latch 住，当前 turn 一结束就开 | **受** |
/// | [`WakeAction::Steer`] | Idle | 退化成唤醒，开一个（规格 §4.5） | **受** |
/// | [`WakeAction::Steer`] | Running / Maintenance | 投进那个已经开着的 turn | 不受 |
/// | [`WakeAction::Interrupt`] | 任何 | 打断当前 turn 再开一个 | 不受（见下） |
///
/// 这条预算保护的东西：一个把高频 kind 映射成 `wake`（或者在 Idle 的脑上映射成 `steer`）
/// 的策略，不能让脑一秒钟开一百个 turn。它**不**该顺手让脑对世界装聋——`brain/answer`
/// （慢脑回话）、`act/refused`（提案被拒）、`safety/fsm`（世界变了）在 `wake/fast.yaml`
/// 里全是 steer，脑正在跑的时候它们要投进的那个 turn 本来就已经开着，一条都不该因为
/// "刚醒过"而被丢掉。
///
/// [`WakeAction::Interrupt`] 确实会开一个新 turn，但它**永远**不受预算约束：
/// `safety/estop` 这类事件正是靠它进来的，宁可多开一个 turn，也不能让一句"停下"被一条
/// 限频规则吞掉。它开出来的 turn 照样 [`BrainCursor::note_wake`]，**计入**后面的账。
///
/// 定向事件（`Brains::steer`/`interrupt`/`inject`）另说：它们连 [`BrainCursor::decide`]
/// 都不经过。定向唤醒真的开出一个 turn 时同样 `note_wake`，**计入**预算。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WakeBudget {
    /// 连续开几个 turn 之后停手（中间隔了 `min_interval_ms` 还没被唤醒就清零，见
    /// [`BrainCursor::note_idle`]）。
    #[serde(default = "d_max")]
    pub max_consecutive_wakes: u32,
    /// 两次唤醒之间至少隔多久。
    #[serde(default = "d_interval")]
    pub min_interval_ms: u64,
}
fn d_max() -> u32 {
    3
}
fn d_interval() -> u64 {
    500
}
impl Default for WakeBudget {
    fn default() -> Self {
        Self {
            max_consecutive_wakes: d_max(),
            min_interval_ms: d_interval(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WakePolicy {
    #[serde(default = "d_default")]
    pub default: WakeAction,
    #[serde(default)]
    pub rules: BTreeMap<String, WakeAction>,
    #[serde(default)]
    pub budget: WakeBudget,
}
fn d_default() -> WakeAction {
    WakeAction::Accumulate
}
impl Default for WakePolicy {
    fn default() -> Self {
        Self {
            default: d_default(),
            rules: BTreeMap::new(),
            budget: WakeBudget::default(),
        }
    }
}

impl WakePolicy {
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        serde_yaml_ng::from_str(text).map_err(|e| e.to_string())
    }
    /// 精确 kind → `<prefix>/*` → default。
    pub fn action_for(&self, kind: &EventKind) -> WakeAction {
        if let Some(a) = self.rules.get(kind.as_str()) {
            return *a;
        }
        if let Some(a) = self.rules.get(&format!("{}/*", kind.prefix())) {
            return *a;
        }
        self.default
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub action: WakeAction,
    pub suppressed: bool,
}

/// 一个脑的游标与唤醒预算状态。只判定，不做 I/O。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainCursor {
    pub cursor: u64,
    pub consecutive_wakes: u32,
    pub last_wake_ns: Option<u64>,
}

impl BrainCursor {
    pub fn new(cursor: u64) -> Self {
        Self {
            cursor,
            consecutive_wakes: 0,
            last_wake_ns: None,
        }
    }
    /// 现在还开得出一个新 turn 吗（连续次数没用光，且离上一次够远）。
    ///
    /// `pub`：`Brains` 在**执行锁**里还要再问一次。`decide` 那一次是乐观预筛，两次加锁
    /// 之间当前 turn 完全可能已经结束，于是一条本来只是"投进去"的 steer 退化成了"开一个
    /// 新 turn"——那一刻它是一次开 turn，就得过这一关。
    ///
    /// ms → ns 用 `saturating_mul`：`min_interval_ms` 来自 YAML，一个够大的值会让
    /// `* 1_000_000` 溢出、门槛回绕成一个很小的数，预算就形同虚设（debug 下则是 panic）。
    pub fn may_open_turn(&self, b: &WakeBudget, now_ns: u64) -> bool {
        self.consecutive_wakes < b.max_consecutive_wakes
            && self
                .last_wake_ns
                .map(|t| now_ns.saturating_sub(t) >= b.min_interval_ms.saturating_mul(1_000_000))
                .unwrap_or(true)
    }

    /// 这次判定要不要过预算这一关——也就是它会不会让这个脑多跑一个 turn。整张表和每一格
    /// 的理由见 [`WakeBudget`]。
    fn budgeted(action: WakeAction, status: BrainStatus) -> bool {
        match action {
            // Idle 时立刻开一个；Running/Maintenance 时被 latch 住，当前 turn 一结束
            // `finish_turn` 就用它开下一个——两种都是"多跑一个 turn"。
            WakeAction::Wake => true,
            // Idle 的脑收到 steer 等同 wake（`route_one` 的 `(Steer, Idle)` →
            // `Plan::StartTurn`，规格 §4.5），那就是一次开 turn；有 turn 在跑的时候它
            // 只是投进去，什么也不开，一条都不该被预算吞掉。
            WakeAction::Steer => status == BrainStatus::Idle,
            // 打断确实会开一个新 turn，但它是安全路径，永远不受限（见 [`WakeBudget`]）。
            WakeAction::Interrupt => false,
            WakeAction::Ignore | WakeAction::Accumulate => false,
        }
    }

    /// 判定一条事件：policy 说该做什么，预算说现在做不做得了。
    ///
    /// `status` 是**判定这一刻**这个脑的状态，因为同一个 `Steer` 在 Idle 和 Running 上
    /// 是两件事（见 [`WakeBudget`] 的表）。被预算挡下的判定降级成
    /// `Accumulate{suppressed: true}`：事件留在日志里，游标不动，调用方记一条
    /// `brain/wake{suppressed:true}`。
    pub fn decide(
        &mut self,
        policy: &WakePolicy,
        ev: &WorldEvent,
        now_ns: u64,
        status: BrainStatus,
    ) -> Decision {
        let action = policy.action_for(&ev.kind);
        if Self::budgeted(action, status) && !self.may_open_turn(&policy.budget, now_ns) {
            return Decision {
                action: WakeAction::Accumulate,
                suppressed: true,
            };
        }
        Decision {
            action,
            suppressed: false,
        }
    }
    pub fn note_wake(&mut self, now_ns: u64) {
        self.consecutive_wakes += 1;
        self.last_wake_ns = Some(now_ns);
    }
    pub fn note_idle(&mut self, now_ns: u64, b: &WakeBudget) {
        if self
            .last_wake_ns
            .map(|t| now_ns.saturating_sub(t) >= b.min_interval_ms.saturating_mul(1_000_000))
            .unwrap_or(true)
        {
            self.consecutive_wakes = 0;
        }
    }
    pub fn advance(&mut self, to_seq: u64) {
        self.cursor = self.cursor.max(to_seq);
    }
}
