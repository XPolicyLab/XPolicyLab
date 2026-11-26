//! 唤醒预算数的是**开 turn 这件事**，不是事件上那个 `wake` / `steer` 的标签。
//!
//! 两条边界，这个文件把它们各钉一遍：
//!
//! - 有 turn 在跑的时候，steer 只是投进那个已经开着的 turn，什么也不开——一条都不该被
//!   预算吞掉。预算一旦连这种 steer 一起压，一个刚被唤醒过的脑就会在接下来整整
//!   `min_interval_ms` 里对世界装聋：`brain/answer`（慢脑回话）、`act/refused`（提案被
//!   拒）、`safety/fsm`（世界变了）在 `wake/fast.yaml` 里全是 steer。症状是"脑收不到
//!   回答"，而根因藏在一个名字里写着 wake 的预算里。
//! - **Idle** 的脑收到 steer 等同 wake（规格 §4.5，`route_one` 的 `(Steer, Idle)` →
//!   `Plan::StartTurn`），那就是实打实地开一个 turn——它必须和 wake 一样受限，否则
//!   `wake/fast.yaml` 里每一个映射成 steer 的 kind 都是一条绕开限频的开 turn 入口，而
//!   `brain/answer` / `brain/inject` 这些 kind 任何一个 RPC 客户端都发得出来。
//!
//! `Interrupt` 两边都不管：它确实会开一个新 turn，但 `safety/estop` 正是靠它进来的。

use oeh_kernel::brains::{BrainSink, BrainStatus, Brains, BrainsConfig, TurnHandle};
use oeh_kernel::observation::Observation;
use oeh_kernel::session_log::{read_session, SessionEntry};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::wake::{BrainCursor, WakeAction, WakePolicy};
use oeh_kernel::{ManualClock, WorldLog};
use oeh_proto::{EventDraft, EventKind, WorldEvent};
use parking_lot::Mutex;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 开着 turn 不放的脑：steer 要投进一个**活着的** turn 才走 `sink.steer`。
struct Holder {
    turns: AtomicU64,
    steers: AtomicU64,
    interrupts: AtomicU64,
    held: Mutex<Option<TurnHandle>>,
}

impl Holder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            turns: AtomicU64::new(0),
            steers: AtomicU64::new(0),
            interrupts: AtomicU64::new(0),
            held: Mutex::new(None),
        })
    }
}

impl BrainSink for Holder {
    fn on_turn(&self, _obs: Observation, turn: TurnHandle) {
        self.turns.fetch_add(1, Ordering::SeqCst);
        *self.held.lock() = Some(turn);
    }
    fn steer(&self, _obs: Observation) {
        self.steers.fetch_add(1, Ordering::SeqCst);
    }
    fn interrupt(&self) {
        self.interrupts.fetch_add(1, Ordering::SeqCst);
    }
}

/// 一开 turn 就收尾的脑：Idle → 开一个 → 回 Idle，于是下一条事件判定时它又是 Idle 的。
struct Quick {
    turns: AtomicU64,
}

impl Quick {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            turns: AtomicU64::new(0),
        })
    }
}

impl BrainSink for Quick {
    fn on_turn(&self, _obs: Observation, turn: TurnHandle) {
        self.turns.fetch_add(1, Ordering::SeqCst);
        turn.done("completed");
    }
    fn steer(&self, _obs: Observation) {}
    fn interrupt(&self) {}
}

fn ev(kind: &str) -> oeh_proto::WorldEvent {
    EventDraft::new(EventKind::new(kind).unwrap(), "t", json!({})).into_event(1, 0)
}

/// 慢脑的回话。**不带 `brain_id`**：带上目标脑自己的名字会被 `route_one` 的防自激守卫
/// 丢掉，那样测的就不是预算了。
fn answer(n: u64) -> EventDraft {
    EventDraft::new(
        EventKind::new("brain/answer").unwrap(),
        "slow",
        json!({"text": format!("answer {n}")}),
    )
}

/// 判定层：预算见底时，同一条 steer 在 Running 的脑上照样通过、在 Idle 的脑上被压住；
/// interrupt 两种状态都通过。
#[test]
fn a_spent_budget_suppresses_a_turn_opening_steer_but_not_one_into_a_live_turn() {
    let p = WakePolicy::from_yaml(
        "default: accumulate\n\
         rules: { obs/person: wake, brain/answer: steer, safety/estop: interrupt }\n\
         budget: { max_consecutive_wakes: 1, min_interval_ms: 10000 }",
    )
    .unwrap();
    let mut c = BrainCursor::new(0);
    let d = c.decide(&p, &ev("obs/person"), 0, BrainStatus::Idle);
    assert_eq!((d.action, d.suppressed), (WakeAction::Wake, false));
    c.note_wake(0);

    // 预算见底（1 次已经用掉，而且 10 s 的间隔一步都没走）。
    // 唤醒两种状态都受限：Idle 时立刻开一个 turn，Running 时被 latch 住、当前 turn 一
    // 结束就开——都是"多跑一个 turn"。
    for st in [BrainStatus::Idle, BrainStatus::Running] {
        let d = c.decide(&p, &ev("obs/person"), 1, st);
        assert_eq!(
            (d.action, d.suppressed),
            (WakeAction::Accumulate, true),
            "唤醒在 {st:?} 上仍然被压住"
        );
    }
    // 有 turn 在跑：steer 投进去，什么也不开。
    for (i, st) in [BrainStatus::Running, BrainStatus::Maintenance]
        .into_iter()
        .cycle()
        .take(5)
        .enumerate()
    {
        let d = c.decide(&p, &ev("brain/answer"), 2 + i as u64, st);
        assert_eq!(
            (d.action, d.suppressed),
            (WakeAction::Steer, false),
            "投进活着的 turn 的 steer 不开 turn，不受预算约束"
        );
    }
    // Idle：同一条 steer 等同 wake，开一个 turn —— 受限。
    let d = c.decide(&p, &ev("brain/answer"), 7, BrainStatus::Idle);
    assert_eq!(
        (d.action, d.suppressed),
        (WakeAction::Accumulate, true),
        "Idle 的脑上 steer 就是一次开 turn，必须和 wake 一样受限"
    );
    // 打断永远通过，两种状态都是。
    for st in [BrainStatus::Idle, BrainStatus::Running] {
        let d = c.decide(&p, &ev("safety/estop"), 8, st);
        assert_eq!((d.action, d.suppressed), (WakeAction::Interrupt, false));
    }
    // 被压住的判定不计账，投进去的 steer 也不计账：门槛一点没动。
    assert_eq!(c.consecutive_wakes, 1);
}

/// 路由层：一次唤醒把预算用光之后，五条 `brain/answer` 全部投进那个还开着的 turn
/// （五条 `SessionEntry::Steer`），而第二次唤醒仍旧记 `brain/wake{suppressed:true}`。
#[tokio::test]
async fn a_spent_wake_budget_still_delivers_every_steer_to_the_live_turn() {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock.clone(),
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    let sink = Holder::new();
    let policy = WakePolicy::from_yaml(
        "default: accumulate\n\
         rules: { obs/person: wake, brain/answer: steer, brain/*: ignore }\n\
         budget: { max_consecutive_wakes: 1, min_interval_ms: 10000 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();

    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    assert_eq!(sink.turns.load(Ordering::SeqCst), 1);
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));

    // 时钟一步不走：`min_interval_ms: 10000` 之内，预算对唤醒始终是关着的。
    for i in 0..5 {
        log.append(answer(i));
    }
    brains.settle().await;
    assert_eq!(
        sink.steers.load(Ordering::SeqCst),
        5,
        "五条 steer 一条都不能被唤醒预算吞掉"
    );

    // 唤醒那一侧不变：还是压着。
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    assert_eq!(
        sink.turns.load(Ordering::SeqCst),
        1,
        "预算没恢复，第二次唤醒不开 turn"
    );
    let suppressed: Vec<WorldEvent> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "brain/wake" && e.payload["suppressed"] == json!(true))
        .collect();
    assert_eq!(
        suppressed.len(),
        1,
        "只有那一条唤醒被抑制；五条 steer 一条抑制记录都不该有"
    );
    assert_eq!(suppressed[0].payload["kind"], "obs/person");

    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    let steers = session
        .iter()
        .filter(|e| matches!(e, SessionEntry::Steer { .. }))
        .count();
    assert_eq!(steers, 5, "会话日志里也要看得见这五条");
}

/// 路由层（Idle 的那一半）：一条策略 `steer` 落在 Idle 的脑上就是一次开 turn，所以窗口
/// 之内的第二条被压住，并且照常记一条 `brain/wake{suppressed:true}`。
///
/// 时钟一步不走，`min_interval_ms: 10_000` 于是永远没过去；`max_consecutive_wakes: 5`
/// 写得很宽，好让失败的原因只可能是间隔那一条（而不是"连续次数用光了"）。
#[tokio::test]
async fn a_policy_steer_that_opens_a_turn_on_an_idle_brain_is_rate_limited() {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock.clone(),
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    let sink = Quick::new();
    let policy = WakePolicy::from_yaml(
        "default: accumulate\n\
         rules: { obs/surprise: steer, brain/*: ignore }\n\
         budget: { max_consecutive_wakes: 5, min_interval_ms: 10000 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();

    log.append(EventDraft::new(
        EventKind::new("obs/surprise").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    assert_eq!(
        sink.turns.load(Ordering::SeqCst),
        1,
        "Idle 的脑收到 steer 等同 wake，开一个 turn"
    );
    assert_eq!(
        brains.status("fast"),
        Some(BrainStatus::Idle),
        "turn 收尾了"
    );

    log.append(EventDraft::new(
        EventKind::new("obs/surprise").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    assert_eq!(
        sink.turns.load(Ordering::SeqCst),
        1,
        "窗口之内的第二条 steer 不该再开一个 turn"
    );
    let suppressed: Vec<WorldEvent> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "brain/wake" && e.payload["suppressed"] == json!(true))
        .collect();
    assert_eq!(suppressed.len(), 1, "被压住的那一条要留下记录");
    assert_eq!(suppressed[0].payload["kind"], "obs/surprise");
}
