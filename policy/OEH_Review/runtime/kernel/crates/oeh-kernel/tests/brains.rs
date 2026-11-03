use oeh_kernel::brains::{BrainSink, BrainStatus, Brains, BrainsConfig, BrainsError, TurnHandle};
use oeh_kernel::observation::Observation;
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, Proposer};
use oeh_kernel::session_log::{read_session, SessionEntry};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::{ManualClock, WorldLog};
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

struct Recorder {
    turns: Mutex<Vec<Observation>>,
    traces: Mutex<Vec<String>>,
    steers: AtomicU64,
    interrupts: AtomicU64,
    auto_done: bool,
    pending: Mutex<Option<TurnHandle>>,
}

impl Recorder {
    fn new(auto_done: bool) -> Arc<Self> {
        Arc::new(Self {
            turns: Mutex::new(vec![]),
            traces: Mutex::new(vec![]),
            steers: AtomicU64::new(0),
            interrupts: AtomicU64::new(0),
            auto_done,
            pending: Mutex::new(None),
        })
    }
}

impl BrainSink for Recorder {
    fn on_turn(&self, obs: Observation, turn: TurnHandle) {
        self.turns.lock().push(obs);
        self.traces.lock().push(turn.trace_id.clone());
        if self.auto_done {
            turn.done("completed");
        } else {
            *self.pending.lock() = Some(turn);
        }
    }
    fn steer(&self, _obs: Observation) {
        self.steers.fetch_add(1, Ordering::SeqCst);
    }
    fn interrupt(&self) {
        self.interrupts.fetch_add(1, Ordering::SeqCst);
    }
}

/// 什么都放行的准入服务：这些用例验的是会话日志，不是准入链。
struct Yes;
impl Proposer for Yes {
    fn propose(&self, _p: Proposal) -> BoxFuture<Outcome> {
        Box::pin(async { Outcome::Executed { result: json!({}) } })
    }
}

fn ev(kind: &str) -> EventDraft {
    EventDraft::new(EventKind::new(kind).unwrap(), "t", json!({"x": 1})).with_collapse_key(kind)
}

/// `Brains` 的零件，还没造 `Brains`：构造要在 runtime 里（见
/// `brains_new_outside_runtime_is_an_error`），想自带 `BrainsConfig` 的用例从这里起步。
fn setup_parts() -> (
    Arc<ManualClock>,
    Arc<WorldLog>,
    Arc<StateRegistry>,
    tempfile::TempDir,
) {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    (clock, log, state, dir)
}

fn setup() -> (
    Arc<ManualClock>,
    Arc<WorldLog>,
    Arc<StateRegistry>,
    Brains,
    tempfile::TempDir,
) {
    let (clock, log, state, dir) = setup_parts();
    let brains = Brains::new(
        clock.clone(),
        log.clone(),
        state.clone(),
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    (clock, log, state, brains, dir)
}

#[tokio::test]
async fn wake_builds_observation_advances_cursor_and_logs() {
    let (clock, log, state, brains, dir) = setup();
    state.set("fsm", json!("STANDING"), "t");
    let sink = Recorder::new(true);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, brain/*: ignore }\nbudget: { max_consecutive_wakes: 5, min_interval_ms: 0 }",
    )
    .unwrap();
    // attach 自己写了一条 brain/attach（seq 1），所以观测从 seq 1 起。
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/detections"));
    log.append(ev("obs/detections"));
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 0, "accumulate does not wake");
    clock.advance(1_000);
    log.append(ev("obs/person"));
    brains.settle().await;
    let turns = sink.turns.lock();
    assert_eq!(turns.len(), 1);
    let obs = &turns[0];
    assert_eq!(obs.source_seq_range, (1, 4));
    assert_eq!(obs.wake_reason.seq, 4);
    assert_eq!(obs.state["fsm"], json!("STANDING"));
    // brain/attach 被策略的 `brain/*: ignore` 挡在观测之外；两条都是 salience 0，
    // `build_observation` 的次序是 salience 降序后 last_seq 升序 → detections(3) 在 person(4) 前。
    assert_eq!(
        obs.events
            .iter()
            .map(|e| (e.collapse_key.as_str(), e.count))
            .collect::<Vec<_>>(),
        vec![("obs/detections", 2), ("obs/person", 1)]
    );
    drop(turns);
    assert_eq!(brains.cursor("fast"), Some(4));
    assert_eq!(
        brains.status("fast"),
        Some(BrainStatus::Idle),
        "auto_done returns to idle"
    );
    let kinds: Vec<String> = log
        .since(4)
        .unwrap()
        .iter()
        .map(|e| e.kind.as_str().to_string())
        .collect();
    assert_eq!(
        kinds,
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/turn_end"
        ]
    );
    // 会话日志的写盘在写线程上；读文件之前先排空。
    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    // I5: 每份会话日志以一条 Attached 开场——turn 号是 per-attach 的，读的人必须看得见
    // 这条分界线。
    assert!(matches!(&session[0], SessionEntry::Attached { brain_id, .. } if brain_id == "fast"));
    assert!(matches!(
        session[1],
        SessionEntry::TurnStart { turn: 1, .. }
    ));
    assert!(
        matches!(&session[2], SessionEntry::Observation { turn: 1, observation } if observation.source_seq_range == (1, 4))
    );
    assert!(
        matches!(&session[3], SessionEntry::TurnEnd { turn: 1, reason, .. } if reason == "completed")
    );
}

#[tokio::test]
async fn wake_while_running_is_latched_and_delivered_on_done() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 5, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(
        sink.turns.lock().len(),
        1,
        "second wake latched, not delivered"
    );
    let h = sink.pending.lock().take().unwrap();
    h.done("completed");
    brains.settle().await;
    assert_eq!(
        sink.turns.lock().len(),
        2,
        "latched wake delivered after done"
    );
    // turn 1 唤醒于 seq 2（brain/attach 是 seq 1），游标推进到 2；turn-1 的 bookkeeping 是
    // brain/wake(3) brain/turn_start(4) brain/cursor(5)，第二个 obs/person 是 6，
    // brain/turn_end 是 7。C2（fix round 1）之后 `observe_for` 按 `routed_seq` 封顶：
    // `brain/turn_end` 是 BOOKKEEPING，从不经过 `route_one`，从不推高 `routed_seq`——虽然
    // `finish_turn` 在调用 `start_turn` 之前就已经把它落了盘，但它"从未被路由过"，所以不
    // 出现在这次重新唤醒的观测里。latch 的 turn 观测因此止于 6，而不是意外带上 7。
    assert_eq!(sink.turns.lock()[1].source_seq_range, (3, 6));
}

/// C2 (fix round 1): an observation must never swallow an event that has not been routed
/// yet just because it happens to already be sitting in the world log. `fast` is
/// `Running`; `obs/surprise` (→ steer) and `safety/estop` (→ interrupt) are appended
/// back-to-back before a single `settle()`. Before the fix, the steer's `observe_for`
/// pulled `since(cursor)` unbounded, saw the estop already in the log, and advanced the
/// cursor straight past it — the estop was then dropped as "already consumed" and the
/// interrupt never happened at all.
#[tokio::test]
async fn steer_does_not_swallow_a_same_batch_interrupt() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, obs/surprise: steer, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person")); // turn 1 running
    brains.settle().await;
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    let first = sink.pending.lock().take().unwrap();

    log.append(ev("obs/surprise")); // steer, appended first
    log.append(ev("safety/estop")); // interrupt, appended right behind it — before settle()
    brains.settle().await;

    assert_eq!(
        sink.steers.load(Ordering::SeqCst),
        1,
        "the steer still reaches the live turn"
    );
    assert_eq!(
        sink.interrupts.load(Ordering::SeqCst),
        1,
        "the interrupt must not be swallowed by the steer's observation"
    );
    assert!(
        !first.is_current(),
        "the interrupted turn's handle is stale"
    );
    assert_eq!(
        sink.turns.lock().len(),
        2,
        "interrupt cancels the turn and starts a new one"
    );
}

/// C2 (fix round 1), the `Idle`/`start_turn` counterpart of the test above: `obs/speech`
/// (→ wake) and `safety/estop` (→ interrupt) are appended back-to-back while `fast` is
/// still `Idle`, before any `settle()`. Before the fix, `start_turn`'s `observe_for` for
/// the wake advanced the cursor past the already-logged estop, so the brain started a
/// turn that should have been immediately interrupted, and never was.
#[tokio::test]
async fn wake_does_not_swallow_a_same_batch_interrupt_from_idle() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/speech: wake, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();

    log.append(ev("obs/speech")); // wakes fast from idle
    log.append(ev("safety/estop")); // right behind it, before any settle()
    brains.settle().await;

    assert_eq!(
        sink.turns.lock().len(),
        2,
        "the wake starts a turn, the interrupt cancels it and starts a fresh one"
    );
    assert_eq!(sink.interrupts.load(Ordering::SeqCst), 1);
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
}

/// C2 (fix round 1), latch fidelity: a latched wake's observation, once it is finally
/// delivered by `finish_turn`, must still cover everything accumulated in the backlog
/// during the turn it was latched behind — capping `observe_for` at the triggering
/// event's own seq (an earlier, rejected fix) would have thrown the detections away.
#[tokio::test]
async fn latched_wake_observation_covers_events_accumulated_before_done() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let _d = brains
        .attach("fast", wake_on_person(), sink.clone())
        .unwrap();
    log.append(ev("obs/person")); // turn 1 running
    brains.settle().await;
    let latched = log.append(ev("obs/person")); // latched wake
    log.append(ev("obs/detections"));
    log.append(ev("obs/detections"));
    let last = log.append(ev("obs/detections"));
    brains.settle().await;
    assert_eq!(
        sink.turns.lock().len(),
        1,
        "latch does not open a new turn while running"
    );
    let h = sink.pending.lock().take().unwrap();
    h.done("completed");
    brains.settle().await;
    assert_eq!(
        sink.turns.lock().len(),
        2,
        "latched wake delivered after done"
    );
    assert_eq!(sink.turns.lock()[1].wake_reason.seq, latched.seq);
    assert!(
        sink.turns.lock()[1].source_seq_range.1 >= last.seq,
        "the latched turn's observation covers the events accumulated before done: {:?}",
        sink.turns.lock()[1].source_seq_range
    );
}

#[tokio::test]
async fn steer_and_interrupt_semantics() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, safety/fsm: steer, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("safety/fsm"));
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 1, "steer while idle wakes");
    log.append(ev("safety/fsm"));
    brains.settle().await;
    assert_eq!(
        sink.steers.load(Ordering::SeqCst),
        1,
        "steer while running delivers a steer"
    );
    log.append(ev("safety/estop"));
    brains.settle().await;
    assert_eq!(sink.interrupts.load(Ordering::SeqCst), 1);
    assert_eq!(
        sink.turns.lock().len(),
        2,
        "interrupt ends the turn and starts a new one"
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == "interrupted"));
}

#[tokio::test]
async fn suppressed_wake_is_logged_and_double_attach_rejected() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(true);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 1, min_interval_ms: 1000 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy.clone(), sink.clone()).unwrap();
    assert!(brains.attach("fast", policy.clone(), sink.clone()).is_err());
    // brain_id 会拼进会话日志的路径，路径分隔符和 `..` 必须在 attach 就被挡掉。
    assert!(matches!(
        brains.attach("../x", policy.clone(), sink.clone()),
        Err(BrainsError::InvalidBrainId(_))
    ));
    log.append(ev("obs/person"));
    brains.settle().await;
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 1);
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "brain/wake" && e.payload["suppressed"] == json!(true)));
}

/// C1: 打断发生时如果还 latch 着一个唤醒，只能开出**一个**新 turn（打断自己的那个）。
/// 否则 finish_turn 会用 latch 开 turn N+1、route 再开 turn N+2，两个重叠，而 N+1 的
/// 句柄过不了 finish_turn 的陈旧判定，永远收不到 turn_end。
#[tokio::test]
async fn interrupt_with_latched_wake_starts_exactly_one_turn() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person")); // turn 1 running
    brains.settle().await;
    log.append(ev("obs/person")); // latched
    brains.settle().await;
    log.append(ev("safety/estop"));
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 2, "exactly one new turn");
    assert_eq!(sink.interrupts.load(Ordering::SeqCst), 1);
    let ends: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "brain/turn_end")
        .collect();
    assert_eq!(ends.len(), 1, "exactly one turn_end: {ends:?}");
    assert_eq!(ends[0].payload["turn"], json!(1));
    assert_eq!(ends[0].payload["reason"], json!("interrupted"));
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    assert_eq!(sink.turns.lock()[1].wake_reason.kind, "safety/estop");
}

fn brain_wakes(log: &WorldLog) -> usize {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() == "brain/wake")
        .count()
}

/// I2: 两个都不 ignore `brain/*` 又都超预算的脑，不能靠互相唤醒把栈打穿。
#[tokio::test]
async fn two_brains_over_budget_do_not_recurse() {
    let (_clock, log, _state, brains, _dir) = setup();
    let a = Recorder::new(true);
    let b = Recorder::new(true);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, brain/mock_turn: wake }\nbudget: { max_consecutive_wakes: 1, min_interval_ms: 100000 }",
    )
    .unwrap();
    let _da = brains.attach("a", policy.clone(), a.clone()).unwrap();
    let _db = brains.attach("b", policy, b.clone()).unwrap();
    log.append(ev("obs/person")); // 必须返回：不爆栈
    brains.settle().await;
    assert_eq!(a.turns.lock().len(), 1);
    assert_eq!(b.turns.lock().len(), 1);
    let wakes = brain_wakes(&log);
    assert!(wakes < 20, "bounded brain/wake events, got {wakes}");

    // 真正互激的那一档：策略直接把 `brain/*` 当唤醒源。没有 BOOKKEEPING 过滤时 a 的
    // `brain/wake` 唤醒 b、b 的再唤醒 a，两边都超预算之后靠 `Plan::Suppressed` 的
    // `brain/wake` 无限对敲 —— 修复前这一段必然爆栈。
    let hot = WakePolicy::from_yaml(
        "rules: { obs/person: wake, brain/*: wake }\nbudget: { max_consecutive_wakes: 1, min_interval_ms: 100000 }",
    )
    .unwrap();
    let (c, d) = (Recorder::new(true), Recorder::new(true));
    let _dc = brains.attach("c", hot.clone(), c.clone()).unwrap();
    let _dd = brains.attach("d", hot, d.clone()).unwrap();
    let before = brain_wakes(&log);
    log.append(ev("obs/person"));
    brains.settle().await;
    let grew = brain_wakes(&log) - before;
    assert!(grew < 40, "mutual excitation stays bounded, grew by {grew}");
    assert_eq!(c.turns.lock().len(), 1);
    assert_eq!(d.turns.lock().len(), 1);
}

/// I3: 游标滑出保留窗口 = 这个脑漏了一段事件，必须在世界日志里留痕，光 warn 不算数。
#[tokio::test]
async fn cursor_too_old_is_recorded() {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()).with_retain(3));
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
    // 保留窗口只有 3 条，`since` 读不回全部历史 —— 用观察者抄一份。
    let seen: Arc<Mutex<Vec<(String, serde_json::Value)>>> = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    let _collector = log.observe(Arc::new(move |e: &oeh_proto::WorldEvent| {
        s2.lock()
            .push((e.kind.as_str().to_string(), e.payload.clone()));
    }));
    let sink = Recorder::new(true);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    for _ in 0..6 {
        log.append(ev("obs/detections")); // 游标停在 1，窗口转过去了
    }
    brains.settle().await;
    let person = log.append(ev("obs/person"));
    brains.settle().await;
    assert!(
        seen.lock()
            .iter()
            .any(|(k, p)| k == "brain/cursor" && p["reset"] == json!(true)),
        "cursor reset is on the world log: {:?}",
        seen.lock()
    );
    assert_eq!(sink.turns.lock()[0].source_seq_range.1, person.seq);
}

/// m2: detach 之后重新 attach，旧 attach 发出的 TurnHandle 不能关掉新 attach 的 turn
/// （两次 attach 的 turn 号都从 1 起，只有 epoch 能区分）。
#[tokio::test]
async fn a_stale_turn_handle_cannot_close_the_new_attachment() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let d = brains.attach("fast", policy.clone(), sink.clone()).unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    let stale = sink.pending.lock().take().unwrap();
    assert_eq!(stale.turn, 1);
    d();
    assert_eq!(brains.status("fast"), None, "detached");
    // m6: detach 也要在世界日志里留痕，否则 attach 没有配对的收尾。
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "brain/detach" && e.payload["brain_id"] == json!("fast")));
    let _d2 = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 2);
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    stale.done("completed");
    brains.settle().await;
    assert_eq!(
        brains.status("fast"),
        Some(BrainStatus::Running),
        "stale handle is a no-op"
    );
    assert_eq!(sink.turns.lock().len(), 2, "and starts no further turn");
}

/// C3: latch 住的唤醒可能在 turn 结束前就被一次 steer 消费掉了（steer 推进了游标）。
/// 此时再按它开一个 turn，观测里根本没有那条证据，`wake_reason.seq` 会落在
/// `source_seq_range` 之外。必须丢掉并留痕。
#[tokio::test]
async fn latched_wake_after_steer_is_dropped_as_already_delivered() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, safety/fsm: steer }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person")); // turn 1 running
    brains.settle().await;
    let latched = log.append(ev("obs/person")); // latch
    brains.settle().await;
    log.append(ev("safety/fsm")); // steer：观测覆盖到 latch 那条，游标推过去了
    brains.settle().await;
    assert_eq!(sink.steers.load(Ordering::SeqCst), 1);
    assert!(brains.cursor("fast").unwrap() >= latched.seq);
    let h = sink.pending.lock().take().unwrap();
    h.done("completed");
    brains.settle().await;
    assert_eq!(
        sink.turns.lock().len(),
        1,
        "latched wake must not open a turn"
    );
    assert_eq!(brains.status("fast"), Some(BrainStatus::Idle));
    let dropped: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| {
            e.kind.as_str() == "brain/wake" && e.payload["reason"] == json!("already_delivered")
        })
        .collect();
    assert_eq!(
        dropped.len(),
        1,
        "one already_delivered record: {dropped:?}"
    );
    assert_eq!(dropped[0].payload["seq"], json!(latched.seq));
    assert_eq!(dropped[0].payload["suppressed"], json!(true));
}

fn wake_on_person() -> WakePolicy {
    WakePolicy::from_yaml(
        "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap()
}

/// 超时很短的一套 Brains：turn 卡住/进入 Maintenance 的两个用例共用。
fn setup_with_timeout(
    turn_timeout_ms: u64,
) -> (Arc<ManualClock>, Arc<WorldLog>, Brains, tempfile::TempDir) {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock.clone(),
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig {
            turn_timeout_ms,
            ..Default::default()
        },
    )
    .unwrap();
    (clock, log, brains, dir)
}

async fn wait_for_status(brains: &Brains, want: BrainStatus) {
    for _ in 0..100 {
        if brains.status("fast") == Some(want) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// I3: 一个 turn 的全部 `brain/*` bookkeeping 事件共享一个 trace_id（§12），`TurnHandle`
/// 把同一个 id 交给脑，脑产出的 `act/*` 才能串回来。
#[tokio::test]
async fn brain_events_of_a_turn_share_one_trace_id() {
    let (_clock, log, _state, brains, _dir) = setup();
    let sink = Recorder::new(true);
    let _d = brains
        .attach("fast", wake_on_person(), sink.clone())
        .unwrap();
    let person = log.append(ev("obs/person"));
    brains.settle().await;
    let traced: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.prefix() == "brain" && e.kind.as_str() != "brain/attach")
        .collect();
    assert_eq!(
        traced.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/turn_end"
        ]
    );
    let ids: std::collections::BTreeSet<_> = traced.iter().map(|e| e.trace_id.clone()).collect();
    assert_eq!(ids.len(), 1, "one trace per turn: {ids:?}");
    let trace = traced[0].trace_id.clone().expect("trace_id is set");
    let parts: Vec<&str> = trace.split('-').collect();
    assert_eq!(parts.len(), 4, "t-<epoch>-<turn>-<seq>: {trace}");
    assert_eq!(parts[0], "t");
    assert_eq!(parts[2], "1", "turn number");
    assert_eq!(parts[3], person.seq.to_string(), "wake seq");
    assert_eq!(
        sink.traces.lock()[0],
        trace,
        "TurnHandle carries the same id"
    );
}

/// I7: 脑不调 `done` 就永远醒不过来了（§11：脑 Provider 异常 → 终止当前 turn，记 brain/error）。
#[tokio::test]
async fn a_stuck_turn_times_out_with_brain_error() {
    let (clock, log, brains, dir) = setup_with_timeout(50);
    let sink = Recorder::new(false); // 永不 done
    let _d = brains
        .attach("fast", wake_on_person(), sink.clone())
        .unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    clock.advance(60_000_000); // 内核时钟走过 60ms
    wait_for_status(&brains, BrainStatus::Idle).await;
    assert_eq!(brains.status("fast"), Some(BrainStatus::Idle));
    let err = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "brain/error")
        .expect("brain/error is on the world log");
    assert_eq!(err.payload["reason"], json!("turn_timeout"));
    assert_eq!(err.payload["turn"], json!(1));
    assert!(err.trace_id.is_some(), "brain/error carries the turn trace");
    // 会话日志的写盘在写线程上；读文件之前先排空。
    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    assert!(
        session
            .iter()
            .any(|e| matches!(e, SessionEntry::TurnEnd { reason, .. } if reason == "timeout")),
        "{session:?}"
    );
    assert_eq!(sink.turns.lock().len(), 1, "no turn was reopened");
}

/// I7: `Maintenance` 是压缩/记忆写入/子任务收尾的相位——不被超时掐断，唤醒只 latch。
#[tokio::test]
async fn maintenance_is_exempt_from_the_timeout_and_still_latches() {
    let (clock, log, brains, _dir) = setup_with_timeout(50);
    let sink = Recorder::new(false);
    let _d = brains
        .attach("fast", wake_on_person(), sink.clone())
        .unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    {
        let g = sink.pending.lock();
        let h = g.as_ref().unwrap();
        assert!(h.is_current());
        h.enter_maintenance();
    }
    assert_eq!(brains.status("fast"), Some(BrainStatus::Maintenance));
    clock.advance(500_000_000); // 远超 50ms
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert_eq!(
        brains.status("fast"),
        Some(BrainStatus::Maintenance),
        "maintenance is exempt"
    );
    log.append(ev("obs/person")); // Maintenance 期间只 latch
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 1, "no overlapping turn");
    let h = {
        let mut g = sink.pending.lock();
        g.take().unwrap()
    };
    h.leave_maintenance();
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    assert!(h.is_current());
    h.done("completed");
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 2, "latched wake delivered on done");
}

/// m4: 陈旧句柄不能往会话日志里写。turn 被 interrupt / 超时关掉之后它记下的 say 会挂在
/// **下一个** turn 的编号下面，读日志的人无从分辨。
///
/// `say` / `ask_human` 没有旁路：它们和别的能力一样走 `propose`，会话日志里的
/// `Say` / `AskHuman` 是那次提案的副产品（内核只做记账，准不准是准入服务的事）。
#[tokio::test]
async fn a_stale_turn_handle_cannot_write_to_the_session_log() {
    let (_clock, log, _state, brains, dir) = setup();
    let _p = brains.set_proposer(Arc::new(Yes));
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    let stale = sink.pending.lock().take().unwrap();
    stale.propose("say", json!({"text": "during turn 1"})).await;
    log.append(ev("safety/estop")); // turn 1 被打断，turn 2 开始
    brains.settle().await;
    assert!(!stale.is_current());
    stale.propose("say", json!({"text": "after turn 1"})).await;
    stale
        .propose("ask_human", json!({"question": "late question"}))
        .await;
    // 会话日志的写盘在写线程上；读文件之前先排空。
    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    let says: Vec<(u64, String)> = session
        .iter()
        .filter_map(|e| match e {
            SessionEntry::Say { turn, text } => Some((*turn, text.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(says, vec![(1, "during turn 1".to_string())]);
    assert!(!session
        .iter()
        .any(|e| matches!(e, SessionEntry::AskHuman { .. })));
}

/// 超时扫描的顺序：先 `finish_turn` 把 turn 真的关掉，关成功了才记 `brain/error`。
/// 反过来的话，`done()` 抢先收尾的 turn 也会被记一条 `brain/error`——把正常收尾诬告成
/// 脑 Provider 异常。日志里 `brain/turn_end` 必须排在 `brain/error` 前面。
#[tokio::test]
async fn timeout_sweep_closes_turn_before_logging_error() {
    let (clock, log, state, dir) = setup_parts();
    let brains = Brains::new(
        clock.clone(),
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig {
            obs_budget: Default::default(),
            turn_timeout_ms: 50,
        },
    )
    .unwrap();
    let sink = Recorder::new(false); // 永不 done
    let policy = WakePolicy::from_yaml(
        "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(brains.status("fast"), Some(BrainStatus::Running));
    clock.advance(60_000_000); // 内核时钟走过 60ms > 50ms
    brains.sweep_now();
    brains.settle().await;
    let kinds: Vec<String> = log
        .since(0)
        .unwrap()
        .into_iter()
        .map(|e| e.kind.as_str().to_string())
        .collect();
    let ends: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| k.as_str() == "brain/turn_end")
        .map(|(i, _)| i)
        .collect();
    let errs: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| k.as_str() == "brain/error")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(ends.len(), 1, "exactly one turn_end: {kinds:?}");
    assert_eq!(errs.len(), 1, "exactly one brain/error: {kinds:?}");
    assert!(
        ends[0] < errs[0],
        "the turn is closed before the error is logged: {kinds:?}"
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == json!("timeout")));
    assert_eq!(brains.status("fast"), Some(BrainStatus::Idle));
}

/// `Brains::new` 在 runtime 外是错误，不是 panic：插件 apply 里的 panic 会穿过整棵 fiber 树。
#[tokio::test]
async fn brains_new_outside_runtime_is_an_error() {
    let (clock, log, state, dir) = setup_parts();
    let r = std::thread::spawn(move || {
        Brains::new(
            clock,
            log,
            state,
            dir.path().to_path_buf(),
            BrainsConfig::default(),
        )
        .is_err()
    })
    .join()
    .unwrap();
    assert!(r, "Brains::new must not panic outside a runtime");
}

/// M1：detach 不能把会话日志的析构（flush，最长 1 s）关进 `inner.brains` 那把锁里。
///
/// 场景就是这条路径的真实过载形状：写线程卡在盘上（这里用 `stall_writer_for_tests`
/// 精确复现），1024 的队列被填满，于是 `SessionLog::drop` 里的 `flush()` 必须一直等到
/// 写线程醒过来。detach 跑在连接的读任务（`brain/detach`、断连清理）或**内核唯一的那条
/// Op 循环**（`Op::DisposeFiber`）上；只要它在锁里等，所有脑的唤醒/steer/interrupt 就
/// 跟着停——而 `status()` 正是拿同一把锁的最短路径，用它当探针。
///
/// 判据不是"快"而是"不被那段 I/O 牵连"：写线程睡 300 ms，探针必须在 50 ms 内回来。
/// 修复前这条断言会量到整整一个 stall。
#[tokio::test]
async fn detach_does_not_hold_the_brains_lock_while_the_session_log_drains() {
    let (_clock, _log, _state, brains, dir) = setup();
    let sink = Recorder::new(true);
    let detach = brains
        .attach("fast", WakePolicy::default(), sink.clone())
        .unwrap();

    // 写线程睡 300 ms；排在它后面的行把队列填满，`Drop` 的 flush 于是一定要等它。
    {
        let session = brains.session_log_for_tests("fast").unwrap();
        session.stall_writer_for_tests(Duration::from_millis(300));
        for i in 0..2000u64 {
            session
                .record(&SessionEntry::ModelOutput {
                    turn: 1,
                    step: i,
                    text: "x".into(),
                })
                .unwrap();
        }
        // 这个块作用域很要紧：测试自己攥着的这个 `Arc` 一放掉，brains 里那份才是最后
        // 一个强引用，detach 才会真的去析构它。
    }

    let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = started.clone();
    // 专用 OS 线程：`Disposer` 是同步的，它在 runtime 的 worker 上阻塞正是问题的一半。
    let detaching = std::thread::spawn(move || {
        flag.store(true, Ordering::SeqCst);
        detach();
    });
    while !started.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    // 让 detach 真的进到它的临界区/析构里再探（这段等待只是布景，不是判据）。
    std::thread::sleep(Duration::from_millis(30));

    // **这是一条拿墙钟当判据的断言**（本分支上另有两条：`recall_glob` 的病态模式计时，C1），因为这条性质（"这把锁没有被一个慢
    // 盘占住"）没有同步点可用：要测的正是"没有等待"。余量是 6×（50 ms 对 300 ms 的
    // 阻塞），所以它也是最可能在一台负载很重的机器上偶发失败的用例——失败时先看负载，
    // 再看 detach 是不是真的把 `SessionLog::drop` 搬回锁里去了。
    let probe = std::time::Instant::now();
    let _ = brains.status("fast");
    let waited = probe.elapsed();
    assert!(
        waited < Duration::from_millis(50),
        "status() waited {waited:?} on the brains lock — detach is draining the session log \
         inside it"
    );

    detaching.join().unwrap();
    assert_eq!(
        brains.status("fast"),
        None,
        "detach must still have happened"
    );
    drop(dir);
}

/// `TurnHandle::note`：脑这一侧的三种记账进会话日志，`turn` 由句柄决定（调用方填的被
/// 覆盖），别的变体一条都不接。
///
/// 这条边界是 Plan 4 的脑（`oeh-brain` 的原生循环、Task 4 的 Codex 包装）往会话日志里
/// 写字的**唯一**入口：`Proposal`/`Outcome`/`Say` 有提案编号与结局配对的不变式，
/// `TurnStart`/`TurnEnd` 是内核的记账——都不能由脑自己伪造。
#[tokio::test]
async fn note_writes_the_three_brain_side_entries_and_rejects_the_rest() {
    let (_clock, log, _state, brains, dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml("rules: { obs/person: wake }").unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    log.append(ev("obs/person"));
    brains.settle().await;
    let turn = sink.pending.lock().take().unwrap();
    assert_eq!(turn.turn, 1);

    // `turn: 99` 是调用方填错的那种，写进去的必须是句柄自己的 turn。
    turn.note(SessionEntry::ModelOutput {
        turn: 99,
        step: 0,
        text: "thinking".into(),
    })
    .unwrap();
    turn.note(SessionEntry::ToolCall {
        turn: 99,
        step: 0,
        call_id: "c1".into(),
        name: "robot_walk".into(),
        args: json!({"vx": 0.1}),
    })
    .unwrap();
    turn.note(SessionEntry::ToolResult {
        turn: 99,
        step: 0,
        call_id: "c1".into(),
        ok: true,
        result: json!({"outcome": "executed"}),
    })
    .unwrap();
    for rejected in [
        SessionEntry::Say {
            turn: 1,
            text: "forged".into(),
        },
        SessionEntry::TurnEnd {
            turn: 1,
            t_kernel: 0,
            reason: "forged".into(),
        },
        SessionEntry::Interrupt {
            turn: 1,
            t_kernel: 0,
        },
    ] {
        assert!(turn.note(rejected).is_err());
    }

    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    let mine: Vec<&SessionEntry> = session
        .iter()
        .filter(|e| {
            matches!(
                e,
                SessionEntry::ModelOutput { .. }
                    | SessionEntry::ToolCall { .. }
                    | SessionEntry::ToolResult { .. }
            )
        })
        .collect();
    assert!(
        matches!(
            mine.as_slice(),
            [
                SessionEntry::ModelOutput {
                    turn: 1,
                    step: 0,
                    ..
                },
                SessionEntry::ToolCall {
                    turn: 1,
                    step: 0,
                    ..
                },
                SessionEntry::ToolResult {
                    turn: 1,
                    step: 0,
                    ok: true,
                    ..
                },
            ]
        ),
        "{mine:?}"
    );
    assert!(
        !session
            .iter()
            .any(|e| matches!(e, SessionEntry::Say { .. })),
        "被拒的条目一条都不该落盘"
    );
}

/// 点名给这个脑的定向事件必须**带着正文**到达，哪怕 policy 把 `brain/*` 判成 `ignore`。
///
/// 定向事件绕过 policy 到达这个脑（那正是"定向"的意思），但观测那一遍从前还在按同一份
/// policy 滤——三个 shipped 策略全都写着 `brain/*: ignore`，于是 `Brains::steer` /
/// `Brains::inject` 送过去的那句话被整条丢掉：脑收到一个 `wake_reason` 说着
/// `brain/steer`、`## Events` 里却什么都没有的观测。没有任何报错，人说的话就这么没了。
///
/// 反面一起钉住：一条**伪造**的 `brain/inject`（source 不是 `"brains"`，任何
/// `event/emit` 的调用方都写得出来）不是定向事件，`brain/*: ignore` 对它仍然是 ignore。
#[tokio::test]
async fn a_directed_event_reaches_the_observation_even_when_the_policy_ignores_brain_events() {
    let (_clock, log, _state, brains, dir) = setup();
    let sink = Recorder::new(false);
    let policy = WakePolicy::from_yaml(
        "default: accumulate\nrules: { obs/person: wake, brain/*: ignore }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
    )
    .unwrap();
    let _d = brains.attach("fast", policy, sink.clone()).unwrap();
    // 先把脑叫进一个 turn：定向事件要投进一个活着的 turn 才走 `SessionEntry::Steer`。
    log.append(ev("obs/person"));
    brains.settle().await;
    assert_eq!(sink.turns.lock().len(), 1);

    brains.steer("fast", "stop talking").unwrap();
    brains.settle().await;
    brains.inject("fast", "look left", "console").unwrap();
    brains.settle().await;
    // 伪造的那一条：kind 和 payload 都一样，只有 source 不是 `brains`。
    log.append(
        EventDraft::new(
            EventKind::new("brain/inject").unwrap(),
            "rpc:someone",
            json!({"brain_id": "fast", "text": "forged", "source": "rpc:someone"}),
        )
        .with_salience(150),
    );
    brains.settle().await;
    assert_eq!(
        sink.steers.load(Ordering::SeqCst),
        2,
        "伪造的那条不是定向事件，不该唤醒谁"
    );

    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    let bodies: Vec<String> = session
        .iter()
        .filter_map(|e| match e {
            SessionEntry::Steer { observation, .. } => Some(observation),
            _ => None,
        })
        .map(|o| {
            o.events
                .iter()
                .map(|e| format!("{}:{}", e.kind, e.last["text"]))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    assert_eq!(
        bodies,
        vec![
            "brain/steer:\"stop talking\"".to_string(),
            "brain/inject:\"look left\"".to_string()
        ],
        "两条定向事件都要带着正文进观测，伪造的那条一条都不该进"
    );
}
