use std::sync::Arc;

use oeh_kernel::brains::{BrainSink, BrainStatus, Brains, BrainsConfig, TurnHandle};
use oeh_kernel::clock::ManualClock;
use oeh_kernel::observation::Observation;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::wake::{WakeAction, WakePolicy};
use oeh_kernel::world_log::WorldLog;
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;

/// `turns` 存 `(Observation, TurnHandle)`：光有 handle 看不出这个 turn 是被什么唤醒的
/// （idle 时的定向 steer/interrupt 走的是 `Plan::StartTurn`，不是 `sink.steer`/
/// `sink.interrupt`），也看不出它的观测范围盖没盖住打断前积累的事件。
#[derive(Default)]
struct Rec {
    steers: Mutex<Vec<String>>,
    interrupts: Mutex<u32>,
    turns: Mutex<Vec<(Observation, TurnHandle)>>,
}
impl BrainSink for Rec {
    fn on_turn(&self, obs: Observation, turn: TurnHandle) {
        self.turns.lock().push((obs, turn));
    }
    fn steer(&self, obs: Observation) {
        self.steers.lock().push(obs.wake_reason.kind.clone());
    }
    fn interrupt(&self) {
        *self.interrupts.lock() += 1;
    }
}

fn brains(dir: &std::path::Path) -> (Arc<Brains>, Arc<WorldLog>) {
    let clock = Arc::new(ManualClock::new(1_000));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let b = Brains::new(
        clock,
        log.clone(),
        state,
        dir.to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    (Arc::new(b), log)
}

fn ev(kind: &str, payload: serde_json::Value) -> EventDraft {
    EventDraft::new(EventKind::new(kind).unwrap(), "t", payload)
}

/// A brain that ignores everything by default, plus one rule that wakes it on `obs/kick`
/// so a test can get it into `Running` (or `Maintenance`, via `TurnHandle::enter_maintenance`)
/// without relying on the directed API itself.
fn ignore_but_wake_on_kick() -> WakePolicy {
    let mut p = WakePolicy {
        default: WakeAction::Ignore,
        ..Default::default()
    };
    p.rules.insert("obs/kick".into(), WakeAction::Wake);
    p
}

/// Controller ruling (fix round 1, C1+I1): a directed `brain/steer`/`brain/interrupt` is
/// only a classifier that bypasses `BrainCursor::decide` (policy/budget/suppression); the
/// resulting `WakeAction` then goes through the *existing* `(action, status)` matrix and
/// the *existing* `Plan::StartTurn`/`Steer`/`Interrupt` executors, unchanged. At `Idle`
/// that means steer/interrupt are "wake" (spec §4.5): they open a new turn via
/// `sink.on_turn`, they never call `sink.steer`/`sink.interrupt` directly.
#[tokio::test]
async fn steer_and_interrupt_reach_only_the_named_brain() {
    let dir = tempfile::tempdir().unwrap();
    let (b, log) = brains(dir.path());
    let a = Arc::new(Rec::default());
    let c = Arc::new(Rec::default());
    let policy = WakePolicy {
        default: WakeAction::Ignore,
        ..Default::default()
    };
    let _da = b.attach("fast", policy.clone(), a.clone()).unwrap();
    let _dc = b.attach("slow", policy, c.clone()).unwrap();

    // `fast` is freshly attached (Idle): a directed steer goes through the existing
    // (Steer, Idle) => Plan::StartTurn arm, same as spec §4.5's "idle steer == wake" —
    // it never calls sink.steer, it opens a new turn whose wake_reason is this event.
    b.steer("fast", "look left").unwrap();
    // The directed interrupt right behind it finds `fast` already Running (the turn the
    // steer just opened): (Interrupt, Running) => Plan::Interrupt — a real cancel-and-
    // rewake, not a no-op notification.
    b.interrupt("fast").unwrap();
    b.settle().await;

    assert!(
        a.steers.lock().is_empty(),
        "idle steer starts a turn, it never calls sink.steer"
    );
    {
        let turns = a.turns.lock();
        assert_eq!(
            turns.len(),
            2,
            "the steer's start_turn, then the interrupt's re-wake start_turn"
        );
        assert_eq!(turns[0].0.wake_reason.kind, "brain/steer");
        assert_eq!(turns[1].0.wake_reason.kind, "brain/interrupt");
        assert!(
            !turns[0].1.is_current(),
            "the interrupted turn's handle is stale"
        );
    }
    assert_eq!(*a.interrupts.lock(), 1);
    assert!(c.steers.lock().is_empty());
    assert!(
        c.turns.lock().is_empty(),
        "a directed event never reaches a brain it doesn't name"
    );
    assert_eq!(*c.interrupts.lock(), 0);

    let kinds: Vec<String> = log
        .since(0)
        .unwrap()
        .iter()
        .map(|e| e.kind.as_str().to_string())
        .collect();
    assert!(
        kinds.contains(&"brain/steer".to_string())
            && kinds.contains(&"brain/interrupt".to_string())
    );
    assert!(matches!(
        b.steer("nobody", "x"),
        Err(oeh_kernel::brains::BrainsError::NotAttached(_))
    ));
    assert!(matches!(
        b.interrupt("nobody"),
        Err(oeh_kernel::brains::BrainsError::NotAttached(_))
    ));
}

/// Controller ruling (fix round 1, I2): only events actually appended by
/// `Brains::steer`/`interrupt` (source `"brains"`) are directed. A `brain/steer` from any
/// other source is not a directed event at all — and what happens to it then depends on
/// *who is reading it*, which is what this test pins down (Task 4, parked finding (b): the
/// first version gave both brains `default: ignore`, so "nothing happened" was the expected
/// outcome for every brain and the test would have passed just as happily if the forged
/// event had never been routed at all):
///
/// - the brain the payload **names** never sees it: `route_one`'s anti-self-excite guard
///   drops any non-directed `brain/*` whose `payload.brain_id` is this brain, *before* the
///   policy is consulted — so even `brain/steer: wake` in its own policy does not wake it;
/// - any **other** brain falls back to its ordinary `WakePolicy`, where `brain/steer` is
///   just another `brain/*` event — with `brain/steer: wake` it wakes, exactly like before
///   the directed-steer feature existed.
///
/// Both brains run the *same* policy here, so the asymmetry can only come from the guard.
#[tokio::test]
async fn only_events_from_the_brains_source_are_treated_as_directed() {
    let dir = tempfile::tempdir().unwrap();
    let (b, log) = brains(dir.path());
    let named = Arc::new(Rec::default());
    let other = Arc::new(Rec::default());
    let mut policy = WakePolicy {
        default: WakeAction::Ignore,
        ..Default::default()
    };
    policy.rules.insert("brain/steer".into(), WakeAction::Wake);
    let _dn = b.attach("fast", policy.clone(), named.clone()).unwrap();
    let _do = b.attach("slow", policy, other.clone()).unwrap();

    // Forge an event that looks exactly like a directed steer (same kind, same payload
    // shape) but from a source other than "brains".
    log.append(EventDraft::new(
        EventKind::new("brain/steer").unwrap(),
        "someone-else",
        serde_json::json!({"brain_id": "fast", "text": "not really directed"}),
    ));
    b.settle().await;

    assert!(
        named.steers.lock().is_empty() && named.turns.lock().is_empty(),
        "a forged steer never reaches the brain it names, not even through that brain\'s \
         own `brain/steer: wake` rule"
    );
    assert_eq!(*named.interrupts.lock(), 0);
    assert!(
        other.steers.lock().is_empty(),
        "and it is not directed at the other brain either — it is an ordinary wake there"
    );
    assert_eq!(
        other.turns.lock().len(),
        1,
        "the event *was* routed: a brain the payload does not name judges it by its own \
         WakePolicy, where `brain/steer: wake` wakes it"
    );
    assert_eq!(other.turns.lock()[0].0.wake_reason.kind, "brain/steer");
    assert_eq!(*other.interrupts.lock(), 0);
}

/// Controller ruling (fix round 1, I3): a directed steer while the brain is already
/// `Running` must reach the live turn via `sink.steer`, and the `TurnHandle` the brain
/// already holds must remain current afterwards (no turn is started or ended).
#[tokio::test]
async fn directed_steer_while_running_reaches_the_live_turn() {
    let dir = tempfile::tempdir().unwrap();
    let (b, log) = brains(dir.path());
    let a = Arc::new(Rec::default());
    let _d = b
        .attach("fast", ignore_but_wake_on_kick(), a.clone())
        .unwrap();

    log.append(ev("obs/kick", serde_json::json!({})));
    b.settle().await;
    assert_eq!(a.turns.lock().len(), 1, "obs/kick woke fast into Running");

    b.steer("fast", "look left").unwrap();
    b.settle().await;

    assert_eq!(a.steers.lock().as_slice(), ["brain/steer"]);
    assert_eq!(
        a.turns.lock().len(),
        1,
        "a running steer does not start a new turn"
    );
    assert!(
        a.turns.lock()[0].1.is_current(),
        "the live turn's handle is still current after a steer"
    );
}

/// Controller ruling (fix round 1, C1+I3): a directed interrupt while `Running` must
/// actually cancel the current turn (the old handle stops being current) and re-wake with
/// a fresh one (a second `on_turn`), and — since it goes through the existing
/// `Plan::Interrupt` → `finish_turn` → `start_turn` path instead of a manual cursor jump —
/// events accumulated in the inbox before the interrupt must survive into the new turn's
/// observation, not be silently discarded.
#[tokio::test]
async fn directed_interrupt_while_running_cancels_re_wakes_and_preserves_the_inbox() {
    let dir = tempfile::tempdir().unwrap();
    let (b, log) = brains(dir.path());
    let a = Arc::new(Rec::default());
    let mut policy = ignore_but_wake_on_kick();
    policy
        .rules
        .insert("obs/context".into(), WakeAction::Accumulate);
    let _d = b.attach("fast", policy, a.clone()).unwrap();

    log.append(ev("obs/kick", serde_json::json!({})));
    b.settle().await;
    assert_eq!(a.turns.lock().len(), 1);

    // Two accumulated events land in the world log before the directed interrupt.
    let c1 = log.append(ev("obs/context", serde_json::json!({"n": 1})));
    let c2 = log.append(ev("obs/context", serde_json::json!({"n": 2})));
    b.interrupt("fast").unwrap();
    b.settle().await;

    assert_eq!(*a.interrupts.lock(), 1);
    let turns = a.turns.lock();
    assert_eq!(
        turns.len(),
        2,
        "interrupt cancels the turn and re-wakes with a fresh one"
    );
    assert!(
        !turns[0].1.is_current(),
        "the interrupted turn's handle is stale"
    );
    let (obs2, _) = &turns[1];
    assert!(
        obs2.source_seq_range.0 <= c1.seq && c2.seq <= obs2.source_seq_range.1,
        "the new turn's observation covers the events accumulated before the interrupt: {:?}",
        obs2.source_seq_range
    );
}

/// Controller ruling (fix round 1, I3): a directed interrupt during `Maintenance` behaves
/// like `Running` (it is not treated as idle) — the maintenance turn is torn down and a
/// fresh turn starts in `Running`.
#[tokio::test]
async fn directed_interrupt_during_maintenance_behaves_like_running() {
    let dir = tempfile::tempdir().unwrap();
    let (b, log) = brains(dir.path());
    let a = Arc::new(Rec::default());
    let _d = b
        .attach("fast", ignore_but_wake_on_kick(), a.clone())
        .unwrap();

    log.append(ev("obs/kick", serde_json::json!({})));
    b.settle().await;
    assert_eq!(a.turns.lock().len(), 1);
    {
        let turns = a.turns.lock();
        turns[0].1.enter_maintenance();
    }
    assert_eq!(b.status("fast"), Some(BrainStatus::Maintenance));

    b.interrupt("fast").unwrap();
    b.settle().await;

    assert_eq!(*a.interrupts.lock(), 1);
    assert_eq!(
        a.turns.lock().len(),
        2,
        "a maintenance turn is torn down and a fresh one starts, same as running"
    );
    assert_eq!(
        b.status("fast"),
        Some(BrainStatus::Running),
        "the re-wake turn starts in Running, not Maintenance"
    );
}

#[tokio::test]
async fn update_policy_changes_routing_for_later_events() {
    let dir = tempfile::tempdir().unwrap();
    let (b, log) = brains(dir.path());
    let a = Arc::new(Rec::default());
    let _d = b
        .attach(
            "fast",
            WakePolicy {
                default: WakeAction::Ignore,
                ..Default::default()
            },
            a.clone(),
        )
        .unwrap();
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        serde_json::json!({}),
    ));
    b.settle().await;
    assert!(a.turns.lock().is_empty());

    let mut p = WakePolicy::default();
    p.rules.insert("obs/person".into(), WakeAction::Wake);
    b.update_policy("fast", p).unwrap();
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        serde_json::json!({}),
    ));
    b.settle().await;
    assert_eq!(a.turns.lock().len(), 1);
}
