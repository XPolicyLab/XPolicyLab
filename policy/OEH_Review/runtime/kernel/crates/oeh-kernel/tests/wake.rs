use oeh_kernel::wake::{BrainCursor, WakeAction, WakePolicy};
use oeh_kernel::BrainStatus;
use oeh_proto::{EventDraft, EventKind};
use serde_json::json;

fn ev(kind: &str, seq: u64) -> oeh_proto::WorldEvent {
    EventDraft::new(EventKind::new(kind).unwrap(), "t", json!({})).into_event(seq, 0)
}

#[test]
fn policy_lookup_precedence_and_yaml() {
    let p = WakePolicy::from_yaml(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../wake/fast.yaml"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        p.action_for(&EventKind::new("obs/person").unwrap()),
        WakeAction::Wake
    );
    assert_eq!(
        p.action_for(&EventKind::new("brain/wake").unwrap()),
        WakeAction::Ignore,
        "prefix wildcard"
    );
    assert_eq!(
        p.action_for(&EventKind::new("obs/detections").unwrap()),
        WakeAction::Accumulate,
        "default"
    );
    assert_eq!(p.budget.max_consecutive_wakes, 3);
    assert!(WakePolicy::from_yaml("default: fly").is_err());
}

#[test]
fn budget_suppresses_wakes_but_never_interrupt() {
    let p = WakePolicy::from_yaml("default: accumulate\nrules: { obs/person: wake, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 2, min_interval_ms: 100 }").unwrap();
    let mut c = BrainCursor::new(0);
    let ms = |m: u64| m * 1_000_000;
    let d1 = c.decide(&p, &ev("obs/person", 1), ms(0), BrainStatus::Idle);
    assert_eq!((d1.action, d1.suppressed), (WakeAction::Wake, false));
    c.note_wake(ms(0));
    let d2 = c.decide(&p, &ev("obs/person", 2), ms(50), BrainStatus::Idle);
    assert_eq!(
        (d2.action, d2.suppressed),
        (WakeAction::Accumulate, true),
        "min_interval not elapsed"
    );
    let d3 = c.decide(&p, &ev("obs/person", 3), ms(150), BrainStatus::Idle);
    assert_eq!(d3.action, WakeAction::Wake);
    c.note_wake(ms(150));
    let d4 = c.decide(&p, &ev("obs/person", 4), ms(400), BrainStatus::Idle);
    assert_eq!(
        (d4.action, d4.suppressed),
        (WakeAction::Accumulate, true),
        "consecutive cap reached"
    );
    let d5 = c.decide(&p, &ev("safety/estop", 5), ms(401), BrainStatus::Idle);
    assert_eq!((d5.action, d5.suppressed), (WakeAction::Interrupt, false));
    c.note_idle(ms(300), &p.budget);
    assert_eq!(
        c.consecutive_wakes, 0,
        "idle after min_interval resets the streak"
    );
    assert_eq!(
        c.decide(&p, &ev("obs/person", 6), ms(400), BrainStatus::Idle)
            .action,
        WakeAction::Wake
    );
    c.advance(6);
    assert_eq!(c.cursor, 6);
}
