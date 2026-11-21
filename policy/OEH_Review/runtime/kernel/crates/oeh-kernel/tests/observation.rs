use oeh_kernel::observation::{build_observation, ObservationBudget, WakeReason};
use oeh_kernel::wake::{WakeAction, WakePolicy};
use oeh_proto::{EventDraft, EventKind, FrameRef, WorldEvent};
use serde_json::json;
use std::collections::BTreeMap;

fn ev(kind: &str, seq: u64, sal: u8, key: Option<&str>, payload: serde_json::Value) -> WorldEvent {
    let mut d = EventDraft::new(EventKind::new(kind).unwrap(), "t", payload)
        .with_salience(sal)
        .with_frame_refs(vec![FrameRef {
            stream_id: "s".into(),
            frame_seq: seq,
            t_kernel: seq,
        }]);
    if let Some(k) = key {
        d = d.with_collapse_key(k);
    }
    d.into_event(seq, seq * 10)
}

fn policy() -> WakePolicy {
    WakePolicy::from_yaml("rules: { brain/*: ignore, obs/person: wake }").unwrap()
}

#[test]
fn collapses_by_key_keeps_first_last_count_and_orders_by_salience() {
    let events = vec![
        ev("obs/detections", 10, 20, None, json!({"n": 1})),
        ev("obs/person", 11, 200, Some("person"), json!({"near": true})),
        ev("brain/cursor", 12, 0, None, json!({})),
        ev("obs/detections", 13, 20, None, json!({"n": 2})),
        ev(
            "obs/person",
            14,
            200,
            Some("person"),
            json!({"near": false}),
        ),
        ev(
            "health/stale",
            15,
            220,
            Some("stale:x"),
            json!({"stale": true}),
        ),
    ];
    let mut state = BTreeMap::new();
    state.insert("fsm".to_string(), json!("STANDING"));
    let obs = build_observation(
        "fast",
        &events,
        state,
        WakeReason {
            kind: "obs/person".into(),
            seq: 14,
            action: WakeAction::Wake,
        },
        999,
        &policy(),
        &ObservationBudget::default(),
    );
    assert_eq!(obs.source_seq_range, (10, 15));
    assert_eq!(obs.t_kernel, 999);
    let keys: Vec<(&str, u64)> = obs
        .events
        .iter()
        .map(|e| (e.collapse_key.as_str(), e.count))
        .collect();
    assert_eq!(
        keys,
        vec![("stale:x", 1), ("person", 2), ("obs/detections", 2)],
        "salience desc; brain/* ignored"
    );
    let person = &obs.events[1];
    assert_eq!((person.first_seq, person.last_seq), (11, 14));
    assert_eq!(
        (&person.first, &person.last),
        (&json!({"near": true}), &json!({"near": false}))
    );
    assert_eq!(person.first_frame_refs[0].frame_seq, 11);
    assert_eq!(person.last_frame_refs[0].frame_seq, 14);
    assert_eq!(obs.state["fsm"], json!("STANDING"));
    assert_eq!(obs.truncated, 0);
    let s = serde_json::to_string(&obs).unwrap();
    assert!(s.contains("\"wake_reason\""));
}

#[test]
fn truncates_by_count_then_bytes_but_keeps_wake_reason_event() {
    let mut events = Vec::new();
    for i in 0..50u64 {
        events.push(ev(
            "obs/detections",
            i + 1,
            10,
            Some(&format!("k{i}")),
            json!({"pad": "x".repeat(200)}),
        ));
    }
    events.push(ev(
        "obs/person",
        100,
        200,
        Some("person"),
        json!({"near": true}),
    ));
    let budget = ObservationBudget {
        max_events: 8,
        max_bytes: 1200,
    };
    let obs = build_observation(
        "fast",
        &events,
        BTreeMap::new(),
        WakeReason {
            kind: "obs/person".into(),
            seq: 100,
            action: WakeAction::Wake,
        },
        0,
        &policy(),
        &budget,
    );
    assert!(obs.events.len() <= 8);
    assert!(
        serde_json::to_vec(&obs).unwrap().len() <= 1200,
        "byte budget"
    );
    assert_eq!(
        obs.events[0].collapse_key, "person",
        "wake-reason event survives truncation"
    );
    assert_eq!(obs.truncated, 51 - obs.events.len());
}

#[test]
fn empty_input_is_well_formed() {
    let obs = build_observation(
        "fast",
        &[],
        BTreeMap::new(),
        WakeReason {
            kind: "brain/ask".into(),
            seq: 0,
            action: WakeAction::Wake,
        },
        1,
        &policy(),
        &ObservationBudget::default(),
    );
    assert_eq!(obs.source_seq_range, (0, 0));
    assert!(obs.events.is_empty());
}

/// I8: 事件砍到只剩唤醒那一条还超字节预算时，撑爆预算的是 `state` 本身。整份换成一个
/// 明确的占位——脑知道“状态被省略了”，总好过收到一份超预算的观测。
#[test]
fn oversized_state_is_elided_wholesale() {
    let events = vec![ev(
        "obs/person",
        1,
        200,
        Some("person"),
        json!({"near": true}),
    )];
    let mut state = BTreeMap::new();
    for i in 0..40 {
        state.insert(format!("k{i}"), json!("v".repeat(100)));
    }
    let obs = build_observation(
        "fast",
        &events,
        state,
        WakeReason {
            kind: "obs/person".into(),
            seq: 1,
            action: WakeAction::Wake,
        },
        0,
        &policy(),
        &ObservationBudget {
            max_events: 32,
            max_bytes: 1024,
        },
    );
    assert_eq!(obs.state["_elided"], json!(true));
    assert_eq!(obs.state["keys"], json!(40));
    assert_eq!(obs.state.len(), 2);
    assert_eq!(obs.events.len(), 1, "the wake reason still survives");
    assert!(
        serde_json::to_vec(&obs).unwrap().len() <= 1024,
        "back inside the byte budget"
    );
}

/// 预算够用时 `state` 原样保留（省略只在真的超预算时发生）。
#[test]
fn state_survives_when_it_fits() {
    let mut state = BTreeMap::new();
    state.insert("fsm".to_string(), json!("STANDING"));
    let obs = build_observation(
        "fast",
        &[],
        state,
        WakeReason {
            kind: "obs/person".into(),
            seq: 1,
            action: WakeAction::Wake,
        },
        0,
        &policy(),
        &ObservationBudget::default(),
    );
    assert_eq!(obs.state["fsm"], json!("STANDING"));
    assert!(!obs.state.contains_key("_elided"));
}
