use std::sync::Arc;

use oeh_kernel::fsm::{FsmError, FsmState, RobotFsm};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{ManualClock, WorldLog};
use serde_json::json;

fn fsm() -> (RobotFsm, Arc<WorldLog>, Arc<StateRegistry>) {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock));
    (RobotFsm::new(log.clone(), state.clone()), log, state)
}

#[test]
fn legal_path_emits_events_and_state() {
    let (f, log, state) = fsm();
    assert_eq!(f.state(), FsmState::Boot);
    f.transition(FsmState::Standing, "policy active", "test")
        .unwrap();
    f.transition(FsmState::Engaged, "user engaged", "test")
        .unwrap();
    f.transition(FsmState::Acting, "walk", "test").unwrap();
    assert!(f.allows_motion());
    let ev = log.since(0).unwrap();
    assert_eq!(ev.len(), 3);
    assert_eq!(ev[2].kind.as_str(), "safety/fsm");
    assert_eq!(
        ev[2].payload,
        json!({"from": "ENGAGED", "to": "ACTING", "reason": "walk"})
    );
    assert_eq!(ev[2].collapse_key.as_deref(), Some("fsm"));
    assert_eq!(state.get("fsm"), Some(json!("ACTING")));
}

#[test]
fn illegal_and_noop_transitions() {
    let (f, log, _) = fsm();
    assert!(matches!(
        f.transition(FsmState::Acting, "x", "t"),
        Err(FsmError::Illegal {
            from: FsmState::Boot,
            to: FsmState::Acting
        })
    ));
    f.transition(FsmState::Boot, "same", "t").unwrap();
    assert_eq!(log.next_seq(), 1, "no-op emits nothing");
    f.transition(FsmState::EmergencyStop, "tipping", "t")
        .unwrap();
    assert!(!f.allows_motion());
    assert!(
        f.transition(FsmState::Standing, "x", "t").is_err(),
        "must go through RECOVERING"
    );
    f.transition(FsmState::Recovering, "manual", "t").unwrap();
    f.transition(FsmState::Standing, "ok", "t").unwrap();
    assert_eq!(
        serde_json::to_string(&FsmState::EmergencyStop).unwrap(),
        "\"EMERGENCY_STOP\""
    );
}

/// C2: `transition` 对并发调用者必须是一个原子步骤。状态翻转之后、写寄存器与 append 之前
/// 放开状态锁，另一个线程就能挤进来先把它那条 `safety/fsm` 写出去——事件链断裂（某条的
/// `from` 接不上上一条的 `to`），`state()` 与 `fsm` 寄存器也会各说各话。
#[test]
fn concurrent_transitions_keep_the_event_chain_contiguous() {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()).with_retain(8192));
    let state = Arc::new(StateRegistry::new(clock));
    let f = Arc::new(RobotFsm::new(log.clone(), state.clone()));
    f.transition(FsmState::Standing, "init", "test").unwrap();
    let mut threads = Vec::new();
    for _ in 0..2 {
        let f = f.clone();
        threads.push(std::thread::spawn(move || {
            // STANDING → EMERGENCY_STOP → RECOVERING → STANDING 全是合法边；非法/无操作
            // 的那些自己就返回了，不写事件。
            for i in 0..200u32 {
                let to = match i % 3 {
                    0 => FsmState::EmergencyStop,
                    1 => FsmState::Recovering,
                    _ => FsmState::Standing,
                };
                let _ = f.transition(to, "race", "test");
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    let evs: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "safety/fsm")
        .collect();
    assert!(
        evs.len() > 10,
        "transitions actually happened: {}",
        evs.len()
    );
    for w in evs.windows(2) {
        assert_eq!(
            w[0].payload["to"], w[1].payload["from"],
            "chain broken between seq {} and {}: {:?} then {:?}",
            w[0].seq, w[1].seq, w[0].payload, w[1].payload
        );
    }
    let last = evs.last().unwrap();
    assert_eq!(last.payload["to"], json!(f.state()), "event vs state()");
    assert_eq!(
        state.get("fsm"),
        Some(json!(f.state())),
        "register vs state()"
    );
}
