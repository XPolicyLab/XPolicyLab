use oeh_kernel::brains::{BrainSink, Brains, BrainsConfig, TurnHandle};
use oeh_kernel::observation::Observation;
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, Proposer};
use oeh_kernel::session_log::{read_session, SessionEntry};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::{ManualClock, WorldLog};
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde_json::json;
use std::sync::Arc;

struct Echo {
    seen: Mutex<Vec<Proposal>>,
}
impl Proposer for Echo {
    fn propose(&self, p: Proposal) -> BoxFuture<Outcome> {
        self.seen.lock().push(p.clone());
        Box::pin(async move {
            if p.capability == "walk" {
                Outcome::Executed {
                    result: json!({"walked": true}),
                }
            } else {
                Outcome::Refused {
                    reason: "unknown".into(),
                }
            }
        })
    }
}
struct Holder {
    handle: Mutex<Option<TurnHandle>>,
}
impl BrainSink for Holder {
    fn on_turn(&self, _o: Observation, t: TurnHandle) {
        *self.handle.lock() = Some(t);
    }
    fn steer(&self, _o: Observation) {}
    fn interrupt(&self) {}
}

#[tokio::test]
async fn propose_routes_through_proposer_and_records_session() {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock,
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    let echo = Arc::new(Echo {
        seen: Mutex::new(vec![]),
    });
    let _p = brains.set_proposer(echo.clone());
    let holder = Arc::new(Holder {
        handle: Mutex::new(None),
    });
    let _d = brains
        .attach(
            "fast",
            WakePolicy::from_yaml(
                "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
            )
            .unwrap(),
            holder.clone(),
        )
        .unwrap();
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    let h = holder.handle.lock().take().unwrap();
    let out = h.propose("walk", json!({"vx": 0.2})).await;
    assert_eq!(
        out,
        Outcome::Executed {
            result: json!({"walked": true})
        }
    );
    let out2 = h.propose("fly", json!({})).await;
    assert!(matches!(out2, Outcome::Refused { .. }));
    // 块作用域而不是 `drop(seen)`：`clippy::await_holding_lock` 只认前者。
    {
        let seen = echo.seen.lock();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].trace_id, h.trace_id);
        assert_eq!(seen[0].proposal_id.0, format!("{}-p1", h.trace_id));
        assert_eq!(seen[1].proposal_id.0, format!("{}-p2", h.trace_id));
    }
    let trace = h.trace_id.clone();
    h.done("completed");
    brains.settle().await;
    // 会话日志的写盘在写线程上；读文件之前先排空。
    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    assert!(session.iter().any(
        |e| matches!(e, SessionEntry::Proposal { capability, trace_id, .. } if capability == "walk" && *trace_id == trace)
    ));
    assert!(session.iter().any(|e| matches!(
        e,
        SessionEntry::Outcome {
            outcome: Outcome::Executed { .. },
            ..
        }
    )));
}

#[tokio::test]
async fn propose_without_proposer_or_stale_turn_is_refused() {
    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock,
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    let holder = Arc::new(Holder {
        handle: Mutex::new(None),
    });
    let _d = brains
        .attach(
            "fast",
            WakePolicy::from_yaml(
                "rules: { obs/person: wake, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
            )
            .unwrap(),
            holder.clone(),
        )
        .unwrap();
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    let h = holder.handle.lock().take().unwrap();
    assert!(
        matches!(h.propose("walk", json!({})).await, Outcome::Refused { reason } if reason.contains("no admission"))
    );
    log.append(EventDraft::new(
        EventKind::new("safety/estop").unwrap(),
        "t",
        json!({"engaged": true}),
    ));
    brains.settle().await;
    assert!(!h.is_current());
    assert!(
        matches!(h.propose("walk", json!({})).await, Outcome::Refused { reason } if reason.contains("not current"))
    );
}

/// 提案在途时 turn 被打断：结局仍然要落进会话日志，带着它自己那个（已经过期的）turn 号。
/// 会话日志里的 `Outcome` 自带 `turn` 和 `proposal_id`，不会和新 turn 的条目混淆；真要是
/// 按“当前 turn”去卡这条写入，日志里就会留下一条永远没有下文的 `Proposal`——读的人无从
/// 知道那次提案到底做没做、结果是什么。
#[tokio::test]
async fn outcome_of_an_interrupted_turn_still_reaches_the_session_log() {
    struct Slow {
        rx: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    }
    impl Proposer for Slow {
        fn propose(&self, _p: Proposal) -> BoxFuture<Outcome> {
            let rx = self.rx.lock().take();
            Box::pin(async move {
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                Outcome::Executed {
                    result: json!({"late": true}),
                }
            })
        }
    }

    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock,
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _p = brains.set_proposer(Arc::new(Slow {
        rx: Mutex::new(Some(rx)),
    }));
    let holder = Arc::new(Holder {
        handle: Mutex::new(None),
    });
    let _d = brains
        .attach(
            "fast",
            WakePolicy::from_yaml(
                "rules: { obs/person: wake, safety/estop: interrupt }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
            )
            .unwrap(),
            holder.clone(),
        )
        .unwrap();
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    let h = holder.handle.lock().take().unwrap();
    assert_eq!(h.turn, 1);
    // 提案发出去了，但准入服务还没给结局。
    let pending = tokio::spawn(h.propose("walk", json!({"vx": 0.1})));
    // turn 1 在这里被打断，brains 开出 turn 2。
    log.append(EventDraft::new(
        EventKind::new("safety/estop").unwrap(),
        "t",
        json!({"engaged": true}),
    ));
    brains.settle().await;
    assert!(!h.is_current(), "turn 1 should have been interrupted");
    assert_eq!(holder.handle.lock().as_ref().map(|t| t.turn), Some(2));
    tx.send(()).unwrap();
    assert!(matches!(pending.await.unwrap(), Outcome::Executed { .. }));
    // 会话日志的写盘在写线程上；读文件之前先排空。
    brains.flush_sessions();
    let session = read_session(&dir.path().join("fast.session.jsonl")).unwrap();
    assert!(
        session.iter().any(|e| matches!(
            e,
            SessionEntry::Outcome {
                turn: 1,
                proposal_id,
                outcome: Outcome::Executed { .. },
            } if proposal_id.ends_with("-p1")
        )),
        "{session:#?}"
    );
}

/// 脑把 `propose` 返回的 future 丢掉——Plan 3 的 RPC 客户端断线、Plan 4 的脑任务被
/// abort 都是这个形状。结局仍然要落进会话日志：否则日志里留下一条永远没有下文的
/// `Proposal`，读的人无从知道那次提案到底做没做、做成了什么。
#[tokio::test]
async fn a_dropped_propose_future_still_records_its_outcome() {
    struct Slow {
        rx: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    }
    impl Proposer for Slow {
        fn propose(&self, _p: Proposal) -> BoxFuture<Outcome> {
            let rx = self.rx.lock().take();
            Box::pin(async move {
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                Outcome::Executed {
                    result: json!({"late": true}),
                }
            })
        }
    }

    let clock = Arc::new(ManualClock::new(0));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let dir = tempfile::tempdir().unwrap();
    let brains = Brains::new(
        clock,
        log.clone(),
        state,
        dir.path().to_path_buf(),
        BrainsConfig::default(),
    )
    .unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _p = brains.set_proposer(Arc::new(Slow {
        rx: Mutex::new(Some(rx)),
    }));
    let holder = Arc::new(Holder {
        handle: Mutex::new(None),
    });
    let _d = brains
        .attach(
            "fast",
            WakePolicy::from_yaml(
                "rules: { obs/person: wake }\nbudget: { max_consecutive_wakes: 9, min_interval_ms: 0 }",
            )
            .unwrap(),
            holder.clone(),
        )
        .unwrap();
    log.append(EventDraft::new(
        EventKind::new("obs/person").unwrap(),
        "t",
        json!({}),
    ));
    brains.settle().await;
    let h = holder.handle.lock().take().unwrap();

    // 脑放弃了这次等待。准入服务那边照样在跑。
    drop(h.propose("walk", json!({"vx": 0.1})));
    tx.send(()).unwrap();

    let path = dir.path().join("fast.session.jsonl");
    for _ in 0..200 {
        let session = read_session(&path).unwrap_or_default();
        if session.iter().any(|e| {
            matches!(e, SessionEntry::Outcome { outcome: Outcome::Executed { .. }, proposal_id, .. }
                if proposal_id.ends_with("-p1"))
        }) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!(
        "被丢掉的提案没有把结局写进会话日志: {:#?}",
        read_session(&path).unwrap_or_default()
    );
}
