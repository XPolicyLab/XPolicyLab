//! 准入服务的端到端行为：词汇 → 规则 → 审批 → 执行（租约、抢占、FSM 驱动）。
//!
//! 这些测试用 `Kernel::boot` 把真实的插件图拼起来（clock / world-log / state / fsm /
//! brains / admission / mock-executor），然后直接调用 `Admission::propose`——不经过
//! brains 的 turn 循环，这样断言的是准入流水线本身，而不是唤醒策略。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oeh_admission as _;
use oeh_admission::admission::{Admission, AdmissionConfig};
use oeh_admission::approval::{Approval, AskRequest, AutoApproval, ChannelApproval, Decision};
use oeh_admission::capability::{CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor};
use oeh_admission::rules::scene::SceneRule;
use oeh_admission::verdict::{Rule, RuleCtx, Verdict};
use oeh_kernel::clock::{Clock, MonotonicClock};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Kernel, ManualClock, WorldLog};
// `distill-estop` 的工厂在 `oeh-distill` 里，是 `inventory` 注册的：不引用这个 crate，
// 它就不会被链进这个测试二进制，那一行会解析不出来。
use oeh_distill as _;
use serde_json::{json, Value};

fn walk_spec() -> Value {
    json!({
        "name": "walk",
        "class": "motion",
        "risk_level": "medium",
        "params_schema": {"type": "object", "properties": {"vx": {"type": "number"}, "duration_s": {"type": "number"}}},
        "lease_ttl_s": 0.3
    })
}

fn rows(extra: Vec<EntryRow>) -> Vec<EntryRow> {
    let mut r = vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains"),
        EntryRow::new("admission")
            .with_config(json!({"ask_timeout_ms": 200, "lease_check_ms": 20})),
        EntryRow::new("mock-executor").with_config(
            json!({"capabilities": [walk_spec()], "duration_s": 0.1, "heartbeat_ms": 20}),
        ),
    ];
    r.extend(extra);
    r
}

fn proposal(cap: &str, params: Value, n: u32) -> Proposal {
    Proposal {
        proposal_id: ProposalId(format!("t-p{n}")),
        brain_id: "fast".into(),
        turn: 1,
        trace_id: "t".into(),
        capability: cap.into(),
        params,
    }
}

fn kinds(log: &WorldLog, trace: &str) -> Vec<String> {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace))
        .map(|e| e.kind.as_str().to_string())
        .collect()
}

/// 世界日志里每一条 `safety/fsm` 的 `to`，按顺序。
fn fsm_tos(log: &WorldLog) -> Vec<String> {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() == "safety/fsm")
        .map(|e| e.payload["to"].as_str().unwrap().to_string())
        .collect()
}

/// 轮询等一个条件成立（最多 1s）。比固定 sleep 稳：等的是状态，不是时间。
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..200 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

fn motion_spec(name: &str, schema: Value) -> CapabilitySpec {
    serde_json::from_value(json!({
        "name": name, "class": "motion", "risk_level": "low",
        "params_schema": schema, "lease_ttl_s": 60.0
    }))
    .unwrap()
}

/// 不心跳、也永远不返回的执行器：失联的那一种。`stop()` 把内核回头叫停它的那一下记下来。
#[derive(Default)]
struct Forever {
    stopped: Mutex<Vec<ProposalId>>,
}
impl Executor for Forever {
    fn execute(&self, _c: ExecutionCall, _x: ExecCtx) -> BoxFuture<ExecOutcome> {
        Box::pin(std::future::pending())
    }
    fn stop(&self, proposal_id: &ProposalId) -> BoxFuture<()> {
        self.stopped.lock().unwrap().push(proposal_id.clone());
        Box::pin(async {})
    }
}

/// 一直跑到被撤销为止的执行器，顺手记下"我确实看到了取消"。
struct UntilCancelled {
    saw_cancel: Arc<AtomicBool>,
}
impl Executor for UntilCancelled {
    fn execute(&self, _c: ExecutionCall, mut ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let saw = self.saw_cancel.clone();
        Box::pin(async move {
            ctx.heartbeat();
            ctx.cancelled().await;
            saw.store(true, Ordering::SeqCst);
            ExecOutcome {
                ok: true,
                result: json!({}),
                error: None,
            }
        })
    }
}

/// 一条把判定收紧成 `Ask` 的规则：每次提案都得去走审批。
struct AskAlways;
impl Rule for AskAlways {
    fn name(&self) -> &str {
        "ask-always"
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
        Verdict::Ask {
            reason: "a human should look".into(),
        }
    }
}

struct AllowAll;
impl Rule for AllowAll {
    fn name(&self) -> &str {
        "allow-all"
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
        i
    }
}

#[tokio::test]
async fn empty_rule_set_asks_and_no_approval_denies() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let out = adm.propose(proposal("walk", json!({"vx": 0.1}), 1)).await;
    // 两个理由都要还给脑：链为什么要问人，以及人（这里是内核自己）为什么说不行。
    // 日志里两条分开记着（`act/ask` / `act/decision`），脑只拿得到这一个字符串。
    assert_eq!(
        out,
        Outcome::Refused {
            reason: "no rules registered; no approval provider".into()
        }
    );
    assert_eq!(
        kinds(&log, "t"),
        vec!["act/proposed", "act/ask", "act/decision", "act/refused"]
    );
    let out = adm.propose(proposal("fly", json!({}), 2)).await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("unknown capability")),
        "{out:?}"
    );
}

#[tokio::test]
async fn allow_executes_drives_fsm_and_records_result() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let out = adm.propose(proposal("walk", json!({"vx": 0.1}), 1)).await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    let ks = kinds(&log, "t");
    assert_eq!(
        ks,
        vec!["act/proposed", "act/admitted", "act/started", "act/result"]
    );
    let fsm_events: Vec<String> = log
        .since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() == "safety/fsm")
        .map(|e| e.payload["to"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        fsm_events,
        vec!["STANDING", "ENGAGED", "ACTING", "STANDING"]
    );
    assert_eq!(fsm.state(), FsmState::Standing);
}

#[tokio::test]
async fn ask_is_answered_by_channel_and_motion_is_preempted() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let (chan, mut rx) = ChannelApproval::new();
    let _a = adm.set_approval(chan);
    let answer = tokio::spawn(async move {
        let (_r, tx) = rx.recv().await.unwrap();
        tx.send(Decision::Allow {
            by: "op".into(),
            reason: "ok".into(),
        })
        .unwrap();
    });
    let adm2 = adm.clone();
    let first = tokio::spawn(async move {
        adm2.propose(proposal("walk", json!({"vx": 0.1, "duration_s": 5.0}), 1))
            .await
    });
    answer.await.unwrap();
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(adm.active_motion().is_some());
    // 第二个提案直接 Allow（规则链非空且无缺失的必需规则 → 初始 Allow）
    let _r = adm.register_rule(Arc::new(AllowAll));
    let second = adm
        .propose(proposal("walk", json!({"vx": 0.2, "duration_s": 0.05}), 2))
        .await;
    assert!(matches!(second, Outcome::Executed { .. }), "{second:?}");
    assert_eq!(first.await.unwrap(), Outcome::Preempted);
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/preempted" && e.payload["by"] == "t-p2"));
    assert_eq!(fsm.state(), FsmState::Standing);
}

#[tokio::test]
async fn lease_expiry_cancels_and_fsm_preemption_cancels() {
    // 执行器不 heartbeat：用 duration 长 + heartbeat_ms 大于 ttl
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor");
    r.push(EntryRow::new("mock-executor").with_config(
        json!({"capabilities": [walk_spec()], "duration_s": 5.0, "heartbeat_ms": 2000}),
    ));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let out = adm.propose(proposal("walk", json!({"vx": 0.1}), 1)).await;
    assert!(
        matches!(out, Outcome::Cancelled { ref reason } if reason.contains("lease")),
        "{out:?}"
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/lease_expired"));
    assert_eq!(fsm.state(), FsmState::Standing);

    // FSM 被蒸馏器抢占：运行中 → EmergencyStop → motion 取消
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor");
    r.push(EntryRow::new("mock-executor").with_config(
        json!({"capabilities": [walk_spec()], "duration_s": 5.0, "heartbeat_ms": 20}),
    ));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let adm2 = adm.clone();
    let running =
        tokio::spawn(async move { adm2.propose(proposal("walk", json!({"vx": 0.1}), 3)).await });
    tokio::time::sleep(Duration::from_millis(60)).await;
    fsm.transition(FsmState::EmergencyStop, "tipping", "test")
        .unwrap();
    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "fsm".into()
        }
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/cancelled" && e.payload["reason"] == "fsm"));
    // recover 是急停唯一的出路：allow-all 规则集下它直接 Allow → 执行 →
    // EMERGENCY_STOP → RECOVERING → STANDING。
    let _a = adm.set_approval(Arc::new(AutoApproval { allow: true }));
    let out = adm.propose(proposal("recover", json!({}), 5)).await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    assert_eq!(fsm.state(), FsmState::Standing);
}

#[tokio::test]
async fn tamper_is_denied_and_recorded() {
    struct Loosen;
    impl Rule for Loosen {
        fn name(&self) -> &str {
            "loosen"
        }
        fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
            Verdict::Allow
        }
    }
    struct AskFirst;
    impl Rule for AskFirst {
        fn name(&self) -> &str {
            "ask"
        }
        fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
            Verdict::Ask { reason: "x".into() }
        }
    }
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let _r1 = adm.register_rule(Arc::new(AskFirst));
    let _r2 = adm.register_rule(Arc::new(Loosen));
    let out = adm.propose(proposal("walk", json!({"vx": 0.1}), 1)).await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "safety/tamper_attempt" && e.payload["rule"] == "loosen"));
}

/// 内置能力：`say`（speech）不碰 FSM、不抢 motion；`stop`（control）撤销正在跑的 motion。
#[tokio::test]
async fn speech_leaves_the_motion_alone_and_stop_cancels_it() {
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor");
    r.push(EntryRow::new("mock-executor").with_config(
        json!({"capabilities": [walk_spec()], "duration_s": 5.0, "heartbeat_ms": 20}),
    ));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));

    // speech 类：执行完不改变 FSM，也不登记成 motion。
    let out = adm.propose(proposal("say", json!({"text": "hi"}), 1)).await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    assert_eq!(fsm.state(), FsmState::Standing);
    assert!(adm.active_motion().is_none());

    let adm2 = adm.clone();
    let running =
        tokio::spawn(async move { adm2.propose(proposal("walk", json!({"vx": 0.1}), 2)).await });
    tokio::time::sleep(Duration::from_millis(60)).await;
    let pid = adm.active_motion().unwrap();
    assert_eq!(pid.0, "t-p2");
    // 说话不抢运动。
    let said = adm
        .propose(proposal("say", json!({"text": "on my way"}), 3))
        .await;
    assert!(matches!(said, Outcome::Executed { .. }), "{said:?}");
    assert_eq!(adm.active_motion(), Some(pid));
    assert_eq!(fsm.state(), FsmState::Acting);
    // stop 才抢。
    let stopped = adm.propose(proposal("stop", json!({}), 4)).await;
    assert!(matches!(stopped, Outcome::Executed { .. }), "{stopped:?}");
    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "stop".into()
        }
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/cancelled"
            && e.payload["reason"] == "stop"
            && e.payload["proposal_id"] == "t-p2"));
    assert_eq!(fsm.state(), FsmState::Standing);
}

/// 执行器：每次调用领一张"号码牌"，挂在 oneshot 上直到测试放行。用来把"被抢占者什么
/// 时候收尾"攥在手里。
struct Held {
    gates: Mutex<VecDeque<tokio::sync::oneshot::Receiver<()>>>,
}
impl Executor for Held {
    fn execute(&self, _call: ExecutionCall, ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let rx = self.gates.lock().unwrap().pop_front();
        Box::pin(async move {
            ctx.heartbeat();
            if let Some(rx) = rx {
                let _ = rx.await;
            }
            ExecOutcome {
                ok: true,
                result: json!({}),
                error: None,
            }
        })
    }
}

/// 抢占是一次**交接**：被抢占者的收尾不许再把 FSM 交还成 STANDING。
///
/// 这个测试把交接窗口**按住**来验：接班者 B 在 gate 里认领了 A（`active_motion` 已经
/// 让出、B 还没登记）之后会记一条 `act/preempted`——世界日志的观察者是同步回调，所以
/// 测试挂一个观察者在那条事件上，放开 A 的执行器、然后原地自旋，直到 A 的收尾真的跑
/// 完为止（`hit_window` 记录这一点，窗口没被命中就让测试响）。没有交接标记的话，A 的
/// 收尾会在这个窗口里看到"没有别的 motion 在跑"，于是多发一次 `ACTING → STANDING`，
/// B 随后被登记进一个 STANDING 的机器人里，起跑后的复查把它以 `reason:"fsm"` 取消掉。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn preemption_hands_the_fsm_over_without_a_spurious_standing() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let (tx_a, rx_a) = tokio::sync::oneshot::channel();
    let (tx_b, rx_b) = tokio::sync::oneshot::channel();
    let _c = adm
        .capabilities()
        .register(
            motion_spec("hold", json!({})),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx_a, rx_b])),
            }),
        )
        .unwrap();

    let a_done = Arc::new(AtomicBool::new(false));
    let adm2 = adm.clone();
    let flag = a_done.clone();
    let a = tokio::spawn(async move {
        let out = adm2.propose(proposal("hold", json!({}), 1)).await;
        flag.store(true, Ordering::SeqCst);
        out
    });
    wait_until("A to become the active motion", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;
    assert_eq!(fsm_tos(&log), vec!["STANDING", "ENGAGED", "ACTING"]);

    // 放行 A 的那一下必须来自 runtime 之外的线程：从正在自旋的 worker 里唤醒任务，
    // 它会落进那个 worker 的 LIFO 槽——别的 worker 偷不走，而这个 worker 正忙着自旋。
    let arm = Arc::new(AtomicBool::new(false));
    let armed = arm.clone();
    let releaser = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !armed.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let _ = tx_a.send(());
    });

    // 交接窗口里：放 A 收尾，并把窗口按住到它跑完。
    let hit_window = Arc::new(AtomicBool::new(false));
    let (hit, flag) = (hit_window.clone(), a_done.clone());
    let _o = log.observe(Arc::new(move |ev: &oeh_proto::WorldEvent| {
        if ev.kind.as_str() != "act/preempted" {
            return;
        }
        arm.store(true, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if flag.load(Ordering::SeqCst) {
                hit.store(true, Ordering::SeqCst);
                return;
            }
            std::thread::yield_now();
        }
    }));

    let adm3 = adm.clone();
    let b = tokio::spawn(async move { adm3.propose(proposal("hold", json!({}), 2)).await });
    assert_eq!(a.await.unwrap(), Outcome::Preempted);
    releaser.join().unwrap();
    // B 登记成 active motion 是观察者返回之后才可能发生的事（它就卡在 B 的 gate 里），
    // 所以等到这里再读 `hit_window`——`a.await` 只说明 A 收完了尾，自旋的观察者可能还
    // 差最后一圈没看见。
    wait_until("B to start", || {
        adm.active_motion() == Some(ProposalId("t-p2".into()))
    })
    .await;
    assert!(
        hit_window.load(Ordering::SeqCst),
        "被抢占者的收尾没有落在交接窗口里，这个测试就没验到东西"
    );
    // A 收完尾了，ACTING 还在交接中：没有多出来的 STANDING，也没有重新驱一遍。
    assert_eq!(
        fsm_tos(&log),
        vec!["STANDING", "ENGAGED", "ACTING"],
        "被抢占者的收尾不该交还 ACTING"
    );
    assert_eq!(fsm.state(), FsmState::Acting);

    tx_b.send(()).unwrap();
    assert!(matches!(b.await.unwrap(), Outcome::Executed { .. }));
    assert_eq!(
        fsm_tos(&log),
        vec!["STANDING", "ENGAGED", "ACTING", "STANDING"]
    );
}

/// 执行器手里那份 params 必须是夹过的——两个字段都是。`call.proposal.params` 要是还留着
/// 提案原样，随手读它的执行器就把整条规则链绕过去了。
#[tokio::test]
async fn the_executor_only_ever_sees_clamped_params() {
    struct ClampVx;
    impl Rule for ClampVx {
        fn name(&self) -> &str {
            "clamp-vx"
        }
        fn evaluate(&self, p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
            if p.get("vx")
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v > 0.05)
            {
                p["vx"] = json!(0.05);
            }
            i
        }
    }
    struct Recorder {
        seen: Mutex<Vec<ExecutionCall>>,
    }
    impl Executor for Recorder {
        fn execute(&self, call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
            self.seen.lock().unwrap().push(call);
            Box::pin(async {
                ExecOutcome {
                    ok: true,
                    result: json!({}),
                    error: None,
                }
            })
        }
    }

    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(ClampVx));
    let rec = Arc::new(Recorder {
        seen: Mutex::new(vec![]),
    });
    let _c = adm
        .capabilities()
        .register(
            motion_spec(
                "creep",
                json!({"type": "object", "properties": {"vx": {"type": "number"}}}),
            ),
            rec.clone(),
        )
        .unwrap();

    let out = adm.propose(proposal("creep", json!({"vx": 0.9}), 1)).await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    let seen = rec.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].params, json!({"vx": 0.05}), "params 必须是夹过的");
    assert_eq!(
        seen[0].proposal.params, seen[0].params,
        "proposal.params 必须和 params 一致（都是夹过的那份）"
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/clamped" && e.payload["rule"] == "clamp-vx"));
}

/// 规则可以收紧参数，但不能把它夹成执行器没法接的形状：夹过之后要重新过一遍能力自己的
/// schema，过不了就拒。
#[tokio::test]
async fn clamped_params_that_violate_the_schema_are_refused() {
    struct BreakSchema;
    impl Rule for BreakSchema {
        fn name(&self) -> &str {
            "break-schema"
        }
        fn evaluate(&self, p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
            p["vx"] = json!("fast");
            i
        }
    }
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let _r = adm.register_rule(Arc::new(BreakSchema));
    let out = adm.propose(proposal("walk", json!({"vx": 0.1}), 1)).await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("clamped params violate schema")),
        "{out:?}"
    );
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/refused"
            && e.payload["stage"] == "rules"
            && e.payload["reason"]
                .as_str()
                .unwrap()
                .contains("clamped params violate schema")));
    // 起跑相关的事件一条都不该有。
    assert_eq!(
        kinds(&log, "t"),
        vec!["act/proposed", "act/clamped", "act/refused"]
    );
}

/// 登记表按 `proposal_id` 索引：一个重名的提案会把前一次执行悄悄顶掉——先收尾的那个
/// 把另一个的登记一起摘走，剩下那次执行的租约、FSM 和撤销权就再也没人管了。重复的 id
/// 在登记之前（抢占之前）就被挡下来：拒一个重名的提案，不该顺带把正在跑的 motion 抢掉。
#[tokio::test]
async fn a_duplicate_proposal_id_is_refused_before_it_preempts_anything() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let (tx_a, rx_a) = tokio::sync::oneshot::channel();
    let _c = adm
        .capabilities()
        .register(
            motion_spec("hold", json!({})),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx_a])),
            }),
        )
        .unwrap();

    let adm2 = adm.clone();
    let first = tokio::spawn(async move { adm2.propose(proposal("hold", json!({}), 1)).await });
    wait_until("第一次执行起跑", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;

    // 同一个 proposal_id 再来一次。
    let dup = adm.propose(proposal("hold", json!({}), 1)).await;
    assert!(
        matches!(dup, Outcome::Refused { ref reason } if reason == "duplicate proposal_id"),
        "{dup:?}"
    );
    let evs = log.since(0).unwrap();
    assert!(evs.iter().any(|e| e.kind.as_str() == "act/refused"
        && e.payload["stage"] == "vocabulary"
        && e.payload["reason"] == "duplicate proposal_id"));
    assert!(
        !evs.iter().any(|e| e.kind.as_str() == "act/preempted"),
        "重名的提案不该抢占任何东西: {:?}",
        kinds(&log, "t")
    );
    // 第一次执行毫发无伤：还在跑，还是 active motion，收尾照常。
    assert_eq!(adm.active_motion(), Some(ProposalId("t-p1".into())));
    assert_eq!(fsm.state(), FsmState::Acting);
    tx_a.send(()).unwrap();
    assert!(matches!(first.await.unwrap(), Outcome::Executed { .. }));
    assert_eq!(fsm.state(), FsmState::Standing);
}

/// 第三方代码可以在 `execute()` / `decide()` **本身**里 panic——那一刻 future 还没造出来，
/// `tokio::spawn` 和 `timeout` 都挡不住它：unwind 直接穿过 `start` → `pipeline` → 脑的
/// 任务。执行器那一侧最要命：收尾永远不跑，登记摘不掉，FSM 就永远停在 ACTING 而并没有
/// 任何 motion 在跑，谁也不会自愈。两处都按"调用本身就可能炸"来写。
#[tokio::test]
async fn a_panicking_executor_or_approval_does_not_strand_the_pipeline() {
    struct PanicNow;
    impl Executor for PanicNow {
        fn execute(&self, _c: ExecutionCall, _x: ExecCtx) -> BoxFuture<ExecOutcome> {
            panic!("executor exploded before returning a future");
        }
    }
    struct PanicApproval;
    impl Approval for PanicApproval {
        fn decide(&self, _r: AskRequest) -> BoxFuture<Decision> {
            panic!("approval exploded before returning a future");
        }
        fn name(&self) -> &str {
            "panic-approval"
        }
    }
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let _c = adm
        .capabilities()
        .register(motion_spec("boom", json!({})), Arc::new(PanicNow))
        .unwrap();

    // panic 会走默认 hook 打一大段到 stderr；这两下是故意的，别污染测试输出。
    std::panic::set_hook(Box::new(|_| {}));
    let out = adm.propose(proposal("boom", json!({}), 1)).await;
    let _a = adm.set_approval(Arc::new(PanicApproval));
    let _r2 = adm.register_rule(Arc::new(AskAlways));
    let asked = adm.propose(proposal("say", json!({"text": "hi"}), 2)).await;
    let _ = std::panic::take_hook();

    // 执行器：结局是 Failed，`act/result{ok:false}` 照记，FSM 交还，位子腾空。
    assert!(
        matches!(out, Outcome::Failed { ref error } if error == "executor panicked"),
        "{out:?}"
    );
    let evs = log.since(0).unwrap();
    assert!(evs.iter().any(|e| e.kind.as_str() == "act/result"
        && e.payload["proposal_id"] == "t-p1"
        && e.payload["ok"] == false
        && e.payload["error"] == "executor panicked"));
    assert_eq!(fsm.state(), FsmState::Standing);
    assert!(adm.active_motion().is_none());

    // 审批：一个 panic 的审批服务 = 拒绝，不是挂起，也不是放行。
    assert!(matches!(asked, Outcome::Refused { .. }), "{asked:?}");
    let evs = log.since(0).unwrap();
    assert!(evs.iter().any(|e| e.kind.as_str() == "act/decision"
        && e.payload["proposal_id"] == "t-p2"
        && e.payload["decision"] == "deny"
        && e.payload["by"] == "panic-approval"
        && e.payload["reason"] == "approval panicked"));
    assert!(evs.iter().any(|e| e.kind.as_str() == "act/refused"
        && e.payload["proposal_id"] == "t-p2"
        && e.payload["stage"] == "approval"));

    // 登记摘干净了：同一个 id 不再被当成重复的那一个。
    let again = adm
        .propose(proposal("say", json!({"text": "again"}), 1))
        .await;
    assert!(
        !matches!(again, Outcome::Refused { ref reason } if reason == "duplicate proposal_id"),
        "panic 之后登记没摘干净: {again:?}"
    );
}

/// 收尾里"还有没有 motion 在跑？→ 把 ACTING 交还成 STANDING"是两步，中间另一个 motion
/// 完全可能刚好抢到位子：它看到 FSM 还是 ACTING（于是不驱动）、登记自己、随后前一个的
/// 收尾把 FSM 落回 STANDING——B 就在一台自称 STANDING 的机器人上跑完了整趟。`safety/fsm`
/// 那条事件的 source 是 admission 自己，观察者会把它过滤掉，没人会自愈。
///
/// 所以收尾的这个判断挪进了 `admit` gate，和"抢占 → 驱 FSM → 登记 → 起跑"那一段互斥。
/// 这里把 gate 按住来验：B 在 gate 里（`act/started` 的观察者是同步回调，在那儿自旋），
/// 同时放开 A 的执行器。A 的收尾这时**不许**跑完——它得排在 gate 后面。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_finish_cannot_take_its_fsm_decision_while_another_motion_is_in_the_gate() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let (tx_a, rx_a) = tokio::sync::oneshot::channel();
    let (tx_b, rx_b) = tokio::sync::oneshot::channel();
    let _c = adm
        .capabilities()
        .register(
            motion_spec("hold", json!({})),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx_a, rx_b])),
            }),
        )
        .unwrap();

    let a_done = Arc::new(AtomicBool::new(false));
    let adm2 = adm.clone();
    let flag = a_done.clone();
    let a = tokio::spawn(async move {
        let out = adm2.propose(proposal("hold", json!({}), 1)).await;
        flag.store(true, Ordering::SeqCst);
        out
    });
    wait_until("A 成为正在跑的 motion", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;

    // `stop` 把 A 撤掉：位子当场让出来，但 A 的执行器还被 oneshot 按着没返回，所以它的
    // 收尾还没发生。A 不是被**抢占**的，收尾时仍然要交还 ACTING——正是有风险的那条路。
    let stopped = adm.propose(proposal("stop", json!({}), 9)).await;
    assert!(matches!(stopped, Outcome::Executed { .. }), "{stopped:?}");
    assert!(adm.active_motion().is_none());
    assert_eq!(fsm.state(), FsmState::Acting, "A 还没收尾，ACTING 还在");

    // 放行 A 的那一下必须来自 runtime 之外的线程：从正在自旋的 worker 里唤醒任务，它会
    // 落进那个 worker 的 LIFO 槽，而这个 worker 正忙着自旋。
    let arm = Arc::new(AtomicBool::new(false));
    let armed = arm.clone();
    let releaser = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !armed.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let _ = tx_a.send(());
    });

    // B 的 `act/started` 记在 gate 里：观察者在这里自旋，gate 就被按住了。
    let leaked = Arc::new(AtomicBool::new(false));
    let (leak, flag) = (leaked.clone(), a_done.clone());
    let _o = log.observe(Arc::new(move |ev: &oeh_proto::WorldEvent| {
        if ev.kind.as_str() != "act/started" || ev.payload["proposal_id"] != "t-p2" {
            return;
        }
        arm.store(true, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + Duration::from_millis(400);
        while std::time::Instant::now() < deadline {
            if flag.load(Ordering::SeqCst) {
                leak.store(true, Ordering::SeqCst);
                return;
            }
            std::thread::yield_now();
        }
    }));

    let adm3 = adm.clone();
    let b = tokio::spawn(async move { adm3.propose(proposal("hold", json!({}), 2)).await });
    wait_until("B 成为正在跑的 motion", || {
        adm.active_motion() == Some(ProposalId("t-p2".into()))
    })
    .await;
    releaser.join().unwrap();
    assert_eq!(
        a.await.unwrap(),
        Outcome::Cancelled {
            reason: "stop".into()
        }
    );
    assert!(
        !leaked.load(Ordering::SeqCst),
        "A 的收尾在 B 还占着 gate 的时候就做完了 FSM 判断——正是那个 TOCTOU"
    );
    // B 全程跑在 ACTING 上，没有多出来的 STANDING。
    assert_eq!(fsm.state(), FsmState::Acting);
    assert_eq!(fsm_tos(&log), vec!["STANDING", "ENGAGED", "ACTING"]);

    tx_b.send(()).unwrap();
    assert!(matches!(b.await.unwrap(), Outcome::Executed { .. }));
    assert_eq!(
        fsm_tos(&log),
        vec!["STANDING", "ENGAGED", "ACTING", "STANDING"]
    );
}

/// 急停按下时正在跑的运动必须停。`rule-estop` 只挡**后续**提案；一台在操作员按 ESC 前
/// 一毫秒被准入的 `walk`，在这条链修好之前会揣着自己的租约和 ACTING 继续对适配器下指令。
/// `safety/fsm` 的那条观察者→通道→任务本来就在，`safety/estop{engaged:true}` 走同一条。
#[tokio::test]
async fn an_engaged_estop_cancels_the_motion_that_is_already_running() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let _c = adm
        .capabilities()
        .register(
            motion_spec("creep", json!({})),
            Arc::new(UntilCancelled {
                saw_cancel: saw_cancel.clone(),
            }),
        )
        .unwrap();

    let adm2 = adm.clone();
    let running = tokio::spawn(async move { adm2.propose(proposal("creep", json!({}), 1)).await });
    wait_until("运动起跑", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;

    // 操作员按下急停：`distill-estop` 发的就是这一条（source 不是 admission）。
    log.append(
        oeh_proto::EventDraft::new(
            oeh_proto::EventKind::new("safety/estop").unwrap(),
            "distill-estop",
            json!({"engaged": true, "flag_path": "/tmp/oeh_estop"}),
        )
        .with_salience(255),
    );

    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "estop".into()
        }
    );
    assert!(saw_cancel.load(Ordering::SeqCst), "执行器应该看到取消");
    assert!(log
        .since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == "act/cancelled"
            && e.payload["reason"] == "estop"
            && e.payload["proposal_id"] == "t-p1"));
    assert!(adm.active_motion().is_none());
    wait_until("FSM 交还 STANDING", || fsm.state() == FsmState::Standing).await;
}

/// 规格 §11：规则插件 panic = deny，而且要留下一条**健康**事件。`evaluate_chain` 已经
/// 把这次判定变成 Deny 并在 trace 里标 `panicked`，但那只说明"这一次没放行"；一条坏掉
/// 的规则本身是要被看见的（控制台、Plan 3 的 `plugin/state` 都要读它），所以另记一条
/// `health/distiller_error{rule, reason:"panic", stage:"admission"}`。
#[tokio::test]
async fn a_panicking_rule_is_denied_and_recorded_as_a_health_event() {
    struct Panicky;
    impl Rule for Panicky {
        fn name(&self) -> &str {
            "panicky"
        }
        fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
            panic!("boom")
        }
    }
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let _r = adm.register_rule(Arc::new(Panicky));

    std::panic::set_hook(Box::new(|_| {}));
    let out = adm.propose(proposal("walk", json!({"vx": 0.1}), 1)).await;
    let _ = std::panic::take_hook();

    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("panicked")),
        "{out:?}"
    );
    let evs = log.since(0).unwrap();
    let health = evs
        .iter()
        .find(|e| e.kind.as_str() == "health/distiller_error")
        .expect("规则 panic 要记一条 health/distiller_error");
    assert_eq!(health.payload["rule"], "panicky");
    assert_eq!(health.payload["reason"], "panic");
    assert_eq!(health.payload["stage"], "admission");
    assert_eq!(health.salience, 200);
    assert_eq!(health.trace_id.as_deref(), Some("t"));
    assert!(!kinds(&log, "t").contains(&"act/started".to_string()));
}

/// 规格 §5.3：`walk` 这类持续能力在执行期间要按周期复检，失败即 cancel。准入只看了
/// 起跑那一瞬间的世界——一个人走到正在行走的机器人前面，在这之前不会产生任何撤销。
#[tokio::test]
async fn an_in_flight_motion_is_cancelled_when_the_rules_stop_allowing_it() {
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor" && e.id != "admission");
    r.push(
        EntryRow::new("admission")
            .with_config(json!({"ask_timeout_ms": 200, "lease_check_ms": 20, "recheck_ms": 10})),
    );
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    state.set_many(
        &[
            ("clear_path", json!(true)),
            ("nearest_obstacle_m", Value::Null),
            ("nearest_person_m", Value::Null),
        ],
        "test",
    );
    let _r = adm.register_rule(Arc::new(SceneRule::default()));
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let _c = adm
        .capabilities()
        .register(
            motion_spec("walk", json!({})),
            Arc::new(UntilCancelled {
                saw_cancel: saw_cancel.clone(),
            }),
        )
        .unwrap();

    let adm2 = adm.clone();
    let running = tokio::spawn(async move { adm2.propose(proposal("walk", json!({}), 1)).await });
    wait_until("运动起跑", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;

    // 世界变了：路不空了。复检下一拍就该把它叫停。
    state.set("clear_path", json!(false), "test");

    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "rules".into()
        }
    );
    assert!(saw_cancel.load(Ordering::SeqCst), "执行器应该看到取消");
    let cancelled = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "act/cancelled")
        .expect("act/cancelled");
    assert_eq!(cancelled.payload["reason"], "rules");
    let detail = cancelled.payload["detail"].as_str().unwrap_or_default();
    assert!(
        detail.contains("rule-scene") && detail.contains("path is not clear"),
        "撤销要说清楚是哪条规则、为什么: {detail}"
    );
    wait_until("FSM 交还 STANDING", || fsm.state() == FsmState::Standing).await;
}

/// 审批可能让人想上 20 秒（`ask_timeout_ms` 默认就是这个数）。链在**问之前**跑过一次，
/// 人点头之后直接起跑——那份判定可能已经是 20 秒前的世界了。放行之后必须再过一遍链：
/// 此刻 Deny 就是 `act/refused{stage:"recheck"}`；此刻还是 Ask 不再问第二遍，但留痕。
#[tokio::test]
async fn an_approved_proposal_is_re_checked_against_a_fresh_snapshot() {
    // 少一条必需规则 → 初始判定是 Ask，于是每次提案都要走审批。
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "admission");
    r.push(EntryRow::new("admission").with_config(json!({
        "required_rules": ["rule-fsm"], "ask_timeout_ms": 2000, "lease_check_ms": 20, "recheck_ms": 0
    })));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    state.set_many(
        &[
            ("clear_path", json!(true)),
            ("nearest_obstacle_m", Value::Null),
            ("nearest_person_m", Value::Null),
        ],
        "test",
    );
    let _r = adm.register_rule(Arc::new(SceneRule::default()));
    let (chan, mut rx) = ChannelApproval::new();
    let _a = adm.set_approval(chan);

    // 第一次：人点头时世界没变。复检的判定仍是 `Ask`（还缺 `rule-fsm`）——不再问第二遍，
    // 但记一条 `act/recheck`，因为批准的和起跑的不是同一份判定。
    let adm2 = adm.clone();
    let first = tokio::spawn(async move { adm2.propose(proposal("walk", json!({}), 1)).await });
    let (_req, tx) = rx.recv().await.unwrap();
    tx.send(Decision::Allow {
        by: "op".into(),
        reason: "ok".into(),
    })
    .unwrap();
    assert!(matches!(first.await.unwrap(), Outcome::Executed { .. }));
    assert_eq!(
        kinds(&log, "t"),
        vec![
            "act/proposed",
            "act/ask",
            "act/decision",
            "act/recheck",
            "act/admitted",
            "act/started",
            "act/result"
        ]
    );
    let recheck = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "act/recheck")
        .unwrap();
    assert_eq!(recheck.payload["verdict"], "ask");
    assert_eq!(recheck.payload["proposal_id"], "t-p1");

    // 第二次：人还在想的时候路被占了。批准回来时复检 Deny，绝不能起跑。
    let adm3 = adm.clone();
    let second = tokio::spawn(async move { adm3.propose(proposal("walk", json!({}), 2)).await });
    let (_req, tx) = rx.recv().await.unwrap();
    state.set("clear_path", json!(false), "test");
    tx.send(Decision::Allow {
        by: "op".into(),
        reason: "ok".into(),
    })
    .unwrap();
    assert!(
        matches!(second.await.unwrap(), Outcome::Refused { ref reason } if reason == "path is not clear"),
    );
    let refused = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "act/refused" && e.payload["proposal_id"] == "t-p2")
        .expect("act/refused");
    assert_eq!(refused.payload["stage"], "recheck");
    assert!(refused.payload["trace"].is_array(), "复检也要留完整 trace");
    assert_eq!(
        log.since(0)
            .unwrap()
            .iter()
            .filter(|e| e.kind.as_str() == "act/started")
            .count(),
        1,
        "复检 Deny 的那次绝不该起跑"
    );
}

/// 租约过期说明执行器**失联**了：它可能还在让机器人动，只是不再报告。原来的做法是当场
/// 把 motion 的位子腾出来，于是下一个运动可以立刻起跑——在前一个可能仍在物理执行的时候，
/// 两条运动指令重叠。规格 §5.3 还要求这一刻"内核向适配器发 stop"。
///
/// 现在：记 `act/lease_expired` + 撤销，位子**继续**由失联者占着（新的 motion 只能走
/// 抢占那条明确的路），同时回头叫**这个执行器自己**的 `stop()`，等到 `lease_grace_ms`
/// 还没等回它就强行摘掉登记、交还 ACTING。
#[tokio::test]
async fn an_expired_lease_holds_the_slot_sends_stop_and_frees_it_after_the_grace() {
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor" && e.id != "admission");
    r.push(EntryRow::new("admission").with_config(json!({
        "ask_timeout_ms": 200, "lease_check_ms": 20, "lease_grace_ms": 300, "recheck_ms": 0
    })));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let spec: CapabilitySpec = serde_json::from_value(json!({
        "name": "stuck", "class": "motion", "risk_level": "low",
        "params_schema": {}, "lease_ttl_s": 0.05
    }))
    .unwrap();
    let stuck = Arc::new(Forever::default());
    let _c = adm.capabilities().register(spec, stuck.clone()).unwrap();

    let adm2 = adm.clone();
    let _lost = tokio::spawn(async move { adm2.propose(proposal("stuck", json!({}), 1)).await });
    wait_until("租约过期", || {
        log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/lease_expired")
    })
    .await;

    // 位子还在失联者手里：没人能从一个"空出来的"位子上悄悄起跑。
    assert_eq!(adm.active_motion(), Some(ProposalId("t-p1".into())));
    assert_eq!(fsm.state(), FsmState::Acting);

    // 内核回头叫**失联的那个执行器**停下来（规格 §5.3），叫的就是它自己那次执行。
    wait_until("执行器收到 stop", || {
        !stuck.stopped.lock().unwrap().is_empty()
    })
    .await;
    assert_eq!(
        *stuck.stopped.lock().unwrap(),
        vec![ProposalId("t-p1".into())]
    );

    // 宽限期过完：执行器还是没回来，强行摘掉登记，位子和 ACTING 都还回去。
    wait_until("宽限期后位子腾空", || adm.active_motion().is_none()).await;
    wait_until("FSM 交还 STANDING", || fsm.state() == FsmState::Standing).await;
}

/// 提案方把 future 丢掉（Plan 3 的 RPC 断线、Plan 4 的脑任务被 abort）之后，一次已经
/// 记了 `act/proposed` 的提案不能就此消失：登记、租约、FSM 都还等着它收尾，而审计里
/// 一条没有终局事件的提案，读的人无从知道机器人到底动了没有。
#[tokio::test]
async fn a_dropped_proposal_future_still_reaches_a_terminal_event() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AskAlways));
    // 永远不应答的审批端：提案会一直卡在 `act/ask` 上，直到 `ask_timeout_ms`（200ms）。
    let (chan, _rx) = ChannelApproval::new();
    let _a = adm.set_approval(chan);

    // 提案发出去了，然后脑走了。
    drop(adm.propose(proposal("walk", json!({"vx": 0.1}), 1)));

    wait_until("提案走到终局", || {
        log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/refused" && e.payload["proposal_id"] == "t-p1")
    })
    .await;
    let evs = log.since(0).unwrap();
    // 每一条 `act/proposed` 都有一个终局事件。
    for proposed in evs.iter().filter(|e| e.kind.as_str() == "act/proposed") {
        let pid = &proposed.payload["proposal_id"];
        assert!(
            evs.iter().any(|e| {
                matches!(
                    e.kind.as_str(),
                    "act/refused"
                        | "act/result"
                        | "act/cancelled"
                        | "act/preempted"
                        | "act/lease_expired"
                ) && &e.payload["proposal_id"] == pid
            }),
            "提案 {pid} 没有终局事件: {:?}",
            kinds(&log, "t")
        );
    }
    assert_eq!(
        find_payload(&log, "act/refused", "t-p1")["stage"],
        "approval"
    );
}

/// 某个 kind + proposal_id 的 payload。
fn find_payload(log: &WorldLog, kind: &str, pid: &str) -> Value {
    log.since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == kind && e.payload["proposal_id"] == pid)
        .unwrap_or_else(|| panic!("没有 {kind}({pid})"))
        .payload
}

/// 租约过期时叫停的必须是**失联的那一个**，不是"当前正在跑的那个 motion"。
///
/// `sweep_leases` 不按 class 过滤，而所有内置能力的默认租约都是 5 秒：一次挂住的
/// speech（Plan 3 里一个不再应答的远端 `say`）租约到期，如果兜底的做法是去调内置 `stop`
/// 能力，那条能力的语义恰恰是"撤销当前 motion"——于是一台完全健康的 `walk` 被一次和它
/// 毫无关系的超时停掉。叫停因此走 `Executor::stop`，打给那次执行自己的执行器。
#[tokio::test]
async fn an_expired_non_motion_lease_leaves_a_healthy_motion_alone() {
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor" && e.id != "admission");
    r.push(EntryRow::new("admission").with_config(json!({
        "ask_timeout_ms": 200, "lease_check_ms": 20, "lease_grace_ms": 60000, "recheck_ms": 0
    })));
    // 心跳很勤、跑很久的 walk：健康得不能再健康。
    let mut long_walk = walk_spec();
    long_walk["lease_ttl_s"] = json!(5.0);
    r.push(
        EntryRow::new("mock-executor").with_config(
            json!({"capabilities": [long_walk], "duration_s": 5.0, "heartbeat_ms": 20}),
        ),
    );
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));

    // 一次会挂住的非 motion 执行，租约很短。
    let hung = Arc::new(Forever::default());
    let spec: CapabilitySpec = serde_json::from_value(json!({
        "name": "announce", "class": "speech", "risk_level": "none",
        "params_schema": {}, "lease_ttl_s": 0.05
    }))
    .unwrap();
    let _c = adm.capabilities().register(spec, hung.clone()).unwrap();

    let adm2 = adm.clone();
    let _walking = tokio::spawn(async move { adm2.propose(proposal("walk", json!({}), 1)).await });
    wait_until("walk 起跑", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;
    let adm3 = adm.clone();
    let _hanging =
        tokio::spawn(async move { adm3.propose(proposal("announce", json!({}), 2)).await });
    wait_until("announce 的租约过期", || {
        log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/lease_expired" && e.payload["proposal_id"] == "t-p2")
    })
    .await;
    // 叫停的是挂住的那一个。
    wait_until("挂住的执行器收到 stop", || {
        !hung.stopped.lock().unwrap().is_empty()
    })
    .await;
    assert_eq!(
        *hung.stopped.lock().unwrap(),
        vec![ProposalId("t-p2".into())]
    );

    // 健康的 walk 毫发无伤。
    let evs = log.since(0).unwrap();
    assert!(
        !evs.iter().any(|e| e.kind.as_str() == "act/cancelled"),
        "一次无关的 speech 超时不该撤销任何东西: {:?}",
        kinds(&log, "t")
    );
    assert_eq!(adm.active_motion(), Some(ProposalId("t-p1".into())));
    assert_eq!(fsm.state(), FsmState::Acting);

    // 挂住的那次提案恰好一个终局事件。
    let terminals: Vec<&str> = evs
        .iter()
        .filter(|e| {
            e.payload["proposal_id"] == "t-p2"
                && matches!(
                    e.kind.as_str(),
                    "act/refused"
                        | "act/result"
                        | "act/cancelled"
                        | "act/preempted"
                        | "act/lease_expired"
                )
        })
        .map(|e| e.kind.as_str())
        .collect();
    assert_eq!(terminals, vec!["act/lease_expired"]);
}

/// 复检任务可以活得比它那次执行更久。`ProposalId` 一旦被重新登记，一个陈旧的任务就会
/// 拿着**旧**能力的 spec 和旧的 params 去评估**新**的那次执行——一次 Deny 就把一个和它
/// 毫无关系的执行以 `reason:"rules"` 撤掉，`detail` 里还写着另一条能力的规则。内核生成的
/// id 每 turn 唯一，但 `Admission::propose` 是公开的，Plan 3 会把它摆到更远的地方。
///
/// 登记因此带一个全局单调的世代号：陈旧的任务认不出新的那一个。
#[tokio::test]
async fn a_stale_recheck_task_cannot_cancel_a_re_registered_proposal_id() {
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor" && e.id != "admission");
    // recheck_ms 0：复检由测试自己驱动，不靠时钟。
    r.push(
        EntryRow::new("admission")
            .with_config(json!({"ask_timeout_ms": 200, "lease_check_ms": 20, "recheck_ms": 0})),
    );
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    state.set_many(
        &[
            ("clear_path", json!(true)),
            ("nearest_obstacle_m", Value::Null),
            ("nearest_person_m", Value::Null),
        ],
        "test",
    );
    let _r = adm.register_rule(Arc::new(SceneRule::default()));
    // `walk` 在 `rule-scene` 的 motion_caps 里；`creep` 不在，它的判定不看 clear_path。
    let walk = motion_spec("walk", json!({}));
    let (tx_a, rx_a) = tokio::sync::oneshot::channel();
    let (tx_b, rx_b) = tokio::sync::oneshot::channel();
    let _c1 = adm
        .capabilities()
        .register(
            walk.clone(),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx_a])),
            }),
        )
        .unwrap();
    let _c2 = adm
        .capabilities()
        .register(
            motion_spec("creep", json!({})),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx_b])),
            }),
        )
        .unwrap();

    let pid = ProposalId("t-p1".into());
    let adm2 = adm.clone();
    let first = tokio::spawn(async move { adm2.propose(proposal("walk", json!({}), 1)).await });
    wait_until("walk 起跑", || adm.active_motion() == Some(pid.clone())).await;
    // 一个复检任务从起跑那一刻起攥着的东西。
    let stale_epoch = adm.exec_epoch(&pid).expect("登记的世代号");

    tx_a.send(()).unwrap();
    assert!(matches!(first.await.unwrap(), Outcome::Executed { .. }));
    wait_until("walk 的登记摘掉", || adm.exec_epoch(&pid).is_none()).await;

    // 同一个 proposal_id 再登记一次，这次是另一条能力。
    let adm3 = adm.clone();
    let second = tokio::spawn(async move { adm3.propose(proposal("creep", json!({}), 1)).await });
    wait_until("creep 起跑", || adm.active_motion() == Some(pid.clone())).await;
    let fresh_epoch = adm.exec_epoch(&pid).expect("新登记的世代号");
    assert_ne!(stale_epoch, fresh_epoch, "重新登记必须换一个世代号");

    // 世界变了：`walk` 现在过不了 `rule-scene`，`creep` 不受影响。
    state.set("clear_path", json!(false), "test");

    // 陈旧的任务回来敲门：它该发现自己已经过期，什么都不碰。
    assert!(
        !adm.recheck_now(&pid, stale_epoch, &walk, &json!({})),
        "陈旧的复检任务该就此结束"
    );
    assert_eq!(
        adm.active_motion(),
        Some(pid.clone()),
        "陈旧的复检不该撤销另一次执行"
    );
    assert!(
        !log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/cancelled"),
        "{:?}",
        kinds(&log, "t")
    );

    // 同一组判据配上**当前**的世代号确实会撤销——证明上面那一下不是因为规则没咬住。
    assert!(!adm.recheck_now(&pid, fresh_epoch, &walk, &json!({})));
    // `Held` 只认它那个 oneshot，不看取消：放它走，收尾才跑得完。
    drop(tx_b);
    assert_eq!(
        second.await.unwrap(),
        Outcome::Cancelled {
            reason: "rules".into()
        }
    );
}

/// 接班者走到一个**租约已经过期**的 motion 头上。那一个的终局事件（`act/lease_expired`）
/// 在过期那一刻就记过了，所以绝不能再记第二条；但 ACTING 这块地确实换了主人，交接必须
/// 正经立起来——`claim` 遇到已经带着 `cancel_reason` 的登记直接返回 `None`，于是
/// `handed_over` 一直是 false，日志里也看不出这次起跑是从谁手里接的：读日志的人只看到
/// 一条 `act/lease_expired`，然后一条凭空出现在已经 ACTING 的机器人上的 `act/started`。
#[tokio::test]
async fn a_successor_takes_over_a_lease_expired_motion_without_a_second_terminal_event() {
    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor" && e.id != "admission");
    r.push(EntryRow::new("admission").with_config(json!({
        "ask_timeout_ms": 200, "lease_check_ms": 20, "lease_grace_ms": 200, "recheck_ms": 0
    })));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let stuck_spec: CapabilitySpec = serde_json::from_value(json!({
        "name": "stuck", "class": "motion", "risk_level": "low",
        "params_schema": {}, "lease_ttl_s": 0.05
    }))
    .unwrap();
    let _c1 = adm
        .capabilities()
        .register(stuck_spec, Arc::new(Forever::default()))
        .unwrap();
    let (tx_b, rx_b) = tokio::sync::oneshot::channel();
    let _c2 = adm
        .capabilities()
        .register(
            motion_spec("hold", json!({})),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx_b])),
            }),
        )
        .unwrap();

    let adm2 = adm.clone();
    let _lost = tokio::spawn(async move { adm2.propose(proposal("stuck", json!({}), 1)).await });
    wait_until("A 的租约过期", || {
        log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/lease_expired" && e.payload["proposal_id"] == "t-p1")
    })
    .await;
    assert_eq!(
        adm.active_motion(),
        Some(ProposalId("t-p1".into())),
        "失联者还占着位子"
    );

    // 接班者在宽限期内起跑。
    let adm3 = adm.clone();
    let b = tokio::spawn(async move { adm3.propose(proposal("hold", json!({}), 2)).await });
    wait_until("B 接手", || {
        adm.active_motion() == Some(ProposalId("t-p2".into()))
    })
    .await;

    let evs = log.since(0).unwrap();
    let started = evs
        .iter()
        .find(|e| e.kind.as_str() == "act/started" && e.payload["proposal_id"] == "t-p2")
        .expect("B 的 act/started");
    assert_eq!(
        started.payload["took_over_expired"], "t-p1",
        "接班者要说清楚它是从谁手里接的 ACTING: {}",
        started.payload
    );
    assert!(
        !evs.iter().any(|e| e.kind.as_str() == "act/preempted"),
        "失联者的终局早记过了，不能再补一条 act/preempted"
    );

    // 宽限期过完，失联者的登记被强行摘掉——ACTING 归 B，不许多出一条 STANDING。
    wait_until("A 的登记被摘掉", || {
        adm.exec_epoch(&ProposalId("t-p1".into())).is_none()
    })
    .await;
    assert_eq!(fsm.state(), FsmState::Acting);
    assert_eq!(fsm_tos(&log), vec!["STANDING", "ENGAGED", "ACTING"]);

    // 每条提案恰好一个终局事件。
    tx_b.send(()).unwrap();
    assert!(matches!(b.await.unwrap(), Outcome::Executed { .. }));
    let evs = log.since(0).unwrap();
    let terminals = |pid: &str| -> Vec<&str> {
        evs.iter()
            .filter(|e| {
                e.payload["proposal_id"] == pid
                    && matches!(
                        e.kind.as_str(),
                        "act/refused"
                            | "act/result"
                            | "act/cancelled"
                            | "act/preempted"
                            | "act/lease_expired"
                    )
            })
            .map(|e| e.kind.as_str())
            .collect()
    };
    assert_eq!(terminals("t-p1"), vec!["act/lease_expired"]);
    assert_eq!(terminals("t-p2"), vec!["act/result"]);
    assert_eq!(
        fsm_tos(&log),
        vec!["STANDING", "ENGAGED", "ACTING", "STANDING"]
    );
}

/// 执行期间的复检每一拍都把整条链重跑一遍。一条"想放水"的规则要是每拍都被记一条
/// `safety/tamper_attempt`，salience 255——操作员的告警通道——就会被同一条消息刷屏。
/// 同一次执行里，同一条规则只响一次。
///
/// （今天链把篡改一律强制成 Deny，所以第一拍就会把这次执行撤掉、后两拍在读登记那一步
/// 就返回了；计数保持在 1 于是有两道保险，去重是更靠后、也更不依赖链语义的那一道。）
#[tokio::test]
async fn the_in_flight_recheck_reports_a_tampering_rule_only_once() {
    /// 起跑之后才开始想放水的规则：准入那一遍不能篡改，否则提案根本进不来。
    struct LoosenWhenArmed {
        armed: Arc<AtomicBool>,
    }
    impl Rule for LoosenWhenArmed {
        fn name(&self) -> &str {
            "loosen-later"
        }
        fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
            if self.armed.load(Ordering::SeqCst) {
                Verdict::Allow
            } else {
                i
            }
        }
    }

    let mut r = rows(vec![]);
    r.retain(|e| e.id != "mock-executor" && e.id != "admission");
    // 少一条必需规则 → 初始判定是 Ask，于是"放松成 Allow"才有得放松。
    r.push(EntryRow::new("admission").with_config(json!({
        "required_rules": ["rule-fsm"], "ask_timeout_ms": 200, "lease_check_ms": 20, "recheck_ms": 0
    })));
    let k = Kernel::boot(r).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let armed = Arc::new(AtomicBool::new(false));
    let _r = adm.register_rule(Arc::new(LoosenWhenArmed {
        armed: armed.clone(),
    }));
    let _a = adm.set_approval(Arc::new(AutoApproval { allow: true }));
    let spec = motion_spec("hold", json!({}));
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _c = adm
        .capabilities()
        .register(
            spec.clone(),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::from([rx])),
            }),
        )
        .unwrap();

    let pid = ProposalId("t-p1".into());
    let adm2 = adm.clone();
    let running = tokio::spawn(async move { adm2.propose(proposal("hold", json!({}), 1)).await });
    wait_until("起跑", || adm.active_motion() == Some(pid.clone())).await;
    let epoch = adm.exec_epoch(&pid).expect("登记的世代号");

    armed.store(true, Ordering::SeqCst);
    for _ in 0..3 {
        adm.recheck_now(&pid, epoch, &spec, &json!({}));
    }

    let tampers: Vec<oeh_proto::WorldEvent> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "safety/tamper_attempt")
        .collect();
    assert_eq!(tampers.len(), 1, "同一条规则在一次执行里只该响一次");
    assert_eq!(tampers[0].payload["rule"], "loosen-later");
    assert_eq!(tampers[0].salience, 255);

    drop(tx);
    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "rules".into()
        }
    );
}

/// 急停的观察者必须 fail-closed：`engaged` 读不出来（缺字段、类型不对）就当**按下了**。
/// 它的兄弟分支（`safety/fsm` 的 `to`）早就是这个方向——读不出来就当"不是 ACTING"，
/// 宁可多撤一次。只有明确的 `engaged: false` 才是"松开了"，那一条不撤任何东西。
/// 树内的 `distill-estop` 总是带上这个字段，但 Plan 3 的 `event/emit` 会让外部生产者
/// 直接写 `safety/*`。
#[tokio::test]
async fn an_estop_event_without_a_readable_engaged_flag_still_cancels() {
    let k = Kernel::boot(rows(vec![])).unwrap();
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let _c = adm
        .capabilities()
        .register(
            motion_spec("creep", json!({})),
            Arc::new(UntilCancelled {
                saw_cancel: saw_cancel.clone(),
            }),
        )
        .unwrap();

    let adm2 = adm.clone();
    let running = tokio::spawn(async move { adm2.propose(proposal("creep", json!({}), 1)).await });
    wait_until("运动起跑", || {
        adm.active_motion() == Some(ProposalId("t-p1".into()))
    })
    .await;

    // 松开急停的那一条不撤任何东西。
    log.append(oeh_proto::EventDraft::new(
        oeh_proto::EventKind::new("safety/estop").unwrap(),
        "somebody",
        json!({"engaged": false}),
    ));
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(
        adm.active_motion(),
        Some(ProposalId("t-p1".into())),
        "`engaged: false` 是松开，不该撤销"
    );

    // 读不出 `engaged` 的那一条当成按下了。
    log.append(oeh_proto::EventDraft::new(
        oeh_proto::EventKind::new("safety/estop").unwrap(),
        "somebody",
        json!({}),
    ));
    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "estop".into()
        }
    );
    assert!(saw_cancel.load(Ordering::SeqCst));
    wait_until("FSM 交还 STANDING", || fsm.state() == FsmState::Standing).await;
}

/// 服务被撤销时，还在途的提案不能就这么消失：它们已经记过 `act/proposed`，而一条没有
/// 终局事件的提案在审计里读起来就是"机器人到底动没动？不知道"。流水线跑在自己 spawn
/// 出来的任务上，所以那些任务由 `Inner` 的 `JoinSet` 攥着，`Drop` 里一并 abort——
/// `Terminal` 守卫到这里才真正派上用场。
#[tokio::test]
async fn dropping_the_service_gives_every_in_flight_proposal_a_terminal_event() {
    // 手工搭零件而不是 `Kernel::boot`：这个用例要把 `Admission` 自己 drop 掉，而世界
    // 日志必须活过它才读得到那条补记。
    let clock: Arc<dyn Clock> = Arc::new(MonotonicClock::new());
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let fsm = Arc::new(RobotFsm::new(log.clone(), state.clone()));
    let adm = Admission::new(
        clock,
        log.clone(),
        state,
        fsm,
        AdmissionConfig {
            // 永远等不到审批超时：能给这次提案一个终局的只有服务撤销本身。
            ask_timeout_ms: 600_000,
            lease_check_ms: 20,
            ..Default::default()
        },
    )
    .unwrap();
    let _r = adm.register_rule(Arc::new(AskAlways));
    let (chan, _rx) = ChannelApproval::new();
    let _a = adm.set_approval(chan);
    let _c = adm
        .capabilities()
        .register(
            motion_spec("hold", json!({})),
            Arc::new(Held {
                gates: Mutex::new(VecDeque::new()),
            }),
        )
        .unwrap();

    // 返回的 future 不持有 `Arc<Admission>`，丢掉它不妨碍服务被 drop；流水线已经在
    // 自己的任务上跑着了。
    drop(adm.propose(proposal("hold", json!({}), 1)));
    wait_until("提案卡在审批上", || {
        log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/ask")
    })
    .await;

    drop(adm);

    for _ in 0..200 {
        let evs = log.since(0).unwrap();
        if evs.iter().any(|e| {
            e.kind.as_str() == "act/cancelled"
                && e.payload["reason"] == "abandoned"
                && e.payload["proposal_id"] == "t-p1"
        }) {
            // 恰好一个终局事件。
            let terminals: Vec<&str> = evs
                .iter()
                .filter(|e| {
                    e.payload["proposal_id"] == "t-p1"
                        && matches!(
                            e.kind.as_str(),
                            "act/refused"
                                | "act/result"
                                | "act/cancelled"
                                | "act/preempted"
                                | "act/lease_expired"
                        )
                })
                .map(|e| e.kind.as_str())
                .collect();
            assert_eq!(terminals, vec!["act/cancelled"]);
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "服务撤销之后这条提案没有终局事件: {:?}",
        log.since(0)
            .unwrap()
            .iter()
            .map(|e| e.kind.as_str().to_string())
            .collect::<Vec<_>>()
    );
}

/// 租约过期由**注入的时钟**决定，不由墙钟决定。
///
/// `lease_expiry_cancels_and_fsm_preemption_cancels` 验的是同一件事，但它要真的等上
/// 0.3 s（`walk_spec` 的 `lease_ttl_s`）——那段等待既不是判据也不是被测行为，它只是
/// `MonotonicClock` 快进不了。这里手工搭零件，用 `ManualClock` 把时间推过租约，再同步
/// 跑一次扫描（`sweep_leases_now`，生产路径上由 `lease_check_ms` 的定时任务调同一个
/// 函数），于是"过期"这件事在一次 `await` 里就发生完了。
///
/// 时钟从 1 µs 起而不是 0：`sweep_leases` 拿 `last_heartbeat_ns == 0` 当"一次心跳都没
/// 有过"的哨兵，起点为 0 的话"没心跳"和"在 0 时刻心跳过"分不开，这个用例就验不到它
/// 想验的那条路。
#[tokio::test]
async fn advancing_the_injected_clock_past_the_lease_expires_it_at_once() {
    let clock = Arc::new(ManualClock::new(1_000));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let fsm = Arc::new(RobotFsm::new(log.clone(), state.clone()));
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let grace_ms = 500;
    let adm = Admission::new(
        clock.clone(),
        log.clone(),
        state,
        fsm.clone(),
        AdmissionConfig {
            // 定时扫描形同虚设（一分钟一次）：这个用例自己按快门。
            lease_check_ms: 60_000,
            lease_grace_ms: grace_ms,
            recheck_ms: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let _r = adm.register_rule(Arc::new(AllowAll));
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let ttl_s = 2.0;
    let _c = adm
        .capabilities()
        .register(
            serde_json::from_value(json!({
                "name": "creep", "class": "motion", "risk_level": "low",
                "params_schema": {"type": "object"}, "lease_ttl_s": ttl_s
            }))
            .unwrap(),
            Arc::new(UntilCancelled {
                saw_cancel: saw_cancel.clone(),
            }),
        )
        .unwrap();

    let adm2 = adm.clone();
    let running = tokio::spawn(async move { adm2.propose(proposal("creep", json!({}), 1)).await });
    // 同步点：执行真的起跑了（`act/started` 已经在日志里），租约从这一刻算起。等的是
    // 事件，不是时间。
    wait_until("运动起跑", || {
        log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "act/started")
    })
    .await;

    // 时间一步跨过租约 + 宽限期，然后按一次快门。
    clock.advance((ttl_s * 1e9) as u64 + grace_ms * 1_000_000);
    adm.sweep_leases_now().await;

    // 立刻：没有轮询，没有第二次扫描。
    let evs = log.since(0).unwrap();
    let expired: Vec<&oeh_proto::WorldEvent> = evs
        .iter()
        .filter(|e| e.kind.as_str() == "act/lease_expired")
        .collect();
    assert_eq!(expired.len(), 1, "{:?}", kinds(&log, "t"));
    assert_eq!(expired[0].payload["proposal_id"], "t-p1");
    assert_eq!(expired[0].payload["reason"], "lease expired");
    assert_eq!(expired[0].payload["lease_ttl_s"], json!(ttl_s));

    assert_eq!(
        running.await.unwrap(),
        Outcome::Cancelled {
            reason: "lease expired".into()
        }
    );
    assert!(saw_cancel.load(Ordering::SeqCst), "执行器收到了取消");
    // 没有 motion 在跑了，ACTING 交还回去。
    wait_until("FSM 交还 STANDING", || fsm.state() == FsmState::Standing).await;
}

/// 急停的两条路必须指着同一个文件。
///
/// 急停的权威在内核外面（`estop-listener` 写一个标志文件）。内核这边有**两个**读者：
/// `distill-estop` 把它蒸馏成 `estop` 寄存器 + `safety/estop` 事件，`admission` 自己在取
/// 快照时也直接看一眼那个文件（`Admission::estop_engaged` 取或）。两个各有各的默认路径，
/// 于是一份只改了其中一个的 profile 会让另一个继续盯着 `/tmp/oeh_estop`——急停那一刻它
/// 说"没按下"，而写这份 profile 的人完全有理由以为自己已经把路径改好了。
///
/// 所以不一致是**配置错误**，在 boot 当场 fail-closed，而不是等到急停那天才发现。
#[tokio::test]
async fn two_different_estop_flag_paths_fail_the_admission_plugin_at_boot() {
    let tmp = tempfile::tempdir().unwrap();
    let mut r = rows(vec![]);
    for row in &mut r {
        if row.id == "admission" {
            row.config["estop_flag"] = json!(tmp.path().join("a"));
        }
    }
    r.push(EntryRow::new("distill-estop").with_config(json!({"flag_path": tmp.path().join("b")})));
    let k = Kernel::boot(r).unwrap();
    let failed = &k.report().failed;
    let (_, err) = failed
        .iter()
        .find(|(id, _)| id == "admission")
        .unwrap_or_else(|| panic!("admission 应该起不来: {:?}", k.report()));
    assert!(
        err.contains("estop flag mismatch"),
        "错误里得说清是哪两条路径对不上: {err}"
    );

    // `flag_path` 在那儿但不是字符串：说清楚是这个字段坏了，而不是退回默认值再报一个
    // 指着一条没人写过的路径的"不一致"。
    let mut r = rows(vec![]);
    for row in &mut r {
        if row.id == "admission" {
            row.config["estop_flag"] = json!(tmp.path().join("a"));
        }
    }
    r.push(EntryRow::new("distill-estop").with_config(json!({"flag_path": 123})));
    let k = Kernel::boot(r).unwrap();
    let (_, err) = k
        .report()
        .failed
        .iter()
        .find(|(id, _)| id == "admission")
        .unwrap_or_else(|| panic!("admission 应该起不来: {:?}", k.report()));
    assert!(
        err.contains("distill-estop.flag_path is not a string"),
        "{err}"
    );

    // 两条指向同一个文件时照常起来（路径写法不同也算同一个：`./x` 和 `x`）。
    let tmp2 = tempfile::tempdir().unwrap();
    let flag = tmp2.path().join("estop");
    let mut r = rows(vec![]);
    for row in &mut r {
        if row.id == "admission" {
            row.config["estop_flag"] = json!(flag);
        }
    }
    r.push(EntryRow::new("distill-estop").with_config(json!({
        "flag_path": tmp2.path().join(".").join("estop")
    })));
    let k = Kernel::boot(r).unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
}
