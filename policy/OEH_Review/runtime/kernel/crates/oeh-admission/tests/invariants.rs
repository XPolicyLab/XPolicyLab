//! 规格 §11 里和准入链有关的四条不变式，一条一个用例。
//!
//! 和 `tests/admission.rs` 一样用 `Kernel::boot` 把真实的插件图拼起来，然后直接调
//! `Admission::propose`——不经过 brains 的 turn 循环，断言的是准入链本身。
//!
//! - **#1** 每条 `act/started` 之前都有同 trace 的 `act/admitted`，且它的 `verdict_trace`
//!   逐步单调（`outgoing.rank >= incoming.rank`）：规则只能收紧，起跑必须有准入背书。
//! - **#3** Provider 消失 → 能力面收缩：注销那条能力注册之后，下一次提案连词汇这一关
//!   都过不去（`act/refused{stage:"vocabulary"}`）；注销一条**必需规则**之后，下一次
//!   提案不再是 Allow（链的初始判定变成 `Ask`，`act/ask.reason` 点名缺的那条规则，
//!   没人放行就是拒）。
//! - **#5** 想把判定放松的规则按篡改处理：`safety/tamper_attempt` + Refused，审批放行
//!   也救不回来。
//! - **#6** 急停标志文件在，有风险的能力一律拒；`stop`（risk none）照常执行。

use std::sync::Arc;

use oeh_admission as _;
use oeh_admission::admission::Admission;
use oeh_admission::approval::AutoApproval;
use oeh_admission::capability::{
    CapabilityClass, CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor,
};
use oeh_admission::rules::estop::EstopRule;
use oeh_admission::rules::fsm::FsmRule;
use oeh_admission::rules::whitelist::WhitelistRule;
use oeh_admission::verdict::{Rule, RuleCtx, Verdict};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::{Kernel, WorldLog};
use oeh_proto::WorldEvent;
use serde_json::{json, Value};

// ---------------------------------------------------------------- 脚手架

fn walk_spec() -> Value {
    json!({
        "name": "walk",
        "class": "motion",
        "risk_level": "medium",
        "params_schema": {"type": "object", "properties": {"vx": {"type": "number"}}},
        "lease_ttl_s": 0.5
    })
}

/// `tests/admission.rs::rows()` 那一套行，外加这个用例自己的 `admission` 配置。
fn rows(admission: Value) -> Vec<EntryRow> {
    vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains"),
        EntryRow::new("admission").with_config(admission),
        EntryRow::new("mock-executor").with_config(
            json!({"capabilities": [walk_spec()], "duration_s": 0.05, "heartbeat_ms": 20}),
        ),
    ]
}

fn admission_cfg(required: &[&str]) -> Value {
    json!({"required_rules": required, "ask_timeout_ms": 200, "lease_check_ms": 20})
}

/// 启动 + 拿服务 + 把 FSM 推到 STANDING（motion 的起点）。
///
/// `Kernel` 一并返回：它一 drop，fiber 树就把插件全撤了（准入服务连着任务一起没）。
/// 用例把它绑在 `_k` 上留到作用域结束。这些用例不发 Op，所以不跑 `Kernel::run`。
fn boot(admission: Value) -> (Kernel, Arc<Admission>, Arc<WorldLog>) {
    let k = Kernel::boot(rows(admission)).unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    (k, adm, log)
}

fn proposal(cap: &str, params: Value, trace: &str) -> Proposal {
    Proposal {
        proposal_id: ProposalId(format!("{trace}-p1")),
        brain_id: "fast".into(),
        turn: 1,
        trace_id: trace.into(),
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

fn find(log: &WorldLog, trace: &str, kind: &str) -> WorldEvent {
    log.since(0)
        .unwrap()
        .iter()
        .find(|e| e.trace_id.as_deref() == Some(trace) && e.kind.as_str() == kind)
        .unwrap_or_else(|| panic!("trace {trace} 里没有 {kind}: {:?}", kinds(log, trace)))
        .clone()
}

/// 判定的紧度：allow < ask < deny。从序列化后的标签读，不碰 `Verdict` 本身——不变式
/// 断言的是**日志里**那份 trace，读日志的人只有这个标签。
fn rank(v: &Value) -> u8 {
    match v["verdict"].as_str() {
        Some("allow") => 0,
        Some("ask") => 1,
        Some("deny") => 2,
        other => panic!("unknown verdict tag {other:?} in {v}"),
    }
}

/// 一条放行一切的规则：链非空才有 Allow 的初始判定。
struct AllowAll;
impl Rule for AllowAll {
    fn name(&self) -> &str {
        "allow-all"
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
        i
    }
}

// ---------------------------------------------------------------- #1

/// 不变式 #1：没有准入就没有起跑，而准入的 `verdict_trace` 每一步都只会更紧。
#[tokio::test]
async fn every_started_has_an_earlier_admitted_with_a_monotone_trace() {
    let (_k, adm, log) = boot(admission_cfg(&[]));

    /// motion 一律要人确认：把 Allow 收紧成 Ask，是合法的一步（rank 0 → 1）。
    struct AskForMotion;
    impl Rule for AskForMotion {
        fn name(&self) -> &str {
            "ask-for-motion"
        }
        fn evaluate(&self, _p: &mut Value, c: &RuleCtx, i: Verdict) -> Verdict {
            if c.spec.class == CapabilityClass::Motion {
                return Verdict::Ask {
                    reason: "motion needs a human".into(),
                };
            }
            i
        }
    }
    let _r1 = adm.register_rule(Arc::new(AllowAll));
    let _r2 = adm.register_rule(Arc::new(AskForMotion));
    let _a = adm.set_approval(Arc::new(AutoApproval { allow: true }));

    // 三条不同的 trace：一条直接 Allow 的 speech，两条经审批放行的 motion。
    let say = adm
        .propose(proposal("say", json!({"text": "hi"}), "t1"))
        .await;
    assert!(matches!(say, Outcome::Executed { .. }), "{say:?}");
    for (n, vx) in [(2, 0.1), (3, 0.2)] {
        let out = adm
            .propose(proposal("walk", json!({ "vx": vx }), &format!("t{n}")))
            .await;
        assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    }

    let evs = log.since(0).unwrap();
    let started: Vec<&WorldEvent> = evs
        .iter()
        .filter(|e| e.kind.as_str() == "act/started")
        .collect();
    assert_eq!(started.len(), 3, "三次执行都起跑了");
    for s in started {
        let trace = s.trace_id.as_deref().expect("act/* 必须带 trace_id");
        let admitted = evs
            .iter()
            .find(|e| {
                e.kind.as_str() == "act/admitted"
                    && e.trace_id.as_deref() == Some(trace)
                    && e.seq < s.seq
            })
            .unwrap_or_else(|| panic!("act/started({trace}) 之前没有同 trace 的 act/admitted"));
        let steps = admitted.payload["verdict_trace"].as_array().unwrap();
        assert!(!steps.is_empty(), "空的 verdict_trace 证明不了任何事");
        for step in steps {
            assert!(
                rank(&step["outgoing"]) >= rank(&step["incoming"]),
                "规则 {} 把判定放松了: {step}",
                step["rule"]
            );
        }
    }
}

// ---------------------------------------------------------------- #3

/// 不变式 #3 的两半。
///
/// **Provider 消失**（规格 §11.3 说的那一半）：能力注册表就是能力面。提供 `crouch` 的
/// 那个 Provider 一撤，注册跟着撤，下一次提案连词汇这一关都过不去——`Allow` 无从谈起。
///
/// **必需规则消失**：链的初始判定变成 `Ask{missing required rules: ..}`，没有审批放行
/// 就是拒。配错规则集的结果是"什么都要人来确认"，不是"什么都放行"。
#[tokio::test]
async fn a_vanished_provider_or_required_rule_stops_the_next_proposal_from_being_allowed() {
    let (_k, adm, log) = boot(admission_cfg(&["rule-whitelist", "rule-fsm"]));
    let _w = adm.register_rule(Arc::new(WhitelistRule {
        allowed: vec!["walk".into(), "crouch".into()],
    }));
    let drop_fsm_rule = adm.register_rule(Arc::new(FsmRule));
    // 无人值守：`Ask` 的结论是拒绝。第一次提案根本走不到这里——它是 Allow。
    let _a = adm.set_approval(Arc::new(AutoApproval { allow: false }));

    let out = adm
        .propose(proposal("walk", json!({"vx": 0.1}), "t1"))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    assert_eq!(
        kinds(&log, "t1"),
        vec!["act/proposed", "act/admitted", "act/started", "act/result"],
        "必需规则齐全 → 直接 Allow，不问人"
    );

    // Provider 消失：`crouch` 本来跑得好好的，注册一撤，能力面就少了这一条。
    struct Nop;
    impl Executor for Nop {
        fn execute(&self, _c: ExecutionCall, _x: ExecCtx) -> BoxFuture<ExecOutcome> {
            Box::pin(async {
                ExecOutcome {
                    ok: true,
                    result: json!({}),
                    error: None,
                }
            })
        }
    }
    let spec: CapabilitySpec = serde_json::from_value(json!({
        "name": "crouch", "class": "motion", "risk_level": "low",
        "params_schema": {}, "lease_ttl_s": 1.0
    }))
    .unwrap();
    let drop_crouch = adm.capabilities().register(spec, Arc::new(Nop)).unwrap();
    let out = adm.propose(proposal("crouch", json!({}), "t3")).await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");

    drop_crouch();
    assert!(
        !adm.capabilities().list().iter().any(|s| s.name == "crouch"),
        "撤销之后能力面里不该还有它"
    );
    let out = adm.propose(proposal("crouch", json!({}), "t4")).await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("unknown capability")),
        "{out:?}"
    );
    assert_eq!(
        kinds(&log, "t4"),
        vec!["act/proposed", "act/refused"],
        "词汇这一关就挡下了，规则链根本不该跑"
    );
    assert_eq!(
        find(&log, "t4", "act/refused").payload["stage"],
        "vocabulary"
    );

    drop_fsm_rule();

    let out = adm
        .propose(proposal("walk", json!({"vx": 0.1}), "t2"))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    assert_eq!(
        kinds(&log, "t2"),
        vec!["act/proposed", "act/ask", "act/decision", "act/refused"]
    );
    let ask = find(&log, "t2", "act/ask");
    let reason = ask.payload["reason"].as_str().unwrap();
    assert!(reason.contains("rule-fsm"), "{reason}");
    assert_eq!(find(&log, "t2", "act/refused").payload["stage"], "approval");
}

// ---------------------------------------------------------------- #5

/// 不变式 #5：规则不能放水。一条从 `Ask` 里返回 `Allow` 的规则被记成篡改
/// （`safety/tamper_attempt`），实际生效的判定是 Deny——哪怕审批那头是"全放行"。
#[tokio::test]
async fn a_rule_that_loosens_the_verdict_is_tampering_and_refused() {
    struct Loosen;
    impl Rule for Loosen {
        fn name(&self) -> &str {
            "loosen"
        }
        fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
            Verdict::Allow
        }
    }

    // 少一条必需规则 → 初始判定是 Ask；`loosen` 想把它变回 Allow。
    let (_k, adm, log) = boot(admission_cfg(&["rule-whitelist", "rule-fsm"]));
    let _w = adm.register_rule(Arc::new(WhitelistRule {
        allowed: vec!["walk".into()],
    }));
    let _l = adm.register_rule(Arc::new(Loosen));
    let _a = adm.set_approval(Arc::new(AutoApproval { allow: true }));

    let out = adm
        .propose(proposal("walk", json!({"vx": 0.1}), "t1"))
        .await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("loosen")),
        "{out:?}"
    );
    assert_eq!(
        kinds(&log, "t1"),
        vec!["act/proposed", "safety/tamper_attempt", "act/refused"],
        "篡改直接变 Deny：审批那一步根本不该发生"
    );
    let tamper = find(&log, "t1", "safety/tamper_attempt");
    assert_eq!(tamper.payload["rule"], "loosen");
    assert_eq!(tamper.payload["incoming"]["verdict"], "ask");
    assert_eq!(tamper.payload["outgoing"]["verdict"], "deny");
    assert_eq!(tamper.salience, 255);

    let refused = find(&log, "t1", "act/refused");
    assert_eq!(refused.payload["stage"], "rules");
    let step = refused.payload["trace"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["rule"] == "loosen")
        .unwrap();
    assert_eq!(step["tamper"], true);
    assert_eq!(step["outgoing"]["verdict"], "deny");
}

// ---------------------------------------------------------------- #6

/// 不变式 #6：急停标志文件一出现，有风险的能力一律拒（`rule-estop`）；`stop` 是 risk
/// none 的控制能力，急停期间还得能再按一次停，所以它照常执行。
#[tokio::test]
async fn the_estop_flag_denies_risky_capabilities_but_not_stop() {
    let dir = tempfile::tempdir().unwrap();
    let flag = dir.path().join("oeh_estop");
    let mut cfg = admission_cfg(&["rule-whitelist", "rule-estop"]);
    cfg["estop_flag"] = json!(flag);
    let (_k, adm, log) = boot(cfg);
    let _w = adm.register_rule(Arc::new(WhitelistRule {
        allowed: vec!["walk".into(), "stop".into()],
    }));
    let _e = adm.register_rule(Arc::new(EstopRule));

    // 标志文件还不在：walk 照常跑。
    let out = adm
        .propose(proposal("walk", json!({"vx": 0.1}), "t1"))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");

    std::fs::write(&flag, b"").unwrap();

    let out = adm
        .propose(proposal("walk", json!({"vx": 0.1}), "t2"))
        .await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason == "estop engaged"),
        "{out:?}"
    );
    let refused = find(&log, "t2", "act/refused");
    assert_eq!(refused.payload["stage"], "rules");
    let step = refused.payload["trace"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["rule"] == "rule-estop")
        .unwrap();
    assert_eq!(
        step["outgoing"],
        json!({"verdict": "deny", "reason": "estop engaged"})
    );
    assert!(!kinds(&log, "t2").contains(&"act/started".to_string()));

    // `stop` 不带风险：急停期间照常执行。
    let out = adm.propose(proposal("stop", json!({}), "t3")).await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    assert_eq!(
        kinds(&log, "t3"),
        vec!["act/proposed", "act/admitted", "act/started", "act/result"]
    );

    // 标志文件撤掉，但 `distill-estop` 写下的 `estop` 寄存器还是 true：一样算按下。
    // 两条路（`admission.estop_flag` 与 `distill-estop.flag_path`）各有各的默认值，
    // 只改一个的 profile 不该悄悄把急停这条链解掉——寄存器和文件取或。
    std::fs::remove_file(&flag).unwrap();
    let state = _k
        .registry()
        .get::<oeh_kernel::state::StateRegistry>(&"state".into())
        .unwrap();
    state.set("estop", json!(true), "distill-estop");
    let out = adm
        .propose(proposal("walk", json!({"vx": 0.1}), "t4"))
        .await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason == "estop engaged"),
        "{out:?}"
    );
}
