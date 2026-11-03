//! 九条内置安全规则的判定用例：每条规则直接构造出来、直接调 `evaluate`，上下文是一份
//! 手搭的蒸馏状态快照（`StateEntry` 的 BTreeMap）——不经过内核，断言的是规则本身的语义。
//!
//! 另外两个用例走真内核：一条规则插件经 `Kernel::boot` 注册进准入服务（提案被
//! `rule-whitelist` 拦下），以及 `distill-estop` 在标志文件出现/消失时发的两条边沿。

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::capability::CapabilitySpec;
use oeh_admission::rules::clamp::ClampRule;
use oeh_admission::rules::dryrun::DryRunRule;
use oeh_admission::rules::estop::EstopRule;
use oeh_admission::rules::fsm::FsmRule;
use oeh_admission::rules::pose::PoseRule;
use oeh_admission::rules::run_mode::{RunMode, RunModeRule};
use oeh_admission::rules::scene::SceneRule;
use oeh_admission::rules::watchdog::WatchdogRule;
use oeh_admission::rules::whitelist::WhitelistRule;
use oeh_admission::verdict::{Rule, RuleCtx, Verdict};
use oeh_distill as _;
use oeh_kernel::fsm::FsmState;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::{Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::state::{StateEntry, StateRegistry};
use oeh_kernel::{Kernel, WorldLog};
use oeh_predict::kinematic::KinematicPredictor;
use oeh_predict::predictor::{ActionStep, PredictedState, Prediction, Predictor, StateRef};
use oeh_proto::WorldEvent;
use serde_json::{json, Value};

// ---------------------------------------------------------------- 脚手架

/// 蒸馏状态里的一条记录。规则读 `value`，`rule-watchdog` 之外的规则都不看 `t_kernel`。
fn entry(v: Value) -> StateEntry {
    StateEntry {
        value: v,
        t_kernel: 0,
        source: "test".into(),
    }
}

fn spec(name: &str, class: &str, risk: &str) -> CapabilitySpec {
    serde_json::from_value(json!({
        "name": name, "class": class, "risk_level": risk,
        "params_schema": {}, "lease_ttl_s": 5.0
    }))
    .unwrap()
}

/// motion / medium：规则链里最"贵"的那一类。
fn walk() -> CapabilitySpec {
    spec("walk", "motion", "medium")
}
/// speech / none：class 门与 estop 门都该放过它。
fn say() -> CapabilitySpec {
    spec("say", "speech", "none")
}
fn gesture() -> CapabilitySpec {
    spec("gesture", "motion", "low")
}
fn recover() -> CapabilitySpec {
    spec("recover", "control", "high")
}

/// 一次判定看到的世界：FSM 态、内核时间、estop、蒸馏状态。
struct Snap {
    fsm: FsmState,
    now_ns: u64,
    estop: bool,
    state: BTreeMap<String, StateEntry>,
}

impl Default for Snap {
    fn default() -> Self {
        Self {
            fsm: FsmState::Standing,
            now_ns: 0,
            estop: false,
            state: BTreeMap::new(),
        }
    }
}

impl Snap {
    fn with(pairs: &[(&str, Value)]) -> Self {
        let mut s = Snap::default();
        for (k, v) in pairs {
            s.state.insert((*k).to_string(), entry(v.clone()));
        }
        s
    }
    fn ctx<'a>(&'a self, spec: &'a CapabilitySpec) -> RuleCtx<'a> {
        RuleCtx {
            now_ns: self.now_ns,
            fsm: self.fsm,
            state: &self.state,
            spec,
            estop_engaged: self.estop,
        }
    }
    /// 空 params + `Allow` 入参：大多数规则不看 params。
    fn eval(&self, rule: &dyn Rule, spec: &CapabilitySpec) -> Verdict {
        rule.evaluate(&mut json!({}), &self.ctx(spec), Verdict::Allow)
    }
    fn eval_params(
        &self,
        rule: &dyn Rule,
        spec: &CapabilitySpec,
        params: &mut Value,
        incoming: Verdict,
    ) -> Verdict {
        rule.evaluate(params, &self.ctx(spec), incoming)
    }
    /// 同 `eval_params`，外带规则留在 trace 里的 detail（`RuleTrace::detail`）。
    fn eval_detail(
        &self,
        rule: &dyn Rule,
        spec: &CapabilitySpec,
        params: &mut Value,
        incoming: Verdict,
    ) -> (Verdict, Option<Value>) {
        rule.evaluate_detailed(params, &self.ctx(spec), incoming)
    }
}

#[track_caller]
fn deny_reason(v: &Verdict) -> String {
    match v {
        Verdict::Deny { reason } => reason.clone(),
        other => panic!("expected Deny, got {other:?}"),
    }
}

#[track_caller]
fn ask_reason(v: &Verdict) -> String {
    match v {
        Verdict::Ask { reason } => reason.clone(),
        other => panic!("expected Ask, got {other:?}"),
    }
}

// ---------------------------------------------------------------- rule-whitelist

#[test]
fn whitelist_allows_listed_and_denies_the_rest() {
    let rule = WhitelistRule {
        allowed: vec!["say".into(), "stop".into()],
    };
    assert_eq!(rule.name(), "rule-whitelist");
    let s = Snap::default();
    assert_eq!(s.eval(&rule, &say()), Verdict::Allow);
    let v = s.eval(&rule, &walk());
    assert!(deny_reason(&v).contains("walk"), "{v:?}");
}

#[test]
fn whitelist_never_loosens_and_an_empty_list_denies() {
    let rule = WhitelistRule {
        allowed: vec!["say".into()],
    };
    let s = Snap::default();
    // 放行 = 原样传回上游判定，不是把它放松成 Allow。
    let incoming = Verdict::Ask {
        reason: "upstream".into(),
    };
    let out = s.eval_params(&rule, &say(), &mut json!({}), incoming);
    assert_eq!(ask_reason(&out), "upstream");
    // 一条都没配 = 什么都不放行（fail-closed），而不是什么都放行。
    let empty = WhitelistRule { allowed: vec![] };
    assert!(matches!(s.eval(&empty, &say()), Verdict::Deny { .. }));
}

// ---------------------------------------------------------------- rule-fsm

#[test]
fn fsm_denies_motion_outside_the_motion_states() {
    let rule = FsmRule;
    assert_eq!(rule.name(), "rule-fsm");
    let boot = Snap {
        fsm: FsmState::Boot,
        ..Snap::default()
    };
    assert_eq!(deny_reason(&boot.eval(&rule, &walk())), "fsm BOOT");
    let estopped = Snap {
        fsm: FsmState::EmergencyStop,
        ..Snap::default()
    };
    assert_eq!(
        deny_reason(&estopped.eval(&rule, &walk())),
        "fsm EMERGENCY_STOP"
    );
    // STANDING / ENGAGED / ACTING 放行；非 motion 的能力任何态都放行。
    for st in [FsmState::Standing, FsmState::Engaged, FsmState::Acting] {
        let s = Snap {
            fsm: st,
            ..Snap::default()
        };
        assert_eq!(s.eval(&rule, &walk()), Verdict::Allow, "{st}");
    }
    assert_eq!(boot.eval(&rule, &say()), Verdict::Allow);
}

#[test]
fn fsm_allows_recover_only_from_emergency_stop_or_fault() {
    let rule = FsmRule;
    for st in [FsmState::EmergencyStop, FsmState::Fault] {
        let s = Snap {
            fsm: st,
            ..Snap::default()
        };
        assert_eq!(s.eval(&rule, &recover()), Verdict::Allow, "{st}");
    }
    for st in [FsmState::Standing, FsmState::Boot, FsmState::Recovering] {
        let s = Snap {
            fsm: st,
            ..Snap::default()
        };
        let v = s.eval(&rule, &recover());
        assert!(deny_reason(&v).contains("recover"), "{st}: {v:?}");
    }
}

// ---------------------------------------------------------------- rule-run-mode

#[test]
fn run_mode_gates_motion_by_mode() {
    let s = Snap::default();
    let observe = RunModeRule {
        mode: RunMode::Observe,
    };
    assert_eq!(observe.name(), "rule-run-mode");
    assert_eq!(deny_reason(&s.eval(&observe, &walk())), "run_mode observe");
    let confirm = RunModeRule {
        mode: RunMode::Confirm,
    };
    assert_eq!(ask_reason(&s.eval(&confirm, &walk())), "run_mode confirm");
    let active = RunModeRule {
        mode: RunMode::Active,
    };
    assert_eq!(s.eval(&active, &walk()), Verdict::Allow);
}

#[test]
fn run_mode_leaves_non_motion_alone_and_never_loosens() {
    let s = Snap::default();
    for mode in [RunMode::Observe, RunMode::Confirm, RunMode::Active] {
        let rule = RunModeRule { mode };
        assert_eq!(s.eval(&rule, &say()), Verdict::Allow, "{mode:?}");
    }
    // confirm 想要 Ask，但上游已经 Deny：不能把它放松回 Ask。
    let confirm = RunModeRule {
        mode: RunMode::Confirm,
    };
    let incoming = Verdict::Deny {
        reason: "upstream".into(),
    };
    let out = s.eval_params(&confirm, &walk(), &mut json!({}), incoming);
    assert_eq!(deny_reason(&out), "upstream");
}

// ---------------------------------------------------------------- rule-watchdog

fn watchdog_rule() -> WatchdogRule {
    WatchdogRule {
        streams: [("lowstate".to_string(), 0.5)].into_iter().collect(),
    }
}

#[test]
fn watchdog_allows_fresh_streams() {
    let rule = watchdog_rule();
    assert_eq!(rule.name(), "rule-watchdog");
    let mut s = Snap::with(&[("last_frame_t_kernel.lowstate", json!(900_000_000u64))]);
    s.now_ns = 1_000_000_000; // 帧年龄 0.1 s < 0.5 s
    assert_eq!(s.eval(&rule, &walk()), Verdict::Allow);
}

#[test]
fn watchdog_denies_stale_or_missing_streams() {
    let rule = watchdog_rule();
    let mut stale = Snap::with(&[("last_frame_t_kernel.lowstate", json!(0u64))]);
    stale.now_ns = 1_000_000_000; // 帧年龄 1.0 s > 0.5 s
    let v = stale.eval(&rule, &walk());
    assert!(deny_reason(&v).contains("lowstate"), "{v:?}");
    // 键不存在（没有蒸馏器在写）和值为 null（流从没来过帧）都算缺失。
    let missing = Snap::default();
    assert!(matches!(missing.eval(&rule, &walk()), Verdict::Deny { .. }));
    let never = Snap::with(&[("last_frame_t_kernel.lowstate", json!(null))]);
    assert!(matches!(never.eval(&rule, &walk()), Verdict::Deny { .. }));
}

#[test]
fn watchdog_only_gates_motion() {
    let rule = watchdog_rule();
    let s = Snap::default(); // 什么流都没有
    assert_eq!(s.eval(&rule, &say()), Verdict::Allow);
}

// ---------------------------------------------------------------- rule-pose

#[test]
fn pose_allows_upright_and_denies_tipping_or_unknown() {
    let rule = PoseRule::default();
    assert_eq!(rule.name(), "rule-pose");
    assert_eq!(rule.min_gravity_z, -0.85);
    let upright = Snap::with(&[("gravity_proj_z", json!(-0.98))]);
    assert_eq!(upright.eval(&rule, &walk()), Verdict::Allow);
    let tipping = Snap::with(&[("gravity_proj_z", json!(-0.5))]);
    let v = tipping.eval(&rule, &walk());
    assert!(deny_reason(&v).contains("gravity_proj_z"), "{v:?}");
    // 没人在写这个寄存器：不知道姿态就不许动。
    let unknown = Snap::default();
    assert!(matches!(unknown.eval(&rule, &walk()), Verdict::Deny { .. }));
    assert_eq!(unknown.eval(&rule, &say()), Verdict::Allow);
}

// ---------------------------------------------------------------- rule-clamp

fn clamp_rule() -> ClampRule {
    ClampRule {
        envelopes: serde_json::from_value(json!({
            "walk": {"vx": [-0.2, 0.3], "vy": [-0.1, 0.1], "wz": [-0.5, 0.5], "duration_s": [0, 2.0]}
        }))
        .unwrap(),
    }
}

#[test]
fn clamp_shrinks_params_into_the_envelope_without_touching_the_verdict() {
    let rule = clamp_rule();
    assert_eq!(rule.name(), "rule-clamp");
    let s = Snap::default();
    let mut params = json!({"vx": 0.9, "duration_s": 5.0, "note": "hi"});
    let incoming = Verdict::Ask {
        reason: "upstream".into(),
    };
    let out = s.eval_params(&rule, &walk(), &mut params, incoming);
    assert_eq!(ask_reason(&out), "upstream", "钳位规则不改判定");
    assert_eq!(params["vx"], json!(0.3));
    assert_eq!(params["duration_s"], json!(2.0));
    assert_eq!(params["note"], json!("hi"), "非数值参数原样保留");
}

#[test]
fn clamp_does_not_add_missing_params_or_touch_other_capabilities() {
    let rule = clamp_rule();
    let s = Snap::default();
    let mut params = json!({"vx": -0.9});
    assert_eq!(
        s.eval_params(&rule, &walk(), &mut params, Verdict::Allow),
        Verdict::Allow
    );
    assert_eq!(params, json!({"vx": -0.2}), "缺的参数不补");
    // 包络里没有的能力：params 一个字节都不动。
    let mut untouched = json!({"vx": 9.0});
    assert_eq!(
        s.eval_params(&rule, &say(), &mut untouched, Verdict::Allow),
        Verdict::Allow
    );
    assert_eq!(untouched, json!({"vx": 9.0}));
}

#[test]
fn clamp_leaves_in_range_params_alone() {
    let rule = clamp_rule();
    let s = Snap::default();
    let mut params = json!({"vx": 0.1, "wz": -0.25});
    s.eval_params(&rule, &walk(), &mut params, Verdict::Allow);
    assert_eq!(params, json!({"vx": 0.1, "wz": -0.25}));
}

// ---------------------------------------------------------------- rule-scene

fn clear_scene() -> Snap {
    Snap::with(&[
        ("clear_path", json!(true)),
        ("nearest_obstacle_m", json!(2.0)),
        ("nearest_person_m", json!(2.0)),
    ])
}

#[test]
fn scene_allows_a_clear_path_with_far_distances() {
    let rule = SceneRule::default();
    assert_eq!(rule.name(), "rule-scene");
    assert_eq!(clear_scene().eval(&rule, &walk()), Verdict::Allow);
    // null 距离 = 无穷远：视野里什么都没有，不是"不知道"。
    let empty_world = Snap::with(&[
        ("clear_path", json!(true)),
        ("nearest_obstacle_m", json!(null)),
        ("nearest_person_m", json!(null)),
    ]);
    assert_eq!(empty_world.eval(&rule, &walk()), Verdict::Allow);
}

#[test]
fn scene_denies_obstacles_people_and_an_unknown_path() {
    let rule = SceneRule::default();
    let blocked = Snap::with(&[
        ("clear_path", json!(false)),
        ("nearest_obstacle_m", json!(2.0)),
        ("nearest_person_m", json!(2.0)),
    ]);
    assert!(matches!(blocked.eval(&rule, &walk()), Verdict::Deny { .. }));
    let obstacle = Snap::with(&[
        ("clear_path", json!(true)),
        ("nearest_obstacle_m", json!(0.3)),
        ("nearest_person_m", json!(2.0)),
    ]);
    let v = obstacle.eval(&rule, &walk());
    assert!(deny_reason(&v).contains("obstacle"), "{v:?}");
    let person = Snap::with(&[
        ("clear_path", json!(true)),
        ("nearest_obstacle_m", json!(2.0)),
        ("nearest_person_m", json!(0.5)),
    ]);
    let v = person.eval(&rule, &walk());
    assert!(deny_reason(&v).contains("person"), "{v:?}");
    // 没有 `clear_path` 键：没有蒸馏器在看地面，不许走。
    let v = Snap::default().eval(&rule, &walk());
    assert!(deny_reason(&v).contains("clear_path"), "{v:?}");
}

#[test]
fn scene_gates_gestures_on_person_distance_only() {
    let rule = SceneRule::default();
    // 手势不看 clear_path，只看人有多近（默认 0.5 m）。
    let close = Snap::with(&[("nearest_person_m", json!(0.3))]);
    let v = close.eval(&rule, &gesture());
    assert!(deny_reason(&v).contains("person"), "{v:?}");
    let far = Snap::with(&[("nearest_person_m", json!(1.0))]);
    assert_eq!(far.eval(&rule, &gesture()), Verdict::Allow);
    // 既不是 motion_caps 也不是 gesture_caps 的能力：透传。
    assert_eq!(Snap::default().eval(&rule, &say()), Verdict::Allow);
}

// ---------------------------------------------------------------- rule-estop

#[test]
fn estop_denies_everything_with_risk_while_engaged() {
    let rule = EstopRule;
    assert_eq!(rule.name(), "rule-estop");
    let engaged = Snap {
        estop: true,
        ..Snap::default()
    };
    assert_eq!(deny_reason(&engaged.eval(&rule, &walk())), "estop engaged");
    assert_eq!(
        deny_reason(&engaged.eval(&rule, &recover())),
        "estop engaged",
        "high 风险的 recover 也一样：解除急停走审批，不走规则链"
    );
    // risk_level none（say / ask_human / stop）照常：急停时还得能说话、能停。
    assert_eq!(engaged.eval(&rule, &say()), Verdict::Allow);
}

#[test]
fn estop_is_transparent_when_not_engaged() {
    let rule = EstopRule;
    let s = Snap::default();
    assert_eq!(s.eval(&rule, &walk()), Verdict::Allow);
    assert_eq!(s.eval(&rule, &say()), Verdict::Allow);
}

// ---------------------------------------------------------------- 内核集成

fn walk_spec_json() -> Value {
    json!({
        "name": "walk",
        "class": "motion",
        "risk_level": "medium",
        "params_schema": {"type": "object", "properties": {"vx": {"type": "number"}}},
        "lease_ttl_s": 1.0
    })
}

fn proposal(cap: &str) -> Proposal {
    Proposal {
        proposal_id: ProposalId("p1".into()),
        brain_id: "fast".into(),
        turn: 1,
        trace_id: "t".into(),
        capability: cap.into(),
        params: json!({"vx": 0.1}),
    }
}

/// 一条规则插件经内核装进准入服务：提案真的被它拦下来，并且 refusal 的 trace 认得出是谁。
#[tokio::test]
async fn rule_plugin_registers_through_the_kernel_and_refuses() {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains"),
        EntryRow::new("admission"),
        EntryRow::new("mock-executor").with_config(json!({"capabilities": [walk_spec_json()]})),
        EntryRow::new("rule-whitelist").with_config(json!({"allowed": ["say"]})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();

    let out = adm.propose(proposal("walk")).await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("walk")),
        "{out:?}"
    );
    let refused = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "act/refused")
        .expect("act/refused");
    assert_eq!(refused.payload["stage"], json!("rules"));
    let trace = refused.payload["trace"].as_array().expect("trace");
    assert!(
        trace
            .iter()
            .any(|t| t["rule"] == json!("rule-whitelist") && t["outgoing"]["verdict"] == "deny"),
        "{trace:?}"
    );
}

// ---------------------------------------------------------------- distill-estop

/// 轮询等一个条件成立（最多 1 s）：等的是状态，不是时间。
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..200 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

fn estop_events(log: &WorldLog) -> Vec<WorldEvent> {
    log.since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "safety/estop")
        .collect()
}

#[tokio::test]
async fn distill_estop_publishes_both_edges_and_the_register() {
    let dir = tempfile::tempdir().unwrap();
    let flag = dir.path().join("oeh_estop");
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("distill-estop").with_config(json!({"flag_path": flag, "period_ms": 10})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();

    // 标志文件不存在：寄存器是 false，但不发事件（没有边沿）。
    wait_until("first tick", || state.get("estop").is_some()).await;
    assert_eq!(state.get("estop"), Some(json!(false)));
    assert!(estop_events(&log).is_empty(), "no flag file, no edge");

    std::fs::write(&flag, b"").unwrap();
    wait_until("engaged", || state.get("estop") == Some(json!(true))).await;
    std::fs::remove_file(&flag).unwrap();
    wait_until("released", || state.get("estop") == Some(json!(false))).await;

    let evs = estop_events(&log);
    let engaged: Vec<Value> = evs.iter().map(|e| e.payload["engaged"].clone()).collect();
    assert_eq!(engaged, vec![json!(true), json!(false)], "两条边沿，各一条");
    assert_eq!(evs[0].collapse_key.as_deref(), Some("estop"));
    assert_eq!(evs[0].salience, 255);
    assert_eq!(evs[0].source, "distill-estop");
}

// ---------------------------------------------------------------- rule-dryrun

/// 默认包络（障碍 0.5 m / 人 1.0 m）+ 保底世界模型。
fn dryrun() -> DryRunRule {
    DryRunRule::new(Arc::new(KinematicPredictor::default()))
}

/// 一个永远报错的预测器：预测失败该走 `Ask`，不该走 `Allow`。
struct BrokenPredictor;
impl Predictor for BrokenPredictor {
    fn name(&self) -> &str {
        "broken"
    }
    fn predict(
        &self,
        _state: &StateRef,
        _actions: &[ActionStep],
        _horizon: Duration,
    ) -> Result<Prediction, String> {
        Err("model is not loaded".into())
    }
}

/// 一个给出 `NaN` 距离的预测器。`NaN < limit` 是假，放它过去就是"看起来很安全"。
struct NanPredictor;
impl Predictor for NanPredictor {
    fn name(&self) -> &str {
        "nan"
    }
    fn predict(
        &self,
        _state: &StateRef,
        _actions: &[ActionStep],
        _horizon: Duration,
    ) -> Result<Prediction, String> {
        Ok(Prediction {
            states: vec![PredictedState {
                t_offset_s: 0.1,
                x: 0.0,
                y: 0.0,
                yaw: 0.0,
                nearest_obstacle_m: Some(f64::NAN),
                nearest_person_m: None,
            }],
            uncertainty: vec![0.05],
        })
    }
}

/// 记下规则递进来的 `StateRef` 与动作，什么也不预测。
#[derive(Default)]
struct RecordingPredictor {
    seen: Mutex<Option<(StateRef, Vec<ActionStep>)>>,
}
impl Predictor for RecordingPredictor {
    fn name(&self) -> &str {
        "recording"
    }
    fn predict(
        &self,
        state: &StateRef,
        actions: &[ActionStep],
        _horizon: Duration,
    ) -> Result<Prediction, String> {
        *self.seen.lock().expect("lock") = Some((state.clone(), actions.to_vec()));
        Ok(Prediction {
            states: Vec::new(),
            uncertainty: Vec::new(),
        })
    }
}

#[track_caller]
fn f64_at(v: &Value, path: &[&str]) -> f64 {
    let mut cur = v;
    for k in path {
        cur = &cur[*k];
    }
    cur.as_f64()
        .unwrap_or_else(|| panic!("{path:?} 不是数字: {v}"))
}

/// 0.8 m 外的障碍：**现在**还过得去（`rule-scene` 的 0.6 m 门都够），但以 0.5 m/s 走
/// 2 s 的预测轨迹会走进 0.5 m 的包络 → 收紧到 `Ask`。留痕进 `detail`，**不进 params**。
#[test]
fn dryrun_asks_when_the_predicted_trajectory_enters_the_envelope() {
    let s = Snap::with(&[
        ("nearest_obstacle_m", json!(0.8)),
        ("nearest_person_m", json!(null)),
    ]);
    let mut params = json!({"vx": 0.5, "duration_s": 2.0});
    let rule = dryrun();
    assert_eq!(rule.name(), "rule-dryrun");
    let (v, detail) = s.eval_detail(&rule, &walk(), &mut params, Verdict::Allow);
    let reason = ask_reason(&v);
    assert!(reason.starts_with("dryrun: "), "{reason}");
    assert!(reason.contains("obstacle"), "{reason}");
    // 交给执行器的那一份参数一个键都没多。
    assert_eq!(
        params,
        json!({"vx": 0.5, "duration_s": 2.0}),
        "留痕不许写进 params"
    );
    // 留痕：这条轨迹最近处是多少。走 1.0 m 之后 0.8 m 的余量被吃光（夹在 0）。
    let d = detail.expect("detail");
    assert!(
        (f64_at(&d, &["dryrun", "min_obstacle_m"]) - 0.0).abs() <= 1e-9,
        "{d}"
    );
    assert_eq!(
        d["dryrun"]["min_person_m"],
        json!(null),
        "不知道的距离不编一个数出来: {d}"
    );
}

/// 5 m 外的障碍：整条轨迹都在包络之外 → 判定原样传下去（Allow 进 Allow 出），留痕照留。
#[test]
fn dryrun_is_transparent_when_the_trajectory_stays_clear() {
    let s = Snap::with(&[
        ("nearest_obstacle_m", json!(5.0)),
        ("nearest_person_m", json!(5.0)),
    ]);
    let mut params = json!({"vx": 0.5, "duration_s": 2.0});
    let (v, detail) = s.eval_detail(&dryrun(), &walk(), &mut params, Verdict::Allow);
    assert_eq!(v, Verdict::Allow);
    assert_eq!(params, json!({"vx": 0.5, "duration_s": 2.0}));
    let d = detail.expect("detail");
    assert!(
        (f64_at(&d, &["dryrun", "min_obstacle_m"]) - 4.0).abs() <= 1e-6,
        "{d}"
    );
}

/// 人的包络（1.0 m）：1.5 m 外的人，走 2 s 之后剩 0.5 m → Ask，理由说的是人。
#[test]
fn dryrun_watches_the_person_envelope_too() {
    let s = Snap::with(&[
        ("nearest_obstacle_m", json!(9.0)),
        ("nearest_person_m", json!(1.5)),
    ]);
    let mut params = json!({"vx": 0.5, "duration_s": 2.0});
    let (v, detail) = s.eval_detail(&dryrun(), &walk(), &mut params, Verdict::Allow);
    let reason = ask_reason(&v);
    assert!(reason.contains("person"), "{reason}");
    let d = detail.expect("detail");
    assert!(
        (f64_at(&d, &["dryrun", "min_person_m"]) - 0.5).abs() <= 1e-6,
        "{d}"
    );
}

/// 非 motion 的能力：不预演、不留痕、不改判定。
#[test]
fn dryrun_only_looks_at_motion() {
    let s = Snap::with(&[("nearest_obstacle_m", json!(0.1))]);
    let mut params = json!({"text": "hi"});
    let (v, detail) = s.eval_detail(&dryrun(), &say(), &mut params, Verdict::Allow);
    assert_eq!(v, Verdict::Allow);
    assert_eq!(params, json!({"text": "hi"}), "params 一个键都不该多");
    assert!(detail.is_none(), "{detail:?}");
}

/// 预测失败 = 不知道接下来会怎样 = 问人。
#[test]
fn a_broken_predictor_asks_instead_of_allowing() {
    let s = Snap::with(&[("nearest_obstacle_m", json!(5.0))]);
    let rule = DryRunRule::new(Arc::new(BrokenPredictor));
    let mut params = json!({"vx": 0.5, "duration_s": 2.0});
    let (v, detail) = s.eval_detail(&rule, &walk(), &mut params, Verdict::Allow);
    assert_eq!(
        ask_reason(&v),
        "dryrun: predictor error",
        "预测失败只有一个理由"
    );
    assert!(detail.is_none(), "没算出什么就没什么可留的: {detail:?}");
}

/// 预测出 `NaN` 距离的模型一样是问人：`NaN < limit` 是假，放过去就是"看起来很安全"。
#[test]
fn a_predictor_that_returns_nan_distances_asks_instead_of_allowing() {
    let s = Snap::with(&[("nearest_obstacle_m", json!(5.0))]);
    let rule = DryRunRule::new(Arc::new(NanPredictor));
    let mut params = json!({"vx": 0.5, "duration_s": 2.0});
    let (v, detail) = s.eval_detail(&rule, &walk(), &mut params, Verdict::Allow);
    assert_eq!(ask_reason(&v), "dryrun: invalid state");
    assert!(detail.is_none(), "{detail:?}");
}

/// `StateRef.v*` 是机器人**现在**的速度（蒸馏状态里有就用，没有就是 0）；提案要的那一份
/// 只进 `ActionStep`。混起来的话，一个从 `state.v*` 起算积分的预测器会把指令速度算两遍。
#[test]
fn the_state_ref_carries_the_current_velocity_not_the_proposed_one() {
    let spy = Arc::new(RecordingPredictor::default());
    let rule = DryRunRule::new(spy.clone());
    let s = Snap::with(&[("nearest_obstacle_m", json!(5.0)), ("vx", json!(0.2))]);
    let mut params = json!({"vx": 0.5, "wz": -0.1, "duration_s": 2.0});
    let (v, _) = s.eval_detail(&rule, &walk(), &mut params, Verdict::Allow);
    assert_eq!(v, Verdict::Allow);
    let seen = spy
        .seen
        .lock()
        .expect("lock")
        .clone()
        .expect("predict 被调用");
    let (state, actions) = seen;
    assert_eq!(state.vx, 0.2, "现在的速度来自蒸馏状态");
    assert_eq!(state.vy, 0.0);
    assert_eq!(state.wz, 0.0, "状态里没有 wz 就是 0，不是提案里的 -0.1");
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].vx, 0.5, "提案要的速度只进动作");
    assert_eq!(actions[0].wz, -0.1);
    assert_eq!(actions[0].duration_s, 2.0);
}

/// 只收紧：上游的 `Ask` 理由不被盖掉，上游的 `Deny` 不被放松成 `Ask`。
#[test]
fn dryrun_never_loosens() {
    let s = Snap::with(&[("nearest_obstacle_m", json!(0.8))]);
    let mut params = json!({"vx": 0.5, "duration_s": 2.0});
    let upstream = Verdict::Ask {
        reason: "upstream".into(),
    };
    assert_eq!(
        s.eval_params(&dryrun(), &walk(), &mut params, upstream.clone()),
        upstream,
        "同紧度留上游的理由"
    );
    let deny = Verdict::Deny {
        reason: "upstream".into(),
    };
    assert_eq!(
        s.eval_params(&dryrun(), &walk(), &mut params, deny.clone()),
        deny
    );
}

/// fail-closed 靠 profile：一份把 `rule-dryrun` 写进 `admission.required_rules`、却没有
/// `predict-kinematic` 行的 profile，每一次 motion 提案都过不去。
///
/// 机制是现成的那一套，不是内核改动：缺了 `predictor`，这条规则停在 `Pending`（boot
/// 报告里看得见），链上没有它 → `evaluate_chain` 的起始判定是
/// `Ask{"missing required rules: rule-dryrun"}` → 走审批 → 没装审批服务 = 拒绝。
/// 注意 `Kernel::boot` 本身**不会**失败：`report().pending` 是这件事唯一的启动期信号，
/// 真正挡住动作的是 `required_rules`。
#[tokio::test]
async fn a_profile_that_requires_dryrun_without_a_predictor_refuses_every_motion() {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains"),
        EntryRow::new("admission").with_config(json!({"required_rules": ["rule-dryrun"]})),
        EntryRow::new("mock-executor").with_config(json!({"capabilities": [walk_spec_json()]})),
        EntryRow::new("rule-whitelist").with_config(json!({"allowed": ["walk"]})),
        EntryRow::new("rule-dryrun"),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let pending = &k.report().pending;
    assert!(
        pending.iter().any(
            |(key, missing)| key == "rule-dryrun" && missing.iter().any(|s| s.0 == "predictor")
        ),
        "rule-dryrun 该因为缺 predictor 停在 Pending: {pending:?}"
    );
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let out = adm.propose(proposal("walk")).await;
    assert!(
        matches!(out, Outcome::Refused { ref reason } if reason.contains("rule-dryrun")),
        "{out:?}"
    );
    let evs = log.since(0).unwrap();
    let ask = evs
        .iter()
        .find(|e| e.kind.as_str() == "act/ask")
        .expect("act/ask");
    assert_eq!(
        ask.payload["reason"],
        json!("missing required rules: rule-dryrun")
    );
    assert!(
        !evs.iter().any(|e| e.kind.as_str() == "act/started"),
        "没有预测器就不该有任何动作起跑"
    );
}

/// 装上 `predict-kinematic` 之后这条规则就上链了——**哪怕它排在 profile 里更靠后**：
/// `FiberTree::settle()` 是一个不动点循环，provide/inject 的解析和行的先后无关。
///
/// 顺带钉住留痕那条路：`dryrun` 出现在 `act/admitted.verdict_trace` 的 `detail` 里，
/// 而整段日志里**没有** `act/clamped`（留痕没有碰 params）。
#[tokio::test]
async fn the_predictor_row_may_come_after_the_rule_that_injects_it() {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains"),
        EntryRow::new("admission").with_config(json!({"required_rules": ["rule-dryrun"]})),
        EntryRow::new("mock-executor").with_config(json!({"capabilities": [walk_spec_json()]})),
        EntryRow::new("rule-dryrun"),
        // 消费者在前、提供者在后：settle() 的第二轮才把 rule-dryrun 激活。
        EntryRow::new("predict-kinematic"),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    assert!(
        k.report().pending.is_empty(),
        "行的先后不该影响 provide/inject 的解析: {:?}",
        k.report().pending
    );
    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    // 世界里一个障碍都没蒸出来（距离未知）：dryrun 对"不知道"不发言，链因此不因为它
    // 而停下——这次提案被准入，证明规则真的在链上跑过了。
    let out = adm.propose(proposal("walk")).await;
    assert!(
        !matches!(out, Outcome::Refused { .. }),
        "没有障碍的世界里 dryrun 不该拦: {out:?}"
    );
    let evs = log.since(0).unwrap();
    let admitted = evs
        .iter()
        .find(|e| e.kind.as_str() == "act/admitted")
        .expect("act/admitted");
    let trace = admitted.payload["verdict_trace"]
        .as_array()
        .expect("verdict_trace");
    let rules: Vec<&str> = trace
        .iter()
        .map(|t| t["rule"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(rules, vec!["rule-dryrun"], "{rules:?}");
    assert!(
        trace[0]["detail"]["dryrun"].is_object(),
        "留痕该出现在 trace 的 detail 里: {}",
        admitted.payload["verdict_trace"]
    );
    assert!(
        !evs.iter().any(|e| e.kind.as_str() == "act/clamped"),
        "留痕没有碰 params，不该有 act/clamped: {:?}",
        evs.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>()
    );
    // 这台机器人的 FSM 还在 BOOT，所以准入之后没有 `act/started`——这个用例要验的是
    // 规则上没上链、留痕走的哪条路，起不起跑是 `plan2b_e2e` 的事。
}
