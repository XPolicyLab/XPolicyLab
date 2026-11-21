//! RoboDojo ARX-X5 操作路线那六条规则的判定用例。
//!
//! 大部分用例直接构造规则、直接调 `evaluate`，上下文是一份手搭的状态快照（`StateEntry`
//! 的 BTreeMap）——断言的是规则本身的语义，不经过内核。文件末尾有一个走真内核的链路
//! 用例：六条规则一起上链，提案真的被夹、被拒、被放行。
//!
//! 三条贯穿全文的纪律（S1 合约 §3、§10）：
//!
//! - **只断言 trace 里"有"某条规则，不断言它"拥有"理由。** 链上同紧度留上游的理由
//!   （`rules::tighten`），而账本未知时 `rule-carry-speed` 和 `rule-hold-no-rotation`
//!   都会 Deny——谁先谁后是注册顺序，是实现细节，不是承诺。
//! - **浮点断言带容差**，不拿 `serde_json::Value` 去比一个算出来的数。
//! - **等条件，不睡固定时间。**

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::capability::{CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor};
use oeh_admission::rules::arx_mode::ArxModeRule;
use oeh_admission::rules::carry_speed::CarrySpeedRule;
use oeh_admission::rules::finish_gate::FinishGateRule;
use oeh_admission::rules::hold_no_rotation::HoldNoRotationRule;
use oeh_admission::rules::keepout::KeepoutRule;
use oeh_admission::rules::step_reserve::StepReserveRule;
use oeh_admission::verdict::{Rule, RuleCtx, Verdict};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::state::{StateEntry, StateRegistry};
use oeh_kernel::{Kernel, WorldLog};
use serde_json::{json, Value};

// ---------------------------------------------------------------- 脚手架

fn entry(v: Value) -> StateEntry {
    StateEntry {
        value: v,
        t_kernel: 0,
        source: "test".into(),
    }
}

/// `CapabilitySpec` 没有 `Default`，一律经这里构造。
fn spec(name: &str, class: &str, risk: &str) -> CapabilitySpec {
    serde_json::from_value(json!({
        "name": name, "class": class, "risk_level": risk,
        "params_schema": {}, "lease_ttl_s": 5.0
    }))
    .unwrap()
}

fn move_relative() -> CapabilitySpec {
    spec("arx_x5_move_relative", "motion", "high")
}
fn move_eef() -> CapabilitySpec {
    spec("arx_x5_move_eef", "motion", "high")
}
fn wait_cap() -> CapabilitySpec {
    spec("arx_x5_wait", "motion", "low")
}
fn set_gripper() -> CapabilitySpec {
    spec("arx_x5_set_gripper", "motion", "medium")
}
fn return_to_start() -> CapabilitySpec {
    spec("arx_x5_return_to_start", "motion", "medium")
}
fn reset_cap() -> CapabilitySpec {
    spec("arx_x5_reset", "motion", "high")
}
fn finish_cap() -> CapabilitySpec {
    spec("arx_x5_finish", "control", "low")
}
/// 同名但是 query：用来证明 class 门真的在那儿。
fn move_relative_as_query() -> CapabilitySpec {
    spec("arx_x5_move_relative", "query", "none")
}

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

fn ask(reason: &str) -> Verdict {
    Verdict::Ask {
        reason: reason.into(),
    }
}
fn denied(reason: &str) -> Verdict {
    Verdict::Deny {
        reason: reason.into(),
    }
}

/// 一条臂夹着东西的账本。
fn holding_left() -> Vec<(&'static str, Value)> {
    vec![
        ("holding.left", json!(true)),
        ("holding.right", json!(false)),
    ]
}
fn holding_neither() -> Vec<(&'static str, Value)> {
    vec![
        ("holding.left", json!(false)),
        ("holding.right", json!(false)),
    ]
}

/// 取出 params 里某个数，断言用。
#[track_caller]
fn num(params: &Value, pointer: &str) -> f64 {
    params
        .pointer(pointer)
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| panic!("{pointer} is not a number in {params}"))
}

/// 每条规则都要过的一组"不许放松"：Ask 和 Deny 的上游理由原样活下来。
#[track_caller]
fn never_loosens(rule: &dyn Rule, snap: &Snap, spec: &CapabilitySpec, params: Value) {
    let mut p = params.clone();
    let v = snap.eval_params(rule, spec, &mut p, ask("upstream"));
    assert!(
        v.rank() >= 1,
        "{}: Ask 进来不该变成 Allow: {v:?}",
        rule.name()
    );
    if v.rank() == 1 {
        assert_eq!(ask_reason(&v), "upstream", "同紧度该留上游的理由");
    }
    let mut p = params;
    let v = snap.eval_params(rule, spec, &mut p, denied("upstream"));
    assert_eq!(
        deny_reason(&v),
        "upstream",
        "{}: Deny 进来永远是上游的理由",
        rule.name()
    );
}

// ================================================================ rule-carry-speed

#[test]
fn carry_speed_has_a_name() {
    assert_eq!(CarrySpeedRule::default().name(), "rule-carry-speed");
}

#[test]
fn carry_speed_clamps_a_full_speed_move_while_holding() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"speed": 1.0, "left": {"dz": 0.01}});
    let v = snap.eval_params(
        &CarrySpeedRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow, "夹参数的规则不动判定");
    assert!((num(&params, "/speed") - 0.4).abs() < 1e-12, "{params}");
    assert!(
        (num(&params, "/left/dz") - 0.01).abs() < 1e-12,
        "别的键一个字节都不动"
    );
}

#[test]
fn carry_speed_does_not_invent_a_speed_that_the_brain_never_sent() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"left": {"dz": 0.01}});
    let before = params.clone();
    let v = snap.eval_params(
        &CarrySpeedRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow);
    assert_eq!(params, before, "默认值归执行器管，不归规则造");
}

#[test]
fn carry_speed_leaves_an_already_slow_move_alone() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"speed": 0.3});
    let before = params.clone();
    assert_eq!(
        snap.eval_params(
            &CarrySpeedRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow
        ),
        Verdict::Allow
    );
    assert_eq!(params, before);
}

#[test]
fn carry_speed_does_nothing_with_empty_hands() {
    let snap = Snap::with(&holding_neither());
    let mut params = json!({"speed": 1.0});
    let before = params.clone();
    assert_eq!(
        snap.eval_params(
            &CarrySpeedRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow
        ),
        Verdict::Allow
    );
    assert_eq!(params, before, "空手全速是允许的");
}

#[test]
fn carry_speed_denies_every_shape_of_unknown_ledger() {
    // 键不存在 / `null` / 类型不对 / NaN（序列化成 null）。
    for state in [
        vec![("holding.right", json!(false))],
        vec![
            ("holding.left", json!(null)),
            ("holding.right", json!(false)),
        ],
        vec![
            ("holding.left", json!("yes")),
            ("holding.right", json!(false)),
        ],
        vec![
            ("holding.left", json!(f64::NAN)),
            ("holding.right", json!(false)),
        ],
        vec![],
    ] {
        let snap = Snap::with(&state);
        let mut params = json!({"speed": 1.0});
        let before = params.clone();
        let v = snap.eval_params(
            &CarrySpeedRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow,
        );
        assert!(
            deny_reason(&v).contains("holding."),
            "{state:?} 该 fail-closed: {v:?}"
        );
        assert_eq!(params, before, "Deny 的时候不该改参数");
    }
}

#[test]
fn carry_speed_ignores_other_capabilities_and_other_classes() {
    let snap = Snap::with(&holding_left());
    for spec in [set_gripper(), wait_cap(), move_relative_as_query()] {
        let mut params = json!({"speed": 1.0});
        let before = params.clone();
        let v = snap.eval_params(
            &CarrySpeedRule::default(),
            &spec,
            &mut params,
            Verdict::Allow,
        );
        assert_eq!(v, Verdict::Allow, "{}", spec.name);
        assert_eq!(params, before, "{} 的参数一个字节都不该动", spec.name);
    }
}

#[test]
fn carry_speed_never_loosens() {
    never_loosens(
        &CarrySpeedRule::default(),
        &Snap::with(&holding_left()),
        &move_relative(),
        json!({"speed": 1.0}),
    );
    // 账本未知那条路同样不许放松。
    never_loosens(
        &CarrySpeedRule::default(),
        &Snap::default(),
        &move_relative(),
        json!({"speed": 1.0}),
    );
}

// ================================================================ rule-hold-no-rotation

#[test]
fn hold_no_rotation_has_a_name() {
    assert_eq!(
        HoldNoRotationRule::default().name(),
        "rule-hold-no-rotation"
    );
}

#[test]
fn hold_no_rotation_scales_a_ten_centimetre_step_to_three() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"left": {"dz": 0.10}});
    let (v, detail) = snap.eval_detail(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow);
    assert!((num(&params, "/left/dz") - 0.03).abs() < 1e-12, "{params}");
    let d = detail.expect("detail");
    assert!(
        (d["arms"][0]["translation_m"].as_f64().unwrap() - 0.10).abs() < 1e-12,
        "留痕记的是夹之前看到的量: {d}"
    );
    assert!(
        params.pointer("/left/dx").is_none() && params.pointer("/left/dy").is_none(),
        "只写回脑用过的键: {params}"
    );
}

#[test]
fn hold_no_rotation_scales_a_mixed_axis_step_to_exactly_three_centimetres() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"left": {"dx": 0.06, "dy": 0.08}});
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow);
    let dx = num(&params, "/left/dx");
    let dy = num(&params, "/left/dy");
    assert!((dx.hypot(dy) - 0.03).abs() < 1e-12, "{params}");
    assert!((dx / dy - 0.75).abs() < 1e-12, "方向不变");
}

#[test]
fn hold_no_rotation_denies_a_wrist_turn_and_leaves_params_untouched() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"left": {"dyaw_deg": 10.0}});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("rotation"), "{v:?}");
    assert_eq!(params, before);
}

/// 先 Deny 再 clamp 的两遍结构：一次同时越界的提案**一个参数都不许改**。
///
/// `evaluate_chain` 只在入参判定已经是 Deny 时才撤销改动，所以"先夹再自己 Deny"会
/// 留下一条 `act/clamped`，记录一步根本没跑过的动作。
#[test]
fn hold_no_rotation_runs_the_deny_pass_before_the_clamp_pass() {
    let snap = Snap::with(&holding_left());
    let mut params = json!({"left": {"dz": 0.10, "dyaw_deg": 10.0}});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("rotation"), "{v:?}");
    assert_eq!(params, before, "Deny 的那一趟不许留下任何 clamp 痕迹");
}

#[test]
fn hold_no_rotation_denies_a_rotvec_over_the_threshold() {
    let snap = Snap::with(&holding_left());
    // ‖[0.06, 0, 0]‖ = 0.06 > 0.05
    let mut params = json!({"left": {"drot_rotvec": [0.06, 0.0, 0.0]}});
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("rotation"), "{v:?}");
    // 0.04 rad 在阈值内。
    let mut ok = json!({"left": {"drot_rotvec": [0.04, 0.0, 0.0]}});
    assert_eq!(
        snap.eval_params(
            &HoldNoRotationRule::default(),
            &move_relative(),
            &mut ok,
            Verdict::Allow
        ),
        Verdict::Allow
    );
}

#[test]
fn hold_no_rotation_scales_an_absolute_move_eef_target_along_the_segment() {
    let mut pairs = holding_left();
    pairs.push(("eef_position.left", json!([0.0, 0.0, 0.3])));
    let snap = Snap::with(&pairs);
    let mut params = json!({"left": {"position": [0.2, 0.0, 0.3]}});
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_eef(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow);
    assert!(
        (num(&params, "/left/position/0") - 0.03).abs() < 1e-12,
        "{params}"
    );
    assert!(num(&params, "/left/position/1").abs() < 1e-12);
    assert!((num(&params, "/left/position/2") - 0.3).abs() < 1e-12);
}

/// 一条臂同时带绝对 `position` 和相对 `dx/dy/dz` 时必须当场 Deny，params 一个字节不动。
///
/// 这曾经是 3 cm 夹紧的一个绕过口。规则把两种写法当二选一（有 `position` 就走绝对分支，
/// `else` 才读增量），而 `arx_x5_move_relative` 的执行体 `_relative_target` 恰恰相反——
/// 只读 `dx/dy/dz`，完全无视 `position`。于是
/// `{"left": {"position": <当前 EEF>, "dx": 0.9}}` 让规则量出 0 m 的位移、什么都不夹、
/// 判 Allow，适配器再照着 `dx` 走出一步 0.9 m 的平移——持物中的单步 0.9 m，正是这条规则
/// 存在的理由。
///
/// 现在两层各自独立地堵上了：词汇关（`_ARM_ABS` / `_ARM_REL` 的
/// `additionalProperties: false`）在 `stage:"schema"` 就拒掉，这条规则在 `stage:"rules"`
/// 再拒一次。这里测的是后一层，所以 `spec` 用的是测试里那个空 schema——即使有人把
/// schema 改松了，规则也不放行。
#[test]
fn hold_no_rotation_denies_an_arm_that_mixes_absolute_position_with_relative_deltas() {
    let mut pairs = holding_left();
    pairs.push(("eef_position.left", json!([0.10, 0.20, 0.30])));
    let snap = Snap::with(&pairs);
    let mut params = json!({"left": {"position": [0.10, 0.20, 0.30], "dx": 0.90}, "speed": 0.3});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    let why = deny_reason(&v);
    assert!(
        why.contains("mixes absolute position with relative deltas") && why.contains("dx"),
        "理由要点名那个含糊的形状: {why}"
    );
    assert_eq!(params, before, "Deny 的那一遍不许动 params");
}

/// 同样的形状在 `arx_x5_move_eef` 上也是 Deny：方向反过来（执行体读 `position`、无视 `dx`），
/// 但"两种写法同时出现"这件事本身就没有正确答案，所以两条能力一视同仁。
#[test]
fn hold_no_rotation_denies_the_mixed_shape_on_move_eef_too() {
    let mut pairs = holding_left();
    pairs.push(("eef_position.left", json!([0.0, 0.0, 0.3])));
    let snap = Snap::with(&pairs);
    let mut params = json!({"left": {"position": [0.0, 0.0, 0.31], "dz": 0.90}});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_eef(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("mixes absolute position with relative deltas"));
    assert_eq!(params, before);
}

/// 不持物时这条规则整条不适用，混合形状也照样穿过去——这里钉的是"拒绝发生在持物这一支里"，
/// 免得有人把它误读成一条通用的形状校验（那是词汇关的活）。
#[test]
fn hold_no_rotation_leaves_the_mixed_shape_alone_when_neither_hand_holds() {
    let mut pairs = holding_neither();
    pairs.push(("eef_position.left", json!([0.10, 0.20, 0.30])));
    let snap = Snap::with(&pairs);
    let mut params = json!({"left": {"position": [0.10, 0.20, 0.30], "dx": 0.90}});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow);
    assert_eq!(params, before);
}

#[test]
fn hold_no_rotation_compares_an_absolute_quaternion_against_the_ledger() {
    let mut pairs = holding_left();
    pairs.push(("eef_position.left", json!([0.0, 0.0, 0.3])));
    pairs.push(("eef_quaternion.left", json!([1.0, 0.0, 0.0, 0.0])));
    let snap = Snap::with(&pairs);
    // 绕 z 转 90°：远远超过 0.05 rad。
    let h = std::f64::consts::FRAC_PI_4;
    let mut params = json!({"left": {"quaternion_wxyz": [h.cos(), 0.0, 0.0, h.sin()]}});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_eef(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("rotation"), "{v:?}");
    assert_eq!(params, before);
    // 同一个姿态（差 0.01 rad）放行。
    let t = 0.005f64;
    let mut small = json!({"left": {"quaternion_wxyz": [t.cos(), 0.0, 0.0, t.sin()]}});
    assert_eq!(
        snap.eval_params(
            &HoldNoRotationRule::default(),
            &move_eef(),
            &mut small,
            Verdict::Allow
        ),
        Verdict::Allow
    );
}

#[test]
fn hold_no_rotation_denies_when_the_pose_registers_are_unknown() {
    // 给了绝对目标，却读不到当前 EEF：不知道这一步走多远。
    let snap = Snap::with(&holding_left());
    let mut params = json!({"left": {"position": [0.2, 0.0, 0.3]}});
    let before = params.clone();
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_eef(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("eef_position.left"), "{v:?}");
    assert_eq!(params, before);

    // 给了绝对姿态，却读不到当前姿态。
    let mut pairs = holding_left();
    pairs.push(("eef_position.left", json!([0.0, 0.0, 0.3])));
    pairs.push(("eef_quaternion.left", json!(null)));
    let snap = Snap::with(&pairs);
    let mut params = json!({"left": {"quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}});
    let v = snap.eval_params(
        &HoldNoRotationRule::default(),
        &move_eef(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("eef_quaternion.left"), "{v:?}");
}

#[test]
fn hold_no_rotation_denies_an_unreadable_field() {
    let snap = Snap::with(&holding_left());
    for bad in [
        json!({"left": {"dz": "0.10"}}),
        json!({"left": {"dz": f64::NAN}}),
        json!({"left": {"drot_rotvec": [0.0, 0.0]}}),
        json!({"left": {"dyaw_deg": null}}),
        json!({"left": [0.0, 0.0, 0.1]}),
    ] {
        let mut params = bad.clone();
        let before = params.clone();
        let v = snap.eval_params(
            &HoldNoRotationRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow,
        );
        assert!(
            matches!(v, Verdict::Deny { .. }),
            "{bad} 该 fail-closed: {v:?}"
        );
        assert_eq!(params, before);
    }
}

#[test]
fn hold_no_rotation_denies_every_shape_of_unknown_ledger() {
    for state in [
        vec![
            ("holding.left", json!(null)),
            ("holding.right", json!(true)),
        ],
        vec![("holding.left", json!(1)), ("holding.right", json!(true))],
        vec![],
    ] {
        let snap = Snap::with(&state);
        let mut params = json!({"left": {"dz": 0.10}});
        let before = params.clone();
        let v = snap.eval_params(
            &HoldNoRotationRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow,
        );
        assert!(deny_reason(&v).contains("holding."), "{state:?}: {v:?}");
        assert_eq!(params, before);
    }
}

#[test]
fn hold_no_rotation_does_nothing_with_empty_hands() {
    let snap = Snap::with(&holding_neither());
    let mut params = json!({"left": {"dz": 0.50, "dyaw_deg": 90.0}});
    let before = params.clone();
    assert_eq!(
        snap.eval_params(
            &HoldNoRotationRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow
        ),
        Verdict::Allow,
        "空手转腕是别的规则的事"
    );
    assert_eq!(params, before);
}

#[test]
fn hold_no_rotation_ignores_other_capabilities_and_other_classes() {
    let snap = Snap::with(&holding_left());
    for spec in [set_gripper(), return_to_start(), move_relative_as_query()] {
        let mut params = json!({"left": {"dz": 0.50, "dyaw_deg": 90.0}});
        let before = params.clone();
        let v = snap.eval_params(
            &HoldNoRotationRule::default(),
            &spec,
            &mut params,
            Verdict::Allow,
        );
        assert_eq!(v, Verdict::Allow, "{}", spec.name);
        assert_eq!(params, before, "{}", spec.name);
    }
}

#[test]
fn hold_no_rotation_never_loosens() {
    let snap = Snap::with(&holding_left());
    never_loosens(
        &HoldNoRotationRule::default(),
        &snap,
        &move_relative(),
        json!({"left": {"dz": 0.10}}),
    );
    never_loosens(
        &HoldNoRotationRule::default(),
        &snap,
        &move_relative(),
        json!({"left": {"dyaw_deg": 10.0}}),
    );
    never_loosens(
        &HoldNoRotationRule::default(),
        &Snap::default(),
        &move_relative(),
        json!({"left": {"dz": 0.10}}),
    );
}

// ================================================================ rule-keepout

fn placed(key: &str, xyz: [f64; 3]) -> (String, Value) {
    (
        key.to_string(),
        json!({"xyz": xyz, "radius_m": 0.10, "native_step": 100, "episode_id": "e"}),
    )
}

/// 左臂在原点，`eef_position.left` 已知。
fn keepout_snap(extra: &[(String, Value)]) -> Snap {
    let mut s = Snap::with(&[("eef_position.left", json!([0.0, 0.0, 0.0]))]);
    for (k, v) in extra {
        s.state.insert(k.clone(), entry(v.clone()));
    }
    s
}

#[test]
fn keepout_has_a_name() {
    assert_eq!(KeepoutRule::default().name(), "rule-keepout");
}

#[test]
fn keepout_asks_in_exploration_and_denies_in_evaluation() {
    // 目标点 [1,0,0]，受保护物在 [1, 0.05, 0]：离终点 5 cm。
    let snap = keepout_snap(&[placed("placement_verified.alarm_clock", [1.0, 0.05, 0.0])]);
    let mut params = json!({"left": {"dx": 1.0}});

    let explore = KeepoutRule::default();
    assert_eq!(explore.verdict, "ask");
    let v = snap.eval_params(&explore, &move_relative(), &mut params, Verdict::Allow);
    assert!(ask_reason(&v).contains("alarm_clock"), "{v:?}");

    let eval = KeepoutRule {
        verdict: "deny".into(),
        ..KeepoutRule::default()
    };
    let v = snap.eval_params(&eval, &move_relative(), &mut params, Verdict::Allow);
    assert!(deny_reason(&v).contains("alarm_clock"), "{v:?}");
}

#[test]
fn keepout_lets_a_distant_target_through() {
    let snap = keepout_snap(&[placed("placement_verified.figurine", [1.0, 0.5, 0.0])]);
    let mut params = json!({"left": {"dx": 1.0}});
    let before = params.clone();
    assert_eq!(
        snap.eval_params(
            &KeepoutRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow
        ),
        Verdict::Allow
    );
    assert_eq!(params, before);
}

/// **g7 的那一课**（设计文档 §1.4）：两个端点都离得远，路上却蹭过去了。
///
/// 只看终点的实现会放行这一步——而 g7 正是这样输掉的：四件东西全放好，回臂时左臂在
/// 显示器旁停滞并蹭掉了 figurine。
#[test]
fn keepout_catches_a_path_that_grazes_a_placed_object_between_two_far_endpoints() {
    let snap = keepout_snap(&[placed("placement_verified.figurine", [0.5, 0.05, 0.0])]);
    let mut params = json!({"left": {"dx": 1.0}});

    // 两个端点都在半米开外。
    let start = [0.0f64, 0.0, 0.0];
    let target = [1.0f64, 0.0, 0.0];
    let p = [0.5f64, 0.05, 0.0];
    for e in [start, target] {
        let d = ((p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2)).sqrt();
        assert!(d > 0.4, "端点距离 {d} 不该触发任何只看端点的判据");
    }

    let v = snap.eval_params(
        &KeepoutRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert!(ask_reason(&v).contains("figurine"), "{v:?}");
}

#[test]
fn keepout_denies_a_protected_entry_it_cannot_read() {
    for bad in [
        json!("over there"),
        json!({"xyz": [0.0, 0.0]}),
        json!({"xyz": [0.0, 0.0, f64::NAN]}),
        json!(null),
        json!({}),
    ] {
        let snap = keepout_snap(&[("placement_verified.keyboard".to_string(), bad.clone())]);
        let mut params = json!({"left": {"dx": 1.0}});
        let v = snap.eval_params(
            &KeepoutRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow,
        );
        assert!(
            deny_reason(&v).contains("placement_verified.keyboard"),
            "{bad} 该 fail-closed: {v:?}"
        );
    }
}

#[test]
fn keepout_denies_when_the_arm_pose_is_unknown() {
    let mut snap = keepout_snap(&[placed("placement_verified.figurine", [5.0, 5.0, 5.0])]);
    snap.state
        .insert("eef_position.left".into(), entry(json!(null)));
    let mut params = json!({"left": {"dx": 1.0}});
    let v = snap.eval_params(
        &KeepoutRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert!(deny_reason(&v).contains("eef_position.left"), "{v:?}");
}

#[test]
fn a_per_entry_radius_overrides_the_configured_one() {
    let wide = (
        "placement_verified.monitor".to_string(),
        json!({"xyz": [1.0, 0.2, 0.0], "radius_m": 0.3, "native_step": 1, "episode_id": "e"}),
    );
    let snap = keepout_snap(&[wide]);
    let mut params = json!({"left": {"dx": 1.0}});
    let v = snap.eval_params(
        &KeepoutRule::default(),
        &move_relative(),
        &mut params,
        Verdict::Allow,
    );
    assert!(
        ask_reason(&v).contains("monitor"),
        "条目自己的 0.3 m 该盖过配置里的 0.10 m: {v:?}"
    );

    // 同样的几何，条目没写半径：20 cm 在默认的 10 cm 之外，放行。
    let narrow = keepout_snap(&[(
        "placement_verified.monitor".to_string(),
        json!({"xyz": [1.0, 0.2, 0.0]}),
    )]);
    assert_eq!(
        narrow.eval_params(
            &KeepoutRule::default(),
            &move_relative(),
            &mut json!({"left": {"dx": 1.0}}),
            Verdict::Allow
        ),
        Verdict::Allow
    );
}

#[test]
fn keepout_passes_when_nothing_is_protected_yet() {
    let snap = keepout_snap(&[]);
    let mut params = json!({"left": {"dx": 1.0}});
    let before = params.clone();
    assert_eq!(
        snap.eval_params(
            &KeepoutRule::default(),
            &move_relative(),
            &mut params,
            Verdict::Allow
        ),
        Verdict::Allow
    );
    assert_eq!(params, before);
}

#[test]
fn keepout_ignores_other_capabilities_and_other_classes() {
    let snap = keepout_snap(&[placed("placement_verified.figurine", [0.5, 0.0, 0.0])]);
    for spec in [wait_cap(), return_to_start(), move_relative_as_query()] {
        let mut params = json!({"left": {"dx": 1.0}});
        let before = params.clone();
        let v = snap.eval_params(&KeepoutRule::default(), &spec, &mut params, Verdict::Allow);
        assert_eq!(v, Verdict::Allow, "{}", spec.name);
        assert_eq!(params, before, "{}", spec.name);
    }
}

#[test]
fn keepout_never_loosens() {
    let snap = keepout_snap(&[placed("placement_verified.figurine", [0.5, 0.0, 0.0])]);
    never_loosens(
        &KeepoutRule::default(),
        &snap,
        &move_relative(),
        json!({"left": {"dx": 1.0}}),
    );
    never_loosens(
        &KeepoutRule {
            verdict: "deny".into(),
            ..KeepoutRule::default()
        },
        &snap,
        &move_relative(),
        json!({"left": {"dx": 1.0}}),
    );
}

// ================================================================ rule-step-reserve

#[test]
fn step_reserve_has_a_name() {
    assert_eq!(StepReserveRule::default().name(), "rule-step-reserve");
}

#[test]
fn step_reserve_denies_one_step_below_the_reserve_and_admits_one_at_it() {
    let low = Snap::with(&[("native_steps_remaining", json!(119))]);
    let v = low.eval(&StepReserveRule::default(), &move_eef());
    assert!(deny_reason(&v).contains("119"), "{v:?}");

    let exact = Snap::with(&[("native_steps_remaining", json!(120))]);
    assert_eq!(
        exact.eval(&StepReserveRule::default(), &move_eef()),
        Verdict::Allow
    );
}

#[test]
fn step_reserve_keeps_the_escape_hatches_reachable_at_zero() {
    let empty = Snap::with(&[("native_steps_remaining", json!(0))]);
    for spec in [wait_cap(), return_to_start(), reset_cap()] {
        assert_eq!(
            empty.eval(&StepReserveRule::default(), &spec),
            Verdict::Allow,
            "{} 必须在预算见底时仍然可达",
            spec.name
        );
    }
}

/// `arx_x5_finish` 是 `control`，靠 class 门排除——**不**靠 `exempt` 名单。
#[test]
fn step_reserve_never_speaks_about_a_control_capability() {
    let empty = Snap::with(&[("native_steps_remaining", json!(0))]);
    let rule = StepReserveRule::default();
    assert!(!rule.exempt.iter().any(|c| c == "arx_x5_finish"));
    assert_eq!(empty.eval(&rule, &finish_cap()), Verdict::Allow);
    // 寄存器完全读不到时也一样：class 门在最前面。
    assert_eq!(Snap::default().eval(&rule, &finish_cap()), Verdict::Allow);
}

#[test]
fn step_reserve_denies_every_shape_of_unknown_budget() {
    for v in [
        None,
        Some(json!(null)),
        Some(json!("800")),
        Some(json!(119.5)),
        Some(json!(f64::NAN)),
        Some(json!(-1)),
    ] {
        let snap = match &v {
            Some(value) => Snap::with(&[("native_steps_remaining", value.clone())]),
            None => Snap::default(),
        };
        let out = snap.eval(&StepReserveRule::default(), &move_eef());
        assert!(
            matches!(out, Verdict::Deny { .. }),
            "{v:?} 该 fail-closed: {out:?}"
        );
    }
}

#[test]
fn step_reserve_leaves_params_alone() {
    let low = Snap::with(&[("native_steps_remaining", json!(0))]);
    let mut params = json!({"speed": 1.0, "left": {"dz": 0.9}});
    let before = params.clone();
    let _ = low.eval_params(
        &StepReserveRule::default(),
        &move_eef(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(params, before);
    let mut params = json!({"steps": 10});
    let before = params.clone();
    let _ = low.eval_params(
        &StepReserveRule::default(),
        &wait_cap(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(params, before);
}

#[test]
fn step_reserve_never_loosens() {
    never_loosens(
        &StepReserveRule::default(),
        &Snap::with(&[("native_steps_remaining", json!(10))]),
        &move_eef(),
        json!({}),
    );
    never_loosens(
        &StepReserveRule::default(),
        &Snap::default(),
        &move_eef(),
        json!({}),
    );
}

// ================================================================ rule-mode

#[test]
fn mode_has_a_name() {
    assert_eq!(ArxModeRule::default().name(), "rule-mode");
}

#[test]
fn evaluation_refuses_reset_and_exploration_allows_it() {
    let snap = Snap::default();
    let eval = ArxModeRule::default();
    assert_eq!(eval.mode, "evaluation");
    let v = snap.eval(&eval, &reset_cap());
    assert!(deny_reason(&v).contains("arx_x5_reset"), "{v:?}");

    let explore = ArxModeRule {
        mode: "exploration".into(),
        ..ArxModeRule::default()
    };
    assert_eq!(snap.eval(&explore, &reset_cap()), Verdict::Allow);
}

#[test]
fn mode_says_nothing_about_the_ordinary_capabilities() {
    let snap = Snap::default();
    for spec in [move_eef(), move_relative(), wait_cap(), finish_cap()] {
        assert_eq!(
            snap.eval(&ArxModeRule::default(), &spec),
            Verdict::Allow,
            "{}",
            spec.name
        );
    }
}

#[test]
fn mode_leaves_params_alone() {
    let snap = Snap::default();
    let mut params = json!({"reason": "the clock slipped"});
    let before = params.clone();
    let v = snap.eval_params(
        &ArxModeRule::default(),
        &reset_cap(),
        &mut params,
        Verdict::Allow,
    );
    assert!(matches!(v, Verdict::Deny { .. }));
    assert_eq!(params, before);
}

#[test]
fn mode_never_loosens() {
    never_loosens(
        &ArxModeRule::default(),
        &Snap::default(),
        &reset_cap(),
        json!({"reason": "x"}),
    );
    never_loosens(
        &ArxModeRule {
            mode: "exploration".into(),
            ..ArxModeRule::default()
        },
        &Snap::default(),
        &reset_cap(),
        json!({"reason": "x"}),
    );
}

// ================================================================ rule-finish-gate

fn home_and_open() -> Vec<(&'static str, Value)> {
    vec![
        ("gripper_cmd.left", json!(1.0)),
        ("gripper_cmd.right", json!(1.0)),
        ("arms_home", json!(true)),
    ]
}

#[test]
fn finish_gate_has_a_name() {
    assert_eq!(FinishGateRule::default().name(), "rule-finish-gate");
}

#[test]
fn a_finish_with_open_grippers_and_arms_home_goes_straight_through() {
    let snap = Snap::with(&home_and_open());
    let mut params = json!({"status": "success", "summary": "all four placed"});
    let before = params.clone();
    let v = snap.eval_params(
        &FinishGateRule::default(),
        &finish_cap(),
        &mut params,
        Verdict::Allow,
    );
    assert_eq!(v, Verdict::Allow);
    assert_eq!(params, before);
}

#[test]
fn a_declared_failure_is_never_gated() {
    // 失败/放弃必须永远能收尾，哪怕夹爪还闭着、臂还在外面。
    let snap = Snap::default();
    let mut params = json!({"status": "failure", "summary": "gave up"});
    assert_eq!(
        snap.eval_params(
            &FinishGateRule::default(),
            &finish_cap(),
            &mut params,
            Verdict::Allow
        ),
        Verdict::Allow
    );
}

#[test]
fn finish_gate_asks_and_never_denies() {
    let cases: Vec<(&str, Vec<(&str, Value)>)> = vec![
        (
            "left gripper still closed",
            vec![
                ("gripper_cmd.left", json!(0.0)),
                ("gripper_cmd.right", json!(1.0)),
                ("arms_home", json!(true)),
            ],
        ),
        (
            "right gripper still closed",
            vec![
                ("gripper_cmd.left", json!(1.0)),
                ("gripper_cmd.right", json!(0.0)),
                ("arms_home", json!(true)),
            ],
        ),
        (
            "arms_home absent",
            vec![
                ("gripper_cmd.left", json!(1.0)),
                ("gripper_cmd.right", json!(1.0)),
            ],
        ),
        (
            "arms_home false",
            vec![
                ("gripper_cmd.left", json!(1.0)),
                ("gripper_cmd.right", json!(1.0)),
                ("arms_home", json!(false)),
            ],
        ),
        (
            "arms_home null",
            vec![
                ("gripper_cmd.left", json!(1.0)),
                ("gripper_cmd.right", json!(1.0)),
                ("arms_home", json!(null)),
            ],
        ),
        (
            "ledger null",
            vec![
                ("gripper_cmd.left", json!(null)),
                ("gripper_cmd.right", json!(null)),
                ("arms_home", json!(true)),
            ],
        ),
        ("nothing known at all", vec![]),
    ];
    for (what, state) in cases {
        let snap = Snap::with(&state);
        let mut params = json!({"status": "success", "summary": "done"});
        let before = params.clone();
        let (v, detail) = snap.eval_detail(
            &FinishGateRule::default(),
            &finish_cap(),
            &mut params,
            Verdict::Allow,
        );
        assert!(
            !matches!(v, Verdict::Deny { .. }),
            "{what}: 被拒的 finish 会让这一局彻底拿不到分数: {v:?}"
        );
        let reason = ask_reason(&v);
        assert!(
            reason.contains("open both grippers and return to start"),
            "{what}: {reason}"
        );
        assert_eq!(params, before, "{what}: 这条规则不碰 params");
        assert!(
            detail.expect("detail")["unmet"].is_array(),
            "{what}: 失败的读数进留痕，不进 params"
        );
    }
}

#[test]
fn finish_gate_says_nothing_about_other_capabilities() {
    let snap = Snap::default();
    for spec in [move_eef(), wait_cap(), reset_cap()] {
        let mut params = json!({"status": "success"});
        let before = params.clone();
        let v = snap.eval_params(
            &FinishGateRule::default(),
            &spec,
            &mut params,
            Verdict::Allow,
        );
        assert_eq!(v, Verdict::Allow, "{}", spec.name);
        assert_eq!(params, before);
    }
}

#[test]
fn finish_gate_never_loosens() {
    never_loosens(
        &FinishGateRule::default(),
        &Snap::default(),
        &finish_cap(),
        json!({"status": "success", "summary": "done"}),
    );
    never_loosens(
        &FinishGateRule::default(),
        &Snap::with(&home_and_open()),
        &finish_cap(),
        json!({"status": "success", "summary": "done"}),
    );
}

// ================================================================ 真内核：六条规则一起上链

/// 记下每一次执行调用的执行器。
///
/// 内置的 `mock-executor` 只记 `stop()`，读不到规则链跑完之后真正交出去的那份 params
/// ——而这正是这个用例要证明的东西。
#[derive(Default)]
struct Recorder {
    calls: Mutex<Vec<ExecutionCall>>,
}

impl Recorder {
    fn params_of(&self, capability: &str) -> Vec<Value> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.capability == capability)
            .map(|c| c.params.clone())
            .collect()
    }
}

impl Executor for Recorder {
    fn execute(&self, call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        self.calls.lock().unwrap().push(call);
        Box::pin(async {
            ExecOutcome {
                ok: true,
                result: json!({"ok": true}),
                error: None,
            }
        })
    }
}

/// S1 合约 §1.1 的 schema 原文（只取这个用例用得到的四条）：夹过的 params 会**再过一遍**
/// 这份 schema（`admission.rs` 的 `clamped params violate schema`），所以照抄，不简化。
fn arx_spec_json(name: &str, class: &str, risk: &str, schema: Value) -> Value {
    json!({"name": name, "class": class, "risk_level": risk,
           "params_schema": schema, "lease_ttl_s": 5.0})
}

fn arm_rel_schema() -> Value {
    json!({"type": ["object", "null"], "properties": {
        "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"},
        "drot_rotvec": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "dyaw_deg": {"type": "number"},
        "gripper": {"type": "number", "minimum": 0, "maximum": 1}}})
}

fn arm_abs_schema() -> Value {
    json!({"type": ["object", "null"], "properties": {
        "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "quaternion_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "gripper": {"type": "number", "minimum": 0, "maximum": 1}}})
}

fn arx_specs() -> Vec<Value> {
    let move_props = |arm: Value| {
        json!({"type": "object", "additionalProperties": false, "properties": {
            "left": arm, "right": arm,
            "speed": {"type": "number", "minimum": 0.1, "maximum": 1.0},
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 200},
            "tolerance_m": {"type": "number", "minimum": 0.001, "maximum": 0.2},
            "tolerance_rad": {"type": "number", "minimum": 0.005, "maximum": 1.0}}})
    };
    vec![
        arx_spec_json(
            "arx_x5_move_relative",
            "motion",
            "high",
            move_props(arm_rel_schema()),
        ),
        arx_spec_json(
            "arx_x5_move_eef",
            "motion",
            "high",
            move_props(arm_abs_schema()),
        ),
        arx_spec_json(
            "arx_x5_wait",
            "motion",
            "low",
            json!({"type": "object", "additionalProperties": false,
                   "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 60}}}),
        ),
        arx_spec_json(
            "arx_x5_return_to_start",
            "motion",
            "medium",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "arms": {"type": "array", "items": {"type": "string", "enum": ["left", "right"]},
                         "minItems": 1, "maxItems": 2},
                "lift_m": {"type": "number", "minimum": 0, "maximum": 0.5},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 400},
                "open_grippers": {"type": "boolean"}}}),
        ),
    ]
}

fn proposal(id: &str, cap: &str, params: Value) -> Proposal {
    Proposal {
        proposal_id: ProposalId(id.into()),
        brain_id: "slow".into(),
        turn: 1,
        trace_id: format!("t-{id}"),
        capability: cap.into(),
        params,
    }
}

/// 等一个条件成立（5 ms × 200 = 1 s 上限）：等的是状态，不是时间。
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..200 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

fn count_kind(log: &WorldLog, kind: &str) -> usize {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() == kind)
        .count()
}

/// 最后一条 `act/refused` 的 trace 里有没有这条规则。
///
/// 断言"有"，不断言"它拥有理由"：同紧度留上游的理由，而账本未知时两条规则都会 Deny。
fn refusal_trace_contains(log: &WorldLog, rule: &str) -> bool {
    let evs = log.since(0).unwrap();
    let Some(refused) = evs.iter().rev().find(|e| e.kind.as_str() == "act/refused") else {
        return false;
    };
    refused.payload["trace"]
        .as_array()
        .map(|t| t.iter().any(|s| s["rule"] == json!(rule)))
        .unwrap_or(false)
}

#[tokio::test]
async fn the_six_rules_boot_together_and_clamp_deny_and_admit_through_a_real_kernel() {
    let six = [
        "rule-mode",
        "rule-step-reserve",
        "rule-carry-speed",
        "rule-hold-no-rotation",
        "rule-keepout",
        "rule-finish-gate",
    ];
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains"),
        EntryRow::new("admission").with_config(json!({"required_rules": six})),
        EntryRow::new("rule-mode").with_config(json!({"mode": "evaluation"})),
        EntryRow::new("rule-step-reserve"),
        EntryRow::new("rule-carry-speed"),
        EntryRow::new("rule-hold-no-rotation"),
        EntryRow::new("rule-keepout").with_config(json!({"verdict": "deny"})),
        EntryRow::new("rule-finish-gate"),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    assert!(k.report().pending.is_empty(), "{:?}", k.report());

    let adm = k.registry().get::<Admission>(&"admission".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    // BOOT 不让动；生产里这一步由 `distill-arx-ledger` 在第一条账本事件上做（合约 §4.3）。
    fsm.transition(FsmState::Standing, "test", "test").unwrap();

    let rec = Arc::new(Recorder::default());
    let mut _disposers = Vec::new();
    for spec_json in arx_specs() {
        let spec: CapabilitySpec = serde_json::from_value(spec_json).unwrap();
        _disposers.push(
            adm.capabilities()
                .register(spec, rec.clone() as Arc<dyn Executor>)
                .unwrap(),
        );
    }

    state.set_many(
        &[
            ("gripper_cmd.left", json!(0.0)),
            ("gripper_cmd.right", json!(0.0)),
            ("holding.left", json!(true)),
            ("holding.right", json!(false)),
            ("native_steps_remaining", json!(800)),
            ("eef_position.left", json!([0.0, 0.0, 0.3])),
            ("eef_quaternion.left", json!([1.0, 0.0, 0.0, 0.0])),
            ("arms_home", json!(false)),
        ],
        "test",
    );

    // (a) 持物 + 全速 + 10 cm：两条规则各夹一处，动作照跑。
    let out = adm
        .propose(proposal(
            "p-carry",
            "arx_x5_move_relative",
            json!({"speed": 1.0, "left": {"dz": 0.10}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    wait_until("two act/clamped events", || {
        count_kind(&log, "act/clamped") == 2
    })
    .await;
    let clamped_rules: Vec<String> = log
        .since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() == "act/clamped")
        .map(|e| e.payload["rule"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        clamped_rules.contains(&"rule-carry-speed".to_string())
            && clamped_rules.contains(&"rule-hold-no-rotation".to_string()),
        "{clamped_rules:?}"
    );
    let calls = rec.params_of("arx_x5_move_relative");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert!(
        (num(&calls[0], "/speed") - 0.4).abs() < 1e-12,
        "执行器拿到的是夹过的那一份: {}",
        calls[0]
    );
    assert!(
        (num(&calls[0], "/left/dz") - 0.03).abs() < 1e-12,
        "{}",
        calls[0]
    );

    // (b) 持物转腕：拒。trace 里**有**这条规则（不断言它拥有理由）。
    let out = adm
        .propose(proposal(
            "p-yaw",
            "arx_x5_move_relative",
            json!({"left": {"dyaw_deg": 10.0}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    assert!(
        refusal_trace_contains(&log, "rule-hold-no-rotation"),
        "trace 里该有 rule-hold-no-rotation"
    );
    assert_eq!(
        rec.params_of("arx_x5_move_relative").len(),
        1,
        "没有第二次执行"
    );

    // (c) 账本变成未知：两条 move 被拒，三个逃生口仍然可达（合约 §2.2）。
    //
    // 这个不对称是**故意的**：内核层的 fail-closed 只覆盖 move_eef / move_relative，
    // `wait` / `return_to_start` / `reset` 必须留着，不然手臂永远回不了家。谁要是把它
    // "修"成全链保证，这个断言会当场失败——先读合约 §2.2。
    state.set_many(
        &[
            ("holding.left", json!(null)),
            ("holding.right", json!(null)),
        ],
        "test",
    );
    let out = adm
        .propose(proposal(
            "p-unknown",
            "arx_x5_move_eef",
            json!({"left": {"position": [0.01, 0.0, 0.3]}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    assert!(
        refusal_trace_contains(&log, "rule-carry-speed"),
        "trace 里该有 rule-carry-speed"
    );

    let out = adm
        .propose(proposal("p-wait", "arx_x5_wait", json!({"steps": 10})))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "账本未知时 wait 必须仍然可达: {out:?}"
    );
    let out = adm
        .propose(proposal(
            "p-home",
            "arx_x5_return_to_start",
            json!({"lift_m": 0.05, "max_steps": 150, "open_grippers": true}),
        ))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "账本未知时 return_to_start 必须仍然可达: {out:?}"
    );

    // (d) 预算见底：move 被拒，wait 仍然放行。
    state.set_many(&[("native_steps_remaining", json!(100))], "test");
    let out = adm
        .propose(proposal("p-broke", "arx_x5_wait", json!({"steps": 8})))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    state.set_many(
        &[
            ("holding.left", json!(false)),
            ("holding.right", json!(false)),
        ],
        "test",
    );
    let out = adm
        .propose(proposal(
            "p-broke-move",
            "arx_x5_move_relative",
            json!({"left": {"dz": 0.01}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    assert!(refusal_trace_contains(&log, "rule-step-reserve"));
}
