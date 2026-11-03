//! S1 端到端：`profiles/robodojo-{eval,explore}.yaml` 走真的 `load_profile` 上真内核。
//!
//! 这个文件验的是**出厂的那一份配置**，不是手搭的行列表：设计文档 §4 的 S1 验收行
//! （持物全速被夹、持物转腕被拒、账本未知的 hold 被拒）必须在真实 bundle + profile +
//! 真实规则链上成立，而不是只在某个测试自己凑出来的六条规则上成立。
//!
//! 两条每次都得打的补丁，少一条这个二进制连 boot 都过不去：
//!
//! 1. `- { id: rpc-server, disabled: true }`。`rpc-server` 唯一的 `inventory::submit!`
//!    在 `crates/oeh-rpc/src/plugin.rs`，而 `crates/oeh-admission/Cargo.toml` 的
//!    dev-dependencies 里**没有** `oeh-rpc`——这个 id 在本二进制里解析不出来，
//!    `Kernel::boot` 会在第一条断言之前就返回 `BootError::UnknownPlugin`。
//!    `crates/oeh-kernel/src/kernel.rs` 在解析 id **之前**就跳过 `disabled` 行，所以
//!    这条补丁不需要动任何 manifest。顺带也避免了在 cwd（`crates/oeh-admission/`，
//!    也就是仓库里面）绑一个真的 `./run/oeh.sock`，以及两个并发测试二进制抢同一个
//!    socket 文件。
//! 2. `distill-placement.spec_path` 改成绝对路径。cargo 跑集成测试时 cwd = 包根目录，
//!    bundle 里的 `./task/robodojo/...` 会解析成 `crates/oeh-admission/task/...`，
//!    而这个 loader 是必需路径：读不到就 `report().failed`。
//!
//! 其余三条（`world-log.dir` / `brains.session_dir` / `distill-estop.flag_path`）挪进
//! `tempfile::tempdir()`。`admission.estop_flag` **不走补丁文件**而是在合成之后单点改
//! 掉：整行 Replace 会把 `required_rules` 一起换掉，那这个用例就验不到真正的规则集了。
//! 两个急停路径必须归一到同一个字符串，否则 `admission` 在 boot 当场 fail-closed。
//!
//! 能力来自一个本二进制里并不存在的远端插件，所以在运行时注册：`params_schema` 照抄
//! S1 合约 §1.1 的原文——夹过的 params 会**再过一遍**这份 schema，照抄才验得到那一步。
//! 执行器是本文件自己的 `Recorder`：内置的 `mock-executor` 只记 `stop()`，读不到规则链
//! 真正交出去的那份 params。
//!
//! 纪律（S1 合约 §3、§10）：只断言 trace 里"有"某条规则，不断言它"拥有"理由；浮点带
//! 容差；等条件不睡固定时间；每个提案一个**不同**的 `ProposalId`（重复的那个会在词汇
//! 阶段就被拒）；绝不 `set_current_dir`（同一个 crate 的测试二进制是并发跑的）。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oeh_admission as _;
use oeh_admission::admission::Admission;
use oeh_admission::capability::{CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor};
use oeh_admission::rules::hold_no_rotation::HoldNoRotationRule;
use oeh_admission::verdict::{Rule, RuleCtx, Verdict};
use oeh_distill as _;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::profile::{load_profile, EntryRow};
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Disposer, Kernel, WorldLog};
use oeh_proto::{EventDraft, EventKind, WorldEvent};
use serde_json::{json, Value};

/// 仓库根。绝不 `set_current_dir`——见文件头。
fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

const EVAL: &str = "profiles/robodojo-eval.yaml";
const EXPLORE: &str = "profiles/robodojo-explore.yaml";

// ---------------------------------------------------------------- 启动

/// 真 profile 的行，四条路径挪进 `tmp`，`rpc-server` 关掉。
///
/// 补丁文件写的是 **JSON**：YAML 1.2 是 JSON 的超集，`serde_yaml_ng` 照收，而这个仓库
/// 的路径里有空格（`Industry grade -Open Embodied Harness`），手写 YAML flow 标量迟早
/// 要在某台机器上炸。
fn rows(profile: &str, tmp: &Path) -> Vec<EntryRow> {
    let repo = root();
    let patch_path = tmp.join("patch.yaml");
    let patch = json!([
        {"id": "rpc-server", "disabled": true},
        {"id": "world-log", "config": {
            "dir": tmp.join("logs"), "segment_mb": 16, "retain": 32768}},
        {"id": "brains", "config": {
            "session_dir": tmp.join("sessions"), "turn_timeout_ms": 1_800_000}},
        {"id": "distill-estop", "config": {"flag_path": tmp.join("estop")}},
        {"id": "distill-placement", "config": {
            "spec_path": repo.join("task/robodojo/organize_table.json")}},
    ]);
    std::fs::write(&patch_path, serde_json::to_string_pretty(&patch).unwrap()).unwrap();

    let mut rows = load_profile(
        &repo.join(profile),
        &repo.join("bundles"),
        std::slice::from_ref(&patch_path),
    )
    .unwrap_or_else(|e| panic!("load_profile({profile}): {e}"));
    for r in &mut rows {
        // 只改这一个键：整行 Replace 会带走 `required_rules`，那就不是在验真规则集了。
        if r.id == "admission" {
            r.config["estop_flag"] = json!(tmp.join("estop"));
        }
    }
    rows
}

/// 跑起来的一局：内核连同测试要直接看的四个服务，和能力注册的 disposer。
struct Live {
    _kernel: Kernel,
    adm: Arc<Admission>,
    log: Arc<WorldLog>,
    state: Arc<StateRegistry>,
    fsm: Arc<RobotFsm>,
    rec: Arc<Recorder>,
    _caps: Vec<Disposer>,
}

/// 启动一条 lane，并把 S1 合约 §1.1 的八条能力注册到 `Recorder` 上。
///
/// 这里**没有** `fsm.transition(...)`：STANDING 必须由生产路径给出，也就是
/// `distill-arx-ledger` 在第一条账本事件上驱动的 `BOOT → STANDING`（合约 §4.3）。
fn boot(profile: &str, tmp: &Path) -> Live {
    let kernel = Kernel::boot(rows(profile, tmp)).unwrap();
    assert!(
        kernel.report().failed.is_empty(),
        "{profile} 有插件激活失败: {:?}",
        kernel.report()
    );
    assert!(
        kernel.report().pending.is_empty(),
        "{profile} 有插件卡在 pending: {:?}",
        kernel.report()
    );
    let reg = kernel.registry().clone();
    let adm = reg.get::<Admission>(&"admission".into()).unwrap();
    let log = reg.get::<WorldLog>(&"world-log".into()).unwrap();
    let state = reg.get::<StateRegistry>(&"state".into()).unwrap();
    let fsm = reg.get::<RobotFsm>(&"fsm".into()).unwrap();

    let rec = Arc::new(Recorder::default());
    let caps = arx_specs()
        .into_iter()
        .map(|s| {
            let spec: CapabilitySpec = serde_json::from_value(s).unwrap();
            adm.capabilities()
                .register(spec, rec.clone() as Arc<dyn Executor>)
                .unwrap()
        })
        .collect();

    Live {
        _kernel: kernel,
        adm,
        log,
        state,
        fsm,
        rec,
        _caps: caps,
    }
}

// ---------------------------------------------------------------- 执行器

/// 记下每一次执行调用。内置的 `mock-executor` 只记 `stop()`。
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
    fn names(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.capability.clone())
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

// ---------------------------------------------------------------- 能力（合约 §1.1 原文）

/// 每条臂的子 schema 上那一行 `additionalProperties: false` 是承重的，不是整洁癖：两个
/// 执行体读的键集互不相交（`_absolute_target` 读 `position`/`quaternion_wxyz` 而无视
/// `dx/dy/dz`，`_relative_target` 反过来），所以一条臂带着**另一条能力**的键，就是一份
/// "规则量到的"和"适配器执行的"不是同一件事的请求。校验器对缺省的 `additionalProperties`
/// 按 true 处理（`capability.rs`：`.and_then(as_bool).unwrap_or(true)`），少这一行的后果
/// 是 `move_relative` 的一条臂夹带 `position` 时，0.9 m 的 `dx` 整个绕过了 3 cm 夹紧。
/// 这里照抄 `arx_plugin.py` 的 `_ARM_ABS` / `_ARM_REL`。
fn arm_rel() -> Value {
    json!({"type": ["object", "null"], "additionalProperties": false, "properties": {
        "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"},
        "drot_rotvec": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "dyaw_deg": {"type": "number"},
        "gripper": {"type": "number", "minimum": 0, "maximum": 1}}})
}

fn arm_abs() -> Value {
    json!({"type": ["object", "null"], "additionalProperties": false, "properties": {
        "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "quaternion_wxyz": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "gripper": {"type": "number", "minimum": 0, "maximum": 1}}})
}

fn cap(name: &str, class: &str, risk: &str, schema: Value) -> Value {
    json!({"name": name, "class": class, "risk_level": risk,
           "params_schema": schema, "lease_ttl_s": 5.0})
}

/// 两条 move 能力共用的顶层 schema。自由函数而不是 `arx_specs` 里的闭包：词汇关那条用例
/// 要直接拿这一份去 `validate_params`，而"注册用的"和"测试用的"必须是同一份。
fn move_schema(arm: Value) -> Value {
    json!({"type": "object", "additionalProperties": false, "properties": {
        "left": arm, "right": arm,
        "speed": {"type": "number", "minimum": 0.1, "maximum": 1.0},
        "max_steps": {"type": "integer", "minimum": 1, "maximum": 200},
        "tolerance_m": {"type": "number", "minimum": 0.001, "maximum": 0.2},
        "tolerance_rad": {"type": "number", "minimum": 0.005, "maximum": 1.0}}})
}

/// 合约 §1 的八条 motion/control 能力，class/risk/schema 全部照抄。
fn arx_specs() -> Vec<Value> {
    vec![
        cap(
            "arx_x5_vla_act",
            "motion",
            "high",
            json!({"type": "object", "additionalProperties": false,
                   "required": ["observation_id", "subgoal", "execute_steps"],
                   "properties": {
                       "observation_id": {"type": "string", "minLength": 1},
                       "subgoal": {"type": "string", "minLength": 1, "maxLength": 512},
                       "execute_steps": {"type": "integer", "minimum": 1, "maximum": 50}}}),
        ),
        cap("arx_x5_move_eef", "motion", "high", move_schema(arm_abs())),
        cap(
            "arx_x5_move_relative",
            "motion",
            "high",
            move_schema(arm_rel()),
        ),
        cap(
            "arx_x5_set_gripper",
            "motion",
            "medium",
            json!({"type": "object", "additionalProperties": false,
                   "required": ["arm", "opening"], "properties": {
                       "arm": {"type": "string", "enum": ["left", "right"]},
                       "opening": {"type": "number", "minimum": 0, "maximum": 1},
                       "steps": {"type": "integer", "minimum": 1, "maximum": 40}}}),
        ),
        cap(
            "arx_x5_wait",
            "motion",
            "low",
            json!({"type": "object", "additionalProperties": false, "properties": {
                "steps": {"type": "integer", "minimum": 1, "maximum": 60}}}),
        ),
        cap(
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
        // 合约 §1 的第一条刻意偏离：`reset` 是 **motion**，不是 control——只有 motion
        // 才走 `execution.rs` 的运动槽/抢占门，control 的 reset 会在一个还活着的
        // `_servo` 底下把 backend 换掉，而没有任何东西能拒绝它。
        cap(
            "arx_x5_reset",
            "motion",
            "high",
            json!({"type": "object", "additionalProperties": false, "required": ["reason"],
                   "properties": {"reason": {"type": "string", "minLength": 1, "maxLength": 512}}}),
        ),
        // 第二条刻意偏离：`finish` 留在 **control**。`rules/fsm.rs` 只拒 Motion，所以
        // 一局已经死掉的 episode 仍然收得了尾——那是拿到原生分数的唯一途径。
        cap(
            "arx_x5_finish",
            "control",
            "low",
            json!({"type": "object", "additionalProperties": false,
                   "required": ["status", "summary"], "properties": {
                       "status": {"type": "string", "enum": ["success", "failure"]},
                       "summary": {"type": "string", "minLength": 1, "maxLength": 4096}}}),
        ),
    ]
}

// ---------------------------------------------------------------- 账本事件

/// 合约 §4.1 的账本载荷。`seq` 单调，`registration` 一局不变——蒸馏器的排序保护就认这个。
#[allow(clippy::too_many_arguments)]
fn ledger(seq: i64, gripper: Value, holding: Value, remaining: i64, extra: Value) -> Value {
    let mut payload = json!({
        "schema": "oeh.arx_x5.ledger.v1",
        "episode_id": "ep-s1",
        "seq": seq,
        "registration": 1,
        "primitive": null,
        "outcome": null,
        "stop_reason": null,
        "executed_steps": 0,
        "gripper_cmd": gripper,
        "gripper_measured": {"left": 0.71, "right": 0.99},
        "holding": holding,
        "native_step": 1000 - remaining,
        "native_step_limit": 1000,
        "native_steps_remaining": remaining,
        "native_terminal": false,
        "native_success": false,
        "arms_home": false,
        "eef": {
            "left": {"position": [0.0, 0.0, 0.30], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "right": {"position": [0.0, -0.4, 0.30], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
        }
    });
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        payload[k] = v;
    }
    payload
}

/// 左臂夹着东西、预算充足的账本。
fn holding_left(seq: i64) -> Value {
    ledger(
        seq,
        json!({"left": 0.0, "right": 1.0}),
        json!({"left": true, "right": false}),
        800,
        json!({}),
    )
}

/// 账本未知：两个通道都是 JSON null（规则读作"未知"，fail-closed）。
fn unknown_ledger(seq: i64) -> Value {
    ledger(
        seq,
        json!({"left": null, "right": null}),
        json!({"left": null, "right": null}),
        800,
        json!({}),
    )
}

/// 手空、臂已回家、夹爪张开：`rule-finish-gate` 的通过条件。
fn home_and_open(seq: i64) -> Value {
    ledger(
        seq,
        json!({"left": 1.0, "right": 1.0}),
        json!({"left": false, "right": false}),
        800,
        json!({"arms_home": true}),
    )
}

/// 直接 `append`：观察者是同步调用的，所以这一行返回时蒸馏器已经写完寄存器、
/// 也已经把 FSM 推过去了。不需要 `kernel.run()`，也就没有"等多久才算写完"的问题。
fn emit(live: &Live, kind: &str, payload: Value) {
    live.log.append(EventDraft::new(
        EventKind::new(kind).unwrap(),
        "test",
        payload,
    ));
}

/// 账本事件 + 立刻断言 FSM 已经站起来。生产路径上这是同一件事。
fn establish(live: &Live, payload: Value) {
    emit(live, "obs/arx_x5_ledger", payload);
}

// ---------------------------------------------------------------- 断言工具

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

fn events(live: &Live) -> Vec<WorldEvent> {
    live.log.since(0).unwrap_or_default()
}

fn count_kind(live: &Live, kind: &str) -> usize {
    events(live)
        .iter()
        .filter(|e| e.kind.as_str() == kind)
        .count()
}

fn kinds(live: &Live) -> Vec<String> {
    events(live)
        .iter()
        .map(|e| e.kind.as_str().to_string())
        .collect()
}

/// 某个 trace 下最后一条 `act/refused`。
fn refusal<'a>(evs: &'a [WorldEvent], trace: &str) -> &'a WorldEvent {
    evs.iter()
        .rev()
        .find(|e| e.kind.as_str() == "act/refused" && e.trace_id.as_deref() == Some(trace))
        .unwrap_or_else(|| {
            panic!(
                "trace {trace} 没有 act/refused: {:?}",
                evs.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>()
            )
        })
}

/// 这条拒绝的规则 trace 里**有**这条规则。
///
/// 断言"有"，不断言"拥有理由"：同紧度留上游的理由，而账本未知时 `rule-carry-speed`
/// 与 `rule-hold-no-rotation` 都会 Deny，谁先谁后是注册顺序，是实现细节。
#[track_caller]
fn trace_contains(evs: &[WorldEvent], trace: &str, rule: &str) -> bool {
    refusal(evs, trace).payload["trace"]
        .as_array()
        .map(|t| t.iter().any(|s| s["rule"] == json!(rule)))
        .unwrap_or(false)
}

/// 等一个条件成立：20 ms × 100 = 2 s 上限。等的是状态，不是时间——`scripts/ci.sh` 给
/// 整段 `cargo test --workspace` 的硬上限是 `timeout 300`，而这个文件每个场景一次 boot。
async fn until(live: &Live, what: &str, mut cond: impl FnMut(&Live) -> bool) {
    for _ in 0..100 {
        if cond(live) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("等 {what} 超过 2 s；世界日志: {:?}", kinds(live));
}

#[track_caller]
fn num(params: &Value, pointer: &str) -> f64 {
    params
        .pointer(pointer)
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| panic!("{pointer} 不是数字: {params}"))
}

// ================================================================ 1. 两条 lane 都 boot 干净

/// 出厂的两份 profile 都能从 `load_profile` 一路 boot 到干净的报告。
///
/// 这是这个文件里唯一一个"配置本身对不对"的断言，也是 P5 那五个 YAML 的**唯一**绑定
/// 检查：`pyyaml` 不在那个 venv 里，P5 的 Python 测试只能做文本 grep。
#[tokio::test]
async fn both_profiles_boot_with_a_clean_report() {
    for profile in [EVAL, EXPLORE] {
        let tmp = tempfile::tempdir().unwrap();
        let live = boot(profile, tmp.path());
        // boot 之后、任何账本事件之前：FSM 还在 BOOT，寄存器是空的。这不是竞态，是
        // fail-closed 的起点（bundle 头部那段注释说的就是这件事）。
        assert_eq!(live.fsm.state(), FsmState::Boot, "{profile}");
        assert_eq!(live.state.get("gripper_cmd.left"), None, "{profile}");
    }
}

// ================================================================ 2. 生产路径把机器人站起来

/// 一条 seq-0 的账本事件 → 寄存器全填 → `BOOT → STANDING`。全程没有手工 transition。
#[tokio::test]
async fn the_seq_zero_ledger_event_fills_every_register_and_stands_the_robot_up() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0));

    // 合约 §4.4：`distill-arx-ledger` 拥有的全部键。
    for key in [
        "gripper_cmd.left",
        "gripper_cmd.right",
        "gripper_measured.left",
        "gripper_measured.right",
        "holding.left",
        "holding.right",
        "native_step",
        "native_step_limit",
        "native_steps_remaining",
        "native_terminal",
        "native_success",
        "arms_home",
        "eef_position.left",
        "eef_position.right",
        "eef_quaternion.left",
        "eef_quaternion.right",
        "arx_episode_id",
        "arx_ledger_seq",
        "arx_ledger_registration",
    ] {
        assert!(
            live.state.get(key).is_some(),
            "账本事件之后 `{key}` 仍然没有值"
        );
    }
    assert_eq!(live.state.get("holding.left"), Some(json!(true)));
    assert_eq!(live.state.get("native_steps_remaining"), Some(json!(800)));
    // 这一步是生产路径给的（合约 §4.3）：本文件里没有任何 `fsm.transition`。
    assert_eq!(live.fsm.state(), FsmState::Standing);
}

// ================================================================ 3. S1 验收行 1：carry clamp

/// 持物 + `speed 1.0` + 10 cm：两条规则各夹一处，动作照跑，执行器拿到的是夹过的那份。
#[tokio::test]
async fn s1_row_1_a_full_speed_carry_is_clamped_to_four_tenths_and_three_centimetres() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0));

    let out = live
        .adm
        .propose(proposal(
            "p-carry",
            "arx_x5_move_relative",
            json!({"speed": 1.0, "left": {"dz": 0.10}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");

    until(&live, "两条 act/clamped", |l| {
        count_kind(l, "act/clamped") == 2
    })
    .await;
    let clamped: Vec<String> = events(&live)
        .iter()
        .filter(|e| e.kind.as_str() == "act/clamped")
        .map(|e| e.payload["rule"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        clamped.contains(&"rule-carry-speed".to_string())
            && clamped.contains(&"rule-hold-no-rotation".to_string()),
        "{clamped:?}"
    );

    let calls = live.rec.params_of("arx_x5_move_relative");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert!(
        (num(&calls[0], "/speed") - 0.4).abs() < 1e-12,
        "执行器拿到的 speed: {}",
        calls[0]
    );
    assert!(
        (num(&calls[0], "/left/dz") - 0.03).abs() < 1e-12,
        "执行器拿到的 dz: {}",
        calls[0]
    );
    assert_eq!(count_kind(&live, "act/refused"), 0, "{:?}", kinds(&live));
}

// ================================================================ 4. S1 验收行 2：持物转腕

/// 持物时转腕：拒，`stage: "rules"`，trace 里**有** `rule-hold-no-rotation`，params 没被动过。
#[tokio::test]
async fn s1_row_2_a_wrist_turn_while_holding_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0));

    let out = live
        .adm
        .propose(proposal(
            "p-yaw",
            "arx_x5_move_relative",
            json!({"left": {"dyaw_deg": 10.0}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");

    let evs = events(&live);
    let refused = refusal(&evs, "t-p-yaw");
    assert_eq!(refused.payload["stage"], "rules");
    assert!(
        trace_contains(&evs, "t-p-yaw", "rule-hold-no-rotation"),
        "trace: {}",
        refused.payload["trace"]
    );
    // Deny 在 clamp 之前跑完（合约 §3 的两遍结构），所以一个参数都没被改过。
    assert!(
        live.rec.params_of("arx_x5_move_relative").is_empty(),
        "被拒的提案不该起跑"
    );
}

/// 3 cm 夹紧曾经有一个**合 schema 的**绕过口：一条臂同时写绝对 `position` 和相对 `dx`。
///
/// 规则把两种写法当二选一（有 `position` 就走绝对分支，`else` 才读增量），而
/// `arx_x5_move_relative` 的执行体只读 `dx/dy/dz`、完全无视 `position`。于是
/// `{"left": {"position": <当前 EEF>, "dx": 0.9}}` 被量成 0 m 的位移、什么都不夹、判
/// Allow，适配器再照着 `dx` 走出持物中的单步 0.9 m。
///
/// 现在两道关各自独立地挡住它，这条用例把两道都走一遍：
/// 1. 词汇关——每条臂的子 schema 是 `additionalProperties: false`，所以真实提案在
///    `stage:"vocabulary"` 就被拒，准入链一条规则都没跑；
/// 2. 规则关——把同一份 params 直接喂给规则（绕过词汇关），仍然是 Deny 且 params 不动。
///    即使将来有人把 schema 改松了，这一层也不放行。
#[tokio::test]
async fn a_move_relative_arm_that_mixes_position_with_a_delta_is_refused_at_both_layers() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0));

    // --- 第一道：词汇关。
    let params = json!({"left": {"position": [0.10, 0.20, 0.30], "dx": 0.90}, "speed": 0.3});
    let out = live
        .adm
        .propose(proposal("p-mixed", "arx_x5_move_relative", params.clone()))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    let evs = events(&live);
    let refused = refusal(&evs, "t-p-mixed");
    assert_eq!(
        refused.payload["stage"], "vocabulary",
        "per-arm 的 additionalProperties 应该在准入之前就拒掉它: {}",
        refused.payload
    );
    assert!(
        refused.payload["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("additional property `position` is not allowed"),
        "{}",
        refused.payload
    );
    assert!(
        live.rec.params_of("arx_x5_move_relative").is_empty(),
        "0.9 m 的一步绝不该到达执行器"
    );

    // 同一份 schema 直接校验一次，把"词汇关确实认得这个形状"钉死，而不是靠上面那条
    // 拒绝可能来自别的原因。
    let schema = move_schema(arm_rel());
    let why = oeh_admission::capability::validate_params(&schema, &params)
        .expect_err("混合形状必须过不了词汇关");
    assert!(why.contains("dx") || why.contains("position"), "{why}");
    // 各自单独出现时都是合法的，所以拒的确实是"同时出现"这件事。
    assert!(oeh_admission::capability::validate_params(
        &schema,
        &json!({"left": {"dx": 0.01}, "speed": 0.3})
    )
    .is_ok());
    assert!(oeh_admission::capability::validate_params(
        &move_schema(arm_abs()),
        &json!({"left": {"position": [0.1, 0.2, 0.3]}})
    )
    .is_ok());

    // --- 第二道：规则关。schema 换成空的，词汇关放行，只剩规则说话。
    let bare: CapabilitySpec =
        serde_json::from_value(cap("arx_x5_move_relative", "motion", "high", json!({}))).unwrap();
    let rule = HoldNoRotationRule::default();
    let mut p = params.clone();
    let entries = live.state.snapshot_entries();
    let ctx = RuleCtx {
        now_ns: 0,
        fsm: FsmState::Standing,
        state: &entries,
        spec: &bare,
        estop_engaged: false,
    };
    let verdict = Rule::evaluate(&rule, &mut p, &ctx, Verdict::Allow);
    match &verdict {
        Verdict::Deny { reason } => assert!(
            reason.contains("mixes absolute position with relative deltas"),
            "{reason}"
        ),
        other => panic!("expected Deny, got {other:?}"),
    }
    assert_eq!(p, params, "Deny 的那一遍不许动 params");
}

// ================================================================ 5. S1 验收行 3：账本未知

/// 账本未知：两条 move 被链拒掉，`wait` / `return_to_start` 仍然可达。
///
/// 这个不对称是**故意的**（合约 §2.2）。内核层的 fail-closed 只覆盖 `move_eef` /
/// `move_relative`；`set_gripper` / `wait` / `return_to_start` 由 Python body 抛
/// `ArxLedgerUnknown`，结果是 `act/result{ok:false}` 而不是 `act/refused`。它们必须留着，
/// 否则账本一旦丢失手臂就永远回不了家。谁要把这里"修"成全链保证，先读 §2.2。
#[tokio::test]
async fn s1_row_3_an_unknown_ledger_refuses_the_two_moves_and_leaves_the_escape_hatches_open() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    // 先把机器人站起来（seq 0 是正常的一局），再让账本变成未知（seq 1）。
    establish(&live, holding_left(0));
    establish(&live, unknown_ledger(1));
    assert_eq!(live.state.get("holding.left"), Some(Value::Null));

    let out = live
        .adm
        .propose(proposal(
            "p-unknown-eef",
            "arx_x5_move_eef",
            json!({"left": {"position": [0.01, 0.0, 0.30]}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    let evs = events(&live);
    assert!(
        trace_contains(&evs, "t-p-unknown-eef", "rule-carry-speed"),
        "trace 里该有 rule-carry-speed"
    );

    let out = live
        .adm
        .propose(proposal(
            "p-unknown-rel",
            "arx_x5_move_relative",
            json!({"left": {"dz": 0.01}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");

    // 三个逃生口里的两个（`reset` 在评测 lane 另有 `rule-mode` 挡着，见下一个用例）。
    let out = live
        .adm
        .propose(proposal("p-wait", "arx_x5_wait", json!({"steps": 10})))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "账本未知时 wait 必须仍然可达（合约 §2.2）: {out:?}"
    );
    let out = live
        .adm
        .propose(proposal(
            "p-home",
            "arx_x5_return_to_start",
            json!({"lift_m": 0.05, "max_steps": 150, "open_grippers": true}),
        ))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "账本未知时 return_to_start 必须仍然可达（合约 §2.2）: {out:?}"
    );
    assert_eq!(
        live.rec.names(),
        vec!["arx_x5_wait", "arx_x5_return_to_start"]
    );
}

// ================================================================ 6. rule-mode / rule-whitelist

/// `arx_x5_reset`：评测 lane 拒，探索 lane 放行。
///
/// trace 里**有** `rule-whitelist` **或** `rule-mode` 就够了——两条规则在评测 lane 都会
/// Deny（白名单里没有这个名字，且 `rule-mode` 是 evaluation），谁先谁后是注册顺序。
#[tokio::test]
async fn reset_is_refused_under_evaluation_and_admitted_under_exploration() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0));
    let out = live
        .adm
        .propose(proposal(
            "p-reset",
            "arx_x5_reset",
            json!({"reason": "wedged"}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "评测 lane: {out:?}");
    let evs = events(&live);
    assert!(
        trace_contains(&evs, "t-p-reset", "rule-whitelist")
            || trace_contains(&evs, "t-p-reset", "rule-mode"),
        "trace: {}",
        refusal(&evs, "t-p-reset").payload["trace"]
    );

    let tmp2 = tempfile::tempdir().unwrap();
    let live2 = boot(EXPLORE, tmp2.path());
    establish(&live2, holding_left(0));
    let out = live2
        .adm
        .propose(proposal(
            "p-reset",
            "arx_x5_reset",
            json!({"reason": "wedged"}),
        ))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "探索 lane 必须放行 reset: {out:?}"
    );
}

// ================================================================ 7. rule-finish-gate 与两条 lane

/// `finish(success)` 在夹爪没开、臂没回家时 `Ask`；两条 lane 对这个 Ask 的答复不同。
///
/// 评测 lane 的 `approval-auto{allow:false}` 把它变成 `act/refused{stage:"approval"}`；
/// 探索 lane 的 `allow:true` 让它过去。规则本身永远不 Deny——一次被 Deny 的 finish 会让
/// 这一局拿不到原生分数，而那正是这条路线存在的理由。
#[tokio::test]
async fn the_finish_gate_asks_and_the_two_lanes_answer_it_differently() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0)); // gripper_cmd 0.0, arms_home false

    let out = live
        .adm
        .propose(proposal(
            "p-finish",
            "arx_x5_finish",
            json!({"status": "success", "summary": "claimed too early"}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "评测 lane: {out:?}");
    assert_eq!(count_kind(&live, "act/ask"), 1, "{:?}", kinds(&live));
    let evs = events(&live);
    assert_eq!(refusal(&evs, "t-p-finish").payload["stage"], "approval");

    // 同一份账本、同一个提案，探索 lane 放行（`approval-auto{allow:true}` 答了那个 Ask）。
    let tmp2 = tempfile::tempdir().unwrap();
    let live2 = boot(EXPLORE, tmp2.path());
    establish(&live2, holding_left(0));
    let out = live2
        .adm
        .propose(proposal(
            "p-finish",
            "arx_x5_finish",
            json!({"status": "success", "summary": "claimed too early"}),
        ))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "探索 lane: {out:?}；日志 {:?}",
        kinds(&live2)
    );
    assert_eq!(count_kind(&live2, "act/ask"), 1);

    // 门槛满足：一路直过，没有 `act/ask`。
    let tmp3 = tempfile::tempdir().unwrap();
    let live3 = boot(EVAL, tmp3.path());
    establish(&live3, home_and_open(0));
    let out = live3
        .adm
        .propose(proposal(
            "p-finish-clean",
            "arx_x5_finish",
            json!({"status": "success", "summary": "arms home, grippers open"}),
        ))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");
    assert_eq!(count_kind(&live3, "act/ask"), 0, "{:?}", kinds(&live3));
}

// ================================================================ 8. rule-step-reserve

/// 原生预算见底：`move_eef` 被拒，`wait` 仍然放行（它在 `exempt` 里）。
#[tokio::test]
async fn the_step_reserve_refuses_a_move_and_still_admits_a_wait() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    // 手空（免得 carry/rotation 两条规则抢走这次拒绝的理由），只剩 100 步。
    establish(
        &live,
        ledger(
            0,
            json!({"left": 1.0, "right": 1.0}),
            json!({"left": false, "right": false}),
            100,
            json!({}),
        ),
    );
    assert_eq!(live.state.get("native_steps_remaining"), Some(json!(100)));

    let out = live
        .adm
        .propose(proposal(
            "p-broke",
            "arx_x5_move_eef",
            json!({"left": {"position": [0.01, 0.0, 0.30]}}),
        ))
        .await;
    assert!(matches!(out, Outcome::Refused { .. }), "{out:?}");
    let evs = events(&live);
    assert!(trace_contains(&evs, "t-p-broke", "rule-step-reserve"));

    let out = live
        .adm
        .propose(proposal("p-last-wait", "arx_x5_wait", json!({"steps": 8})))
        .await;
    assert!(
        matches!(out, Outcome::Executed { .. }),
        "wait 在 exempt 里: {out:?}"
    );
}

// ================================================================ 9. 两个蒸馏器

/// 预算标记一次一条，原生终止一次一条，而且**都不碰 FSM**。
///
/// `distill-native` 不驱动 FSM 边是合约 §4.3 的刻意设计：从世界日志观察者里推
/// `* → FAULT` 会同步走到准入的 `safety/fsm` 观察者，把那条还在能力体里跑着的执行撤销掉
/// ——也就是几乎每一局的最后一条原语。
#[tokio::test]
async fn the_budget_and_terminal_distillers_fire_once_each_and_leave_the_fsm_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    establish(&live, holding_left(0)); // remaining 800：一个标记都不该响
    assert_eq!(count_kind(&live, "health/native_budget"), 0);
    let standing = live.fsm.state();
    assert_eq!(standing, FsmState::Standing);

    // 跨过 300：恰好一条。
    establish(
        &live,
        ledger(
            1,
            json!({"left": 0.0, "right": 1.0}),
            json!({"left": true, "right": false}),
            300,
            json!({}),
        ),
    );
    assert_eq!(
        count_kind(&live, "health/native_budget"),
        1,
        "{:?}",
        kinds(&live)
    );
    // 299：已经锁住，不再响。
    establish(
        &live,
        ledger(
            2,
            json!({"left": 0.0, "right": 1.0}),
            json!({"left": true, "right": false}),
            299,
            json!({}),
        ),
    );
    assert_eq!(count_kind(&live, "health/native_budget"), 1);

    // 原生终止：恰好一条，FSM 原地不动。
    let fsm_events_before = count_kind(&live, "safety/fsm");
    establish(
        &live,
        ledger(
            3,
            json!({"left": 0.0, "right": 1.0}),
            json!({"left": true, "right": false}),
            0,
            json!({"native_terminal": true, "native_success": true}),
        ),
    );
    assert_eq!(count_kind(&live, "safety/native_terminal"), 1);
    establish(
        &live,
        ledger(
            4,
            json!({"left": 0.0, "right": 1.0}),
            json!({"left": true, "right": false}),
            0,
            json!({"native_terminal": true, "native_success": true}),
        ),
    );
    assert_eq!(count_kind(&live, "safety/native_terminal"), 1, "只该响一次");
    assert_eq!(live.fsm.state(), standing, "distill-native 不驱动 FSM 边");
    assert_eq!(
        count_kind(&live, "safety/fsm"),
        fsm_events_before,
        "原生终止不该产生任何 FSM 事件"
    );
}

// ================================================================ 10. 一条正常的短 recipe

/// 一条正常的短 recipe 全程被放行：没有一条 `act/refused`，也没有一条 `act/ask`。
///
/// 风险清单里最实在的一条是"fail-closed 会不会在正常流程上误拒"（设计文档 §5）。这个
/// 用例就是那条风险的证伪面：建立账本 → 闭合夹爪 → carry 搬运 → 静置 → 回臂 → 收尾。
#[tokio::test]
async fn a_normal_short_recipe_is_admitted_end_to_end_with_no_refusal() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(EVAL, tmp.path());
    // `arx_x5_vla_act` 建立账本之前，寄存器还是空的：第一条提案要先有账本。生产路径上
    // 这条事件由插件的 `register()` 发出（seq 0，`primitive: null`）。
    establish(&live, holding_left(0));

    let recipe: Vec<(&str, &str, Value)> = vec![
        (
            "r1",
            "arx_x5_vla_act",
            json!({"observation_id": "obs-1", "subgoal": "grasp the alarm clock",
                   "execute_steps": 12}),
        ),
        (
            "r2",
            "arx_x5_set_gripper",
            json!({"arm": "left", "opening": 0.0, "steps": 8}),
        ),
        (
            "r3",
            "arx_x5_move_relative",
            json!({"speed": 0.3, "left": {"dz": 0.02}, "max_steps": 30}),
        ),
        ("r4", "arx_x5_wait", json!({"steps": 12})),
        (
            "r5",
            "arx_x5_return_to_start",
            json!({"lift_m": 0.05, "max_steps": 150, "open_grippers": true}),
        ),
    ];
    for (id, cap, params) in &recipe {
        let out = live.adm.propose(proposal(id, cap, params.clone())).await;
        assert!(
            matches!(out, Outcome::Executed { .. }),
            "{cap} 该被放行: {out:?}；日志 {:?}",
            kinds(&live)
        );
    }

    // 回臂之后账本换成"手空 + 到家"，收尾才不会触发 finish gate 的 Ask。
    establish(&live, home_and_open(1));
    let out = live
        .adm
        .propose(proposal(
            "r6",
            "arx_x5_finish",
            json!({"status": "success", "summary": "alarm clock placed upright"}),
        ))
        .await;
    assert!(matches!(out, Outcome::Executed { .. }), "{out:?}");

    assert_eq!(
        live.rec.names(),
        vec![
            "arx_x5_vla_act",
            "arx_x5_set_gripper",
            "arx_x5_move_relative",
            "arx_x5_wait",
            "arx_x5_return_to_start",
            "arx_x5_finish"
        ]
    );
    for kind in ["act/refused", "act/ask", "act/clamped"] {
        assert_eq!(
            count_kind(&live, kind),
            0,
            "正常流程不该出现 {kind}: {:?}",
            kinds(&live)
        );
    }
    assert_eq!(count_kind(&live, "act/admitted"), 6);
}
