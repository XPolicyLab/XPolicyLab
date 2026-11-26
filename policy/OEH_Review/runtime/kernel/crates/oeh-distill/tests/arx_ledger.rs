//! `distill-arx-ledger` 的端到端测试：真内核、真世界日志、真寄存器、真 FSM。
//!
//! 第一行的 `use oeh_distill as _;` 是必需的——没有它，链接器会把整个 crate 丢掉，
//! `Kernel::boot` 也就找不到 `inventory` 注册的工厂。

use oeh_distill as _;
use oeh_distill::arx_ledger::{ArxLedgerPlugin, ID};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::plugin::Ctx;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::services::FiberId;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Kernel, Plugin, WorldLog};
use oeh_proto::{EventDraft, EventKind};
use serde_json::{json, Value};

fn base_rows() -> Vec<EntryRow> {
    vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
    ]
}

fn boot() -> Kernel {
    let mut rows = base_rows();
    rows.push(EntryRow::new(ID));
    let k = Kernel::boot(rows).unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    assert!(k.report().pending.is_empty(), "{:?}", k.report());
    k
}

/// CONTRACT §4.1 的完整载荷。
fn full(episode: &str, registration: i64, seq: i64) -> Value {
    json!({
        "schema": "oeh.arx_x5.ledger.v1",
        "episode_id": episode,
        "seq": seq,
        "registration": registration,
        "primitive": "arx_x5_move_relative",
        "outcome": "executed",
        "stop_reason": "reached",
        "executed_steps": 4,
        "gripper_cmd": {"left": 0.0, "right": 1.0},
        "gripper_measured": {"left": 0.71, "right": 0.99},
        "holding": {"left": true, "right": false},
        "native_step": 200,
        "native_step_limit": 1000,
        "native_steps_remaining": 800,
        "native_terminal": false,
        "native_success": false,
        "arms_home": false,
        "eef": {
            "left": {"position": [0.1, 0.2, 0.3], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "right": {"position": [0.4, 0.5, 0.6], "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0]}
        }
    })
}

fn append(log: &WorldLog, payload: Value) {
    log.append(EventDraft::new(
        EventKind::new("obs/arx_x5_ledger").unwrap(),
        "test",
        payload,
    ));
}

/// CONTRACT §4.4 里本蒸馏器拥有的全部键。
const OWNED_KEYS: &[&str] = &[
    "arms_home",
    "arx_episode_id",
    "arx_ledger_registration",
    "arx_ledger_seq",
    "eef_position.left",
    "eef_position.right",
    "eef_quaternion.left",
    "eef_quaternion.right",
    "gripper_cmd.left",
    "gripper_cmd.right",
    "gripper_measured.left",
    "gripper_measured.right",
    "holding.left",
    "holding.right",
    "native_step",
    "native_step_limit",
    "native_steps_remaining",
    "native_success",
    "native_terminal",
];

#[tokio::test]
async fn one_event_writes_every_contract_key_in_a_single_set_many() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    append(&log, full("ep-1", 0, 0));

    let entries = state.snapshot_entries();
    let mut stamps = std::collections::BTreeSet::new();
    for key in OWNED_KEYS {
        let e = entries
            .get(*key)
            .unwrap_or_else(|| panic!("missing register `{key}`"));
        assert_eq!(e.source, ID, "{key}");
        stamps.insert(e.t_kernel);
    }
    // 一次 `set_many` ⇒ 全部键共享同一个 `t_kernel`：规则读不到撕裂的账本。
    assert_eq!(stamps.len(), 1, "{stamps:?}");

    assert_eq!(state.get("gripper_cmd.left"), Some(json!(0.0)));
    assert_eq!(state.get("gripper_measured.left"), Some(json!(0.71)));
    assert_eq!(state.get("holding.left"), Some(json!(true)));
    assert_eq!(state.get("holding.right"), Some(json!(false)));
    assert_eq!(state.get("arms_home"), Some(json!(false)));
    assert_eq!(state.get("native_steps_remaining"), Some(json!(800)));
    assert_eq!(state.get("eef_position.left"), Some(json!([0.1, 0.2, 0.3])));
    assert_eq!(
        state.get("eef_quaternion.right"),
        Some(json!([0.0, 1.0, 0.0, 0.0]))
    );
    assert_eq!(state.get("arx_episode_id"), Some(json!("ep-1")));
    assert_eq!(state.get("arx_ledger_seq"), Some(json!(0)));
    assert_eq!(state.get("arx_ledger_registration"), Some(json!(0)));
}

#[tokio::test]
async fn a_null_channel_is_written_as_null_not_omitted() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let mut p = full("ep-1", 0, 0);
    p["gripper_cmd"]["left"] = Value::Null;
    p["holding"]["left"] = Value::Null;
    p["arms_home"] = Value::Null;
    append(&log, p);
    // null = 未知（fail-closed）。省略这些键会让规则读到上一局的旧值。
    assert_eq!(state.get("gripper_cmd.left"), Some(Value::Null));
    assert_eq!(state.get("holding.left"), Some(Value::Null));
    assert_eq!(state.get("arms_home"), Some(Value::Null));
    assert!(state.snapshot_entries().contains_key("gripper_cmd.left"));
    // 键缺席 = 未知的那一组：载荷里没有就不写。
    let mut q = full("ep-1", 0, 1);
    q["gripper_measured"] = json!({});
    q["eef"] = json!({});
    let k2 = boot();
    let log2 = k2.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state2 = k2.registry().get::<StateRegistry>(&"state".into()).unwrap();
    append(&log2, q);
    let snap = state2.snapshot_entries();
    assert!(!snap.contains_key("gripper_measured.left"));
    assert!(!snap.contains_key("eef_position.left"));
    assert!(snap.contains_key("gripper_cmd.left"));
}

#[tokio::test]
async fn ordering_guard_drops_stale_seq_but_accepts_a_reconnect() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();

    append(&log, full("ep-1", 0, 5));
    assert_eq!(state.get("arx_ledger_seq"), Some(json!(5)));

    // 同一局、同一次注册、更旧的 seq → 丢弃，寄存器不动。
    let mut stale = full("ep-1", 0, 3);
    stale["native_steps_remaining"] = json!(111);
    append(&log, stale);
    assert_eq!(state.get("arx_ledger_seq"), Some(json!(5)));
    assert_eq!(state.get("native_steps_remaining"), Some(json!(800)));

    // 重连：同一局，seq 回到 0，但 registration 加一 → **接受**。
    // 只按 seq 判重的守卫会让寄存器在重连后永远冻结。
    let mut reconnect = full("ep-1", 1, 0);
    reconnect["native_steps_remaining"] = json!(222);
    append(&log, reconnect);
    assert_eq!(state.get("arx_ledger_seq"), Some(json!(0)));
    assert_eq!(state.get("arx_ledger_registration"), Some(json!(1)));
    assert_eq!(state.get("native_steps_remaining"), Some(json!(222)));

    // 换一局同样重置水位。
    let mut other = full("ep-2", 0, 0);
    other["native_steps_remaining"] = json!(333);
    append(&log, other);
    assert_eq!(state.get("arx_episode_id"), Some(json!("ep-2")));
    assert_eq!(state.get("native_steps_remaining"), Some(json!(333)));
}

#[tokio::test]
async fn registers_are_written_before_boot_to_standing_and_only_once() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    assert_eq!(fsm.state(), FsmState::Boot);

    append(&log, full("ep-1", 0, 0));

    // 一次断言同时读寄存器和 FSM：`WorldLog::append` 只保证同一次 append 内观察者按注册
    // 顺序跑（`world_log.rs:110-111` 明确不保证**并发** append 之间的顺序），所以"寄存器
    // 先于迁移"这件事必须被断言出来，而不是默认成立。
    assert_eq!(
        (
            state.get("gripper_cmd.left"),
            state.get("native_steps_remaining"),
            fsm.state()
        ),
        (Some(json!(0.0)), Some(json!(800)), FsmState::Standing)
    );

    // 第二条事件不再迁移：`safety/fsm` 只有一条。
    append(&log, full("ep-1", 0, 1));
    let fsm_events: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "safety/fsm")
        .collect();
    assert_eq!(fsm_events.len(), 1, "{fsm_events:?}");
    assert_eq!(fsm_events[0].payload["to"], json!("STANDING"));
    assert_eq!(
        fsm_events[0].payload["reason"],
        json!("arx_x5 ledger established")
    );
    assert_eq!(fsm_events[0].source, ID);
}

#[tokio::test]
async fn a_malformed_payload_is_one_sampled_error_and_the_distiller_keeps_consuming() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();

    append(&log, json!({"episode_id": 7, "seq": 0, "registration": 0}));
    let errs: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "health/distiller_error")
        .collect();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(errs[0].payload["plugin"], json!(ID));
    assert_eq!(errs[0].payload["count"], json!(1));
    assert_eq!(
        (errs[0].salience, errs[0].collapse_key.as_deref()),
        (200, Some("distiller_error:distill-arx-ledger"))
    );
    assert!(state.get("arx_episode_id").is_none());

    // 坏事件之后仍然消费好事件——一条坏载荷不能让这条蒸馏器停摆。
    append(&log, full("ep-1", 0, 0));
    assert_eq!(state.get("arx_episode_id"), Some(json!("ep-1")));
}

#[tokio::test]
async fn the_observer_is_registered_once_and_removed_on_dispose() {
    // 基线：不带本行的内核。
    let a = Kernel::boot(base_rows()).unwrap();
    let log_a = a.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let n = log_a.observer_count();

    // 带本行的内核：正好多一个观察者。
    let b = boot();
    let log_b = b.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    assert_eq!(log_b.observer_count(), n + 1);

    // 撤销这一半在内核外面做：`Kernel` 既没有公开的 dispose，也没有会跑 disposer 的
    // `Drop`，所以直接在基线内核的服务注册表上手搭一个 `Ctx` 挂载插件再撤销它。
    let ctx = Ctx::new(FiberId(9_999), ID, a.registry().clone());
    ArxLedgerPlugin.apply(&ctx, &Value::Null).unwrap();
    assert_eq!(log_a.observer_count(), n + 1);
    ctx.take_effects_and_kill().dispose();
    assert_eq!(log_a.observer_count(), n);
}
