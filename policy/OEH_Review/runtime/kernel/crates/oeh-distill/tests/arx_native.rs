//! `distill-native` 的端到端测试。`use oeh_distill as _;` 是必需的。
//!
//! 最重要的一条断言是"FSM 没有动"：见 `arx_native.rs` 的模块文档（CONTRACT §4.3）——
//! 从世界日志观察者里驱动 `* → FAULT` 会同步撤销正卡在发事件那个能力体里的执行。

use oeh_distill as _;
use oeh_distill::arx_native::{ArxNativePlugin, ID};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::plugin::Ctx;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::services::FiberId;
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

fn ledger(episode: &str, native_step: i64, terminal: bool, success: bool) -> Value {
    json!({
        "schema": "oeh.arx_x5.ledger.v1",
        "episode_id": episode,
        "seq": native_step,
        "registration": 0,
        "native_step": native_step,
        "native_terminal": terminal,
        "native_success": success,
    })
}

fn append(log: &WorldLog, payload: Value) {
    log.append(EventDraft::new(
        EventKind::new("obs/arx_x5_ledger").unwrap(),
        "test",
        payload,
    ));
}

fn distilled(log: &WorldLog) -> Vec<String> {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() != "obs/arx_x5_ledger")
        .map(|e| format!("{}:{}", e.kind, e.payload))
        .collect()
}

#[tokio::test]
async fn terminal_fires_once_per_episode_and_never_touches_the_fsm() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();

    append(&log, ledger("ep-1", 10, false, false));
    assert!(distilled(&log).is_empty());

    append(&log, ledger("ep-1", 830, true, true));
    append(&log, ledger("ep-1", 831, true, true)); // 已锁存 → 不再发
    append(&log, ledger("ep-2", 42, true, false)); // 新一局 → 再发一条

    assert_eq!(
        distilled(&log),
        vec![
            r#"safety/native_terminal:{"episode_id":"ep-1","native_step":830,"native_success":true,"reason":"native success"}"#.to_string(),
            r#"safety/native_terminal:{"episode_id":"ep-2","native_step":42,"native_success":false,"reason":"terminated or truncated"}"#.to_string(),
        ]
    );
    let ev = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "safety/native_terminal")
        .unwrap();
    assert_eq!(
        (ev.salience, ev.collapse_key.as_deref(), ev.source.as_str()),
        (240, Some("native_terminal"), ID)
    );

    // 这条蒸馏器**不驱动任何 FSM 边**（CONTRACT §4.3）。别把它"修"成会转 FAULT。
    assert_eq!(fsm.state(), FsmState::Boot);
    assert!(
        !log.since(0)
            .unwrap()
            .iter()
            .any(|e| e.kind.as_str() == "safety/fsm"),
        "distill-native must drive no FSM edge"
    );
}

#[tokio::test]
async fn a_malformed_payload_is_one_sampled_error_and_the_distiller_keeps_consuming() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    append(&log, json!({"episode_id": 7, "native_terminal": true}));
    let errs: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "health/distiller_error")
        .collect();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(
        (errs[0].salience, errs[0].collapse_key.as_deref()),
        (200, Some("distiller_error:distill-native"))
    );
    append(&log, ledger("ep-1", 5, true, false));
    assert_eq!(
        log.since(0)
            .unwrap()
            .iter()
            .filter(|e| e.kind.as_str() == "safety/native_terminal")
            .count(),
        1
    );
}

#[tokio::test]
async fn the_observer_is_registered_once_and_removed_on_dispose() {
    let a = Kernel::boot(base_rows()).unwrap();
    let log_a = a.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let n = log_a.observer_count();
    let b = boot();
    let log_b = b.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    assert_eq!(log_b.observer_count(), n + 1);
    let ctx = Ctx::new(FiberId(9_999), ID, a.registry().clone());
    ArxNativePlugin.apply(&ctx, &Value::Null).unwrap();
    assert_eq!(log_a.observer_count(), n + 1);
    ctx.take_effects_and_kill().dispose();
    assert_eq!(log_a.observer_count(), n);
}
