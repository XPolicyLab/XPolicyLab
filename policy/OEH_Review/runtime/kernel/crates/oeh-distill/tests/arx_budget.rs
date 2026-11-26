//! `distill-budget` 的端到端测试。`use oeh_distill as _;` 是必需的。

use oeh_distill as _;
use oeh_distill::arx_budget::{ArxBudgetPlugin, ID};
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

fn ledger(episode: &str, remaining: i64) -> Value {
    json!({
        "schema": "oeh.arx_x5.ledger.v1",
        "episode_id": episode,
        "seq": remaining,
        "registration": 0,
        "native_step_limit": 1000,
        "native_steps_remaining": remaining,
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
async fn each_mark_fires_once_per_episode_and_a_new_episode_rearms_them() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();

    append(&log, ledger("ep-1", 320)); // 还没到任何水位
    assert!(distilled(&log).is_empty());
    append(&log, ledger("ep-1", 300)); // 300 触发
    append(&log, ledger("ep-1", 299)); // 300 已锁存，150/120 还没到 → 无
    append(&log, ledger("ep-1", 150)); // 150 触发
    append(&log, ledger("ep-2", 150)); // 换局重新武装：300 与 150 都补发

    assert_eq!(
        distilled(&log),
        vec![
            r#"health/native_budget:{"mark":300,"native_step_limit":1000,"native_steps_remaining":300}"#.to_string(),
            r#"health/native_budget:{"mark":150,"native_step_limit":1000,"native_steps_remaining":150}"#.to_string(),
            r#"health/native_budget:{"mark":300,"native_step_limit":1000,"native_steps_remaining":150}"#.to_string(),
            r#"health/native_budget:{"mark":150,"native_step_limit":1000,"native_steps_remaining":150}"#.to_string(),
        ]
    );
    let ev = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "health/native_budget")
        .unwrap();
    assert_eq!(
        (ev.salience, ev.collapse_key.as_deref(), ev.source.as_str()),
        (200, Some("native_budget"), ID)
    );
}

#[tokio::test]
async fn a_single_drop_past_several_marks_backfills_them_largest_first() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    append(&log, ledger("ep-1", 100));
    assert_eq!(
        distilled(&log),
        vec![
            r#"health/native_budget:{"mark":300,"native_step_limit":1000,"native_steps_remaining":100}"#.to_string(),
            r#"health/native_budget:{"mark":150,"native_step_limit":1000,"native_steps_remaining":100}"#.to_string(),
            r#"health/native_budget:{"mark":120,"native_step_limit":1000,"native_steps_remaining":100}"#.to_string(),
        ]
    );
}

#[tokio::test]
async fn a_missing_remaining_count_is_simply_not_actionable() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    append(
        &log,
        json!({"episode_id": "ep-1", "seq": 0, "registration": 0}),
    );
    assert!(distilled(&log).is_empty());
}

#[tokio::test]
async fn a_malformed_payload_is_one_sampled_error_and_the_distiller_keeps_consuming() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    append(
        &log,
        json!({"episode_id": "ep-1", "native_steps_remaining": "soon"}),
    );
    let errs: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "health/distiller_error")
        .collect();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(
        (errs[0].salience, errs[0].collapse_key.as_deref()),
        (200, Some("distiller_error:distill-budget"))
    );
    append(&log, ledger("ep-1", 300));
    assert_eq!(
        distilled(&log).last().map(String::as_str),
        Some(
            r#"health/native_budget:{"mark":300,"native_step_limit":1000,"native_steps_remaining":300}"#
        )
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
    ArxBudgetPlugin.apply(&ctx, &Value::Null).unwrap();
    assert_eq!(log_a.observer_count(), n + 1);
    ctx.take_effects_and_kill().dispose();
    assert_eq!(log_a.observer_count(), n);
}
