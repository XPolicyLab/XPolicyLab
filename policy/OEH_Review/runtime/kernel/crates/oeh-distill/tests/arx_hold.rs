//! `distill-hold` 的端到端测试。第一行的 `use oeh_distill as _;` 是必需的——没有它工厂
//! 注册会被链接器丢掉。

use oeh_distill as _;
use oeh_distill::arx_hold::{ArxHoldPlugin, ID};
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

/// 契约里那一个持物谓词（`oeh_vla.arx_ledger.holding`）：命令 < 0.5 **且**测量 > 0.05。
///
/// 真实事件里的 `holding` 就是 Python 侧用它算好写进来的，所以这里也用它算，而不是让每个
/// 用例手写一个布尔——手写的那一份迟早会和被测的那一份漂开，而"两处各算一遍持物"正是这条
/// 蒸馏器出过的那个洞。命令或测量未知时给 `null`（未知），与真实生产者一致。
fn contract_holding(cmd: &Value, measured: &Value) -> Value {
    match (cmd.as_f64(), measured.as_f64()) {
        (Some(c), Some(m)) if c.is_finite() && m.is_finite() => json!(c < 0.5 && m > 0.05),
        _ => Value::Null,
    }
}

/// 一条账本事件，只带 `distill-hold` 关心的字段；`holding` 按契约谓词从同一条读数算出。
fn ledger(episode: &str, native_step: i64, left: (Value, Value), right: (Value, Value)) -> Value {
    let held = (
        contract_holding(&left.0, &left.1),
        contract_holding(&right.0, &right.1),
    );
    ledger_holding(episode, native_step, left, right, held)
}

/// 同上，但 `holding` 由用例显式给出——用来钉住"未知的 `holding` 保持上一次的锁存"。
fn ledger_holding(
    episode: &str,
    native_step: i64,
    left: (Value, Value),
    right: (Value, Value),
    holding: (Value, Value),
) -> Value {
    json!({
        "schema": "oeh.arx_x5.ledger.v1",
        "episode_id": episode,
        "seq": native_step,
        "registration": 0,
        "gripper_cmd": {"left": left.0, "right": right.0},
        "gripper_measured": {"left": left.1, "right": right.1},
        "holding": {"left": holding.0, "right": holding.1},
        "native_step": native_step,
    })
}

fn append(log: &WorldLog, payload: Value) {
    log.append(EventDraft::new(
        EventKind::new("obs/arx_x5_ledger").unwrap(),
        "test",
        payload,
    ));
}

/// 世界日志里除输入事件以外的全部事件，`kind:payload` 形式——一次断言同时钉住顺序、
/// 条数和"只发边沿"。
fn distilled(log: &WorldLog) -> Vec<String> {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() != "obs/arx_x5_ledger")
        .map(|e| format!("{}:{}", e.kind, e.payload))
        .collect()
}

#[tokio::test]
async fn the_rpent_edge_sequence_slips_then_goes_empty_and_then_stays_quiet() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    // 右臂全程张开（命令 1.0）：既不会"抓空"也不会"打滑"，所以序列里只应有左臂。
    let right = (json!(1.0), json!(0.99));

    append(
        &log,
        ledger("ep-1", 100, (json!(0.0), json!(0.71)), right.clone()),
    );
    append(
        &log,
        ledger("ep-1", 101, (json!(0.0), json!(0.55)), right.clone()),
    );
    append(
        &log,
        ledger("ep-1", 102, (json!(0.0), json!(0.02)), right.clone()),
    );
    append(
        &log,
        ledger("ep-1", 103, (json!(0.0), json!(0.02)), right.clone()),
    );

    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/hold_slip:{"arm":"left","drop":0.15999999999999992,"gripper_cmd":0.0,"gripper_measured":0.55,"native_step":101,"previous":0.71}"#.to_string(),
            r#"obs/grasp_empty:{"arm":"left","gripper_cmd":0.0,"gripper_measured":0.02,"native_step":102}"#.to_string(),
        ]
    );

    let empty = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "obs/grasp_empty")
        .unwrap();
    assert_eq!(
        (
            empty.salience,
            empty.collapse_key.as_deref(),
            empty.source.as_str()
        ),
        (210, Some("grasp_empty:left"), ID)
    );
    let slip = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "obs/hold_slip")
        .unwrap();
    assert_eq!(
        (
            slip.salience,
            slip.collapse_key.as_deref(),
            slip.source.as_str()
        ),
        (190, Some("hold_slip:left"), ID)
    );

    // 寄存器写的是"这条事件的判定"：最后一条事件上仍然是抓空，但不再叫醒脑。
    assert_eq!(state.get("grasp_empty.left"), Some(json!(true)));
    assert_eq!(state.get("hold_slip.left"), Some(json!(false)));
    assert_eq!(state.get("grasp_empty.right"), Some(json!(false)));
    assert_eq!(
        state
            .snapshot_entries()
            .get("grasp_empty.left")
            .unwrap()
            .source,
        ID
    );
}

#[tokio::test]
async fn empty_wins_over_slip_on_the_same_event() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let right = (json!(1.0), json!(0.99));
    // 0.71 → 0.01 同时满足"下降 > 0.1"和"塌到 0.05 以下"；只应报抓空。
    append(
        &log,
        ledger("ep-1", 1, (json!(0.0), json!(0.71)), right.clone()),
    );
    append(&log, ledger("ep-1", 2, (json!(0.0), json!(0.01)), right));
    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/grasp_empty:{"arm":"left","gripper_cmd":0.0,"gripper_measured":0.01,"native_step":2}"#.to_string()
        ]
    );
}

#[tokio::test]
async fn an_unknown_channel_skips_the_arm_without_clearing_the_latch() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let right = (json!(1.0), json!(0.99));
    append(
        &log,
        ledger("ep-1", 1, (json!(0.0), json!(0.71)), right.clone()),
    );
    // 测量未知 → 这条臂整条跳过：不发事件，也不把 0.71 这个锁存抹掉。
    append(
        &log,
        ledger("ep-1", 2, (json!(0.0), Value::Null), right.clone()),
    );
    assert!(distilled(&log).is_empty());
    // 锁存还在 0.71，所以这一条仍然是相对 0.71 的下降。
    append(&log, ledger("ep-1", 3, (json!(0.0), json!(0.55)), right));
    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/hold_slip:{"arm":"left","drop":0.15999999999999992,"gripper_cmd":0.0,"gripper_measured":0.55,"native_step":3,"previous":0.71}"#.to_string()
        ]
    );
}

#[tokio::test]
async fn a_new_episode_clears_both_latches() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let right = (json!(1.0), json!(0.99));
    append(
        &log,
        ledger("ep-1", 1, (json!(0.0), json!(0.02)), right.clone()),
    );
    // 同一局重复 → 边沿被抑制。
    append(
        &log,
        ledger("ep-1", 2, (json!(0.0), json!(0.02)), right.clone()),
    );
    // 换一局 → 锁存清空，同样的读数重新是一条边沿。
    append(&log, ledger("ep-2", 3, (json!(0.0), json!(0.02)), right));
    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/grasp_empty:{"arm":"left","gripper_cmd":0.0,"gripper_measured":0.02,"native_step":1}"#.to_string(),
            r#"obs/grasp_empty:{"arm":"left","gripper_cmd":0.0,"gripper_measured":0.02,"native_step":3}"#.to_string(),
        ]
    );
}

#[tokio::test]
async fn a_malformed_payload_is_one_sampled_error_and_the_distiller_keeps_consuming() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    append(&log, json!({"episode_id": 7}));
    let errs: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "health/distiller_error")
        .collect();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(
        (errs[0].salience, errs[0].collapse_key.as_deref()),
        (200, Some("distiller_error:distill-hold"))
    );
    assert_eq!(errs[0].payload["plugin"], json!(ID));
    // 之后仍然消费好事件。
    append(
        &log,
        ledger(
            "ep-1",
            1,
            (json!(0.0), json!(0.02)),
            (json!(1.0), json!(0.99)),
        ),
    );
    assert_eq!(state.get("grasp_empty.left"), Some(json!(true)));
}

#[tokio::test]
async fn the_observer_is_registered_once_and_removed_on_dispose() {
    let a = Kernel::boot(base_rows()).unwrap();
    let log_a = a.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let n = log_a.observer_count();
    let b = boot();
    let log_b = b.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    assert_eq!(log_b.observer_count(), n + 1);
    // `Kernel` 没有公开的 dispose 也没有会跑 disposer 的 `Drop`，所以撤销这一半在
    // 基线内核的服务注册表上手搭 `Ctx` 做。
    let ctx = Ctx::new(FiberId(9_999), ID, a.registry().clone());
    ArxHoldPlugin.apply(&ctx, &Value::Null).unwrap();
    assert_eq!(log_a.observer_count(), n + 1);
    ctx.take_effects_and_kill().dispose();
    assert_eq!(log_a.observer_count(), n);
}

// -- "持物中"读的是账本里的 `holding`，不是重算出来的 ---------------------------------------

#[tokio::test]
async fn a_light_squeeze_on_a_wide_object_still_reports_the_slip() {
    // 宽物体的正常握法：`arx_x5_set_gripper(opening=0.40)`。命令 0.40 < 0.5，测量 0.71 > 0.05，
    // 所以契约谓词说"持物中"，`rule-carry-speed` 会夹速度、`rule-hold-no-rotation` 会拒转腕——
    // 整条路线的其余部分都认为机器人正夹着东西。
    //
    // 这条蒸馏器一度自己重算持物（`cmd < empty_cmd_below`，也就是 0.3），于是 0.40 的轻夹
    // 在它眼里不算持物，0.71 → 0.55（下降 0.16 > 0.1）的滑脱一条事件都不发，脑永远不会被
    // 叫醒。现在它读账本里那一个权威字段，所以这一条必须发出来。
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let right = (json!(1.0), json!(0.99));

    let first = ledger("ep-1", 1, (json!(0.40), json!(0.71)), right.clone());
    assert_eq!(
        first["holding"]["left"],
        json!(true),
        "0.40 的命令按契约谓词就是持物中；这条前提塌了下面的断言就没有意义"
    );
    append(&log, first);
    append(
        &log,
        ledger("ep-1", 2, (json!(0.40), json!(0.55)), right.clone()),
    );

    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/hold_slip:{"arm":"left","drop":0.15999999999999992,"gripper_cmd":0.4,"gripper_measured":0.55,"native_step":2,"previous":0.71}"#.to_string()
        ]
    );
    assert_eq!(state.get("hold_slip.left"), Some(json!(true)));
    assert_eq!(state.get("grasp_empty.left"), Some(json!(false)));
}

#[tokio::test]
async fn a_command_between_the_two_thresholds_is_not_a_grasp_empty() {
    // 0.45 的命令：按持物谓词（< 0.5）是持物，按抓空谓词（< 0.3）不是。两个阈值是两件事，
    // 这条用例把它们分开钉住——否则把 `empty_cmd_below` 调到 0.5 也能让上一条通过。
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let right = (json!(1.0), json!(0.99));
    append(
        &log,
        ledger("ep-1", 1, (json!(0.45), json!(0.71)), right.clone()),
    );
    append(&log, ledger("ep-1", 2, (json!(0.45), json!(0.02)), right));
    // 测量塌到 0.02，但命令 0.45 不在"抓空"的命令区间里 → 不是抓空；持物中掉了 0.69 → 打滑。
    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/hold_slip:{"arm":"left","drop":0.69,"gripper_cmd":0.45,"gripper_measured":0.02,"native_step":2,"previous":0.71}"#.to_string()
        ]
    );
    assert_eq!(state.get("grasp_empty.left"), Some(json!(false)));
}

#[tokio::test]
async fn an_unknown_holding_keeps_the_previous_latch() {
    // `holding.<arm>` 为 null 是"不知道"，不是"没持物"：读不到的一条观测不该把已知的状态
    // 抹掉，和"命令或测量未知就跳过这条臂"是同一条纪律。
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let right = (json!(1.0), json!(0.99));
    append(
        &log,
        ledger("ep-1", 1, (json!(0.0), json!(0.71)), right.clone()),
    );
    // holding 未知的一条：锁存保持 true，测量照常更新到 0.70。
    append(
        &log,
        ledger_holding(
            "ep-1",
            2,
            (json!(0.0), json!(0.70)),
            right.clone(),
            (Value::Null, Value::Null),
        ),
    );
    assert!(distilled(&log).is_empty(), "0.01 的下降不到 slip_delta");
    append(&log, ledger("ep-1", 3, (json!(0.0), json!(0.55)), right));
    assert_eq!(
        distilled(&log),
        vec![
            r#"obs/hold_slip:{"arm":"left","drop":0.1499999999999999,"gripper_cmd":0.0,"gripper_measured":0.55,"native_step":3,"previous":0.7}"#.to_string()
        ]
    );
}

#[tokio::test]
async fn an_explicit_not_holding_suppresses_the_slip() {
    // 反向控制：同一组读数，只把权威的 `holding` 翻成 false，就不该再有 `hold_slip`。
    // 这一条证明上面那几条走的确实是账本字段，而不是又一次重算出来的谓词。
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let right = (json!(1.0), json!(0.99));
    append(
        &log,
        ledger_holding(
            "ep-1",
            1,
            (json!(0.40), json!(0.71)),
            right.clone(),
            (json!(false), json!(false)),
        ),
    );
    append(
        &log,
        ledger_holding(
            "ep-1",
            2,
            (json!(0.40), json!(0.55)),
            right,
            (json!(false), json!(false)),
        ),
    );
    assert!(distilled(&log).is_empty());
}
