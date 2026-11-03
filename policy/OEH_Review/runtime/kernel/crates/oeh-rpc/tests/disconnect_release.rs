//! 断连即释放（Plan 3 遗留 I4）。
//!
//! 一个远端执行器所在的连接关掉，意味着**再没有人在看着机器人动**：内核既收不到结果，
//! 也送不出取消，连一声"停"都没地方喊。所以这条路不等租约、不等宽限期、也不叫
//! `Executor::stop`——在跑的执行当场判 `Cancelled{reason:"adapter disconnected"}`，
//! motion 的位子当场空出来，运维那一侧另有一条 `safety/adapter_lost`。
//!
//! 时间在这里是承重结构，但**不是判据**：`lease_grace_ms: 5000` 和 `lease_ttl_s: 5.0`
//! 的作用是让"走了租约那条路"和"走了断连这条路"在**语义**上分得开（那条路记的是
//! `act/lease_expired{reason:"lease expired"}`），而不是靠掐表。

mod common;

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

/// 远端适配器：一个 motion 能力 `wave`，租约 5 s（够长，测试期内不会自己过期）。
fn manifest() -> Value {
    json!({
        "manifest": {"name": "py-arm", "inject": [], "provide": [], "streams": []},
        "capabilities": [{"name": "wave", "class": "motion", "risk_level": "low",
                          "params_schema": {"type": "object"}, "lease_ttl_s": 5.0}],
        "heartbeat_ms": 10_000
    })
}

/// 世界日志里某一类事件的全部 payload。
fn events(kind: &str) -> Vec<Value> {
    common::world_log()
        .since(0)
        .unwrap_or_default()
        .iter()
        .filter(|e| e.kind.as_str() == kind)
        .map(|e| e.payload.clone())
        .collect()
}

async fn next_turn(
    c: &oeh_rpc::client::Client,
    n: &mut tokio::sync::broadcast::Receiver<oeh_proto::rpc::Notification>,
) -> u64 {
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    common::next_notification(n, "brain/turn").await["turn"]
        .as_u64()
        .unwrap()
}

/// 一个只接受、永不回结果的远端执行器，外加一条脑连接开出来的第一个 turn。
/// 返回 `(适配器连接, 脑的通知流, turn 号)`。
async fn adapter_and_turn(
    brain: &oeh_rpc::client::Client,
) -> (
    oeh_rpc::client::Client,
    tokio::sync::broadcast::Receiver<oeh_proto::rpc::Notification>,
    u64,
) {
    // 适配器在**另一条**连接上：断的是它，而脑那一条要活着看结果。
    let adapter = common::connect().await;
    adapter.on_request(
        "capability/execute",
        // 接下来就没动静了：一个正在让机器人动、还没回结果的执行器。
        Arc::new(|_p: Value| Box::pin(async { Ok(json!({"accepted": true})) })),
    );
    adapter.call("plugin/register", manifest()).await.unwrap();

    let mut n = brain.notifications();
    // 唤醒预算写死成"不限"：这些测试要不止一个 turn，而默认的 `min_interval_ms: 500` 会
    // 把第二次唤醒压掉——那样测的就成了"两次 event/emit 之间隔了多久"，一个跟被测的东西
    // 毫无关系的墙钟条件（和 `router.rs` 里那个两 turn 的测试同一个理由）。
    brain
        .call(
            "brain/attach",
            json!({"brain_id": "fast",
                   "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"},
                                   "budget": {"max_consecutive_wakes": 100, "min_interval_ms": 0}}}),
        )
        .await
        .unwrap();
    let turn = next_turn(brain, &mut n).await;
    (adapter, n, turn)
}

/// 谁先发现断连**不该改变结局**。
///
/// `Conn::mark_closed` 先唤醒 `wait_closed`、再跑 `on_close` 钩子，所以在多核上这两条路
/// 是真并发：钩子可能抢到那道"只扫一次"的闸门之后被调度走，而执行 future 先跑完自己的
/// 收尾。修复之前，那个交错里这次执行会以 `Failed{"connection lost"}` 收场，而且
/// **一条 `safety/adapter_lost` 都不会有**——"一台机器人动到一半，看着它的那一头没了"
/// 这件事在审计里就此没有记录，事后也重建不出来（Fix round 1，Important #1）。
///
/// 所以这条测试用多线程 runtime 让两条路真并发，然后断言的是一个**与交错无关**的结论：
/// 不管谁先，结局都是 `Cancelled{"adapter disconnected"}`，事件不多不少正好一条。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_outcome_does_not_depend_on_who_notices_the_disconnect_first() {
    let (brain, kh, run) = common::boot_with(|rows| {
        for r in rows.iter_mut() {
            if r.id == "admission" {
                r.config["lease_grace_ms"] = json!(5000);
            }
        }
    })
    .await;
    let (adapter, _n, turn) = adapter_and_turn(&brain).await;

    let b2 = brain.clone();
    let propose = tokio::spawn(async move {
        b2.call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "wave", "params": {}}),
        )
        .await
        .unwrap()
    });
    common::await_condition("the remote motion started", || {
        !events("act/started").is_empty()
    })
    .await;
    let pid = events("act/started")[0]["proposal_id"]
        .as_str()
        .unwrap()
        .to_string();

    adapter.close().await;

    let out = tokio::time::timeout(Duration::from_secs(5), propose)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "cancelled", "{out}");
    assert_eq!(out["outcome"]["reason"], "adapter disconnected", "{out}");

    common::await_condition("safety/adapter_lost", || {
        !events("safety/adapter_lost").is_empty()
    })
    .await;
    let lost = events("safety/adapter_lost");
    assert_eq!(
        lost.len(),
        1,
        "exactly one, whoever got there first: {lost:?}"
    );
    assert_eq!(lost[0]["executions"], json!([pid]), "{lost:?}");
    let cancelled = events("act/cancelled");
    assert_eq!(cancelled.len(), 1, "{cancelled:?}");
    assert_eq!(
        cancelled[0]["reason"], "adapter disconnected",
        "{cancelled:?}"
    );

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn closing_the_adapter_connection_cancels_its_executions_and_frees_the_motion_slot() {
    // `lease_grace_ms: 5000`：这条路要是还在等宽限期，下面每一条断言都会以超时收场。
    let (brain, kh, run) = common::boot_with(|rows| {
        for r in rows.iter_mut() {
            if r.id == "admission" {
                r.config["lease_grace_ms"] = json!(5000);
            }
        }
    })
    .await;

    let (adapter, mut n, turn) = adapter_and_turn(&brain).await;
    let b2 = brain.clone();
    let propose = tokio::spawn(async move {
        b2.call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "wave", "params": {}}),
        )
        .await
        .unwrap()
    });

    // 等到执行真的在飞（`act/started` 已经记下）再断连：断的早晚正是被测的那件事，
    // 睡一个固定的窗口只是在赌调度。
    common::await_condition("the remote motion started", || {
        !events("act/started").is_empty()
    })
    .await;
    let started = events("act/started");
    let pid = started[0]["proposal_id"].as_str().unwrap().to_string();

    adapter.close().await;

    // 脑看到的结局：`Cancelled{reason}`，不是 `Failed{error}`——机器人没有"做失败"，
    // 是它的执行器不在了。
    let out = tokio::time::timeout(Duration::from_secs(5), propose)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "cancelled", "{out}");
    assert_eq!(out["outcome"]["reason"], "adapter disconnected", "{out}");

    // 审计那一侧：终局事件 + 一条给运维的 `safety/adapter_lost`。
    common::await_condition("act/cancelled and safety/adapter_lost", || {
        events("act/cancelled")
            .iter()
            .any(|p| p["reason"] == "adapter disconnected")
            && !events("safety/adapter_lost").is_empty()
    })
    .await;
    let cancelled = events("act/cancelled");
    assert_eq!(cancelled.len(), 1, "{cancelled:?}");
    assert_eq!(cancelled[0]["proposal_id"], json!(pid), "{cancelled:?}");
    let lost = events("safety/adapter_lost");
    assert_eq!(lost.len(), 1, "{lost:?}");
    assert_eq!(lost[0]["plugin"], "py-arm", "{lost:?}");
    assert_eq!(lost[0]["executions"], json!([pid]), "{lost:?}");
    // 租约那条路没有被走过：走过的话记的是 `act/lease_expired`，而且要等 5 s。
    assert!(events("act/lease_expired").is_empty());

    // 位子真的空出来了。判据是**下一个 motion 有没有抢占谁**：位子还占着的话，
    // `walk` 会把它从上一个执行手里接过来并记一条 `act/preempted`。
    // （`say` 这类非 motion 能力从来就不碰这个位子，证明不了任何事。）
    //
    // 先关掉上一个 turn：一个脑在当前 turn 结束之前不会被开出新 turn（唤醒被 latch 住）。
    brain
        .call(
            "turn/done",
            json!({"brain_id": "fast", "turn": turn, "reason": "adapter gone"}),
        )
        .await
        .unwrap();
    let turn = next_turn(&brain, &mut n).await;
    let out = brain
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "walk",
                   "params": {"vx": 0.1, "duration_s": 0.05}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "executed", "{out}");
    assert!(
        events("act/preempted").is_empty(),
        "the motion slot was still held"
    );

    let st = brain.call("robot/state", json!({})).await.unwrap();
    assert_eq!(st["fsm"], "STANDING", "{st}");
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
