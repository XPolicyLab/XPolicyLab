//! `brain/inject{brain_id, text} -> {seq}`：把一句话塞进一个脑的上下文。
//!
//! 和 `turn/steer` 是一对：两者都是**定向**事件（`source == "brains"`，绕过目标脑的
//! policy 和唤醒预算），走的也是同一条 `(action, status)` 矩阵。差别只有一个，而且全在
//! 脑那一侧：`brain/inject` 的观测带着 `wake_reason.kind == "brain/inject"`，原生脑据此
//! 把它投进 `Lane::Inject`（"捎带上的上下文"）而不是 `Lane::Steer`（"下一步照它办"）。
//!
//! 错误的分法和边上的 `turn/steer` 逐字相同：点不到那个脑 → `NOT_ATTACHED`，文本空或
//! 超限 → `INVALID_PARAMS`。同一个问题在两个相邻的方法上不能有两种答案。

mod common;

use oeh_kernel::MAX_INJECT_TEXT;
use serde_json::json;

/// 正好越界一个字节。上限本身在内核那边（[`MAX_INJECT_TEXT`]），这里不复述那个数字。
const TOO_LONG: usize = MAX_INJECT_TEXT + 1;

#[tokio::test]
async fn an_injection_reaches_the_live_turn_as_a_brain_inject_steer() {
    let (c, kh, run) = common::boot().await;
    let mut n = c.notifications();
    // 预算写死成"不限"：定向事件不受预算约束，但把脑叫进第一个 turn 的那条
    // `obs/person` 是普通唤醒，默认的 `min_interval_ms: 500` 会让它和真实时钟赛跑。
    c.call(
        "brain/attach",
        json!({"brain_id": "fast",
               "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"},
                               "budget": {"max_consecutive_wakes": 100, "min_interval_ms": 0}}}),
    )
    .await
    .unwrap();

    // 先把脑叫进一个 turn：Idle 的脑收到定向事件等同 wake（走 `brain/turn`），
    // `brain/steer` 通知只在投进一个**活着的** turn 时才有。
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    common::next_notification(&mut n, "brain/turn").await;

    let r = c
        .call(
            "brain/inject",
            json!({"brain_id": "fast", "text": "look left"}),
        )
        .await
        .unwrap();
    let seq = r["seq"].as_u64().expect("result carries the event seq");

    let m = common::next_notification(&mut n, "brain/steer").await;
    assert_eq!(m["brain_id"], "fast");
    assert_eq!(
        m["observation"]["wake_reason"]["kind"], "brain/inject",
        "脑要分得出这是注入还是 steer"
    );
    assert_eq!(m["observation"]["wake_reason"]["seq"], json!(seq));

    // 世界日志里那条事件：定向事件的 source 永远是 `brains`（`event/emit` 伪造不出来），
    // payload 带着注入方的身份。
    let log = common::world_log();
    let ev = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "brain/inject")
        .expect("brain/inject 进了世界日志");
    assert_eq!(ev.seq, seq);
    assert_eq!(ev.source, "brains");
    assert_eq!(ev.payload["brain_id"], "fast");
    assert_eq!(ev.payload["text"], "look left");
    assert_eq!(ev.payload["source"], "rpc:fast");
    assert_eq!(ev.salience, 150);
    assert_eq!(ev.trace_id, None, "不是 turn 内事件，不挂 trace_id");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_missing_brain_is_not_attached_and_a_bad_text_is_invalid_params() {
    let (c, kh, run) = common::boot().await;
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
    )
    .await
    .unwrap();

    // 点不到那个脑：和 `turn/steer` 同一个答案。
    let e = c
        .call("brain/inject", json!({"brain_id": "nobody", "text": "hi"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED, "{e:?}");
    let steer = c
        .call("turn/steer", json!({"brain_id": "nobody", "text": "hi"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, steer.code, "两个相邻的方法不能各答各的");

    let e = c
        .call("brain/inject", json!({"brain_id": "fast", "text": ""}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS, "{e:?}");

    let e = c
        .call(
            "brain/inject",
            json!({"brain_id": "fast", "text": "x".repeat(TOO_LONG)}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS, "{e:?}");

    // 边界的另一侧：正好 4 KiB 是合法的。
    c.call(
        "brain/inject",
        json!({"brain_id": "fast", "text": "x".repeat(TOO_LONG - 1)}),
    )
    .await
    .unwrap();

    // 三次拒绝一条日志都不该留下：被拒的调用没有发生过。
    let injected = common::world_log()
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "brain/inject")
        .count();
    assert_eq!(injected, 1);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
