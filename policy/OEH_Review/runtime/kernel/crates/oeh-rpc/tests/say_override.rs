//! 适配器的 `say` 顶替内核的内置 `say`，断连之后内核把它接回去。
//!
//! 这是 `crates/oeh-admission/tests/override.rs` 那套注册表语义在**边界协议**上的样子：
//! 一次普通的 `plugin/register` 就是覆盖（客户端不必知道"覆盖"这个词），`robot/
//! capabilities` 上多出来的 `provider` 是"这句话由谁说出来"的唯一凭据，而断连——不是
//! 一次专门的清理代码——是内核把嗓子收回来的那一下。

mod common;

use serde_json::{json, Value};
use std::sync::Arc;

/// 一个只声明 `say` 的跨进程插件。心跳周期故意很长：这个测试里任何一次超时都是
/// **错误的**原因（能力面缩回去、`capability/restored`，心跳超时一样全都满足）。
fn manifest() -> Value {
    json!({
        "manifest": {"name": "py-voice", "inject": ["clock"], "provide": [], "streams": []},
        "capabilities": [{
            "name": "say",
            "class": "speech",
            "risk_level": "none",
            "params_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                              "required": ["text"]},
            "lease_ttl_s": 1.0
        }],
        "heartbeat_ms": 10_000
    })
}

/// 能力面上叫这个名字的那一项。
fn cap(caps: &Value, name: &str) -> Value {
    caps["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("`{name}` is not on the capability face: {caps}"))
        .clone()
}

/// 世界日志里有没有这么一条事件。
fn logged(log: &oeh_kernel::WorldLog, kind: &str, payload: Value) -> bool {
    log.since(0)
        .unwrap()
        .iter()
        .any(|e| e.kind.as_str() == kind && e.payload == payload)
}

/// 叫醒脑，拿到这一轮的 turn 号。
async fn turn(
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

#[tokio::test]
async fn an_adapter_takes_over_say_and_the_kernel_takes_it_back_when_the_adapter_goes() {
    let (adapter, kh, run) = common::boot().await;
    let log = common::world_log();

    // 适配器那一侧：说过的话记下来，结果用自己的形状回（内核的内置 `say` 回的是
    // `{"text": ...}`，两者不一样正是"谁执行的"的判据）。
    let spoken = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
    let said = spoken.clone();
    let a2 = adapter.clone();
    adapter.on_request(
        "capability/execute",
        Arc::new(move |p: Value| {
            said.lock()
                .push(p["params"]["text"].as_str().unwrap_or_default().to_string());
            let c = a2.clone();
            Box::pin(async move {
                let exec_id = p["exec_id"].as_str().unwrap().to_string();
                tokio::spawn(async move {
                    c.notify(
                        "capability/result",
                        json!({"exec_id": exec_id, "ok": true, "result": {"spoken": true}}),
                    )
                    .unwrap();
                });
                Ok(json!({"accepted": true}))
            })
        }),
    );

    // 覆盖就是一次普通的注册：客户端什么特别的都没做。
    let r = adapter.call("plugin/register", manifest()).await.unwrap();
    assert_eq!(r["state"], "active", "{r}");

    let brain = common::connect().await;
    let caps = brain.call("robot/capabilities", json!({})).await.unwrap();
    let say = cap(&caps, "say");
    assert_eq!(say["provider"], "py-voice", "{say}");
    assert_eq!(say["overridable"], false, "{say}");
    // 生效的是适配器那一份 spec，不只是执行器。
    assert_eq!(say["lease_ttl_s"], 1.0, "{say}");
    // 内核自己留着的名字一点没动。
    let stop = cap(&caps, "stop");
    assert_eq!(stop["provider"], "kernel", "{stop}");
    assert_eq!(stop["overridable"], false, "{stop}");
    assert!(
        logged(
            &log,
            "capability/overridden",
            json!({"name": "say", "by": "py-voice"})
        ),
        "no capability/overridden in the world log"
    );

    // 脑提 `say` → 话到了机器人那一侧。
    let mut n = brain.notifications();
    brain
        .call(
            "brain/attach",
            json!({"brain_id": "fast",
                   "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"},
                                   "budget": {"max_consecutive_wakes": 100, "min_interval_ms": 0}}}),
        )
        .await
        .unwrap();
    let t = turn(&brain, &mut n).await;
    let out = brain
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": t, "capability": "say",
                   "params": {"text": "你好"}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "executed", "{out}");
    assert_eq!(out["outcome"]["result"]["spoken"], true, "{out}");
    assert_eq!(spoken.lock().as_slice(), ["你好"]);
    brain
        .call(
            "turn/done",
            json!({"brain_id": "fast", "turn": t, "reason": "ok"}),
        )
        .await
        .unwrap();

    // 机器人走了：能力面收回内置的那一份，而且日志里说得清楚是从谁那里收回来的。
    adapter.close().await;
    common::await_condition("capability/restored", || {
        logged(
            &log,
            "capability/restored",
            json!({"name": "say", "from": "py-voice"}),
        )
    })
    .await;
    let say = cap(
        &brain.call("robot/capabilities", json!({})).await.unwrap(),
        "say",
    );
    assert_eq!(say["provider"], "kernel", "{say}");
    assert_eq!(say["overridable"], true, "{say}");
    assert_eq!(say["lease_ttl_s"], 5.0, "{say}");

    // 而 `say` 仍然能提：内置的兜底回来了，脑的工具表没有因为少了一台机器人而变形。
    let t = turn(&brain, &mut n).await;
    let out = brain
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": t, "capability": "say",
                   "params": {"text": "还在"}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "executed", "{out}");
    // 内核的 `Echo`：把参数原样回给 `act/result`，而适配器一个字都没再收到。
    assert_eq!(out["outcome"]["result"]["text"], "还在", "{out}");
    assert_eq!(spoken.lock().len(), 1);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
