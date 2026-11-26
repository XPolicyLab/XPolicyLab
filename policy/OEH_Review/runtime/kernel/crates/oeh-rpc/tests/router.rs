//! `KernelRouter` 的端到端：一个真内核（`common::boot()`）+ 一条真 UDS 连接，每个方法
//! 都从客户端那一侧验。
//!
//! 通知一律走 `common::next_notification`（带超时）而不是裸 `recv()`：一条没来的通知在
//! 裸 `recv()` 下表现为整个测试挂死，报出来的是超时，不是"哪条通知没来"。

mod common;

use serde_json::json;

#[tokio::test]
async fn robot_methods_report_capabilities_state_and_health() {
    let (c, kh, run) = common::boot().await;
    let caps = c.call("robot/capabilities", json!({})).await.unwrap();
    let caps = caps["capabilities"].as_array().unwrap();
    assert!(caps.iter().any(|x| x["name"] == "walk"));
    // 线格式是 serde 的小写名，不是 Rust 的 Debug 名。
    let walk = caps.iter().find(|x| x["name"] == "walk").unwrap();
    assert_eq!(walk["class"], "motion");
    assert_eq!(walk["risk_level"], "medium");
    // 内置能力（admission 插件注册的那四个）也在表里。
    assert!(caps.iter().any(|x| x["name"] == "say"));

    let st = c.call("robot/state", json!({})).await.unwrap();
    // FsmState 是 SCREAMING_SNAKE_CASE。
    assert!(["BOOT", "STANDING"].contains(&st["fsm"].as_str().unwrap()));
    assert!(st["t_kernel"].as_u64().unwrap() > 0);
    assert!(st["state"]["fsm"].is_string(), "{st}");

    let h = c.call("robot/health", json!({})).await.unwrap();
    assert_eq!(h["connections"], 1);
    assert!(h["fibers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f["plugin"] == "rpc-server" && f["state"] == "active"));
    let lowstate = h["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["stream_id"] == "lowstate")
        .unwrap_or_else(|| panic!("no lowstate stream: {h}"))
        .clone();
    assert!(lowstate["frames"].as_u64().unwrap() > 0, "{lowstate}");
    assert!(lowstate["age_ms"].is_u64(), "{lowstate}");
    assert_eq!(h["log_sink_errors"], 0);

    // stream/list 是同一批流的声明（没有帧计数）。
    let streams = c.call("stream/list", json!({})).await.unwrap();
    assert!(streams["streams"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["stream_id"] == "lowstate" && s["tier"] == "small"));

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `stream/subscribe`：一条**本地**产生的流（`mock-stream` 的 `lowstate`）被接到 iceoryx2
/// 共享内存通道上，另一个 Node（这里就是测试自己起的那个）能从上面读到帧。
///
/// 两件事一起验，因为它们是同一个承诺的两半：result 里的那个名字是**真**通道名（不是一个
/// 算出来好看的字符串），而通道上真的有帧在流。不存在的流给 `NOT_ATTACHED`——给个通道名
/// 反而更糟，客户端会拿着它去等一条永远不会响的通道。
///
/// run 标签由测试脚手架显式给（`common::run_tag()`，每个内核一个）：iceoryx2 的服务名是
/// 机器全局的，两个测试内核不能共用一条通道。
#[tokio::test]
async fn stream_subscribe_hands_out_a_channel_another_node_can_read() {
    let (c, kh, run) = common::boot().await;

    let e = c
        .call("stream/subscribe", json!({"stream_id": "nope"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED);

    let ch = c
        .call("stream/subscribe", json!({"stream_id": "lowstate"}))
        .await
        .unwrap();
    assert_eq!(ch["stream_id"], "lowstate");
    assert_eq!(
        ch["channel"],
        format!("oeh/{}/lowstate", common::run_tag()),
        "{ch}"
    );
    assert_eq!(ch["schema_id"], "oeh.LowState@json", "{ch}");
    // 第二次订阅同一条流复用同一条 egress（懒起、只起一次），回的是同一个名字。
    let again = c
        .call("stream/subscribe", json!({"stream_id": "lowstate"}))
        .await
        .unwrap();
    assert_eq!(again, ch);

    let channel = ch["channel"].as_str().unwrap().to_string();
    // 读在 `spawn_blocking` 里：iceoryx2 的 port 不是 `Send`，Node 连同订阅者都得待在
    // 这一个线程上。
    let frames = tokio::task::spawn_blocking(move || {
        use iceoryx2::prelude::*;
        let node = NodeBuilder::new().create::<ipc::Service>().unwrap();
        let service = node
            .service_builder(&ServiceName::new(&channel).unwrap())
            .publish_subscribe::<[u8]>()
            .subscriber_max_buffer_size(oeh_rpc::frames::SUB_BUFFER)
            .history_size(oeh_rpc::frames::HISTORY)
            .max_publishers(oeh_rpc::frames::MAX_PUBLISHERS)
            .max_subscribers(oeh_rpc::frames::MAX_SUBSCRIBERS)
            .open_or_create()
            .unwrap();
        let sub = service.subscriber_builder().create().unwrap();
        let mut got: Vec<Vec<u8>> = Vec::new();
        let t0 = std::time::Instant::now();
        // 500 Hz 的流，三帧是 6 ms 的事；3 s 的上限只是为了让一次真正的失败报成
        // "一帧都没读到"，而不是挂死。
        while t0.elapsed() < std::time::Duration::from_secs(3) && got.len() < 3 {
            while let Ok(Some(s)) = sub.receive() {
                got.push(s.payload().to_vec());
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        got
    })
    .await
    .unwrap();
    assert!(frames.len() >= 3, "{} frames off the channel", frames.len());
    // 通道上的字节就是 `FrameStore` 里那一份（`LowState::to_bytes()`），没有再包一层。
    let first = String::from_utf8(frames[0].clone()).unwrap();
    assert!(first.contains("gravity_proj_z"), "{first}");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn attach_brain_receive_turn_propose_and_done() {
    let (c, kh, run) = common::boot().await;
    let mut n = c.notifications();
    let r = c
        .call(
            "brain/attach",
            json!({"brain_id": "fast", "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"}}}),
        )
        .await
        .unwrap();
    assert!(r["cursor"].is_u64());

    c.call(
        "event/emit",
        json!({"kind": "obs/person", "payload": {"distance_m": 1.0}}),
    )
    .await
    .unwrap();

    let turn = common::next_notification(&mut n, "brain/turn").await;
    assert_eq!(turn["brain_id"], "fast");
    assert_eq!(turn["observation"]["wake_reason"]["kind"], "obs/person");
    let t = turn["turn"].as_u64().unwrap();

    let out = c
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": t, "capability": "walk",
                   "params": {"vx": 0.1, "duration_s": 0.1}}),
        )
        .await
        .unwrap();
    assert!(
        out["proposal_id"].as_str().unwrap().ends_with("-p1"),
        "{out}"
    );
    assert_eq!(out["outcome"]["outcome"], "executed", "{out}");

    let say = c
        .call(
            "turn/say",
            json!({"brain_id": "fast", "turn": t, "text": "hi"}),
        )
        .await
        .unwrap();
    assert!(
        say["proposal_id"].as_str().unwrap().ends_with("-p2"),
        "{say}"
    );
    assert_eq!(say["outcome"]["outcome"], "executed", "{say}");

    c.call(
        "turn/done",
        json!({"brain_id": "fast", "turn": t, "reason": "ok"}),
    )
    .await
    .unwrap();

    let cur = c
        .call("brain/cursor", json!({"brain_id": "fast"}))
        .await
        .unwrap();
    assert_eq!(cur["status"], "idle");
    assert!(cur["cursor"].as_u64().unwrap() >= r["cursor"].as_u64().unwrap());

    // 句柄是被**取走**的：同一个 turn 关不了第二次。
    let e = c
        .call(
            "turn/done",
            json!({"brain_id": "fast", "turn": t, "reason": "again"}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 定向 steer / interrupt / 换 policy 三件事都真的落到了内核的 `Brains` 上。
///
/// 相比 brief 里的版本多了两步"把脑放进一个 turn 里"和一次 `turn/done`，因为内核的语义
/// 是这样的（`Brains` 的 `(action, status)` 矩阵，规格 §4.5）：
///
/// - **Idle** 的脑收到定向 steer 等同于 wake——它走 `Plan::StartTurn` → `sink.on_turn`，
///   发出来的是 `brain/turn`，**不是** `brain/steer`。`brain/steer` 只在 steer 投进一个
///   **活着的** turn 时才有（`sink.steer`）；
/// - 同理，`obs/gesture` 那一次唤醒要求脑此刻是 Idle：Running 的脑遇到 Wake 只会把它
///   latch 住，等当前 turn 结束才开新 turn。中间那次 `turn/done` 就是为此。
///
/// 唤醒预算也写死成"不限"：定向唤醒不受预算约束但**计入**预算（见 `Brains::steer` 的
/// 文档），默认的 `min_interval_ms: 500` 会让紧随其后的 `obs/gesture` 被抑制，测试就成了
/// 一个和真实时钟赛跑的东西。
#[tokio::test]
async fn steer_interrupt_and_policy_update_are_routed() {
    let (c, kh, run) = common::boot().await;
    let mut n = c.notifications();
    let budget = json!({"max_consecutive_wakes": 100, "min_interval_ms": 0});
    c.call(
        "brain/attach",
        json!({"brain_id": "fast",
               "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"},
                               "budget": budget}}),
    )
    .await
    .unwrap();

    // 先把脑叫进一个 turn，steer 才有地方投。
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    common::next_notification(&mut n, "brain/turn").await;

    c.call("turn/steer", json!({"brain_id": "fast", "text": "look"}))
        .await
        .unwrap();
    let m = common::next_notification(&mut n, "brain/steer").await;
    assert_eq!(m["brain_id"], "fast");
    assert_eq!(m["observation"]["wake_reason"]["kind"], "brain/steer");

    // 打断当前 turn：一条 brain/interrupt，紧跟着重新唤醒出来的一个新 turn。
    c.call("turn/interrupt", json!({"brain_id": "fast"}))
        .await
        .unwrap();
    let m = common::next_notification(&mut n, "brain/interrupt").await;
    assert_eq!(m["brain_id"], "fast");
    let re_woken = common::next_notification(&mut n, "brain/turn").await;
    assert_eq!(
        re_woken["observation"]["wake_reason"]["kind"],
        "brain/interrupt"
    );

    // 关掉它，脑回到 Idle，下一条唤醒才会真的开 turn 而不是被 latch。
    c.call(
        "turn/done",
        json!({"brain_id": "fast", "turn": re_woken["turn"], "reason": "ok"}),
    )
    .await
    .unwrap();

    c.call(
        "brain/wake_policy/update",
        json!({"brain_id": "fast",
               "wake_policy": {"default": "ignore", "rules": {"obs/gesture": "wake"},
                               "budget": budget}}),
    )
    .await
    .unwrap();
    c.call(
        "event/emit",
        json!({"kind": "obs/gesture", "payload": {"name": "wave"}}),
    )
    .await
    .unwrap();
    let m = common::next_notification(&mut n, "brain/turn").await;
    assert_eq!(m["observation"]["wake_reason"]["kind"], "obs/gesture");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `brain/detach` 只拆得掉自己那一份，别人的脑一律 `NOT_ATTACHED`。
#[tokio::test]
async fn detach_only_reaches_this_connections_brains() {
    let (c, kh, run) = common::boot().await;
    let c2 = common::connect().await;
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
    )
    .await
    .unwrap();

    // 另一条连接看得见它的游标（控制台要用），但拆不掉它。
    let cur = c2
        .call("brain/cursor", json!({"brain_id": "fast"}))
        .await
        .unwrap();
    assert_eq!(cur["status"], "idle");
    let e = c2
        .call("brain/detach", json!({"brain_id": "fast"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED);

    // 同一个 brain_id 也不能被第二条连接抢走。
    let e = c2
        .call(
            "brain/attach",
            json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED);

    // 名字不合法（会被拼进会话日志的文件名）是参数错误，不是策略拒绝。
    let e = c2
        .call("brain/attach", json!({"brain_id": "../escape"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS);

    // 自己拆自己：拆完之后别人就能接手了。
    c.call("brain/detach", json!({"brain_id": "fast"}))
        .await
        .unwrap();
    c2.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
    )
    .await
    .unwrap();

    let e = c
        .call("brain/cursor", json!({"brain_id": "nobody"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED);
    let e = c
        .call("turn/steer", json!({"brain_id": "nobody", "text": "x"}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn event_emit_refuses_reserved_kinds_and_stamps_source() {
    let (c, kh, run) = common::boot().await;
    for k in ["act/proposed", "safety/fsm", "safety/tamper_attempt"] {
        let e = c
            .call("event/emit", json!({"kind": k, "payload": {}}))
            .await
            .unwrap_err();
        assert_eq!(e.code, oeh_proto::rpc::code::REFUSED, "{k}");
    }
    let e = c
        .call("event/emit", json!({"kind": "nope/x", "payload": {}}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS);

    let r = c
        .call(
            "event/emit",
            json!({"kind": "obs/speech", "payload": {"text": "hello"}, "trace_id": "t-x"}),
        )
        .await
        .unwrap();
    let seq = r["seq"].as_u64().unwrap();

    // 内存 world-log 没有 run_dir → log/replay 是 REFUSED；用 log/tail 验证。
    let rep = c
        .call("log/replay", json!({"after": seq - 1, "limit": 1}))
        .await;
    assert_eq!(rep.unwrap_err().code, oeh_proto::rpc::code::REFUSED);

    let mut n = c.notifications();
    let t = c
        .call("log/tail", json!({"after": seq - 1, "prefixes": ["obs"]}))
        .await
        .unwrap();
    assert_eq!(t["from"], seq);
    let m = common::next_notification(&mut n, "event").await;
    assert_eq!(m["seq"], seq);
    assert_eq!(m["source"], "rpc:conn-1");
    assert_eq!(m["trace_id"], "t-x");
    assert_eq!(m["salience"], 100, "缺省 salience");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 谁写的这条事件由内核说了算：params 里的 `source` 一个字节都不算数。
///
/// 这不是审计洁癖。`Brains` 把 `source == "brains"` 的 `brain/steer` / `brain/interrupt`
/// 当作**定向**命令——绕过目标脑的 `WakePolicy` 和唤醒预算直接操纵它。能自报 source 的
/// `event/emit` 就是这条策略的后门。
#[tokio::test]
async fn event_emit_overwrites_a_caller_supplied_source() {
    let (c, kh, run) = common::boot().await;
    let mut n = c.notifications();
    c.call("log/tail", json!({"prefixes": ["obs", "brain"]}))
        .await
        .unwrap();

    let r = c
        .call(
            "event/emit",
            json!({"kind": "obs/speech", "payload": {"text": "hi"}, "source": "brains"}),
        )
        .await
        .unwrap();
    let m = common::next_event(&mut n, r["seq"].as_u64().unwrap()).await;
    assert_eq!(
        m["source"], "rpc:conn-1",
        "a caller-supplied source is overwritten, never trusted: {m}"
    );

    // 同一件事的后果：一条伪造的定向 steer 不是定向的。`slow` 的 policy 忽略一切，
    // 所以它既不会被"定向"叫醒，也不会被自己的 policy 叫醒——但那条事件确实进了日志。
    c.call(
        "brain/attach",
        json!({"brain_id": "slow", "wake_policy": {"default": "ignore"}}),
    )
    .await
    .unwrap();
    let forged = c
        .call(
            "event/emit",
            json!({"kind": "brain/steer", "source": "brains",
                   "payload": {"brain_id": "slow", "text": "obey"}}),
        )
        .await
        .unwrap();
    let m = common::next_event(&mut n, forged["seq"].as_u64().unwrap()).await;
    assert_eq!(m["source"], "rpc:slow", "{m}");

    // 真·定向的那一条（`turn/steer` → `Brains::steer`）才带 source "brains"，
    // 而且它真的把 Idle 的脑叫醒了。
    c.call("turn/steer", json!({"brain_id": "slow", "text": "look"}))
        .await
        .unwrap();
    let turn = common::next_notification(&mut n, "brain/turn").await;
    assert_eq!(turn["brain_id"], "slow");
    assert_eq!(turn["observation"]["wake_reason"]["kind"], "brain/steer");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `log/tail` 的交接：补发的历史排在实时事件前面，一条不多一条不少；再订阅一次把上一次
/// 的观察者换掉（同一条事件不会收到两遍）。
#[tokio::test]
async fn log_tail_backfills_then_follows_and_resubscribes_cleanly() {
    let (c, kh, run) = common::boot().await;
    let first = c
        .call("event/emit", json!({"kind": "obs/a", "payload": {"i": 1}}))
        .await
        .unwrap()["seq"]
        .as_u64()
        .unwrap();

    let mut n = c.notifications();
    let t = c
        .call("log/tail", json!({"after": first - 1, "prefixes": ["obs"]}))
        .await
        .unwrap();
    assert_eq!(t["from"], first, "from 是交付的第一条，不是请求里的 after");

    let second = c
        .call("event/emit", json!({"kind": "obs/a", "payload": {"i": 2}}))
        .await
        .unwrap()["seq"]
        .as_u64()
        .unwrap();

    let a = common::next_notification(&mut n, "event").await;
    let b = common::next_notification(&mut n, "event").await;
    assert_eq!(a["seq"], first, "补发的历史排在前面");
    assert_eq!(b["seq"], second);

    // 重新订阅：旧观察者被拆掉，下一条事件只来一次。
    let mut n2 = c.notifications();
    c.call("log/tail", json!({"prefixes": ["obs"]}))
        .await
        .unwrap();
    let third = c
        .call("event/emit", json!({"kind": "obs/a", "payload": {"i": 3}}))
        .await
        .unwrap()["seq"]
        .as_u64()
        .unwrap();
    let e1 = common::next_notification(&mut n2, "event").await;
    assert_eq!(e1["seq"], third);
    let fourth = c
        .call("event/emit", json!({"kind": "obs/a", "payload": {"i": 4}}))
        .await
        .unwrap()["seq"]
        .as_u64()
        .unwrap();
    let e2 = common::next_notification(&mut n2, "event").await;
    assert_eq!(
        e2["seq"], fourth,
        "第三条只来了一次：旧订阅已经被拆掉了，不然这里会是它的重复"
    );

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `plugin/state` 广播给**所有**连接，`robot/health` 的 fibers 就是它攒出来的快照。
#[tokio::test]
async fn plugin_state_is_broadcast_to_every_connection() {
    let (c, kh, run) = common::boot().await;
    let c2 = common::connect().await;
    let mut n1 = c.notifications();
    let mut n2 = c2.notifications();

    // 撤掉一个 Fiber（`frame-counter` 不在 profile 里，用一个新加的行来演）：加进去、
    // 再撤掉，两条连接都应该看到状态变化。
    let added = c.call("robot/health", json!({})).await.unwrap()["fibers"]
        .as_array()
        .unwrap()
        .len();
    assert!(added > 0);

    // 内核在每次 settle 之后广播变化；这里靠 `Op::DisposeFiber` 制造一次。
    let fiber_id = c.call("robot/health", json!({})).await.unwrap()["fibers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["plugin"] == "mock-stream")
        .unwrap()["fiber_id"]
        .as_u64()
        .unwrap() as u32;
    kh.dispose_fiber(oeh_kernel::FiberId(fiber_id))
        .unwrap()
        .await
        .unwrap();

    for n in [&mut n1, &mut n2] {
        let m = common::next_notification(n, "plugin/state").await;
        assert_eq!(m["fiber_id"], fiber_id);
        assert_eq!(m["state"], "disposed");
    }
    let h = c.call("robot/health", json!({})).await.unwrap();
    let f = h["fibers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["fiber_id"] == fiber_id)
        .unwrap();
    assert_eq!(f["state"], "disposed", "{h}");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn disconnect_detaches_brains() {
    let (c, kh, run) = common::boot().await;
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
    )
    .await
    .unwrap();
    c.close().await;

    let c2 = common::connect().await;
    // 重连后同名 attach 必须成功：旧连接的脑已经 detach。断连清理是异步的（服务端要先
    // 看到 EOF），所以给它一点时间，但不多。
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if c2
                .call(
                    "brain/attach",
                    json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
                )
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 没接的方法是 `METHOD_NOT_FOUND`，不是静默的空回复。
#[tokio::test]
async fn unknown_and_not_yet_implemented_methods_are_method_not_found() {
    let (c, kh, run) = common::boot().await;
    for m in ["nope/nope", "stream/nope"] {
        let e = c.call(m, json!({})).await.unwrap_err();
        assert_eq!(e.code, oeh_proto::rpc::code::METHOD_NOT_FOUND, "{m}");
    }
    // 对照组：这两个都接上了，所以一个**空** params 得到的是参数错误，而不是"没这个
    // 方法"——两者对客户端是完全不同的两种"不行"。边界协议的方法表到这里就齐了
    // （`stream/subscribe` 是最后一个接上的）。
    for m in ["plugin/register", "stream/subscribe"] {
        let e = c.call(m, json!({})).await.unwrap_err();
        assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS, "{m}: {e:?}");
    }
    // log/flush 对内存日志是空操作，不是错误。
    c.call("log/flush", json!({})).await.unwrap();
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `log/replay` 的实盘路径：给 world-log 一个目录，`log/flush` 之后段文件里就有东西了。
/// 相对的 `dir` 按"run 目录的兄弟"解释，所以一个 run 标签（`run-<ms>`）就能指回来。
#[tokio::test]
async fn log_replay_reads_segments_from_disk() {
    let logs = tempfile::tempdir().unwrap();
    let dir = logs.path().to_path_buf();
    let (c, kh, run) = common::boot_with(move |rows| {
        for r in rows.iter_mut() {
            if r.id == "world-log" {
                *r = oeh_kernel::profile::EntryRow::new("world-log")
                    .with_config(json!({"dir": dir, "segment_mb": 1}));
            }
        }
    })
    .await;

    let seq = c
        .call(
            "event/emit",
            json!({"kind": "obs/speech", "payload": {"text": "on disk"}}),
        )
        .await
        .unwrap()["seq"]
        .as_u64()
        .unwrap();
    // 段文件是带缓冲的：不 flush 的话刚写的这条还在 BufWriter 里。
    c.call("log/flush", json!({})).await.unwrap();

    let r = c
        .call("log/replay", json!({"after": seq - 1, "limit": 1}))
        .await
        .unwrap();
    let events = r["events"].as_array().unwrap();
    assert_eq!(events.len(), 1, "{r}");
    assert_eq!(events[0]["seq"], seq);
    assert_eq!(events[0]["kind"], "obs/speech");
    assert_eq!(events[0]["source"], "rpc:conn-1");
    assert_eq!(r["done"], true, "只有这一条比 after 新: {r}");

    // 从头翻一页：limit 之外还有更多 → done=false。
    let head = c
        .call("log/replay", json!({"after": 0, "limit": 2}))
        .await
        .unwrap();
    assert_eq!(head["events"].as_array().unwrap().len(), 2);
    assert_eq!(head["events"][0]["seq"], 1);
    assert_eq!(head["done"], false);

    // 相对 `dir` = run 目录的名字（`RouterConfig::run_tag`），解释成它的兄弟目录。
    let run_tag = std::fs::read_dir(logs.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().starts_with("run-"))
        .unwrap()
        .file_name()
        .to_string_lossy()
        .into_owned();
    let same = c
        .call(
            "log/replay",
            json!({"dir": run_tag, "after": seq - 1, "limit": 1}),
        )
        .await
        .unwrap();
    assert_eq!(same["events"][0]["seq"], seq);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Fix round 1, H1: two `log/tail` calls in flight on one connection must leave exactly one
/// subscription behind.
///
/// `dispatch_line` gives every Request its own `tokio::spawn`, so on a multi-thread runtime
/// two `log/tail` requests on one connection really do run in parallel. Without a per-
/// connection critical section both read "no previous tail", both `observe()` the world log,
/// and the second store overwrites the first `Disposer` — which is `FnOnce`, so dropping it
/// does not run it. The result is permanent: every matching event delivered twice for the
/// life of the connection, and an observer that stays in `WorldLog`'s list forever (cloned on
/// every single append, on a 500 Hz control loop) with nothing left that could remove it.
///
/// Both halves are asserted, because either one alone can be satisfied by accident: the
/// observer count (the leak) *and* the delivery count (the duplication). The 400-event
/// backfill is there to widen the window — with an empty log both handlers finish too fast
/// to interleave.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_tail_is_serialised_per_connection() {
    let (c, kh, run) = common::boot().await;
    let log = common::world_log();
    for i in 0..400 {
        log.append(
            oeh_proto::EventDraft::new(
                oeh_proto::EventKind::new("obs/filler").unwrap(),
                "test",
                json!({ "i": i }),
            )
            .with_salience(1),
        );
    }
    // Also proves the thread-local harness survives a multi-thread flavor (see common's
    // doc comment): this `connect()` runs after several await points.
    let other = common::connect().await;

    let before = log.observer_count();
    let (a, b) = tokio::join!(
        c.call("log/tail", json!({"after": 0, "prefixes": ["obs"]})),
        c.call("log/tail", json!({"after": 0, "prefixes": ["obs"]})),
    );
    a.unwrap();
    b.unwrap();
    assert_eq!(
        log.observer_count(),
        before + 1,
        "one connection, one tail observer — the loser's must have been disposed, not dropped"
    );

    let mut n = c.notifications();
    let seq = c
        .call("event/emit", json!({"kind": "obs/after", "payload": {}}))
        .await
        .unwrap()["seq"]
        .as_u64()
        .unwrap();
    common::next_event(&mut n, seq).await;
    // Anything that arrives in the next window is a duplicate: the notification above was
    // already delivered, and nothing else emits `obs/*` in this profile.
    let dupes = tokio::time::timeout(std::time::Duration::from_millis(300), async {
        loop {
            let ev = common::next_notification(&mut n, "event").await;
            if ev["seq"] == json!(seq) {
                return 1;
            }
        }
    })
    .await
    .unwrap_or(0);
    assert_eq!(
        dupes, 0,
        "the event was delivered twice: two live subscriptions"
    );

    other.close().await;
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Fix round 1, M3 + L4: the per-connection turn table holds the *current* turn, not every
/// turn this brain ever started.
///
/// Only `turn/done` removes a handle, and there are two other ways a turn ends — an
/// interrupt, and the kernel's turn-timeout sweep, which does not notify the sink at all.
/// A long-lived cross-process brain interrupted a few thousand times over a shift would
/// otherwise accumulate one dead `TurnHandle` per interrupt with nothing able to reclaim
/// them until the connection closes.
#[tokio::test]
async fn interrupted_turns_do_not_accumulate_handles() {
    let (c, kh, run) = common::boot().await;
    let mut n = c.notifications();
    let budget = json!({"max_consecutive_wakes": 100, "min_interval_ms": 0});
    c.call(
        "brain/attach",
        json!({"brain_id": "fast",
               "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"},
                               "budget": budget}}),
    )
    .await
    .unwrap();
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    common::next_notification(&mut n, "brain/turn").await;

    // Two interrupts, no `turn/done` in between: each one closes the running turn and
    // re-wakes into a fresh one.
    for _ in 0..2 {
        c.call("turn/interrupt", json!({"brain_id": "fast"}))
            .await
            .unwrap();
        common::next_notification(&mut n, "brain/interrupt").await;
        common::next_notification(&mut n, "brain/turn").await;
    }

    let conn = common::conns()
        .iter()
        .into_iter()
        .find(|x| x.name() == "fast")
        .expect("the attached connection is named after its brain");
    assert_eq!(
        conn.turn_count("fast"),
        1,
        "three turns started, two of them long since closed by interrupt"
    );

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Fix round 1, L1: a `log/tail` backfill is capped, and `from` says where the coverage
/// really starts.
///
/// The backfill is a synchronous run of `conn.notify` (`try_send`); an overflowing outbound
/// queue does not drop a line, it kills the connection (`mark_closed` + a
/// `health/rpc_backpressure` record). Retention and the queue are both 4096, so an uncapped
/// `log/tail {"after": 0}` on a warm kernel fills the queue exactly — and any notification
/// already sitting in it tips it over. "Console reconnects and asks for the backlog" would
/// be the normal way to hit that.
#[tokio::test]
async fn log_tail_caps_the_backfill_and_reports_where_coverage_starts() {
    let (c, kh, run) = common::boot().await;
    let log = common::world_log();
    for i in 0..1_200 {
        log.append(oeh_proto::EventDraft::new(
            oeh_proto::EventKind::new("obs/filler").unwrap(),
            "test",
            json!({ "i": i }),
        ));
    }
    let hi_before = log.next_seq() - 1;
    let t = c.call("log/tail", json!({"after": 0})).await.unwrap();
    let hi_after = log.next_seq() - 1;
    let from = t["from"].as_u64().unwrap();

    assert!(from > 1, "the cap did not apply: from={from}");
    // `from` is the seq of the 1024th-newest retained event, for whatever the high-water
    // mark was at the instant the handler read the log — bracketed by the two reads above.
    assert!(
        from >= hi_before.saturating_sub(1023) && from <= hi_after.saturating_sub(1023),
        "from={from} outside [{}, {}]",
        hi_before.saturating_sub(1023),
        hi_after.saturating_sub(1023)
    );
    // The connection survived (an uncapped backfill would have been killed by backpressure).
    c.call("robot/state", json!({})).await.unwrap();

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Fix round 1, M1: `log/replay`'s `dir` may only name something inside this run's log root.
///
/// Before this, `dir` was joined onto the root without normalising and used verbatim when
/// absolute, so any client on the socket could read any `seg-*.jsonl` the kernel process can
/// open — and, from the `io::Error` in the message, use it as an existence/permission oracle
/// for *any* path. Three gates now: a run dir must exist at all; `dir` must be a relative
/// path of `Normal` components; and the canonicalised result must still be under the
/// canonicalised root (which is what stops a symlink planted inside the root).
#[tokio::test]
async fn log_replay_refuses_paths_outside_the_log_root() {
    let logs = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    // A directory that really is a readable world log, so the only thing that can refuse it
    // is the sandbox and not "there was nothing there".
    std::fs::write(
        outside.path().join("seg-000000000001.jsonl"),
        "{\"seq\":1,\"t_kernel\":1,\"kind\":\"obs/secret\",\"source\":\"t\",\"payload\":{\"password\":\"hunter2\"}}\n",
    )
    .unwrap();
    let dir = logs.path().to_path_buf();
    let (c, kh, run) = common::boot_with(move |rows| {
        for r in rows.iter_mut() {
            if r.id == "world-log" {
                *r = oeh_kernel::profile::EntryRow::new("world-log")
                    .with_config(json!({"dir": dir}));
            }
        }
    })
    .await;

    let escape = format!(
        "../{}",
        outside.path().file_name().unwrap().to_string_lossy()
    );
    for bad in [
        escape.as_str(),
        "../secret",
        outside.path().to_str().unwrap(),
        "/etc",
        "./run-1",
        // Fix round 2, R2: the empty path has no components, so "every component is
        // `Normal`" is vacuously true and `join("")` lands back on the root — an
        // undocumented, untested alias for "the newest run". Rejected explicitly.
        "",
    ] {
        let e = c
            .call("log/replay", json!({"dir": bad, "after": 0, "limit": 5}))
            .await
            .unwrap_err();
        assert_eq!(
            e.code,
            oeh_proto::rpc::code::INVALID_PARAMS,
            "`{bad}` must be rejected as a malformed `dir`, not resolved"
        );
    }

    // A symlink planted inside the log root passes the component check and is caught only by
    // the canonicalised containment check.
    std::os::unix::fs::symlink(outside.path(), logs.path().join("sneaky")).unwrap();
    let e = c
        .call(
            "log/replay",
            json!({"dir": "sneaky", "after": 0, "limit": 5}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED);
    assert_eq!(
        e.message, "replay dir not available",
        "the error text must not distinguish missing from forbidden: it is an oracle"
    );
    // Same fixed text for a name that simply is not there — that is the point.
    let e = c
        .call(
            "log/replay",
            json!({"dir": "run-does-not-exist", "after": 0}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED);
    assert_eq!(e.message, "replay dir not available");

    // Fix round 2, R5: a name that is inside the sandbox but cannot be read as a log
    // directory answers with the same fixed text — "exists but is not readable" and "is
    // not there" must not be distinguishable either.
    std::fs::write(logs.path().join("not-a-dir"), b"x").unwrap();
    let e = c
        .call("log/replay", json!({"dir": "not-a-dir", "after": 0}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED);
    assert_eq!(e.message, "replay dir not available");

    // A genuine sibling run tag still works.
    c.call("event/emit", json!({"kind": "obs/a", "payload": {}}))
        .await
        .unwrap();
    c.call("log/flush", json!({})).await.unwrap();
    let run_tag = std::fs::read_dir(logs.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().starts_with("run-"))
        .unwrap()
        .file_name()
        .to_string_lossy()
        .into_owned();
    let r = c
        .call(
            "log/replay",
            json!({"dir": run_tag, "after": 0, "limit": 5}),
        )
        .await
        .unwrap();
    assert!(!r["events"].as_array().unwrap().is_empty());

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Fix round 2, R1: a `brain/attach` that cannot store its `Disposer` must run it, and a
/// `brain/detach` must never claim to have detached a reservation.
///
/// `Brains::attach` opens a session-log file, which is slow enough for a concurrent
/// `brain/detach` — or a disconnect — to remove the reservation before it returns. A
/// `Disposer` is a boxed `FnOnce`: dropping it does not run it, so `Brains` would keep the
/// attachment with a sink pointing at a dying connection and the `brain_id` would be burnt
/// for the life of the process — every later `brain/attach` for that name, from any
/// connection, `AlreadyAttached` → `REFUSED`, with `robot/health` showing nothing wrong.
/// A client that pipelines attach + detach without waiting for the first reply (an ordinary
/// shutdown path for an SDK brain) is all it takes.
///
/// Two assertions per round, and they cover the two halves of the bug:
/// - a `brain/detach` that answered `Ok` must really have left the brain detached (the old
///   code answered `Ok` after removing a *reservation*, having disposed nothing);
/// - the id must still be attachable from a fresh connection (it is never stranded).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn attach_racing_detach_never_strands_the_brain_id() {
    let (c0, kh, run) = common::boot().await;
    for i in 0..60 {
        let c = common::connect().await;
        let (attached, detached) = tokio::join!(
            c.call(
                "brain/attach",
                json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}})
            ),
            c.call("brain/detach", json!({"brain_id": "fast"})),
        );
        if detached.is_ok() {
            let e = c0
                .call("brain/cursor", json!({"brain_id": "fast"}))
                .await
                .unwrap_err();
            assert_eq!(
                e.code,
                oeh_proto::rpc::code::NOT_ATTACHED,
                "round {i}: brain/detach answered Ok but the brain is still attached \
                 (attach={attached:?})"
            );
        }
        c.close().await;

        // The id must still be takeable. Retry: if the attach won the race the brain is
        // only released when the server sees this connection's EOF.
        let probe = common::connect().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if probe
                    .call(
                        "brain/attach",
                        json!({"brain_id": "fast", "wake_policy": {"default": "ignore"}}),
                    )
                    .await
                    .is_ok()
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("round {i}: `fast` is stranded — nobody can attach it again"));
        probe
            .call("brain/detach", json!({"brain_id": "fast"}))
            .await
            .unwrap();
        probe.close().await;
    }
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Plan 3 最终评审 C2：`log/replay` 的 `limit` 被**静默**夹在一页上。
///
/// 不夹的话，`{"after": 0, "limit": 100000000}`（线上六十个字节）会让内核把整个 run 的
/// 世界日志读进一个 `Vec<WorldEvent>`，再 `serde_json` 序列化成**一行**——既远超
/// `MAX_LINE`（对端永远读不到），又在出站队列里再拷一份；放大倍数约 1:10⁸。静默是对的：
/// `done: false` 本来就是"还有，接着翻页"，调用方照着翻就是了。
///
/// 5000 条事件直接 append（而不是 5000 次 `event/emit` 往返）：这里要测的是回放这一页的
/// 边界，不是 RPC 的吞吐。
#[tokio::test]
async fn log_replay_clamps_an_absurd_limit_to_one_page() {
    let logs = tempfile::tempdir().unwrap();
    let dir = logs.path().to_path_buf();
    let (c, kh, run) = common::boot_with(move |rows| {
        for r in rows.iter_mut() {
            if r.id == "world-log" {
                *r = oeh_kernel::profile::EntryRow::new("world-log")
                    .with_config(json!({"dir": dir, "segment_mb": 1}));
            }
        }
    })
    .await;

    let log = common::world_log();
    for i in 0..5000u32 {
        log.append(oeh_proto::EventDraft::new(
            oeh_proto::EventKind::new("obs/bulk").unwrap(),
            "test".to_string(),
            json!({"i": i}),
        ));
    }
    c.call("log/flush", json!({})).await.unwrap();

    let r = c
        .call("log/replay", json!({"after": 0, "limit": 100_000_000u64}))
        .await
        .unwrap();
    let events = r["events"].as_array().unwrap();
    assert_eq!(
        events.len(),
        4096,
        "limit must be clamped to max_replay_page, got {} events",
        events.len()
    );
    assert_eq!(
        r["done"], false,
        "被夹掉的那一段要靠 done:false 告诉调用方: {r}"
    );
    // 夹的是上界，不是每一次调用：小的 `limit` 原样生效。
    let one = c
        .call("log/replay", json!({"after": 0, "limit": 1}))
        .await
        .unwrap();
    assert_eq!(one["events"].as_array().unwrap().len(), 1, "{one}");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Plan 3 最终评审 I1：`log/tail` 的 `prefixes` 有上限。
///
/// 匹配跑在世界日志的观察者链上，也就是同步跑在 `WorldLog::append` 里——包括 500 Hz 的帧
/// ingest 路径上那些 append。一条带一百万个前缀的订阅（远在 `MAX_LINE` 之内）会让此后每
/// 一条事件先做一百万次比较：帧 ingest 卡住，看门狗报 `health/stale`，规则链拒掉一切。
#[tokio::test]
async fn log_tail_refuses_more_prefixes_than_the_append_path_can_afford() {
    let (c, kh, run) = common::boot().await;
    let prefixes = |n: usize| (0..n).map(|i| format!("p{i}")).collect::<Vec<_>>();
    let e = c
        .call("log/tail", json!({"prefixes": prefixes(17)}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS, "{e:?}");
    // 16 个照收——上限是给"一百万"设的，不是给操作者那两三个 `--prefix` 设的。
    c.call("log/tail", json!({"prefixes": prefixes(16)}))
        .await
        .unwrap();

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 最终评审再审 R1：上限护的是**留下来**的东西，而 payload 只是其中之一。
///
/// `WorldEvent` 同样留着 `kind`、`collapse_key` 和 `trace_id`，三者和 payload 一起进那个
/// 按条数（4096）保留的环、进每个 `log/tail` 观察者的通知、进每个 attach 着的脑的收件箱。
/// 第一版只量 payload，于是 I3 要堵的那个放大原样从 `kind` 里走了一遍：实测一条 100 KiB
/// 的 `collapse_key` 和一条 100 KiB 的 `kind` 都越过了 64 KiB 的上限，各自拿到了一个 seq。
#[tokio::test]
async fn event_emit_bounds_what_it_retains_not_just_the_payload() {
    let (c, kh, run) = common::boot().await;
    let huge = "x".repeat(100 * 1024);
    let cases = [
        (
            "collapse_key",
            json!({"kind": "obs/blob", "payload": {}, "collapse_key": huge}),
        ),
        (
            "kind",
            json!({"kind": format!("obs/{huge}"), "payload": {}}),
        ),
        (
            "trace_id",
            json!({"kind": "obs/blob", "payload": {}, "trace_id": huge}),
        ),
    ];
    for (what, params) in cases {
        let e = c.call("event/emit", params).await.unwrap_err();
        assert_eq!(
            e.code,
            oeh_proto::rpc::code::INVALID_PARAMS,
            "{what}: {e:?}"
        );
    }
    // 总量也管这三个：一个刚好在单字段上限之内的 kind 加上一个几乎占满配额的 payload，
    // 加起来越线 —— 这一条是"逐字段上限"单独挡不住的。
    let e = c
        .call(
            "event/emit",
            json!({"kind": format!("obs/{}", "k".repeat(200)),
                   "collapse_key": "c".repeat(200),
                   "payload": {"blob": "x".repeat(64 * 1024 - 300)}}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS, "{e:?}");

    // 正常尺寸的三件套照常写进去。
    let ok = c
        .call(
            "event/emit",
            json!({"kind": "obs/blob", "payload": {"blob": "x"},
                   "collapse_key": "blob:1", "trace_id": "t-1"}),
        )
        .await
        .unwrap();
    assert!(ok["seq"].as_u64().unwrap() > 0, "{ok}");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Plan 3 最终评审 I3：`event/emit` 的 payload 有字节上限。
///
/// 世界日志按**条数**保留（默认 4096 条），不按字节：4096 条各带一个 16 MiB 的 payload
/// 就是 64 GiB 常驻内存，外加每个 `log/tail` 观察者、每个 attach 着的脑各一份克隆。
/// `plugin/register` 早就有同一形状的护栏（`max_register_bytes`），`event/emit` 漏了。
#[tokio::test]
async fn event_emit_refuses_an_oversized_payload() {
    let (c, kh, run) = common::boot().await;
    let e = c
        .call(
            "event/emit",
            json!({"kind": "obs/blob", "payload": {"blob": "x".repeat(64 * 1024)}}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::INVALID_PARAMS, "{e:?}");
    // 一条正常大小的观测照常写进去。
    let ok = c
        .call(
            "event/emit",
            json!({"kind": "obs/blob", "payload": {"blob": "x".repeat(1024)}}),
        )
        .await
        .unwrap();
    assert!(ok["seq"].as_u64().unwrap() > 0, "{ok}");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
