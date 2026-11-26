//! 跨进程插件的端到端：一个客户端用 `plugin/register` 把自己变成内核树上的一个 Fiber，
//! 它声明的能力经 `capability/execute` 反向请求跑在客户端那一侧，它的流出现在
//! `stream/list` 里；断连或心跳超时会把这个 Fiber 连同它注册的一切一起撤掉；
//! `approval/attach` 把 `Ask` 判定送到挂上的那条连接。
//!
//! 时间在这里是承重结构（`heartbeat_ms: 100` → 300 ms 超时），所以每个"等一条通知"
//! 都套着显式超时：一条没来的通知要报成"它没来"，而不是整个测试挂死。

mod common;

use serde_json::{json, Value};
use std::sync::Arc;

/// 默认 manifest 的心跳周期是 10 s（→ 30 s 的失联期限），**故意**长到心跳监视在任何一个
/// 测试的生命期里都不可能开火：只有 `heartbeat_timeout_disposes_the_fiber` 用短周期，也只
/// 有它该受时间影响。
///
/// 这不只是防抖：`disconnect_disposes_the_fiber_and_the_capability_face_shrinks` 的每一条
/// 断言（`disposed`、能力没了、流没了）一次心跳超时也全都满足，所以只要那条路还可能触发，
/// 那个测试就可能因为**错误的原因**通过。30 s 的期限把它变回一个真正只测断连的测试。
fn manifest() -> Value {
    manifest_beating(10_000)
}

fn manifest_beating(heartbeat_ms: u64) -> Value {
    json!({
        "manifest": {"name": "py-arm", "inject": ["clock"], "provide": ["adapter.arm"],
                     "streams": [{"stream_id": "arm_state", "schema": "oeh.ArmState", "nominal_hz": 50.0, "tier": "small"}]},
        "capabilities": [{"name": "wave", "class": "motion", "risk_level": "low", "params_schema": {"type": "object", "properties": {"times": {"type": "number"}}}, "lease_ttl_s": 1.0}],
        "heartbeat_ms": heartbeat_ms
    })
}

#[tokio::test]
async fn remote_plugin_becomes_an_active_fiber_and_its_capability_executes_over_rpc() {
    let (c, kh, run) = common::boot().await;
    let executed = Arc::new(parking_lot::Mutex::new(Vec::<Value>::new()));
    let ex = executed.clone();
    let c2 = c.clone();
    c.on_request(
        "capability/execute",
        Arc::new(move |p: Value| {
            ex.lock().push(p.clone());
            let c3 = c2.clone();
            Box::pin(async move {
                let exec_id = p["exec_id"].as_str().unwrap().to_string();
                tokio::spawn(async move {
                    c3.notify("capability/heartbeat", json!({ "exec_id": exec_id }))
                        .unwrap();
                    c3.notify(
                        "capability/result",
                        json!({"exec_id": exec_id, "ok": true, "result": {"waved": 2}}),
                    )
                    .unwrap();
                });
                Ok(json!({"accepted": true}))
            })
        }),
    );
    let r = c.call("plugin/register", manifest()).await.unwrap();
    assert_eq!(r["state"], "active", "{r}");
    assert_eq!(r["streams"][0]["stream_id"], "arm_state");
    assert!(r["streams"][0]["channel"]
        .as_str()
        .unwrap()
        .ends_with("/arm_state"));
    assert_eq!(r["streams"][0]["schema_id"], "oeh.ArmState@json");
    let caps = c.call("robot/capabilities", json!({})).await.unwrap();
    assert!(caps["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x["name"] == "wave"));
    let streams = c.call("stream/list", json!({})).await.unwrap();
    assert!(streams["streams"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["stream_id"] == "arm_state"));

    // 订阅一条**远端在发**的流：拿到的就是它自己那条通道（同一个 iceoryx2 服务，多一个
    // 订阅者），内核不会为它再起一个发布者——那会和这条流的 ingress 组成一个自激回路。
    let ch = c
        .call("stream/subscribe", json!({"stream_id": "arm_state"}))
        .await
        .unwrap();
    assert_eq!(ch["channel"], r["streams"][0]["channel"], "{ch}");
    assert_eq!(ch["schema_id"], "oeh.ArmState@json", "{ch}");

    // 用一个脑提 wave
    let mut n = c.notifications();
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"}}}),
    )
    .await
    .unwrap();
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    let turn = common::next_notification(&mut n, "brain/turn").await["turn"]
        .as_u64()
        .unwrap();
    let out = c
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "wave", "params": {"times": 2}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "executed", "{out}");
    assert_eq!(out["outcome"]["result"]["waved"], 2);
    assert_eq!(executed.lock().len(), 1);
    assert_eq!(executed.lock()[0]["params"]["times"], 2);
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn disconnect_disposes_the_fiber_and_the_capability_face_shrinks() {
    let (c, kh, run) = common::boot().await;
    c.on_request(
        "capability/execute",
        Arc::new(|_p: Value| Box::pin(async { Ok(json!({"accepted": true})) })),
    );
    let r = c.call("plugin/register", manifest()).await.unwrap();
    let fid = r["fiber_id"].as_u64().unwrap();
    let watcher = common::connect().await;
    let mut n = watcher.notifications();
    c.close().await;
    let m = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let m = n.recv().await.unwrap();
            if m.method == "plugin/state"
                && m.params["fiber_id"] == fid
                && m.params["state"] == "disposed"
            {
                break m;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(m.params["plugin"], "py-arm");
    let caps = watcher.call("robot/capabilities", json!({})).await.unwrap();
    assert!(!caps["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x["name"] == "wave"));
    let streams = watcher.call("stream/list", json!({})).await.unwrap();
    assert!(!streams["streams"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["stream_id"] == "arm_state"));
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn heartbeat_timeout_disposes_the_fiber() {
    let (c, kh, run) = common::boot().await;
    // 一台"监控用"的控制台，挂在另一条连接上：它也得知道是谁掉了、为什么掉。
    let watcher = common::connect().await;
    let mut w = watcher.notifications();
    let r = c
        .call("plugin/register", manifest_beating(100))
        .await
        .unwrap(); // heartbeat_ms 100 → 300 ms 超时
    let fid = r["fiber_id"].as_u64().unwrap();
    let mut n = c.notifications();
    let m = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let m = n.recv().await.unwrap();
            if m.method == "plugin/state"
                && m.params["fiber_id"] == fid
                && m.params["state"] == "disposed"
            {
                break m;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(m.params["error"], "heartbeat timeout");
    // 失联的原因广播给所有连接，和 `plugin/state` 的其余部分一样：只告诉掉了的那一条
    // 连接，等于只告诉最可能收不到的那个。
    let seen = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let m = w.recv().await.unwrap();
            if m.method == "plugin/state"
                && m.params["fiber_id"] == fid
                && m.params["error"] == "heartbeat timeout"
            {
                break m;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(seen.params["plugin"], "py-arm");
    // 运维那一侧的记录：失联和正常退出在审计里必须分得开。
    assert!(common::world_log()
        .since(0)
        .unwrap_or_default()
        .iter()
        .any(|e| e.kind.as_str() == "health/plugin_heartbeat_timeout"
            && e.payload["plugin"] == "py-arm"));
    let e = c
        .call("plugin/heartbeat", json!({ "fiber_id": fid }))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED);
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 断连即释放（Plan 3 遗留 I4）在这里只验"脑看到的结局"这一半：完整的一条（终局事件、
/// `safety/adapter_lost`、motion 位子当场空出来）在 `tests/disconnect_release.rs`。
#[tokio::test]
async fn in_flight_remote_execution_fails_closed_when_the_plugin_disconnects() {
    let (c, kh, run) = common::boot().await;
    // 执行器接受但永不回结果
    c.on_request(
        "capability/execute",
        Arc::new(|_p: Value| Box::pin(async { Ok(json!({"accepted": true})) })),
    );
    c.call("plugin/register", manifest()).await.unwrap();
    let brain = common::connect().await;
    let mut n = brain.notifications();
    brain
        .call(
            "brain/attach",
            json!({"brain_id": "fast", "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"}}}),
        )
        .await
        .unwrap();
    brain
        .call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    let turn = common::next_notification(&mut n, "brain/turn").await["turn"]
        .as_u64()
        .unwrap();
    let b2 = brain.clone();
    let propose = tokio::spawn(async move {
        b2.call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "wave", "params": {"times": 1}}),
        )
        .await
        .unwrap()
    });
    // 等到内核那一侧确实把 `capability/execute` 发出来了（提案在飞）再断连——睡一个
    // 固定的窗口只是在赌调度，而这里断的早晚正是被测的那件事。
    common::await_condition("the remote executor accepted the call", || {
        common::world_log()
            .since(0)
            .unwrap_or_default()
            .iter()
            .any(|e| e.kind.as_str() == "act/started")
    })
    .await;
    c.close().await;
    let out = tokio::time::timeout(std::time::Duration::from_secs(3), propose)
        .await
        .unwrap()
        .unwrap();
    // `Cancelled`，不是 `Failed`：机器人没有"做失败"，是看着它动的那一头没了。理由是
    // 对外契约（`CancelReason::Disconnected`），脑据此知道这次动作可以重来。
    assert_eq!(out["outcome"]["outcome"], "cancelled", "{out}");
    assert_eq!(out["outcome"]["reason"], "adapter disconnected", "{out}");
    let st = brain.call("robot/state", json!({})).await.unwrap();
    assert_eq!(st["fsm"], "STANDING");
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn approval_requests_go_to_the_attached_approver() {
    // profile 里 approval-auto 是 allow；这里把它去掉，并把 rule-run-mode 设成 confirm
    // （每一步都 Ask），让提案走到审批
    let (c, kh, run) = common::boot_with(|rows| {
        rows.retain(|r| r.id != "approval-auto");
        for r in rows.iter_mut() {
            if r.id == "rule-run-mode" {
                r.config = json!({"mode": "confirm"});
            }
        }
    })
    .await;
    let approver = common::connect().await;
    approver
        .call("approval/attach", json!({"name": "console"}))
        .await
        .unwrap();
    let seen = Arc::new(parking_lot::Mutex::new(None::<Value>));
    let s2 = seen.clone();
    approver.on_request(
        "approval/request",
        Arc::new(move |p: Value| {
            *s2.lock() = Some(p);
            Box::pin(async {
                Ok(json!({"decision": "deny", "by": "console", "reason": "not now"}))
            })
        }),
    );
    let mut n = c.notifications();
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"}}}),
    )
    .await
    .unwrap();
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    let turn = common::next_notification(&mut n, "brain/turn").await["turn"]
        .as_u64()
        .unwrap();
    let out = c
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "walk", "params": {"vx": 0.1, "duration_s": 0.1}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "refused", "{out}");
    assert!(
        out["outcome"]["reason"]
            .as_str()
            .unwrap()
            .contains("not now"),
        "{out}"
    );
    assert_eq!(seen.lock().as_ref().unwrap()["spec"]["name"], "walk");
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 一个名字在一条连接上只能注册一次，注销之后可以再来。
///
/// 重名检查是"这个 Fiber 归谁"这件事的地基：两个同名 Fiber 同时活着的话，`plugin/state`
/// 里两条状态流共用一个 `plugin` 名字（内核允许，见 `FiberTree::add`），而客户端自己的
/// 两次注册也再分不清谁是谁——除了 `fiber_id`，没有别的线索。
#[tokio::test]
async fn a_name_registers_once_per_connection_and_unregister_gives_it_back() {
    let (c, kh, run) = common::boot().await;
    let r = c.call("plugin/register", manifest()).await.unwrap();
    let fid = r["fiber_id"].as_u64().unwrap();

    let e = c.call("plugin/register", manifest()).await.unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED, "{e:?}");

    c.call("plugin/unregister", json!({ "fiber_id": fid }))
        .await
        .unwrap();
    // `plugin/unregister` 等内核撤完才回复，所以能力面**已经**缩回去了——不用轮询。
    let caps = c.call("robot/capabilities", json!({})).await.unwrap();
    assert!(!caps["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x["name"] == "wave"));
    let e = c
        .call("plugin/heartbeat", json!({ "fiber_id": fid }))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED, "{e:?}");

    // 名字还回来了：同一条连接可以重新注册（一个重连/重载的适配器就是这么做的）。
    let again = c.call("plugin/register", manifest()).await.unwrap();
    assert_eq!(again["state"], "active", "{again}");
    assert_ne!(again["fiber_id"], r["fiber_id"]);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 执行器**拒收**一次调用（`{accepted:false}`）：提案以 `failed` 收场，理由原样回到脑，
/// 而且机器人交还 `STANDING`——没起跑的运动不该把 FSM 留在 ACTING。
#[tokio::test]
async fn a_refused_capability_execute_fails_the_proposal() {
    let (c, kh, run) = common::boot().await;
    c.on_request(
        "capability/execute",
        Arc::new(|_p: Value| {
            Box::pin(async { Ok(json!({"accepted": false, "reason": "arm is parked"})) })
        }),
    );
    c.call("plugin/register", manifest()).await.unwrap();
    let mut n = c.notifications();
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"}}}),
    )
    .await
    .unwrap();
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    let turn = common::next_notification(&mut n, "brain/turn").await["turn"]
        .as_u64()
        .unwrap();
    let out = c
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "wave", "params": {"times": 1}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "failed", "{out}");
    assert_eq!(out["outcome"]["error"], "arm is parked", "{out}");
    let st = c.call("robot/state", json!({})).await.unwrap();
    assert_eq!(st["fsm"], "STANDING", "{st}");
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `approval/detach` 之后没人把关 → 内核自己拒（`no approval provider`），不是放行。
/// 这是整个审批接缝的 fail-closed 判据：一个"审批端走了就默认同意"的系统，只要把
/// 控制台关掉就能让机器人做任何事。
#[tokio::test]
async fn detaching_the_approver_denies_instead_of_allowing() {
    let (c, kh, run) = common::boot_with(|rows| {
        rows.retain(|r| r.id != "approval-auto");
        for r in rows.iter_mut() {
            if r.id == "rule-run-mode" {
                r.config = json!({"mode": "confirm"});
            }
        }
    })
    .await;
    let approver = common::connect().await;
    approver
        .call("approval/attach", json!({"name": "console"}))
        .await
        .unwrap();
    approver.on_request(
        "approval/request",
        Arc::new(|_p: Value| {
            Box::pin(async { Ok(json!({"decision": "allow", "by": "console", "reason": "ok"})) })
        }),
    );
    approver.call("approval/detach", json!({})).await.unwrap();
    // 拆第二次：这条连接已经没有审批服务了。
    let e = approver
        .call("approval/detach", json!({}))
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED, "{e:?}");

    let mut n = c.notifications();
    c.call(
        "brain/attach",
        json!({"brain_id": "fast", "wake_policy": {"default": "ignore", "rules": {"obs/person": "wake"}}}),
    )
    .await
    .unwrap();
    c.call("event/emit", json!({"kind": "obs/person", "payload": {}}))
        .await
        .unwrap();
    let turn = common::next_notification(&mut n, "brain/turn").await["turn"]
        .as_u64()
        .unwrap();
    let out = c
        .call(
            "turn/propose",
            json!({"brain_id": "fast", "turn": turn, "capability": "walk", "params": {"vx": 0.1, "duration_s": 0.1}}),
        )
        .await
        .unwrap();
    assert_eq!(out["outcome"]["outcome"], "refused", "{out}");
    assert!(
        out["outcome"]["reason"]
            .as_str()
            .unwrap()
            .contains("no approval provider"),
        "{out}"
    );
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 反复"注册→注销"不会让内核和 `robot/health` 随运行时长无限变长，但**刚**撤掉的那个
/// 仍然看得见。
///
/// 这是跨进程插件带来的新形状：在它之前，"往树上加一行"只发生在 boot；有了它，一个带
/// 监督进程的适配器每崩溃一次就加一行，而 `FiberTree`/`Kernel::emitted`/路由的
/// `FiberStates` 三张表都只增不减，`settle()` 每次还要走一遍整棵树。40 轮远超 32 的上限，
/// 所以"有界"这件事是被真的验到的，不是恰好没超。
#[tokio::test]
async fn register_unregister_churn_stays_bounded_but_keeps_the_latest_corpse() {
    let (c, kh, run) = common::boot().await;
    let live_at_boot = c.call("robot/health", json!({})).await.unwrap()["fibers"]
        .as_array()
        .unwrap()
        .len();

    let mut last = 0;
    for i in 0..40 {
        let m = json!({"manifest": {"name": format!("churn-{i}")}, "heartbeat_ms": 10_000});
        let r = c.call("plugin/register", m).await.unwrap();
        last = r["fiber_id"].as_u64().unwrap();
        c.call("plugin/unregister", json!({ "fiber_id": last }))
            .await
            .unwrap();
    }

    let fibers = c.call("robot/health", json!({})).await.unwrap()["fibers"]
        .as_array()
        .unwrap()
        .clone();
    // 上限：boot 时活着的那些 + 最多 32 条墓碑。没有 GC 的话这里是 boot + 40。
    assert!(
        fibers.len() <= live_at_boot + 32,
        "{} fibers after 40 churn cycles (was {live_at_boot} live at boot)",
        fibers.len()
    );
    // 但最后撤掉的那一个必须还在：控制台要看见"刚刚谁掉了"。
    let latest = fibers.iter().find(|f| f["fiber_id"] == last);
    assert!(
        latest.is_some_and(|f| f["state"] == "disposed"),
        "the just-disposed fiber {last} must still be visible: {fibers:?}"
    );
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Plan 3 最终评审 C1 的复现：一条声明里的 `nominal_hz` 必须在**边界上**被挡住。
///
/// 内核按 `nominal_hz × window_s` 给每条流预分配环形缓冲，所以 `1e12` 是一次 176 TB 的
/// 分配请求，而分配失败走 `handle_alloc_error`——abort，不是 panic，`FiberTree::activate`
/// 的 `catch_unwind` 接不住：原来这一个调用就能让整个内核进程消失（实测 SIGABRT，
/// "memory allocation of 176000000000000 bytes failed"）。注册的六道护栏数的是"几条流、
/// 几个能力"，从来不看声明**里面**的数字。
///
/// 这个测试断言的正是它复现时缺的那一半：调用被拒（`INVALID_PARAMS`），**而内核还在**
/// （`robot/health` 照常答，树上没有这一行），后面一次正常的注册照样成功。
#[tokio::test]
async fn an_absurd_nominal_hz_is_refused_and_the_kernel_survives() {
    let (c, kh, run) = common::boot().await;
    let declaring = |hz: Value| {
        json!({
            "manifest": {"name": "firehose",
                         "streams": [{"stream_id": "s", "schema": "oeh.X", "nominal_hz": hz, "tier": "small"}]},
            "heartbeat_ms": 10_000
        })
    };
    // 1e12 是复现用的那个数；0 和负数不是"慢流"而是写错了；20001 刚好越过默认上限。
    for hz in [
        json!(1e12),
        json!(1e8),
        json!(0.0),
        json!(-1.0),
        json!(20_001.0),
    ] {
        let e = c
            .call("plugin/register", declaring(hz.clone()))
            .await
            .unwrap_err();
        assert_eq!(
            e.code,
            oeh_proto::rpc::code::INVALID_PARAMS,
            "nominal_hz {hz}: {e:?}"
        );
    }

    // 内核还活着，而且被拒的注册一行都没留下。
    let h = c.call("robot/health", json!({})).await.unwrap();
    assert!(
        !h["fibers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["plugin"] == "firehose"),
        "a refused registration must not leave a fiber: {h}"
    );
    // 上限之内的照常通过——这道护栏不该顺手挡住真实适配器（最快的声明是 500 Hz）。
    let ok = c
        .call("plugin/register", declaring(json!(500.0)))
        .await
        .unwrap();
    assert_eq!(ok["state"], "active", "{ok}");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `plugin/register` 的每一道护栏都拒绝，而且一个 Fiber 都不留下——大小上限是
/// `INVALID_PARAMS`，每连接的活 Fiber 数上限是 `REFUSED`（Task 5 re-review R2）。
///
/// 为什么大小上限是 `INVALID_PARAMS` 而不是一个 `Failed` 的 Fiber：超限是调用方写错了
/// 参数，和写错一个 `class` 同一类。编成 Fiber 的话，客户端得先拿到 200、再去读 `state`
/// 才知道自己被拒了，而内核已经为此在树上留了一行。
///
/// 为什么每连接的 Fiber 数上限不一样：它不是参数错误——同样的 params 在一次
/// `plugin/unregister` 之后原样重发就能成功，这正是"现在不行"（`REFUSED`）而不是"这次
/// 调用本身就是错的，别重试"（`INVALID_PARAMS`）。
///
/// 为什么这些护栏存在：`RemotePlugin::apply` 跑在内核那条任务上，一份声明了十万条流的
/// manifest 就是十万次同步注册占着它，期间每个真实传感器的帧都在排队。
#[tokio::test]
async fn register_bounds_are_invalid_params_and_leave_no_fiber() {
    let (c, kh, run) = common::boot().await;
    let stream = |i: usize| json!({"stream_id": format!("s{i}"), "schema": "oeh.X", "nominal_hz": 1.0, "tier": "small"});
    let cap = |i: usize| json!({"name": format!("c{i}"), "class": "query", "risk_level": "none", "lease_ttl_s": 1.0});
    let names = |n: usize| (0..n).map(|i| format!("svc{i}")).collect::<Vec<_>>();

    let cases: Vec<(&str, Value)> = vec![
        (
            "streams",
            json!({"manifest": {"name": "over", "streams": (0..65).map(stream).collect::<Vec<_>>()}}),
        ),
        (
            "capabilities",
            json!({"manifest": {"name": "over"}, "capabilities": (0..65).map(cap).collect::<Vec<_>>()}),
        ),
        (
            "inject",
            json!({"manifest": {"name": "over", "inject": names(65)}}),
        ),
        (
            "provide",
            json!({"manifest": {"name": "over", "provide": names(65)}}),
        ),
        (
            "bytes",
            // 一个 300 KiB 的 config：数量上一条流都没有，但字节数超了。这一条是唯一
            // 把 `config` / `params_schema` / manifest 一次罩住的护栏。
            json!({"manifest": {"name": "over"}, "config": {"blob": "x".repeat(300 * 1024)}}),
        ),
    ];
    for (what, params) in cases {
        let e = c.call("plugin/register", params).await.unwrap_err();
        assert_eq!(
            e.code,
            oeh_proto::rpc::code::INVALID_PARAMS,
            "{what}: {e:?}"
        );
    }

    // 每条连接的活 Fiber 数也有上限——重名检查挡不住它（`f0`…`f16` 全是不同的名字）。
    // 这一条是 `REFUSED`，不是 `INVALID_PARAMS`：满了是状态，不是参数错误（R2）。
    for i in 0..16 {
        c.call(
            "plugin/register",
            json!({"manifest": {"name": format!("f{i}")}, "heartbeat_ms": 10_000}),
        )
        .await
        .unwrap_or_else(|e| panic!("registration {i} should fit under the limit: {e:?}"));
    }
    let e = c
        .call(
            "plugin/register",
            json!({"manifest": {"name": "f16"}, "heartbeat_ms": 10_000}),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED, "{e:?}");

    // 被拒的那些一个 Fiber 都没留下：树上只有 boot 的那些 + 刚才成功的 16 个。
    let h = c.call("robot/health", json!({})).await.unwrap();
    let fibers = h["fibers"].as_array().unwrap();
    assert!(
        !fibers.iter().any(|f| f["plugin"] == "over"),
        "a refused registration must not leave a fiber: {h}"
    );
    assert!(
        !fibers.iter().any(|f| f["plugin"] == "f16"),
        "the 17th registration must not exist either: {h}"
    );
    assert_eq!(
        (0..16)
            .filter(|i| fibers
                .iter()
                .any(|f| f["plugin"] == format!("f{i}") && f["state"] == "active"))
            .count(),
        16,
        "the 16 that fit under the limit are all live: {h}"
    );
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// `approval/attach` 是抢占式的，被顶掉的那条连接会被**告知**。
///
/// 没有这条通知的话，一个控制台会一直显示"我是审批人"，却再也不会被问到任何事——而它
/// 的 `approval/detach` 还会回 `Ok`（那个句柄早就是个空操作）。这是这条接缝最容易骗人
/// 的一种状态。
#[tokio::test]
async fn attaching_an_approver_displaces_the_previous_one_and_says_so() {
    let (_c, kh, run) = common::boot().await;
    let first = common::connect().await;
    let mut n = first.notifications();
    first
        .call("approval/attach", json!({"name": "console-a"}))
        .await
        .unwrap();

    let second = common::connect().await;
    second
        .call("approval/attach", json!({"name": "console-b"}))
        .await
        .unwrap();

    let m = common::next_notification(&mut n, "approval/detached").await;
    assert_eq!(m["reason"], "replaced");
    // 顶掉之后，第一条连接手里已经没有审批登记了——`approval/detach` 如实说"没有"，
    // 而不是回一个什么都没拆的 `Ok`。
    let e = first.call("approval/detach", json!({})).await.unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::NOT_ATTACHED, "{e:?}");
    // 接管的那一条还在。
    second.call("approval/detach", json!({})).await.unwrap();
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 一次**被拒绝**的 `plugin/register` 不能动别人的东西。
///
/// 场景（R1）：一个控制台订阅了 `lowstate`（`mock-stream` 产的本地流），正从共享内存通道上
/// 读帧。另一个客户端——手滑的或者恶意的——注册一个声明了同一个 `stream_id` 的插件。那次
/// 注册必须失败（流名已经被占），而且必须**只**失败：控制台的通道上得继续有帧。
#[tokio::test]
async fn a_refused_registration_does_not_silence_an_existing_frame_consumer() {
    let (c, kh, run) = common::boot().await;
    let ch = c
        .call("stream/subscribe", json!({"stream_id": "lowstate"}))
        .await
        .unwrap();
    let channel = ch["channel"].as_str().unwrap().to_string();

    // 同一个 `stream_id`，来自另一个客户端。
    let c2 = common::connect().await;
    let r = c2
        .call(
            "plugin/register",
            json!({
                "manifest": {"name": "squatter",
                             "streams": [{"stream_id": "lowstate", "schema": "oeh.LowState",
                                          "nominal_hz": 50.0, "tier": "small"}]},
                "heartbeat_ms": 10_000
            }),
        )
        .await
        .unwrap();
    assert_eq!(r["state"], "failed", "{r}");
    assert!(
        r["error"].as_str().unwrap_or_default().contains("lowstate"),
        "{r}"
    );

    // 通道上照样有帧（500 Hz 的流，三帧是 6 ms 的事；3 s 只是失败的期限）。
    let got = tokio::task::spawn_blocking(move || {
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
        let t0 = std::time::Instant::now();
        let mut n = 0;
        while t0.elapsed() < std::time::Duration::from_secs(3) && n < 3 {
            while let Ok(Some(_)) = sub.receive() {
                n += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        n
    })
    .await
    .unwrap();
    assert!(
        got >= 3,
        "the egress went silent after an unrelated registration was refused ({got} frames)"
    );

    c2.close().await;
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
