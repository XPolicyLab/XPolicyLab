//! 本机鉴权（Plan 3 遗留 I5）：UDS socket 一绑出来就是 `0600`，accept 之后还要再看一次
//! `SO_PEERCRED`——白名单之外的 uid 连 `ConnState` 都进不去，只在世界日志里留下一条
//! `safety/auth_refused{uid, reason}`。
//!
//! 两道是故意的，不是重复：权限位挡的是"别的用户 connect(2) 根本连不上"，白名单挡的是
//! "同一个用户（或 root）下的另一个进程"——前者由内核代劳、便宜且可靠，后者是这条边界
//! 自己的策略，而且是唯一一道能在日志里留下痕迹的。

mod common;

use std::os::unix::fs::PermissionsExt;

use serde_json::{json, Value};

/// 一份最小的注册：不声明流、不声明能力——这些测试问的是"这条连接算不算数"，
/// 不是"注册了什么"。
fn manifest() -> Value {
    json!({
        "manifest": {"name": "py-arm", "inject": [], "provide": [], "streams": []},
        "capabilities": [],
        "heartbeat_ms": 10_000
    })
}

/// (a) socket 文件的权限位。从 RPC 那一侧完全看不见，所以只能直接看文件。
#[tokio::test]
async fn the_uds_socket_is_bound_with_mode_0600() {
    let (_c, kh, run) = common::boot().await;
    let mode = std::fs::metadata(common::sock_path())
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "socket mode is {:o}", mode & 0o777);
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// (b) 白名单之外的 uid：socket 连得上（connect(2) 是内核按权限位判的，而这个测试就是
/// 当前这个用户自己），但第一条请求就得到"连接关了"，世界日志里有
/// `safety/auth_refused{uid}`，而活跃连接表从头到尾没有过这条连接。
///
/// `peer_uids: [4294967294]`（`u32::MAX - 1`）是一个不可能存在的 uid：这样"被拒"就不
/// 依赖测试跑在哪个用户下。
#[tokio::test]
async fn a_peer_outside_the_whitelist_never_becomes_a_connection() {
    let (kh, run) = common::boot_unconnected(|rows| {
        for r in rows.iter_mut() {
            if r.id == "rpc-server" {
                r.config["peer_uids"] = json!([4_294_967_294u64]);
            }
        }
    })
    .await;
    let c = common::connect().await;

    // **主信号**是那条事件：socket 层面 connect(2) 是成功的（权限位允许当前用户），被拒
    // 发生在 accept 之后，所以"服务器拒了我"这件事在客户端那一侧只表现为对端挂断——而
    // 挂断和"内核正在关"、"socket 路径写错了"长得一模一样。日志里这条才是判据。
    let uid = oeh_rpc::transport::current_uid().expect("the current uid");
    common::await_condition("safety/auth_refused in the world log", || {
        common::world_log()
            .since(0)
            .unwrap_or_default()
            .iter()
            .any(|e| e.kind.as_str() == "safety/auth_refused" && e.payload["uid"] == json!(uid))
    })
    .await;
    let refusal = common::world_log()
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "safety/auth_refused")
        .unwrap();
    assert_eq!(refusal.payload["reason"], "uid is not in peer_uids");
    assert_eq!(refusal.salience, 200);
    assert_eq!(
        refusal.collapse_key.as_deref(),
        Some(format!("auth_refused:{uid}").as_str())
    );

    // 客户端那一侧只能看见"对端挂断了"。连文本一起断言：`REFUSED` 这个码客户端自己在
    // 任何一次 EOF 上都会合成出来，光看码分不出"服务器拒了我"和"服务器回了一个拒绝"。
    let e = c.call("plugin/register", manifest()).await.unwrap_err();
    assert_eq!(e.code, oeh_proto::rpc::code::REFUSED, "{e:?}");
    assert_eq!(e.message, "connection closed", "{e:?}");
    // 被拒的对端不进连接表：`robot/health.connections` 不该把它数进去，监听器的
    // `Disposer` 也不该有一条它要去通知的连接。
    assert_eq!(common::conns().len(), 0);
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// (c) 默认白名单就是**当前 uid**：不写任何配置，正常的一次注册一路到底。
#[tokio::test]
async fn the_default_whitelist_admits_the_current_uid() {
    let (c, kh, run) = common::boot().await;
    let r = c.call("plugin/register", manifest()).await.unwrap();
    assert_eq!(r["state"], "active", "{r}");
    assert!(common::world_log()
        .since(0)
        .unwrap_or_default()
        .iter()
        .all(|e| e.kind.as_str() != "safety/auth_refused"));
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
