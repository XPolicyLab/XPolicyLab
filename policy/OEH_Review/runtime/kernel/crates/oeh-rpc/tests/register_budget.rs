//! 握手总预算与保留服务名（Plan 3 遗留 I6 / M12）。
//!
//! `plugin/register` 是边界协议里唯一一个会**改变内核树**的方法，而它的代价由调用方的
//! manifest 说了算：每条声明的流都是内核任务上的一次同步注册加一条 ingress。数量护栏
//! （`max_streams_per_register` 那六道）管的是"最多几条"，管不了"每条有多慢"——
//! `register_timeout_ms` 是那个缺口上的总预算。
//!
//! 慢是用测试钩子 `stream_setup_delay_ms` 造出来的（只在 `test-hooks` feature 下存在，
//! 产品构建里它根本不是一个配置键）：异步 sleep，所以运行时照常转，超预算走的是
//! `select!` 里那条**把收据交给清理任务**的路——两条清理路径里更难的那一条。

mod common;

use oeh_proto::rpc::code;
use serde_json::{json, Value};

const PLUGIN: &str = "slow-arm";
/// 这个插件声明的服务名。重试那一次靠它跟前一次的"幽灵"撞车：服务名冲突会让 Fiber
/// `Failed`，所以它是"上一行到底撤干净了没有"在外部唯一看得见的判据。
const SERVICE: &str = "adapter.slow";
/// 预算，和下面两个数字一起构成这个文件的算术：2 条流 × 400 ms = 800 ms > 600 ms。
const BUDGET_MS: u64 = 600;
const PER_STREAM_MS: u64 = 400;

fn manifest_with_streams(n: usize) -> Value {
    let streams: Vec<Value> = (0..n)
        .map(|i| {
            json!({"stream_id": format!("s{i}"), "schema": "oeh.Probe",
                   "nominal_hz": 1.0, "tier": "small"})
        })
        .collect();
    json!({
        "manifest": {"name": PLUGIN, "inject": [], "provide": [SERVICE], "streams": streams},
        "capabilities": [],
        "heartbeat_ms": 10_000
    })
}

fn slow_rpc_server(rows: &mut [oeh_kernel::profile::EntryRow], budget_ms: u64) {
    for r in rows.iter_mut() {
        if r.id == "rpc-server" {
            r.config["register_timeout_ms"] = json!(budget_ms);
            r.config["stream_setup_delay_ms"] = json!(PER_STREAM_MS);
        }
    }
}

/// 2 条流 × 400 ms = 800 ms，越过 600 ms 的预算。应答是
/// `REFUSED{"register budget exceeded"}`，而且**什么都不留下**。
///
/// "不留下"在这里是一条**直接**断言，不是轮询等出来的：超预算那条路把注册闸门一起交给
/// 了清理任务，撤干净才放开（见 `handlers::plugin::dispose_over_budget`）。所以同一条
/// 连接上紧接着的那次重试——它要等的正是那把闸门——返回的时候，上一行连同它的流已经
/// 不在了。重试声明同一个服务名，冲突因此是真的：幽灵还在，它就只能是 `failed`。
///
/// 为什么不是计划里写的 64 条流：每条声明的流都会真开一条 iceoryx2 ingress（一个线程 +
/// 一段共享内存，开口有 250 ms 的预算），而 `cargo test` 的测试二进制之间也是并发的——
/// 64 条一次性压上去，倒下的不是这个测试，是同一时刻正在开自己那条通道的**别的**测试。
/// 被测性质（"一次慢注册必须被预算掐掉，并且不留痕迹"）跟条数无关，跟总时长有关。
#[tokio::test]
async fn a_registration_over_the_budget_is_refused_and_leaves_nothing_behind() {
    let (c, kh, run) = common::boot_with(|rows| slow_rpc_server(rows, BUDGET_MS)).await;

    let e = c
        .call("plugin/register", manifest_with_streams(2))
        .await
        .unwrap_err();
    assert_eq!(e.code, code::REFUSED, "{e:?}");
    assert_eq!(e.message, "register budget exceeded", "{e:?}");

    // 立刻重试，同一条连接、同一个插件名、同一个服务名，但不声明流（于是它自己不慢）。
    // 它唯一要等的就是那把还攥在清理任务手里的闸门。
    let retry = c
        .call("plugin/register", manifest_with_streams(0))
        .await
        .unwrap();
    assert_eq!(retry["state"], "active", "{retry}");
    assert_eq!(retry["error"], json!(null), "{retry}");

    // 重试回来了 = 清理跑完了，所以下面两条不用等。
    let streams = c.call("stream/list", json!({})).await.unwrap();
    let leaked: Vec<&str> = streams["streams"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["stream_id"].as_str())
        .filter(|id| id.starts_with('s') && id[1..].chars().all(|ch| ch.is_ascii_digit()))
        .collect();
    assert!(
        leaked.is_empty(),
        "the refused registration left {leaked:?}"
    );
    let health = c.call("robot/health", json!({})).await.unwrap();
    let live: Vec<&Value> = health["fibers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["plugin"] == PLUGIN && f["state"] != "disposed")
        .collect();
    assert_eq!(
        live.len(),
        1,
        "only the retry's fiber should be live: {live:?}"
    );

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 控制组：**只**把预算放宽，别的一个字不改——同一份 manifest、同一个 400 ms/条的钩子。
/// 少了它，上面那条测试的失败可能来自 manifest 本身。
#[tokio::test]
async fn the_same_manifest_registers_when_only_the_budget_is_wider() {
    let (c, kh, run) = common::boot_with(|rows| slow_rpc_server(rows, 30_000)).await;
    let r = c
        .call("plugin/register", manifest_with_streams(2))
        .await
        .unwrap();
    assert_eq!(r["state"], "active", "{r}");
    assert_eq!(r["streams"].as_array().unwrap().len(), 2, "{r}");
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 保留服务名（M12）：`provide` 里出现内核自己的名字 → `INVALID_PARAMS`，而且是在
/// **造出任何东西之前**——一个占着 `clock` 名字的远端服务会让每一个 `inject: [clock]`
/// 的进程内插件在下一次 settle 里接到一个它完全不认识的东西。
#[tokio::test]
async fn a_reserved_service_name_is_refused_before_anything_is_created() {
    let (c, kh, run) = common::boot().await;
    let e = c
        .call(
            "plugin/register",
            json!({
                "manifest": {"name": "impostor", "inject": [], "provide": ["clock"], "streams": []},
                "capabilities": [],
                "heartbeat_ms": 10_000
            }),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, code::INVALID_PARAMS, "{e:?}");
    assert_eq!(e.message, "reserved service name: clock", "{e:?}");
    // 树上没有这一行：被拒的注册连 `Op::AddFiber` 都没投出去过。
    let health = c.call("robot/health", json!({})).await.unwrap();
    assert!(
        !health["fibers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["plugin"] == "impostor"),
        "{health}"
    );
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
