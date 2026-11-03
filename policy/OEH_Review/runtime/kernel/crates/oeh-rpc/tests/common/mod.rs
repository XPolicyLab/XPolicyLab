//! 路由测试的脚手架：起一个带 `rpc-server` 的真内核，连上去，等它进入一个确定的状态。
//!
//! 每个测试一个 tempdir，UDS 路径写进 `rpc-server` 那一行的 **config**——不走
//! `OEH_RPC_UDS` 环境变量：环境变量是进程全局的，而同一个测试二进制里的测试是并发跑的，
//! 一个 `set_var` 会把别的测试的 socket 路径改掉。`ws: null` 是同一件事说得再明白一点
//! （它本来就是默认值）：监听端口只有一个，两个并发的测试会在绑定这一步就撞车。
//!
//! profile 是 `bundles/dev-mock.yaml` 去掉 `mock-brain`（脑由测试自己通过
//! `brain/attach` 扮演）、去掉持久化（世界日志只在内存里，`log/replay` 因此必然
//! `REFUSED`，这正是其中一个测试要验的），并且把规则链收窄到 `rule-whitelist` +
//! `rule-fsm`：
//!
//! - `admission.required_rules` 仍然是 dev-mock 的完整八条，所以链的**初始判定是
//!   `Ask`**（`evaluate_chain`：缺必需规则 = 什么都要问人）——配上 `approval-auto`
//!   的 `allow: true`，一次提案会实打实地走完 ask → decision → recheck → admitted →
//!   started → result 整条路，而不是抄近道；
//! - 没有 `rule-scene`/`rule-pose`/`rule-watchdog` 意味着测试不用先喂一条
//!   `obs/detections`、不用等蒸馏器写够寄存器，就能拿到一个确定的 `executed`。这些规则
//!   本身由 `oeh-admission` 的规则测试和 `plan2b_e2e` 覆盖，不是边界协议要验的东西。

// 每个集成测试二进制都会把这个模块**单独**编译一遍，而它们各用其中的一个子集
// （`tests/router.rs` 不用 `await_condition`，`tests/remote.rs` 不用 `conns()`……）。
// 没有这一行的话，"另一个二进制在用"的辅助函数会在这个二进制里报成死代码，而
// `scripts/ci.sh` 的 clippy 是 `-D warnings`。
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::profile::EntryRow;
use oeh_kernel::{BootError, Kernel, KernelHandle, WorldLog};
use oeh_rpc::client::Client;
use oeh_rpc::transport::ConnRegistry;
use serde_json::json;

// 插件工厂是 `inventory` 注册的：这两个 crate 不被引用就不会被链进测试二进制，
// `admission` / `rule-*` / `distill-fsm` 这些行会解析不出来。
use oeh_admission as _;
use oeh_distill as _;

pub type Run = tokio::task::JoinHandle<Result<(), BootError>>;

/// 一次 `boot()` 的临时目录（socket、会话日志、急停标志都在里面），外加几个测试要直接
/// 看的内核服务。存在 thread-local 里而不是还给测试：`common::connect()` 按 brief 的签名
/// 是不带参数的，它得有办法找到**这个测试的** socket；而 `TempDir` 一旦被 drop，目录连同
/// socket 文件就没了。
///
/// 为什么 thread-local 够用（也是它的使用边界）：`boot()` 在测试 future 的某一次 `poll`
/// 里写，`connect()`/`world_log()` 在后来的某一次 `poll` 里读，所以**这些 poll 必须在同一
/// 个线程上**。`Runtime::block_on` 在调用它的那个线程上驱动根 future，`#[tokio::test]` 的
/// 两种 flavor 都一样，而 libtest 又给每个测试自己的线程——所以它成立，前提是测试体本身
/// 没有被 `tokio::spawn` 出去（那样根 future 才会跑到 worker 上）。
/// `log_tail_is_serialised_per_connection` 是 multi_thread flavor 且调用了 `connect()`，
/// 所以这个前提有人替它守着：一旦不再成立，那个测试会带着下面的 `expect` 消息当场响。
struct Harness {
    _tmp: tempfile::TempDir,
    sock: PathBuf,
    log: Arc<WorldLog>,
    conns: Arc<ConnRegistry>,
    run_tag: String,
}

thread_local! {
    static HARNESS: RefCell<Option<Harness>> = const { RefCell::new(None) };
}

fn with_harness<T>(f: impl FnOnce(&Harness) -> T) -> T {
    HARNESS.with(|h| {
        f(h.borrow()
            .as_ref()
            .expect("common::boot() must run before this, on the same thread"))
    })
}

/// 这个内核的 UDS socket 路径。`auth.rs` 直接看它的权限位——0600 是一条从 RPC 那一侧
/// 完全看不见的不变式。
pub fn sock_path() -> PathBuf {
    with_harness(|h| h.sock.clone())
}

/// 这个内核的帧通道命名空间（`rpc-server` 的 `run_tag` 配置项）。
///
/// 每个 booted kernel 一个唯一的名字（`t<pid>-<序号>`），因为 iceoryx2 的服务名是**机器
/// 全局**的：内存日志没有 run 目录，所有测试内核否则都会叫 `local-<pid>`，同一个进程里
/// 并发跑的测试就会共用 `oeh/local-<pid>/arm_state`——订阅者位置会被挤满，而且一个测试
/// 内核会把另一个测试发的帧当成自己的流收进 `FrameStore`。
pub fn run_tag() -> String {
    with_harness(|h| h.run_tag.clone())
}

/// 见 [`run_tag`]：进程内唯一，且带上 pid 之后跨进程也唯一（两个并发的 `cargo test`）。
fn next_run_tag() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    format!(
        "t{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

/// 这个内核的世界日志。测试用它直接 append（比几百次 `event/emit` 往返快得多），以及
/// 读 `observer_count()` —— 一条泄漏的观察者从 RPC 那一侧完全看不出来。
pub fn world_log() -> Arc<WorldLog> {
    with_harness(|h| h.log.clone())
}

/// 这个内核的活跃连接表（`rpc-server` 插件 `provide` 的 `"rpc"` 服务）。测试用它够到
/// 服务端的 `Conn`，去看那些只存在于连接自己状态里的东西（比如 turn 句柄表的大小）。
pub fn conns() -> Arc<ConnRegistry> {
    with_harness(|h| h.conns.clone())
}

/// dev-mock 的行，减去 mock-brain 与持久化，加上 `rpc-server`。
fn rows(tmp: &Path, sock: &Path, run_tag: &str) -> Vec<EntryRow> {
    vec![
        EntryRow::new("clock"),
        // 没有 `dir`：纯内存日志，没有 run 目录，也就没有段文件可回放。
        EntryRow::new("world-log"),
        EntryRow::new("frame-store").with_config(json!({"window_s": 2.0})),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("mock-stream")
            .with_config(json!({"stream_id": "lowstate", "hz": 500.0, "payload": "lowstate"})),
        // lowstate 帧 → `gravity_proj_z` / `rl_policy_active` 寄存器 → BOOT → STANDING。
        // motion 类能力要过 `rule-fsm` 就得先有这一步。
        EntryRow::new("distill-fsm").with_config(json!({"stream_id": "lowstate"})),
        EntryRow::new("brains").with_config(json!({"session_dir": tmp.join("sessions")})),
        EntryRow::new("admission").with_config(json!({
            "required_rules": [
                "rule-whitelist", "rule-fsm", "rule-run-mode", "rule-watchdog",
                "rule-pose", "rule-clamp", "rule-scene", "rule-estop"
            ],
            // 默认是 `/tmp/oeh_estop`：开发机上真有这么一个文件的话，每次提案都会被
            // `rule-estop` 拦下。测试只该看见自己造的世界。
            "estop_flag": tmp.join("estop"),
        })),
        EntryRow::new("approval-auto").with_config(json!({"allow": true})),
        EntryRow::new("mock-executor").with_config(json!({
            "capabilities": [{
                "name": "walk",
                "class": "motion",
                "risk_level": "medium",
                "params_schema": {
                    "type": "object",
                    "properties": {"vx": {"type": "number"}, "duration_s": {"type": "number"}}
                },
                "lease_ttl_s": 2.0
            }],
            // dev-mock 是 0.5 s；测试里没必要等那么久。
            "duration_s": 0.05
        })),
        // `wave` 是 `tests/remote.rs` 里那个跨进程插件注册的能力：白名单是运维的静态
        // 声明，一个还没注册的名字列在这里是合法的（`rule-whitelist` 只按名字放行）。
        EntryRow::new("rule-whitelist").with_config(
            json!({"allowed": ["walk", "wave", "say", "ask_human", "stop", "recover"]}),
        ),
        EntryRow::new("rule-fsm"),
        // `active` = 这条规则不自己收紧判定，交给链上其余规则（等价于它不在链里，所以
        // 别的测试的判定一条都没变）。`tests/remote.rs` 的审批测试把它改成 `confirm`，
        // 那是唯一一个**稳定**产生 `Ask` 的开关。
        EntryRow::new("rule-run-mode").with_config(json!({"mode": "active"})),
        EntryRow::new("rpc-server")
            .with_config(json!({"uds": sock, "ws": null, "run_tag": run_tag})),
    ]
}

pub async fn boot() -> (Client, KernelHandle, Run) {
    boot_with(|_| {}).await
}

/// `boot()`，但先让调用方改几行（加一个插件、换一个配置）。
///
/// 返回之前会等到一个**确定的**起始状态：`rpc-server` 这个 Fiber 已经出现在
/// `plugin/state` 快照里（内核 `run()` 的第一次 `publish_states` 跑过了），而且 FSM 已经
/// 被 `distill-fsm` 从 BOOT 推到了 STANDING（500 Hz 的流，第一帧 2 ms 就到）。不等的话，
/// 一次 `turn/propose walk` 会因为 `rule-fsm` 看到 BOOT 而被拒，且拒不拒全看调度。
pub async fn boot_with(patch: impl FnOnce(&mut Vec<EntryRow>)) -> (Client, KernelHandle, Run) {
    let (kh, run) = boot_unconnected(patch).await;
    let c = connect().await;
    await_ready(&c).await;
    (c, kh, run)
}

/// `boot_with`，但**不替调用方连接**。
///
/// 给鉴权测试用：`peer_uids` 把当前 uid 挡在外面时，`boot_with` 自己那条连接会在
/// `await_ready` 的第一次 `robot/health` 上死掉——而"第一条连接就被拒"正是那个测试要
/// 验的东西，不该由夹具来撞。也因此这里不等 FSM 到 STANDING：那一步要过 RPC。
pub async fn boot_unconnected(patch: impl FnOnce(&mut Vec<EntryRow>)) -> (KernelHandle, Run) {
    let tmp = tempfile::tempdir().unwrap();
    let sock = tmp.path().join("oeh.sock");
    let run_tag = next_run_tag();
    let mut rows = rows(tmp.path(), &sock, &run_tag);
    patch(&mut rows);

    let kernel = Kernel::boot(rows).unwrap();
    assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
    assert!(kernel.report().pending.is_empty(), "{:?}", kernel.report());
    // 趁 `run()` 还没把 `Kernel` 吃掉，从服务注册表里抓两个测试要直接看的服务。
    let log = kernel
        .registry()
        .get::<WorldLog>(&"world-log".into())
        .unwrap();
    let conns = kernel
        .registry()
        .get::<ConnRegistry>(&"rpc".into())
        .unwrap();
    let kh = kernel.handle();
    let run = tokio::spawn(kernel.run());

    HARNESS.with(|h| {
        *h.borrow_mut() = Some(Harness {
            _tmp: tmp,
            sock: sock.clone(),
            log,
            conns,
            run_tag,
        })
    });
    (kh, run)
}

/// 再开一条到同一个内核的连接（`boot()` 那条之外）。
pub async fn connect() -> Client {
    let sock = sock_path();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match Client::connect_uds(&sock).await {
            Ok(c) => return c,
            Err(e) if std::time::Instant::now() >= deadline => {
                panic!("connect {}: {e}", sock.display())
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
}

async fn await_ready(c: &Client) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let health = c.call("robot/health", json!({})).await.unwrap();
        let fibers_up = health["fibers"]
            .as_array()
            .is_some_and(|f| f.iter().any(|f| f["plugin"] == "rpc-server"));
        let state = c.call("robot/state", json!({})).await.unwrap();
        if fibers_up && state["fsm"] == "STANDING" {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "kernel never settled: health={health}, state={state}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// 等一个条件成立（5 ms 一轮，10 s 上限）。给那些**没有通知可等**的同步点用（"内核
/// 那一侧已经把请求发出去了"这类），比睡一个固定的窗口好：窗口猜短了测的就不是要测的
/// 那件事，猜长了每次跑都白等。
pub async fn await_condition(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if cond() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// 等一条指定方法的通知。所有等通知的测试都走这里：裸 `recv()` 在断言失败时会一直挂着，
/// 报出来的是一个 60 s 的测试超时，而不是"这条通知没来"。
pub async fn next_notification(
    n: &mut tokio::sync::broadcast::Receiver<oeh_proto::rpc::Notification>,
    method: &str,
) -> serde_json::Value {
    let got = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match n.recv().await {
                Ok(m) if m.method == method => return m.params,
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(e) => panic!("notifications closed while waiting for {method}: {e}"),
            }
        }
    })
    .await;
    got.unwrap_or_else(|_| panic!("timed out waiting for a `{method}` notification"))
}

/// 等 `log/tail` 推来的某一条**具体**事件（按 seq）。订阅里混着别的事件（`brain/*` 的
/// bookkeeping、别的 `obs/*`）是常态，"下一条 event" 不等于"我要的那条"。
pub async fn next_event(
    n: &mut tokio::sync::broadcast::Receiver<oeh_proto::rpc::Notification>,
    seq: u64,
) -> serde_json::Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let ev = next_notification(n, "event").await;
        if ev["seq"] == json!(seq) {
            return ev;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for event seq {seq}"
        );
    }
}
