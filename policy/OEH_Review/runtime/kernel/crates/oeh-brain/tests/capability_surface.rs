//! 能力面是活的：一个**进程外**插件注册 / 断连之后，进程内的脑在**下一个** turn 就看见
//! 了新的工具表，而正在跑的那个 turn 抱着它开始时的那一张跑完。
//!
//! 这是 Plan 5 最终评审 I2 的回归点。在它之前，`brain-native` 的工具表是能力面在
//! `apply` 那一刻的一份快照：这套 profile（以及 `g1-sim` / `g1-real`）里 `walk` 是适配器
//! 连上来之后才注册的，于是一个进程内的脑**永远**只看得见内核自带的那几条，调
//! `robot_walk` 只会拿到 `unknown tool`。
//!
//! 三件事在一次 boot 里验完（起内核不便宜，而这三件本来就是同一个故事的三段）：
//!
//! 1. 适配器连上来之前：工具表里没有 `robot_walk`；
//! 2. 一个 turn 跑到一半时远端注册了 `walk` → **这个** turn 的第二个 step 仍然用旧表
//!    （中途换表会让请求里出现一条指向已消失工具的 `tool_result`，三家厂商都会拒掉整个
//!    请求），而**下一个** turn 的表里有 `robot_walk`；
//! 3. 远端断连之后：再下一个 turn 的表里又没有它了。
//!
//! 判别信号全部是世界日志里的事件：`brain/request.tool_names` 是"这个 step 手上有哪些
//! 工具"的唯一凭据，`capability/registered` / `capability/unregistered` 是"能力面动过了"
//! 的同步点，慢工具的 `Notify` 是"这个 turn 真的跑到一半了"的同步点。没有一处用 sleep
//! 当判据。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::capability::{
    CapabilityClass, CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor, RiskLevel,
};
use oeh_brain as _;
use oeh_distill as _;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::{Ctx, Kernel, KernelHandle, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, ServiceName, WorldEvent};
use oeh_rpc::client::Client;
use serde_json::{json, Value};
use tokio::sync::Notify;

// ---------------------------------------------------------------- 慢工具

/// 测试与"慢"能力之间的两个同步点：执行器按下 `started`，测试按下 `release`。
/// `Notify::notify_one` 会把许可存下来，所以两边谁先到都行。
#[derive(Default)]
struct Gate {
    started: Notify,
    release: Notify,
}

struct SlowExec {
    gate: Arc<Gate>,
}

impl Executor for SlowExec {
    fn execute(&self, _call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let g = self.gate.clone();
        Box::pin(async move {
            g.started.notify_one();
            g.release.notified().await;
            ExecOutcome {
                ok: true,
                result: json!({"released": true}),
                error: None,
            }
        })
    }
}

/// 只在这个测试文件里存在的插件：注册一个 `slow` 能力（query 类，不占 motion 的位子），
/// 它的执行器一直等到测试放行。有了它，"一个 turn 正跑到一半"才是一个可等的同步点。
struct SlowPlugin {
    gate: Arc<Gate>,
}

impl Plugin for SlowPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("test-slow").inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, _config: &Value) -> Result<(), PluginError> {
        let adm = ctx.get::<Admission>("admission")?;
        let d = adm
            .capabilities()
            .register(
                CapabilitySpec {
                    name: "slow".into(),
                    class: CapabilityClass::Query,
                    risk_level: RiskLevel::None,
                    params_schema: json!({"type": "object"}),
                    // 测试自己决定这次执行什么时候结束，租约不该替它决定。
                    lease_ttl_s: 600.0,
                    overridable: false,
                },
                Arc::new(SlowExec {
                    gate: self.gate.clone(),
                }),
            )
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(d);
        Ok(())
    }
}

// ---------------------------------------------------------------- 脚手架

/// 只让 `obs/gesture` 唤醒，`capability/*` 明确 ignore——shipped 的三份唤醒策略都是这么
/// 写的，而这个测试正是要证明**进程内**的脑不靠那两条事件也能看见能力面的变化。
const WAKE_YAML: &str = "\
default: accumulate
rules:
  obs/gesture: wake
  brain/*: ignore
  act/*: ignore
  health/*: ignore
  safety/*: ignore
  plugin/*: ignore
  capability/*: ignore
budget: { max_consecutive_wakes: 32, min_interval_ms: 0 }
";

/// 每个 turn 一步，都只说一句话——除了第一个 turn，它调一次那个慢工具，好让测试在
/// "turn 跑到一半"的那一刻做点别的。
const SCRIPT: &str = "\
- text: \"turn 1 step 0\"
  tool_calls: [{ name: robot_slow, args: {} }]
- text: \"turn 1 step 1\"
- text: \"turn 2\"
- text: \"turn 3\"
- text: \"turn 4\"
";

/// 远端插件的 manifest：一条 `walk` 能力，没有流。心跳周期长到心跳监视在这个测试的生命
/// 期里不可能开火——这里要验的是**断连**，不是失联。
fn remote_manifest() -> Value {
    json!({
        "manifest": {"name": "py-arm", "inject": ["clock"], "provide": ["adapter.arm"]},
        "capabilities": [{
            "name": "walk", "class": "motion", "risk_level": "medium",
            "params_schema": {"type": "object", "properties": {"vx": {"type": "number"}}},
            "lease_ttl_s": 2.0
        }],
        "heartbeat_ms": 10000
    })
}

struct Booted {
    h: KernelHandle,
    run: tokio::task::JoinHandle<Result<(), oeh_kernel::BootError>>,
    log: Arc<WorldLog>,
    _tmp: tempfile::TempDir,
    sock: PathBuf,
}

impl Booted {
    /// 把脑叫醒。
    fn wake(&self) {
        self.h
            .emit(EventDraft::new(
                EventKind::new("obs/gesture").unwrap(),
                "test",
                json!({"gesture": "wave"}),
            ))
            .unwrap();
    }

    fn events(&self) -> Vec<WorldEvent> {
        self.log.since(0).unwrap_or_default()
    }

    async fn stop(self) {
        self.h.shutdown().unwrap();
        self.run.await.unwrap().unwrap();
    }
}

fn boot(gate: Arc<Gate>) -> Booted {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let script = dir.join("script.yaml");
    std::fs::write(&script, SCRIPT).unwrap();
    let wake = dir.join("wake.yaml");
    std::fs::write(&wake, WAKE_YAML).unwrap();
    let sock = dir.join("oeh.sock");
    let rows = vec![
        EntryRow::new("clock"),
        // 没有 `dir` = 只在内存里，读日志走 `WorldLog::since`。
        EntryRow::new("world-log"),
        EntryRow::new("frame-store"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("brains").with_config(json!({
            "session_dir": dir.join("sessions"),
            // 第一个 turn 一直开着直到测试放行慢工具；内核的超时不该插手。
            "turn_timeout_ms": 0,
        })),
        EntryRow::new("admission").with_config(json!({
            "required_rules": ["rule-whitelist"],
            "estop_flag": dir.join("estop"),
        })),
        EntryRow::new("approval-auto").with_config(json!({"allow": true})),
        EntryRow::new("rule-whitelist").with_config(json!({"allowed": ["walk", "slow"]})),
        EntryRow::new("test-slow"),
        // 每个测试一个 socket，`ws: null`：环境变量是进程全局的，而同一个二进制里的
        // 测试是并发跑的。`run_tag` 同理（iceoryx2 的服务名是机器全局的）。
        EntryRow::new("rpc-server").with_config(json!({
            "uds": sock, "ws": null,
            "run_tag": format!("t{}-caps", std::process::id()),
        })),
        EntryRow::new("brain-native").with_config(json!({
            "brain_id": "fast",
            "wake": wake,
            "persona": "you are a fast brain",
            "model": {"scripted": script},
            "max_steps": 4,
            "oeh_home": dir.join("oeh-home"),
            "protected_paths": [dir.join("guarded")],
            "review": {"every_turns": 0, "on_session_end": false},
        })),
    ];
    let k = Kernel::boot_with(rows, &|id| {
        if id == "test-slow" {
            let gate = gate.clone();
            return Some(Box::new(SlowPlugin { gate }));
        }
        oeh_kernel::find_factory(id).map(|f| (f.build)())
    })
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    assert!(k.report().pending.is_empty(), "{:?}", k.report());
    let log = k
        .registry()
        .get::<WorldLog>(&ServiceName::from("world-log"))
        .unwrap();
    let h = k.handle();
    let run = tokio::spawn(k.run());
    Booted {
        h,
        run,
        log,
        _tmp: tmp,
        sock,
    }
}

/// 等世界日志里出现第一条满足 `pred` 的事件（注册观察者**之前**已经进去的也算）。
async fn wait_for(
    log: &Arc<WorldLog>,
    pred: impl Fn(&WorldEvent) -> bool + Send + Sync + 'static,
) -> WorldEvent {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let pred = Arc::new(pred);
    let p = pred.clone();
    let dispose = log.observe(Arc::new(move |e: &WorldEvent| {
        if p(e) {
            let _ = tx.send(e.clone());
        }
    }));
    let found = match log
        .since(0)
        .unwrap_or_default()
        .into_iter()
        .find(|e| pred(e))
    {
        Some(e) => e,
        None => tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("timed out waiting for a world event")
            .expect("world log observer channel closed"),
    };
    dispose();
    found
}

async fn connect(sock: &Path) -> Client {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match Client::connect_uds(sock).await {
            Ok(c) => return c,
            Err(e) if std::time::Instant::now() >= deadline => {
                panic!("connect {}: {e}", sock.display())
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
}

/// 某个 trace 下每一条 `brain/request` 的 `tool_names`，按 seq。
fn tool_names(evs: &[WorldEvent], trace: &str) -> Vec<Vec<String>> {
    evs.iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace) && e.kind.as_str() == "brain/request")
        .map(|e| {
            e.payload["tool_names"]
                .as_array()
                .expect("tool_names 是数组")
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .collect()
}

/// 第 n 个 turn 的 trace_id（按 `brain/turn_start` 的出现顺序）。
fn trace_of_turn(evs: &[WorldEvent], n: usize) -> String {
    evs.iter()
        .filter(|e| e.kind.as_str() == "brain/turn_start")
        .nth(n)
        .and_then(|e| e.trace_id.clone())
        .unwrap_or_else(|| panic!("没有第 {n} 个 turn"))
}

/// 等第 n 个 turn 开起来，拿它的 trace_id。
///
/// `wake()` 把事件投进内核的 op 通道就返回了，turn 是在内核那一侧开的——"第 n 条
/// `brain/turn_start` 出现了"是这件事唯一的凭据。有界轮询（5 ms 一轮，10 s 上限）是因为
/// "第 n 条"这种带序数的条件没有单条通知可等；判别信号仍然是**事件出现**，轮询间隔只
/// 决定发现得多快。
async fn nth_turn_trace(b: &Booted, n: usize) -> String {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let evs = b.events();
        if let Some(e) = evs
            .iter()
            .filter(|e| e.kind.as_str() == "brain/turn_start")
            .nth(n)
        {
            return e.trace_id.clone().expect("turn_start 带 trace_id");
        }
        assert!(
            std::time::Instant::now() < deadline,
            "第 {n} 个 turn 没开起来"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn next_turn_end(log: &Arc<WorldLog>, trace: String) {
    wait_for(log, move |e| {
        e.kind.as_str() == "brain/turn_end" && e.trace_id.as_deref() == Some(trace.as_str())
    })
    .await;
}

#[tokio::test]
async fn a_remote_capability_reaches_the_next_turns_tool_table_and_leaves_the_running_one_alone() {
    let gate = Arc::new(Gate::default());
    let b = boot(gate.clone());

    // ---- turn 1：远端还没来，表里没有 `robot_walk`。
    b.wake();
    gate.started.notified().await; // 这个 turn 的第一个 step 已经在跑工具了
    let t1 = trace_of_turn(&b.events(), 0);
    let before = tool_names(&b.events(), &t1);
    assert_eq!(before.len(), 1, "慢工具还没放行，只该有第一个 step");
    assert!(
        !before[0].contains(&"robot_walk".to_string()),
        "适配器还没连上来，`walk` 不该在表里：{before:?}"
    );
    assert!(
        before[0].contains(&"robot_slow".to_string()),
        "进程内注册的 `slow` 该在表里：{before:?}"
    );

    // ---- 就在这个 turn 跑到一半的时候，一个跨进程插件把 `walk` 挂上来。
    let c = connect(&b.sock).await;
    let r = c.call("plugin/register", remote_manifest()).await.unwrap();
    assert_eq!(r["state"], "active", "{r}");
    // 注册**真的**落到能力面上了：这条事件由注册表自己在改表的那个临界区之后写。
    wait_for(&b.log, |e| {
        e.kind.as_str() == "capability/registered" && e.payload["name"] == "walk"
    })
    .await;

    // 放行慢工具 → 这个 turn 的第二个 step。它必须还用**旧**表：中途换表会让这个请求
    // 里带着一条 `tool_result`，而它的 `tool_use` 指向一个已经不在 `tools` 里的名字。
    gate.release.notify_one();
    next_turn_end(&b.log, t1.clone()).await;
    let steps = tool_names(&b.events(), &t1);
    assert_eq!(steps.len(), 2, "这个 turn 该有两个 step：{steps:?}");
    assert_eq!(steps[0], steps[1], "一个 turn 之内工具表不许变");

    // ---- turn 2：新表。
    b.wake();
    let t2 = nth_turn_trace(&b, 1).await;
    next_turn_end(&b.log, t2.clone()).await;
    let after = tool_names(&b.events(), &t2);
    assert!(
        after[0].contains(&"robot_walk".to_string()),
        "适配器连上来之后的下一个 turn 该看见 `robot_walk`：{after:?}"
    );
    // 旧表里有的东西一个都没少（重投是从底表重建的，不是差分）。
    for name in &steps[0] {
        assert!(after[0].contains(name), "{name} 在重投之后丢了：{after:?}");
    }

    // ---- 远端断连 → 再下一个 turn 又没有它了。
    c.close().await;
    wait_for(&b.log, |e| {
        e.kind.as_str() == "capability/unregistered" && e.payload["name"] == "walk"
    })
    .await;
    b.wake();
    let t3 = nth_turn_trace(&b, 2).await;
    next_turn_end(&b.log, t3.clone()).await;
    let gone = tool_names(&b.events(), &t3);
    assert!(
        !gone[0].contains(&"robot_walk".to_string()),
        "适配器走了，`robot_walk` 该从表上下去：{gone:?}"
    );
    assert_eq!(gone[0], steps[0], "回到远端来之前的那张表");

    b.stop().await;
}
