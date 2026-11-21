//! `brain-codex` 的端到端：真实内核 + 真实准入链 + 一个真的子进程（`mock-codex`）。
//!
//! 七个场景，每个一个 `#[tokio::test]`：
//!
//! 1. 一次 turn：`say` → `robot_walk`（穿过整条准入链）→ `say` → `turn/completed`；
//!    世界日志里这个 trace 只有内核的记账 + `act/*`，模型说的话只在会话日志里；
//! 2. 审批：codex 发 `item/commandExecution/requestApproval`，装着的
//!    `approval-auto{allow:false}` 说不行 → mock 收到 `decline`，世界日志
//!    `brain/codex_approval{decision:"deny"}`；
//! 3. steer：慢工具把 turn 卡住 → `Brains::steer` → `turn/steer` 真的到了 mock 手里
//!    （下一句话里回显）；
//! 4. 子进程中途死掉 → `brain/error{codex exited}` + turn 以 `error: codex exited` 收尾；
//!    再唤醒一次立刻 `error: codex unavailable`（不自动重启）；
//! 5. interrupt：慢工具跑到一半被 `Brains::interrupt` 打断 → `turn/interrupt` 先于下一个
//!    `turn/start` 到达 mock（脑的出站是**一条**有序队列），turn 以 `interrupted` 收尾，
//!    脑接着跑完下一个 turn；
//! 6. 子进程根本起不来 → `brain/error{brain_id, reason}`，而且什么都没挂上；
//! 7. `turn/completed{status}` 不是 `completed` → `brain/turn_end.reason` 照实说；
//!    点名了别的 turn 的增量被丢掉**并计数**（`brain/codex_dropped`）。
//!
//! 另有一条"根本没装审批服务 → decline"紧跟在场景 2 后面。
//!
//! **没有一处用 sleep 当判别信号**：等世界事件用日志观察者（`wait_for`），等"慢工具真的
//! 开始跑了"用 `Notify`，等子进程死掉用它死之后必然出现的那条世界事件。唯一的墙钟是
//! `wait_for` 外面那层 10 s 兜底超时——它只负责让一个坏掉的测试失败。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::capability::{
    CapabilityClass, CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor, RiskLevel,
};
use oeh_brain::codex::CodexConfig;
use oeh_distill as _;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::session_log::{read_session, SessionEntry};
use oeh_kernel::{Brains, Ctx, Kernel, KernelHandle, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, ServiceName, WorldEvent};
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
/// 它的执行器一直等到测试放行。codex 那边按 `robot_slow` 调用它。
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

const WAKE_YAML: &str = "\
default: accumulate
rules:
  obs/gesture: wake
  brain/steer: steer
  brain/*: ignore
  act/*: ignore
  health/*: ignore
  safety/*: ignore
  plugin/*: ignore
budget: { max_consecutive_wakes: 32, min_interval_ms: 0 }
";

fn walk_spec() -> Value {
    json!({
        "name": "walk",
        "class": "motion",
        "risk_level": "medium",
        "params_schema": {
            "type": "object",
            "properties": {"vx": {"type": "number"}, "duration_s": {"type": "number"}}
        },
        "lease_ttl_s": 2.0
    })
}

struct Fixture {
    tmp: tempfile::TempDir,
    script: PathBuf,
    wake: PathBuf,
}

impl Fixture {
    fn new(script_yaml: &str) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let script = tmp.path().join("script.yaml");
        std::fs::write(&script, script_yaml).unwrap();
        let wake = tmp.path().join("wake.yaml");
        std::fs::write(&wake, WAKE_YAML).unwrap();
        Fixture { tmp, script, wake }
    }

    fn path(&self) -> &Path {
        self.tmp.path()
    }

    /// `mock-codex` 的启动命令。脚本走 `MOCK_CODEX_SCRIPT` 环境变量，而
    /// `std::env::set_var` 是**进程级**的（同一个测试二进制里的用例并行跑），所以用
    /// `sh -c "VAR=… exec …"` 给这一个子进程单独设：`exec` 之后进程就是 mock 自己，
    /// `kill_on_drop` 杀的也是它。
    fn command(&self) -> Vec<String> {
        vec![
            "/bin/sh".into(),
            "-c".into(),
            format!(
                "MOCK_CODEX_SCRIPT='{}' exec '{}'",
                self.script.display(),
                env!("CARGO_BIN_EXE_mock-codex")
            ),
        ]
    }

    /// `approve`：`Some(true/false)` = 装一个固定放行/固定拒绝的 `approval-auto`；
    /// `None` = **根本不装审批服务**（裁定 2 的那条分支）。
    fn rows(&self, with_slow: bool, approve: Option<bool>) -> Vec<EntryRow> {
        let tmp = self.path();
        let mut rows = vec![
            EntryRow::new("clock"),
            EntryRow::new("world-log"),
            EntryRow::new("frame-store"),
            EntryRow::new("state-registry"),
            EntryRow::new("robot-fsm"),
            EntryRow::new("mock-stream")
                .with_config(json!({"stream_id": "lowstate", "hz": 500.0, "payload": "lowstate"})),
            EntryRow::new("distill-fsm").with_config(json!({"stream_id": "lowstate"})),
            EntryRow::new("distill-watchdog")
                .with_config(json!({"streams": {"lowstate": 0.5}, "period_ms": 20})),
            EntryRow::new("brains").with_config(json!({
                "session_dir": tmp.join("sessions"),
                // 场景 3/5 的 turn 一直开着直到测试放行慢工具；内核的超时不该插手。
                "turn_timeout_ms": 0,
            })),
            EntryRow::new("admission").with_config(json!({
                "required_rules": ["rule-whitelist", "rule-fsm", "rule-run-mode", "rule-watchdog"],
                "estop_flag": tmp.join("estop"),
                "lease_check_ms": 50,
            })),
            EntryRow::new("mock-executor").with_config(
                json!({"capabilities": [walk_spec()], "duration_s": 0.02, "heartbeat_ms": 10}),
            ),
            EntryRow::new("rule-whitelist").with_config(json!({"allowed": ["walk", "slow"]})),
            EntryRow::new("rule-fsm"),
            EntryRow::new("rule-run-mode").with_config(json!({"mode": "active"})),
            EntryRow::new("rule-watchdog").with_config(json!({"streams": {"lowstate": 0.5}})),
        ];
        if let Some(allow) = approve {
            rows.push(EntryRow::new("approval-auto").with_config(json!({ "allow": allow })));
        }
        if with_slow {
            rows.push(EntryRow::new("test-slow"));
        }
        // 脑永远是最后一行：能力面是 `apply` 那一刻的快照（D7）。
        rows.push(EntryRow::new("brain-codex").with_config(json!({
            "brain_id": "codex",
            "wake": self.wake,
            "command": self.command(),
            "cwd": tmp,
            "turn_timeout_ms": 10000,
        })));
        rows
    }
}

struct Booted {
    h: KernelHandle,
    run: tokio::task::JoinHandle<Result<(), oeh_kernel::BootError>>,
    log: Arc<WorldLog>,
    brains: Arc<Brains>,
    session: Option<PathBuf>,
}

fn boot(rows: Vec<EntryRow>, gate: Option<Arc<Gate>>) -> Booted {
    let k = Kernel::boot_with(rows, &|id| {
        if id == "test-slow" {
            let gate = gate.clone().expect("test-slow row without a gate");
            return Some(Box::new(SlowPlugin { gate }));
        }
        oeh_kernel::find_factory(id).map(|f| (f.build)())
    })
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    assert!(k.report().pending.is_empty(), "{:?}", k.report());
    let reg = k.registry().clone();
    let log = reg
        .get::<WorldLog>(&ServiceName::from("world-log"))
        .unwrap();
    let brains = reg.get::<Brains>(&ServiceName::from("brains")).unwrap();
    let h = k.handle();
    let run = tokio::spawn(k.run());
    Booted {
        h,
        run,
        log,
        brains,
        session: None,
    }
}

impl Booted {
    /// 数据面就位 **且** 脑挂上了。`brain-codex` 的 attach 发生在一个 tokio 任务里
    /// （子进程要先起来），所以"挂上了"必须等 `brain/attach`，不能假定 boot 返回时就好了。
    async fn ready(&mut self) {
        wait_for(&self.log, |e| {
            e.kind.as_str() == "health/stale"
                && e.payload["stream"] == "lowstate"
                && e.payload["stale"] == false
        })
        .await;
        wait_for(&self.log, |e| {
            e.kind.as_str() == "safety/fsm" && e.payload["to"] == "STANDING"
        })
        .await;
        wait_for(&self.log, |e| {
            e.kind.as_str() == "brain/attach" && e.payload["brain_id"] == "codex"
        })
        .await;
        // 会话日志的路径要在 detach 之前拿到（关机会把这个脑摘掉）。
        self.session = Some(
            self.brains
                .session_log_for_tests("codex")
                .expect("codex is attached")
                .path()
                .to_path_buf(),
        );
    }

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

    fn session(&self) -> PathBuf {
        self.session.clone().expect("ready() 还没跑")
    }

    async fn stop(self) {
        self.h.shutdown().unwrap();
        self.run.await.unwrap().unwrap();
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

/// 某个 trace 下的事件的 kind，按 seq。
fn kinds_of(evs: &[WorldEvent], trace: &str) -> Vec<String> {
    evs.iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace))
        .map(|e| e.kind.as_str().to_string())
        .collect()
}

fn of<'a>(evs: &'a [WorldEvent], trace: &str, kind: &str) -> Vec<&'a WorldEvent> {
    evs.iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace) && e.kind.as_str() == kind)
        .collect()
}

fn one<'a>(evs: &'a [WorldEvent], trace: &str, kind: &str) -> &'a WorldEvent {
    let hits = of(evs, trace, kind);
    assert_eq!(
        hits.len(),
        1,
        "trace {trace} 里的 {kind} 应该恰好一条: {:?}",
        kinds_of(evs, trace)
    );
    hits[0]
}

fn first_trace(evs: &[WorldEvent]) -> String {
    evs.iter()
        .find(|e| e.kind.as_str() == "brain/turn_start")
        .expect("没有 brain/turn_start")
        .trace_id
        .clone()
        .expect("brain/* 必须带 trace_id")
}

/// 会话日志里模型说过的每一句话。
fn model_outputs(session: &Path) -> Vec<String> {
    read_session(session)
        .unwrap()
        .into_iter()
        .filter_map(|e| match e {
            SessionEntry::ModelOutput { text, .. } => Some(text),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------- 场景 1

/// 一次完整的 turn：两句话夹着一次 `robot_walk`。世界日志里这个 trace 是内核的记账 +
/// 准入链，脑自己一条事件都不发（说的话只进会话日志）。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_turn_says_calls_a_robot_tool_and_ends_on_turn_completed() {
    let fx = Fixture::new(
        "- - say: \"hi\"\n  \
             - tool: { name: robot_walk, arguments: { vx: 0.1, duration_s: 0.02 } }\n  \
             - say: \"done\"\n",
    );
    let mut b = boot(fx.rows(false, Some(true)), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let session = b.session();
    let evs = b.events();
    b.stop().await;

    let trace = end.trace_id.clone().unwrap();
    assert_eq!(trace, first_trace(&evs));
    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "act/proposed",
            "act/admitted",
            "act/started",
            "act/result",
            "brain/turn_end",
        ],
    );
    assert_eq!(one(&evs, &trace, "act/result").payload["ok"], true);
    assert_eq!(end.payload["reason"], "end_turn");

    // 两句话进了会话日志；世界日志里一个字都没有（上面那条序列断言已经说明脑一条事件
    // 都没发，这里再确认全文确实落在会话日志里）。
    assert_eq!(model_outputs(&session), vec!["hi", "done"]);
}

// ---------------------------------------------------------------- 场景 2

/// codex 要执行一条命令 → 转给内核装着的审批服务 → 它说不行 → 回 `decline`。
///
/// 事件里两个字段各说一件事：`verdict` 是审批方的**判定**，`decision` 是我们在线上给
/// codex 的那个**答复**；外加被审批的那条命令（截断过的）与 cwd。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_command_approval_request_goes_to_the_installed_approver_and_can_be_denied() {
    let fx = Fixture::new(
        "- - approval: { command: \"ls -l /tmp\", cwd: \"/work\" }\n  \
             - say: \"after\"\n",
    );
    let mut b = boot(fx.rows(false, Some(false)), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let session = b.session();
    let evs = b.events();
    b.stop().await;

    let trace = end.trace_id.clone().unwrap();
    let ev = one(&evs, &trace, "brain/codex_approval");
    assert_eq!(ev.payload["decision"], "decline", "线上给出的答复");
    assert_eq!(ev.payload["verdict"], "deny", "审批方的判定");
    assert_eq!(ev.payload["by"], "approval-auto");
    assert_eq!(ev.payload["command"], "ls -l /tmp", "拒的是什么要看得见");
    assert_eq!(ev.payload["cwd"], "/work");
    assert_eq!(ev.payload["brain_id"], "codex");
    // mock 那头收到的是 `decline`：它把收到的答复回显进下一句话。
    let said = model_outputs(&session).join("\n");
    assert!(said.contains("approval:decline"), "{said:?}");
}

/// **根本没装审批服务**：fail closed——回 `decline`，事件里没有 `verdict`（没人可问就
/// 没有判定可言），`reason` 说清楚为什么。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_no_approver_installed_the_request_is_declined_and_says_so() {
    let fx = Fixture::new(
        "- - approval: { command: \"rm -rf /\" }\n  \
             - say: \"after\"\n",
    );
    let mut b = boot(fx.rows(false, None), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let session = b.session();
    let evs = b.events();
    b.stop().await;

    let trace = end.trace_id.clone().unwrap();
    let ev = one(&evs, &trace, "brain/codex_approval");
    assert_eq!(ev.payload["decision"], "decline");
    assert_eq!(ev.payload["reason"], "no approver");
    assert!(ev.payload.get("verdict").is_none(), "{:?}", ev.payload);
    assert_eq!(ev.payload["command"], "rm -rf /");
    let said = model_outputs(&session).join("\n");
    assert!(said.contains("approval:decline"), "{said:?}");
}

// ---------------------------------------------------------------- 场景 3

/// 慢工具把 turn 卡住 → `Brains::steer` → `turn/steer` 真的到了 mock 手里。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_steer_during_a_running_turn_reaches_codex() {
    let fx = Fixture::new(
        "- - tool: { name: robot_slow, arguments: {} }\n  \
             - say: \"echo\"\n",
    );
    let gate = Arc::new(Gate::default());
    let mut b = boot(fx.rows(true, Some(true)), Some(gate.clone()));
    b.ready().await;
    b.wake();
    // 慢工具真的开始跑了 = turn 一定还开着，而且 `turn/start` 早就回来了。
    gate.started.notified().await;
    b.brains.steer("codex", "please turn left").unwrap();
    // `brain/codex_steer` 是 `turn/steer` **答复回来之后**才记的：等到它，就知道 mock
    // 已经把这句话收进去了，这时候再放行慢工具，回显一定在下一句话里。
    let ack = wait_for(&b.log, |e| e.kind.as_str() == "brain/codex_steer").await;
    assert_eq!(ack.payload["ok"], true);
    assert_eq!(ack.payload["brain_id"], "codex");
    gate.release.notify_one();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let session = b.session();
    b.stop().await;

    assert_eq!(end.payload["reason"], "end_turn");
    let said = model_outputs(&session).join("\n");
    assert!(said.contains("please turn left"), "{said:?}");
}

// ---------------------------------------------------------------- 场景 4

/// 子进程中途死掉：当前 turn 拿到终态，世界日志有 `brain/error{codex exited}`，
/// 之后每次唤醒立刻 `error: codex unavailable`——不自动重启。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subprocess_that_dies_mid_turn_ends_the_turn_and_stays_dead() {
    let fx = Fixture::new(
        "- - say: \"starting\"\n  \
             - exit: true\n",
    );
    let mut b = boot(fx.rows(false, Some(true)), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let trace = end.trace_id.clone().unwrap();
    assert_eq!(end.payload["reason"], "error: codex exited");
    {
        let evs = b.events();
        let err = one(&evs, &trace, "brain/error");
        assert_eq!(err.payload["reason"], "codex exited");
        assert_eq!(err.payload["brain_id"], "codex");
    }

    // 再叫一次：立刻收尾，不重启。
    b.wake();
    let end2 = wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == "error: codex unavailable"
    })
    .await;
    let evs = b.events();
    b.stop().await;

    let trace2 = end2.trace_id.clone().unwrap();
    assert_ne!(trace, trace2);
    assert_eq!(
        kinds_of(&evs, &trace2),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/turn_end",
        ],
    );
}

// ---------------------------------------------------------------- 场景 5

/// 打断：`turn/interrupt` **先于**下一个 `turn/start` 到达 mock（出站是一条有序队列），
/// 这个 turn 以 `interrupted` 收尾（脑自己不 `done`），下一个 turn 照常跑完。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_interrupt_reaches_codex_before_the_next_turn_starts() {
    let fx = Fixture::new(
        "- - tool: { name: robot_slow, arguments: {} }\n  \
             - say: \"one\"\n\
         - - say: \"two\"\n",
    );
    let gate = Arc::new(Gate::default());
    let mut b = boot(fx.rows(true, Some(true)), Some(gate.clone()));
    b.ready().await;
    b.wake();
    gate.started.notified().await;
    b.brains.interrupt("codex").unwrap();
    let first = wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == "interrupted"
    })
    .await;
    let second = wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == "end_turn"
    })
    .await;
    // 卡着的慢工具放行，好让关机不用等租约。
    gate.release.notify_one();
    let session = b.session();
    let evs = b.events();
    b.stop().await;

    assert_ne!(first.trace_id, second.trace_id);
    // 第一个 turn 被打断之后，脑没有再往这个 trace 里写任何东西（尤其没有第二条
    // turn_end）。
    let t1 = first.trace_id.clone().unwrap();
    assert_eq!(of(&evs, &t1, "brain/turn_end").len(), 1);
    // mock 收到了 `turn/interrupt`：它把收到的控制消息回显进下一句话。
    let said = model_outputs(&session).join("\n");
    assert!(said.contains("two"), "{said:?}");
    assert!(said.contains("interrupt"), "{said:?}");
    assert!(
        !said.contains("one"),
        "被打断的那个 turn 不该再说话: {said:?}"
    );
}

// ---------------------------------------------------------------- 场景 6

/// 子进程根本起不来：世界日志里有 `brain/error{brain_id, reason}`，而且**什么都没挂上**
/// ——起进程发生在 `apply` 之后的一个任务里，失败不能悄无声息。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subprocess_that_cannot_be_spawned_is_reported_and_nothing_attaches() {
    let fx = Fixture::new("- - say: \"never\"\n");
    let mut rows = fx.rows(false, Some(true));
    rows.pop();
    rows.push(EntryRow::new("brain-codex").with_config(json!({
        "brain_id": "codex",
        "wake": fx.wake,
        "command": ["/nonexistent/oeh-codex-does-not-exist"],
    })));
    let b = boot(rows, None);
    let err = wait_for(&b.log, |e| e.kind.as_str() == "brain/error").await;
    assert_eq!(err.payload["brain_id"], "codex");
    assert!(
        err.payload["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("spawn"),
        "{:?}",
        err.payload
    );
    // 这条事件不属于任何一个 turn，所以没有 trace。
    assert!(err.trace_id.is_none());
    assert!(b.brains.status("codex").is_none(), "不该挂上");
    b.stop().await;
}

// ---------------------------------------------------------------- 场景 7

/// `turn/completed{status}` 不是 `completed` 时，`brain/turn_end.reason` 必须说出真相
/// ——一次失败的 turn 和一次正常收尾在日志里长得一样，正是操作员最不需要的那种体贴。
///
/// 顺带把"点名了别的 turn 的增量"一并验掉：它不能进会话日志，也不能被**静默**丢掉。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_turn_that_ends_with_a_failure_status_says_so_and_stray_deltas_are_counted() {
    let fx = Fixture::new(
        "- - stray: \"ghost\"\n  \
             - say: \"real\"\n  \
             - status: \"rate_limited\"\n",
    );
    let mut b = boot(fx.rows(false, Some(true)), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let session = b.session();
    let evs = b.events();
    b.stop().await;

    assert_eq!(end.payload["reason"], "error: codex turn rate_limited");
    let trace = end.trace_id.clone().unwrap();
    let dropped = one(&evs, &trace, "brain/codex_dropped");
    assert_eq!(dropped.payload["deltas"], 1);
    assert_eq!(dropped.payload["brain_id"], "codex");
    // 别的 turn 的那句话没有混进来；这个 turn 自己的话在。
    assert_eq!(model_outputs(&session), vec!["real"]);
}

// ---------------------------------------------------------------- 配置

/// `CodexConfig::default()` 就是插件文档里写的那一份默认值：两处各写一遍的数字，
/// 得有一个地方把它们钉在一起。
#[test]
fn the_documented_defaults_are_the_config_s_defaults() {
    let c = CodexConfig::default();
    assert_eq!(c.command, vec!["codex", "app-server"]);
    assert_eq!(c.turn_timeout_ms, 120_000);
    assert!(c.cwd.is_none());
}
