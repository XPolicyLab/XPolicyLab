//! 原生 turn 循环的端到端：真实内核 + 真实准入链 + `brain-native{scripted}`。
//!
//! 十个场景，每个一个 `#[tokio::test]`，全部用 trace_id 过滤之后断言精确的事件序列：
//!
//! 1. 工具调用走完整条准入链（`brain/tool_call` → `act/proposed…act/result` →
//!    `brain/tool_result`），第二个 step 结束 turn；
//! 2. `max_steps: 1` → `done("max_steps")`，只跑一批工具；
//! 3. 慢工具跑到一半被 `Brains::interrupt` 打断 → 任务被 abort，`brain/turn_end`
//!    `{reason:"interrupted"}`，这个 turn 没有第二次 `brain/request`；
//! 4. 慢工具期间 `Brains::steer` → 下一个 step 的 user message 里带着那句话；
//! 5. 模型还 `MissingKey` → `brain/request-error` + `done("error: missing api key …")`，
//!    turn 不挂；
//! 6. 慢工具期间撤销脑这一行（fiber dispose）→ 任务当场死掉，那个还开着的 turn 拿到
//!    唯一一条终态 `brain/turn_end{reason:"detached"}`，之后这个 trace 再没有一条
//!    `brain/*`（fix round 1 · M2；final · M1）；
//! 7. 流吐了两段增量和一次工具调用之后断掉 → 那半句话进会话日志，`brain/request-error`
//!    带 `partial_chars` / `partial_tool_calls`（fix round 1 · M3，round 2 · N2）；
//! 8. 模型在流中途 panic → turn 仍然有终态：`done("error: panicked")` +
//!    `brain/turn_end`，脑回到 Idle（round 2 · I4）。
//! 9. 工具永远不回来 → step 的截止时刻也管工具批：`brain/tool_result{error:"timeout"}`
//!    + `done("error: step timeout")`，turn 不挂着（final · I4）。
//! 10. 慢工具期间 `Brains::inject` → 下一个 step 的 user message 里是 `[inject] <那句话>`
//!     （`Lane::Inject`，不是 steer 那条 lane）。
//!
//! **没有一处用 sleep 当判别信号。** 等世界事件用日志观察者（`wait_for`），等"慢工具
//! 真的开始跑了"用 `Notify`（测试自己注册的那个能力的执行器按下它），等"steer 真的路由
//! 到了"用 `Brains::settle`。唯一的墙钟是 `wait_for` 外面那层 10 s 兜底超时——它只负责
//! 让一个坏掉的测试失败，而不是让一个好的测试通过。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::capability::{
    CapabilityClass, CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor, RiskLevel,
};
use oeh_brain::builtin::scripted_model::ScriptedModel;
use oeh_brain::model::Message;
use oeh_distill as _;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::session_log::{read_session, SessionEntry};
use oeh_kernel::{
    BrainStatus, Brains, Ctx, FiberId, Kernel, KernelHandle, Plugin, PluginError, WorldLog,
};
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
/// 它的执行器一直等到测试放行。`brain-native` 会把它投影成 `robot_slow`。
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

/// 只让 `obs/gesture` 唤醒：别的事件一律 accumulate 或 ignore，于是"这个脑什么时候醒、
/// 醒来看见什么"完全由测试决定，不由 500 Hz 的数据面决定。
///
/// `brain/*: ignore` 这一行是场景 4 与场景 10 的一半判据，**不是**需要绕开的东西：定向
/// 事件（`Brains::steer` / `Brains::inject`）既绕过 policy 到达这个脑，也必须绕过观测
/// 那一遍的 policy 过滤，否则脑收到的是一份 `wake_reason` 说着 `brain/steer`、正文里
/// 却什么都没有的观测。这里刻意**不**写 `brain/steer: steer` / `brain/inject: steer`：
/// 写了就等于把这条豁免测没了（shipped 的 `wake/*.yaml` 也不写）。
const WAKE_YAML: &str = "\
default: accumulate
rules:
  obs/gesture: wake
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
    /// 剧本与唤醒策略都写进 tempdir，用**绝对路径**喂给插件：`brain-native` 和
    /// `mock-brain` 一样按进程 cwd 解析 `wake`，而测试的 cwd 是 crate 目录。
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

    fn rows(&self, max_steps: u32, with_slow: bool) -> Vec<EntryRow> {
        self.rows_with(max_steps, with_slow, 5000)
    }

    /// `step_timeout_ms` 可调的那一版。场景 9 要一个**短**的 step 预算（工具批也在它
    /// 里面），别的场景要一个宽到不会插手的。
    fn rows_with(&self, max_steps: u32, with_slow: bool, step_timeout_ms: u64) -> Vec<EntryRow> {
        let tmp = self.path();
        let mut rows = vec![
            EntryRow::new("clock"),
            // 没有 `dir` = 只在内存里，读日志走 `WorldLog::since`。
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
                // 场景 3/4 的 turn 一直开着直到测试放行慢工具；内核的超时不该插手。
                "turn_timeout_ms": 0,
            })),
            EntryRow::new("admission").with_config(json!({
                "required_rules": ["rule-whitelist", "rule-fsm", "rule-run-mode", "rule-watchdog"],
                // 开发机上真有一个 /tmp/oeh_estop 的话，默认值会让这些用例莫名其妙被拒。
                "estop_flag": tmp.join("estop"),
                "lease_check_ms": 50,
            })),
            EntryRow::new("approval-auto").with_config(json!({"allow": true})),
            EntryRow::new("mock-executor").with_config(
                json!({"capabilities": [walk_spec()], "duration_s": 0.02, "heartbeat_ms": 10}),
            ),
            EntryRow::new("rule-whitelist").with_config(json!({"allowed": ["walk", "slow"]})),
            EntryRow::new("rule-fsm"),
            EntryRow::new("rule-run-mode").with_config(json!({"mode": "active"})),
            EntryRow::new("rule-watchdog").with_config(json!({"streams": {"lowstate": 0.5}})),
        ];
        if with_slow {
            rows.push(EntryRow::new("test-slow"));
        }
        // 脑永远是最后一行：工具表是能力面在 `apply` 那一刻的快照。
        rows.push(EntryRow::new("brain-native").with_config(json!({
            "brain_id": "fast",
            "wake": self.wake,
            "persona": "you are a fast brain",
            "model": {"scripted": self.script},
            "max_steps": max_steps,
            "step_timeout_ms": step_timeout_ms,
            // RSI 的家进 tempdir（默认是进程 cwd 下的 `./run/oeh-home`，而测试的 cwd 是
            // crate 目录），维护相位整个关掉：这个文件问的是 turn 循环，一次会话末
            // review 只会给每个场景多出一次模型调用和一条事件。RSI 自己的端到端在
            // `tests/rsi_review.rs`。
            "oeh_home": tmp.join("oeh-home"),
            "protected_paths": [tmp.join("guarded")],
            "review": {"every_turns": 0, "on_session_end": false},
        })));
        rows
    }
}

struct Booted {
    h: KernelHandle,
    run: tokio::task::JoinHandle<Result<(), oeh_kernel::BootError>>,
    log: Arc<WorldLog>,
    brains: Arc<Brains>,
    model: Arc<ScriptedModel>,
    session: PathBuf,
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
    let model = reg
        .get::<ScriptedModel>(&ServiceName::from("brain-native.fast.model"))
        .unwrap();
    // 会话日志的路径要在 detach 之前拿到（关机会把这个脑摘掉）。
    let session = brains
        .session_log_for_tests("fast")
        .expect("fast is attached")
        .path()
        .to_path_buf();
    let h = k.handle();
    let run = tokio::spawn(k.run());
    Booted {
        h,
        run,
        log,
        brains,
        model,
        session,
    }
}

impl Booted {
    /// 数据面就位：lowstate 有过帧（`rule-watchdog` 读得到）且 FSM 到了 STANDING
    /// （`rule-fsm` 放行 motion）。两件事都有对应的世界事件，不用睡。
    async fn ready(&self) {
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
    }

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

    /// 世界日志此刻的全部事件。断言用的快照**由测试自己决定时点**——关机本身会再产生
    /// 一批事件（`plugin/state`、撤销时给在途提案补的 `act/cancelled`），混进来会让
    /// "这个 turn 的事件序列"这类断言变成时序竞赛。
    fn events(&self) -> Vec<WorldEvent> {
        self.log.since(0).unwrap_or_default()
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
    // 观察者只看得见注册之后的 append，所以补扫一遍已有的。两边都命中时取先找到的那条，
    // 语义一样（都是"这件事发生过"）。
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

/// 在 `window` 内反复确认某件事**没有**发生：每 20 ms 查一次，一出现就立刻失败。
///
/// 否定断言没有单一同步点可用——"再也不会有"只能看一段时间。判别信号仍然是**事件出现**
/// （每 20 ms 查一次，出现即失败），窗口只决定看多久；修好之前，那条 `brain/tool_result`
/// 在慢工具放行后几微秒内就会出现，这个窗口给了它四个数量级的余量。
async fn stays_absent(
    log: &Arc<WorldLog>,
    what: &str,
    pred: impl Fn(&WorldEvent) -> bool,
    window: Duration,
) {
    let deadline = tokio::time::Instant::now() + window;
    loop {
        if let Some(e) = log.since(0).unwrap_or_default().into_iter().find(&pred) {
            panic!("{what} 不该出现，却有一条 {} (seq {})", e.kind, e.seq);
        }
        if tokio::time::Instant::now() >= deadline {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// `plugin/state` 里那一行的 `fiber_id`。内核在 `run()` 一开始就广播一遍状态，所以这是
/// 测试拿到 `FiberId` 的公开途径（`Kernel::states()` 不给 id）。
async fn fiber_id(log: &Arc<WorldLog>, plugin: &str) -> FiberId {
    let want = plugin.to_string();
    let ev = wait_for(log, move |e| {
        e.kind.as_str() == "plugin/state" && e.payload["plugin"] == want.as_str()
    })
    .await;
    FiberId(ev.payload["fiber_id"].as_u64().expect("fiber_id") as u32)
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

/// 第一个 turn 的 trace_id。
fn first_trace(evs: &[WorldEvent]) -> String {
    evs.iter()
        .find(|e| e.kind.as_str() == "brain/turn_start")
        .unwrap_or_else(|| panic!("没有 brain/turn_start: {:?}", kinds_all(evs)))
        .trace_id
        .clone()
        .expect("brain/* 必须带 trace_id")
}

fn kinds_all(evs: &[WorldEvent]) -> Vec<&str> {
    evs.iter().map(|e| e.kind.as_str()).collect()
}

/// 一次请求里所有 user message 的正文。
fn user_texts(msgs: &[Message]) -> Vec<String> {
    msgs.iter()
        .filter_map(|m| match m {
            Message::User { content } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------- 场景 1

/// 第 1 步调 `robot_walk`，第 2 步收尾。一个 turn 里两次 `brain/request`，中间那次工具
/// 调用完整地穿过准入链，结果回到模型手里，turn 以 `end_turn` 收尾，脑回到 Idle。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tool_call_runs_through_admission_and_the_turn_ends_on_the_next_step() {
    let fx = Fixture::new(
        "- text: \"路看起来是空的，我走两步。\"\n  \
           tool_calls:\n    - { name: robot_walk, args: { vx: 0.1, duration_s: 0.02 } }\n\
         - text: \"走完了。\"\n",
    );
    let b = boot(fx.rows(8, false), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let session = b.session.clone();
    let model = b.model.clone();
    let evs = b.events();
    b.stop().await;

    let trace = end.trace_id.clone().unwrap();
    assert_eq!(trace, first_trace(&evs));
    // 这个 turn 的全部事件，按序。`brain/wake`…`brain/turn_end` 是内核的记账，中间那
    // 一段是脑和准入链。
    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/model_output",
            "brain/tool_call",
            "act/proposed",
            "act/admitted",
            "act/started",
            "act/result",
            "brain/tool_result",
            "brain/request",
            "brain/model_output",
            "brain/turn_end",
        ],
    );

    let reqs = of(&evs, &trace, "brain/request");
    assert_eq!(reqs[0].payload["step"], 0);
    assert_eq!(reqs[1].payload["step"], 1);
    assert_eq!(reqs[0].payload["provider"], "scripted");
    assert_eq!(reqs[0].payload["model"], "scripted");
    assert_eq!(reqs[0].payload["api_mode"], "scripted");
    assert_eq!(reqs[0].payload["brain_id"], "fast");
    let names: Vec<&str> = reqs[0].payload["tool_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(names.contains(&"robot_walk"), "{names:?}");
    // prompt 全文不进世界日志。
    assert!(reqs[0].payload.get("system").is_none());
    assert!(reqs[0].payload.get("messages").is_none());

    let call = one(&evs, &trace, "brain/tool_call");
    assert_eq!(call.payload["name"], "robot_walk");
    assert_eq!(
        call.payload["args_bytes"],
        r#"{"duration_s":0.02,"vx":0.1}"#.len(),
        "args 的大小进日志，全文不进"
    );
    let result = one(&evs, &trace, "brain/tool_result");
    assert_eq!(result.payload["name"], "robot_walk");
    assert_eq!(result.payload["ok"], true);
    assert_eq!(one(&evs, &trace, "act/result").payload["ok"], true);

    let outs = of(&evs, &trace, "brain/model_output");
    assert_eq!(outs[0].payload["tool_calls"], 1);
    assert_eq!(outs[0].payload["chars"], 13);
    assert_eq!(outs[1].payload["tool_calls"], 0);
    // 输出全文只在会话日志里。
    assert!(outs[0].payload.get("text").is_none());
    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "end_turn"
    );

    // 会话日志：两条 ModelOutput（带全文）夹着一对 ToolCall/ToolResult，turn 号都对。
    let entries = read_session(&session).unwrap();
    let brain_side: Vec<&SessionEntry> = entries
        .iter()
        .filter(|e| {
            matches!(
                e,
                SessionEntry::ModelOutput { .. }
                    | SessionEntry::ToolCall { .. }
                    | SessionEntry::ToolResult { .. }
            )
        })
        .collect();
    match brain_side.as_slice() {
        [SessionEntry::ModelOutput {
            turn: t0,
            step: 0,
            text,
        }, SessionEntry::ToolCall {
            turn: t1,
            step: 0,
            call_id,
            name,
            args,
        }, SessionEntry::ToolResult {
            turn: t2,
            step: 0,
            call_id: rid,
            ok: true,
            ..
        }, SessionEntry::ModelOutput {
            turn: t3,
            step: 1,
            text: text1,
        }] => {
            assert_eq!((t0, t1, t2, t3), (&1, &1, &1, &1));
            assert_eq!(text, "路看起来是空的，我走两步。");
            assert_eq!(text1, "走完了。");
            assert_eq!(name, "robot_walk");
            assert_eq!(args, &json!({"vx": 0.1, "duration_s": 0.02}));
            assert_eq!(call_id, rid, "ToolCall 与 ToolResult 靠 call_id 配对");
        }
        other => panic!("会话日志里的脑侧条目不对: {other:?}"),
    }

    // 第一个 step 的 user message 里有观测；system 前缀里有工具索引。
    let reqs = model.requests();
    assert_eq!(reqs.len(), 2);
    assert!(reqs[0]
        .system
        .starts_with("you are a fast brain\n\n# Tools\n"));
    assert!(reqs[0]
        .system
        .contains("robot_walk: motion/medium capability walk\n"));
    assert!(user_texts(&reqs[0].messages)[0].contains("wake: obs/gesture"));
    // 第二个 step 带着 assistant + tool 结果。
    assert!(matches!(reqs[1].messages[1], Message::Assistant { .. }));
    assert!(matches!(reqs[1].messages[2], Message::Tool { .. }));
}

// ---------------------------------------------------------------- 场景 2

/// `max_steps: 1`：剧本每一步都要调工具，但这个 turn 只跑一个 step——一次模型调用、
/// 一批工具，然后 `done("max_steps")`。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_steps_stops_the_turn_after_one_model_call_and_one_tool_batch() {
    let fx = Fixture::new(
        "- text: one\n  tool_calls: [{ name: robot_walk, args: { vx: 0.1, duration_s: 0.02 } }]\n\
         - text: two\n  tool_calls: [{ name: robot_walk, args: { vx: 0.1, duration_s: 0.02 } }]\n",
    );
    let b = boot(fx.rows(1, false), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let model = b.model.clone();
    let evs = b.events();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();

    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/model_output",
            "brain/tool_call",
            "act/proposed",
            "act/admitted",
            "act/started",
            "act/result",
            "brain/tool_result",
            "brain/turn_end",
        ],
    );
    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "max_steps"
    );
    // 剧本第二步没被消费：`max_steps` 是在 step 之间停，不是在流里停。
    assert_eq!(model.remaining(), 1);
}

// ---------------------------------------------------------------- 场景 3

/// 打断：第 1 步的工具停在一个只有测试能放行的执行器上，这时 `Brains::interrupt` →
/// 任务被 abort（没有第二次 `brain/request`、没有 `brain/tool_result`），turn 以
/// `interrupted` 收尾。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_interrupt_aborts_the_turn_task_mid_tool_call() {
    let fx = Fixture::new(
        "- text: calling the slow tool\n  tool_calls: [{ name: robot_slow, args: {} }]\n\
         - text: this line belongs to the turn after the interrupt\n",
    );
    let gate = Arc::new(Gate::default());
    let b = boot(fx.rows(8, true), Some(gate.clone()));
    b.ready().await;
    b.wake();
    // 同步点：慢工具真的跑起来了。
    gate.started.notified().await;
    b.brains.interrupt("fast").unwrap();
    let end = wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == "interrupted"
    })
    .await;
    // 快照要在放行之前取：放行之后那次执行会补一条 `act/result`（同一个 trace），
    // 它和"被打断的 turn 长什么样"这条断言抢跑。
    let evs = b.events();
    // 被打断之后内核立刻开了下一个 turn；放行慢工具只是为了别让那次执行留在半路。
    gate.release.notify_one();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();
    assert_eq!(trace, first_trace(&evs));

    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/model_output",
            "brain/tool_call",
            "act/proposed",
            "act/admitted",
            "act/started",
            "brain/turn_end",
        ],
    );
    assert_eq!(
        one(&evs, &trace, "brain/tool_call").payload["name"],
        "robot_slow"
    );
    assert!(
        of(&evs, &trace, "brain/tool_result").is_empty(),
        "被打断的那一批工具没有结果"
    );
    assert_eq!(of(&evs, &trace, "brain/request").len(), 1);
}

// ---------------------------------------------------------------- 场景 4

/// 运行中被 steer：慢工具还没回来的时候 `Brains::steer(fast, "stop talking")` →
/// 下一个 step 的 user message 里带着那句话（走的是 `Inbox::Steer` 那条 lane）。
///
/// 这份唤醒策略把 `brain/*` 判成 `ignore`（shipped 的三份也都是），所以"那句话还在不在"
/// 同时是内核那条定向豁免的判据：`build_observation` 只对**点名给这个脑**的定向事件破例，
/// 破例没了这里就只剩一份没有正文的观测。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_steer_while_running_lands_in_the_next_step_s_user_message() {
    let fx = Fixture::new(
        "- text: calling the slow tool\n  tool_calls: [{ name: robot_slow, args: {} }]\n\
         - text: ok, stopping\n",
    );
    let gate = Arc::new(Gate::default());
    let b = boot(fx.rows(8, true), Some(gate.clone()));
    b.ready().await;
    b.wake();
    gate.started.notified().await;
    b.brains.steer("fast", "stop talking").unwrap();
    // 同步点：这条 steer 已经路由完（`sink.steer` 调过了，条目已经在收件箱里）。
    b.brains.settle().await;
    gate.release.notify_one();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let model = b.model.clone();
    let evs = b.events();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();

    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "end_turn"
    );
    assert_eq!(of(&evs, &trace, "brain/request").len(), 2);

    let reqs = model.requests();
    assert_eq!(reqs.len(), 2);
    let first = user_texts(&reqs[0].messages).join("\n");
    assert!(
        !first.contains("stop talking"),
        "steer 不该回填到第 1 步: {first}"
    );
    let second = user_texts(&reqs[1].messages).join("\n");
    assert!(
        // 缩进：steer 的正文是一整份观测渲染，后续行必须缩进在 `[steer]` 底下，不然
        // 它自己的 `# Observation` 标题会平铺在 `# Inbox` 里（fix round 1 · L5）。
        second.contains("# Inbox\n[steer] # Observation\n    ") && second.contains("stop talking"),
        "第 2 步的 user message 里没有那句 steer: {second}"
    );
}

// ---------------------------------------------------------------- 场景 5

/// 没有 API key：模型第一步就还 `MissingKey` → `brain/request-error` + `done("error: …")`。
/// turn 不挂，脑回到 Idle。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_api_key_ends_the_turn_with_an_error_instead_of_hanging_it() {
    let fx = Fixture::new("- missing_key: OPENAI_API_KEY\n");
    let b = boot(fx.rows(8, false), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let evs = b.events();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();

    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/request-error",
            "brain/turn_end",
        ],
    );
    // 这条是**重试瀑布**记的（Task 3）：调用根本没成功，所以带的是"找谁、第几次"。
    // 没有 key 不重试也没有 fallback（链上只有剧本模型），所以恰好一条。
    let err = one(&evs, &trace, "brain/request-error");
    assert_eq!(err.payload["reason"], "missing api key OPENAI_API_KEY");
    assert_eq!(err.payload["provider"], "scripted");
    assert_eq!(err.payload["model"], "scripted");
    assert_eq!(err.payload["attempt"], 1);
    assert_eq!(err.payload["brain_id"], "fast");
    // 第几步也在（fix round 1 · M1）：多 step 的 turn 出错时，靠事件的相对位置去数太
    // 容易数错，而 `brain/request` 本来就带着它。
    assert_eq!(err.payload["step"], 0);
    assert!(of(&evs, &trace, "brain/retry").is_empty(), "缺 key 不重试");
    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "error: missing api key OPENAI_API_KEY"
    );
    // 失败的那一步没有 ModelOutput：模型什么都没说。
    assert!(of(&evs, &trace, "brain/model_output").is_empty());
}

// ---------------------------------------------------------------- 场景 6

/// 撤销脑那一行的 fiber：还在跑的 turn 任务当场死掉。
///
/// 没有这条保证的话，任务会一直活到下一个 await 点之后——默认 `step_timeout_ms` 60 s，
/// 也就是 `brain/detach` 写进日志之后最多一分钟还会冒出这个死 turn 的 `brain/tool_result`
/// 和下一个 step 的整批工具执行。判别点就是那条 `brain/tool_result`：慢工具被放行之后
/// 它本该出现，修好之后一条都不该有。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disposing_the_brain_fiber_stops_the_turn_task_dead() {
    let fx = Fixture::new(
        "- text: calling the slow tool\n  tool_calls: [{ name: robot_slow, args: {} }]\n\
         - text: this line must never be requested\n",
    );
    let gate = Arc::new(Gate::default());
    let b = boot(fx.rows(8, true), Some(gate.clone()));
    b.ready().await;
    b.wake();
    gate.started.notified().await;
    let trace = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_start")
        .await
        .trace_id
        .expect("brain/* 必须带 trace_id");

    // 撤销 `brain-native` 那一行。ack 回来 = 这一行的 Effects 已经全跑完（先 abort，
    // 后 detach），不是"大概已经跑完了"。
    let fid = fiber_id(&b.log, "brain-native").await;
    b.h.dispose_fiber(fid).unwrap().await.unwrap();
    let detach = wait_for(&b.log, |e| e.kind.as_str() == "brain/detach").await;
    // M1：detach 也是一条 turn 的终结路径，它也要给终态。否则世界日志的回放会看到一条
    // 永远没有下文的 `brain/turn_start`。
    let end = one(&b.events(), &trace, "brain/turn_end").clone();
    assert_eq!(end.payload["reason"], "detached");
    assert_eq!(end.payload["brain_id"], "fast");
    assert!(end.seq < detach.seq, "终态排在 brain/detach 前面");

    // 放行慢工具：不放行的话"任务死了"和"任务只是还卡在同一个 await 上"分不开。这条
    // 执行属于 admission，不属于脑，所以它照常跑完。
    gate.release.notify_one();
    // 因果上的前驱：`act/result` 之后，那个已经没人认领的 turn 任务本该被唤醒并写下
    // `brain/tool_result`——修好之前就是这么发生的。
    let result = wait_for(&b.log, |e| e.kind.as_str() == "act/result").await;
    assert!(result.seq > detach.seq, "act/result 应该晚于 brain/detach");
    b.brains.settle().await;
    stays_absent(
        &b.log,
        "detach 之后这个 turn 的 brain/tool_result",
        |e| e.kind.as_str() == "brain/tool_result",
        Duration::from_millis(400),
    )
    .await;

    let model = b.model.clone();
    let evs = b.events();
    b.stop().await;

    // 这个 turn 的 `brain/*` 到 `brain/tool_call` 为止，一条不多。
    let mine: Vec<String> = evs
        .iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace.as_str()) && e.kind.prefix() == "brain")
        .map(|e| e.kind.as_str().to_string())
        .collect();
    assert_eq!(
        mine,
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/model_output",
            "brain/tool_call",
            // detach 给的终态。这个 turn **恰好**一条 `brain/turn_end`：脑自己那次
            // `done` 在 detach 之后是空操作（`finish_turn` 找不到这个脑），所以不会重。
            "brain/turn_end",
        ],
        "detach 之后不该再有这个 turn 的 brain/*（整段 trace: {:?}）",
        kinds_of(&evs, &trace)
    );
    // 第 2 步根本没发生：剧本的第二条还在。
    assert_eq!(model.remaining(), 1);
}

// ---------------------------------------------------------------- 场景 7

/// 流吐了两段增量之后断掉：那半句话必须留下来。
///
/// 全文只有会话日志这一个去处，所以断流时把它丢掉等于"操作员无从知道模型有没有说过
/// 什么"。世界日志拿到的是长度（`partial_chars`），不是正文。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_mid_stream_error_keeps_the_text_the_model_had_already_sent() {
    let fx = Fixture::new(
        "- text: 路是空的\n  tool_calls: [{ name: robot_walk, args: { vx: 0.1 } }]\n  \
         stream_error: connection reset\n",
    );
    let b = boot(fx.rows(8, false), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let session = b.session.clone();
    let evs = b.events();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();

    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/request-error",
            "brain/turn_end",
        ],
    );
    let err = one(&evs, &trace, "brain/request-error");
    assert_eq!(err.payload["reason"], "transport: connection reset");
    assert_eq!(err.payload["partial_chars"], 4);
    // 断流之前那次工具调用也报了数——它不会被执行（上面的序列里没有 `brain/tool_call`、
    // 没有 `act/*`），但"模型要求过一次动作"和"模型什么都没说"必须分得开。
    assert_eq!(err.payload["partial_tool_calls"], 1);
    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "error: transport: connection reset"
    );
    // 世界日志里没有正文，会话日志里有。
    assert!(err.payload.get("text").is_none());
    let entries = read_session(&session).unwrap();
    assert!(
        entries.iter().any(|e| matches!(
            e,
            SessionEntry::ModelOutput { step: 0, text, .. } if text == "路是空的"
        )),
        "断流之前的半句话没进会话日志: {entries:?}"
    );
}

// ---------------------------------------------------------------- 场景 8

/// turn 里的 panic 也必须有终态。
///
/// 这是这个子系统里最难查的故障：没有 `done` 就没有 `brain/turn_end`、没有下一个 turn、
/// 日志里一个字都没有，脑永远停在 Running。内核的 turn 超时扫描是最后一道，但
/// `turn_timeout_ms: 0`（这些用例自己就这么配，一个"让快脑慢慢想"的 profile 也会）把那道
/// 也关了。Task 3 马上要把真正的 SSE 解析塞进这条循环的收流那一截——正是最容易在一个畸形
/// 字节上 panic 的代码。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_panic_inside_the_turn_still_reaches_a_terminal() {
    let fx = Fixture::new("- text: 我先想一下\n  panic: boom\n- text: 第二个 turn\n");
    let b = boot(fx.rows(8, false), None);
    b.ready().await;
    b.wake();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let evs = b.events();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();

    // panic 掉的那一步没有 `brain/model_output`（`collect` 根本没回来），但 turn 照样
    // 收尾了。
    assert_eq!(
        kinds_of(&evs, &trace),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/turn_end",
        ],
    );
    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "error: panicked"
    );
}

// ---------------------------------------------------------------- 场景 9

/// 工具永远不回来：step 的截止时刻必须把**工具批**也管上。
///
/// 一个 step 的定义是"一次模型调用 + 它要求的那一批工具"，但从前只有前半截被
/// `step_timeout_ms` 包着。一个永不回话的执行器（或者一次跑很久的 `recall_*`）因此能把
/// 这个 step 无限开着，而唯一的兜底是内核的 `turn_timeout_ms`——这些 e2e 自己就把它配成
/// 0，一个"让脑慢慢想"的 profile 也会。于是这个 turn 永远停在 Running：没有
/// `brain/turn_end`、没有下一个 turn、日志里没有任何一句话说出了什么事。
///
/// 判别信号是那条 `brain/turn_end{reason:"error: step timeout"}`（等事件，不是等时钟）；
/// `step_timeout_ms: 300` 只决定这个测试跑多久。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tool_that_never_returns_ends_the_step_and_the_turn_instead_of_hanging_it() {
    let fx = Fixture::new(
        "- text: calling the slow tool\n  tool_calls: [{ name: robot_slow, args: {} }]\n\
         - text: this line must never be requested\n",
    );
    let gate = Arc::new(Gate::default());
    let b = boot(fx.rows_with(8, true, 300), Some(gate.clone()));
    b.ready().await;
    b.wake();
    // 同步点：工具真的跑起来了，而它永远不会回来（测试不按 `release`）。
    gate.started.notified().await;
    let end = wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/turn_end" && e.payload["reason"] == "error: step timeout"
    })
    .await;
    let trace = end.trace_id.clone().expect("brain/* 必须带 trace_id");
    let status = b.brains.status("fast");
    let evs = b.events();
    let model = b.model.clone();
    // 放行只是为了别让那次执行留在半路（和场景 3 同一个理由），和断言无关。
    gate.release.notify_one();
    b.stop().await;

    assert_eq!(trace, first_trace(&evs));
    // 超时的那一批：每一条工具调用都有一条 `ok:false` 的结果，理由是 timeout。整批一起
    // 被丢掉，所以"哪几条其实跑完了"这件事已经无从知道，也没有一条结果进得了 transcript。
    let results = of(&evs, &trace, "brain/tool_result");
    assert_eq!(results.len(), 1, "{:?}", kinds_of(&evs, &trace));
    assert_eq!(results[0].payload["name"], "robot_slow");
    assert_eq!(results[0].payload["ok"], false);
    assert_eq!(results[0].payload["error"], "timeout");
    assert_eq!(
        one(&evs, &trace, "brain/request-error").payload["reason"],
        "step timeout: tools"
    );
    // 第 2 步根本没发生：只有一次 `brain/request`，剧本的第二条还在。
    assert_eq!(of(&evs, &trace, "brain/request").len(), 1);
    assert_eq!(model.remaining(), 1);
    // turn 有终态 = 脑回到 Idle，下一次唤醒还能开 turn。
    assert_eq!(status, Some(BrainStatus::Idle));
}

// ---------------------------------------------------------------- 场景 10

/// 运行中被 **inject**：`Brains::inject(fast, "look left", …)` → 下一个 step 的 user
/// message 里是 `[inject] look left`，走的是 `Lane::Inject` 那条 lane。
///
/// 和场景 4 一样，这份策略把 `brain/*` 判成 `ignore`：`look left` 能出现在 prompt 里，
/// 靠的是内核对点名给这个脑的定向事件的观测豁免。
///
/// 和场景 4（steer）的差别只有一处，而它正是这条 lane 存在的理由：inject 是"人捎带进来
/// 的一句话"，第一行就得是那句话本身。steer 的条目第一行是 `# Observation`（正文是一整份
/// 观测渲染），注入要是也那样，模型在 `# Inbox` 里看到的就是两份长得一模一样的观测，
/// 分不出哪一份带着人的话。观测照旧跟在后面缩进着——这次 steer 已经把游标推过去了，
/// 丢掉它等于把那一截事件悄悄吞掉。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_injection_while_running_lands_in_the_next_step_s_inject_lane() {
    let fx = Fixture::new(
        "- text: calling the slow tool\n  tool_calls: [{ name: robot_slow, args: {} }]\n\
         - text: ok, looking left\n",
    );
    let gate = Arc::new(Gate::default());
    let b = boot(fx.rows(8, true), Some(gate.clone()));
    b.ready().await;
    b.wake();
    gate.started.notified().await;
    b.brains.inject("fast", "look left", "console").unwrap();
    // 同步点：这条注入已经路由完（`sink.steer` 调过了，条目已经在收件箱里）。
    b.brains.settle().await;
    gate.release.notify_one();
    let end = wait_for(&b.log, |e| e.kind.as_str() == "brain/turn_end").await;
    let model = b.model.clone();
    let evs = b.events();
    b.stop().await;
    let trace = end.trace_id.clone().unwrap();

    assert_eq!(
        one(&evs, &trace, "brain/turn_end").payload["reason"],
        "end_turn"
    );
    let reqs = model.requests();
    assert_eq!(reqs.len(), 2);
    let first = user_texts(&reqs[0].messages).join("\n");
    assert!(
        !first.contains("look left"),
        "注入不该回填到第 1 步: {first}"
    );
    let second = user_texts(&reqs[1].messages).join("\n");
    assert!(
        second.contains("# Inbox\n[inject] look left\n    # Observation\n    "),
        "第 2 步的 user message 里没有那条注入: {second}"
    );
}
