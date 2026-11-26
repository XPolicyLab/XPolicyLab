//! 脑间 ask / answer 的端到端：真实内核 + 真实准入链 + 两个 `brain-native{scripted}`。
//!
//! 一条完整的链路，全程只走世界日志——没有任何脑对脑的直连：
//!
//! ```text
//! 快脑 tool ask_slow_brain{q}  → 提案 → 准入 → 执行器记 brain/ask{from: fast}
//!                                            → act/result{ask_id}
//! 慢脑（wake 策略对 brain/ask 是 wake）醒来 → robot_say{text}
//!                                            → NativeBrain 补记 brain/answer{to: fast}
//! 快脑（wake 策略对 brain/answer 是 steer）收到它 → 新 turn（此刻它已经 Idle）
//! ```
//!
//! 慢脑在同一个 turn 里说两句话，而世界日志里只有**一条** `brain/answer`（第一句）：一次
//! 提问换一次回答，否则一段三句话的答复会在提问者那边变成三次唤醒。
//!
//! 两条**反证**也在这里：内核的防自激守卫只看 `payload.brain_id`，所以
//!
//! - 快脑的 wake 策略里写着 `brain/ask: wake`，它仍然不该被**自己的**提问叫醒；
//! - 慢脑的 wake 策略里写着 `brain/answer: steer`，它也不该被**自己的**回答叫醒。
//!
//! 两个脑各自的 wake 策略都写成"只有这一条能叫醒我"，所以"谁什么时候醒"完全由这条链路
//! 决定，不由 500 Hz 的数据面决定。
//!
//! **没有一处用 sleep 当判别信号。** 等事件用世界日志观察者（`wait_for`）；"快脑的第一个
//! turn 已经收尾"这件事由一个卡住的能力（`gate`）保证——慢脑停在它的第一个 step 上，直到
//! 测试放行，于是答案**必然**晚于快脑的 `brain/turn_end`。

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
use oeh_kernel::wake::{WakeAction, WakePolicy};
use oeh_kernel::{BrainStatus, Brains, Ctx, Kernel, KernelHandle, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, ServiceName, WorldEvent};
use serde_json::{json, Value};
use tokio::sync::Notify;

// ---------------------------------------------------------------- 卡住的能力

/// `gate`：一个一直等到测试放行的 query 能力。慢脑的剧本第一步调它，于是"慢脑还没回答"
/// 在测试放行之前是**确定**的，不是赌出来的。
///
/// `Notify::notify_one` 会把许可存下来，所以测试先放行、执行器后到达也没问题。
struct GateExec {
    release: Arc<Notify>,
}

impl Executor for GateExec {
    fn execute(&self, _call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let release = self.release.clone();
        Box::pin(async move {
            release.notified().await;
            ExecOutcome {
                ok: true,
                result: json!({"released": true}),
                error: None,
            }
        })
    }
}

struct GatePlugin {
    release: Arc<Notify>,
}

impl Plugin for GatePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("test-gate").inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, _config: &Value) -> Result<(), PluginError> {
        let adm = ctx.get::<Admission>("admission")?;
        let d = adm
            .capabilities()
            .register(
                CapabilitySpec {
                    name: "gate".into(),
                    class: CapabilityClass::Query,
                    risk_level: RiskLevel::None,
                    params_schema: json!({"type": "object"}),
                    // 这次执行什么时候结束由测试决定，租约不该替它决定。
                    lease_ttl_s: 600.0,
                    overridable: false,
                },
                Arc::new(GateExec {
                    release: self.release.clone(),
                }),
            )
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(d);
        Ok(())
    }
}

// ---------------------------------------------------------------- 脚手架

/// 快脑：只有 `obs/gesture` 和 `brain/answer` 能动它。`brain/ask: wake` 是故意写上的——
/// 它自己的提问必须被内核的防自激守卫挡在门外，而不是靠"策略里没写"侥幸。
const FAST_WAKE: &str = "\
default: accumulate
rules:
  obs/gesture: wake
  brain/ask: wake
  brain/answer: steer
  brain/*: ignore
  act/*: ignore
  health/*: ignore
  safety/*: ignore
  plugin/*: ignore
budget: { max_consecutive_wakes: 32, min_interval_ms: 0 }
";

/// 慢脑：只有 `brain/ask` 能叫醒它。`brain/answer: steer` 同样是反证——它自己的回答不该
/// 把它再叫醒一次。
const SLOW_WAKE: &str = "\
default: accumulate
rules:
  brain/ask: wake
  brain/answer: steer
  brain/*: ignore
  act/*: ignore
  health/*: ignore
  safety/*: ignore
  plugin/*: ignore
budget: { max_consecutive_wakes: 32, min_interval_ms: 0 }
";

/// 快脑：turn 1 提问然后收尾（两个 step），turn 2 被回答叫醒。
const FAST_SCRIPT: &str = "\
- text: \"要过马路，先问问慢脑。\"
  tool_calls:
    - { name: ask_slow_brain, args: { q: \"safe to cross?\" } }
- text: \"问出去了，先等着。\"
- text: \"收到回答。\"
";

/// 慢脑：先卡在 `robot_gate` 上（测试放行之前不回答），再用 `robot_say` 回答，**然后再说
/// 一句**——第二句只是说话，不是第二个回答（一次提问只换一次回答），最后收尾。
const SLOW_SCRIPT: &str = "\
- text: \"让我看一眼路口。\"
  tool_calls:
    - { name: robot_gate, args: {} }
- text: \"可以过。\"
  tool_calls:
    - { name: robot_say, args: { text: \"yes\" } }
- text: \"再补一句。\"
  tool_calls:
    - { name: robot_say, args: { text: \"and mind the curb\" } }
- text: \"答完了。\"
";

struct Fixture {
    tmp: tempfile::TempDir,
}

impl Fixture {
    /// 剧本与唤醒策略都写进 tempdir，用**绝对路径**喂给插件：`brain-native` 按进程 cwd
    /// 解析 `wake`，而测试的 cwd 是 crate 目录。
    fn new() -> Self {
        let fx = Fixture {
            tmp: tempfile::tempdir().unwrap(),
        };
        fx.write("fast.wake.yaml", FAST_WAKE);
        fx.write("slow.wake.yaml", SLOW_WAKE);
        fx.write("fast.script.yaml", FAST_SCRIPT);
        fx.write("slow.script.yaml", SLOW_SCRIPT);
        fx
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let p = self.tmp.path().join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    fn at(&self, name: &str) -> PathBuf {
        self.tmp.path().join(name)
    }

    fn rows(&self) -> Vec<EntryRow> {
        let tmp = self.tmp.path();
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
                // 慢脑的 turn 一直开着直到测试放行 `gate`；内核的超时不该插手。
                "turn_timeout_ms": 0,
            })),
            EntryRow::new("admission").with_config(json!({
                "required_rules": ["rule-whitelist", "rule-fsm", "rule-run-mode", "rule-watchdog"],
                "estop_flag": tmp.join("estop"),
                "lease_check_ms": 50,
            })),
            EntryRow::new("approval-auto").with_config(json!({"allow": true})),
            // 提问和回答都走 whitelist，和 `say` 一模一样：脑间通信不是准入的旁路。
            EntryRow::new("rule-whitelist")
                .with_config(json!({"allowed": ["say", "ask_slow_brain", "gate"]})),
            EntryRow::new("rule-fsm"),
            EntryRow::new("rule-run-mode").with_config(json!({"mode": "active"})),
            EntryRow::new("rule-watchdog").with_config(json!({"streams": {"lowstate": 0.5}})),
            EntryRow::new("test-gate"),
        ];
        // 两个脑永远是最后两行：工具表是能力面在 `apply` 那一刻的快照。
        for (id, wake, script) in [
            ("fast", "fast.wake.yaml", "fast.script.yaml"),
            ("slow", "slow.wake.yaml", "slow.script.yaml"),
        ] {
            rows.push(
                EntryRow::new("brain-native")
                    .with_name(format!("brain-{id}"))
                    .with_config(json!({
                        "brain_id": id,
                        "wake": self.at(wake),
                        "persona": format!("you are the {id} brain"),
                        "model": {"scripted": self.at(script)},
                        "max_steps": 8,
                        "step_timeout_ms": 5000,
                        // RSI 的家进 tempdir：默认是进程 cwd 下的 `./run/oeh-home`，
                        // 而测试的 cwd 是 crate 目录——一次 `cargo test` 不该在仓库里
                        // 长出一个 `crates/oeh-brain/run/`。
                        "oeh_home": self.tmp.path().join("oeh-home"),
                        "protected_paths": [self.tmp.path().join("guarded")],
                    })),
            );
        }
        rows
    }
}

struct Booted {
    h: KernelHandle,
    run: tokio::task::JoinHandle<Result<(), oeh_kernel::BootError>>,
    log: Arc<WorldLog>,
    brains: Arc<Brains>,
    fast_model: Arc<ScriptedModel>,
    fast_session: PathBuf,
    slow_session: PathBuf,
}

fn boot(rows: Vec<EntryRow>, release: Arc<Notify>) -> Booted {
    let k = Kernel::boot_with(rows, &|id| {
        if id == "test-gate" {
            return Some(Box::new(GatePlugin {
                release: release.clone(),
            }));
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
    let fast_model = reg
        .get::<ScriptedModel>(&ServiceName::from("brain-native.fast.model"))
        .unwrap();
    // 会话日志的路径要在 detach 之前拿到（关机会把两个脑都摘掉）。
    let session = |id: &str| {
        brains
            .session_log_for_tests(id)
            .unwrap_or_else(|| panic!("{id} is attached"))
            .path()
            .to_path_buf()
    };
    let (fast_session, slow_session) = (session("fast"), session("slow"));
    let h = k.handle();
    let run = tokio::spawn(k.run());
    Booted {
        h,
        run,
        log,
        brains,
        fast_model,
        fast_session,
        slow_session,
    }
}

impl Booted {
    /// 数据面就位：lowstate 有过帧（`rule-watchdog` 读得到）且 FSM 到了 STANDING。
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

    /// 把快脑叫醒。
    fn wake_fast(&self) {
        self.h
            .emit(EventDraft::new(
                EventKind::new("obs/gesture").unwrap(),
                "test",
                json!({"gesture": "wave"}),
            ))
            .unwrap();
    }

    /// 世界日志此刻的全部事件。断言用的快照由测试自己决定时点——关机本身还会产生一批。
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

/// 某个脑被叫醒过几次，以及每次是被什么叫醒的。
fn wakes(evs: &[WorldEvent], brain: &str) -> Vec<String> {
    evs.iter()
        .filter(|e| e.kind.as_str() == "brain/wake" && e.payload["brain_id"] == brain)
        .map(|e| e.payload["kind"].as_str().unwrap_or("?").to_string())
        .collect()
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

fn brain_turn_end(brain: &str) -> impl Fn(&WorldEvent) -> bool + Send + Sync + 'static {
    let brain = brain.to_string();
    move |e: &WorldEvent| {
        e.kind.as_str() == "brain/turn_end" && e.payload["brain_id"] == brain.as_str()
    }
}

// ---------------------------------------------------------------- 场景 1

/// 完整链路：快脑提问 → 慢脑醒来 → 慢脑的 `robot_say` 同时被记成 `brain/answer` →
/// 快脑被这条回答叫醒。两条防自激的反证也在这里。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_ask_reaches_the_slow_brain_and_its_say_comes_back_as_an_answer() {
    let fx = Fixture::new();
    let release = Arc::new(Notify::new());
    let b = boot(fx.rows(), release.clone());
    b.ready().await;
    b.wake_fast();

    // 快脑的第一个 turn：提问，然后收尾。慢脑此刻正卡在 `gate` 上。
    let end1 = wait_for(&b.log, brain_turn_end("fast")).await;
    let trace1 = end1.trace_id.clone().expect("brain/* 必须带 trace_id");
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    // 慢脑确实是被这条提问叫醒的，而且还没来得及回答。
    let slow_wake = wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/wake" && e.payload["brain_id"] == "slow"
    })
    .await;
    assert_eq!(slow_wake.payload["kind"], "brain/ask");
    let slow_trace = slow_wake.trace_id.clone().expect("brain/* 必须带 trace_id");

    // 放行慢脑 → 回答出现 → 快脑（已经 Idle）为它开一个新 turn。
    release.notify_one();
    let answer = wait_for(&b.log, |e| e.kind.as_str() == "brain/answer").await;
    let t1 = trace1.clone();
    let end2 = wait_for(&b.log, move |e| {
        brain_turn_end("fast")(e) && e.trace_id.as_deref() != Some(t1.as_str())
    })
    .await;
    wait_for(&b.log, brain_turn_end("slow")).await;

    let fast_session = b.fast_session.clone();
    let slow_session = b.slow_session.clone();
    let fast_model = b.fast_model.clone();
    let evs = b.events();
    b.stop().await;

    // -------- 快脑的第一个 turn：提问穿过整条准入链，`brain/ask` 记在执行里
    assert_eq!(
        kinds_of(&evs, &trace1),
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
            "brain/ask",
            "act/result",
            "brain/tool_result",
            "brain/request",
            "brain/model_output",
            "brain/turn_end",
        ],
    );
    let ask = one(&evs, &trace1, "brain/ask");
    let ask_id = ask.payload["ask_id"].as_str().expect("ask_id").to_string();
    assert_eq!(ask.payload["from"], "fast");
    // `brain_id` 是提问者：防自激守卫读的就是这个字段（全局约束：每条 brain/* 都有它）。
    assert_eq!(ask.payload["brain_id"], "fast");
    assert_eq!(ask.payload["q"], "safe to cross?");
    assert_eq!(ask.salience, 150);
    let proposal_id = one(&evs, &trace1, "act/proposed").payload["proposal_id"].clone();
    assert_eq!(
        Value::String(ask_id.clone()),
        proposal_id,
        "ask_id 就是提案 id"
    );
    // 工具结果回到模型手里的是 `{ask_id}`。
    let result = one(&evs, &trace1, "act/result");
    assert_eq!(result.payload["ok"], true);
    assert_eq!(result.payload["result"]["ask_id"], ask_id.as_str());
    assert_eq!(
        one(&evs, &trace1, "brain/turn_end").payload["reason"],
        "end_turn"
    );

    // -------- 慢脑那个 turn：**第一句**说出口的话之后多出一条 `brain/answer`，
    // 第三个 step 的第二句话就只是说话了。
    assert_eq!(
        kinds_of(&evs, &slow_trace),
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
            "brain/tool_call",
            "act/proposed",
            "act/admitted",
            "act/started",
            "act/result",
            "brain/tool_result",
            "brain/answer",
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
    // 一次提问只换一次回答：两句话，一条 `brain/answer`，而且是第一句。
    assert_eq!(of(&evs, &slow_trace, "brain/answer").len(), 1);
    assert_eq!(answer.trace_id.as_deref(), Some(slow_trace.as_str()));
    assert_eq!(answer.payload["to"], "fast");
    assert_eq!(answer.payload["brain_id"], "slow");
    assert_eq!(answer.payload["ask_id"], ask_id.as_str());
    assert_eq!(answer.payload["text"], "yes");
    // 没截断就没有 `truncated` 字段；salience 和提问同级（150），按 salience 过滤日志时
    // 问和答一起出现。
    assert!(answer.payload.get("truncated").is_none());
    assert_eq!(answer.salience, 150);

    // -------- 路由：各自只醒一次该醒的那一次
    //
    // 快脑的策略里 `brain/ask: wake` 写着，它仍然没有被自己的提问叫醒；慢脑的策略里
    // `brain/answer: steer` 写着，它也没有被自己的回答叫醒。
    assert_eq!(wakes(&evs, "fast"), vec!["obs/gesture", "brain/answer"]);
    assert_eq!(wakes(&evs, "slow"), vec!["brain/ask"]);
    assert_eq!(
        kinds_of(&evs, end2.trace_id.as_deref().unwrap()),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/model_output",
            "brain/turn_end",
        ],
    );

    // 快脑是 Idle 时收到回答的 → 新 turn，不是 steer。两个脑都没有一条 steer 记录。
    for (who, path) in [("fast", &fast_session), ("slow", &slow_session)] {
        let steers = read_session(path)
            .unwrap()
            .into_iter()
            .filter(|e| matches!(e, SessionEntry::Steer { .. }))
            .count();
        assert_eq!(steers, 0, "{who} 不该收到 steer");
    }
    // 两句话都照常进慢脑自己的会话日志（`brain/answer` 只是把第一句转给提问的脑，不是
    // 说话的唯一去处）。
    let says: Vec<String> = read_session(&slow_session)
        .unwrap()
        .into_iter()
        .filter_map(|e| match e {
            SessionEntry::Say { text, .. } => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(
        says,
        vec!["yes".to_string(), "and mind the curb".to_string()]
    );

    // 回答真的到了模型眼前：第三次请求（turn 2 的 step 0）的观测里有它。
    //
    // 会话末那次 RSI review（`on_session_end` 默认开着）也会调一次模型，但它的工具表
    // 是**另一张**（只有 `skill_manage` / `memory_manage` / `read`），在这里滤掉——这个
    // 用例问的是脑间问答，不是维护相位。
    let reqs: Vec<_> = fast_model
        .requests()
        .into_iter()
        .filter(|r| !r.tools.iter().any(|t| t.name == "skill_manage"))
        .collect();
    assert_eq!(reqs.len(), 3, "turn 1 两个 step + turn 2 一个 step");
    let obs_text = user_texts(&reqs[2].messages).join("\n");
    assert!(obs_text.contains("wake: brain/answer"), "{obs_text}");
    assert!(obs_text.contains("yes"), "{obs_text}");
    // 提问的工具没有 `robot_` 前缀：问一个脑不是对世界的动作。工具描述还必须说清楚它
    // **不阻塞**——回答是之后一次单独的唤醒。
    assert!(reqs[0]
        .system
        .contains("ask_slow_brain: query capability ask_slow_brain"));
    assert!(
        reqs[0].system.contains("does NOT wait"),
        "工具描述不能让模型以为自己会等到答案: {}",
        reqs[0].system
    );
    assert!(
        !reqs[0].system.contains("robot_ask_slow_brain"),
        "同一个能力不该有两个工具名: {}",
        reqs[0].system
    );
}

// ---------------------------------------------------------------- 场景 2

/// 仓库里那两份 wake 策略本身：慢脑对 `brain/ask` 是 `wake`，快脑对 `brain/answer` 是
/// `steer`，而各自对**对方那一半**都还是 `brain/*: ignore`。上面那个 e2e 用的是测试自己
/// 的策略（预算放开、只留一条唤醒源），所以发货的这两份要单独钉一次。
#[test]
fn the_shipped_wake_policies_route_the_ask_and_the_answer() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let load = |name: &str| {
        let text = std::fs::read_to_string(root.join("wake").join(name))
            .unwrap_or_else(|e| panic!("wake/{name}: {e}"));
        WakePolicy::from_yaml(&text).unwrap_or_else(|e| panic!("wake/{name}: {e}"))
    };
    let kind = |k: &str| EventKind::new(k).unwrap();
    let fast = load("fast.yaml");
    assert_eq!(fast.action_for(&kind("brain/answer")), WakeAction::Steer);
    assert_eq!(fast.action_for(&kind("brain/ask")), WakeAction::Ignore);
    let slow = load("slow.yaml");
    assert_eq!(slow.action_for(&kind("brain/ask")), WakeAction::Wake);
    assert_eq!(slow.action_for(&kind("brain/answer")), WakeAction::Ignore);
}
