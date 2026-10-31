//! Codex app-server 包装：把一个 `codex app-server` 子进程变成内核眼里的一个
//! [`BrainSink`]。
//!
//! ```text
//! on_turn(obs, turn)               ← brains 路由任务，必须立刻返回
//!   └─ 存槽位 → 把 `turn/start` 推进出站队列 → spawn(run_turn)
//!        run_turn: 等 turn_id → 收通知
//!            item/agentMessage/delta  → 攒文本
//!            item/completed{agentMessage} → turn.note(ModelOutput)
//!            turn/completed{status}   → turn.done(end_turn | interrupted | error: …)
//! codex → item/tool/call{robot_*}    → turn.propose(能力, arguments) → {output, success}
//! codex → item/commandExecution/requestApproval → 内核装着的 Approval → accept/decline
//! ```
//!
//! 和 [`crate::native`] 逐条相同的三条纪律（理由也相同）：**先存句柄再发信号**、
//! **一个脑同一时刻只有一个 turn**、**detach 之后不复活**。[`BrainSink::interrupt`]
//! 同样只拆槽位、**不** `done`——内核在调用它的前后自己会 `finish_turn("interrupted")`。
//!
//! 这个模块自己多两条：
//!
//! 1. **出站是一条有序队列，每次发送各有自己的界。** `turn/start` / `turn/steer` /
//!    `turn/interrupt` 全部经过
//!    同一个 [`mpsc`] 队列、由同一个任务（[`pump`]）按序发出。内核是**同步**地先
//!    `sink.interrupt()` 再 `sink.on_turn()` 的，各自 `spawn` 一个任务去发的话，
//!    codex 完全可能先收到下一个 turn 的 `turn/start`、再收到上一个 turn 的
//!    `turn/interrupt`——那是一条它无从恢复的乱序。入队是同步的，所以队列里的顺序就是
//!    内核调用的顺序。队头阻塞由 [`SEND_TIMEOUT`] 兜住：一次不回话的 `turn/steer`
//!    最多压住后面的 `turn/interrupt` 五秒，而不是一个 turn 的超时那么久。**队列深度
//!    也有界**（[`MAX_OUTBOUND`]）：满了就丢**最老的一条 steer**，绝不丢
//!    `turn/start`、`turn/interrupt` 或一次待答复的请求。
//! 2. **子进程死了就是死了。** 一条看门狗任务守着连接（[`watch_exit`]）：EOF 的那一刻
//!    记一条 `brain/error{reason:"codex exited"}`、把当前 turn 以
//!    `error: codex exited` 收掉，之后每次唤醒立刻 `error: codex unavailable`。
//!    **不自动重启**——重启是 Plan 5 的 console 的事，一个自己悄悄复活的脑会把"它到底
//!    在不在"变成日志里看不出来的东西。

pub mod wire;

use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::approval::{decide_with_timeout, AskRequest, Decision};
use oeh_admission::capability::{CapabilityClass, CapabilitySpec, RiskLevel};
use oeh_kernel::observation::Observation;
use oeh_kernel::proposal::{Outcome, Proposal, ProposalId};
use oeh_kernel::session_log::SessionEntry;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{BrainSink, TurnHandle};
use oeh_proto::rpc::{code, Notification, RpcError};
use oeh_proto::{EventDraft, EventKind};
use oeh_rpc::client::Client;
use oeh_rpc::codec::LineTransport;
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, oneshot, Notify};
use tokio::task::JoinHandle;

use crate::native::prompt::render_observation;
use crate::tools::{ToolRegistry, ASK_SLOW_BRAIN};

/// 这个模块写进世界日志的事件的 `source`。
const SOURCE: &str = "brain";

/// 转给审批服务的那次询问最多等多久。和 `Admission` 默认的 `ask_timeout_ms` 一个数：
/// codex 那头正等着一个 `{decision}`，不能永远等下去。
const APPROVAL_TIMEOUT: Duration = Duration::from_secs(20);

/// 出站队列上**一次**发送最多等多久。远小于 `turn_timeout_ms`（`Client` 的调用超时）：
/// 泵是串行的，一次不回话的 `turn/steer` 会把后面排着的 `turn/interrupt` 一起压住，而
/// "打断送不到"要有一个比两分钟短得多的界。超时就记一条、接着处理下一件。
const SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// 出站队列最多排这么多件事。
///
/// 每一件都攥着一份渲染好的观测（最多 32 KiB），所以"无界"在这里的意思是"内存无界"。
/// 今天 steer 受唤醒预算约束，队列几乎不可能涨起来——这条界是为了让它**不可能**，
/// 而不是为了让它不太可能。
const MAX_OUTBOUND: usize = 256;

/// `brain/codex_approval` 里那条命令最多记多少字节（折行之后，按字符边界截断）。
const COMMAND_MAX: usize = 256;

/// 合成给审批方看的那个"能力"的名字。它**不在**能力注册表里，也不走规则链——codex 要跑
/// 的是一条命令，不是机器人的动作；这里只是借 `AskRequest` 这张表格去问同一个人。
const CODEX_COMMAND: &str = "codex.command";

/// 一个 codex 脑的运行参数。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodexConfig {
    pub brain_id: String,
    /// 子进程的 argv，第一项是可执行文件（例如 `["codex", "app-server"]`）。
    pub command: Vec<String>,
    /// 子进程的工作目录。`None` = 跟着本进程。
    pub cwd: Option<PathBuf>,
    /// 一个 turn 最多开多久（含等 codex 的每一次答复）。0 = 不设限——只有内核的
    /// `turn_timeout_ms` 兜底。
    pub turn_timeout_ms: u64,
}

impl Default for CodexConfig {
    fn default() -> Self {
        CodexConfig {
            brain_id: "codex".into(),
            command: vec!["codex".into(), "app-server".into()],
            cwd: None,
            turn_timeout_ms: 120_000,
        }
    }
}

/// 起一个 codex 脑失败的四种法子。都发生在 [`CodexBrain::spawn`] 里，也就是 attach
/// **之前**——插件把它们原样写进 `brain/error{reason}`。
#[derive(Debug, thiserror::Error)]
pub enum CodexError {
    #[error("codex command is empty")]
    EmptyCommand,
    #[error("spawn `{0}`: {1}")]
    Spawn(String, std::io::Error),
    #[error("codex subprocess has no piped stdio")]
    NoStdio,
    #[error("thread/start: {0}")]
    ThreadStart(String),
}

/// 出站队列上的一件事。入队同步、出队按序（见模块头的纪律 1）。
enum Out {
    Start {
        turn_no: u64,
        text: String,
        /// `turn/start` 的结果（codex 的 turn_id，或一句短错）回给 [`run_turn`]。
        reply: oneshot::Sender<Result<String, String>>,
    },
    Steer {
        turn_no: u64,
        trace_id: String,
        text: String,
    },
    Interrupt {
        turn_no: u64,
        trace_id: String,
    },
}

impl Out {
    /// 满了的时候可以丢哪一件。
    ///
    /// 只有 `turn/steer` 可以：它是"给正在跑的 turn 补一句话"，丢了 codex 少听见一句，
    /// 日志里记一条。`turn/start` 带着一个**必须被答复**的 oneshot（不答复 `run_turn`
    /// 就永远等下去），`turn/interrupt` 丢了就是一次送不到的打断——那两件事比排队久
    /// 严重得多。
    fn droppable(&self) -> bool {
        matches!(self, Out::Steer { .. })
    }
}

/// 出站队列：入队同步、出队按序、深度有界（[`MAX_OUTBOUND`]）。
///
/// 手写而不是 `mpsc::channel(N)`：有界 channel 满了只有"等"和"丢新来的"两条路，而这里
/// 要的是第三条——**丢最老的那条 steer**，把位置留给后面可能排着的 `turn/interrupt`。
#[derive(Default)]
struct OutQueue {
    inner: Mutex<OutInner>,
    notify: Notify,
}

#[derive(Default)]
struct OutInner {
    items: VecDeque<Out>,
    closed: bool,
}

impl OutQueue {
    /// 入队。`Ok(true)` = 为了腾位置丢掉了一条 steer；`Err(item)` = 队列已经关了
    /// （收摊之后），调用方自己给终态。
    fn push(&self, item: Out) -> Result<bool, Out> {
        let dropped = {
            let mut g = self.inner.lock();
            if g.closed {
                return Err(item);
            }
            let dropped = if g.items.len() >= MAX_OUTBOUND {
                match g.items.iter().position(Out::droppable) {
                    Some(i) => {
                        g.items.remove(i);
                        true
                    }
                    // 一件可丢的都没有（排着的全是 start/interrupt）：照样入队。这两种
                    // 一个 turn 各只有一条，天然有界，而丢掉它们的代价见 `droppable`。
                    None => false,
                }
            } else {
                false
            };
            g.items.push_back(item);
            dropped
        };
        self.notify.notify_one();
        Ok(dropped)
    }

    /// 取下一件。队列关了且排空了就是 `None`（泵收工）。
    async fn recv(&self) -> Option<Out> {
        loop {
            // `notified()` 要先 `enable()` 注册再检查条件，否则通知会落在"检查完、还没
            // 注册"的窗口里。
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut g = self.inner.lock();
                if let Some(o) = g.items.pop_front() {
                    return Some(o);
                }
                if g.closed {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// 收摊：之后的 `push` 一律 `Err`，排着的全丢，泵的 `recv` 还 `None`。
    fn close(&self) {
        {
            let mut g = self.inner.lock();
            g.closed = true;
            g.items.clear();
        }
        self.notify.notify_waiters();
    }
}

/// 当前 turn 的槽位。`on_turn` 建，`interrupt` / 收尾 / 子进程死掉时拆。
struct Current {
    turn_no: u64,
    handle: TurnHandle,
    task: Option<JoinHandle<()>>,
}

#[derive(Default)]
struct State {
    current: Option<Current>,
    /// 最近一次 `turn/start` 答复给出的 `(内核 turn 号, codex turn_id)`。
    ///
    /// 和槽位分开存**是有意的**：`interrupt()` 会把槽位整个拆掉，但那一刻 codex 的
    /// turn_id 完全可能还没回来（泵还在等 `turn/start` 的答复）。泵是严格按序的，
    /// `Out::Interrupt` 一定排在同一个 turn 的 `Out::Start` 后面——也就是说真正要发
    /// `turn/interrupt` 的那一刻这个信息**一定**有了。在入队时刻去读槽位就会白白丢掉
    /// 一次打断，而 codex 会继续跑一个内核已经关掉的 turn，紧接着又收到下一个
    /// `turn/start`。
    last_started: Option<(u64, String)>,
}

/// `CodexBrain` 的共享内脏。出站泵、看门狗、两个反向请求处理器拿的都是它的
/// [`Weak`]——攥强引用就意味着 detach 之后这些任务还活着（而子进程也就不会被杀）。
struct Shared {
    cfg: CodexConfig,
    log: Arc<WorldLog>,
    admission: Arc<Admission>,
    client: Client,
    /// `thread/start` 之后才有；之后永不变化。
    thread_id: OnceLock<String>,
    /// 能力面的投影，每个 `turn/start` 作为 `additional_context` 带上。
    ///
    /// **每个 turn 对一次账，不是每个 step**：`additional_context` 只在 `turn/start` 上
    /// 发，而它每变一次就作废一次 codex 自己的 prompt 缓存。所以这里和 `brain-native`
    /// 的工具表是同一条纪律（[`crate::native`] 的 `Projection`）——按
    /// [`CapabilityRegistry::generation`] 比一次，能力面没动过就原样复用同一个 `Arc`，
    /// 那一份逐字节不变；动过了才重投，而那一次的缓存失效正是要付的代价。
    tools_context: Mutex<ToolsCtx>,
    child: Mutex<Option<Child>>,
    state: Mutex<State>,
    out: Arc<OutQueue>,
    /// 泵与看门狗。`shutdown` 一并 abort。
    tasks: Mutex<Vec<JoinHandle<()>>>,
    /// 子进程已经没了。之后每次 `on_turn` 立刻 `done("error: codex unavailable")`。
    dead: AtomicBool,
    /// 我们自己正在收摊（detach）。看门狗靠它区分"codex 死了"和"我们杀了它"。
    shutting_down: AtomicBool,
}

/// 一个由 `codex app-server` 子进程驱动的脑。
pub struct CodexBrain {
    s: Arc<Shared>,
}

impl CodexBrain {
    /// 起子进程 → 接上 JSON-RPC → `thread/start`。返回之后这个脑就可以 attach 了。
    ///
    /// 失败时子进程已经被收掉（`Shared` 连同 [`Child`] 一起析构，`kill_on_drop`）。
    pub async fn spawn(
        cfg: CodexConfig,
        admission: Arc<Admission>,
        log: Arc<WorldLog>,
    ) -> Result<Arc<Self>, CodexError> {
        let (exe, args) = cfg.command.split_first().ok_or(CodexError::EmptyCommand)?;
        let mut cmd = Command::new(exe);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr 原样透传：codex 自己的日志是操作员排障的第一手材料，吞掉它等于把
            // 子进程变成一个黑盒。
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        if let Some(dir) = &cfg.cwd {
            cmd.current_dir(dir);
        }
        let mut child = cmd.spawn().map_err(|e| CodexError::Spawn(exe.clone(), e))?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(CodexError::NoStdio);
        };
        let mut client = Client::from_transport(LineTransport::new(stdout, stdin));
        if cfg.turn_timeout_ms > 0 {
            client = client.with_timeout(Duration::from_millis(cfg.turn_timeout_ms));
        }
        let caps = admission.capabilities();
        let (ctx, skipped) = tools_context(&caps.list());
        if !skipped.is_empty() {
            warn_skipped(&skipped);
        }
        let tools_context = Mutex::new(ToolsCtx {
            generation: caps.generation(),
            context: Arc::new(ctx),
            skipped,
        });
        let out = Arc::new(OutQueue::default());
        let s = Arc::new(Shared {
            cfg,
            log,
            admission,
            client,
            thread_id: OnceLock::new(),
            tools_context,
            child: Mutex::new(Some(child)),
            state: Mutex::new(State::default()),
            out: out.clone(),
            tasks: Mutex::new(Vec::new()),
            dead: AtomicBool::new(false),
            shutting_down: AtomicBool::new(false),
        });
        // 处理器装在 `thread/start` **之前**：codex 完全可以在答复线程之前就先问点什么，
        // 而一个没人认领的反向请求只会让它干等到超时。
        install_handlers(&s);
        let v = s
            .client
            .call(wire::THREAD_START, json!({}))
            .await
            .map_err(|e| CodexError::ThreadStart(e.message))?;
        let started: wire::ThreadStarted =
            serde_json::from_value(v).map_err(|e| CodexError::ThreadStart(e.to_string()))?;
        let _ = s.thread_id.set(started.thread_id);

        let pump_task = tokio::spawn(pump(Arc::downgrade(&s), out));
        let watch_task = tokio::spawn(watch_exit(Arc::downgrade(&s), s.client.clone()));
        s.tasks.lock().extend([pump_task, watch_task]);
        Ok(Arc::new(CodexBrain { s }))
    }
}

impl BrainSink for CodexBrain {
    fn on_turn(&self, obs: Observation, turn: TurnHandle) {
        if self.s.dead.load(Ordering::SeqCst) {
            defer_done(turn, "error: codex unavailable");
            return;
        }
        let n = turn.turn;
        let text = render_observation(&obs);
        // 订阅在 `turn/start` 入队**之前**：codex 完全可能在答复 turn_id 的同一口气里就
        // 把第一段增量发出来，晚订阅一步就永远看不到它。
        let rx = self.s.client.notifications();
        let (tx, reply) = oneshot::channel();
        // 槽位先就位（纪律 1、2）。被换下来的那个出锁之后才 abort。
        let prev = {
            let mut g = self.s.state.lock();
            g.current.replace(Current {
                turn_no: n,
                handle: turn.clone(),
                task: None,
            })
        };
        if let Some(t) = prev.and_then(|c| c.task) {
            t.abort();
        }
        match self.s.out.push(Out::Start {
            turn_no: n,
            text,
            reply: tx,
        }) {
            Ok(dropped) => self.s.note_queue_full(dropped, Some(&turn.trace_id)),
            Err(_) => {
                // 泵没了（只会在收摊之后发生）。
                let _ = self.s.take_current(Some(n));
                defer_done(turn, "error: codex unavailable");
                return;
            }
        }
        let s = self.s.clone();
        let task = tokio::spawn(run_turn(s, turn, rx, reply));
        let mut g = self.s.state.lock();
        match g.current.as_mut() {
            Some(c) if c.turn_no == n => c.task = Some(task),
            // 这一瞬间槽位已经被下一个 turn（或一次 interrupt）换掉了：任务自己会在第一次
            // `is_current` 检查时退出，但别留下一个谁都 abort 不了的孤儿。
            _ => task.abort(),
        }
    }

    /// 进行中的 turn 收到一句话：渲染成文本推进出站队列（`turn/steer`）。
    ///
    /// 没有 turn 在跑就丢掉——内核对 idle 的脑发定向 steer 走的是 `start_turn`，那条路
    /// 进的是 [`BrainSink::on_turn`]，不是这里。
    fn steer(&self, obs: Observation) {
        let text = render_observation(&obs);
        let cur = {
            let g = self.s.state.lock();
            g.current
                .as_ref()
                .map(|c| (c.turn_no, c.handle.trace_id.clone()))
        };
        let Some((turn_no, trace_id)) = cur else {
            return;
        };
        if let Ok(dropped) = self.s.out.push(Out::Steer {
            turn_no,
            trace_id: trace_id.clone(),
            text,
        }) {
            self.s.note_queue_full(dropped, Some(&trace_id));
        }
    }

    /// 只拆槽位 + 转发 `turn/interrupt`，**不** `done`：内核已经
    /// `finish_turn("interrupted")` 过了，脑再 `done` 一次只会被陈旧判定挡下来。
    fn interrupt(&self) {
        let Some(cur) = self.s.take_current(None) else {
            return;
        };
        if let Some(t) = cur.task {
            t.abort();
        }
        // 同步入队（纪律 1）：内核接着就会 `on_turn` 下一个 turn，`turn/interrupt` 必须
        // 排在那个 `turn/start` 前面。codex 的 turn_id 这一刻可能还没回来，由泵在真正
        // 发送时解析（见 [`State::last_started`]）。
        if let Ok(dropped) = self.s.out.push(Out::Interrupt {
            turn_no: cur.turn_no,
            trace_id: cur.handle.trace_id.clone(),
        }) {
            self.s.note_queue_full(dropped, Some(&cur.handle.trace_id));
        }
    }
}

/// detach 即刻收摊：泵、看门狗、turn 任务全 abort，子进程当场杀掉。
///
/// `Brains::attach` 存的 `Arc<dyn BrainSink>` 是这个 sink 唯一的强引用（插件交出去之后
/// 只留一个 `Weak`），所以 detach 把 `Attached` 摘掉的那一刻就是这里。**必须收**：没有
/// 这一下，一个已经没人认领的 codex 子进程会一直活到进程退出。
impl Drop for CodexBrain {
    fn drop(&mut self) {
        self.s.shutdown();
    }
}

// ---------------------------------------------------------------- 一个 turn

/// 一个 turn：等 `turn/start` 的答复，然后收通知直到 `turn/completed`。
async fn run_turn(
    s: Arc<Shared>,
    turn: TurnHandle,
    mut rx: broadcast::Receiver<Notification>,
    reply: oneshot::Receiver<Result<String, String>>,
) {
    let n = turn.turn;
    let turn_id = match reply.await {
        Ok(Ok(id)) => id,
        Ok(Err(e)) => {
            s.emit(
                Some(&turn.trace_id),
                "brain/error",
                json!({"reason": "codex turn/start failed", "detail": e}),
            );
            s.finish(n, "error: codex turn/start failed");
            return;
        }
        // 泵没了（收摊）。终态不归我们管：`shutdown` 已经把槽位拆了。
        Err(_) => return,
    };
    let deadline = (s.cfg.turn_timeout_ms > 0)
        .then(|| tokio::time::Instant::now() + Duration::from_millis(s.cfg.turn_timeout_ms));
    // 一条 agent 消息的文本。`item/completed` 是它的收尾；收尾之前的增量都攒在这里。
    let mut text = String::new();
    let mut step = 0u64;
    // 点名了别的 turn、被丢掉的增量条数。一个 turn 记一次（见 [`note_dropped`]）。
    let mut dropped = 0u64;
    loop {
        let next = match by(deadline, rx.recv()).await {
            Ok(n) => n,
            Err(_) => {
                flush(&turn, &mut text, &mut step);
                note_dropped(&s, &turn, dropped);
                s.emit(
                    Some(&turn.trace_id),
                    "brain/error",
                    json!({"reason": "codex turn timeout"}),
                );
                s.finish(n, "error: codex turn timeout");
                return;
            }
        };
        let ev = match next {
            Ok(ev) => ev,
            // 跟不上广播就跳过丢掉的那几条（文本会缺一段，好过整个 turn 卡死）。
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            // 只有 `Client` 整个析构才会到这里，而我们攥着 `Shared`。
            Err(broadcast::error::RecvError::Closed) => return,
        };
        // 陈旧就直接退出——不发事件、不 `done`（纪律 3）。
        if !turn.is_current() {
            return;
        }
        let Notification { method, params, .. } = ev;
        match method.as_str() {
            wire::AGENT_MESSAGE_DELTA => {
                if let Ok(d) = serde_json::from_value::<wire::AgentMessageDelta>(params) {
                    match d.turn_id.as_deref() {
                        // 没说属于谁 = 属于当前 turn。真的 app-server 不保证每条增量都带
                        // 这个字段，而"宁可丢掉全部输出"是最坏的一种保守。
                        None => text.push_str(&d.delta),
                        Some(id) if id == turn_id => text.push_str(&d.delta),
                        // 点名了别的 turn：丢掉，但**数着**——静默丢弃正是要避免的那件事。
                        Some(_) => dropped += 1,
                    }
                }
            }
            wire::ITEM_COMPLETED => {
                if let Ok(c) = serde_json::from_value::<wire::ItemCompleted>(params) {
                    if c.item.item_type == wire::ITEM_AGENT_MESSAGE {
                        flush(&turn, &mut text, &mut step);
                    }
                }
            }
            wire::TURN_COMPLETED => {
                let Ok(c) = serde_json::from_value::<wire::TurnCompleted>(params) else {
                    continue;
                };
                if c.turn_id != turn_id {
                    continue;
                }
                // 没收尾的那半句也要落地：`turn/completed` 就是最后的收尾信号。
                flush(&turn, &mut text, &mut step);
                note_dropped(&s, &turn, dropped);
                // `status` 决定 `brain/turn_end.reason`。一次失败的 turn 和一次正常收尾
                // 在日志里长得一模一样，正是操作员最不需要的那种体贴。
                match c.status.as_str() {
                    "" | "completed" => s.finish(n, "end_turn"),
                    // 打断通常是**内核**发起的（`Brains::interrupt` → `sink.interrupt()`
                    // → `finish_turn("interrupted")`），那一路上这个 turn 早就关掉了，
                    // 循环顶上的 `is_current` 检查已经把我们请出去了。走到这里只剩
                    // "codex 自己把 turn 打断了"这一种，那就由脑来给终态。
                    "interrupted" if turn.is_current() => s.finish(n, "interrupted"),
                    "interrupted" => {}
                    other => s.finish(n, &format!("error: codex turn {other}")),
                }
                return;
            }
            _ => {}
        }
    }
}

/// 一个 turn 里被丢掉的增量条数 → 一条事件。0 条就不记。
fn note_dropped(s: &Shared, turn: &TurnHandle, dropped: u64) {
    if dropped > 0 {
        s.emit(
            Some(&turn.trace_id),
            "brain/codex_dropped",
            json!({ "deltas": dropped }),
        );
    }
}

/// 攒到现在的这一段 agent 文本 → 会话日志。**全文只在那里**（§12：世界日志里不放模型
/// 说的话）。
fn flush(turn: &TurnHandle, text: &mut String, step: &mut u64) {
    if text.is_empty() {
        return;
    }
    let _ = turn.note(SessionEntry::ModelOutput {
        turn: 0,
        step: *step,
        text: std::mem::take(text),
    });
    *step += 1;
}

// ---------------------------------------------------------------- 出站泵 / 看门狗

/// 出站队列的唯一消费者：一件一件按序发出去（见模块头的纪律 1）。
///
/// 对 `Shared` 拿 [`Weak`]：攥强引用的话这个任务会把 `Shared`（连同子进程）永远留住。
/// 队列本身是单独的 [`Arc`]，`Shared::shutdown` 把它 `close` 掉，`recv()` 于是还 `None`、
/// 这个任务自己收工。
async fn pump(w: Weak<Shared>, q: Arc<OutQueue>) {
    while let Some(o) = q.recv().await {
        let Some(s) = w.upgrade() else { return };
        match o {
            Out::Start {
                turn_no,
                text,
                reply,
            } => {
                let r = s.start_turn(turn_no, text).await;
                let _ = reply.send(r);
            }
            Out::Steer {
                turn_no,
                trace_id,
                text,
            } => s.send_steer(turn_no, &trace_id, text).await,
            Out::Interrupt { turn_no, trace_id } => s.send_interrupt(turn_no, &trace_id).await,
        }
    }
}

/// 子进程死了（或者连接断了）：记一条、把当前 turn 收掉、从此不再接活。
async fn watch_exit(w: Weak<Shared>, client: Client) {
    client.wait_closed().await;
    let Some(s) = w.upgrade() else { return };
    // 是我们自己杀的（detach）——那不是故障，也没有 turn 要收。
    if s.shutting_down.load(Ordering::SeqCst) {
        return;
    }
    s.dead.store(true, Ordering::SeqCst);
    match s.take_current(None) {
        Some(c) => {
            if let Some(t) = c.task {
                t.abort();
            }
            s.emit(
                Some(&c.handle.trace_id),
                "brain/error",
                json!({"reason": "codex exited"}),
            );
            c.handle.done("error: codex exited");
        }
        // 没有 turn 在跑：这条事件不属于任何一个 turn，所以没有 trace 可挂——但仍然要记，
        // 否则"这个脑怎么不动了"在日志里一点线索都没有。
        None => s.emit(None, "brain/error", json!({"reason": "codex exited"})),
    }
}

// ---------------------------------------------------------------- Shared

/// [`Shared::tools_context`] 的内容：投出它时能力面的代号、渲染好的
/// `additional_context`，以及那一轮被跳过的能力名（见 [`Shared::tools_context_for_turn`]）。
struct ToolsCtx {
    generation: u64,
    context: Arc<String>,
    skipped: Vec<String>,
}

impl Shared {
    /// 这个 turn 的 `additional_context`。**一个 turn 只调一次**（`turn/start` 是唯一
    /// 带它的消息）。代号相等就是同一个 `Arc`：codex 那边看到的字节一个都没变。
    fn tools_context_for_turn(&self) -> Arc<String> {
        let caps = self.admission.capabilities();
        let generation = caps.generation();
        {
            let g = self.tools_context.lock();
            if g.generation == generation {
                return g.context.clone();
            }
        }
        let (ctx, skipped) = tools_context(&caps.list());
        let mut g = self.tools_context.lock();
        if skipped != g.skipped {
            warn_skipped(&skipped);
        }
        *g = ToolsCtx {
            generation,
            context: Arc::new(ctx),
            skipped,
        };
        g.context.clone()
    }

    fn thread_id(&self) -> String {
        self.thread_id.get().cloned().unwrap_or_default()
    }

    /// 一次出站调用，**带自己的截止时刻**（[`SEND_TIMEOUT`]）。
    ///
    /// `Client` 自己的超时是 `turn_timeout_ms`（默认两分钟）——那是"一个 turn 能想多久"，
    /// 不是"一条控制消息能发多久"。泵是串行的，用前者当界的话，一次不回话的
    /// `turn/steer` 会把排在后面的 `turn/interrupt` 压住两分钟。
    async fn call_bounded(&self, method: &str, params: Value) -> Result<Value, String> {
        match tokio::time::timeout(SEND_TIMEOUT, self.client.call(method, params)).await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e.message),
            Err(_) => Err("send timeout".into()),
        }
    }

    /// `turn/start`。成功时记下 `(turn 号, codex turn_id)`——steer / interrupt 要指名道姓。
    async fn start_turn(&self, turn_no: u64, text: String) -> Result<String, String> {
        let params = serde_json::to_value(wire::TurnStart {
            thread_id: self.thread_id(),
            input: vec![wire::Input::Text { text }],
            additional_context: Some((*self.tools_context_for_turn()).clone()),
        })
        .map_err(|e| e.to_string())?;
        let v = self.call_bounded(wire::TURN_START, params).await?;
        let started: wire::TurnStarted = serde_json::from_value(v).map_err(|e| e.to_string())?;
        // 记在槽位**外面**：`interrupt()` 随时可能把槽位拆掉，而那次打断还要这个 id。
        self.state.lock().last_started = Some((turn_no, started.turn_id.clone()));
        Ok(started.turn_id)
    }

    /// 某个内核 turn 号对应的 codex turn_id。泵在**真正要发**的那一刻才解析。
    fn turn_id_for(&self, turn_no: u64) -> Option<String> {
        match &self.state.lock().last_started {
            Some((n, id)) if *n == turn_no => Some(id.clone()),
            _ => None,
        }
    }

    async fn send_steer(&self, turn_no: u64, trace_id: &str, text: String) {
        let Some(turn_id) = self.turn_id_for(turn_no) else {
            // `turn/start` 根本没成功过：没有 turn 可以插话。
            self.send_failed(Some(trace_id), wire::TURN_STEER, "no codex turn");
            self.emit(Some(trace_id), "brain/codex_steer", json!({"ok": false}));
            return;
        };
        let params = match serde_json::to_value(wire::TurnSteer {
            thread_id: self.thread_id(),
            turn_id,
            text,
        }) {
            Ok(p) => p,
            Err(e) => return self.send_failed(Some(trace_id), wire::TURN_STEER, &e.to_string()),
        };
        let ok = match self.call_bounded(wire::TURN_STEER, params).await {
            Ok(_) => true,
            Err(e) => {
                self.send_failed(Some(trace_id), wire::TURN_STEER, &e);
                false
            }
        };
        // 一句 steer 到底有没有真的进到 codex 那个 turn 里，除了这条事件在日志里看不
        // 出来（内容按纪律不进世界日志，所以只报成败）。
        self.emit(Some(trace_id), "brain/codex_steer", json!({ "ok": ok }));
    }

    async fn send_interrupt(&self, turn_no: u64, trace_id: &str) {
        let Some(turn_id) = self.turn_id_for(turn_no) else {
            // 这个 turn 的 `turn/start` 从来没成功过 —— codex 那边压根没有它，也就没有
            // 什么要打断的。
            return;
        };
        let params = match serde_json::to_value(wire::TurnRef {
            thread_id: self.thread_id(),
            turn_id,
        }) {
            Ok(p) => p,
            Err(e) => {
                return self.send_failed(Some(trace_id), wire::TURN_INTERRUPT, &e.to_string())
            }
        };
        if let Err(e) = self.call_bounded(wire::TURN_INTERRUPT, params).await {
            self.send_failed(Some(trace_id), wire::TURN_INTERRUPT, &e);
        }
    }

    /// 队列满了、为了腾位置丢掉一条 steer：记一条。`dropped == false` 就什么都不做。
    ///
    /// 丢弃必须在日志里看得见——"操作员 steer 了一句，codex 那边没反应"是这个子系统最
    /// 难查的一种故障，而静默丢弃正是它的成因。
    fn note_queue_full(&self, dropped: bool, trace: Option<&str>) {
        if dropped {
            self.emit(
                trace,
                "brain/error",
                json!({"reason": "codex outbound queue full", "dropped": "turn/steer"}),
            );
        }
    }

    /// 一次发不出去的出站调用。泵**不**因此停下——记一条，接着处理下一件。
    fn send_failed(&self, trace: Option<&str>, method: &str, detail: &str) {
        self.emit(
            trace,
            "brain/error",
            json!({"reason": "codex send failed", "method": method, "detail": detail}),
        );
    }

    /// codex 调一个 DynamicTool。工具面上有的走提案，别的名字明确说"没这个工具"。
    async fn on_tool_call(&self, params: Value) -> wire::ToolReply {
        let Ok(call) = serde_json::from_value::<wire::ToolCall>(params) else {
            return wire::ToolReply::failed("bad item/tool/call params");
        };
        let Some(capability) = capability_of(&call.name) else {
            // 说清楚是"没这个工具"而不是"执行失败"：codex 会因此换一个工具，而不是重试。
            return wire::ToolReply::failed("unknown tool");
        };
        // 句柄先克隆出来再 await——锁不能跨 `.await`。
        //
        // `item/tool/call` 的线格式里**没有** turn 标识（真的 app-server 也只给
        // `{name, arguments, call_id}`），所以这里能做的是把"进来的那一刻的槽位"钉住，
        // 并在提案前后各校一次它还是不是当前 turn。少了这一步，一次属于 turn N 的调用
        // 会在 N+1 的 trace 底下被提案、被执行——不是安全洞（照样过准入链），但审计上
        // 是一次张冠李戴。
        let handle = self.state.lock().current.as_ref().map(|c| c.handle.clone());
        let Some(handle) = handle else {
            return wire::ToolReply::failed("no turn");
        };
        if !handle.is_current() {
            return wire::ToolReply::failed("no turn");
        }
        // 唯一的下场路径：提案。FSM、执行器、租约都在准入链那一侧，这里碰不到。
        let outcome = handle.propose(capability, call.arguments).await;
        // 等提案跑完的这段时间里 turn 可能已经收尾/被打断了。结局本身在世界日志里有
        // （`act/*` 挂在它自己的 trace 下），但**这条答复**已经不属于任何一个活着的
        // turn 了，照实说。
        if !handle.is_current() {
            return wire::ToolReply::failed("no turn");
        }
        let success = matches!(outcome, Outcome::Executed { .. });
        let output = serde_json::to_string(&outcome)
            .unwrap_or_else(|e| format!("{{\"error\":\"unserialisable outcome: {e}\"}}"));
        wire::ToolReply { output, success }
    }

    /// codex 要执行一条命令：问内核**此刻**装着的审批服务。没装就拒——fail closed。
    async fn on_approval(&self, params: Value) -> wire::ApprovalDecision {
        let req: wire::ApprovalRequest = serde_json::from_value(params).unwrap_or_default();
        let (trace, turn_no) = {
            let g = self.state.lock();
            match g.current.as_ref() {
                Some(c) => (Some(c.handle.trace_id.clone()), c.turn_no),
                None => (None, 0),
            }
        };
        // 日志里必须看得出**拒的是什么**：一条提交给人类审批的命令是 `act/*` 参数那一
        // 类的事实，不是模型独白（§12 禁的是后者）。全文不记——折行之后截到
        // [`COMMAND_MAX`] 字节。
        let command = command_summary(&req.command);
        let cwd = req.cwd.clone();
        let Some(approval) = self.admission.approval() else {
            // 没有人可问 → 没有 `verdict` 可言，只有我们在线上给出的那个答复。
            // fail closed。
            self.emit(
                trace.as_deref(),
                "brain/codex_approval",
                json!({
                    "decision": "decline",
                    "reason": "no approver",
                    "command": command,
                    "cwd": cwd,
                }),
            );
            return wire::ApprovalDecision::Decline;
        };
        let trace_id = trace.clone().unwrap_or_default();
        let ask = AskRequest {
            proposal: Proposal {
                proposal_id: ProposalId(format!("{trace_id}-codex-{}", req.item_id)),
                brain_id: self.cfg.brain_id.clone(),
                turn: turn_no,
                trace_id,
                capability: CODEX_COMMAND.into(),
                params: json!({"command": req.command}),
            },
            // 这份 spec 不进规则链（它不是注册过的能力），只是给审批方看的说明：一条
            // 命令是控制类、高风险，没有租约可言。
            spec: CapabilitySpec {
                name: CODEX_COMMAND.into(),
                class: CapabilityClass::Control,
                risk_level: RiskLevel::High,
                params_schema: json!({"type": "object"}),
                lease_ttl_s: 0.0,
                overridable: false,
            },
            reason: "codex requestApproval".into(),
            trace: Vec::new(),
        };
        // 走 `decide_with_timeout` 而不是直接 `decide`：审批方 panic 或者干脆不回话时，
        // codex 那头还在等一个答复。
        // 两个字段，两件事：`verdict` 是审批方的判定（只有真的问过人才有），
        // `decision` 是我们在线上给 codex 的那个答复。一个字段混着三种取值
        // （allow/deny/decline）只会让日志查询写错。
        let (verdict, by, reason, decision) =
            match decide_with_timeout(Some(approval), ask, APPROVAL_TIMEOUT).await {
                Decision::Allow { by, reason } => {
                    ("allow", by, reason, wire::ApprovalDecision::Accept)
                }
                Decision::Deny { by, reason } => {
                    ("deny", by, reason, wire::ApprovalDecision::Decline)
                }
            };
        self.emit(
            trace.as_deref(),
            "brain/codex_approval",
            json!({
                "decision": decision,
                "verdict": verdict,
                "by": by,
                "reason": reason,
                "command": command,
                "cwd": cwd,
            }),
        );
        decision
    }

    /// 交出槽位。`which == Some(n)` 时只交 turn 号对得上的那个。
    fn take_current(&self, which: Option<u64>) -> Option<Current> {
        let mut g = self.state.lock();
        let mine = match (&g.current, which) {
            (Some(c), Some(n)) => c.turn_no == n,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if mine {
            g.current.take()
        } else {
            None
        }
    }

    /// 收尾：交出槽位并 `done`。turn 号对不上（已经被打断/换人/看门狗收掉）就什么都不做。
    ///
    /// **不** abort 槽位里的任务：调用它的通常就是那个任务自己。
    fn finish(&self, turn_no: u64, reason: &str) {
        let Some(cur) = self.take_current(Some(turn_no)) else {
            return;
        };
        cur.handle.done(reason);
    }

    /// detach：把所有还在跑的东西收掉，子进程当场杀死。幂等。
    fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        // 队列先关：之后每次 `on_turn` 都会在 `push` 上拿到 `Err`，立刻给终态，而不是
        // 把一条永远发不出去的 `turn/start` 排进一个没人读的队列。
        self.out.close();
        if let Some(c) = self.take_current(None) {
            if let Some(t) = c.task {
                t.abort();
            }
        }
        for t in self.tasks.lock().drain(..) {
            t.abort();
        }
        // `kill_on_drop` 是兜底，但 `Shared` 可能被还没退干净的任务多攥一会儿，而一个
        // 没人管的 codex 进程会一直活到本进程退出。
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.start_kill();
        }
    }

    /// 一条 `brain/*`：`brain_id` 由这里补上，`trace_id` 有 turn 的时候才有（spawn 失败
    /// 和"没有 turn 在跑时子进程死掉"这两条本来就不属于任何一个 turn）。
    fn emit(&self, trace: Option<&str>, kind: &str, mut payload: Value) {
        if let Value::Object(m) = &mut payload {
            m.insert("brain_id".into(), Value::String(self.cfg.brain_id.clone()));
        }
        let mut draft =
            EventDraft::new(EventKind::new(kind).expect("static kind"), SOURCE, payload);
        if let Some(t) = trace {
            draft = draft.with_trace_id(t);
        }
        self.log.append(draft);
    }
}

// ---------------------------------------------------------------- 小零件

/// 把两个反向请求（codex → 我们）接上。处理器里拿 [`Weak`]：`Client` 的处理器表活在
/// `Shared` 里面，攥强引用就是一个谁也析构不了的环。
fn install_handlers(s: &Arc<Shared>) {
    let w = Arc::downgrade(s);
    s.client.on_request(
        wire::TOOL_CALL,
        Arc::new(move |params: Value| {
            let w = w.clone();
            Box::pin(async move {
                let reply = match w.upgrade() {
                    Some(s) => s.on_tool_call(params).await,
                    None => wire::ToolReply::failed("brain detached"),
                };
                as_result(&reply)
            })
        }),
    );
    let w = Arc::downgrade(s);
    s.client.on_request(
        wire::REQUEST_APPROVAL,
        Arc::new(move |params: Value| {
            let w = w.clone();
            Box::pin(async move {
                let decision = match w.upgrade() {
                    // detach 之后还来问：fail closed。
                    None => wire::ApprovalDecision::Decline,
                    Some(s) => s.on_approval(params).await,
                };
                as_result(&wire::ApprovalReply { decision })
            })
        }),
    );
}

/// 一条命令（argv 数组或一整行字符串）→ 一段能进世界日志的短串：紧凑 JSON、所有空白
/// 折成单个空格（一条多行 heredoc 不该把日志撑成一页）、按**字符**边界截到
/// [`COMMAND_MAX`] 字节。
fn command_summary(v: &Value) -> String {
    let raw = match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<unserialisable>".into()),
    };
    let one_line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.len() <= COMMAND_MAX {
        return one_line;
    }
    const MARK: &str = "…";
    let room = COMMAND_MAX - MARK.len();
    let cut = (0..=room)
        .rev()
        .find(|i| one_line.is_char_boundary(*i))
        .unwrap_or(0);
    let mut s = one_line[..cut].to_string();
    s.push_str(MARK);
    s
}

fn as_result<T: Serialize>(v: &T) -> Result<Value, RpcError> {
    serde_json::to_value(v).map_err(|e| RpcError::new(code::INTERNAL, e.to_string()))
}

/// codex 的工具名 → 能力名。
///
/// `robot_*` 是能力面的投影；`ask_slow_brain` 是唯一一个**不带**前缀的能力工具（问一个脑
/// 不是对机器人的动作），两个脑的工具表上都是这个名字，所以这里也得认它——只广播不受理
/// 等于告诉模型有这个工具然后一直回 "unknown tool"。
fn capability_of(tool: &str) -> Option<&str> {
    tool.strip_prefix(wire::ROBOT_PREFIX)
        .or_else(|| (tool == ASK_SLOW_BRAIN).then_some(tool))
}

/// 能力面 → `additional_context`。复用 `brain-native` 那套投影（`robot_` 前缀、描述、
/// schema 兜底）**加上那张表里手写的 `ask_slow_brain`**，两个脑看到的工具名才是同一套
/// ——`project_capabilities` 会跳过那个能力（它的工具名没有前缀），少了这一行，codex 脑
/// 就是唯一一个不知道自己能提问的脑。
///
/// 返回的第二项是被跳过的能力名（工具名用不了的那些）。谁调用谁决定这条告警说几遍：
/// 这个函数现在**每次能力面变动**都会被调一次，而一条每次都重复的告警等于没有告警。
fn tools_context(specs: &[CapabilitySpec]) -> (String, Vec<String>) {
    let mut reg = ToolRegistry::new();
    let skipped = reg.project_capabilities(specs);
    reg.add_ask_slow_brain();
    let tools: Vec<Value> = reg
        .specs()
        .iter()
        .map(|t| json!({"name": t.name, "description": t.description, "parameters": t.parameters}))
        .collect();
    let ctx = serde_json::to_string(&json!({ "tools": tools }))
        .unwrap_or_else(|_| r#"{"tools":[]}"#.into());
    (ctx, skipped)
}

/// 说一次"这个能力上不了工具表"。三家厂商对工具名的字符集都有要求，一个
/// `robot_walk.fast` 换来的是整个请求的 400——那时候所有工具都用不了，而日志里没有任何
/// 线索。
fn warn_skipped(names: &[String]) {
    tracing::warn!(
        capabilities = %names.join(", "),
        "capability names are not usable tool names ([a-z0-9_]); they are not advertised to codex"
    );
}

/// 立刻给这个 turn 一个终态，但**不在调用者的栈上**：`done` 会让内核接着开下一个 turn，
/// 而那条路会同步回到 [`BrainSink::on_turn`]——在那里直接 `done` 就是一次递归。
fn defer_done(turn: TurnHandle, reason: &str) {
    let reason = reason.to_string();
    match tokio::runtime::Handle::try_current() {
        Ok(h) => {
            h.spawn(async move { turn.done(&reason) });
        }
        // 没有 runtime 可用：递归的风险好过一个永远收不了尾的 turn。
        Err(_) => turn.done(&reason),
    }
}

/// `f`，但最晚到 `deadline`。`None` = 不设限。
async fn by<F: Future>(
    deadline: Option<tokio::time::Instant>,
    f: F,
) -> Result<F::Output, tokio::time::error::Elapsed> {
    match deadline {
        Some(d) => tokio::time::timeout_at(d, f).await,
        None => Ok(f.await),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;

    fn spec(name: &str, class: CapabilityClass) -> CapabilitySpec {
        CapabilitySpec {
            name: name.into(),
            class,
            risk_level: RiskLevel::None,
            params_schema: json!({"type": "object"}),
            lease_ttl_s: 1.0,
            overridable: false,
        }
    }

    /// codex 脑广播的工具表和 `brain-native` 的是同一套名字——包括那个**不带前缀**的
    /// `ask_slow_brain`（`project_capabilities` 跳过它，所以它得单独加回来）。
    #[test]
    fn the_advertised_tools_include_the_unprefixed_ask_slow_brain() {
        let (ctx, skipped) = tools_context(&[
            spec("walk", CapabilityClass::Motion),
            spec(ASK_SLOW_BRAIN, CapabilityClass::Query),
        ]);
        assert!(skipped.is_empty(), "{skipped:?}");
        let v: Value = serde_json::from_str(&ctx).expect("additional_context 是 JSON");
        let names: Vec<&str> = v["tools"]
            .as_array()
            .expect("tools 是数组")
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(names, vec![ASK_SLOW_BRAIN, "robot_walk"]);
        // schema 和 `brain-native` 那张表上的一模一样：同一个 `add_ask_slow_brain`。
        let ask = v["tools"]
            .as_array()
            .and_then(|a| a.iter().find(|t| t["name"] == ASK_SLOW_BRAIN))
            .expect("ask_slow_brain 在表上");
        assert_eq!(ask["parameters"]["required"], json!(["q"]));
    }

    /// M7：队列满了丢的是**最老的一条 steer**，`turn/start` / `turn/interrupt` 一条都
    /// 不丢，而且剩下的顺序原样保留（乱序是这个泵存在的全部理由）。
    #[tokio::test]
    async fn a_full_outbound_queue_drops_the_oldest_steer_and_nothing_else() {
        let q = OutQueue::default();
        let steer = |n: u64| Out::Steer {
            turn_no: n,
            trace_id: format!("t{n}"),
            text: String::new(),
        };
        let interrupt = |n: u64| Out::Interrupt {
            turn_no: n,
            trace_id: format!("t{n}"),
        };
        // 一条 interrupt 在最前面，后面全是 steer，正好填满。
        assert_eq!(q.push(interrupt(0)).ok(), Some(false));
        for n in 1..MAX_OUTBOUND as u64 {
            assert_eq!(q.push(steer(n)).ok(), Some(false));
        }
        // 再来一条：丢掉最老的**steer**（1 号），不是队头那条 interrupt。
        assert_eq!(q.push(steer(999)).ok(), Some(true));
        let mut turns = Vec::new();
        while let Some(o) = q.recv().now_or_never().flatten() {
            turns.push(match o {
                Out::Start { turn_no, .. } => turn_no,
                Out::Steer { turn_no, .. } => turn_no,
                Out::Interrupt { turn_no, .. } => turn_no,
            });
        }
        assert_eq!(turns.len(), MAX_OUTBOUND);
        assert_eq!(turns[0], 0, "队头的 interrupt 还在");
        assert_eq!(turns[1], 2, "被丢掉的是最老的那条 steer（1 号）");
        assert_eq!(turns[MAX_OUTBOUND - 1], 999, "新来的排在最后，顺序不变");

        // 关掉之后一律 `Err`：调用方据此立刻给终态，而不是排一条发不出去的消息。
        q.close();
        assert!(q.push(steer(1)).is_err());
        assert!(q.recv().now_or_never().flatten().is_none());
    }

    /// 广播了就得受理：两种工具名都能映回能力名，别的都不行。
    #[test]
    fn every_advertised_tool_name_maps_back_to_a_capability() {
        assert_eq!(capability_of("robot_walk"), Some("walk"));
        assert_eq!(capability_of(ASK_SLOW_BRAIN), Some(ASK_SLOW_BRAIN));
        assert_eq!(capability_of("walk"), None);
        assert_eq!(capability_of("ask_fast_brain"), None);
    }
}
