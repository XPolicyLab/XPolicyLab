use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use oeh_proto::{EventDraft, EventKind, WorldEvent};
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;

use crate::clock::Clock;
use crate::observation::{build_observation, Observation, ObservationBudget, WakeReason};
use crate::plugin::Disposer;
use crate::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use crate::session_log::{SessionEntry, SessionLog};
use crate::state::StateRegistry;
use crate::wake::{BrainCursor, Decision, WakeAction, WakePolicy};
use crate::world_log::WorldLog;

/// `Brains` 服务的运行参数。
#[derive(Clone, Debug, PartialEq)]
pub struct BrainsConfig {
    pub obs_budget: ObservationBudget,
    /// Running 状态下超过这么久还没 `done` 的 turn 会被强制结束并记 `brain/error`
    /// （§11：脑 Provider 异常 → 终止当前 turn，记 `brain/error`）。0 = 关闭。
    /// `Maintenance` 不受此约束。
    pub turn_timeout_ms: u64,
}
fn d_turn_timeout_ms() -> u64 {
    30_000
}
impl Default for BrainsConfig {
    fn default() -> Self {
        Self {
            obs_budget: ObservationBudget::default(),
            turn_timeout_ms: d_turn_timeout_ms(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrainStatus {
    Idle,
    Running,
    Maintenance,
}

/// 脑 Provider 实现的接口。回调可能来自路由任务、超时任务或调用 `done()` 的任务；
/// `Attached` 的每次变更都在 brains 锁下并受 epoch/turn/status 守护；回调必须非阻塞。
pub trait BrainSink: Send + Sync {
    fn on_turn(&self, obs: Observation, turn: TurnHandle);
    fn steer(&self, obs: Observation);
    fn interrupt(&self);
}

#[derive(Debug, thiserror::Error)]
pub enum BrainsError {
    #[error("brain `{0}` already attached")]
    AlreadyAttached(String),
    #[error("invalid brain id `{0}`")]
    InvalidBrainId(String),
    /// 调用方把参数写坏了（空的 / 超限的注入文本）。边界上映射成 `INVALID_PARAMS`。
    #[error("invalid params: {0}")]
    InvalidParams(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("Brains must be constructed inside a tokio runtime")]
    NoRuntime,
    #[error("brain `{0}` is not attached")]
    NotAttached(String),
}

/// 每次 attach 一个新 epoch：detach 之后再 attach 同一个 brain_id 时，旧 attach 发出去的
/// `TurnHandle` 的 turn 号会和新 attach 的撞上，epoch 让陈旧句柄认不出新的那个。
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// 路由队列容量。满了就丢事件并置 `catch_up`：世界日志才是真相，路由任务会用
/// `log.since(cursor)` 把漏掉的补回来，绝不让 append 侧阻塞。
const ROUTE_QUEUE: usize = 1024;

/// [`Brains::flush_sessions`] 的总预算，**在所有已挂载的脑之间共享**。关机路径调它，
/// 所以它必须有界：一块卡住的盘不能让 `oeh run` 停不下来。
const SESSION_FLUSH_BUDGET: Duration = Duration::from_secs(1);

struct Attached {
    epoch: u64,
    policy: WakePolicy,
    cursor: BrainCursor,
    /// 这个脑迄今为止**真正被 `route_one` 看过、做过一次路由决策**的最大 seq
    /// （不管那次决策是不是 `Ignore`/`Accumulate`）。`observe_for` 用它给一份观测的
    /// `since(cursor)` 结果封顶：世界日志的**尾巴上**此刻可能正摆着一截更晚、这个脑还
    /// 没轮到的事件（trailing un-routed events——还排在路由队列里，或者正轮到别的脑），
    /// 不能因为顺手扫到它们就把游标一并带过去，把它们各自的路由决策悄悄吞掉（C2）。
    /// 不变式：`routed_seq >= cursor.cursor`（游标只在 `observe_for` 里推进，而那次推进
    /// 封顶在 `routed_seq` 上；游标被重置时 `routed_seq` 跟着抬上去）。
    routed_seq: u64,
    status: BrainStatus,
    turn: u64,
    /// latch 住的唤醒 + latch 当时的游标。turn 结束时若游标已经越过 `reason.seq`，
    /// 这条证据早被某次 steer 消费掉了，再开 turn 只会拿到一个不含它的观测。
    pending_wake: Option<(WakeReason, u64)>,
    /// 当前 turn 的 trace_id（§12：`act/*` 与 `brain/*` 事件靠它串起来）。
    current_trace: Option<String>,
    /// 当前 turn 进入 Running 的内核时刻；超时扫描用。Maintenance 期间不看它。
    turn_started_ns: Option<u64>,
    /// 已经记过 `brain/wake{suppressed:true}` 的最大 seq。被抑制的事件不推进游标，
    /// 队列溢出后的 catch-up 会把整段 `since(cursor)` 再放一遍——没有这个水位线，
    /// 同一条事件每补齐一次就多一条抑制记录。
    last_suppressed_seq: u64,
    /// 本 turn 已经提过几次提案。提案 id 用 `{trace_id}-p{n}`，trace_id 每 turn 一换，
    /// 所以计数器每次开 turn 归零。
    proposal_counter: u64,
    sink: Arc<dyn BrainSink>,
    session: Arc<SessionLog>,
}

/// `Brains` 的共享状态。`TurnHandle` 只持 `Weak<Inner>`，`session_for_handle` 这类
/// 从句柄侧进来的路径需要拿到它，所以对 crate 内可见。
pub(crate) struct Inner {
    brains: Mutex<HashMap<String, Attached>>,
    /// 构造时捕获的 runtime 句柄。`propose` 要在这上面 spawn 记结局的那一截——脑可能
    /// 从任何线程调它，不能指望调用处正好在 runtime 上下文里。
    rt: tokio::runtime::Handle,
    clock: Arc<dyn Clock>,
    log: Arc<WorldLog>,
    state: Arc<StateRegistry>,
    session_dir: PathBuf,
    cfg: BrainsConfig,
    /// 准入服务。同一时刻至多一个；后设者覆盖前一个。
    proposer: Mutex<Option<Arc<dyn Proposer>>>,
    /// 队列溢出过：下一次路由前先按每个脑的游标补齐。
    catch_up: AtomicBool,
    dropped: AtomicU64,
    /// `settle` 用的水位：入队计数与已路由计数（单消费者 FIFO，两者一一对应）。
    enqueued: AtomicU64,
    routed: AtomicU64,
    notify: Notify,
}

/// 把世界日志按每个脑的游标/策略/预算变成 turn。
pub struct Brains {
    inner: Arc<Inner>,
    observer: Mutex<Option<Disposer>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl Drop for Brains {
    /// 先停路由任务再撤观察者：`Disposer` 是 `FnOnce`，光 drop 它不会执行，留着会在
    /// `WorldLog` 的观察者表里堆死项。
    fn drop(&mut self) {
        for t in self.tasks.lock().drain(..) {
            t.abort();
        }
        if let Some(d) = self.observer.lock().take() {
            d();
        }
    }
}

/// 一个进行中的 turn；`done` 结束它。
///
/// `Clone`：跨进程的脑（Plan 3 的 RPC 边界）把句柄存在一张 `turn → TurnHandle` 的表里，
/// `turn/propose` 要在**不持**那张表的锁的情况下 `.await` 一次提案，只能先克隆一份出来。
/// 克隆不放大任何权力：`propose`/`record` 走 `is_current()` 与 epoch+turn 双重检查，
/// `done` 走 `finish_turn` 的 epoch+turn+status 检查——两个克隆各调一次 `done`，第二次是
/// 空操作，不会凭空多出一条 `brain/turn_end`。
#[derive(Clone)]
pub struct TurnHandle {
    pub brain_id: String,
    pub turn: u64,
    /// 本 turn 的全部 `brain/*` 事件都带这个 id；脑产出的 `act/*` 也应带上（Plan 2b）。
    pub trace_id: String,
    epoch: u64,
    brains: Weak<Inner>,
}

impl TurnHandle {
    /// 一个不连着任何 `Brains` 的句柄，给下游 crate 的**测试**用（脑侧的工具、批执行
    /// 这类代码要一个 `TurnHandle` 才能调用，但并不关心提案的结局）。
    ///
    /// 它持的 `Weak` 永远 upgrade 不了，所以 `propose` 立刻还
    /// `Outcome::Refused { reason: "brains gone" }`，`record`/`done`/`is_current` 都是
    /// 空操作——和"`Brains` 已经被撤销之后还拿着的旧句柄"是同一条路径，不是新语义。
    #[doc(hidden)]
    pub fn detached_for_tests(brain_id: &str) -> TurnHandle {
        TurnHandle {
            brain_id: brain_id.to_string(),
            turn: 0,
            trace_id: format!("{brain_id}-detached"),
            epoch: 0,
            brains: Weak::new(),
        }
    }

    pub fn done(self, reason: &str) {
        if let Some(inner) = self.brains.upgrade() {
            let _ = Brains::finish_turn(&inner, &self.brain_id, self.turn, self.epoch, reason);
        }
    }
    /// 把一次提案交给准入服务。内核在这里只做三件事：挡住陈旧 turn 和没装服务的情况、
    /// 给提案编号并记一条 `SessionEntry::Proposal`、在结局回来之后补一条
    /// `SessionEntry::Outcome`。准不准、怎么执行都是准入服务的事。
    ///
    /// `say` / `ask_human` 没有旁路。它们和别的能力一样是提案（规格 §5.1：脑的三种产出
    /// 走同一条链），会话日志里的 `Say` / `AskHuman` 只是这次提案的副产品——曾经有一对
    /// `record_say` / `record_ask_human` 直接往会话日志里写，那是一条绕过能力注册表、
    /// 规则链和 `act/*` 的暗道，`turn/say` 这类边界方法一旦按名字接上去就会默认走它。
    ///
    /// 返回的 future 不持有 brains 锁，也不持有 `Inner` 的强引用：脑可以把它拿去和别的
    /// future select，`Brains` 在等待期间被撤销也只是让结局写不进会话日志而已。
    pub fn propose(&self, capability: &str, params: Value) -> BoxFuture<Outcome> {
        self.propose_with_id(capability, params).1
    }

    /// [`TurnHandle::propose`]，外加这次提案的 [`ProposalId`]——`propose` 只还结局，而
    /// 结局里没有 id。跨进程的脑（`turn/propose`）必须在**提案发出的那一刻**就拿到它：
    /// 调用方要能把自己这次调用和世界日志里的 `act/*`（都带 `proposal_id`）对上号，而不
    /// 是等结局回来之后去猜是哪一条。
    ///
    /// 没能真的提出去（brains 已撤销、turn 陈旧、没装准入服务）时 id 是
    /// `{trace_id}-p0`：`proposal_counter` 从 1 起编号，`p0` 是"这次提案根本没被编号"的
    /// 哨兵，日志里永远不会出现同名的一条。
    pub fn propose_with_id(
        &self,
        capability: &str,
        params: Value,
    ) -> (ProposalId, BoxFuture<Outcome>) {
        let unnumbered = ProposalId(format!("{}-p0", self.trace_id));
        let refused = |reason: &str| -> (ProposalId, BoxFuture<Outcome>) {
            let reason = reason.to_string();
            (
                unnumbered.clone(),
                Box::pin(async move { Outcome::Refused { reason } }) as BoxFuture<Outcome>,
            )
        };
        let Some(inner) = self.brains.upgrade() else {
            return refused("brains gone");
        };
        if !self.is_current() {
            return refused("turn is not current");
        }
        let proposer = inner.proposer.lock().clone();
        let Some(proposer) = proposer else {
            return refused("no admission service");
        };
        // 编号在锁里取，提案本身在锁外记：`record` 会再拿这把锁（`parking_lot` 不可重入），
        // 而且会写盘。
        let n = {
            let mut g = inner.brains.lock();
            match g.get_mut(&self.brain_id) {
                Some(a) if a.epoch == self.epoch && a.turn == self.turn => {
                    a.proposal_counter += 1;
                    a.proposal_counter
                }
                _ => return refused("turn is not current"),
            }
        };
        let proposal_id = ProposalId(format!("{}-p{n}", self.trace_id));
        let p = Proposal {
            proposal_id: proposal_id.clone(),
            brain_id: self.brain_id.clone(),
            turn: self.turn,
            trace_id: self.trace_id.clone(),
            capability: capability.to_string(),
            params: params.clone(),
        };
        self.record(SessionEntry::Proposal {
            turn: self.turn,
            capability: capability.to_string(),
            params: params.clone(),
            trace_id: self.trace_id.clone(),
        });
        // 会话日志是"这个脑说过什么、问过什么"的那一份记录，`Proposal` 只记得住
        // "它提了一次 say"。两条都记：读会话的人要的是对话，读 `Proposal`/`Outcome`
        // 的人要的是准入。
        match capability {
            "say" => self.record(SessionEntry::Say {
                turn: self.turn,
                text: str_param(&params, "text"),
            }),
            "ask_human" => self.record(SessionEntry::AskHuman {
                turn: self.turn,
                question: str_param(&params, "question"),
            }),
            _ => {}
        }
        let fut = proposer.propose(p.clone());
        let weak = self.brains.clone();
        let (bid, epoch, turn, pid) = (
            self.brain_id.clone(),
            self.epoch,
            self.turn,
            p.proposal_id.0.clone(),
        );
        // 记结局的这一截跑在自己的任务上：脑把返回的 future 丢掉（Plan 3 的 RPC 客户端
        // 断线、Plan 4 的脑任务被 abort，都是这个形状）之后，会话日志里不能留下一条
        // 永远没有下文的 `Proposal`——读的人无从知道那次提案到底做没做。调用方照常
        // 从返回的 future 里拿到同一个结局。
        let task = inner.rt.spawn(async move {
            let out = fut.await;
            if let Some(inner) = weak.upgrade() {
                // epoch 级而不是 epoch+turn 级：提案在途时这个 turn 完全可能被打断
                // （`safety/estop` 之类），结局回来时 brains 已经在跑下一个 turn 了。
                // 条目自带 `turn`/`proposal_id`，落在哪个时刻都对得上号；漏掉它才是
                // 真正的审计空洞——日志里会留下一条永远没有结局的 `Proposal`。
                Brains::session_for_epoch(
                    &inner,
                    &bid,
                    epoch,
                    SessionEntry::Outcome {
                        turn,
                        proposal_id: pid,
                        outcome: out.clone(),
                    },
                );
            }
            out
        });
        (
            proposal_id,
            Box::pin(async move {
                match task.await {
                    Ok(out) => out,
                    // 任务被 abort（runtime 关停）：结局无从得知，但提案方总得拿到点什么。
                    Err(_) => Outcome::Cancelled {
                        reason: "proposal task aborted".into(),
                    },
                }
            }),
        )
    }

    /// 脑这一侧的记账：模型这一步说了什么、调了哪些工具、工具回了什么。
    ///
    /// 只接受 [`SessionEntry::ModelOutput`] / [`SessionEntry::ToolCall`] /
    /// [`SessionEntry::ToolResult`] 三种；别的变体返回 `Err` 而不是悄悄写进去。会话日志
    /// 里 `Attached`/`TurnStart`/`Observation`/`TurnEnd` 是**内核**的记账，`Proposal`/
    /// `Outcome`/`Say`/`AskHuman` 是 [`TurnHandle::propose`] 的副产品——两类都有各自的
    /// 不变式（epoch 级 vs. 句柄级、提案编号与结局配对）。开一个让脑自由写任意条目的口子
    /// 等于把那些不变式交给脑去维护，而脑是这个工作区里最外层的扩展点。
    ///
    /// `turn` 一律**改写**成本句柄的 turn：调用方不需要（也不应该）自己填对，填错的后果
    /// 是一条挂在别的 turn 名下、读日志的人分辨不出来的记录。
    ///
    /// 非阻塞：底层 [`SessionLog::record`] 只入队，不碰文件。陈旧句柄（turn 已被
    /// interrupt / 超时关掉）照样返回 `Ok`——它走的是和 [`TurnHandle::record`] 同一条
    /// epoch+turn 双检查路径，写不进去只是没有落盘，不是调用错误。
    pub fn note(&self, entry: SessionEntry) -> Result<(), &'static str> {
        let turn = self.turn;
        let entry = match entry {
            SessionEntry::ModelOutput { step, text, .. } => {
                SessionEntry::ModelOutput { turn, step, text }
            }
            SessionEntry::ToolCall {
                step,
                call_id,
                name,
                args,
                ..
            } => SessionEntry::ToolCall {
                turn,
                step,
                call_id,
                name,
                args,
            },
            SessionEntry::ToolResult {
                step,
                call_id,
                ok,
                result,
                ..
            } => SessionEntry::ToolResult {
                turn,
                step,
                call_id,
                ok,
                result,
            },
            _ => return Err("note accepts only ModelOutput / ToolCall / ToolResult"),
        };
        self.record(entry);
        Ok(())
    }

    /// 陈旧句柄不能往会话日志里写：turn 已经被 interrupt / 超时关掉之后，它记下的 say
    /// 会挂在**下一个** turn 的编号下面，读日志的人无从分辨。epoch 和 turn 都要对上。
    fn record(&self, e: SessionEntry) {
        if let Some(inner) = self.brains.upgrade() {
            Brains::session_for_handle(&inner, &self.brain_id, self.epoch, self.turn, e);
        }
    }

    /// 这个句柄是否还指向该脑**当前**进行中的 turn（没被 interrupt / 超时 / detach 掉）。
    pub fn is_current(&self) -> bool {
        self.brains.upgrade().is_some_and(|inner| {
            inner.brains.lock().get(&self.brain_id).is_some_and(|a| {
                a.epoch == self.epoch && a.turn == self.turn && a.status != BrainStatus::Idle
            })
        })
    }

    /// 进入 `Maintenance`（压缩、记忆写入、子任务收尾，§7.1）：对唤醒来说与 Running 等价
    /// （只 latch，不重复开 turn），但不受 turn 超时约束。
    pub fn enter_maintenance(&self) {
        self.set_status(BrainStatus::Running, BrainStatus::Maintenance);
    }

    /// 回到 `Running`。超时时钟从这一刻重新起算：Maintenance 期间的耗时不算进 turn 超时。
    pub fn leave_maintenance(&self) {
        self.set_status(BrainStatus::Maintenance, BrainStatus::Running);
    }

    fn set_status(&self, from: BrainStatus, to: BrainStatus) {
        let Some(inner) = self.brains.upgrade() else {
            return;
        };
        let now = inner.clock.now_ns();
        let mut g = inner.brains.lock();
        let Some(a) = g.get_mut(&self.brain_id) else {
            return;
        };
        if a.epoch == self.epoch && a.turn == self.turn && a.status == from {
            a.status = to;
            a.turn_started_ns = (to == BrainStatus::Running).then_some(now);
        }
    }
}

/// `params` 里的一个字符串字段。不是字符串（能力的 schema 会在准入那边挡下来）就记空串：
/// 会话日志是记账，不是校验的地方。
fn str_param(params: &Value, field: &str) -> String {
    params
        .get(field)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

/// `route` 在锁内决定要做什么，出锁之后才执行——会话日志落盘和 sink 回调都可能反过来调
/// `TurnHandle`（`done` 会再拿这把锁），而 `parking_lot::Mutex` 不可重入。
enum Plan {
    Nothing,
    Suppressed,
    StartTurn(WakeReason),
    Steer(WakeReason),
    Interrupt(WakeReason),
}

/// 内核为脑写的 bookkeeping 事件：永远不路由给**任何**脑。它们就是路由任务自己产生的，
/// 再喂回 `route` 只会绕着队列打转——两个都不 `ignore` `brain/*` 又都超预算的脑，会靠
/// `Plan::Suppressed` 的 `brain/wake` 无限对敲。脑自己写的 `brain/*`（例如
/// `brain/mock_turn`、`brain/ask`）不在此列：别的脑本来就该看见它们。
const BOOKKEEPING: &[&str] = &[
    "brain/attach",
    "brain/cursor",
    "brain/detach",
    "brain/turn_end",
    "brain/turn_start",
    "brain/wake",
];

/// `brain_id` 会被拼进会话日志的文件名；路径分隔符和 `..` 会让它逃出 session_dir。
fn valid_brain_id(id: &str) -> bool {
    !id.is_empty() && !id.contains('/') && !id.contains('\\') && !id.contains("..")
}

fn kind(s: &str) -> EventKind {
    EventKind::new(s).expect("static kind")
}

/// 一条 `brain/inject` 的 `text` 上限（4 KiB）。
///
/// 注入是**边界**方法（控制台、运维脚本、另一个进程）喂进来的一段任意文本，而世界日志的
/// payload 本身没有上限：没有这道闸，一次 `brain/inject` 就能往日志里塞进一个 8 MB 的
/// 字符串，并且此后这个 turn 的**每一个 step** 都要为它重新付一次 token（收件箱条目会
/// 一直留到被认领，见 `oeh_brain::native::prompt`）。4 KiB 是"人能说出来的一句话"的
/// 量级——真要传一大坨材料，那是一次提案或者一条流，不是一句捎带进来的话。
pub const MAX_INJECT_TEXT: usize = 4096;

/// 一条 `brain/inject` 的 salience。和 `act/admitted` / `brain/ask` 同级（150）：有人对
/// 这个脑说了句话是要被看见的事，但它不是安全事件（`brain/steer` 的 200 是"现在就改主意"，
/// 注入没有那么急）。
const SAL_INJECT: u8 = 150;

/// `brain/steer` / `brain/interrupt` / `brain/inject` 的定向分类，只用来绕过 `BrainCursor::decide`
/// （policy / 预算 / suppressed）——分类出来的 `WakeAction` 之后照样走既有的
/// `(action, status)` 矩阵和既有的 `Plan::StartTurn`/`Steer`/`Interrupt` 执行器，
/// 不会有第二套语义。
///
/// `Some(Steer|Interrupt)` 当 `ev` 点名的正是 `id`，`Some(Ignore)` 当它点了别的脑，
/// `None` 当 `ev` 根本不是这三种事件之一，或者 `source` 不是 `"brains"`——只有
/// `Brains::steer`/`interrupt`/`inject` 写的这三种事件才算定向，任何别的来源伪造的
/// `brain/steer`/`brain/interrupt`/`brain/inject` 都不是定向事件（不然任何能写世界日志的
/// 东西——包括 `event/emit` 这样的边界方法——都能绕开目标脑的 policy 直接操纵它）。
///
/// `brain/inject` 和 `brain/steer` 落在**同一格**上，不是新语义：两者都是"有话要对这个
/// 脑说"，区别只在脑那一侧怎么读（原生脑把 `wake_reason.kind == "brain/inject"` 的观测
/// 投进 `Lane::Inject` 而不是 `Lane::Steer`）。内核这边分两种 kind 只是为了让那一侧分得
/// 出来，路由、预算、(action, status) 矩阵一模一样。
///
/// 返回 `None` 之后这条事件的下场分两种，别记成"一律落回目标脑的 `WakePolicy`"：
/// - **被点名的那个脑**（`payload.brain_id == id`）根本看不到它——`route_one` 的防自激
///   守卫（`directed.is_none() && kind.prefix() == "brain" && payload.brain_id == id`）
///   在判定之前就 return 了，policy 里即便写着 `brain/steer: wake` 也不会醒；
/// - **没被点名的脑**才真的落回自己的 `WakePolicy`：`brain/steer` 对它们来说只是一条
///   普通的 `brain/*` 事件。
pub(crate) fn directed_action(ev: &WorldEvent, id: &str) -> Option<WakeAction> {
    if ev.source != "brains" {
        return None;
    }
    let is_steer = matches!(ev.kind.as_str(), "brain/steer" | "brain/inject");
    let is_interrupt = ev.kind.as_str() == "brain/interrupt";
    if !is_steer && !is_interrupt {
        return None;
    }
    let targeted = ev.payload.get("brain_id").and_then(|v| v.as_str()) == Some(id);
    Some(match (targeted, is_steer) {
        (true, true) => WakeAction::Steer,
        (true, false) => WakeAction::Interrupt,
        (false, _) => WakeAction::Ignore,
    })
}

impl Brains {
    /// **必须在 tokio runtime 内调用**：`Brains` 捕获当前 `Handle` 并在上面 spawn 一个专属
    /// 路由任务。世界日志的观察者（跑在 append/ingest 线程上，500 Hz）只做一次 `try_send`，
    /// 所有 `append`、会话日志写盘和 sink 回调都发生在那个任务上。runtime 外构造返回
    /// `NoRuntime` 而不是 panic：插件的 `apply` 里 panic 会穿过整棵 fiber 树。
    pub fn new(
        clock: Arc<dyn Clock>,
        log: Arc<WorldLog>,
        state: Arc<StateRegistry>,
        session_dir: PathBuf,
        cfg: BrainsConfig,
    ) -> Result<Self, BrainsError> {
        let rt = tokio::runtime::Handle::try_current().map_err(|_| BrainsError::NoRuntime)?;
        let turn_timeout_ms = cfg.turn_timeout_ms;
        let inner = Arc::new(Inner {
            brains: Mutex::new(HashMap::new()),
            rt: rt.clone(),
            clock,
            log: log.clone(),
            state,
            session_dir,
            cfg,
            proposer: Mutex::new(None),
            catch_up: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
            enqueued: AtomicU64::new(0),
            routed: AtomicU64::new(0),
            notify: Notify::new(),
        });
        let (tx, mut rx) = mpsc::channel::<Arc<WorldEvent>>(ROUTE_QUEUE);
        // 路由任务持 Weak：`Brains` 被 drop（服务被撤销）之后不能再靠它把 Inner 吊着。
        let weak = Arc::downgrade(&inner);
        let task = rt.spawn(async move {
            while let Some(ev) = rx.recv().await {
                let Some(i) = weak.upgrade() else { break };
                if i.catch_up.swap(false, Ordering::SeqCst) {
                    Self::catch_up(&i);
                }
                Self::route(&i, &ev);
                i.routed.fetch_add(1, Ordering::SeqCst);
                i.notify.notify_waiters();
            }
        });
        let weak_obs = Arc::downgrade(&inner);
        let observer = log.observe(Arc::new(move |ev: &WorldEvent| {
            let Some(i) = weak_obs.upgrade() else { return };
            match tx.try_send(Arc::new(ev.clone())) {
                Ok(()) => {
                    i.enqueued.fetch_add(1, Ordering::SeqCst);
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // 丢事件但不丢信息：游标还停在原处，catch-up 会用 `since` 补回来。
                    i.catch_up.store(true, Ordering::SeqCst);
                    i.dropped.fetch_add(1, Ordering::SeqCst);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {}
            }
        }));
        let mut tasks = vec![task];
        if turn_timeout_ms > 0 {
            // 独立的计时任务：路由任务只在有事件时才跑，卡死的 turn 正好是“没事件”的那种。
            let period = Duration::from_millis((turn_timeout_ms / 4).clamp(5, 1_000));
            let weak_t = Arc::downgrade(&inner);
            tasks.push(rt.spawn(async move {
                let mut iv = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
                iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    iv.tick().await;
                    let Some(i) = weak_t.upgrade() else { break };
                    Self::sweep_timeouts(&i);
                }
            }));
        }
        Ok(Self {
            inner,
            observer: Mutex::new(Some(observer)),
            tasks: Mutex::new(tasks),
        })
    }

    /// 测试用：同步跑一次超时扫描（生产路径由定时任务调用同一函数）。
    #[doc(hidden)]
    pub fn sweep_now(&self) {
        Self::sweep_timeouts(&self.inner);
    }

    /// 装上准入服务。同一时刻至多一个：后设者直接覆盖前一个。返回的 `Disposer` 只在
    /// 当前装着的仍是**自己**那一个时才清除——否则重复 dispose 会把后来者拆掉。
    pub fn set_proposer(&self, p: Arc<dyn Proposer>) -> Disposer {
        *self.inner.proposer.lock() = Some(p.clone());
        let inner = self.inner.clone();
        Box::new(move || {
            let mut g = inner.proposer.lock();
            if g.as_ref().is_some_and(|cur| Arc::ptr_eq(cur, &p)) {
                *g = None;
            }
        })
    }

    /// 强制结束跑过头的 turn：记 `brain/error{reason:"turn_timeout"}`，会话日志记
    /// `TurnEnd{reason:"timeout"}`，状态回 Idle。`Maintenance` 豁免。
    fn sweep_timeouts(inner: &Arc<Inner>) {
        let timeout_ns = inner.cfg.turn_timeout_ms.saturating_mul(1_000_000);
        if timeout_ns == 0 {
            return;
        }
        let now = inner.clock.now_ns();
        let expired: Vec<(String, u64, u64, Option<String>)> = inner
            .brains
            .lock()
            .iter()
            .filter(|(_, a)| a.status == BrainStatus::Running)
            .filter_map(|(id, a)| {
                let started = a.turn_started_ns?;
                (now.saturating_sub(started) >= timeout_ns)
                    .then(|| (id.clone(), a.turn, a.epoch, a.current_trace.clone()))
            })
            .collect();
        for (id, turn, epoch, trace) in expired {
            // 先关再记：快照和这里之间 `done()`/interrupt 可能已经把这个 turn 收了尾，
            // 那时 `finish_turn` 是空操作，一条 `brain/error` 会把正常收尾诬告成脑异常。
            if !Self::finish_turn(inner, &id, turn, epoch, "timeout") {
                continue;
            }
            tracing::warn!(brain = %id, turn, "turn timed out; forcing it closed");
            Self::append_brain(
                inner,
                "brain/error",
                json!({"brain_id": id, "turn": turn, "reason": "turn_timeout"}),
                trace.as_deref(),
                200,
            );
        }
    }

    /// 带 trace_id 的 bookkeeping 事件（§12：`brain/*` 靠 trace_id 串起一个 turn）。
    fn append_brain(
        inner: &Arc<Inner>,
        k: &str,
        payload: Value,
        trace: Option<&str>,
        salience: u8,
    ) {
        let mut d = EventDraft::new(kind(k), "brains", payload).with_salience(salience);
        if let Some(t) = trace {
            d = d.with_trace_id(t);
        }
        inner.log.append(d);
    }

    /// 此刻为止入队的事件全部路由完毕后返回（路由过程中派生出来的事件也一并等到）。
    /// 路由是异步的，测试与关机前的静默点用它取代 sleep。
    pub async fn settle(&self) {
        loop {
            let target = self.inner.enqueued.load(Ordering::SeqCst);
            loop {
                // `notified()` 要先 `enable()` 注册再检查条件，否则 notify_waiters 会在
                // “检查完、还没注册”的窗口里丢掉唤醒。
                let notified = self.inner.notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.inner.routed.load(Ordering::SeqCst) >= target {
                    break;
                }
                notified.await;
            }
            if self.inner.enqueued.load(Ordering::SeqCst) == target {
                return;
            }
        }
    }

    /// 路由队列溢出而被丢弃的事件数（丢弃后靠 catch-up 补齐，这里只用于观测背压）。
    pub fn dropped_events(&self) -> u64 {
        self.inner.dropped.load(Ordering::SeqCst)
    }

    /// 测试用：拿到某个脑的会话日志句柄。`#[doc(hidden)]`——生产路径没有理由绕过
    /// `record`/`flush_sessions` 直接动它。**注意它是一个强引用**：拿着它的时候 detach
    /// 不会析构这份日志。
    #[doc(hidden)]
    pub fn session_log_for_tests(&self, brain_id: &str) -> Option<Arc<SessionLog>> {
        self.inner
            .brains
            .lock()
            .get(brain_id)
            .map(|a| a.session.clone())
    }

    /// 把所有已挂载的脑的会话日志排空到盘上，再回来。**阻塞**，总预算 1 s
    /// （[`SESSION_FLUSH_BUDGET`]，在所有脑之间**共享**，不是每个脑一秒）。
    ///
    /// 两个调用者：
    ///
    /// 1. **关机**（[`crate::Kernel::run`]，`settle()` 之后、`dispose_all` 之前）。撤销时
    ///    `SessionLog::drop` 只在还剩最后一个强引用时才排空，而关机那一刻 `Brains` 的
    ///    `Arc` 很可能还被别人攥着（RPC 服务器、还没归还的句柄）——那样最后一段 turn /
    ///    提案 / 结局的记账就留在队列里，随进程一起没了。这一步把"有序关机之后会话日志
    ///    是完整的"变成一条不依赖引用计数的保证。
    /// 2. **读日志的测试**：`record` 从 Plan 4 起只入队，"刚记过的那条已经在文件里"不再
    ///    自动成立。
    ///
    /// 写线程真的卡住时它会吃满预算然后放弃（每份日志只 warn 一句）——有界是硬要求：
    /// 关机路径不能因为一块卡住的盘而永远回不来。
    pub fn flush_sessions(&self) {
        // 句柄在锁里取、flush 在锁外做：`flush` 会等写线程，那段时间里路由任务不能被
        // 这把锁挡住。
        let logs: Vec<Arc<SessionLog>> = self
            .inner
            .brains
            .lock()
            .values()
            .map(|a| a.session.clone())
            .collect();
        let deadline = std::time::Instant::now() + SESSION_FLUSH_BUDGET;
        for l in logs {
            // 预算是**共享**的：n 个脑不能变成 n 秒。用光了还是各调一次（剩余预算为 0
            // 时 `flush_within` 只做一次不等待的尝试），这样队列本来就空的那些仍然能
            // 干净收尾。
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if let Err(err) = l.flush_within(left) {
                tracing::warn!(path = %l.path().display(), %err, "session log flush failed");
            }
        }
    }

    pub fn attach(
        &self,
        brain_id: &str,
        policy: WakePolicy,
        sink: Arc<dyn BrainSink>,
    ) -> Result<Disposer, BrainsError> {
        if !valid_brain_id(brain_id) {
            return Err(BrainsError::InvalidBrainId(brain_id.to_string()));
        }
        // 重名的快速预检，在开文件**之前**。两件事都靠它：
        //
        // - **错误优先级**：`InvalidBrainId` → `AlreadyAttached` → `Io`。少了这一步，对一个
        //   已经挂着的脑重复 attach、而此刻会话目录恰好不可写，返回的会是 `Io`（RPC 侧
        //   `INTERNAL`）而不是 `AlreadyAttached`（RPC 侧 `REFUSED`）——客户端对"策略性
        //   拒绝"和"服务器内部错误"的反应完全不同。
        // - **被拒的请求不该比被接受的贵**：`brain/attach` 是一条客户端可达的 RPC，重复
        //   调用只该是一次 `HashMap` 查表，不是 `create_dir_private` + `open` +
        //   `thread::spawn` 再加一次析构时的 flush 往返。
        //
        // 权威判定仍然是下面锁里的那次（这中间可能有人抢先 attach）；这里只是不让最常见
        // 的那条被拒路径付钱。
        if self.inner.brains.lock().contains_key(brain_id) {
            return Err(BrainsError::AlreadyAttached(brain_id.into()));
        }
        // 会话日志在**锁外**打开：`SessionLog::open` 要建目录、开文件、再起一条写线程
        // （OS 线程创建是几十 µs），而 `inner.brains` 是路由任务每条事件都要拿的那把锁。
        // 输给了并发 attach 时这份刚开的日志在函数返回时析构——那也在锁外（`g` 是内层
        // 作用域的局部，先于它释放）。文件本身是 append 打开的，白开一次不会动到已经挂着
        // 的那个脑的内容。
        let session = Arc::new(SessionLog::open(&self.inner.session_dir, brain_id)?);
        let (start, epoch, previous) = {
            let mut g = self.inner.brains.lock();
            if g.contains_key(brain_id) {
                return Err(BrainsError::AlreadyAttached(brain_id.into()));
            }
            // 挂载点之前的事件不属于这个脑：游标从“当前最后一条”起。
            let start = self.inner.log.next_seq().saturating_sub(1);
            let epoch = NEXT_EPOCH.fetch_add(1, Ordering::SeqCst);
            // 返回值绑住再出锁析构：`insert` 的返回值在分号处 drop，而一个被顶掉的
            // `Attached` 里是 `Arc<SessionLog>`——它的 `Drop` 会等写线程。上面的
            // `contains_key` 保证这里永远是 `None`，绑定是为了让"锁里不 drop
            // `Attached`"这条纪律不依赖于那个保证。
            let previous = g.insert(
                brain_id.to_string(),
                Attached {
                    epoch,
                    policy,
                    cursor: BrainCursor::new(start),
                    routed_seq: start,
                    status: BrainStatus::Idle,
                    turn: 0,
                    pending_wake: None,
                    current_trace: None,
                    turn_started_ns: None,
                    last_suppressed_seq: 0,
                    proposal_counter: 0,
                    sink,
                    session,
                },
            );
            (start, epoch, previous)
        };
        // 断言在**锁外**求值：万一它真的触发，展开时 `previous`（声明在 `g` 之后）会先于
        // `g` 析构，也就是在锁里跑 `SessionLog::drop`——正是上面那段注释要避免的形状。
        debug_assert!(previous.is_none(), "contains_key already ruled this out");
        drop(previous);
        Self::session_for_epoch(
            &self.inner,
            brain_id,
            epoch,
            SessionEntry::Attached {
                epoch,
                brain_id: brain_id.to_string(),
                t_kernel: self.inner.clock.now_ns(),
            },
        );
        self.inner.log.append(EventDraft::new(
            kind("brain/attach"),
            "brains",
            json!({"brain_id": brain_id, "cursor": start}),
        ));
        let inner = self.inner.clone();
        let id = brain_id.to_string();
        Ok(Box::new(move || {
            // 只摘掉还是这个 epoch 的那一份：重复 dispose 不能误伤后来的 attach。
            //
            // 摘下来的 `Attached` **绑住**，出锁之后才析构。`g.remove(&id);` 作为一条
            // 语句，返回的 `Option<Attached>` 在分号处就 drop 了——而 `Attached` 里是
            // `Arc<SessionLog>`，detach 时通常就是最后一个强引用，于是 `SessionLog::drop`
            // （flush，最长 `FLUSH_BUDGET`）会在 `inner.brains` 这把锁里跑：一块慢盘就能
            // 让所有脑的唤醒/steer/interrupt 停上一秒，而 `brain/detach`、断连清理和
            // `Op::DisposeFiber` 这三条路分别跑在连接的读任务和**内核唯一的 Op 循环**上。
            // 这正是这个任务要消灭的那种阻塞，只是入口从 `record` 换到了 detach。
            //
            // 还在跑的那个 turn 也要有终态。一个 turn 的终态有四条路（`done`、被
            // interrupt、内核的超时扫描、这里），而 detach 是唯一一条从前什么都不记的：
            // 世界日志的回放于是看到一条 `brain/turn_start` 永远没有下文（profile 重载、
            // fiber dispose、RPC 断连都会这样）。`done` 补不上这一条——sink 的 `Drop`
            // 就在这一刻，而 detach 之后的 `done` 按纪律 3 本来就是一次空操作
            // （`finish_turn` 找不到这个脑，`is_current` 也是假）。所以恰好一条，不会重。
            let (gone, removed) = {
                let mut g = inner.brains.lock();
                let mine = g.get(&id).map(|a| a.epoch) == Some(epoch);
                (mine, if mine { g.remove(&id) } else { None })
            };
            let live_turn = removed.as_ref().and_then(|a| {
                (a.status != BrainStatus::Idle).then(|| (a.turn, a.current_trace.clone()))
            });
            drop(removed);
            if gone {
                if let Some((turn, trace)) = live_turn {
                    Self::append_brain(
                        &inner,
                        "brain/turn_end",
                        json!({"brain_id": id, "turn": turn, "reason": "detached"}),
                        trace.as_deref(),
                        0,
                    );
                }
                inner.log.append(EventDraft::new(
                    kind("brain/detach"),
                    "brains",
                    json!({"brain_id": id, "epoch": epoch}),
                ));
            }
        }))
    }

    pub fn status(&self, brain_id: &str) -> Option<BrainStatus> {
        self.inner.brains.lock().get(brain_id).map(|a| a.status)
    }
    pub fn cursor(&self, brain_id: &str) -> Option<u64> {
        self.inner
            .brains
            .lock()
            .get(brain_id)
            .map(|a| a.cursor.cursor)
    }

    /// 定向让某个脑 Steer：append 一条 `brain/steer{brain_id,text}`（source `"brains"`，
    /// salience 200）。`route_one` 只认 `source == "brains"` 的 `brain/steer` 为定向
    /// 事件——也就是**只有这个方法**（不是任何伪造了同样 kind/payload 但换了 source 的
    /// 事件）写出来的那条——才会绕过 `brain_id` 指定的那个脑的 policy 和唤醒预算，直接
    /// 按 `Steer` 处理（Idle 时等同 wake，开一个新 turn；Running/Maintenance 时投进当前
    /// turn）；别的脑当普通 Ignore 处理，看不见它醒来。任何其他来源写的
    /// `brain/steer`/`brain/interrupt`（比如插件、蒸馏器、`event/emit`）都不是定向事件
    /// ——不然任何能写世界日志的东西都能绕开脑的 policy 直接操纵它；它们对**被点名的**
    /// 那个脑会被防自激守卫直接丢掉，对别的脑则照旧按各自的 `WakePolicy` 判定
    /// （见 [`directed_action`]）。
    ///
    /// "绕过预算"只指**不受**预算约束（不会被判成 `suppressed`）：定向唤醒真的开出一个
    /// turn 时照样走 `note_wake`，**计入**预算。连着几次定向 steer/interrupt 之后，紧随
    /// 其后的一次普通唤醒完全可能因为 `max_consecutive_wakes` / `min_interval_ms` 被抑
    /// 制——控制台连点几下之后脑对世界"没反应"，根因通常在这里。
    pub fn steer(&self, brain_id: &str, text: &str) -> Result<(), BrainsError> {
        self.ensure_attached(brain_id)?;
        self.inner.log.append(
            EventDraft::new(
                kind("brain/steer"),
                "brains",
                json!({"brain_id": brain_id, "text": text}),
            )
            .with_salience(200),
        );
        Ok(())
    }

    /// 定向让某个脑 Interrupt：append 一条 `brain/interrupt{brain_id}`。语义同 `steer`
    /// （包括 source 门槛和"不受预算约束、但计入预算"——见 [`Brains::steer`] 的文档），
    /// 只是落在 `WakeAction::Interrupt` 上：Idle 时等同 wake；Running/Maintenance 时打断
    /// 当前 turn（`sink.interrupt`、`finish_turn("interrupted")`，inbox 保留）再开一个
    /// 新 turn。
    pub fn interrupt(&self, brain_id: &str) -> Result<(), BrainsError> {
        self.ensure_attached(brain_id)?;
        self.inner.log.append(
            EventDraft::new(
                kind("brain/interrupt"),
                "brains",
                json!({"brain_id": brain_id}),
            )
            .with_salience(200),
        );
        Ok(())
    }

    /// 把一句话**注入**某个脑的上下文：append 一条 `brain/inject{brain_id, text, source}`
    /// （事件的 source 是 `"brains"`，payload 里那个 `source` 是**注入方**的身份，例如
    /// `"rpc:console"`），返回这条事件的 seq。
    ///
    /// 路由上它和 [`Brains::steer`] 是同一件事：定向事件，绕过目标脑的 policy 与唤醒预算，
    /// 走同一条 `(action, status)` 矩阵（Idle → 开一个新 turn；Running/Maintenance →
    /// 投进当前 turn）。差别只在脑那一侧：观测的 `wake_reason.kind` 是 `brain/inject`，
    /// 原生脑据此把它投进 `Lane::Inject`（"捎带上的上下文"）而不是 `Lane::Steer`
    /// （"下一步照它办"）。[`BrainSink`] 的契约一个字没改——收到的仍然是 `steer(Observation)`。
    ///
    /// 事件**不带 trace_id**：注入不属于任何一个 turn（发出它的时候目标脑可能还在 Idle），
    /// 挂上当前 turn 的 trace 只会让"这个 turn 干了什么"里多出一条不是这个 turn 干的事。
    ///
    /// 两种 `InvalidParams`：空文本（一条什么也没说的注入除了推进游标之外只有坏处——它
    /// 会把一份观测消费掉）和超过 [`MAX_INJECT_TEXT`] 的文本。校验在 append **之前**：
    /// 被拒的调用不该在世界日志里留下痕迹。
    pub fn inject(&self, brain_id: &str, text: &str, source: &str) -> Result<u64, BrainsError> {
        self.ensure_attached(brain_id)?;
        if text.is_empty() {
            return Err(BrainsError::InvalidParams("text is empty".into()));
        }
        if text.len() > MAX_INJECT_TEXT {
            return Err(BrainsError::InvalidParams(format!(
                "text is {} bytes; the limit is {MAX_INJECT_TEXT}",
                text.len()
            )));
        }
        let ev = self.inner.log.append(
            EventDraft::new(
                kind("brain/inject"),
                "brains",
                json!({"brain_id": brain_id, "text": text, "source": source}),
            )
            .with_salience(SAL_INJECT),
        );
        Ok(ev.seq)
    }

    /// 就地替换一个已 attach 的脑的 `WakePolicy`；只影响这次调用之后路由的事件
    /// （已经在路由队列里、还没判定的事件仍按旧 policy 走，判定发生在拿到锁之后）。
    pub fn update_policy(&self, brain_id: &str, policy: WakePolicy) -> Result<(), BrainsError> {
        let mut g = self.inner.brains.lock();
        let a = g
            .get_mut(brain_id)
            .ok_or_else(|| BrainsError::NotAttached(brain_id.into()))?;
        a.policy = policy;
        Ok(())
    }

    fn ensure_attached(&self, brain_id: &str) -> Result<(), BrainsError> {
        if self.inner.brains.lock().contains_key(brain_id) {
            Ok(())
        } else {
            Err(BrainsError::NotAttached(brain_id.into()))
        }
    }

    /// 队列溢出后的补齐：按每个脑自己的游标重放世界日志里还在保留窗口内的事件。
    fn catch_up(inner: &Arc<Inner>) {
        let cursors: Vec<(String, u64)> = inner
            .brains
            .lock()
            .iter()
            .map(|(k, a)| (k.clone(), a.cursor.cursor))
            .collect();
        for (id, cursor) in cursors {
            let events = match inner.log.since(cursor) {
                Ok(v) => v,
                Err(e) => inner
                    .log
                    .since(e.oldest_retained.saturating_sub(1))
                    .unwrap_or_default(),
            };
            for ev in events {
                if BOOKKEEPING.contains(&ev.kind.as_str()) {
                    continue;
                }
                Self::route_one(inner, &id, &ev);
            }
        }
    }

    fn route(inner: &Arc<Inner>, ev: &WorldEvent) {
        if BOOKKEEPING.contains(&ev.kind.as_str()) {
            return;
        }
        let ids: Vec<String> = inner.brains.lock().keys().cloned().collect();
        for id in ids {
            Self::route_one(inner, &id, ev);
        }
    }

    fn route_one(inner: &Arc<Inner>, id: &str, ev: &WorldEvent) {
        // 只用来绕过 `decide`（policy / 预算 / suppressed）——分类结果照样走下面既有的
        // `(action, status)` 矩阵，没有第二套语义。
        let directed = directed_action(ev, id);
        let now = inner.clock.now_ns();
        let plan = {
            let mut g = inner.brains.lock();
            let Some(a) = g.get_mut(id) else { return };
            // 每条到达这里的事件都推高这个脑的"已路由"水位——不管接下来判定成什么。
            // `observe_for` 用它给一份观测的 `since(cursor)` 结果封顶：此刻世界日志里
            // 可能已经摆着更晚、这个脑还没轮到的事件，不能因为顺手扫到它们就把游标
            // 一并带过去，把它们各自的路由决策悄悄吞掉（C2）。放在任何 early return
            // 之前，包括下面的防自激和"已消费"判定。
            a.routed_seq = a.routed_seq.max(ev.seq);
            // 防自激：自己写进世界日志的 brain/* 事件不能再唤醒自己。定向事件
            // （`directed` 非 `None`）不是脑自己写的（是 `Brains::steer`/`interrupt`
            // 这两个 API 写的），不受这条限制——否则点名的那个脑自己永远收不到。
            if directed.is_none()
                && ev.kind.prefix() == "brain"
                && ev.payload.get("brain_id").and_then(|v| v.as_str()) == Some(id)
            {
                return;
            }
            // 已经被这个脑消费过的事件不再判定：观察者在并发 append 之间是无序的，
            // catch-up 也会重放整段 `since(cursor)`，同一条事件可能来两次。
            if ev.seq <= a.cursor.cursor {
                return;
            }
            let d = match directed {
                // 定向事件不经过 `decide`：不受 policy 和预算约束，也不会被判定为
                // suppressed。
                Some(action) => Decision {
                    action,
                    suppressed: false,
                },
                // 状态要一起给：预算数的是 turn，而同一个 `steer` 在 Idle 的脑上开一个
                // 新 turn、在跑着的脑上只是投进去（见 `WakeBudget` 的表）。
                None => a.cursor.decide(&a.policy, ev, now, a.status),
            };
            let reason = WakeReason {
                kind: ev.kind.as_str().to_string(),
                seq: ev.seq,
                action: d.action,
            };
            if d.suppressed {
                Plan::Suppressed
            } else {
                match (d.action, a.status) {
                    // 游标不动：事件留在日志里，下一次观测会带上它们。
                    (WakeAction::Ignore | WakeAction::Accumulate, _) => Plan::Nothing,
                    (WakeAction::Wake, BrainStatus::Idle)
                    | (WakeAction::Steer, BrainStatus::Idle) => Plan::StartTurn(reason),
                    (WakeAction::Wake, _) => {
                        a.pending_wake = Some((reason, a.cursor.cursor));
                        Plan::Nothing
                    }
                    (WakeAction::Steer, _) => Plan::Steer(reason),
                    (WakeAction::Interrupt, BrainStatus::Idle) => Plan::StartTurn(reason),
                    (WakeAction::Interrupt, _) => Plan::Interrupt(reason),
                }
            }
        };
        match plan {
            Plan::Nothing => {}
            Plan::Suppressed => Self::note_suppressed(inner, id, ev),
            Plan::StartTurn(reason) => Self::start_turn(inner, id, reason),
            Plan::Steer(reason) => {
                // 上面的判定只是一次乐观预筛：两次加锁之间 turn 可能已经结束。执行锁里
                // 重新判定，Idle 时 steer 等同 wake（§4.5），不能把 steer 投给不存在的 turn。
                //
                // 而"等同 wake"就意味着它要开一个**新** turn——预筛那一次是按"脑在跑"
                // 判的，没过预算这一关，所以在这里补上（`directed` 的除外：定向事件从不
                // 受预算约束）。少了它，一条落在这个窄窗口里的 policy steer 就是一个
                // 绕开限频的开 turn 入口。
                let steer = {
                    let mut g = inner.brains.lock();
                    let Some(a) = g.get_mut(id) else { return };
                    if a.status == BrainStatus::Idle {
                        let may = directed.is_some()
                            || a.cursor
                                .may_open_turn(&a.policy.budget, inner.clock.now_ns());
                        Err(may)
                    } else {
                        let (obs, was_reset) = Self::observe_for(inner, id, a, reason.clone());
                        Ok((
                            obs,
                            a.sink.clone(),
                            a.turn,
                            a.cursor.cursor,
                            a.epoch,
                            was_reset,
                            a.current_trace.clone(),
                        ))
                    }
                };
                let (obs, sink, turn, cursor, epoch, was_reset, trace) = match steer {
                    Ok(v) => v,
                    Err(true) => {
                        Self::start_turn(inner, id, reason);
                        return;
                    }
                    Err(false) => {
                        Self::note_suppressed(inner, id, ev);
                        return;
                    }
                };
                Self::note_cursor_reset(inner, id, cursor, was_reset, trace.as_deref());
                Self::session_for_epoch(
                    inner,
                    id,
                    epoch,
                    SessionEntry::Steer {
                        turn,
                        observation: obs.clone(),
                    },
                );
                sink.steer(obs);
            }
            Plan::Interrupt(reason) => {
                // 同 Steer：执行锁里重新判定，Idle 时 interrupt 退化成直接开一个新 turn。
                // 被打断的 turn 上 latch 的唤醒在**同一个**临界区里作废：打断本身就要开一个
                // 新 turn，留着 latch 会让 finish_turn 再开一个，两个 turn 重叠——而且后开的
                // 那个永远过不了 finish_turn 的陈旧判定，收不到 turn_end。
                let cur = {
                    let mut g = inner.brains.lock();
                    let Some(a) = g.get_mut(id) else { return };
                    if a.status == BrainStatus::Idle {
                        None
                    } else {
                        a.pending_wake = None;
                        Some((a.sink.clone(), a.turn, a.epoch))
                    }
                };
                let Some((sink, turn, epoch)) = cur else {
                    Self::start_turn(inner, id, reason);
                    return;
                };
                sink.interrupt();
                Self::session_for_epoch(
                    inner,
                    id,
                    epoch,
                    SessionEntry::Interrupt {
                        turn,
                        t_kernel: now,
                    },
                );
                let _ = Self::finish_turn(inner, id, turn, epoch, "interrupted");
                // 打断后的新 turn 不受预算限制，但仍然计入 note_wake。
                Self::start_turn(inner, id, reason);
            }
        }
    }

    /// 被唤醒预算挡下的那条事件记一条 `brain/wake{suppressed:true}`。
    ///
    /// 抑制**不推进游标**，所以队列溢出后的 catch-up 会把同一条事件再判一次。按 seq 去重
    /// （`last_suppressed_seq` 是水位线），一条事件只留一条抑制记录。
    fn note_suppressed(inner: &Arc<Inner>, id: &str, ev: &WorldEvent) {
        let trace = {
            let mut g = inner.brains.lock();
            let Some(a) = g.get_mut(id) else { return };
            if ev.seq <= a.last_suppressed_seq {
                return;
            }
            a.last_suppressed_seq = ev.seq;
            a.current_trace.clone()
        };
        Self::append_brain(
            inner,
            "brain/wake",
            json!({"brain_id": id, "kind": ev.kind.as_str(), "seq": ev.seq, "suppressed": true}),
            trace.as_deref(),
            0,
        );
    }

    /// 在锁内构造观测并推进游标（不做 I/O、不做回调）。返回值第二项是“游标滑出保留窗口、
    /// 已经被重置”——这里持着 `brains` 锁，不能 `append`（`append` 会同步跑观察者），由调用方
    /// 出锁之后补一条 `brain/cursor { reset: true }`。
    ///
    /// `since(a.cursor.cursor)` 拿到的事件按 `a.routed_seq` 封顶（C2）：世界日志此刻可能
    /// 已经摆着比这次唤醒晚、这个脑还没被 `route_one` 单独判定过的事件，直接把它们塞进这
    /// 份观测会连带把游标一起带过去，让它们各自的路由决策永远等不到（`ev.seq <= cursor`
    /// 的“已消费”判定会把它们吞掉，不留一点痕迹）。latch 住的唤醒不受影响：latch 期间
    /// 累积的事件在各自被路由时已经把 `routed_seq` 推过去了，`finish_turn` 补开的这个
    /// turn 一样能看见它们。
    fn observe_for(
        inner: &Arc<Inner>,
        id: &str,
        a: &mut Attached,
        reason: WakeReason,
    ) -> (Observation, bool) {
        // `Attached` 的不变式（见 `routed_seq` 上的注释）。破掉它就意味着下面那次按
        // `routed_seq` 的封顶会把整份观测滤空，而脑拿到的将是一个解释不了自己为什么醒
        // 的空观测——宁可在 debug 构建里当场响。
        debug_assert!(
            a.routed_seq >= a.cursor.cursor,
            "routed_seq {} fell behind cursor {}",
            a.routed_seq,
            a.cursor.cursor
        );
        let mut was_reset = false;
        let events = match inner.log.since(a.cursor.cursor) {
            Ok(v) => v,
            Err(e) => {
                was_reset = true;
                tracing::warn!(
                    cursor = a.cursor.cursor,
                    oldest_retained = e.oldest_retained,
                    "brain cursor predates the retained window; resetting"
                );
                a.cursor = BrainCursor {
                    cursor: e.oldest_retained.saturating_sub(1),
                    ..a.cursor.clone()
                };
                // 游标被抬到保留窗口的起点，`routed_seq` 有可能还落在它后面（被跳过的
                // 那一整段事件早就滑出窗口，再也不会被 `route_one` 看到，`routed_seq`
                // 也就永远不会自己走到那里）。不跟着抬上去的话，下面按 `routed_seq`
                // 的封顶会把这次重置之后读到的事件**全部**滤掉：脑拿到一份空观测，
                // 而且此后每次唤醒都还是空的。
                a.routed_seq = a.routed_seq.max(a.cursor.cursor);
                inner.log.since(a.cursor.cursor).unwrap_or_default()
            }
        };
        let events: Vec<WorldEvent> = events
            .into_iter()
            .filter(|e| e.seq <= a.routed_seq)
            .collect();
        let obs = build_observation(
            id,
            &events,
            inner.state.snapshot(),
            reason,
            inner.clock.now_ns(),
            &a.policy,
            &inner.cfg.obs_budget,
        );
        a.cursor.advance(obs.source_seq_range.1);
        (obs, was_reset)
    }

    /// 游标被重置意味着这个脑漏掉了一段事件——必须在世界日志里留下痕迹，光 warn 不算数。
    fn note_cursor_reset(
        inner: &Arc<Inner>,
        id: &str,
        cursor: u64,
        was_reset: bool,
        trace: Option<&str>,
    ) {
        if was_reset {
            Self::append_brain(
                inner,
                "brain/cursor",
                json!({"brain_id": id, "cursor": cursor, "reset": true}),
                trace,
                0,
            );
        }
    }

    fn start_turn(inner: &Arc<Inner>, id: &str, reason: WakeReason) {
        let now = inner.clock.now_ns();
        // 不变式：认领（判定 Idle）和状态翻转（置 Running）必须在**同一个**临界区里完成。
        // `route_one` 里的判定只是一次乐观预筛——两次加锁之间另一个线程可能已经开了 turn，
        // 所以这里重新判定，非 Idle 就退化成 latch，绝不并发开出两个重叠的 turn。
        let (obs, sink, turn, cursor, epoch, was_reset, trace) = {
            let mut g = inner.brains.lock();
            let Some(a) = g.get_mut(id) else { return };
            if a.status != BrainStatus::Idle {
                a.pending_wake = Some((reason, a.cursor.cursor));
                return;
            }
            a.turn += 1;
            a.status = BrainStatus::Running;
            a.turn_started_ns = Some(now);
            a.proposal_counter = 0;
            a.cursor.note_wake(now);
            let trace = format!("t-{}-{}-{}", a.epoch, a.turn, reason.seq);
            a.current_trace = Some(trace.clone());
            let (obs, was_reset) = Self::observe_for(inner, id, a, reason);
            // 不变式：观测非空时，唤醒证据必须在 `source_seq_range` 里——否则脑拿到的是
            // 一个解释不了自己为什么醒的观测。
            let (lo, hi) = obs.source_seq_range;
            let wake_seq = obs.wake_reason.seq;
            if !obs.events.is_empty() && !(lo <= wake_seq && wake_seq <= hi) {
                tracing::error!(
                    brain = id,
                    wake_seq,
                    lo,
                    hi,
                    "wake reason falls outside the observation range"
                );
                debug_assert!(
                    false,
                    "wake reason seq {wake_seq} outside observation range {lo}..={hi}"
                );
            }
            (
                obs,
                a.sink.clone(),
                a.turn,
                a.cursor.cursor,
                a.epoch,
                was_reset,
                trace,
            )
        };
        let tr = Some(trace.as_str());
        Self::append_brain(
            inner,
            "brain/wake",
            json!({"brain_id": id, "kind": obs.wake_reason.kind, "seq": obs.wake_reason.seq, "suppressed": false}),
            tr,
            0,
        );
        Self::append_brain(
            inner,
            "brain/turn_start",
            json!({"brain_id": id, "turn": turn}),
            tr,
            0,
        );
        Self::note_cursor_reset(inner, id, cursor, was_reset, tr);
        Self::append_brain(
            inner,
            "brain/cursor",
            json!({"brain_id": id, "cursor": cursor}),
            tr,
            0,
        );
        Self::session_for_epoch(
            inner,
            id,
            epoch,
            SessionEntry::TurnStart {
                turn,
                t_kernel: now,
            },
        );
        Self::session_for_epoch(
            inner,
            id,
            epoch,
            SessionEntry::Observation {
                turn,
                observation: obs.clone(),
            },
        );
        sink.on_turn(
            obs,
            TurnHandle {
                brain_id: id.to_string(),
                turn,
                trace_id: trace,
                epoch,
                brains: Arc::downgrade(inner),
            },
        );
    }

    /// 返回“这次调用真的关掉了那个 turn”：陈旧句柄和已经收尾的 turn 都返回 `false`，
    /// 超时扫描靠它决定要不要记 `brain/error`。
    fn finish_turn(inner: &Arc<Inner>, id: &str, turn: u64, epoch: u64, reason: &str) -> bool {
        let now = inner.clock.now_ns();
        let (pending, trace) = {
            let mut g = inner.brains.lock();
            let Some(a) = g.get_mut(id) else { return false };
            // 陈旧句柄（别的 attach 发的，或者 turn 已经被 interrupt 关掉换了号）不能再关一次。
            if a.epoch != epoch || a.turn != turn || a.status == BrainStatus::Idle {
                return false;
            }
            a.status = BrainStatus::Idle;
            a.turn_started_ns = None;
            a.cursor.note_idle(now, &a.policy.budget);
            (a.pending_wake.take(), a.current_trace.take())
        };
        Self::append_brain(
            inner,
            "brain/turn_end",
            json!({"brain_id": id, "turn": turn, "reason": reason}),
            trace.as_deref(),
            0,
        );
        Self::session_for_epoch(
            inner,
            id,
            epoch,
            SessionEntry::TurnEnd {
                turn,
                t_kernel: now,
                reason: reason.to_string(),
            },
        );
        if let Some((r, cursor_at_latch)) = pending {
            // 游标已经越过这条证据（中途的 steer 把它消费掉了）：再开 turn 只会给脑一个
            // 不含唤醒原因的观测。丢掉并留痕。
            let consumed = inner
                .brains
                .lock()
                .get(id)
                .is_some_and(|a| a.cursor.cursor >= r.seq);
            if consumed {
                Self::append_brain(
                    inner,
                    "brain/wake",
                    json!({
                        "brain_id": id,
                        "kind": r.kind,
                        "seq": r.seq,
                        "suppressed": true,
                        "reason": "already_delivered",
                        "cursor_at_latch": cursor_at_latch,
                    }),
                    trace.as_deref(),
                    0,
                );
            } else {
                Self::start_turn(inner, id, r);
            }
        }
        true
    }

    /// **epoch 级**的会话日志写入：只要这份 attach 还在（epoch 对得上）就落盘，不看
    /// 当前跑到第几个 turn。
    ///
    /// 内核自己的 bookkeeping（attach/turn_start/turn_end…）天然是这一档；`Outcome` 也是：
    /// 一次提案的结局可能在它那个 turn 被打断、换成下一个 turn 之后才回来，而条目自带
    /// `turn` 和 `proposal_id`，不会和新 turn 的条目混淆。按“当前 turn”去卡它只会让日志里
    /// 留下一条永远没有下文的 `Proposal`——读的人无从知道那次提案到底做没做。
    /// 与之相对，`Proposal`/`Say`/`AskHuman` 是**句柄**在说话，必须走
    /// [`Brains::session_for_handle`] 的 epoch+turn 双重检查：陈旧句柄记下的话会挂到
    /// 下一个 turn 的编号底下。
    pub(crate) fn session_for_epoch(inner: &Arc<Inner>, id: &str, epoch: u64, e: SessionEntry) {
        Self::session_inner(inner, id, epoch, None, e);
    }

    /// 句柄侧的会话日志写入：只在句柄仍指向该脑当前的 epoch+turn 时才落盘。
    /// `TurnHandle::record` 同步走这里。
    pub(crate) fn session_for_handle(
        inner: &Arc<Inner>,
        id: &str,
        epoch: u64,
        turn: u64,
        e: SessionEntry,
    ) {
        Self::session_inner(inner, id, epoch, Some(turn), e);
    }

    fn session_inner(inner: &Arc<Inner>, id: &str, epoch: u64, turn: Option<u64>, e: SessionEntry) {
        let s = inner
            .brains
            .lock()
            .get(id)
            .filter(|a| a.epoch == epoch && turn.is_none_or(|t| a.turn == t))
            .map(|a| a.session.clone());
        if let Some(s) = s {
            if let Err(err) = s.record(&e) {
                tracing::warn!(brain = id, %err, "session log write failed");
            }
        }
    }
}
