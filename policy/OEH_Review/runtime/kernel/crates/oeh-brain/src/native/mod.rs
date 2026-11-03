//! 原生 turn 循环：把内核的 [`BrainSink`] 合约变成"一个 turn = 若干个 step"。
//!
//! 内核给的是一次唤醒（[`Observation`] + [`TurnHandle`]），脑要还回来的是一句
//! `done(reason)`。中间那段就是这个模块：
//!
//! ```text
//! on_turn(obs, turn)            ← brains 路由任务，必须立刻返回
//!   └─ 存槽位（收件箱 + 任务句柄）→ tokio::spawn(run_turn)
//!        step 0: brain/request → model.complete → brain/model_output
//!                有工具调用？ → brain/tool_call ×n → run_batch → brain/tool_result ×n
//!                transcript += assistant + tool 结果 + 认领到的收件箱
//!        step 1: …
//!   └─ turn.done("end_turn" | "max_steps" | "error: …")
//! ```
//!
//! 三条纪律，和 Plan 3 的 `RpcBrainSink` 逐字相同，理由也相同：
//!
//! 1. **先存句柄再发信号**：槽位（收件箱）必须在 `spawn` 之前就位——steer 完全可能在
//!    任务真正跑起来之前到达，没有槽位它就被默默丢掉了。
//! 2. **一个脑同一时刻只有一个 turn**：新 turn 到来时把旧槽位整个换掉并 abort 旧任务。
//!    turn 的结束有三条路（`done`、被 interrupt、被内核的超时扫描强行收掉），只有第一
//!    条经过这个模块，少了这一步，被超时收掉的那些任务会一直跑下去。
//! 3. **detach 之后不复活**：任务每个 step 开头检查 [`TurnHandle::is_current`]，陈旧就
//!    直接退出——不发事件、不 `done`（那本来就是一次空操作）。
//!
//! [`BrainSink::interrupt`] 只 abort，**不** `done`：内核在调用它之前/之后自己会
//! `finish_turn("interrupted")`，脑再 `done` 一次只会被陈旧判定挡下来，白白让人以为
//! turn 是自己收的尾。

pub mod prompt;
pub mod review;
pub mod review_prompt;
pub mod vision;

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{FutureExt, StreamExt};
use oeh_admission::capability::CapabilityRegistry;
use oeh_kernel::observation::Observation;
use oeh_kernel::session_log::SessionEntry;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{BrainSink, TurnHandle};
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::inbox::{Inbox, InboxItem, Lane};
use crate::model::{
    BoxFuture, EventStream, Message, Model, ModelError, ModelEvent, ModelRequest, StopReason,
    ToolCall, ToolSpec,
};
use crate::retry::{complete_with_retry_observed, RetryEvent, RetryPolicy};
use crate::rsi::KnowledgeStore;
use crate::tools::{cap, ToolOutcome, ToolRegistry};

pub use review::{ReviewConfig, ReviewReport};

/// 这个模块写进世界日志的事件的 `source`。
const SOURCE: &str = "brain";

/// 定向注入的 kind（`Brains::inject`）。内核把它和 `brain/steer` 路由成同一件事，
/// 脑这一侧靠它分出两条 lane。
const INJECT_KIND: &str = "brain/inject";

/// 脑间提问与回答的两个 kind。提问由准入的 `ask_slow_brain` 执行器记（提问也是一次提案），
/// 回答由这里记——它没有自己的能力，因为"回答"就是被提问叫醒之后的那一次 `say`。
const ASK_KIND: &str = "brain/ask";
const ANSWER_KIND: &str = "brain/answer";

/// `say` 能力投影出来的工具名（`robot_` 前缀由 [`ToolRegistry::project_capabilities`] 加）。
const SAY_TOOL: &str = "robot_say";

/// `brain/review` 的 salience。比记账（0）高得多，比 `brain/answer`（150）低一档：
/// "这个脑刚刚改了自己"是操作员**一定**想在仪表盘上看见的事，但它不像一次脑间问答那样
/// 需要即刻回应。两个触发器写在这里，日志里的字符串和配置里的字段名对得上。
const SAL_REVIEW: u8 = 100;
const TRIGGER_EVERY_TURNS: &str = "every_turns";
const TRIGGER_SESSION_END: &str = "session_end";

/// 攒给下一次 review 看的会话记录条数上限。超了丢最旧的——[`review_prompt::user`] 还会
/// 再按字节截一遍，这一道只是别让一个从不 review 的脑把内存慢慢吃光。
const MAX_RECENT_ENTRIES: usize = 256;

/// `brain/answer` 的 salience，和 `brain/ask`（准入那边的 `SAL_BRAIN_ASK`）同为 150：按
/// salience 过滤日志的人要么两条都看见，要么两条都不看见——只看得见问题的那半边最糟。
/// 这个脑写的别的 `brain/*` 都是记账，仍然走默认的 0。
const SAL_ANSWER: u8 = 150;

/// `brain/answer.text` 的上限（1 KiB）。这条 payload 只是**给提问的那个脑看的那一份**，
/// 说出去的全文另有去处：`say` 提案的 `act/proposed.params` 和会话日志的 `Say`。所以这里
/// 宁可截断，也不让一句超长的话把每个读日志的人的预算吃光。
const MAX_ANSWER_TEXT: usize = 1024;

/// 一次 steer 回调该进哪条 lane，正文写成什么。
///
/// `brain/inject` → [`Lane::Inject`]（捎带上的上下文，不为它单独开 turn），正文的**第一
/// 行是注入的那句话**，观测跟在后面（`render_item` 会把后续行缩进在 `[inject]` 底下）。
/// 第一行必须是那句话本身：一条 steer 的正文是一整份观测渲染，注入要是也那样，模型在
/// `# Inbox` 里看到的就是两份开头一模一样的观测，分不出哪一份带着人的话。
///
/// 观测**照旧带上**，不是冗余：这次 steer 已经把这个脑的游标推过去了，丢掉它等于把那一
/// 截事件（期间发生的别的事）悄悄吞掉，谁都看不见。
///
/// 注入的正文从观测里取（那条 `brain/inject` 事件的 `text`）。取不到就退回纯观测渲染，
/// 两种成因：观测预算把那条事件挤掉了（`Observation::truncated`），或者同一份观测里落了
/// **两条** `brain/inject` —— `CollapsedEvent` 只留 `last`，于是靠 `last_seq` 对不上
/// 更早那一条的 `wake_reason.seq`。两种都很窄（每次 append 各走一次路由，一条注入对应
/// 一次 steer 回调），而且 `brain/steer` 本来就是同样的形状。`wake_reason` 在任何一种
/// 情况下都仍然说得清发生过什么。
///
/// **不会**因为策略把 `brain/inject` 判成 `ignore` 而取不到：定向事件在
/// `build_observation` 里有豁免（见 `oeh_kernel::observation::build_observation`），
/// `brain/*: ignore` 挡不住点名给这个脑的那一条。
fn lane_of(obs: &Observation) -> (Lane, String) {
    let rendered = prompt::render_observation(obs);
    lane_with_observation(obs, rendered)
}

fn lane_with_observation(obs: &Observation, rendered: String) -> (Lane, String) {
    if obs.wake_reason.kind != INJECT_KIND {
        return (Lane::Steer, rendered);
    }
    let injected = obs
        .events
        .iter()
        .find(|e| e.kind == INJECT_KIND && e.last_seq == obs.wake_reason.seq)
        .and_then(|e| e.last.get("text"))
        .and_then(|v| v.as_str());
    match injected {
        Some(t) => (Lane::Inject, format!("{t}\n{rendered}")),
        None => (Lane::Inject, rendered),
    }
}

/// 一个原生脑的运行参数。
///
/// `recall_root` 在这里而不是在插件配置里独自存在：它是"这个脑能翻自己哪一摞笔记"，
/// 和人格、步数上限一样属于脑的身份。真正把三个 `recall_*` 工具装进工具表的是插件
/// （[`crate::builtin::native_brain`]）——[`NativeBrain::new`] 收的是一张**已经造好**的
/// 工具表，循环本身不认识沙箱。
#[derive(Clone, Debug, PartialEq)]
pub struct NativeConfig {
    pub brain_id: String,
    /// Frozen evaluation policy: one exact camera observation per turn, no state,
    /// accumulated events, inbox, recall or automatic knowledge maintenance.
    pub isolated_camera_turns: bool,
    /// Also archive the exact ModelRequest on a non-isolated policy turn.
    pub record_policy_requests: bool,
    /// Refuse oversized rendered observations/inboxes rather than trimming policy evidence.
    pub require_complete_observation: bool,
    pub allowed_tools: Option<Vec<String>>,
    pub persona: String,
    /// 一个 turn 最多跑几个 step（一个 step = 一次模型调用 + 它要求的那一批工具）。
    pub max_steps: u32,
    pub max_tokens: u32,
    /// 采样温度。`None`（默认）= 不发这个字段，用厂商自己的默认值。见
    /// [`crate::model::ModelRequest::temperature`]。
    pub temperature: Option<f32>,
    pub recall_root: Option<PathBuf>,
    /// RSI review fork 的触发与预算。没有知识存储（[`NativeBrain::with_chain`]）时整块
    /// 不起作用——review 没有可写的地方。
    pub review: ReviewConfig,
    /// 一个 step 的上限：**一次模型调用（含流读完）+ 它要求的那一批工具**。0 = 不设限
    /// ——只有内核的 `turn_timeout_ms` 兜底。
    ///
    /// 工具批也在这个界里面，因为一个 step 的定义就包含它。少了这一半，一个永不回话
    /// 的执行器、或者一次跑很久的 `recall_*` 就能把 step 无限开着，而 `turn_timeout_ms`
    /// 在"让脑慢慢想"的 profile 里完全可能是 0（这些 e2e 自己就这么配）。
    ///
    /// **维护相位不吃这一条的 0。** review 跑在 `Maintenance` 上，而内核的超时扫描只管
    /// `Running` 的脑，所以那里没有"兜底"可言——它的界是
    /// [`ReviewConfig::timeout_ms`]，不许为 0。这一条非 0 时仍然额外管住 review 的单步。
    pub step_timeout_ms: u64,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            brain_id: "fast".into(),
            isolated_camera_turns: false,
            record_policy_requests: false,
            require_complete_observation: false,
            allowed_tools: None,
            persona: String::new(),
            max_steps: 8,
            max_tokens: 2048,
            temperature: None,
            recall_root: None,
            review: ReviewConfig::default(),
            step_timeout_ms: 60_000,
        }
    }
}

/// 当前 turn 的槽位。`on_turn` 建，`interrupt` / turn 收尾时拆。
struct Current {
    turn: u64,
    /// 三 lane 收件箱。`steer` 往里投，每个 step 开头认领一次。
    inbox: Inbox,
    input_error: Option<String>,
    /// 跑这个 turn 的任务。`interrupt` 用它 abort；turn 自己收尾时只是把它丢掉（detach）。
    task: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Default)]
struct State {
    current: Option<Current>,
}

/// `NativeBrain` 的共享内脏。turn 任务拿的是它的 `Arc`，不是 `Arc<NativeBrain>`：sink
/// 本身没有任何任务需要的东西，而 `Arc<Self>` 会逼着 [`NativeBrain::new`] 走
/// `Arc::new_cyclic`。
struct Shared {
    cfg: NativeConfig,
    /// fallback 链，主 runtime 在前。一个 step 的模型调用按它从头走一遍
    /// （[`crate::retry`]）；链本身不记忆，下一个 step 还是从主 runtime 开始。
    chain: Vec<Arc<dyn Model>>,
    retry: RetryPolicy,
    log: Arc<WorldLog>,
    /// 缓存的 system 前缀。每个 step 都要，而它**只在一次成功的 review 写入之后**变
    /// （`# Skills` 段，见 [`prompt::system_prefix`]）——那一次变化就是我们要的缓存失效。
    /// 除此之外它逐字节稳定；放在锁后面而不是每个 step 重新渲染，是为了让"什么时候变"
    /// 只有一个地方（[`Shared::finish_review`]）。
    system: Mutex<String>,
    /// `# Skills` 段的输入：store 里每个 skill 的 rel + 首行。缓存的理由和 `system`
    /// 一样——它只在一次成功的 review 之后变，而重渲染前缀（能力面一动就要重渲染一次）
    /// 不该每次都去走一遍磁盘上的 skill 目录。
    skills: Mutex<Vec<(String, String)>>,
    /// 能力面**此刻**的投影：工具表 + 它的 specs/names，加上投影时能力面的代号。
    ///
    /// 整份换掉，不是就地改三个字段：一个 turn 的每个 step 都要读这三样，而它们必须
    /// 在一个 turn 之内**完全不动**——step 1 之后少掉一个工具的话，step 2 的请求里会带
    /// 着一条 `tool_result`，它的 `tool_use` 指向一个已经不在 `tools` 里的名字，三家厂商
    /// 都会拒掉整个请求。所以 `step_loop` 在开头把这个 `Arc` 抓进一个局部变量，整个
    /// turn 只认那一份；换表只发生在两个 turn 之间。
    projection: Mutex<Arc<Projection>>,
    /// 投影的**底表**：`recall_*` / `ask_slow_brain` 这些不来自能力面的工具。每次重投
    /// 从它克隆一份再叠上能力面——在一张已经投过的表上"减掉上次多的、加上这次少的"是
    /// 一次可以写错的差分，而从底表重建不可能错。
    base_tools: ToolRegistry,
    /// 活的能力注册表。`None` = 这个脑的工具表是构造时那一份，永远不变（进程内直接
    /// `NativeBrain::new` 的调用方，以及所有不关心能力面的测试）。
    caps: Option<CapabilityRegistry>,
    ident: Identity,
    state: Mutex<State>,
    /// 收件箱条目的投递序。turn 之间连续，`claim_for_step` 只看相对大小。
    seq: AtomicU64,
    /// RSI 第一级的知识存储。`None` = 这个脑不做 review（也没有 `# Skills` 段）。
    store: Option<Arc<KnowledgeStore>>,
    /// review 专用的工具表：只有 `skill_manage` / `memory_manage` / `read`。**和上面那张
    /// 表是两张表**，不是同一张的过滤视图——过滤是一次可以写错的操作，两张表不是。
    review_tools: ToolRegistry,
    review_specs: Vec<ToolSpec>,
    /// 上一次 review 之后跑完了几个 turn。
    turns_since_review: AtomicU64,
    /// 攒给下一次 review 看的会话记录（最多 [`MAX_RECENT_ENTRIES`] 条）。
    recent: Mutex<VecDeque<SessionEntry>>,
    /// 最后一个 turn 的 trace_id。会话末那次 review 跑在 detach 之后，那时句柄已经没了，
    /// 但那次 review 总结的正是这些 turn，挂在它们最后一条链上才读得通。
    last_trace: Mutex<Option<String>>,
    /// 同一时刻只允许一个 review 在飞。detach 会 `abort_current()` 之后立刻派生会话末那
    /// 次 review，而 `abort` 只是**预约**取消（任务要到下一个 await 点才真的停），所以
    /// 一次正在跑的周期 review 和会话末那次可以短暂并存。两个 review 并发的后果不是坏
    /// 文件（store 有写锁 + read-before-write），而是两条 `brain/review`、两次前缀重建，
    /// 以及其中一次拿到一个莫名其妙的 `stale read`——都是给人看的东西变脏。
    review_in_flight: std::sync::atomic::AtomicBool,
    /// review fork 在账本里的署名（`review:<brain_id>`）。
    review_actor: String,
    /// 最近一次被 abort 的 turn 任务。会话末那次 review 先等它真的停下来再开始——见
    /// [`NativeBrain::abort_current`]。
    aborted: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// 上一次 review 收尾时账本的最新 seq。下一次 review 拿它当基线，把**这次**新增的
    /// 行认出来——见 [`Shared::finish_review`]。构造期就从磁盘读一次，因为 store 里可能
    /// 本来就有前几次 run 留下的行，基线从 0 起算会把它们全认成这次写的。
    last_ledger_seq: AtomicU64,
}

/// 能力面的一份投影，连同它是从哪个代号投出来的。
///
/// 三样东西一起换或者一起不换：工具表（执行提案的那一份）、它的 specs（发给厂商的
/// `tools` 数组）、它的名字（`brain/request.tool_names`，也就是日志里"这个 step 手上
/// 有哪些工具"的唯一凭据）。分开存过一版，代价是三者可以不一致，而不一致的症状是模型
/// 调了一个日志说它有、表里却没有的工具。
struct Projection {
    /// 投出这一份时 [`CapabilityRegistry::generation`] 的值。下一个 turn 开头拿它和
    /// 当下的代号比一下：相等就直接复用这一份（零工作量，前缀**逐字节**不变是构造
    /// 出来的，不是靠渲染恰好确定性）。
    generation: u64,
    tools: ToolRegistry,
    specs: Vec<ToolSpec>,
    tool_names: Vec<String>,
    /// 这一轮被跳过的能力名（工具名用不了的那些）。留着只为一件事：每个 turn 都投影
    /// 一次的话，`project_capabilities` 那条 `warn!` 会变成每个 turn 一行；只有这张单子
    /// 和上一份不同才说话，于是"有一个能力永远上不了工具表"仍然看得见，而且只说一次。
    skipped: Vec<String>,
}

impl Projection {
    /// 底表 + 能力面 → 一份投影。`caps` 为 `None` 时就是底表本身（工具表定格在构造期
    /// 的那一份）。
    fn build(
        base: &ToolRegistry,
        caps: Option<&CapabilityRegistry>,
        allowed: Option<&[String]>,
    ) -> Projection {
        // 代号**先读**再 `list()`：反过来的话，两者之间落地的一次注册会被这份投影看见
        // 却记在旧代号上，于是它再也不会被重投一次。这个顺序下最坏情况只是多投一次。
        let generation = caps.map(|c| c.generation()).unwrap_or(0);
        let mut tools = base.clone();
        let skipped = match caps {
            Some(c) => tools.project_capabilities(&c.list()),
            None => Vec::new(),
        };
        if let Some(names) = allowed {
            tools.retain_names(names);
        }
        let specs = tools.specs();
        let tool_names = specs.iter().map(|t| t.name.clone()).collect();
        Projection {
            generation,
            tools,
            specs,
            tool_names,
            skipped,
        }
    }
}

/// `enter_maintenance()` / `leave_maintenance()` 的 RAII 配对。**进和出都在这里**——
/// 只有一半的守卫比没有守卫更糟：它看起来像是把这件事管住了。
///
/// 出的那一半不能依赖"后面那一行会被执行到"：漏掉它这个脑就永远停在 `Maintenance`
/// 上，对内核来说它等价于 Running，于是此后每一次唤醒都只 latch 不开 turn，而超时扫描
/// 只管 `Running`，谁也不会来收尾。
///
/// 进的那一半同样要紧，而且症状相反：不进 `Maintenance` 的话这个 turn 整段 review 都
/// 留在 `Running` 里，于是**内核的超时扫描够得着它**——一个已经做完 step 循环、只是在
/// 记笔记的 turn 会被 `sweep_timeouts` 强行关掉，留下一条假的
/// `brain/error{reason:"turn_timeout"}`，而那正是维护相位被发明出来要避免的那个误诊。
/// 紧接着这个 turn 真正的 `done` 会落在一个内核已经收过尾的 turn 上被悄悄丢掉，而脑已
/// 经 `Idle`，一条 latch 住的唤醒可以在 review 还在写 store 的时候开出下一个 turn。
struct MaintenanceGuard<'a>(&'a TurnHandle);

impl<'a> MaintenanceGuard<'a> {
    fn enter(turn: &'a TurnHandle) -> MaintenanceGuard<'a> {
        turn.enter_maintenance();
        MaintenanceGuard(turn)
    }
}

impl Drop for MaintenanceGuard<'_> {
    fn drop(&mut self) {
        self.0.leave_maintenance();
    }
}

/// "同一时刻只有一个 review 在飞"的令牌。拿不到就说明有别人在跑。
///
/// 也是 RAII：一次 panic 出去的 review 不能把这个标志永久留在"在飞"上，否则这个脑此后
/// 再也不 review 了——一个只在出过一次错之后才显现的静默失效。
struct ReviewGuard<'a>(&'a std::sync::atomic::AtomicBool);

impl<'a> ReviewGuard<'a> {
    fn acquire(flag: &'a std::sync::atomic::AtomicBool) -> Option<ReviewGuard<'a>> {
        flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()
            .map(|_| ReviewGuard(flag))
    }
}

impl Drop for ReviewGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// 一个 LLM 驱动的脑。
pub struct NativeBrain {
    s: Arc<Shared>,
}

impl NativeBrain {
    /// `tools` 是 turn 开始时的能力面投影：插件在 `apply` 里造好第一份交进来，之后每个
    /// turn 开始时若 `CapabilityRegistry::generation()` 变了就重新投影，一个 turn 之内不变。
    /// 能力是动态的（执行器插件随时能注册/注销），见 [`crate::builtin::native_brain`]。
    pub fn new(
        cfg: NativeConfig,
        model: Arc<dyn Model>,
        tools: ToolRegistry,
        log: Arc<WorldLog>,
    ) -> Arc<Self> {
        Self::with_chain(cfg, model, Vec::new(), RetryPolicy::default(), tools, log)
    }

    /// 带 fallback 链与退避参数的构造。**不带知识存储**：没有 review，也没有
    /// `# Skills` 段。
    ///
    /// `primary` 与 `fallbacks` 分开收而不是一个 `Vec`：链**不可能为空**（空链只会让每
    /// 个 turn 都以"没有模型"收场，而身份字段无处可取），让类型说出这件事比在运行时
    /// 兜底便宜。
    pub fn with_chain(
        cfg: NativeConfig,
        primary: Arc<dyn Model>,
        fallbacks: Vec<Arc<dyn Model>>,
        retry: RetryPolicy,
        tools: ToolRegistry,
        log: Arc<WorldLog>,
    ) -> Arc<Self> {
        Self::with_knowledge(cfg, primary, fallbacks, retry, tools, log, None, None)
    }

    /// 带 RSI 知识存储的构造：review fork 用的第二张工具表在这里造出来（`actor` 是
    /// `review:<brain_id>`，见 [`crate::tools::ToolRegistry::add_knowledge`]），
    /// `# Skills` 段也在这里第一次渲染。
    ///
    /// `store` 为 `None` 时这两样都不存在，行为和 [`Self::with_chain`] 完全一样。
    ///
    /// `caps` 是**活的**能力注册表。给了它，工具表就在每个 turn 开头按
    /// [`CapabilityRegistry::generation`] 对一次账：能力面动过就重投一份（一个晚到的
    /// 适配器注册的 `walk` 因此会出现在**下一个** turn 的工具表里），没动过就原样复用。
    /// `tools` 在这条路上是**底表**——只放 `recall_*` / `ask_slow_brain` 这类不来自能力
    /// 面的工具，能力面那一半由这里自己投影。
    ///
    /// 不给（`None`）就是里程碑 1 的老行为：`tools` 是全表，定格在构造的这一刻。
    #[allow(clippy::too_many_arguments)]
    pub fn with_knowledge(
        cfg: NativeConfig,
        primary: Arc<dyn Model>,
        fallbacks: Vec<Arc<dyn Model>>,
        retry: RetryPolicy,
        tools: ToolRegistry,
        log: Arc<WorldLog>,
        store: Option<Arc<KnowledgeStore>>,
        caps: Option<CapabilityRegistry>,
    ) -> Arc<Self> {
        let projection = Projection::build(&tools, caps.as_ref(), cfg.allowed_tools.as_deref());
        if !projection.skipped.is_empty() {
            warn_skipped(&projection.skipped);
        }
        let specs = projection.specs.clone();
        // 构造期没有 runtime，这一遍同步读盘正好（store 刚开完，目录树都是热的）。
        let skills = store
            .as_deref()
            .map(review::skill_index)
            .unwrap_or_default();
        let system = prompt::system_prefix(&cfg.persona, &specs, &skills);
        let review_actor = format!("review:{}", cfg.brain_id);
        // 构造期这一遍同步读盘和上面那个 `skill_index` 同一条理由（还没有 runtime，
        // store 刚开完目录树都是热的）。
        let ledger_seq0 = store
            .as_deref()
            .and_then(|st| st.ledger().ok())
            .map(|l| l.iter().map(|e| e.seq).max().unwrap_or(0))
            .unwrap_or(0);
        let mut review_tools = ToolRegistry::new();
        if let Some(st) = &store {
            review_tools.add_knowledge(st.clone(), review_actor.clone());
        }
        let review_specs = review_tools.specs();
        // 身份来自**主** runtime：`brain/request` 是"这个 step 打算找谁"，真的换了人由
        // `brain/fallback` 说。
        let ident = Identity::of(primary.as_ref());
        let mut chain = Vec::with_capacity(1 + fallbacks.len());
        chain.push(primary);
        chain.extend(fallbacks);
        Arc::new(NativeBrain {
            s: Arc::new(Shared {
                cfg,
                chain,
                retry,
                log,
                system: Mutex::new(system),
                skills: Mutex::new(skills),
                projection: Mutex::new(Arc::new(projection)),
                base_tools: tools,
                caps,
                ident,
                state: Mutex::new(State::default()),
                seq: AtomicU64::new(0),
                store,
                review_tools,
                review_specs,
                turns_since_review: AtomicU64::new(0),
                recent: Mutex::new(VecDeque::new()),
                last_trace: Mutex::new(None),
                review_in_flight: std::sync::atomic::AtomicBool::new(false),
                review_actor,
                aborted: Mutex::new(None),
                last_ledger_seq: AtomicU64::new(ledger_seq0),
            }),
        })
    }

    /// 放弃当前 turn：槽位交出来，任务 abort，**不** `done`。
    ///
    /// 三个调用点：[`BrainSink::interrupt`]（内核已经 `finish_turn` 过了）、
    /// [`Drop`]（detach）、以及插件的 `on_dispose`（fiber 撤销时比 detach 更早跑一步）。
    /// 幂等：槽位已经空了就什么也不做。
    ///
    /// 被 abort 的句柄**留在 [`Shared::aborted`] 里**，不是返回给调用方：`abort` 只是
    /// 预约取消（任务要到下一个 await 点才真的停），而会话末那次 review 需要等它真的停
    /// 下来。
    ///
    /// 留在共享状态里而不是顺着返回值传，是因为 detach 这条路上**这个函数会被调用两
    /// 次**：插件的 `on_dispose` 先跑一步，`Drop` 随后。第二次看到的槽位已经空了，于是
    /// 返回值那一版里 `Drop` 永远拿到 `None`——等于根本没等。这个 bug 当场就被
    /// `detaching_on_top_of_a_running_review_still_leaves_one_review_behind` 抓住了。
    pub(crate) fn abort_current(&self) {
        let cur = self.s.state.lock().current.take();
        if let Some(task) = cur.and_then(|c| c.task) {
            task.abort();
            *self.s.aborted.lock() = Some(task);
        }
    }

    /// detach 时补的那一次 review（`on_session_end`）。
    ///
    /// **派生一个任务，绝不在这里等**：这个 `drop` 跑在内核的 detach 路径上，而那条路
    /// 上 brains 的锁可能还握着；在这里 await 几秒模型调用，就是把整个路由链（所有脑的
    /// 唤醒、steer、interrupt）停在那里。界由 [`ReviewConfig::timeout_ms`] 给，和周期
    /// review 是同一道——没有内核的 turn 超时兜底了，这一段必须自己有界。
    ///
    /// 任务**不被任何人持有**：sink 这一刻正在死掉，没有谁能再 abort 它。这也是它必须
    /// 有界的理由。进程在它跑完之前退出的话，这次 review 就没发生——落盘的部分留在账
    /// 本里，没落盘的部分本来也不存在。
    fn spawn_session_end_review(&self) {
        let s = self.s.clone();
        let aborted = s.aborted.lock().take();
        if !s.cfg.review.on_session_end || s.store.is_none() {
            return;
        }
        // 上次 review 之后一个 turn 都没有 = 没有新东西可总结。
        if s.turns_since_review.load(Ordering::SeqCst) == 0 {
            return;
        }
        let Some(trace) = s.last_trace.lock().clone() else {
            return;
        };
        let Ok(rt) = tokio::runtime::Handle::try_current() else {
            // 没有 runtime 的 drop（测试里直接丢掉一个 sink，或者 runtime 已经关完了）。
            // 这里没有任何办法既跑完它又不阻塞，所以说一声然后放弃。
            tracing::warn!(
                brain_id = %s.cfg.brain_id,
                "detached outside a tokio runtime; skipping the session-end RSI review"
            );
            return;
        };
        rt.spawn(async move {
            // **先等那个被 abort 的 turn 任务真的停下来**，再开始。
            //
            // `abort` 只是预约取消。不等的话，一次正好在飞的周期 review 还攥着
            // `ReviewGuard`，这里会当成"有人在跑"而跳过——而那一个随后在它的下一个
            // await 点被丢掉，永远走不到 `finish_review`。结果是**一条 `brain/review`
            // 都没有**、前缀也不重建，比这个守卫要防的"两条"更糟。等到它停下来之后，
            // 令牌已经还回来了（`ReviewGuard` 也是 RAII），而 `turns_since_review` 还没
            // 被清零（清零发生在 `finish_review` 之前那一步），所以这一次照常会跑。
            //
            // `JoinHandle` 在任务被取消之后解析成 `Err(JoinError::Cancelled)`，正常跑完
            // 则是 `Ok(())`；两种都是"它已经不在跑了"，所以结果直接丢掉。
            if let Some(h) = aborted {
                let _ = h.await;
            }
            // detach 之后没有 turn 了——也**不需要**一个。`run_review` 不收句柄，它那三
            // 个工具一个都不提案（`run_batch(.., None)`）。事件挂在最后一个 turn 的
            // trace 上（`trace`），因为这次 review 总结的正是那几个 turn；一个合成的
            // id 谁也串不起来。
            s.review(&trace, TRIGGER_SESSION_END).await;
        });
    }
}

impl BrainSink for NativeBrain {
    fn on_turn(&self, obs: Observation, turn: TurnHandle) {
        let n = turn.turn;
        // 会话末那次 review 要它（detach 之后句柄已经没了）。在这里记而不是在 turn 任务
        // 里：任务可能被 abort 在第一个 await 之前，而那个 turn 照样发生过。
        *self.s.last_trace.lock() = Some(turn.trace_id.clone());
        // 槽位先就位（见模块头的纪律 1、2）。被换下来的那个绑在 `prev` 上，出锁之后才
        // abort：`JoinHandle::abort` 本身不阻塞，但没有理由把它放进临界区。
        let prev = {
            let mut g = self.s.state.lock();
            g.current.replace(Current {
                turn: n,
                inbox: Inbox::new(),
                input_error: None,
                task: None,
            })
        };
        if let Some(t) = prev.and_then(|c| c.task) {
            t.abort();
        }
        let s = self.s.clone();
        let task = tokio::spawn(run_turn(s, obs, turn));
        let mut g = self.s.state.lock();
        match g.current.as_mut() {
            Some(c) if c.turn == n => c.task = Some(task),
            // 这一瞬间槽位已经被下一个 turn（或一次 interrupt）换掉了。任务自己会在第
            // 一次 `is_current` 检查时退出，但别留下一个谁都 abort 不了的孤儿。
            _ => task.abort(),
        }
    }

    /// Running 时：把观测渲染成一条 [`Lane::Steer`]，下一个 step 认领。
    ///
    /// 没有 turn 在跑就丢掉——内核对 idle 的脑发定向 steer 时走的是 `start_turn`
    /// （`Plan::Steer` 在执行锁里重新判定，Idle 等同 wake），那条路径进的是 `on_turn`，
    /// 不是这里。
    ///
    /// 有一个窄窗口会真的丢掉一条 steer：turn 的最后一个 step 已经认领过收件箱、而
    /// `done` 还没回到内核的那几微秒里，内核仍然认为这个脑是 Running，于是走 `Plan::Steer`
    /// （`observe_for` 已经把游标推过那条证据）。这个窗口是内核那套"乐观预筛 + 执行锁里
    /// 复判"的固有产物，不是这里的实现选择——任何"最后一次认领之后还能收到 steer"的脑
    /// 都有它。证据本身没丢（世界日志 + 会话日志的 `SessionEntry::Steer` 都在），丢的是
    /// "模型这一轮看见了它"。
    fn steer(&self, obs: Observation) {
        if self.s.cfg.isolated_camera_turns {
            return;
        }
        let rendered = if self.s.cfg.require_complete_observation {
            prompt::render_complete_observation(&obs).map(|text| lane_with_observation(&obs, text))
        } else {
            Ok(lane_of(&obs))
        };
        let seq = self.s.seq.fetch_add(1, Ordering::SeqCst);
        let mut g = self.s.state.lock();
        if let Some(c) = g.current.as_mut() {
            match rendered {
                Ok((lane, text)) => c.inbox.push(InboxItem { lane, text, seq }),
                Err(error) => c.input_error = Some(error),
            }
        }
    }

    /// 只 abort，不 `done`：内核已经 `finish_turn("interrupted")` 过了。
    ///
    /// 收件箱跟着槽位一起丢——内核在打断之后立刻开一个新 turn，而新 turn 的观测里本来
    /// 就带着那条打断证据；把上一个 turn 的 steer 文本搬过去只会让模型收到两遍。
    fn interrupt(&self) {
        self.abort_current();
    }
}

/// detach 即刻收摊。
///
/// `Brains::attach` 存的 `Arc<dyn BrainSink>` 是这个 sink 唯一的强引用（插件把它交出去
/// 之后只留一个 `Weak`），所以 detach 把 `Attached` 摘掉的那一刻就是这里。**必须收**：
/// turn 任务持的是 `Arc<Shared>` 和一个 `TurnHandle`，谁都不认识 sink，没有这一下它会
/// 一直跑到下一个 await 点之后——默认 `step_timeout_ms` 是 60 s，也就是 `brain/detach`
/// 写进日志之后最多一分钟，还会冒出这个已经死掉的 turn 的 `brain/model_output`、
/// `brain/tool_call`、整整一批工具执行和 `brain/tool_result`。能力工具倒是无害
/// （`TurnHandle::propose` 会以 `turn is not current` 拒掉），但 `recall_*` 根本不看句柄，
/// 会替一个已经不存在的脑去读沙箱。
impl Drop for NativeBrain {
    fn drop(&mut self) {
        self.abort_current();
        self.spawn_session_end_review();
    }
}

/// 一个 turn：跑完 step 循环，交还槽位，`done`。
async fn run_turn(s: Arc<Shared>, obs: Observation, turn: TurnHandle) {
    // panic 也要有终态。一个 turn 的 panic 不该让这个脑**静悄悄地**永远停在 Running：
    // 没有 `done` 就没有 `brain/turn_end`、没有下一个 turn、也没有任何一条日志说出了什么
    // 事。内核的 turn 超时扫描是最后一道，但 `turn_timeout_ms: 0`（一个"让脑慢慢想"的
    // profile 完全会这么配，新的 e2e 自己就这么配）把那道也关了。
    //
    // 包住的是 step 循环，不是 `release_slot`/`done`——收尾这一段必须在任何情况下都跑。
    // 工具处理器早就各自 `catch_unwind` 过（`tools.rs`），这里挡的是循环自己那一截：
    // 模型流的消费（Task 3 会把 SSE 解析塞进来）、prompt 渲染、消息拼装。
    //
    // 对 abort 没有影响：被 abort 的 future 是**被丢掉**，不是 panic，`catch_unwind`
    // 看不到它，detach / interrupt 的行为和第 1 轮一模一样。
    let reason = match AssertUnwindSafe(step_loop(&s, obs, &turn))
        .catch_unwind()
        .await
    {
        Ok(reason) => reason,
        Err(_) => Some("error: panicked".to_string()),
    };
    // 维护相位排在 `done` 之前、`release_slot` 之前：`done` 会让内核立刻开下一个 turn，
    // 那时这次 review 还没跑完，它要总结的那几个 turn 已经又长出一个了；而槽位留着的
    // 好处是 review 期间到达的 steer 仍然会被数进 `brain/inbox_dropped`，而不是无声消失。
    //
    // `reason` 是 `None` 就跳过：这个 turn 已经不归我们管了（超时扫描收了尾，或者
    // detach），一个没人认领的 turn 不该顺手把维护相位也带走。
    if reason.is_some() {
        s.after_turn(&turn).await;
    }
    // 槽位先还（含把收件箱里剩下的东西数出来并丢掉），再 `done`：`done` 会让内核立刻开
    // 下一个 turn（latch 住的唤醒），而那条路会回头调 `on_turn` 去抢同一个槽位。
    let left = s.release_slot(turn.turn);
    let Some(reason) = reason else {
        // turn 已经不是当前的那个（超时扫描收了尾，或者 detach 了）。不发事件，也不
        // `done`——那是一次空操作，只会让读代码的人以为这条路真的关掉了一个 turn。
        return;
    };
    if left.followups > 0 {
        // 里程碑 1：followup 不自动开下一个 turn。留一条痕迹，让"脑自己给自己留的待办
        // 被丢了"在日志里看得见，而不是只在源码注释里。
        s.emit(&turn, "brain/followup", json!({ "count": left.followups }));
    }
    if left.next_step > 0 {
        // 最后一个 step 认领过之后才到的 steer / inject。它们的证据在内核那边已经被游标
        // 吃掉了（`Plan::Steer` 里的 `observe_for`），下一个 turn 的观测不会再带上——
        // 也就是说这句话模型**永远**看不到了。prompt 文本按纪律不进世界日志，所以至少
        // 把"丢了几条"记下来：操作员 steer 了一句却石沉大海时，日志里得有这一行。
        s.emit(
            &turn,
            "brain/inbox_dropped",
            json!({ "count": left.next_step }),
        );
    }
    turn.done(&reason);
}

/// 说一次"这个能力上不了工具表"。
///
/// 三家厂商对工具名的字符集都有要求，一个 `robot_walk.fast` 换来的是整个请求的
/// HTTP 400——那时候**所有**工具都用不了，而日志里没有任何线索。所以它必须被说出来；
/// 而因为投影现在每个 turn 都可能重来一次，它**只在这张单子变了的时候**才说，否则一
/// 个永远上不了表的能力会变成每个 turn 一行的噪音（见 [`Projection::skipped`]）。
fn warn_skipped(names: &[String]) {
    tracing::warn!(
        capabilities = %names.join(", "),
        "capability names are not usable tool names ([a-z0-9_]); they are not projected"
    );
}

/// step 循环。返回 `done` 的理由；`None` = 这个 turn 已经不归我们管了。
async fn step_loop(s: &Shared, obs: Observation, turn: &TurnHandle) -> Option<String> {
    let obs = if s.cfg.isolated_camera_turns {
        match vision::isolate_policy_observation(&obs) {
            Ok(obs) => obs,
            Err(error) => return Some(format!("error: {error}")),
        }
    } else {
        obs
    };
    // 这个 turn 已经不归我们管了的话，连工具表都不必对账。这一下和循环开头那一下是
    // 同一个判断，重复一次是为了让投影**发生在**它后面：`brain/request.tool_names` 必须
    // 描述真的被用上的那张表，而一个还没确认是当前 turn 的脑不该去动共享的投影缓存。
    if !turn.is_current() {
        return None;
    }
    // 能力面 → 工具表，**整个 turn 只对一次账**。抓进局部变量而不是每个 step 读一次
    // `Shared`：两个 step 之间换表会让第二个请求带着一条指向已消失工具的 `tool_result`，
    // 三家厂商都会拒掉整个请求（见 [`Projection`]）。
    let proj = s.turn_projection();
    // 第一个 user message：观测 + 这一刻收件箱里的东西。system 前缀不在 `messages` 里
    // ——[`ModelRequest::system`] 是独立字段（Anthropic 的线格式就是这样，而把前缀和
    // 会变的内容分开是缓存纪律最省事的执行方式）。
    // 这个 turn 是不是被别的脑的提问叫醒的。算一次，整个 turn 用；回答出去之后置空
    // ——**一次提问只换一次回答**。
    let mut asked_by = AskedBy::of(&obs);
    let rendered = if s.cfg.require_complete_observation {
        prompt::render_complete_observation(&obs)
    } else {
        Ok(prompt::render_observation(&obs))
    };
    let mut first = match rendered {
        Ok(text) => text,
        Err(error) => {
            s.emit(
                turn,
                "brain/request-error",
                json!({"reason": error, "step": 0}),
            );
            return Some(format!("error: {error}"));
        }
    };
    let inbox = match s.claim_inbox(turn.turn) {
        Ok(inbox) => inbox,
        Err(error) => {
            s.emit(
                turn,
                "brain/request-error",
                json!({"reason": error, "step": 0}),
            );
            return Some(format!("error: {error}"));
        }
    };
    if let Some(inbox) = inbox {
        first.push('\n');
        first.push_str(&inbox);
    }
    let first = match vision::user_message(&obs, first) {
        Ok(message) => message,
        Err(error) => {
            s.emit(
                turn,
                "brain/request-error",
                json!({"reason": error.to_string(), "step": 0}),
            );
            return Some(format!("error: {error}"));
        }
    };
    let mut messages = vec![first];
    // 0 当 1：`max_steps: 0` 是配置写坏了，不是"一个 step 都不跑"——那样这个脑每次被
    // 唤醒都只会记一条 turn_end，而日志里没有任何线索。
    let max_steps = u64::from(s.cfg.max_steps.max(1));
    let mut step: u64 = 0;
    loop {
        if !turn.is_current() {
            return None;
        }
        s.emit(
            turn,
            "brain/request",
            json!({
                "provider": s.ident.provider,
                "model": s.ident.model,
                "api_mode": s.ident.api_mode,
                "step": step,
                "tool_names": proj.tool_names,
            }),
        );
        let req = ModelRequest {
            system: s.system.lock().clone(),
            messages: messages.clone(),
            tools: proj.specs.clone(),
            max_tokens: s.cfg.max_tokens,
            temperature: s.cfg.temperature,
        };
        if s.cfg.isolated_camera_turns || s.cfg.record_policy_requests {
            s.emit(
                turn,
                "brain/policy_request",
                json!({
                    "step": step, "system": req.system, "messages": req.messages,
                    "tools": req.tools, "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                }),
            );
        }
        // 一个 step 的截止时刻：模型调用和它之后那批工具**共用**这一个预算。
        let deadline = s.deadline();
        // 出错时 `out` 带着**出错之前已经收到的**增量：流可能已经吐了 3 KB 文本才断掉，
        // 那半句话是"模型到底想干什么"唯一的线索。
        let StepOut { out, err, logged } = s.complete(req, &turn.trace_id, step, deadline).await;
        for message in &out.warnings {
            s.emit(
                turn,
                "brain/model_warning",
                json!({"step": step, "message": message}),
            );
        }
        for usage in &out.usage {
            s.emit(
                turn,
                "brain/usage",
                json!({"step": step, "input_tokens": usage.0,
                "output_tokens": usage.1, "cached_input_tokens": usage.2,
                "reasoning_output_tokens": usage.3,
                "complete": err.is_none(), "accounting": "provider_reported_event"}),
            );
        }
        if let Some(e) = err {
            // fail-closed：整条 fallback 链都失败之后才到这里（重试与降级已经在
            // `Shared::complete` 里走完并各记了事件），这个 turn 就此结束，理由原样进
            // `done`（内核把它写进 `brain/turn_end.reason` 和会话日志的 `TurnEnd`）。
            let partial_chars = out.text.chars().count();
            if partial_chars > 0 {
                // 半句话也进会话日志——那里是全文唯一的去处。世界日志只拿到长度。
                let _ = turn.note(SessionEntry::ModelOutput {
                    turn: 0,
                    step,
                    text: out.text,
                });
            }
            // 瀑布已经替这个错记过一条（调用根本没成功，`{provider, model, attempt}`）
            // 就不再记第二条：那条会一个字段都不多——没有流也就没有 partial。这里这条
            // 是**另一种**失败：流已经在吐东西了然后断掉/超时。
            if !logged {
                s.emit(
                    turn,
                    "brain/request-error",
                    json!({
                        "reason": e.to_string(),
                        "step": step,
                        "partial_chars": partial_chars,
                        // 断流之前已经拼完的工具调用也一并报数：它们不会被执行（没有
                        // `brain/tool_call`、没有 `act/*`），而只看 `partial_chars: 0` 的话
                        // "模型什么都没说"和"模型要求了两次动作但流断了"长得一模一样。
                        "partial_tool_calls": out.calls.len(),
                    }),
                );
            }
            return Some(format!("error: {e}"));
        }
        // 全文只进会话日志；世界日志里只有"多长、几个工具调用"。同一条也进 review 的
        // 取景框（[`Shared::push_recent`]）——那是"上次 review 之后这个脑做了什么"的
        // 唯一来源，从磁盘上的会话日志再读一遍只会多一条会漂移的真相。
        let _ = turn.note(s.push_recent(SessionEntry::ModelOutput {
            turn: 0,
            step,
            text: out.text.clone(),
        }));
        s.emit(
            turn,
            "brain/model_output",
            json!({
                "step": step,
                "chars": out.text.chars().count(),
                "tool_calls": out.calls.len(),
            }),
        );
        if out.calls.is_empty() || out.stop == StopReason::EndTurn {
            return Some("end_turn".into());
        }
        // 记账在**执行之前**：一次卡住的工具（或者一次打断）之后，日志里必须留得下
        // "模型确实要求过这次调用"，否则 `brain/tool_call` 会和 `brain/tool_result`
        // 一样只在成功路径上出现。
        for c in &out.calls {
            let _ = turn.note(s.push_recent(SessionEntry::ToolCall {
                turn: 0,
                step,
                call_id: c.id.clone(),
                name: c.name.clone(),
                args: c.args.clone(),
            }));
            s.emit(
                turn,
                "brain/tool_call",
                json!({ "name": c.name, "args_bytes": args_bytes(&c.args) }),
            );
        }
        // 工具批也在 step 的截止时刻之内（I4）。超时的处理和模型调用那条路逐字相同：
        // 记下来、`done("error: …")`、不让 turn 挂着。
        //
        // 包的是**整批**而不是每一条：`by` 超时会把这个 future 整个丢掉，于是哪几条已经
        // 跑完我们已经无从知道（它们的结果和 future 一起没了，也永远进不了 transcript）。
        // 所以这一批的每一条都记成 `timeout`——那正是"模型这一步没拿到任何结果"的事实。
        // 真的已经递出去的提案在 `act/*` 里有自己的一条链，靠同一个 trace_id 对得上。
        //
        // 注意它**不回收**线程：`recall_*` 跑在 `spawn_blocking` 上，那种任务谁也 abort
        // 不了。这条超时管的是"turn 不挂着"，不是"活干完了"。
        let pending: Vec<String> = out.calls.iter().map(|c| c.name.clone()).collect();
        let results = match by(deadline, proj.tools.run_batch(out.calls, Some(turn))).await {
            Ok(r) => r,
            Err(_) => {
                for name in &pending {
                    s.emit(
                        turn,
                        "brain/tool_result",
                        json!({ "name": name, "ok": false, "error": "timeout" }),
                    );
                }
                s.emit(
                    turn,
                    "brain/request-error",
                    json!({
                        "reason": "step timeout: tools",
                        "step": step,
                        "tool_calls": pending.len(),
                    }),
                );
                return Some("error: step timeout".into());
            }
        };
        for (c, o) in &results {
            let _ = turn.note(s.push_recent(SessionEntry::ToolResult {
                turn: 0,
                step,
                call_id: c.id.clone(),
                ok: o.ok,
                result: o.result.clone(),
            }));
            s.emit(
                turn,
                "brain/tool_result",
                json!({ "name": c.name, "ok": o.ok }),
            );
            // 被提问叫醒的 turn 里，**第一句**说出口的话就是那次回答：在 `say` 之外再记
            // 一条 `brain/answer`，提问的那个脑靠它的 wake 策略收到（`brain_id` 是回答者，
            // 所以内核的防自激守卫只挡回答者自己）。
            //
            // 之后的 `robot_say` 只是说话：一次提问换一次回答，否则一段三句话的答复会在
            // 提问者那边变成三次唤醒/steer，而它们说的是同一件事。想补充的脑再开一个 turn
            // ——那时它得重新提问，代价是明的。
            let answer = asked_by.as_ref().and_then(|a| a.answer(c, o));
            if let Some(payload) = answer {
                s.emit_salient(turn, ANSWER_KIND, SAL_ANSWER, payload);
                asked_by = None;
            }
        }
        // assistant 消息里的 tool_calls 必须和下面那串 `tool` 消息一一对应、同序：
        // 两边都从 `results` 来（`run_batch` 按模型给出的顺序返回），不会错位。
        messages.push(Message::Assistant {
            content: out.text,
            tool_calls: results.iter().map(|(c, _)| c.clone()).collect(),
        });
        for (c, o) in results {
            messages.push(Message::Tool {
                call_id: c.id,
                content: serde_json::to_string(&o.result)
                    .unwrap_or_else(|e| format!("{{\"error\":\"unserialisable result: {e}\"}}")),
            });
        }
        let inbox = match s.claim_inbox(turn.turn) {
            Ok(inbox) => inbox,
            Err(error) => {
                s.emit(
                    turn,
                    "brain/request-error",
                    json!({"reason": error, "step": step + 1}),
                );
                return Some(format!("error: {error}"));
            }
        };
        if let Some(text) = inbox {
            messages.push(Message::User { content: text });
        }
        step += 1;
        if step >= max_steps {
            return Some("max_steps".into());
        }
    }
}

/// 叫醒这个 turn 的那一次脑间提问。`None` = 不是提问叫醒的，这个 turn 里的 `robot_say`
/// 就只是说话。
///
/// 为什么是 turn 级而不是 step 级：提问是"这个 turn 存在的理由"。慢脑很可能先翻几步笔记
/// 再开口，那一句回答落在第几个 step 上不该改变它是不是回答。
struct AskedBy {
    ask_id: String,
    from: String,
}

impl AskedBy {
    /// 从叫醒这个 turn 的观测里认出那条 `brain/ask`。
    ///
    /// 观测里的事件是**折叠**过的（同 kind 合成一条，只留首末两份 payload），而
    /// [`prompt::render_observation`] 渲染的只有 `last`——所以这里认的也只能是 `last`：
    /// 回答必须贴着**模型真的读到的那个问题**，不然 `ask_id` / `to` 指向的是一个它从没
    /// 看见过的提问（两条 ask 折进同一条观测时，`first` 和 `last` 是两个不同的提问者）。
    ///
    /// 唤醒证据落在哪一端由内核决定，这里只用它的 seq 认出**是哪一条折叠项**：一个被
    /// latch 住的唤醒（脑还在 Running 时来的）要等到下一次 `observe_for` 才建观测，那时
    /// 后到的提问可能已经被折进同一项的尾巴上了。`CollapsedEvent` 没有"哪一条是唤醒证据"
    /// 的标记（那要改内核），`last` 是现有信息里唯一和 prompt 对得上的那一份。
    fn of(obs: &Observation) -> Option<AskedBy> {
        if obs.wake_reason.kind != ASK_KIND {
            return None;
        }
        let seq = obs.wake_reason.seq;
        let c = obs
            .events
            .iter()
            .find(|c| c.kind == ASK_KIND && c.first_seq <= seq && seq <= c.last_seq)?;
        let field = |k: &str| c.last.get(k).and_then(Value::as_str).map(str::to_string);
        Some(AskedBy {
            ask_id: field("ask_id")?,
            from: field("from")?,
        })
    }

    /// 这次工具调用是不是对提问的回答；是就给出 `brain/answer` 的 payload。
    ///
    /// 只有**执行成功**的 `robot_say` 算：被准入拒掉的那句话根本没说出去，不该被转成一条
    /// 回答（提问的脑看得见那条 `act/refused`，`wake/fast.yaml` 对它是 `steer`）。
    ///
    /// 文本取自模型这次调用的参数，而不是执行结果：回答的是"这个脑说了什么"。真正过了
    /// 规则链之后的那份参数在 `act/*` 里，靠同一个 `trace_id` 就能对上。
    ///
    /// 截断了就带一个 `truncated: true`（没截断时这个字段根本不出现）：一句 2 KiB 的话
    /// 到了提问者那里不能看起来像一句完整的 1 KiB 话。
    fn answer(&self, call: &ToolCall, out: &ToolOutcome) -> Option<Value> {
        if !out.ok || call.name != SAY_TOOL {
            return None;
        }
        let text = call.args.get("text").and_then(Value::as_str)?;
        let (text, cut) = cap(text, MAX_ANSWER_TEXT);
        let mut payload = json!({ "to": self.from, "ask_id": self.ask_id, "text": text });
        if cut {
            payload["truncated"] = Value::Bool(true);
        }
        Some(payload)
    }
}

/// 一次模型调用收下来的全部内容。
struct ModelOut {
    text: String,
    calls: Vec<ToolCall>,
    stop: StopReason,
    /// `(input, output, cached, reasoning)`，见 [`ModelEvent::Usage`]。
    usage: Vec<(u32, u32, u32, Option<u32>)>,
    /// 这次调用附带的不致命告警（[`ModelEvent::Warning`]），原样进 `brain/model_warning`。
    warnings: Vec<String>,
}

/// 一个 step 的模型调用的结局。
struct StepOut {
    out: ModelOut,
    err: Option<ModelError>,
    /// 这个错**已经**在世界日志里有一条 `brain/request-error` 了（瀑布记的，带
    /// `{provider, model, attempt}`）。分开记一个布尔而不是让调用处去猜错的种类：同一个
    /// `ModelError::Transport` 既可能是"连不上"（瀑布记过）也可能是"流断在半路"（瀑布
    /// 没见过），差别只在**它发生在哪一层**。
    logged: bool,
}

/// [`Shared::release_slot`] 交还槽位时，收件箱里还剩下的东西。两个数分开报：followup 是
/// "这个 turn 自己留的待办"（里程碑 1 不自动开下一个 turn），next-step 是"来晚了一步的
/// steer / inject"（那句话模型永远看不到了）——两种丢法的根因完全不同。
#[derive(Default, Debug, PartialEq, Eq)]
struct Leftover {
    followups: usize,
    next_step: usize,
}

impl Shared {
    /// 这个 turn 认的那一份工具表。**一个 turn 只调用一次**（见 [`Projection`]）。
    ///
    /// 常态是一次原子读 + 一次 `Arc` 克隆：能力面没动过，代号相等，上一份原样还回去
    /// ——于是 prompt 前缀逐字节不变是**构造出来的**，不是靠"重渲染恰好是确定性的"。
    /// 动过了才重投一份，而只有**投影出来的工具表真的不一样**时才重渲染前缀：一次
    /// 注册又撤销（代号 +2、表没变）不该作废任何一家厂商的 prompt 缓存。
    fn turn_projection(&self) -> Arc<Projection> {
        let Some(caps) = self.caps.as_ref() else {
            // 静态工具表的脑（`NativeBrain::new`）：这个锁后面的东西一辈子不变。
            return self.projection.lock().clone();
        };
        let current = self.projection.lock().clone();
        if current.generation == caps.generation() {
            return current;
        }
        let next = Arc::new(Projection::build(
            &self.base_tools,
            Some(caps),
            self.cfg.allowed_tools.as_deref(),
        ));
        if next.specs != current.specs {
            // 工具表真的变了：前缀里的那一段跟着变，这一下就是 prompt 缓存失效。它的
            // 代价是**一次**，而且只发生在能力面真的换了东西的时候——这正是要付的那笔。
            let system = prompt::system_prefix(&self.cfg.persona, &next.specs, &self.skills.lock());
            *self.system.lock() = system;
        }
        if next.skipped != current.skipped {
            warn_skipped(&next.skipped);
        }
        *self.projection.lock() = next.clone();
        next
    }

    /// 这一个 step 的截止时刻。`step_timeout_ms: 0` = 不设限。
    ///
    /// 算一次、模型调用和工具批共用：一个 step 的定义是"一次模型调用 + 它要求的那一批
    /// 工具"，给两半各自一个 `step_timeout_ms` 等于把这个数悄悄翻倍。
    fn deadline(&self) -> Option<tokio::time::Instant> {
        (self.cfg.step_timeout_ms > 0)
            .then(|| tokio::time::Instant::now() + Duration::from_millis(self.cfg.step_timeout_ms))
    }

    /// 一次模型调用：返回收到的内容 **和** 是否出错。
    ///
    /// 不是 `Result<ModelOut, _>`：出错的那一刻流上可能已经过了几千字，而
    /// `?` 会把它们连同错误一起扔掉。超时同理——所以超时是**每次 `await` 一个截止时刻**
    /// （`timeout_at`），不是把整段 `collect` 包进一个 `timeout`：包起来的话超时会把整个
    /// future 连同它的累加器一起 drop，"模型说到一半超时"恰恰是最需要那半句的场合。
    async fn complete(
        &self,
        req: ModelRequest,
        trace: &str,
        step: u64,
        deadline: Option<tokio::time::Instant>,
    ) -> StepOut {
        self.complete_in("", req, trace, step, deadline).await
    }

    /// 同上，但带一个相位标记。`phase` 非空时会进 `brain/request-error` /
    /// `brain/fallback` 的 payload：维护相位的模型调用**不记** `brain/request`
    /// （见 [`review`] 的模块头），所以一条 `brain/request-error{step: 0}` 如果不说自己
    /// 是哪个相位的，读日志的人分不清它来自 turn 的第一步还是那次 review 的第一步。
    async fn complete_in(
        &self,
        phase: &str,
        req: ModelRequest,
        trace: &str,
        step: u64,
        deadline: Option<tokio::time::Instant>,
    ) -> StepOut {
        // 退避用真时钟，但仍然在 step 的截止时刻之内：`by(deadline, …)` 包的是**整段**
        // 瀑布（连着重试之间的等待），不然一个配得很宽的 retry 能把 step 超时架空。
        let sleep = |d: Duration| -> BoxFuture<()> { Box::pin(tokio::time::sleep(d)) };
        let on_event = |ev: RetryEvent| self.emit_retry(phase, trace, step, ev);
        let on_attempt = |name: &str, attempt: u32| {
            if self.cfg.record_policy_requests {
                let (provider, model) = split_name(name);
                self.emit_at(
                    trace,
                    "brain/provider_attempt",
                    0,
                    json!({"provider": provider, "model": model, "attempt": attempt,
                           "step": step, "phase": phase}),
                );
            }
        };
        let mut out = ModelOut {
            text: String::new(),
            calls: Vec::new(),
            stop: StopReason::Other,
            usage: Vec::new(),
            warnings: Vec::new(),
        };
        let stream = match by(
            deadline,
            complete_with_retry_observed(
                &self.chain,
                req,
                &self.retry,
                &sleep,
                &on_event,
                &on_attempt,
            ),
        )
        .await
        {
            Ok(Ok(s)) => s,
            // 整条链都失败：每一次尝试瀑布都记过一条。
            Ok(Err(e)) => {
                return StepOut {
                    out,
                    err: Some(e),
                    logged: true,
                }
            }
            // step 的截止时刻在瀑布里走完之前就到了（重试等待也算在里面）。这个错不是
            // 任何一次尝试的错，瀑布没记过它。
            Err(_) => {
                return StepOut {
                    out,
                    err: Some(ModelError::Timeout),
                    logged: false,
                }
            }
        };
        let err = collect(stream, deadline, &mut out).await;
        StepOut {
            out,
            err,
            logged: false,
        }
    }

    /// 瀑布上的一件事 → 一条世界事件。三种失败在日志里必须分得开：请求失败本身
    /// （`brain/request-error`，带 provider / model / 第几次）、决定再试（`brain/retry`，
    /// 带等多久）、换 runtime（`brain/fallback`，带从谁到谁）。
    ///
    /// 和 step 循环里那条流中途断掉的 `brain/request-error`（带 `step` /
    /// `partial_chars`）同名不同形：一个是"这次调用没发出去/没成功"，一个是"已经在收
    /// 内容了然后断了"。
    fn emit_retry(&self, phase: &str, trace: &str, step: u64, ev: RetryEvent) {
        let (kind, payload) = match ev {
            RetryEvent::Error {
                model,
                attempt,
                reason,
            } => {
                let (provider, model) = split_name(&model);
                (
                    "brain/request-error",
                    // `step` 也带上：一个多 step 的 turn 出错时，光靠事件在日志里的相对
                    // 位置去数是第几步太容易数错，而 `brain/request` 本来就带着它。
                    json!({"provider": provider, "model": model, "attempt": attempt, "reason": reason, "step": step}),
                )
            }
            RetryEvent::Retry { attempt, delay_ms } => (
                "brain/retry",
                json!({"attempt": attempt, "delay_ms": delay_ms}),
            ),
            RetryEvent::Fallback { from, to } => {
                ("brain/fallback", json!({"from": from, "to": to}))
            }
        };
        let mut payload = payload;
        if !phase.is_empty() {
            payload["phase"] = Value::String(phase.to_string());
        }
        self.emit_at(trace, kind, 0, payload);
    }

    /// 认领下一个 step 的收件箱条目并渲染。turn 号对不上（槽位已经换人）就当空。
    fn claim_inbox(&self, turn: u64) -> Result<Option<String>, String> {
        let items = {
            let mut g = self.state.lock();
            match g.current.as_mut() {
                Some(c) if c.turn == turn => {
                    if let Some(error) = c.input_error.take() {
                        return Err(error);
                    }
                    c.inbox.claim_for_step()
                }
                _ => Vec::new(),
            }
        };
        if self.cfg.require_complete_observation {
            prompt::render_complete_inbox(&items)
        } else {
            Ok(prompt::render_inbox(&items))
        }
    }

    /// 交还槽位，返回收件箱里剩下的东西。turn 号对不上就什么都不动（槽位已经属于下一个
    /// turn 了）。
    fn release_slot(&self, turn: u64) -> Leftover {
        let mut g = self.state.lock();
        if !g.current.as_ref().is_some_and(|c| c.turn == turn) {
            return Leftover::default();
        }
        let mut c = g.current.take().expect("just checked");
        drop(g);
        let mut left = Leftover::default();
        // `claim_for_step` 一次只放一条 followup（三 lane 的规则），所以要数干净得循环，
        // 直到它什么也认领不出来为止。这个收件箱跟着 turn 一起结束了，排空即丢弃。
        loop {
            let claimed = c.inbox.claim_for_step();
            if claimed.is_empty() {
                return left;
            }
            for i in claimed {
                if i.lane == Lane::Followup {
                    left.followups += 1;
                } else {
                    left.next_step += 1;
                }
            }
        }
    }

    /// 一条 `brain/*`。这个脑写的事件绝大多数是**记账**，salience 0 就够：它们的读者是
    /// 排查问题的人，不是按 salience 过滤的仪表盘。两个例外走
    /// [`Shared::emit_salient`] / [`Shared::emit_at`]：`brain/answer`（150）和
    /// `brain/review`（100）。
    fn emit(&self, turn: &TurnHandle, kind: &str, payload: Value) {
        self.emit_at(&turn.trace_id, kind, 0, payload);
    }

    /// 一条带 salience 的 `brain/*`：`brain_id` 由这里补上（每条都要有，别让调用处一个个
    /// 记得写），`trace_id` 来自句柄（§12：一个 turn 的 `brain/*` 与 `act/*` 靠它串成一
    /// 条链）。
    fn emit_salient(&self, turn: &TurnHandle, kind: &str, salience: u8, payload: Value) {
        self.emit_at(&turn.trace_id, kind, salience, payload);
    }

    /// 同上，但 trace_id 由调用方给。会话末那次 review 跑在 detach 之后，那时已经没有
    /// 活着的句柄了，可那次 review 仍然属于它总结的那几个 turn 的链。
    fn emit_at(&self, trace: &str, kind: &str, salience: u8, mut payload: Value) {
        if let Value::Object(m) = &mut payload {
            m.insert("brain_id".into(), Value::String(self.cfg.brain_id.clone()));
        }
        self.log.append(
            EventDraft::new(EventKind::new(kind).expect("static kind"), SOURCE, payload)
                .with_salience(salience)
                .with_trace_id(trace),
        );
    }

    // -----------------------------------------------------------------------
    // RSI review fork
    // -----------------------------------------------------------------------

    /// 一个 turn 跑完之后：计数，够数就在 `Maintenance` 相位里跑一次 review。
    ///
    /// `is_current` 的检查在计数**之后**：这个 turn 确实发生过，哪怕它的终态已经被别人
    /// 收走了，下一次 review 该把它算进去。
    async fn after_turn(&self, turn: &TurnHandle) {
        let n = self.turns_since_review.fetch_add(1, Ordering::SeqCst) + 1;
        let every = u64::from(self.cfg.review.every_turns);
        if every == 0 || n < every || self.store.is_none() || !turn.is_current() {
            return;
        }
        // `Maintenance`：对唤醒来说等价于 Running（只 latch，不重复开 turn），但**不受
        // 内核的 turn 超时约束**——一个已经做完的 turn 不该因为它的维护相位多花了几秒
        // 就被超时扫描记成 `brain/error`。代价是这一段要自己有界、自己接住 panic，两件
        // 事都在 `Shared::review` 里（界是 `review.timeout_ms`，不可为 0）。
        //
        // 进出都交给 RAII 守卫而不是写成前后两行：这样两边都由类型系统保证，哪怕
        // `review` 有一天从某条新路径上 panic 出去，栈展开也会把"出"执行掉。漏掉"出"
        // 会让这个脑永远只 latch 不开 turn；漏掉"进"会让这个 turn 整段 review 都暴露
        // 在内核的超时扫描下（那正是维护相位要挡的那个误诊）。两种都没有任何东西能事
        // 后救回来——不是值得靠"记得写那两行"来维持的不变量。
        let _phase = MaintenanceGuard::enter(turn);
        // 外层再包一次 `catch_unwind`：`run_turn` 那层只包了 step 循环
        // （`after_turn` 是在它之后 await 的），而这里漏出去的 panic 会连 `done` 一起
        // 吃掉——turn 挂着、没有 `brain/turn_end`、也没有超时扫描会来收尾。
        let _ = AssertUnwindSafe(self.review(&turn.trace_id, TRIGGER_EVERY_TURNS))
            .catch_unwind()
            .await;
    }

    /// 跑一次 review 并把结局写进世界日志。**永远不返回错误、永远不 panic、永远有界**
    /// ——维护失败不是 turn 失败，维护慢也不该变成一个冻住的脑。
    ///
    /// 三种收场在日志里看得出区别：`error: "timeout"`（整段超了
    /// [`ReviewConfig::timeout_ms`]）、`error: "panicked"`、以及 `run_review` 自己记下的
    /// 那句话（模型流断、工具批超时）。三种都保留**已经跑出来的**部分报告：写已经落盘
    /// 入账了，`writes` 必须说得出它们，前缀也必须跟着重建。
    async fn review(&self, trace: &str, trigger: &str) {
        let Some(store) = self.store.clone() else {
            return;
        };
        let Some(_in_flight) = ReviewGuard::acquire(&self.review_in_flight) else {
            // 已经有一个 review 在飞（detach 撞上了一次正在跑的周期 review）。跳过而不
            // 是排队：那一个已经把 `recent` 取走了，它正在总结的就是这段窗口；排在它后
            // 面只会拿着一个空窗口再问模型一次。
            tracing::debug!(
                brain_id = %self.cfg.brain_id,
                trigger,
                "skipping an RSI review: another one is already running"
            );
            return;
        };
        let recent = self.take_recent();
        let report = Mutex::new(ReviewReport::default());
        // 整段的硬上限。`run_review` 的 future 到点就被丢掉，所以进度只能从 `report`
        // 里捞——它是一边跑一边填的，正是为了这一刻。
        let budget = Duration::from_millis(self.cfg.review.timeout_ms.max(1));
        let outcome = tokio::time::timeout(
            budget,
            AssertUnwindSafe(review::run_review(
                self,
                store.clone(),
                &self.cfg.review,
                trace,
                &recent,
                &report,
            ))
            .catch_unwind(),
        )
        .await;
        let mut report = report.into_inner();
        match outcome {
            Ok(Ok(())) => {}
            // panic 也要有终态，理由和 `run_turn` 那一层逐字相同——只是这里的终态是一条
            // `brain/review{error}`，而不是一个 `done`。
            Ok(Err(_)) => report.error = Some("panicked".into()),
            Err(_) => report.error = Some("timeout".into()),
        }
        // 计数在这里清零而不是在触发处：出错的那一次也算跑过了，否则一个每次都失败的
        // review 会在此后的**每一个** turn 上重跑。
        self.turns_since_review.store(0, Ordering::SeqCst);
        self.finish_review(&store, trace, trigger, report).await;
    }

    /// 收尾：对一遍账本、重建 system 前缀、记一条 `brain/review`。
    ///
    /// 读盘都合在一个 `spawn_blocking` 里：它们是同步 I/O，而这里跑在 runtime 的工作线
    /// 程上。
    ///
    /// **`writes` 要和账本对一遍**，不能只信 review 自己记下的那份。知识工具的 store
    /// 调用跑在 `tokio::task::spawn_blocking` 里，而阻塞任务是**取消不掉**的：整段超时
    /// （或者 panic）把 review 的 future 丢掉时，一次在飞的写会照常跑完、落盘、入账，
    /// 只是它的返回值再也回不到那份报告里了。不对账的话，`brain/review{writes: []}` 会
    /// 否认一个**确实**在账本上的改动，而 `ledger_seq` 又恰好指着它——同一条事件里两个
    /// 字段互相矛盾。
    ///
    /// 基线是上一次收尾时的 seq（[`Shared::last_ledger_seq`]），署名是这个 fork 自己的
    /// （同一个 store 可能被同一台机器人的另一个脑共用）。
    async fn finish_review(
        &self,
        store: &Arc<KnowledgeStore>,
        trace: &str,
        trigger: &str,
        report: ReviewReport,
    ) {
        let st = store.clone();
        // 异常收场（超时 / panic）时**无条件**重建前缀：那正是"有没有写成"说不准的情
        // 况，而前缀是从磁盘上的 skill 索引渲染出来的，所以重建一次就是自愈。渲染是确
        // 定性的，真的什么都没变时新前缀和旧的逐字节相同，不会白白作废一次 prompt 缓存。
        let rebuild = !report.writes.is_empty() || report.error.is_some();
        let persona = self.cfg.persona.clone();
        // 前缀里的工具表用**这一刻**投影里的那一份，不是构造期那一份：能力面已经可以
        // 在 turn 之间变了（[`Shared::turn_projection`]），拿一份过期的 specs 重渲染等于
        // 把刚投上去的工具从前缀里撤下来。
        let specs = self.projection.lock().specs.clone();
        let baseline = self.last_ledger_seq.load(Ordering::SeqCst);
        let actor = self.review_actor.clone();
        let (system, skills, ledger_seq, recovered) = tokio::task::spawn_blocking(move || {
            let skills = rebuild.then(|| review::skill_index(&st));
            let system = skills
                .as_ref()
                .map(|sk| prompt::system_prefix(&persona, &specs, sk));
            // 账本读不动就报 0 / 空：这条 payload 是给人对账用的，一个读不出来的账本不
            // 该让整条 `brain/review` 消失。
            let rows = st.ledger().unwrap_or_default();
            let seq = rows.iter().map(|e| e.seq).max().unwrap_or(0);
            let recovered: Vec<String> = rows
                .into_iter()
                .filter(|e| e.seq > baseline && e.actor == actor)
                .map(|e| e.rel)
                .collect();
            (system, skills, seq, recovered)
        })
        .await
        .unwrap_or((None, None, 0, Vec::new()));
        if let Some(skills) = skills {
            // 缓存跟着一起换：下一次因为**能力面**变化而重渲染前缀时用的就是这一份，
            // 从磁盘再读一遍只会多一条会漂移的真相。
            *self.skills.lock() = skills;
        }
        if let Some(system) = system {
            // 这一下就是 prompt 缓存失效：下一个 turn 的前缀和上一个不再逐字节相同。
            *self.system.lock() = system;
        }
        // `fetch_max`，不是 `store`：基线只许往前走。读不动账本（或者整个
        // `spawn_blocking` panic 掉）的那一次上面把 `seq` 兜成了 0，而一个 0 基线会让
        // **下一次** review 把这个 actor 有史以来写过的每一行都算成那一次的写入——一条
        // 谁也对不上的 `brain/review{writes}`。
        self.last_ledger_seq.fetch_max(ledger_seq, Ordering::SeqCst);
        // review 自己看见的那些在前（它们的顺序是模型的动作顺序），账本补上的跟在后面。
        let mut writes = report.writes;
        for rel in recovered {
            if !writes.contains(&rel) {
                writes.push(rel);
            }
        }
        let mut payload = json!({
            "trace_id": trace,
            "trigger": trigger,
            "steps": report.steps,
            "writes": writes,
            "ledger_seq": ledger_seq,
        });
        if let Some(e) = &report.error {
            payload["error"] = Value::String(e.clone());
        }
        self.emit_at(trace, "brain/review", SAL_REVIEW, payload);
    }

    /// 一条会话记录：攒给下一次 review，原样还给调用方（去 `turn.note`）。
    ///
    /// 攒的是**克隆**，不是会话日志文件里那一份：会话日志是异步落盘的有界队列，过载时
    /// 会丢行，而 review 的取景框不该跟着盘的脾气变。
    fn push_recent(&self, e: SessionEntry) -> SessionEntry {
        if self.store.is_none()
            || (self.cfg.review.every_turns == 0 && !self.cfg.review.on_session_end)
        {
            return e;
        }
        let mut g = self.recent.lock();
        if g.len() >= MAX_RECENT_ENTRIES {
            g.pop_front();
        }
        g.push_back(e.clone());
        e
    }

    /// 取走攒下的记录（下一次 review 从空开始）。
    ///
    /// **取走就不还**，哪怕这次 review 失败了：不还的代价是那一段活动不会被再看一次，
    /// 还回去的代价是"这个窗口到底盖着哪几个 turn"变成一件要读两个字段才说得清的事
    /// （计数器无论成败都归零）。而且这不是数据丢失——那些记录的耐久副本在会话日志
    /// 里，人和以后的工具照样读得到。
    fn take_recent(&self) -> Vec<SessionEntry> {
        std::mem::take(&mut *self.recent.lock()).into()
    }
}

/// 把一条模型事件流收进 `out`，返回是否出错。**任何一种收不完的结局都把已经收到的部分
/// 留在 `out` 里。**
///
/// `Done` 是流的最后一条，收到就停——之后还来的事件属于一个已经结束的回合，读下去只会
/// 把下一次调用的内容混进这一次。三种收不完：
///
/// - 流在 `Done` 之前就没了（连接断、传输忘了发）→ `StopReason::Other`，**无错**；
/// - 流上来一条 `Err`（SSE 解析失败、连接中途重置）→ 带着已收到的部分 + 这个错；
/// - 超过 `deadline` → 同上，错是 [`ModelError::Timeout`]。
///
/// 截止时刻加在**每一次 `await`** 上而不是包住整个函数：`tokio::time::timeout(whole)`
/// 超时会把整个 future 连同 `text`/`calls` 一起 drop。
async fn collect(
    mut stream: EventStream,
    deadline: Option<tokio::time::Instant>,
    out: &mut ModelOut,
) -> Option<ModelError> {
    loop {
        let next = match by(deadline, stream.next()).await {
            Ok(n) => n,
            Err(_) => return Some(ModelError::Timeout),
        };
        let ev = next?;
        match ev {
            Ok(ModelEvent::TextDelta(d)) => out.text.push_str(&d),
            Ok(ModelEvent::ToolCall(c)) => out.calls.push(c),
            Ok(ModelEvent::Usage {
                input,
                output,
                cached,
                reasoning,
            }) => out.usage.push((input, output, cached, reasoning)),
            Ok(ModelEvent::Warning(w)) => out.warnings.push(w),
            Ok(ModelEvent::Done { stop }) => {
                out.stop = stop;
                return None;
            }
            Err(e) => return Some(e),
        }
    }
}

/// `f`，但最晚到 `deadline`。`None` = 不设限。
async fn by<F: std::future::Future>(
    deadline: Option<tokio::time::Instant>,
    f: F,
) -> Result<F::Output, tokio::time::error::Elapsed> {
    match deadline {
        Some(d) => tokio::time::timeout_at(d, f).await,
        None => Ok(f.await),
    }
}

/// `brain/request` 里的三个身份字段。
struct Identity {
    provider: String,
    model: String,
    api_mode: String,
}

impl Identity {
    /// 三个字段都从模型自己那里来：名字是 `<provider>/<model>`（传输按 `Runtime` 拼），
    /// wire 是 [`Model::api_mode`]。没有斜杠的名字（剧本模型、测试替身）provider 和
    /// model 就都是它——一个不出网的模型没有"厂商"可言。
    fn of(model: &dyn Model) -> Identity {
        let name = model.name();
        let (provider, model_id) = split_name(&name);
        Identity {
            provider,
            model: model_id,
            api_mode: model.api_mode(),
        }
    }
}

/// `openai/gpt-5` → `("openai", "gpt-5")`；没有斜杠就两边都是原名。
fn split_name(name: &str) -> (String, String) {
    match name.split_once('/') {
        Some((p, m)) => (p.to_string(), m.to_string()),
        None => (name.to_string(), name.to_string()),
    }
}

/// 工具参数的线上大小。全文不进世界日志（可能含地址、人名），长度进——排查"模型往一个
/// 工具里塞了 40 KB"时只需要这个数。
fn args_bytes(args: &Value) -> usize {
    serde_json::to_string(args).map(|s| s.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oeh_kernel::observation::{CollapsedEvent, WakeReason};
    use oeh_kernel::wake::WakeAction;

    /// 一条折叠过的 `brain/ask`：`first`/`last` 是两次不同的提问。
    fn folded_ask(first_seq: u64, last_seq: u64) -> CollapsedEvent {
        let ask = |id: &str, from: &str| json!({"ask_id": id, "from": from, "q": "?"});
        CollapsedEvent {
            kind: ASK_KIND.into(),
            collapse_key: ASK_KIND.into(),
            first_seq,
            last_seq,
            count: 2,
            salience: 150,
            first: ask("a-1", "fast"),
            last: ask("a-2", "other"),
            first_frame_refs: Vec::new(),
            last_frame_refs: Vec::new(),
        }
    }

    fn woken_by(kind: &str, seq: u64, events: Vec<CollapsedEvent>) -> Observation {
        Observation {
            t_kernel: 0,
            wake_reason: WakeReason {
                kind: kind.into(),
                seq,
                action: WakeAction::Wake,
            },
            state: Default::default(),
            events,
            source_seq_range: (0, 99),
            truncated: 0,
        }
    }

    /// 两条提问折进同一条观测时，回答贴的是**模型真的读到的那一条**。
    /// [`prompt::render_observation`] 只渲染 `last`，所以 `ask_id`/`to` 也只能是 `last`
    /// 的——哪怕唤醒证据正是 `first`（Running 时 latch 住的唤醒 + 被预算抑制的后一条提问
    /// 就会这样）。
    #[test]
    fn a_folded_pair_of_asks_is_answered_as_the_one_the_prompt_rendered() {
        let rendered = prompt::render_observation(&woken_by(ASK_KIND, 7, vec![folded_ask(7, 9)]));
        assert!(rendered.contains("a-2"), "{rendered}");
        assert!(!rendered.contains("a-1"), "{rendered}");
        for wake_seq in [7, 8, 9] {
            let obs = woken_by(ASK_KIND, wake_seq, vec![folded_ask(7, 9)]);
            let a = AskedBy::of(&obs).expect("被提问叫醒的 turn");
            assert_eq!((a.ask_id.as_str(), a.from.as_str()), ("a-2", "other"));
        }
    }

    /// review 的互斥令牌：第二个拿不到，第一个放手之后又能拿到了。
    ///
    /// 这就是 detach 撞上一次在飞的周期 review 时发生的事——`abort_current()` 只是**预
    /// 约**取消（任务要到下一个 await 点才真的停），所以会话末那一次会在前一次还没结束
    /// 时被派生出来。
    #[test]
    fn only_one_review_can_be_in_flight_at_a_time() {
        let flag = std::sync::atomic::AtomicBool::new(false);
        let first = ReviewGuard::acquire(&flag).expect("第一个该拿得到");
        assert!(
            ReviewGuard::acquire(&flag).is_none(),
            "第二个 review 必须被挡在外面"
        );
        drop(first);
        assert!(
            ReviewGuard::acquire(&flag).is_some(),
            "放手之后下一次 review 该能拿到"
        );
    }

    /// 令牌在**栈展开**时也会还回去。没有这条保证的话，一次 panic 出去的 review 会把
    /// 标志永久留在"在飞"上，而症状是这个脑此后再也不 review 了——一个只在出过一次错
    /// 之后才显现的静默失效，没人会把它和那次 panic 联系起来。
    #[test]
    fn a_panicking_review_still_hands_the_token_back() {
        let flag = std::sync::atomic::AtomicBool::new(false);
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _g = ReviewGuard::acquire(&flag).expect("拿得到");
            panic!("boom");
        }));
        assert!(r.is_err());
        assert!(ReviewGuard::acquire(&flag).is_some(), "标志没还回来");
    }

    #[test]
    fn a_turn_woken_by_anything_else_has_nobody_to_answer() {
        let obs = woken_by("obs/gesture", 7, vec![folded_ask(7, 9)]);
        assert!(AskedBy::of(&obs).is_none());
        // 提问叫醒的，但那条证据不在观测里（预算把它挤掉了）：同样没有可答的对象，
        // 而不是拿另一条 `brain/ask` 顶上。
        let obs = woken_by(ASK_KIND, 42, vec![folded_ask(7, 9)]);
        assert!(AskedBy::of(&obs).is_none());
    }

    fn say(text: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: SAY_TOOL.into(),
            args: json!({ "text": text }),
        }
    }

    #[test]
    fn only_a_successful_say_answers_and_an_over_long_one_says_it_was_cut() {
        let a = AskedBy {
            ask_id: "a-1".into(),
            from: "fast".into(),
        };
        let ok = ToolOutcome::ok(json!({}));
        let p = a.answer(&say("yes"), &ok).expect("一次成功的 say 就是回答");
        assert_eq!(p["to"], "fast");
        assert_eq!(p["ask_id"], "a-1");
        assert_eq!(p["text"], "yes");
        assert!(p.get("truncated").is_none(), "没截断就不该有这个字段");
        // 超长：截到 1 KiB 并明说截过——不然提问者会把半句话当成全文。
        let long = a
            .answer(&say(&"x".repeat(MAX_ANSWER_TEXT * 2)), &ok)
            .expect("超长的 say 照样是回答");
        assert_eq!(long["text"].as_str().map(str::len), Some(MAX_ANSWER_TEXT));
        assert_eq!(long["truncated"], true);
        // 被拒的 say 没说出口，不是回答；别的工具也不是。
        assert!(a
            .answer(&say("yes"), &ToolOutcome::err("refused"))
            .is_none());
        let mut other = say("yes");
        other.name = "robot_walk".into();
        assert!(a.answer(&other, &ok).is_none());
    }

    /// 一个只报名字和 wire 的模型：`Identity` 只读这两样。
    struct Named(&'static str, &'static str);

    impl Model for Named {
        fn name(&self) -> String {
            self.0.to_string()
        }
        fn api_mode(&self) -> String {
            self.1.to_string()
        }
        fn complete(&self, _req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
            Box::pin(async { Err(ModelError::Transport("not a real model".into())) })
        }
    }

    #[test]
    fn a_transport_reports_its_provider_model_and_wire() {
        let i = Identity::of(&Named("openai/gpt-5", "responses"));
        assert_eq!((i.provider.as_str(), i.model.as_str()), ("openai", "gpt-5"));
        assert_eq!(i.api_mode, "responses");
    }

    #[test]
    fn a_bare_model_name_is_its_own_provider() {
        let i = Identity::of(&Named("scripted", "scripted"));
        assert_eq!(i.provider, "scripted");
        assert_eq!(i.model, "scripted");
        assert_eq!(i.api_mode, "scripted");
    }

    #[test]
    fn args_bytes_counts_the_compact_encoding() {
        assert_eq!(args_bytes(&json!({})), 2);
        assert_eq!(args_bytes(&json!({"vx": 0.1})), r#"{"vx":0.1}"#.len());
    }

    #[test]
    fn the_default_config_matches_the_documented_numbers() {
        let d = NativeConfig::default();
        assert_eq!(d.max_steps, 8);
        assert_eq!(d.step_timeout_ms, 60_000);
    }

    #[tokio::test]
    async fn provider_usage_is_preserved_including_before_a_stream_error() {
        let stream = Box::pin(futures_util::stream::iter(vec![
            Ok(ModelEvent::Warning("fallback model metadata".into())),
            Ok(ModelEvent::Usage {
                input: 123,
                output: 17,
                cached: 80,
                reasoning: Some(9),
            }),
            Err(ModelError::Transport("fixture disconnect".into())),
        ]));
        let mut out = ModelOut {
            text: String::new(),
            calls: vec![],
            stop: StopReason::Other,
            usage: vec![],
            warnings: vec![],
        };
        assert!(collect(stream, None, &mut out).await.is_some());
        assert_eq!(out.usage, vec![(123, 17, 80, Some(9))]);
        assert_eq!(out.warnings, vec!["fallback model metadata".to_string()]);
    }
}
