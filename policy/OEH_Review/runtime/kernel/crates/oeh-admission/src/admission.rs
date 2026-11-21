//! 准入服务：脑的提案（`Proposal`）在这里变成一次真正的执行，或者被挡下来。
//!
//! 一次提案走的流水线是固定的六步，每一步都在世界日志里留痕（全部带发起它的
//! `trace_id`，payload 里都有 `brain_id` / `proposal_id` / `capability`）：
//!
//! 1. `act/proposed`——收到了。
//! 2. **词汇**：能力不在注册表里、或者 params 过不了能力自己的 schema →
//!    `act/refused{stage:"vocabulary"}`。
//! 3. **规则链**：快照（时钟、FSM 态、蒸馏状态、estop 标志）一次取齐，`evaluate_chain`
//!    按注册顺序跑完所有规则。想放松判定的规则记 `safety/tamper_attempt`（salience 255），
//!    夹过 params 的规则记 `act/clamped`。
//! 4. **判定**：`Deny` → `act/refused{stage:"rules"}`；`Ask` → `act/ask` → 审批服务
//!    （带超时，没配审批 = 拒绝）→ `act/decision` →（拒绝的话）`act/refused{stage:"approval"}`。
//! 5. **准入**：`act/admitted{verdict_trace}`。motion 类还要抢占正在跑的 motion
//!    （`act/preempted`）并把 FSM 驱到 `ACTING`。
//! 6. **执行**：`act/started` → 执行器 → `act/result`。执行期间规则链按 `recheck_ms`
//!    复检（`act/cancelled{reason:"rules"}`），租约监视看心跳（`act/lease_expired`，
//!    并回头叫一次**这个执行器自己**的 `Executor::stop`），FSM 离开 `ACTING` 或者急停
//!    按下也会撤销执行（`act/cancelled{reason:"fsm"|"estop"}`）。
//!
//! 审批点头之后还会拿**新鲜**快照再过一遍链（`act/refused{stage:"recheck"}` /
//! `act/recheck`）：人可能想了 20 秒，批准的不能是 20 秒前的那个世界。
//!
//! FSM 归属（见 `oeh_distill::fsm_driver` 的对账）：本服务拥有
//! `STANDING → ENGAGED → ACTING → STANDING`（被准入的 motion 起跑时两步推上去，最后一个
//! motion 以任何理由结束时退回 STANDING），以及内置能力 `recover` 的
//! `EMERGENCY_STOP|FAULT → RECOVERING → STANDING`。除此之外不碰 FSM。
//!
//! 执行那一半（登记、租约、抢占、收尾）住在 [`crate::execution`]：这个文件是"准入"
//! （判定与事件），那个文件是"执行"（一次调用的生老病死）。
//!
//! 锁的纪律：`Inner::s` 这把锁只在"取快照 / 改登记"这样的短临界区里持有，绝不跨
//! `append`、`await`、执行器调用或 FSM 迁移——`WorldLog::append` 会同步跑全部观察者，
//! `RobotFsm::transition` 里也有 `append`，带着锁进去就是死锁的配方。同理，`safety/fsm`
//! 与 `safety/estop` 的观察者只把事件 `send` 到内部通道，真正的撤销发生在自己的任务里。
//!
//! 反过来的一条：**世界日志的观察者不许回头调用 `Admission`**。`admit` gate 会跨着
//! `act()` 和 `fsm.transition()` 持有，而这两个都会同步跑完全部观察者——一个想在观察者
//! 里提案一次 motion 的插件会直接和自己死锁。观察者该做的是把事件 `send` 出去，在自己
//! 的任务里再回来。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::clock::Clock;
use oeh_kernel::frame_store::FrameStore;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::Disposer;
use oeh_proto::{EventDraft, EventKind, FrameRef, WorldEvent};
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::approval::{decide_with_timeout, Approval, AskRequest, Decision};
use crate::capability::{
    validate_params, CancelReason, CapabilityRegistry, CapabilitySpec, ExecControl,
};
use crate::execution::{start, ExecEntry};
use crate::verdict::{evaluate_chain, ChainResult, Rule, RuleCtx, RuleTrace, Verdict};

/// 本服务写进世界日志的 `source`，同时也是"这条 `safety/fsm` 是我自己发的"的判据。
pub const SOURCE: &str = "admission";

/// 流水线在给出任何终局事件之前就被丢掉（运行时关停、任务被 abort）时的撤销理由。
pub(crate) const ABANDONED: &str = "abandoned";

/// 服务被撤销时写进登记的撤销理由。不是 `CancelReason`：它不是任何一次"正常"的撤销，
/// 而是"整个准入服务没了"。
const DISPOSED: &str = "admission disposed";

/// `act/started` / `act/result` 带的帧引用取这条流的最新一帧。
///
/// 为什么是 lowstate：规则链判定时读的那几个事实（姿态、FSM 态、流够不够新鲜）全是从
/// 这条流蒸出来的，所以"机器人是在看着哪一帧开始动的"问的就是它。相机流各有各的名字
/// 和帧率，而 lowstate 是每台机器人都有的那一条。
const LOWSTATE_STREAM: &str = "lowstate";

const SAL_PROPOSED: u8 = 100;
pub(crate) const SAL_STARTED: u8 = 100;
const SAL_CLAMPED: u8 = 120;
pub(crate) const SAL_RESULT: u8 = 120;
const SAL_ADMITTED: u8 = 150;
const SAL_REFUSED: u8 = 150;
const SAL_DECISION: u8 = 150;
pub(crate) const SAL_PREEMPTED: u8 = 150;
const SAL_ASK: u8 = 180;
const SAL_RECHECK: u8 = 150;
pub(crate) const SAL_CANCELLED: u8 = 180;
pub(crate) const SAL_LEASE: u8 = 200;
const SAL_TAMPER: u8 = 255;
const SAL_RULE_ERROR: u8 = 200;

/// 准入服务的运行参数。
#[derive(Clone, Debug, PartialEq)]
pub struct AdmissionConfig {
    /// 必须出现在规则链里的规则名。缺一条，链的初始判定就是 `Ask`（见 `evaluate_chain`）——
    /// 配错规则集的结果是"什么都要人来确认"，而不是"什么都放行"。
    pub required_rules: Vec<String>,
    /// 一次 `Ask` 等审批的上限；超时算拒绝。
    pub ask_timeout_ms: u64,
    /// 租约监视的扫描周期。
    pub lease_check_ms: u64,
    /// 租约过期之后还给失联执行器多久：这段时间里它占着 motion 的位子（新的 motion 只能
    /// 走抢占那条明确的路），过完还没回来就强行摘掉登记、交还 ACTING。
    pub lease_grace_ms: u64,
    /// 执行期间规则复检的周期（规格 §5.3：`walk` 这类持续能力由 `rule-scene` 以 5 Hz
    /// 复检，失败即 cancel）。0 = 关闭复检。
    pub recheck_ms: u64,
    /// 急停标志文件：存在即视为急停按下，作为只读事实传给规则链。
    pub estop_flag: PathBuf,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            required_rules: Vec::new(),
            ask_timeout_ms: 20_000,
            lease_check_ms: 100,
            lease_grace_ms: 500,
            recheck_ms: 200,
            estop_flag: PathBuf::from("/tmp/oeh_estop"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    #[error("Admission must be constructed inside a tokio runtime")]
    NoRuntime,
}

#[derive(Default)]
pub(crate) struct AdmState {
    /// 按注册顺序评估；`u64` 是注册 id，撤销只拆自己那一次。
    rules: Vec<(u64, String, Arc<dyn Rule>)>,
    next_rule_id: u64,
    /// 同一时刻至多一个审批服务；后设者覆盖前一个。
    approval: Option<(u64, Arc<dyn Approval>)>,
    next_approval_id: u64,
    pub(crate) execs: HashMap<ProposalId, ExecEntry>,
    /// 登记世代号的发号器：同一个 `ProposalId` 被重新登记时世代必然不同。
    pub(crate) next_exec_epoch: u64,
    /// 同一时刻至多一个 motion 在跑；新的 motion 被准入时抢占它。
    pub(crate) active_motion: Option<ProposalId>,
}

impl AdmState {
    pub(crate) fn rules_snapshot(&self) -> Vec<(String, Arc<dyn Rule>)> {
        self.rules
            .iter()
            .map(|(_, name, r)| (name.clone(), r.clone()))
            .collect()
    }
}

pub(crate) struct Inner {
    /// 构造时捕获的 runtime 句柄：流水线跑在它 spawn 出来的任务上，提案方从哪个线程
    /// 调 `propose` 都一样。
    rt: tokio::runtime::Handle,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) log: Arc<WorldLog>,
    state: Arc<StateRegistry>,
    /// 可选的帧存储：有它才写得出 `act/started` / `act/result` 的 `frame_refs`。内置
    /// `admission` 插件用 `try_get` 取，取不到就是 `None`（事件照发，只是没有帧引用）。
    frames: Option<Arc<FrameStore>>,
    pub(crate) fsm: Arc<RobotFsm>,
    pub(crate) caps: CapabilityRegistry,
    cfg: AdmissionConfig,
    pub(crate) s: Mutex<AdmState>,
    /// motion 的"抢占 → 驱动 FSM → 登记"是一个整体：两个并发被准入的 motion 否则会
    /// 各自看到"没有 motion 在跑"，一起起跑。用 async 锁而不是 `parking_lot`——临界区里
    /// 要 append、要迁移 FSM，那些操作会同步回调世界日志的观察者。
    pub(crate) admit: tokio::sync::Mutex<()>,
    /// 已经 spawn 出去的流水线任务。攥着它们有两个理由：服务被撤销时要把还在途的提案
    /// 一并 abort（`Terminal` 守卫随后给每一条补一句 `act/cancelled{reason:"abandoned"}`
    /// ——一条已经记过 `act/proposed` 的提案，不该因为服务没了就在审计里没有下文），
    /// 以及跑完的任务得有人收尸：`JoinSet` 会一直留着结果直到被 join。
    pipelines: Mutex<tokio::task::JoinSet<()>>,
}

/// 观察者送进内部任务的安全事件。观察者本身只做一次 `send`。
enum Safety {
    /// 一次不是本服务发起的 FSM 迁移。
    Fsm { seq: u64, to: String },
    /// 急停按下。
    Estop,
}

/// 一次 Deny 的根因，写成"哪条规则：为什么"。链里**第一条**给出 Deny 的规则就是根因：
/// `tighten` 在同紧度时留上游的理由，后面的规则只是把它原样传下去。
pub(crate) fn deny_detail(trace: &[RuleTrace], reason: &str) -> String {
    match trace
        .iter()
        .find(|t| matches!(t.outgoing, Verdict::Deny { .. }))
    {
        Some(t) => format!("{}: {reason}", t.rule),
        None => reason.to_string(),
    }
}

fn kind(s: &str) -> EventKind {
    EventKind::new(s).expect("static kind")
}

/// 一条 `act/*`，只需要日志本身。后台任务（比如租约过期时的兜底 `stop`）攥着的是
/// `Arc<WorldLog>` 而不是整个 `Inner`，走这一个入口，payload 约定不分两套。
pub(crate) fn act_on(log: &WorldLog, p: &Proposal, k: &str, salience: u8, extra: Value) {
    act_on_with_frames(log, p, k, salience, extra, Vec::new());
}

/// 同 [`act_on`]，外加帧引用。
fn act_on_with_frames(
    log: &WorldLog,
    p: &Proposal,
    k: &str,
    salience: u8,
    extra: Value,
    refs: Vec<FrameRef>,
) {
    log.append(
        EventDraft::new(kind(k), SOURCE, act_payload(p, extra))
            .with_salience(salience)
            .with_trace_id(p.trace_id.as_str())
            .with_frame_refs(refs),
    );
}

/// 每条 `act/*`（以及 `safety/tamper_attempt`）都带的三个键，加上这一步自己的字段。
fn act_payload(p: &Proposal, extra: Value) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("brain_id".into(), Value::String(p.brain_id.clone()));
    o.insert("proposal_id".into(), Value::String(p.proposal_id.0.clone()));
    o.insert("capability".into(), Value::String(p.capability.clone()));
    if let Value::Object(m) = extra {
        o.extend(m);
    }
    Value::Object(o)
}

impl Inner {
    /// 把跑完的流水线任务从 `JoinSet` 里摘掉。不这么做的话，`JoinSet` 会一直攥着每次
    /// 提案的结果，内存随提案数单调增长。
    pub(crate) fn reap_pipelines(&self) {
        let mut g = self.pipelines.lock();
        while g.try_join_next().is_some() {}
    }

    pub(crate) fn recheck_ms(&self) -> u64 {
        self.cfg.recheck_ms
    }

    pub(crate) fn lease_grace_ns(&self) -> u64 {
        self.cfg.lease_grace_ms.saturating_mul(1_000_000)
    }

    /// 急停按下了没有。两条独立的路取**或**：准入服务自己的标志文件
    /// （`AdmissionConfig::estop_flag`）和 `distill-estop` 写下的 `estop` 寄存器。两者
    /// 各有各的默认路径，一份只改了其中一个的 profile 不该悄悄把急停这条链解掉——
    /// fail-closed 的方向是"任一说按下了就按下了"。
    fn estop_engaged(&self) -> bool {
        self.cfg.estop_flag.exists() || self.state.get("estop") == Some(Value::Bool(true))
    }

    /// 取一次**新鲜**快照，把整条规则链在 `params` 上跑一遍。一次判定里读到的世界必须
    /// 是同一个瞬间的，所以时钟、FSM 态、蒸馏状态和 estop 一次取齐。准入、审批之后的
    /// 复检、执行期间的周期复检都走这一个入口。
    pub(crate) fn evaluate_chain_now(&self, spec: &CapabilitySpec, params: Value) -> ChainResult {
        let estop_engaged = self.estop_engaged();
        let snapshot = self.state.snapshot_entries();
        let rules = self.s.lock().rules_snapshot();
        let ctx = RuleCtx {
            now_ns: self.clock.now_ns(),
            fsm: self.fsm.state(),
            state: &snapshot,
            spec,
            estop_engaged,
        };
        evaluate_chain(&rules, &self.cfg.required_rules, params, &ctx)
    }

    pub(crate) fn act(&self, p: &Proposal, k: &str, salience: u8, extra: Value) {
        act_on(&self.log, p, k, salience, extra);
    }

    /// 写进 `act/started` / `act/result` 的帧引用：`lowstate` 流的最新一帧。
    /// 没装帧存储、或者一帧都还没到 → 空（事件照发，只是指不回任何一帧）。
    fn lowstate_refs(&self) -> Vec<FrameRef> {
        match self.frames.as_ref().and_then(|f| f.latest(LOWSTATE_STREAM)) {
            Some(f) => vec![f.frame_ref()],
            None => Vec::new(),
        }
    }

    /// 一条带帧引用的 `act/*`。只有执行这一段（`act/started` / `act/result`）用它：
    /// 一条"机器人开始动了 / 动完了"的事件该指回它当时看到的那一帧世界，否则回放的人
    /// 只能拿时间戳去猜哪一帧对应哪次动作。判定那一段（`act/proposed` … `act/admitted`）
    /// 不带：它们读的是蒸馏状态的快照，不是某一帧。
    pub(crate) fn act_framed(&self, p: &Proposal, k: &str, salience: u8, extra: Value) {
        act_on_with_frames(&self.log, p, k, salience, extra, self.lowstate_refs());
    }

    /// 链里想把判定放松的规则：`safety/tamper_attempt`（salience 255）。审计要看得出
    /// "哪条规则想放水"，而不是被它悄悄得逞。
    pub(crate) fn record_tamper(&self, p: &Proposal, trace: &[RuleTrace]) {
        for t in trace.iter().filter(|t| t.tamper) {
            self.log.append(
                EventDraft::new(
                    kind("safety/tamper_attempt"),
                    SOURCE,
                    act_payload(
                        p,
                        json!({"rule": t.rule, "incoming": t.incoming, "outgoing": t.outgoing}),
                    ),
                )
                .with_salience(SAL_TAMPER)
                .with_trace_id(p.trace_id.as_str()),
            );
        }
    }

    /// 链里 panic 掉的规则：`health/distiller_error{rule, reason:"panic", stage:"admission"}`
    /// （规格 §11："panic/超时 = deny"）。`evaluate_chain` 已经把这次判定变成 Deny 并在
    /// trace 里标了 `panicked`，但那只说明"这一次没放行"——一条坏掉的规则本身是要被看见
    /// 的健康事件，控制台和 Plan 3 的 `plugin/state` 都要读它。
    pub(crate) fn record_rule_errors(&self, p: &Proposal, trace: &[RuleTrace]) {
        for t in trace.iter().filter(|t| t.panicked) {
            self.log.append(
                EventDraft::new(
                    kind("health/distiller_error"),
                    SOURCE,
                    json!({"rule": t.rule, "reason": "panic", "stage": "admission"}),
                )
                .with_salience(SAL_RULE_ERROR)
                .with_trace_id(p.trace_id.as_str()),
            );
        }
    }

    pub(crate) fn refuse(
        &self,
        p: &Proposal,
        stage: &str,
        reason: String,
        trace: Option<&[RuleTrace]>,
    ) -> Outcome {
        let mut extra = json!({"stage": stage, "reason": reason});
        if let Some(t) = trace {
            extra["trace"] = json!(t);
        }
        self.act(p, "act/refused", SAL_REFUSED, extra);
        Outcome::Refused { reason }
    }
}

/// 准入服务。`Proposer` 的实现，同时也是能力注册表、规则链和审批服务的挂载点。
pub struct Admission {
    inner: Arc<Inner>,
    observer: Mutex<Option<Disposer>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl Drop for Admission {
    /// 先停任务再撤观察者（`Disposer` 是 `FnOnce`，光 drop 不会执行），然后把还在跑的
    /// 执行器全部撤销：服务都没了，就没人再看着它们的租约了。最后把 FSM 交还——撤销
    /// 之后没有 motion 了，留着 `ACTING` 会让接手的人（蒸馏器、下一次 boot 的日志读者）
    /// 以为机器人还在动。
    fn drop(&mut self) {
        for t in self.tasks.lock().drain(..) {
            t.abort();
        }
        // 还在途的流水线：abort 之后每条的 `Terminal` 守卫会补一句
        // `act/cancelled{reason:"abandoned"}`。一次已经记过 `act/proposed` 的提案，
        // 不该因为服务被撤销就在审计里没有下文。
        self.inner.pipelines.lock().abort_all();
        if let Some(d) = self.observer.lock().take() {
            d();
        }
        let live: Vec<Arc<ExecControl>> = self.inner.s.lock().cancel_all(DISPOSED);
        for c in live {
            c.cancel();
        }
        // 尽力而为：别人已经把 FSM 带去别处（急停、故障）时 `release_fsm_if_idle` 自己
        // 就不动手，非法迁移也只记一条 debug。gate 用 `try_lock`——`Drop` 不能 await，
        // 而抢不到就跳过是可以接受的：这一刻要么有个 `start` 正在 gate 里把 FSM 推向
        // ACTING（它自己会负责那一段），要么进程本来就在往下走，日志里最后一条 ACTING
        // 由下一次 boot 的蒸馏器对账。
        match self.inner.admit.try_lock() {
            Ok(_gate) => self.inner.release_fsm_if_idle("admission shutdown"),
            Err(_) => tracing::debug!("admit gate busy at shutdown; leaving the fsm alone"),
        }
    }
}

impl Admission {
    /// **必须在 tokio runtime 内调用**：服务自带两个任务（租约监视、`safety/fsm` 的消费），
    /// 和 `Brains` 一样捕获当前 `Handle`。runtime 外构造返回 `NoRuntime` 而不是 panic——
    /// 插件 `apply` 里的 panic 会穿过整棵 fiber 树。
    pub fn new(
        clock: Arc<dyn Clock>,
        log: Arc<WorldLog>,
        state: Arc<StateRegistry>,
        fsm: Arc<RobotFsm>,
        cfg: AdmissionConfig,
    ) -> Result<Arc<Self>, AdmissionError> {
        Self::build(clock, log, state, fsm, cfg, None)
    }

    /// 同 [`Admission::new`]，外加帧存储：`act/started` / `act/result` 会带上
    /// `lowstate` 流的最新帧引用。
    ///
    /// 为什么是第二个构造函数而不是给 `new` 加一个参数：帧存储对准入是**可选**的
    /// （内置 `admission` 插件用 `try_get` 取它，profile 里没挂帧存储照样能判定），
    /// 而现有的调用方一个字都不用改。
    pub fn with_frames(
        clock: Arc<dyn Clock>,
        log: Arc<WorldLog>,
        state: Arc<StateRegistry>,
        fsm: Arc<RobotFsm>,
        cfg: AdmissionConfig,
        frames: Arc<FrameStore>,
    ) -> Result<Arc<Self>, AdmissionError> {
        Self::build(clock, log, state, fsm, cfg, Some(frames))
    }

    fn build(
        clock: Arc<dyn Clock>,
        log: Arc<WorldLog>,
        state: Arc<StateRegistry>,
        fsm: Arc<RobotFsm>,
        cfg: AdmissionConfig,
        frames: Option<Arc<FrameStore>>,
    ) -> Result<Arc<Self>, AdmissionError> {
        let rt = tokio::runtime::Handle::try_current().map_err(|_| AdmissionError::NoRuntime)?;
        let period = Duration::from_millis(cfg.lease_check_ms.clamp(1, 60_000));
        // 能力注册表拿着同一份世界日志：能力面的易主（`capability/overridden` /
        // `capability/restored`）是准入这一侧的记账，和 `act/*` 走同一条路、同一个 source。
        let caps = CapabilityRegistry::with_log(log.clone());
        let inner = Arc::new(Inner {
            rt: rt.clone(),
            clock,
            log,
            state,
            frames,
            fsm,
            caps,
            cfg,
            s: Mutex::new(AdmState::default()),
            admit: tokio::sync::Mutex::new(()),
            pipelines: Mutex::new(tokio::task::JoinSet::new()),
        });

        // 租约监视。任务持 Weak：服务被撤销之后不能靠它把 `Inner` 吊着。
        let weak = Arc::downgrade(&inner);
        let lease = rt.spawn(async move {
            let mut iv = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
            iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                iv.tick().await;
                let Some(i) = weak.upgrade() else { break };
                i.sweep_leases().await;
            }
        });

        // 安全事件观察者 → 通道 → 任务。观察者跑在 `append` 的调用线程上，绝不能在那里
        // 碰 FSM 或者再 append（`RobotFsm::transition` 持着自己的锁调用 append，重入必
        // 死锁）。通道是无界的：这两类事件都很稀疏，而丢一条意味着一次该被撤销的运动
        // 没有被撤销——这种事件宁可排队也不能丢。
        let (tx, mut rx) = mpsc::unbounded_channel::<Safety>();
        let observer = inner.log.observe(Arc::new(move |ev: &WorldEvent| {
            match ev.kind.as_str() {
                // 自己发的那条不算：`act/started` 之后我们自己把 FSM 推到 ACTING。
                "safety/fsm" if ev.source != SOURCE => {
                    // 读不出 `to` 就当成"不是 ACTING"：宁可多撤一次，不可漏撤。
                    let to = ev
                        .payload
                        .get("to")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let _ = tx.send(Safety::Fsm { seq: ev.seq, to });
                }
                // 急停不看 source：谁发的都算（`distill-estop`、外部监听器、回放）。
                // fail-closed：`engaged` 读不出来（缺字段、类型不对）就当**按下了**，
                // 和上面那条"读不出 `to` 就当不是 ACTING"同一个方向——宁可多撤一次，
                // 不可漏撤。只有明确的 `engaged: false` 是"松开了"，那一条不撤任何东西
                // （否则它会把刚起跑的运动撤掉）。
                "safety/estop"
                    if ev.payload.get("engaged").and_then(|v| v.as_bool()) != Some(false) =>
                {
                    let _ = tx.send(Safety::Estop);
                }
                _ => {}
            }
        }));
        let weak = Arc::downgrade(&inner);
        let acting = FsmState::Acting.to_string();
        let safety_task = rt.spawn(async move {
            while let Some(sig) = rx.recv().await {
                let Some(i) = weak.upgrade() else { break };
                match sig {
                    Safety::Fsm { seq, to } if to != acting => {
                        // 只撤销这条事件之前就起跑的 motion：消费是异步的，排在前面的
                        // 旧事件（比如开机时那条 `→ STANDING`）不能反过来把后起跑的
                        // 运动撤了。
                        i.cancel_active_motion_since(seq, CancelReason::Fsm);
                    }
                    Safety::Fsm { .. } => {}
                    // 急停没有"起跑得比它晚"这回事：按下的那一刻，所有还在动的都得停。
                    Safety::Estop => {
                        i.cancel_active_motion(CancelReason::Estop);
                    }
                }
            }
        });

        Ok(Arc::new(Self {
            inner,
            observer: Mutex::new(Some(observer)),
            tasks: Mutex::new(vec![lease, safety_task]),
        }))
    }

    pub fn capabilities(&self) -> &CapabilityRegistry {
        &self.inner.caps
    }

    /// 测试用：同步跑一次租约扫描（生产路径由 `lease_check_ms` 的定时任务调用**同一个**
    /// 函数）。
    ///
    /// 有它，租约过期就完全由注入的 `Clock` 说了算：`ManualClock::advance` 把时间推过
    /// 租约，再调一次这里，`act/lease_expired` 当场就在日志里。没有它，测租约的唯一
    /// 办法是让 `MonotonicClock` 真的走完 TTL——那段等待既不是判据也不是被测行为，只是
    /// 墙钟快进不了。
    ///
    /// 为什么是"按快门"而不是"让定时任务走注入的时钟"：定时任务要的是**睡到**某个时刻，
    /// 而 `Clock` 只会报时（`now_ns`），叫不醒任何人。要让它驱动 `interval_at`，得先给
    /// `Clock` 加一套可等待的定时器语义并让每个用到它的地方都改过来——为了一个测试钩子
    /// 换掉内核的时间抽象，不值。
    #[doc(hidden)]
    pub async fn sweep_leases_now(&self) {
        self.inner.sweep_leases().await;
    }

    /// 注册一条规则。链按注册顺序评估；返回的 `Disposer` 只拆自己那一次注册。
    pub fn register_rule(&self, rule: Arc<dyn Rule>) -> Disposer {
        let name = rule.name().to_string();
        let id = {
            let mut g = self.inner.s.lock();
            g.next_rule_id += 1;
            let id = g.next_rule_id;
            g.rules.push((id, name, rule));
            id
        };
        let inner = self.inner.clone();
        Box::new(move || {
            inner.s.lock().rules.retain(|(i, _, _)| *i != id);
        })
    }

    /// 装上审批服务。同一时刻至多一个：后设者覆盖前一个，返回的 `Disposer` 只在装着的
    /// 仍是自己那一个时才清除。
    pub fn set_approval(&self, a: Arc<dyn Approval>) -> Disposer {
        let id = {
            let mut g = self.inner.s.lock();
            g.next_approval_id += 1;
            let id = g.next_approval_id;
            g.approval = Some((id, a));
            id
        };
        let inner = self.inner.clone();
        Box::new(move || {
            let mut g = inner.s.lock();
            if g.approval.as_ref().map(|(i, _)| *i) == Some(id) {
                g.approval = None;
            }
        })
    }

    /// 此刻装着的审批服务（没装 = `None`）。
    ///
    /// 给**不走准入链**、但仍然要问同一个人的调用方用：`brain-codex` 收到 codex 的
    /// `item/commandExecution/requestApproval` 时，要问的不是"这个能力能不能跑"（那是
    /// [`Admission::propose`] 的事，而 codex 要跑的是一条命令，根本不在能力注册表里），
    /// 而是"现在管事的那个人怎么说"。返回的是一份快照：之后 `set_approval` 换了人，
    /// 手上这个 `Arc` 仍然指向换之前的那一个——一次正在进行的询问不该因为审批端被替换
    /// 而无人应答。
    pub fn approval(&self) -> Option<Arc<dyn Approval>> {
        self.inner
            .s
            .lock()
            .approval
            .as_ref()
            .map(|(_, a)| a.clone())
    }

    /// 正在跑的 motion（至多一个）。
    pub fn active_motion(&self) -> Option<ProposalId> {
        self.inner.s.lock().active_motion.clone()
    }

    /// 撤销正在跑的 motion，记 `act/cancelled{reason}`。没有 motion 在跑 → `false`。
    pub fn cancel_active(&self, reason: CancelReason) -> bool {
        self.inner.cancel_active_motion(reason)
    }

    /// 撤销**指定的那一次**执行，记 `act/cancelled{reason}`。已经收尾、或者已经被别人
    /// 撤销过了 → `false`（认领是幂等的，一次提案只会有一个终局事件）。
    ///
    /// 和 [`Admission::cancel_active`] 的分工：那个问的是"现在跑着的 motion 是谁"，
    /// 这个是调用方**已经知道**该撤谁。跑这次执行的适配器断连就是后者——断的是一条
    /// 连接，它上面可能挂着好几次执行，而当前的 motion 完全可能是另一条连接的
    /// （`oeh-rpc` 的断连即释放，Plan 3 I4）。
    pub fn cancel_exec(&self, pid: &ProposalId, reason: CancelReason) -> bool {
        self.inner.cancel_exec(pid, None, reason, None)
    }

    /// 测试钩子：某次执行的登记世代号；登记不在了 → `None`。
    #[doc(hidden)]
    pub fn exec_epoch(&self, pid: &ProposalId) -> Option<u64> {
        self.inner.s.lock().exec_epoch(pid)
    }

    /// 测试钩子：原地跑一次"执行期间的规则复检"，跑的就是复检任务循环体里的那一句。
    /// 这四个参数正是一个复检任务从起跑那一刻起攥在手里的全部东西——传一组**过时**的
    /// 进来，就是在模拟"一个活过了自己那次执行的任务回来敲门"。返回"这个任务还该不该
    /// 继续"。
    #[doc(hidden)]
    pub fn recheck_now(
        &self,
        pid: &ProposalId,
        epoch: u64,
        spec: &CapabilitySpec,
        params: &Value,
    ) -> bool {
        self.inner.recheck_exec(pid, epoch, spec, params)
    }
}

impl Proposer for Admission {
    /// 流水线跑在自己的任务上。提案方（脑、Plan 3 的 RPC 会话）把返回的 future 丢掉，
    /// 不该让一次已经记了 `act/proposed` 的提案就此消失：登记、租约、FSM 都还等着它
    /// 收尾，而审计里一条没有终局事件的提案，读的人无从知道机器人到底动了没有。
    /// 调用方照常从返回的 future 里拿到同一个结局。
    fn propose(&self, p: Proposal) -> BoxFuture<Outcome> {
        let inner = self.inner.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let pipe = inner.clone();
        // 进 `JoinSet` 而不是裸 spawn：服务被撤销时要能把还在途的流水线一并 abort。
        inner.pipelines.lock().spawn_on(
            async move {
                let out = pipeline(pipe, p).await;
                let _ = tx.send(out);
            },
            &inner.rt,
        );
        Box::pin(async move {
            // 发送端随着被 abort 的 future 一起消失（服务撤销、运行时关停）：`Terminal`
            // 守卫这时已经在日志里补过 `act/cancelled{reason:"abandoned"}`。
            rx.await.unwrap_or(Outcome::Cancelled {
                reason: ABANDONED.into(),
            })
        })
    }
}

/// 一条 `act/proposed` 必须有一个终局事件。流水线要是在给出终局之前就被丢掉（运行时
/// 关停、任务被 abort），这个守卫补一条 `act/cancelled{reason:"abandoned"}`：审计里
/// 一条没有下文的提案，比一条明说"被放弃了"的提案难读得多。
///
/// 流水线本身已经 spawn 成独立任务，所以正常情况下这个守卫不会开火；留着是因为
/// "被 abort" 是一个真实存在的终点，而不变式（一次提案一个终局）不该依赖运气。
struct Terminal {
    inner: Arc<Inner>,
    p: Proposal,
    reached: bool,
}

impl Terminal {
    /// 终局马上就要记了，或者已经交给 [`crate::execution::start`] 那边保证。
    fn reached(&mut self) {
        self.reached = true;
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        if self.reached {
            return;
        }
        self.inner.act(
            &self.p,
            "act/cancelled",
            SAL_CANCELLED,
            json!({"reason": ABANDONED}),
        );
    }
}

/// 流水线本体。只捕获 `Arc<Inner>`：`Admission` 这层（任务句柄、观察者）被撤销之后，
/// 已经在途的提案照样能把自己收干净。
async fn pipeline(inner: Arc<Inner>, p: Proposal) -> Outcome {
    let mut terminal = Terminal {
        inner: inner.clone(),
        p: p.clone(),
        reached: false,
    };
    inner.act(
        &p,
        "act/proposed",
        SAL_PROPOSED,
        json!({"params": p.params}),
    );

    // 1. 词汇
    let Some((spec, exec)) = inner.caps.get(&p.capability) else {
        let reason = format!("unknown capability `{}`", p.capability);
        terminal.reached();
        return inner.refuse(&p, "vocabulary", reason, None);
    };
    if let Err(reason) = validate_params(&spec.params_schema, &p.params) {
        terminal.reached();
        return inner.refuse(&p, "vocabulary", reason, None);
    }

    // 2. 规则链。快照一次取齐：规则读到的世界必须是同一个瞬间的。
    let chain = inner.evaluate_chain_now(&spec, p.params.clone());
    inner.record_tamper(&p, &chain.trace);
    inner.record_rule_errors(&p, &chain.trace);
    for t in &chain.trace {
        if let Some(c) = &t.clamped {
            inner.act(
                &p,
                "act/clamped",
                SAL_CLAMPED,
                json!({"rule": t.rule, "before": c["before"], "after": c["after"]}),
            );
        }
    }
    let params = chain.params;

    // 3. 判定 / 审批
    if let Verdict::Deny { reason } = &chain.verdict {
        terminal.reached();
        return inner.refuse(&p, "rules", reason.clone(), Some(&chain.trace));
    }
    // 夹过的 params 要重新过一遍能力自己的 schema：规则可以收紧参数，但不能把它夹成一个
    // 执行器根本没法接的形状（类型不对、越过上下界）。步骤 2 验的是提案原样的那一份，
    // 这里验的是真正会跑的那一份。
    if params != p.params {
        if let Err(e) = validate_params(&spec.params_schema, &params) {
            terminal.reached();
            return inner.refuse(
                &p,
                "rules",
                format!("clamped params violate schema: {e}"),
                Some(&chain.trace),
            );
        }
    }
    // 还要人来确认的话，走审批；`Allow` 直接往下走。
    if let Verdict::Ask { reason } = &chain.verdict {
        let reason = reason.clone();
        // 审批拒绝时要把**两个**理由都还给脑：链为什么要问人（`ask_reason`），以及人
        // 为什么说不行。日志里两条都有（`act/ask` 和 `act/decision`），但脑只拿得到
        // 一个 `Outcome::Refused{reason}`——只给后者的话，它根本不知道自己被问了什么。
        let ask_reason = reason.clone();
        inner.act(&p, "act/ask", SAL_ASK, json!({"reason": reason}));
        let approval = inner.s.lock().approval.as_ref().map(|(_, a)| a.clone());
        let req = AskRequest {
            // 审批看到的是规则夹过之后、真正会跑的那份 params。
            proposal: Proposal {
                params: params.clone(),
                ..p.clone()
            },
            spec: spec.clone(),
            reason,
            trace: chain.trace.clone(),
        };
        let decision = decide_with_timeout(
            approval,
            req,
            Duration::from_millis(inner.cfg.ask_timeout_ms),
        )
        .await;
        match decision {
            Decision::Allow { by, reason } => {
                inner.act(
                    &p,
                    "act/decision",
                    SAL_DECISION,
                    json!({"decision": "allow", "by": by, "reason": reason}),
                );
                // 人可能想了 20 秒（`ask_timeout_ms` 默认就是 20 s）。放行的是**问之前**
                // 那个世界的判定；点头的这一刻，路上完全可能已经站了个人。起跑之前拿
                // 此刻的世界再过一遍链。夹取不重来：跑的是人批准的那一份 params。
                let re = inner.evaluate_chain_now(&spec, params.clone());
                inner.record_tamper(&p, &re.trace);
                inner.record_rule_errors(&p, &re.trace);
                match &re.verdict {
                    Verdict::Deny { reason } => {
                        terminal.reached();
                        return inner.refuse(&p, "recheck", reason.clone(), Some(&re.trace));
                    }
                    // 复检还想问人：人刚刚已经点过头了，不再问第二遍（那会没完没了），
                    // 但这一步要留痕——审批批的和最终起跑的不是同一份判定。
                    Verdict::Ask { reason } => inner.act(
                        &p,
                        "act/recheck",
                        SAL_RECHECK,
                        json!({"verdict": "ask", "reason": reason, "trace": re.trace}),
                    ),
                    Verdict::Allow => {}
                }
            }
            Decision::Deny { by, reason } => {
                inner.act(
                    &p,
                    "act/decision",
                    SAL_DECISION,
                    json!({"decision": "deny", "by": by, "reason": reason}),
                );
                terminal.reached();
                return inner.refuse(
                    &p,
                    "approval",
                    format!("{ask_reason}; {reason}"),
                    Some(&chain.trace),
                );
            }
        }
    }
    inner.act(
        &p,
        "act/admitted",
        SAL_ADMITTED,
        json!({"verdict_trace": chain.trace}),
    );

    // 4. 起跑。从这里开始终局由 `start` 那边保证：执行器跑在自己 spawn 出来的任务上，
    // 收尾（`act/result` / 撤销事件）绝不会跟着调用方一起消失。
    terminal.reached();
    start(&inner, p, spec, params, exec).await
}
