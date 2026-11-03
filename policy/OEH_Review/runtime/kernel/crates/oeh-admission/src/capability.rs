//! 能力面：一个能力（capability）是"脑可以提案要求内核做的一件事"。这个模块定义能力
//! 的静态描述（`CapabilitySpec`）、执行它的合约（`Executor`/`ExecCtx`/`ExecOutcome`）、
//! 把名字映射到实现的注册表（`CapabilityRegistry`），以及校验提案 params 的最小
//! JSON-schema 子集（`validate_params`）。
//!
//! 准入判定（规则链、审批、租约）不在这个模块里——这里只定义"能力是什么、怎么执行、
//! 参数长什么样"，判定逻辑属于 Plan 2b 后续任务。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use oeh_kernel::proposal::{BoxFuture, Proposal, ProposalId};
use oeh_kernel::{Clock, Disposer, WorldLog};
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::admission::SOURCE;

/// 内置能力的提供者名字。`robot/capabilities` 的 `provider` 字段报的就是它，而一个跨进程
/// 插件报的是自己的 `manifest.name`——两者不会撞，因为保留服务名之外的插件名不叫
/// `kernel` 也无所谓：这个字段只用来读，不用来路由。
pub const KERNEL_PROVIDER: &str = "kernel";

/// 内置能力被适配器顶替：`{name, by}`。
pub const KIND_OVERRIDDEN: &str = "capability/overridden";
/// 顶替它的那一位走了，内置能力回到能力面：`{name, from}`。
pub const KIND_RESTORED: &str = "capability/restored";
/// 一个**进程外**插件把一条能力挂上了能力面：`{name, provider, class}`。
pub const KIND_REGISTERED: &str = "capability/registered";
/// 它又下去了（插件断连、或者那一行 fiber 被撤销）：`{name, provider}`。
pub const KIND_UNREGISTERED: &str = "capability/unregistered";

/// 上面四条的 salience。
const SAL_CAPABILITY_FACE: u8 = 100;

/// 能力所属的大类，粗粒度地分组（比如规则链可以按 class 一次性限制所有 Motion 类能力）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CapabilityClass {
    Motion,
    Speech,
    Query,
    Control,
}

/// 风险等级，从低到高排序（`Ord` 用于规则链按等级比较，比如"High 风险必须审批"）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    None,
    Low,
    Medium,
    High,
}

fn d_ttl() -> f64 {
    5.0
}

/// 一个能力的静态描述：注册时提供，注册表按 `name` 索引。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySpec {
    pub name: String,
    pub class: CapabilityClass,
    pub risk_level: RiskLevel,
    #[serde(default)]
    pub params_schema: Value,
    #[serde(default = "d_ttl")]
    pub lease_ttl_s: f64,
    /// 这一项允许被一个适配器的同类能力顶替吗？默认 false，而且**只有内置 `say` 是
    /// true**（见 `builtin::admission`）。
    ///
    /// 为什么这个标记长在 spec 上而不是注册表里的一张名单：它是这个能力自己的性质，
    /// 声明它的人（内置插件）最清楚，而一张名单会让"哪些名字可以被顶替"离开它所描述
    /// 的那个东西，从此两处各改各的。
    ///
    /// 线格式上一个跨进程插件说什么都不算数（`oeh_rpc::handlers::plugin::spec_of` 不读
    /// 这个字段）：自封可顶替等于邀请下一个连上来的进程接管自己的能力。
    #[serde(default)]
    pub overridable: bool,
}

/// 执行器视角的取消/心跳句柄。`Admission`（后续任务）在开始执行时用 `new_pair` 造一对：
/// `ExecCtx` 交给 `Executor::execute`，`ExecControl` 自己留着，随时能撤销租约或读心跳。
pub struct ExecCtx {
    pub trace_id: String,
    pub proposal_id: ProposalId,
    cancel: watch::Receiver<bool>,
    heartbeat_ns: Arc<AtomicU64>,
    clock: Arc<dyn Clock>,
}

impl ExecCtx {
    /// 造一对：执行器拿 `ExecCtx`，调用方拿 `ExecControl`。
    pub fn new_pair(
        trace_id: String,
        proposal_id: ProposalId,
        clock: Arc<dyn Clock>,
    ) -> (ExecCtx, ExecControl) {
        let (cancel_tx, cancel) = watch::channel(false);
        let heartbeat_ns = Arc::new(AtomicU64::new(0));
        let ctx = ExecCtx {
            trace_id,
            proposal_id,
            cancel,
            heartbeat_ns: heartbeat_ns.clone(),
            clock,
        };
        let ctl = ExecControl {
            cancel_tx,
            heartbeat_ns,
        };
        (ctx, ctl)
    }

    /// 执行器在长任务里周期性调用，证明自己还活着（心跳戳记录内核时间，不是墙钟）。
    pub fn heartbeat(&self) {
        self.heartbeat_handle().beat();
    }

    /// 只能盖心跳戳的一个廉价克隆。
    ///
    /// 为什么需要它：一个**跨进程**的执行器收到的心跳是一条后到的消息（`capability/
    /// heartbeat` 通知），处理它的是连接的读循环，不是 `execute()` 返回的那个 future
    /// ——而 `ExecCtx` 已经被移进那个 future 里了，读循环够不着。反过来把
    /// [`ExecControl`] 交给读循环也不行：那是**准入侧**的取消端，交出去就等于把"撤销
    /// 这次执行"的权力交给了被撤销的那一方。这个句柄只有 `beat()` 一件事可做，取消端
    /// 一点都不漏。
    pub fn heartbeat_handle(&self) -> HeartbeatHandle {
        HeartbeatHandle {
            heartbeat_ns: self.heartbeat_ns.clone(),
            clock: self.clock.clone(),
        }
    }

    /// 非阻塞查询：现在有没有被取消。
    pub fn is_cancelled(&self) -> bool {
        *self.cancel.borrow()
    }

    /// 等到被取消为止；如果调用时已经是取消状态，立即返回而不是等下一次变化。
    pub async fn cancelled(&mut self) {
        if *self.cancel.borrow() {
            return;
        }
        while self.cancel.changed().await.is_ok() {
            if *self.cancel.borrow() {
                return;
            }
        }
    }
}

/// 撤销一次执行的理由。`as_str()` 给出的字符串是**对外契约**：它原样出现在
/// `act/cancelled{reason}`、`act/lease_expired{reason}` 与 `Outcome::Cancelled{reason}` 里。
///
/// 用枚举而不是 `&str`：`Preempted` 不只是一个标签，它还宣布"ACTING 这块地已经交给接班
/// 的 motion 了，被抢占者的收尾不要再把 FSM 交还成 STANDING"。外面随手传一个
/// `"preempted"` 进来，就等于宣布了一次根本不存在的交接——没人接班，FSM 会永远停在
/// ACTING。所以这一个变体用 `#[non_exhaustive]` 锁住：crate 外面构造不出来。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelReason {
    /// 内置 `stop` 能力，或者操作员主动叫停。
    Stop,
    /// FSM 离开了 `ACTING`：别人（蒸馏器、操作员）把机器人带去了别处。
    Fsm,
    /// 急停按下（`safety/estop{engaged:true}`）。
    Estop,
    /// 执行期间的规则复检不再放行。
    Rules,
    /// 心跳老过租约：执行器失联。
    LeaseExpired,
    /// 跑这次执行的适配器**断连了**。
    ///
    /// 和 `LeaseExpired` 的差别是承重的，不是措辞：租约过期只说明"心跳停了"——执行器
    /// 可能只是卡住，而机器人还在动，所以那条路要保留位子、还要再喊一声停。断连是
    /// 一件确定的事：那条连接上再也发不出、也收不到任何东西，喊停没有对象，等宽限期
    /// 等的是一条永远不会来的消息。所以这条路当场释放位子（见 `AdmState::claim`）。
    Disconnected,
    /// 被新准入的 motion 抢占——一次**交接**。只有准入服务自己能构造。
    #[non_exhaustive]
    Preempted,
}

impl CancelReason {
    /// 写进事件与 `Outcome` 的字符串。
    pub fn as_str(self) -> &'static str {
        match self {
            CancelReason::Stop => "stop",
            CancelReason::Fsm => "fsm",
            CancelReason::Estop => "estop",
            CancelReason::Rules => "rules",
            CancelReason::LeaseExpired => "lease expired",
            CancelReason::Disconnected => "adapter disconnected",
            CancelReason::Preempted => "preempted",
        }
    }
}

impl std::fmt::Display for CancelReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// [`ExecCtx::heartbeat_handle`] 交出来的东西：盖一次心跳戳，别的什么也做不了。`Clone`
/// 只是多一个指向同一个时间戳的句柄。
#[derive(Clone)]
pub struct HeartbeatHandle {
    heartbeat_ns: Arc<AtomicU64>,
    clock: Arc<dyn Clock>,
}

impl HeartbeatHandle {
    /// 盖一次心跳戳（内核时间，不是墙钟——租约监视读的就是这个刻度）。
    pub fn beat(&self) {
        self.heartbeat_ns
            .store(self.clock.now_ns(), Ordering::SeqCst);
    }
}

/// `Admission` 持有的另一半：撤销执行、读最近一次心跳时间戳。
pub struct ExecControl {
    cancel_tx: watch::Sender<bool>,
    heartbeat_ns: Arc<AtomicU64>,
}

impl ExecControl {
    pub fn cancel(&self) {
        // 接收端（`ExecCtx`）可能已经被丢弃（执行器提前返回）——发送失败是正常情况，
        // 不代表撤销本身出了错，忽略即可。
        let _ = self.cancel_tx.send(true);
    }

    pub fn last_heartbeat_ns(&self) -> u64 {
        self.heartbeat_ns.load(Ordering::SeqCst)
    }

    /// 订阅取消信号。发送端（这次执行的 `ExecControl`）随着登记一起消失时 `changed()`
    /// 会报错——后台任务据此知道"这次执行的故事结束了"，不必等到下一拍才发现。
    pub(crate) fn subscribe(&self) -> watch::Receiver<bool> {
        self.cancel_tx.subscribe()
    }
}

/// 一次执行的结局。`error` 只在 `ok == false` 时有意义，但即便 `ok == true` 也允许携带
/// 空字符串以外的字段默认值，序列化上不强求两者互斥。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecOutcome {
    pub ok: bool,
    #[serde(default)]
    pub result: Value,
    #[serde(default)]
    pub error: Option<String>,
}

/// 执行器实现的合约：给定一次调用和取消/心跳句柄，返回一个 future。`execute` 本身必须
/// 立刻返回（不能阻塞调用它的调度循环），真正的工作发生在返回的 future 里。
pub trait Executor: Send + Sync {
    fn execute(&self, call: ExecutionCall, ctx: ExecCtx) -> BoxFuture<ExecOutcome>;

    /// 租约过期时内核回头叫的那一下（规格 §5.3："心跳超过 lease TTL 记
    /// `act/lease_expired`，内核向适配器发 stop"）。
    ///
    /// 这时 `ExecCtx` 已经被撤销过了，但撤销只是"别再往下做"——一个**失联**的执行器
    /// 按定义就是不再看那个通道的那种，而机器人可能还在动。所以内核另外明确地叫一次
    /// 停，打给**这次执行自己的执行器**：能力注册表里的内置 `stop` 能力语义是"撤销
    /// 当前 motion"，拿它兜底会在一次无关的 speech 超时时停掉一台健康的 `walk`。
    ///
    /// 默认什么也不做：对一个纯计算的执行器来说没有"停"可言。**内核不会 await 它**
    /// ——一个已经失联的执行器不值得再等——所以 `stop` 本身也必须立刻返回。
    fn stop(&self, proposal_id: &ProposalId) -> BoxFuture<()> {
        let _ = proposal_id;
        Box::pin(async {})
    }
}

/// 一次具体的执行调用：能力名字、参数、发起它的提案。
///
/// **两个字段里的 params 都是规则链跑完之后的那一份**：`params == proposal.params`，
/// 准入服务保证这一点。执行器读哪个都行，读到的都是被夹过的值——要是 `proposal` 还
/// 留着提案原样的 params，随手读它的执行器就把整条规则链（限速、限幅）绕过去了。
#[derive(Clone, Debug)]
pub struct ExecutionCall {
    pub capability: String,
    pub params: Value,
    pub proposal: Proposal,
}

#[derive(Debug, thiserror::Error)]
pub enum CapabilityError {
    #[error("capability `{0}` already registered")]
    AlreadyRegistered(String),
}

struct Entry {
    /// 每次注册一个新 id：撤销只拆自己那次注册，过期的 disposer 碰不到后来的重新注册
    /// （同名能力先撤销、再注册的场景）。做法抄自 `oeh_kernel::frame_store::FrameStore`。
    reg_id: u64,
    spec: CapabilitySpec,
    exec: Arc<dyn Executor>,
    /// 谁提供这一项：内置是 [`KERNEL_PROVIDER`]，跨进程插件是它的 `manifest.name`。
    provider: String,
    /// 被这一项顶替掉的内置项（只有覆盖注册才有）。这一份**不在**能力面上，但它还活着：
    /// 顶替它的那次注册一撤销，它就回到台面上。
    fallback: Option<Box<Entry>>,
}

struct Inner {
    entries: HashMap<String, Entry>,
    next_reg: u64,
}

/// 注册表里一项的快照：静态描述 + 谁提供它。
///
/// 和 [`CapabilityRegistry::list`] 的差别就是那个 `provider`，而它只有一个读者：
/// `robot/capabilities`。脑不需要它（对脑来说 `say` 就是 `say`，谁执行的不改变工具表），
/// 运维需要它——"这句话到底有没有从扬声器出来"，答案全在这个字段里。
#[derive(Clone, Debug)]
pub struct CapabilityEntry {
    pub spec: CapabilitySpec,
    pub provider: String,
}

/// 名字 → (描述, 执行器) 的注册表。注册返回的 `Disposer` 只在仍是"那一次"注册时才生效，
/// 这样过期的 disposer 不会误删后来重新注册的同名能力。
///
/// **覆盖**（`overridable` 的那条路）也长在这同一个 `Disposer` 上：顶替生效时被顶替的
/// 那一项存成 fallback，顶替者的 disposer 一跑就把它放回去。于是"适配器断连"和"内核树
/// 上一个 Fiber 被 dispose"仍然是同一件事——没有任何一段"把内置能力装回去"的专门代码，
/// 也就没有一条只在断连时才走、平时没人测的路。
///
/// `Clone` 是**再拿一个句柄**，不是复制一份注册表：三个字段都是 `Arc`，克隆出来的那个
/// 看到的是同一张表、同一个代号。脑要的正是这个——它得在每个 turn 开头问一次"能力面
/// 动过没有"，而 `Admission::capabilities()` 只借得出一个引用，借的东西活不过那一行。
#[derive(Clone)]
pub struct CapabilityRegistry {
    inner: Arc<Mutex<Inner>>,
    /// 能力面的**代号**：每一次改动台面上那张表都 +1（注册、撤销、顶替、恢复）。
    ///
    /// 为什么是一个计数器而不是让读者自己 diff 一遍 `list()`：唯一的读者是脑，而它要问
    /// 的问题是"从我上次投影到现在，能力面动过没有"。一个 `u64` 的原子读答得了这个问题
    /// 而**不用拿锁**，于是"每个 turn 开头确认一次"在没人动过的常态下是零成本的；diff
    /// 则要先克隆一遍整张表才知道答案，而那正是这个计数器要省掉的那一份工作。
    ///
    /// 放在 `Arc` 里而不是 `Inner` 里：撤销走的是 `register_from` 交出去的那个闭包，它
    /// 只攥着 `inner` 和这一份计数器；`generation()` 因此是 lock-free 的，而每一次 +1
    /// 仍然发生在改 `entries` 的那个临界区**里面**（顺序不会反过来）。
    generation: Arc<AtomicU64>,
    /// 覆盖/恢复写去哪。`None` 是"没人听"（`new()` 出来的注册表，测试和进程内的用法）：
    /// 注册表不该因为没有世界日志就不能用。
    log: Option<Arc<WorldLog>>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self::build(None)
    }

    /// 带世界日志的注册表：能力面的易主（[`KIND_OVERRIDDEN`] / [`KIND_RESTORED`]）写进去。
    /// `Admission` 用这个构造器——它本来就持着日志。
    pub fn with_log(log: Arc<WorldLog>) -> Self {
        Self::build(Some(log))
    }

    fn build(log: Option<Arc<WorldLog>>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                entries: HashMap::new(),
                next_reg: 1,
            })),
            generation: Arc::new(AtomicU64::new(0)),
            log,
        }
    }

    /// 能力面此刻的代号。两次读到同一个数 = 这中间台面上那张表一个字都没动过，于是
    /// 上一次投影出来的工具表可以原样再用（[`Self::list`] 会返回逐项相同的东西）。
    ///
    /// 反过来不成立也不要紧：代号变了不代表**脑看得见的**那部分变了（一次注册又撤销
    /// 会让它 +2），代价只是多投影一次，而投影是确定性的——多出来的那一次渲染出逐字节
    /// 相同的 prompt 前缀，什么也不会失效。宁可多一次也不能少一次。
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// 内核自己注册一个能力（`provider == "kernel"`）：内置能力、以及任何一个进程内插件。
    ///
    /// 跨进程插件走 [`register_from`](Self::register_from)，因为"谁提供它"是那一侧唯一
    /// 说得清的事。
    pub fn register(
        &self,
        spec: CapabilitySpec,
        exec: Arc<dyn Executor>,
    ) -> Result<Disposer, CapabilityError> {
        self.register_from(KERNEL_PROVIDER, spec, exec)
    }

    /// 以 `provider` 的名义注册。
    ///
    /// 重名一律 `AlreadyRegistered`——**除了**一种情况：台面上那一项自己说
    /// `overridable`，而新来的这一项是 [`CapabilityClass::Speech`]。那是一次**顶替**：
    /// 内置的那一份退到 fallback 里，新的这一份上台，日志里留一条 `capability/overridden`。
    ///
    /// 为什么类别也要对：可顶替说的是"这个名字背后的实现可以换人"，不是"这个名字可以
    /// 改成别的意思"。脑的工具表里 `say` 是"说一句话"，用一个 motion 顶替它，脑下一次
    /// 调用 `say` 就会让机器人走起来——那不是覆盖，那是换名。
    ///
    /// 顶替过的那一项不再可顶替（它带着 `overridable: false` 上台，见 `spec_of`），所以
    /// 第二台机器人撞上来照旧是 `AlreadyRegistered`：两台机器人抢同一张嘴没有正确答案，
    /// 而先到的那台正在用它。
    pub fn register_from(
        &self,
        provider: &str,
        spec: CapabilitySpec,
        exec: Arc<dyn Executor>,
    ) -> Result<Disposer, CapabilityError> {
        let name = spec.name.clone();
        let class = spec.class;
        let (reg_id, overrode) = {
            let mut g = self.inner.lock();
            let fallback = match g.entries.get(&name) {
                Some(old) if old.spec.overridable && spec.class == CapabilityClass::Speech => {
                    Some(Box::new(g.entries.remove(&name).expect("just matched")))
                }
                Some(_) => return Err(CapabilityError::AlreadyRegistered(name)),
                None => None,
            };
            let overrode = fallback.is_some();
            let reg_id = g.next_reg;
            g.next_reg += 1;
            g.entries.insert(
                name.clone(),
                Entry {
                    reg_id,
                    spec,
                    exec,
                    provider: provider.to_string(),
                    fallback,
                },
            );
            // 台面动了。**在锁里面**，和那一次 `insert` 同一个临界区：一个在锁外 +1 的
            // 计数器可以让"代号没变"和"表已经换了"同时成立，而脑正是靠前者决定不再看
            // 后者一眼。
            self.generation.fetch_add(1, Ordering::SeqCst);
            (reg_id, overrode)
        };
        // 锁外面写日志：`append` 会在**调用线程**上跑一遍观察者，而观察者是别人的代码。
        if overrode {
            self.emit(KIND_OVERRIDDEN, json!({"name": name, "by": provider}));
        }
        // 进程外的插件挂上一条能力才记事件，内核自带的那几条不记：内置能力是**开机就
        // 在**的（bundle 的行序说了算），一条"kernel 注册了 say"只会在每次启动时重复
        // 五遍已经写在 profile 里的事。进程外的那些不一样——它们什么时候来、什么时候
        // 走，只有世界日志说得出来，而这正是一个进程外的脑唯一能看见能力面变化的地方
        // （它读不到注册表）。
        if provider != KERNEL_PROVIDER {
            self.emit(
                KIND_REGISTERED,
                json!({"name": name, "provider": provider, "class": class}),
            );
        }

        let inner = self.inner.clone();
        let generation = self.generation.clone();
        let log = self.log.clone();
        let provider = provider.to_string();
        Ok(Box::new(move || {
            let (removed, restored) = {
                let mut g = inner.lock();
                let touched = match g.entries.get_mut(&name) {
                    // 还是"那一次"注册：撤下来，底下有兜底就让它回台面。
                    Some(e) if e.reg_id == reg_id => {
                        let mut e = g.entries.remove(&name).expect("just matched");
                        match e.fallback.take() {
                            Some(f) => {
                                g.entries.insert(name.clone(), *f);
                                Some((true, true))
                            }
                            None => Some((true, false)),
                        }
                    }
                    // 这一次注册已经被顶替了，它此刻躺在台面那一项的 fallback 里。把它
                    // 从那里摘掉：它的主人（比如被 dispose 的 admission 那一行）已经走了，
                    // 晚些的一次恢复不该把一个已经撤销的执行器请回能力面。
                    //
                    // 台面上那一项没变，但这张表变了，代号照样 +1：它偏保守（多变一次只
                    // 是多投影一次，见 `generation()`），漏变一次则是脑抱着一张过期的表
                    // 跑下去。
                    Some(e) if e.fallback.as_ref().map(|f| f.reg_id) == Some(reg_id) => {
                        e.fallback = e.fallback.take().and_then(|f| f.fallback);
                        Some((false, false))
                    }
                    // 别人的注册（同名先撤后注册），或者早就没了：不碰，代号也不动。
                    _ => None,
                };
                match touched {
                    Some(out) => {
                        generation.fetch_add(1, Ordering::SeqCst);
                        out
                    }
                    None => (false, false),
                }
            };
            if let Some(log) = &log {
                if removed && provider != KERNEL_PROVIDER {
                    append(
                        log,
                        KIND_UNREGISTERED,
                        json!({"name": name, "provider": provider}),
                    );
                }
                if restored {
                    append(log, KIND_RESTORED, json!({"name": name, "from": provider}));
                }
            }
        }))
    }

    fn emit(&self, kind: &'static str, payload: Value) {
        if let Some(log) = &self.log {
            append(log, kind, payload);
        }
    }

    pub fn get(&self, name: &str) -> Option<(CapabilitySpec, Arc<dyn Executor>)> {
        self.inner
            .lock()
            .entries
            .get(name)
            .map(|e| (e.spec.clone(), e.exec.clone()))
    }

    pub fn list(&self) -> Vec<CapabilitySpec> {
        self.inner
            .lock()
            .entries
            .values()
            .map(|e| e.spec.clone())
            .collect()
    }

    /// [`list`](Self::list) 再加上"谁提供它"。
    pub fn list_entries(&self) -> Vec<CapabilityEntry> {
        self.inner
            .lock()
            .entries
            .values()
            .map(|e| CapabilityEntry {
                spec: e.spec.clone(),
                provider: e.provider.clone(),
            })
            .collect()
    }
}

/// 写一条能力面易主的事件。salience 100（中位）：它不是安全事件，但"机器人的嗓子换人了"
/// 是一件读日志的人必须能看见的事——一条静悄悄的顶替，症状是"话没从扬声器出来"，而
/// 原因在别的文件里。
fn append(log: &WorldLog, kind: &'static str, payload: Value) {
    log.append(
        EventDraft::new(EventKind::new(kind).expect("static kind"), SOURCE, payload)
            .with_salience(SAL_CAPABILITY_FACE),
    );
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// 最小 JSON-schema 子集校验：`type`（object|number|integer|string|boolean|array）、
/// object 的 `properties`（每个属性可选 `type`/`minimum`/`maximum`/`enum`）、
/// `required`、`additionalProperties: false`。不认识的关键字一律忽略；空 schema `{}`
/// 接受任何值。
pub fn validate_params(schema: &Value, params: &Value) -> Result<(), String> {
    validate_value(schema, params, "params")
}

fn validate_value(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    let schema_obj = match schema.as_object() {
        Some(o) => o,
        // schema 不是 object（比如 `true`/`null`）：没有可解释的约束，一律放行。
        None => return Ok(()),
    };
    if schema_obj.is_empty() {
        return Ok(());
    }

    if let Some(t) = schema_obj.get("type").and_then(|v| v.as_str()) {
        check_type(t, value, path)?;
    }

    if let Some(enum_vals) = schema_obj.get("enum").and_then(|v| v.as_array()) {
        if !enum_vals.iter().any(|v| v == value) {
            return Err(format!(
                "{path}: value {value} is not one of the allowed enum values {enum_vals:?}"
            ));
        }
    }

    if let Some(n) = value.as_f64() {
        if let Some(min) = schema_obj.get("minimum").and_then(|v| v.as_f64()) {
            if n < min {
                return Err(format!("{path}: {n} is less than minimum {min}"));
            }
        }
        if let Some(max) = schema_obj.get("maximum").and_then(|v| v.as_f64()) {
            if n > max {
                return Err(format!("{path}: {n} is greater than maximum {max}"));
            }
        }
    }

    let is_object_schema = schema_obj.get("type").and_then(|v| v.as_str()) == Some("object")
        || schema_obj.contains_key("properties");
    if is_object_schema {
        let obj = value
            .as_object()
            .ok_or_else(|| format!("{path}: expected type `object`, got {value}"))?;

        if let Some(required) = schema_obj.get("required").and_then(|v| v.as_array()) {
            for r in required {
                if let Some(name) = r.as_str() {
                    if !obj.contains_key(name) {
                        return Err(format!("{path}: missing required property `{name}`"));
                    }
                }
            }
        }

        let props = schema_obj.get("properties").and_then(|v| v.as_object());
        if let Some(props) = props {
            for (k, v) in obj {
                if let Some(pschema) = props.get(k) {
                    validate_value(pschema, v, &format!("{path}.{k}"))?;
                }
            }
        }

        let additional_allowed = schema_obj
            .get("additionalProperties")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if !additional_allowed {
            for k in obj.keys() {
                let known = props.map(|p| p.contains_key(k)).unwrap_or(false);
                if !known {
                    return Err(format!("{path}: additional property `{k}` is not allowed"));
                }
            }
        }
    }

    Ok(())
}

fn check_type(t: &str, value: &Value, path: &str) -> Result<(), String> {
    let ok = match t {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "number" => value.is_number(),
        // 判断是不是整数要看数学意义上的值，不是 serde_json 内部的存储表示：
        // `1.0`（存成 f64）在数学上就是整数，应该和 `1`（存成 i64/u64）一样被接受。
        "integer" => value
            .as_f64()
            .map(|f| f.is_finite() && f.fract() == 0.0)
            .unwrap_or(false),
        // 不认识的 type 关键字：忽略，不阻塞校验。
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("{path}: expected type `{t}`, got {value}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oeh_kernel::ManualClock;

    /// 跨进程执行器要的正是这件事：`ExecCtx` 已经被移进 `execute()` 返回的 future 里，
    /// 但心跳是另一条路（连接的读循环）送来的。句柄盖的戳必须落在**同一个**时间戳上
    /// ——准入侧的租约监视读的就是它——而且句柄除了盖戳什么也做不了（它不带取消端）。
    #[test]
    fn heartbeat_handle_stamps_the_same_cell_as_the_ctx() {
        let clock = Arc::new(ManualClock::new(10));
        let (ctx, ctl) = ExecCtx::new_pair("t".into(), ProposalId("p".into()), clock.clone());
        assert_eq!(ctl.last_heartbeat_ns(), 0);
        let hb = ctx.heartbeat_handle();
        clock.set(77);
        hb.beat();
        assert_eq!(ctl.last_heartbeat_ns(), 77);
        // 句柄活过 `ExecCtx` 也照样能用（执行器的 future 结束了，但登记还没摘掉）。
        drop(ctx);
        clock.set(99);
        hb.beat();
        assert_eq!(ctl.last_heartbeat_ns(), 99);
    }

    struct Nop;
    impl Executor for Nop {
        fn execute(&self, _call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
            Box::pin(async {
                ExecOutcome {
                    ok: true,
                    result: Value::Null,
                    error: None,
                }
            })
        }
    }

    fn spec(name: &str, class: CapabilityClass, overridable: bool) -> CapabilitySpec {
        CapabilitySpec {
            name: name.into(),
            class,
            risk_level: RiskLevel::None,
            params_schema: json!({"type": "object"}),
            lease_ttl_s: 1.0,
            overridable,
        }
    }

    /// 代号在**每一次**台面变动上 +1：注册、撤销、顶替、恢复。脑按它决定要不要重新
    /// 投影工具表，所以漏掉任何一种都是"脑拿着一张过期的表跑一整个 turn"。
    #[test]
    fn the_generation_moves_on_every_change_to_the_face() {
        let r = CapabilityRegistry::new();
        assert_eq!(r.generation(), 0);

        let d1 = r
            .register(spec("say", CapabilityClass::Speech, true), Arc::new(Nop))
            .unwrap();
        let g1 = r.generation();
        assert_eq!(g1, 1);
        // 读一遍不算变动。
        assert_eq!(r.list().len(), 1);
        assert_eq!(r.generation(), g1);

        // 顶替：台面上那一项换人了，虽然名字一个没多。
        let d2 = r
            .register_from(
                "py-arm",
                spec("say", CapabilityClass::Speech, false),
                Arc::new(Nop),
            )
            .unwrap();
        assert_eq!(r.generation(), 2);

        // 恢复：被顶替的内置项回台面，schema 也跟着换回去。
        d2();
        assert_eq!(r.generation(), 3);

        // 撤销一个没有兜底的：名字从台面上消失。
        let d3 = r
            .register_from(
                "py-arm",
                spec("walk", CapabilityClass::Motion, false),
                Arc::new(Nop),
            )
            .unwrap();
        assert_eq!(r.generation(), 4);
        d3();
        assert_eq!(r.generation(), 5);

        // 重名被拒绝那一次**不**算变动：什么都没动过，脑不该为此重投一次。（用一个
        // motion 去撞 `say`：可顶替说的是"实现可以换人"，不是"这个名字可以改成别的
        // 意思"，所以这一次照旧是 `AlreadyRegistered`。）
        let dup = r.register(spec("say", CapabilityClass::Motion, false), Arc::new(Nop));
        assert!(dup.is_err());
        assert_eq!(r.generation(), 5);

        d1();
        assert_eq!(r.generation(), 6);
    }

    /// `capability/registered` / `capability/unregistered` 只替**进程外**的插件记。
    ///
    /// 内置的那几条不记的理由在 `register_from` 里：它们开机就在，一条"kernel 注册了
    /// say"重复的是已经写在 profile 里的事；而进程外的那些什么时候来、什么时候走，
    /// 只有这两条事件说得出来——一个跨进程的脑读不到注册表，这是它唯一的信源。
    #[test]
    fn only_out_of_process_registrations_are_announced_to_the_world_log() {
        let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
        let r = CapabilityRegistry::with_log(log.clone());

        let kernel_d = r
            .register(spec("stop", CapabilityClass::Control, false), Arc::new(Nop))
            .unwrap();
        assert!(
            log.since(0).unwrap_or_default().is_empty(),
            "内置能力不记事件"
        );

        let remote_d = r
            .register_from(
                "py-arm",
                spec("walk", CapabilityClass::Motion, false),
                Arc::new(Nop),
            )
            .unwrap();
        remote_d();
        kernel_d();

        let evs = log.since(0).unwrap_or_default();
        let kinds: Vec<&str> = evs.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(kinds, vec![KIND_REGISTERED, KIND_UNREGISTERED]);
        assert_eq!(evs[0].payload["name"], "walk");
        assert_eq!(evs[0].payload["provider"], "py-arm");
        assert_eq!(evs[0].payload["class"], "motion");
        assert_eq!(evs[0].salience, SAL_CAPABILITY_FACE);
        assert_eq!(
            evs[1].payload,
            json!({"name": "walk", "provider": "py-arm"})
        );
    }

    /// `CancelReason` 的字符串是对外契约：它原样出现在 `act/cancelled{reason}` 和
    /// `Outcome::Cancelled{reason}` 里，脑和日志读者都按它分支。改一个字都是破坏性变更。
    #[test]
    fn cancel_reason_wire_strings_are_stable() {
        assert_eq!(CancelReason::Stop.as_str(), "stop");
        assert_eq!(CancelReason::Fsm.as_str(), "fsm");
        assert_eq!(CancelReason::Estop.as_str(), "estop");
        assert_eq!(CancelReason::Rules.as_str(), "rules");
        assert_eq!(CancelReason::LeaseExpired.as_str(), "lease expired");
        assert_eq!(CancelReason::Disconnected.as_str(), "adapter disconnected");
        assert_eq!(CancelReason::Preempted.as_str(), "preempted");
        assert_eq!(CancelReason::Preempted.to_string(), "preempted");
    }
}
