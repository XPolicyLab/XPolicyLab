//! `Router`：一条连接收到的 Request/Notification 最终都交给它处理。
//!
//! [`KernelRouter`] 是把边界协议接到内核服务上的那一层：一张方法名 → 处理函数的表
//! （实现住在 [`crate::handlers`]），外加一件它自己得记着的状态——每个 Fiber 当前的
//! `plugin/state`（[`FiberStates`]）。
//!
//! 为什么 Fiber 状态要由路由自己维护：`FiberTree` 归内核的 `run` 循环所有，跨线程读它
//! 需要一把这里不该有的锁。内核在每次 settle 之后把变化广播成 `plugin/state` 世界日志
//! 事件，路由订阅它，一边攒成快照给 `robot/health`，一边原样转发给**所有**连接——
//! 控制台要的是"谁刚刚掉了"，不是"我去问一下"。

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::{
    Brains, ClockService, Disposer, FiberId, FrameStore, KernelHandle, RobotFsm, StateRegistry,
    WorldLog,
};
use oeh_proto::rpc::method::*;
use oeh_proto::rpc::types::FiberStateInfo;
use oeh_proto::rpc::{code, RpcError};
use oeh_proto::{EventDraft, EventKind, WorldEvent};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::conn::Conn;
use crate::handlers;
use crate::transport::ConnRegistry;

/// 一条连接的方法路由表。实现者必须保证 `call` 立刻返回一个 future（不能在调用它的地方
/// 阻塞）——`conn.rs` 的读循环把每次 `call` 单独 `tokio::spawn` 出去，但如果实现本身在
/// 构造 future 之前就同步阻塞，照样会拖慢读循环。
pub trait Router: Send + Sync + 'static {
    /// 处理一次 Request；在自己的 tokio 任务里跑，慢方法不阻塞其他请求。
    fn call(
        &self,
        conn: Arc<Conn>,
        method: String,
        params: Value,
    ) -> BoxFuture<Result<Value, RpcError>>;
    /// 处理一次客户端发来的 Notification（比如 `capability/heartbeat`、`capability/result`）。
    fn notify(&self, conn: Arc<Conn>, method: String, params: Value);
}

/// 把 `params` 解成 `T`；失败时统一编码成 `INVALID_PARAMS`，`data` 里带错误文本，
/// 方法实现不用各自拼错误。
pub fn parse_params<T: DeserializeOwned>(v: &Value) -> Result<T, RpcError> {
    serde_json::from_value(v.clone()).map_err(|e| {
        RpcError::new(code::INVALID_PARAMS, "invalid params")
            .with_data(Value::String(e.to_string()))
    })
}

/// `KernelRouter` 要用到的全部服务。前八个字段跟 `rpc-server` 插件的 `inject` 列表一一
/// 对应；后两个（`conns`、`channels`）是 `rpc-server` 自己造出来、连接们共享的东西。
pub struct Services {
    pub kernel: Arc<KernelHandle>,
    pub clock: Arc<ClockService>,
    pub log: Arc<WorldLog>,
    pub frames: Arc<FrameStore>,
    pub state: Arc<StateRegistry>,
    pub fsm: Arc<RobotFsm>,
    pub brains: Arc<Brains>,
    pub admission: Arc<Admission>,
    pub conns: Arc<ConnRegistry>,
    /// 帧通道（[`crate::frames`]）：`stream/subscribe` 的 egress 和跨进程插件的 ingress
    /// 都挂在它上面。和 `frames`（内核的 `FrameStore`）是两件事——那个是帧**存在哪**，
    /// 这个是帧**怎么出进程**。
    pub channels: Arc<crate::frames::Frames>,
    /// `approval/attach` 的互斥闸门（Task 5 re-review R1）：`set_approval` + 装进
    /// `ConnState.approval` + `displace_others` 这三步之间没有别的锁跨着，两条连接
    /// 真并发 attach 时会在几条指令的窗口里插花——一条没被 `displace_others` 看见（它
    /// 的 `ConnState.approval` 还没写上），却在随后自己的 `displace_others` 里把刚
    /// 接管的另一条收掉，两边都不再是审批人却只有一条收到了通知。把整个 `attach` 函数
    /// 体锁在这把闸门下面就消灭了这个交错——函数本身不 `.await`，锁不会跨等待点。
    pub approval_gate: parking_lot::Mutex<()>,
}

/// `rpc-server` 插件配置里跟方法表相关的那部分（`uds`/`ws`/`stdio` 是传输层的事，不在
/// 这里）。`execute_accept_ms`/`cancel_grace_ms` 是跨进程执行器的两个时限；
/// `frames_max_bytes` 被 `rpc-server` 交给 [`crate::frames::Frames`]（帧通道的预留大小），
/// 留在这里是为了让"这个内核的全部旋钮"在一个结构里看得全。
pub struct RouterConfig {
    pub heartbeat_ms: u64,
    pub execute_accept_ms: u64,
    pub cancel_grace_ms: u64,
    /// 这次 run 的帧通道命名空间：`rpc-server` 的 `run_tag` 配置项，没写的话是世界日志
    /// 目录名（`WorldLog::run_dir()` 的 basename），再没有就是 `local-<pid>`
    /// （见 `plugin::resolve_run_tag`——iceoryx2 的服务名是机器全局的，一个常数退路会让
    /// 两个内核共用一条通道）。`plugin/register` 和 `stream/subscribe` 用的是同一个值。
    pub run_tag: String,
    pub frames_max_bytes: usize,
    /// 下面六个是 `plugin/register` 的护栏。它们守的不是内存，是**内核任务**：
    /// `RemotePlugin::apply` 跑在 `Kernel::run` 那一条任务上（`Op::AddFiber` →
    /// `settle()` → `activate()`），一份声明了十万条流的 manifest 就是十万次
    /// `FrameStore::register` 同步地占着那条任务，期间每个真实传感器的
    /// `Op::IngestFrame` 都在排队。一条连得上 socket 的连接不该有这个能力。
    ///
    /// 都是可配的（`rpc-server` 的同名配置项）：默认值按"真实适配器声明 1–3 条流"
    /// 留了 20 倍余量，一个真需要更多的部署把它调大就是了。
    pub max_streams_per_register: usize,
    pub max_capabilities_per_register: usize,
    pub max_inject_per_register: usize,
    pub max_provide_per_register: usize,
    /// **活着**的 Fiber 数上限。重名检查挡不住这个：`a1`/`a2`/`a3`… 全是合法的不同名字。
    pub max_fibers_per_conn: usize,
    /// 一次 `plugin/register` 的 params 序列化之后的字节上限。这是唯一一个真正兜底的
    /// 数字：`config`、每个 `params_schema`、整份 manifest 一次全罩住——包括那些即便
    /// Fiber 被撤掉、也还留在 `FiberTree` 的退休行里占着的字节
    /// （见 `FiberTree::prune_disposed`）。
    pub max_register_bytes: usize,
    /// 一条声明的流允许报到多少赫兹。上面那六道护栏数的是"几条流、几个能力"，从来不看
    /// 声明**里面**的数字，而 `FrameStore::register` 是按 `nominal_hz × window_s` 预分配
    /// 环形缓冲的：`nominal_hz: 1e12` 就是一次 176 TB 的分配，失败走 `handle_alloc_error`
    /// （abort，不是 panic，`catch_unwind` 接不住），内核进程当场没——一条连得上 socket
    /// 的连接、一个适配器 manifest 里的手误，都不该有这个本事（Plan 3 最终评审 C1）。
    ///
    /// 默认 20000 Hz：规格里最快的一条流是 500 Hz，留了 40 倍余量。非有限值和 ≤ 0 一律
    /// 拒（`INVALID_PARAMS`）。内核那一侧另有 `FrameStore::MAX_RING_FRAMES` 兜底，两道
    /// 是故意的：这一道给出的是**调用方能看懂的错误**，那一道保护的是所有调用方。
    pub max_nominal_hz: f64,
    /// 一次 `log/replay` 最多回多少条事件。`limit` 会被**静默**夹进 `1..=max_replay_page`
    /// ——`done: false` 本来就是"还有，接着翻页"的意思，调用方照着翻就是了，不需要为此
    /// 认一个新的错误码。
    ///
    /// 不夹的话，`{"after": 0, "limit": 100000000}`（线上六十个字节）会让内核把整个 run
    /// 读进一个 `Vec<WorldEvent>`、再 `serde_json` 序列化成**一行**——既超 `MAX_LINE`
    /// （对端永远读不到），又在出站队列里再拷一份（Plan 3 最终评审 C2）。
    pub max_replay_page: usize,
    /// 一次 `event/emit` **留下来**的字节上限：`payload` 序列化之后的长度，加上 `kind`、
    /// `collapse_key`、`trace_id` 的长度（那三个另有 `MAX_EVENT_STRING_BYTES` = 256 字节的
    /// 逐字段上限）。量"留下来的"而不是只量 payload，是因为放大来自 `WorldEvent` 被保留、
    /// 被克隆的那一整份，而不是某一个字段（再审 R1）。
    ///
    /// 世界日志按**条数**保留（默认 4096 条），不按字节，所以 4096 条各带一个 16 MiB
    /// payload 的事件就是 64 GiB 常驻内存，外加每个 `log/tail` 观察者、每个 attach 着的
    /// 脑的收件箱各一份克隆。`plugin/register` 早就有同一形状的护栏
    /// （`max_register_bytes`），`event/emit` 漏了（Plan 3 最终评审 I3）。
    ///
    /// 默认 64 KiB：一条观测绰绰有余，而帧按设计根本不走这条路（走 iceoryx2）。
    pub max_event_bytes: usize,
    /// 一次 `plugin/register` 从收到到应答的**总预算**（毫秒，默认 5000）。
    ///
    /// 上面那六道护栏数的是"几条流、几个能力"，一条也数不出**每条有多慢**：
    /// `RemotePlugin::apply` 跑在 `Kernel::run` 那一条任务上，而它要为每条声明的流开一条
    /// ingress（共享内存 + 一个线程）。一个慢到病态的环境、一条在建服务上卡住的流，就能
    /// 让一次注册把那条任务占住任意久——期间每个真实传感器的 `Op::IngestFrame` 都在排队，
    /// 而发起注册的那个客户端只看到一次永远不回来的调用（Plan 3 遗留 I6）。
    ///
    /// 超预算的应答是 `REFUSED{"register budget exceeded"}`（不是 `INVALID_PARAMS`：同样
    /// 的 manifest 在一台不忙的机器上完全可能注册得上），而且**不留痕迹**——内核要是最终
    /// 还是把那一行加上了，一个清理任务会把它撤掉，见 `handlers::plugin::register`。
    pub register_timeout_ms: u64,
    /// **测试钩子**，只在 `test-hooks` feature 下存在（`rpc-server` 的同名配置项在产品
    /// 构建里根本不是一个键）：`plugin/register` 假装内核那一侧给每条声明的流建通道各要
    /// 这么久（毫秒，见 [`MAX_STREAM_SETUP_DELAY_MS`] 的上限）。
    ///
    /// 存在的唯一理由是 `register_timeout_ms` 守的东西（"一次慢得病态的注册"）没有别的
    /// 办法在测试里**确定地**造出来：真去找一台慢机器就是在赌调度，而把预算调到几毫秒
    /// 又会让测试变成在测调度抖动。
    ///
    /// 实现成**异步** sleep，不是 `std::thread::sleep`：内核任务、连接读写任务和这条
    /// 注册都在同一个 runtime 上，阻塞线程模拟出来的是"整个进程卡住"，而要模拟的是
    /// "这一次注册很慢"——两者在预算这条路上走的分支都不一样。
    #[cfg(feature = "test-hooks")]
    pub stream_setup_delay_ms: u64,
}

/// [`RouterConfig::stream_setup_delay_ms`] 的上限：一个测试钩子没有理由让一次注册停超过
/// 一秒，而一个写错的数字不该把这条连接挂到天荒地老。
#[cfg(feature = "test-hooks")]
pub const MAX_STREAM_SETUP_DELAY_MS: u64 = 1000;

/// [`FiberStates`] 的内部状态：表本身，外加一份跟着走的墓碑计数（见该计数字段上的
/// 说明，Task 5 re-review R3）。
#[derive(Default)]
struct FiberStatesInner {
    map: BTreeMap<u32, FiberStateInfo>,
    /// `map` 里 `state == "disposed"` 的行数，跟着每次 `insert` 增量维护。早退判断
    /// (`record`) 原来比的是 `map.len()`（表的**总**大小），一个有超过 `MAX_DISPOSED_
    /// FIBERS` 条活 Fiber 的 profile 会让每一条 `plugin/state` 都落到扫描分支，即使墓碑
    /// 数是零——这个字段让早退比的是它注释里说的那个数量。
    disposed: usize,
}

/// 每个 Fiber 最后一次被广播的 `plugin/state`，按 `fiber_id`。
///
/// 按 `fiber_id` 而不是 `plugin` 分流：`FiberTree::add` 允许同名多行（同一个 key 被
/// dispose 之后重新 add 就是），两个 Fiber 完全可以共享同一个 `plugin` 字符串，按名字
/// 存会让其中一条的状态盖掉另一条。`BTreeMap` 是为了 `robot/health` 的输出顺序稳定。
#[derive(Default)]
pub struct FiberStates(Mutex<FiberStatesInner>);

/// 快照里最多留多少条 `disposed`。对应内核那边的 `FiberTree::MAX_DISPOSED_ROWS`，但**不**
/// 试图跟它保持同一个集合：路由只能从 `plugin/state` 事件里知道 Fiber 的存在，而"这个 id
/// 可以忘了"没有对应的事件。两边各自有界、各自留最近的那些，就够了——这张表唯一的消费者
/// 是 `robot/health.fibers`，它要回答的是"现在树上有什么、刚刚谁掉了"。
const MAX_DISPOSED_FIBERS: usize = 32;

impl FiberStates {
    /// 记一条状态，顺便把 `disposed` 的条目裁到 [`MAX_DISPOSED_FIBERS`]。
    ///
    /// 不裁的话，一个反复重连的跨进程插件会让 `robot/health` 的 `fibers` 数组随运行时长
    /// 无限变长——而那份数组是每个控制台每次刷新都要收一遍的东西。
    ///
    /// `BTreeMap` 按 key 升序迭代，`fiber_id` 单调递增，所以"最老的若干条 disposed"就是
    /// 迭代到的前几条。fiber_id 在生产环境里不会被复用（`FiberTree` 的 id 只增不减），
    /// 所以一个已经 `disposed` 的行实际上不会再变回别的状态；`disposed` 计数按转移维护
    /// （一行第一次变成 disposed 才加一，若它又被记成别的状态就减一）而不是只加不减，
    /// 纯粹是防御性的——这样即便这条假设在某个调用点被打破，计数也不会永久漂移。
    fn record(&self, info: FiberStateInfo) {
        let mut g = self.0.lock();
        let now_disposed = info.state == "disposed";
        let previous = g.map.insert(info.fiber_id, info);
        let was_disposed = previous.is_some_and(|p| p.state == "disposed");
        match (was_disposed, now_disposed) {
            (false, true) => g.disposed += 1,
            (true, false) => g.disposed -= 1,
            _ => {}
        }
        // 早退：这个函数跑在 `WorldLog::append` 的观察者链上（每条 `plugin/state` 一次），
        // 而**墓碑数**比上限还小的时候根本不可能有超额的行——不值得为此扫一遍、分配一个
        // `Vec`。比的是 `disposed`，不是 `map.len()`：一个有几十条活 Fiber 的 profile
        // 不该只因为活着的行多就走到下面的扫描分支。
        if g.disposed <= MAX_DISPOSED_FIBERS {
            return;
        }
        let excess = g.disposed - MAX_DISPOSED_FIBERS;
        let stale: Vec<u32> = g
            .map
            .iter()
            .filter(|(_, v)| v.state == "disposed")
            .map(|(k, _)| *k)
            .take(excess)
            .collect();
        for id in stale {
            g.map.remove(&id);
            g.disposed -= 1;
        }
    }
    pub fn snapshot(&self) -> Vec<FiberStateInfo> {
        self.0.lock().map.values().cloned().collect()
    }
}

/// 方法表。`Arc<Self>`：路由既要被每条连接的 `serve` 任务共享，又要在自己被丢掉的时候
/// 把挂在世界日志上的观察者撤掉（[`Drop`]）。
pub struct KernelRouter {
    pub services: Arc<Services>,
    /// `Arc`：每个 [`crate::remote::RemotePlugin`] 都揣一份（它要 `execute_accept_ms` /
    /// `cancel_grace_ms` 去造执行器），而那些 Fiber 活得比一次方法调用长。
    pub config: Arc<RouterConfig>,
    fibers: Arc<FiberStates>,
    /// 撤销上面那个 `plugin/state` 观察者。`Disposer` 是 `FnOnce`，光 drop 它不会执行，
    /// 留着会在 `WorldLog` 的观察者表里堆一条死项（每次 `rpc-server` 重新 apply 就多
    /// 一条）。
    observer: Mutex<Option<Disposer>>,
    /// 心跳监视任务（见 [`spawn_heartbeat_monitor`]）。
    ///
    /// **真正停下它的是 `abort`**，不是那个 `Weak<Services>`：`Weak` 要等最后一个
    /// `Arc<Services>` 消失，而那要等 `KernelRouter` 本身被丢掉，也就是要等每条连接的
    /// `serve` 任务和监听器都放开各自的 `Arc<dyn Router>`——一条间接得多的链子。
    /// `Weak` 是兜底（任务绝不会让那堆服务多活一秒），`abort` 是开关：
    /// [`KernelRouter::monitor_disposer`] 交给 `rpc-server` 的 `ctx.on_dispose`，于是
    /// "这个 Fiber 被撤了"和"这个任务停了"是同一步；[`Drop`] 里再补一刀，保证一个没走
    /// 插件路径构造出来的路由（测试）也不会漏下一个幽灵任务。
    monitor: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Drop for KernelRouter {
    fn drop(&mut self) {
        if let Some(d) = self.observer.lock().take() {
            d();
        }
        if let Some(h) = self.monitor.lock().take() {
            h.abort();
        }
    }
}

impl KernelRouter {
    /// 必须在 tokio runtime 内调用（心跳监视任务是 `tokio::spawn` 出去的）——`rpc-server`
    /// 插件的 `apply` 本来就已经在起监听器了，同一个前提。
    pub fn new(services: Arc<Services>, config: RouterConfig) -> Arc<Self> {
        let fibers = Arc::new(FiberStates::default());
        // 观察者只攥 `FiberStates` 和 `ConnRegistry`，不攥路由本身：攥了就是一个环
        // （路由 → `Disposer` → 世界日志 → 观察者 → 路由），`Drop` 永远不会跑。
        let sink = fibers.clone();
        let conns = services.conns.clone();
        let observer = services.log.observe(Arc::new(move |ev: &WorldEvent| {
            if ev.kind.as_str() != PLUGIN_STATE {
                return;
            }
            // 跑在 append 线程上：解一次 payload、写一次快照、给每条连接 `try_send`
            // 一条通知，全程非阻塞，不回头 append。
            if let Ok(info) = serde_json::from_value::<FiberStateInfo>(ev.payload.clone()) {
                sink.record(info);
            }
            for conn in conns.iter() {
                let _ = conn.notify(PLUGIN_STATE, ev.payload.clone());
            }
        }));
        let monitor = spawn_heartbeat_monitor(&services, heartbeat_scan_period(&config));
        Arc::new(Self {
            services,
            config: Arc::new(config),
            fibers,
            observer: Mutex::new(Some(observer)),
            monitor: Mutex::new(Some(monitor)),
        })
    }

    /// 停掉心跳监视任务的 `Disposer`，交给 `rpc-server` 的 `ctx.on_dispose`。
    ///
    /// 只攥一个 `AbortHandle`，不攥路由也不攥 `Services`——一个"为了能停掉它而让它活着"
    /// 的 `Disposer` 是自相矛盾的。幂等：`abort` 可以调用任意多次，[`Drop`] 里那一刀
    /// 照常补。
    pub fn monitor_disposer(&self) -> Disposer {
        let handle = self.monitor.lock().as_ref().map(|j| j.abort_handle());
        Box::new(move || {
            if let Some(h) = handle {
                h.abort();
            }
        })
    }
}

/// 心跳监视的扫描周期：默认心跳周期的一半（规格：`heartbeat_ms/2` 扫一次），夹在
/// `[10ms, 5s]` 里。夹下界是因为一个配成 `heartbeat_ms: 1` 的 profile 会让这个任务
/// 变成一个忙循环；夹上界是因为一个配成一小时的心跳不该让"插件失联"这件事一小时后
/// 才被发现。
///
/// 周期看的是**内核默认**的心跳，不是每个插件自报的那个，所以一个报了更短心跳的插件，
/// 失联到被撤之间最坏是 `3×自己的心跳 + 一个扫描周期`，而不是正好 3 拍。这是故意的
/// 取舍：一张按各家心跳排的定时器堆，换来的只是把一个本来就以"3 拍"为粒度的判定再精确
/// 一点。真要更快发现，把 `rpc-server` 的 `heartbeat_ms` 调小——那是这个旋钮的用途。
fn heartbeat_scan_period(cfg: &RouterConfig) -> Duration {
    Duration::from_millis(cfg.heartbeat_ms.max(2) / 2)
        .clamp(Duration::from_millis(10), Duration::from_secs(5))
}

/// 心跳监视：每 `period` 扫一遍连接表，把 deadline 已过的远端 Fiber 撤掉。
///
/// 持 `Weak<Services>`：路由（以及整个 `rpc-server` Fiber）被撤销之后，这个任务不该
/// 成为让那一堆服务活下去的理由。`upgrade` 失败就是"这个 RPC 服务器已经没了"，退出。
fn spawn_heartbeat_monitor(
    services: &Arc<Services>,
    period: Duration,
) -> tokio::task::JoinHandle<()> {
    let weak = Arc::downgrade(services);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let Some(services) = weak.upgrade() else {
                break;
            };
            sweep_heartbeats(&services);
        }
    })
}

/// 一次扫描。跑在自己的任务上，不持有任何 `ConnState` 锁做下面这些事（`notify` 会走
/// 背压路径回头跑 `on_close` 钩子，`append` 会同步跑全部观察者——两条都会回到这把锁）。
fn sweep_heartbeats(services: &Arc<Services>) {
    let now = tokio::time::Instant::now();
    // 一份快照用到底：被判失联的那条 `plugin/state` 要广播给**所有**连接，而不是只给
    // 拥有它的那一条（下面详述）。
    let conns = services.conns.iter();
    for conn in &conns {
        let expired: Vec<(u32, crate::conn::RemoteFiber)> = {
            let mut st = conn.state.lock();
            let ids: Vec<u32> = st
                .fibers
                .iter()
                .filter(|(_, f)| f.deadline <= now)
                .map(|(id, _)| *id)
                .collect();
            ids.into_iter()
                .filter_map(|id| st.fibers.remove(&id).map(|f| (id, f)))
                .collect()
        };
        for (id, f) in expired {
            // 先通知、后 dispose。内核那条"这个 Fiber 没了"的 `plugin/state` 要经 Op
            // 通道绕一圈才回来，而这一条是直接入队的——先发才能保证带着
            // `error: "heartbeat timeout"` 的这条排在前面。反过来的话，客户端先看到一条
            // 没有原因的 `disposed`，"为什么"永远晚一步到，或者干脆因为连接已经在关而
            // 再也不到。
            //
            // 广播给**所有**连接，和 `plugin/state` 的其余部分一个待遇（观察者那边就是
            // `conns.iter()` 逐个发）：只发给失联的那一条，等于说"谁掉了"这件事只有掉了
            // 的那个自己知道——而它恰恰是最可能收不到的那个。一台监控用的控制台挂在另一
            // 条连接上，它看到的将只有内核随后那条 `error: null` 的 `disposed`。
            let params = serde_json::to_value(FiberStateInfo {
                fiber_id: id,
                plugin: f.plugin.clone(),
                state: "disposed".into(),
                error: Some(HEARTBEAT_TIMEOUT.into()),
            })
            .unwrap_or(Value::Null);
            for target in &conns {
                let _ = target.notify(PLUGIN_STATE, params.clone());
            }
            // 运维要看见的那条：一个插件失联和一个插件正常退出，在审计里必须分得开。
            services.log.append(
                EventDraft::new(
                    EventKind::new("health/plugin_heartbeat_timeout").expect("static kind"),
                    "rpc",
                    serde_json::json!({
                        "plugin": f.plugin,
                        "fiber_id": id,
                        "conn": conn.id().0,
                    }),
                )
                .with_salience(200),
            );
            // 和断连清理同一条纪律：`Busy` 不是"撤掉了"。这条扫描本身跑在一个任务上，
            // 但它不是 async fn（下面那串 `notify`/`append` 都不能在锁里 await），
            // 所以重试也甩给一个任务。
            let kernel = f.kernel.clone();
            crate::handlers::send_op_retrying_detached(move || kernel.dispose_fiber(FiberId(id)));
        }
    }
}

/// 心跳超时那条 `plugin/state` 的 `error`。客户端按它分辨"我被判失联了"和"我自己
/// 注销的/内核关机了"。
const HEARTBEAT_TIMEOUT: &str = "heartbeat timeout";

/// 同步处理函数的结果包成 future：`Router::call` 的签名是 future，但大多数方法只是读一把
/// 锁、拷一份快照，没有任何可等的东西。
fn ready(r: Result<Value, RpcError>) -> BoxFuture<Result<Value, RpcError>> {
    Box::pin(async move { r })
}

impl Router for KernelRouter {
    fn call(
        &self,
        conn: Arc<Conn>,
        method: String,
        params: Value,
    ) -> BoxFuture<Result<Value, RpcError>> {
        let s = &self.services;
        match method.as_str() {
            ROBOT_CAPABILITIES => ready(handlers::robot::capabilities(s)),
            ROBOT_STATE => ready(handlers::robot::state(s)),
            ROBOT_HEALTH => ready(handlers::robot::health(s, &self.fibers)),

            BRAIN_ATTACH => ready(handlers::brain::attach(s, &conn, params)),
            BRAIN_DETACH => ready(handlers::brain::detach(&conn, params)),
            BRAIN_CURSOR => ready(handlers::brain::cursor(s, params)),
            BRAIN_WAKE_POLICY_UPDATE => ready(handlers::brain::update_wake_policy(s, params)),
            BRAIN_INJECT => ready(handlers::brain::inject(s, &conn, params)),

            TURN_STEER => ready(handlers::turn::steer(s, params)),
            TURN_INTERRUPT => ready(handlers::turn::interrupt(s, params)),
            TURN_DONE => ready(handlers::turn::done(&conn, params)),
            // 这三个要等一整条准入链（规则、审批、执行）跑完，是这张表里仅有的真·async。
            TURN_PROPOSE => Box::pin(handlers::turn::propose(conn, params)),
            TURN_SAY => Box::pin(handlers::turn::say(conn, params)),
            TURN_ASK_HUMAN => Box::pin(handlers::turn::ask_human(conn, params)),

            STREAM_LIST => ready(handlers::stream::list(s)),
            // 真·async：第一次订阅一条本地流要建 Node、开共享内存段、起线程（见
            // `handlers::stream::subscribe` 里为什么是 `spawn_blocking`）。
            STREAM_SUBSCRIBE => Box::pin(handlers::stream::subscribe(s.clone(), params)),
            EVENT_EMIT => ready(handlers::event::emit(s, &self.config, &conn, params)),

            // `plugin/register` / `plugin/unregister` 都要等内核 settle 完那一轮
            // （`Op::AddFiber` / `Op::DisposeFiber` 的回执）。
            PLUGIN_REGISTER => Box::pin(handlers::plugin::register(
                s.clone(),
                self.config.clone(),
                conn,
                params,
            )),
            PLUGIN_HEARTBEAT => ready(handlers::plugin::heartbeat(&conn, params)),
            PLUGIN_UNREGISTER => {
                Box::pin(async move { handlers::plugin::unregister(&conn, params).await })
            }

            APPROVAL_ATTACH => ready(handlers::approval::attach(s, &conn, params)),
            APPROVAL_DETACH => ready(handlers::approval::detach(&conn, params)),

            // `log/tail` 要跨一把串行化这条连接上所有 tail 的异步锁；`log/replay` 要把
            // 读盘扔进 `spawn_blocking`。两个都是真·async。
            LOG_TAIL => Box::pin(handlers::log::tail(s.clone(), conn, params)),
            LOG_REPLAY => Box::pin(handlers::log::replay(
                s.clone(),
                self.config.clone(),
                params,
            )),
            LOG_FLUSH => ready(handlers::log::flush(s)),

            _ => ready(Err(RpcError::new(code::METHOD_NOT_FOUND, method))),
        }
    }

    /// 客户端发来的通知。跑在连接的读循环上（同步、不 spawn），所以两个处理函数都只做
    /// 一件短事：查一张表、盖个戳或者送一个 oneshot。执行期间的消息走通知而不是请求，
    /// 正是因为它们不需要回复——一个每 100 ms 发一次心跳的执行器不该为此等一次往返。
    fn notify(&self, conn: Arc<Conn>, method: String, params: Value) {
        match method.as_str() {
            CAPABILITY_HEARTBEAT => crate::remote::on_heartbeat(&conn, params),
            CAPABILITY_RESULT => crate::remote::on_result(&conn, params),
            _ => tracing::debug!(%method, "unhandled client notification"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Deserialize)]
    struct P {
        n: u64,
    }

    fn cfg(heartbeat_ms: u64) -> RouterConfig {
        RouterConfig {
            heartbeat_ms,
            execute_accept_ms: 2000,
            cancel_grace_ms: 500,
            run_tag: "local".into(),
            frames_max_bytes: 1024,
            max_streams_per_register: 64,
            max_capabilities_per_register: 64,
            max_inject_per_register: 64,
            max_provide_per_register: 64,
            max_fibers_per_conn: 16,
            max_register_bytes: 256 * 1024,
            max_nominal_hz: 20_000.0,
            max_replay_page: 4096,
            max_event_bytes: 64 * 1024,
            register_timeout_ms: 5000,
            #[cfg(feature = "test-hooks")]
            stream_setup_delay_ms: 0,
        }
    }

    /// 扫描周期是心跳的一半，但两头都夹住：一个 `heartbeat_ms: 1` 的 profile 不该把
    /// 监视任务变成忙循环（下界），一个一小时的心跳也不该让"插件失联"一小时后才被
    /// 发现（上界）。`heartbeat_ms: 0` 走不到这里（`plugin/register` 挡掉了），但除零
    /// 式的 `0/2 = 0` 周期会让 `interval` 直接 panic，所以照样夹住。
    #[test]
    fn heartbeat_scan_period_is_half_the_beat_and_clamped_both_ways() {
        assert_eq!(
            heartbeat_scan_period(&cfg(1000)),
            Duration::from_millis(500)
        );
        assert_eq!(heartbeat_scan_period(&cfg(100)), Duration::from_millis(50));
        assert_eq!(heartbeat_scan_period(&cfg(1)), Duration::from_millis(10));
        assert_eq!(heartbeat_scan_period(&cfg(0)), Duration::from_millis(10));
        assert_eq!(
            heartbeat_scan_period(&cfg(3_600_000)),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn parse_params_ok() {
        let p: P = parse_params(&json!({"n": 3})).unwrap();
        assert_eq!(p.n, 3);
    }

    #[test]
    fn parse_params_err_is_invalid_params() {
        let e = parse_params::<P>(&json!({"n": "x"})).unwrap_err();
        assert_eq!(e.code, code::INVALID_PARAMS);
        assert!(e.data.is_some());
    }

    /// 快照按 `fiber_id` 分流：同名两个 Fiber（一个被 dispose、同名的一行重新 add）
    /// 必须各占一条，而不是后来的那条把前一条盖掉。
    #[test]
    fn fiber_states_key_on_fiber_id_not_plugin_name() {
        let f = FiberStates::default();
        f.record(FiberStateInfo {
            fiber_id: 2,
            plugin: "adapter".into(),
            state: "disposed".into(),
            error: None,
        });
        f.record(FiberStateInfo {
            fiber_id: 7,
            plugin: "adapter".into(),
            state: "active".into(),
            error: None,
        });
        f.record(FiberStateInfo {
            fiber_id: 2,
            plugin: "adapter".into(),
            state: "failed".into(),
            error: Some("boom".into()),
        });
        let snap = f.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].fiber_id, 2);
        assert_eq!(snap[0].state, "failed");
        assert_eq!(snap[1].fiber_id, 7);
        assert_eq!(snap[1].state, "active");
    }
}
