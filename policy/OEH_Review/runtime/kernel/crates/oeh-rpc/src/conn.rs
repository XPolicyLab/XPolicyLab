//! 一条边界连接：出站队列、内核发起的反向请求（带响应回填）、断连清理钩子，以及驱动
//! 一条连接读写的 `serve`。
//!
//! 结构（Fix round 1，C1/H4）：`serve` 起两个独立任务——读任务只碰 `TransportRead`，写任务
//! 只碰 `TransportWrite`，谁也不共享谁的 `&mut`。两个任务通过 `Conn` 上的一个
//! `watch::Sender<bool>` 互相通知"该收工了"：谁先结束都会 `mark_closed()`，把这个 watch
//! 设成 `true`，另一个任务的 `select!` 立刻醒来退出并 `drop` 掉自己那一半传输——这样对端
//! 才会真正看到 EOF，而不是连接卡在"逻辑上关了，socket 还开着"的状态。
//!
//! 锁的纪律：这个文件里的 `parking_lot::Mutex` 只在拿数据/塞数据的一小段临界区里持有，
//! 绝不跨 `.await` 或回调（`on_close` 的 `Disposer` 是同步 `FnOnce`，谁要做异步清理自己
//! `tokio::spawn`）。

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::{Disposer, TurnHandle, WorldLog};
use oeh_proto::rpc::{code, Id, Message, Notification, Request, Response, RpcError};
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot, watch};

use crate::codec::{decode, encode, Transport};
use crate::router::Router;

/// 出站队列容量：超过就是背压，参见 [`Conn::try_enqueue`]。
const OUT_QUEUE_CAPACITY: usize = 4096;

/// 写半单次 `send` 的截止时间：对端不读、写半永远 `pending`，不能拖着整条连接（以及
/// `Client::close()` 之类等着它收工的调用者）永远不退出（Fix round 1，H4）。
pub const SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// `Conn::request`/`Client::call` 没显式给超时时用的默认值（Fix round 1，M4）。
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// [`serve`] 的读任务遇到 EOF/读错误时，`close_after_drain` 用的宽限期（Task 3 Fix round
/// 1，M2）：给已经派发出去、还在算/还没排进出站队列的回复一个机会真正发出去，而不是
/// 对端一断就立刻硬关。
pub const EOF_DRAIN_GRACE: Duration = Duration::from_millis(250);

/// 排水循环发现出站队列空了、但 [`Conn::in_flight`] 还不是零时，重新检查一次之间睡多久
/// （Task 3 Fix round 1，M2 第二轮：第一版用一个固定的空闲窗口猜"是不是排完了"，在跑满
/// 一整个 `dev-mock` profile 的真实内核上，`dispatch_line` 两跳 `tokio::spawn` 之外还要
/// 跟二十多个刚启动、正忙着抢 CPU 的插件任务排队，猜的窗口比真实延迟小，一样会把还在
/// 飞的回复当成"没有了"错过——`in_flight` 是精确计数，不是猜；这个值只决定"多快发现
/// 计数归零"，不是正确性依据）。
const DRAIN_POLL: Duration = Duration::from_millis(5);

/// 一条连接的数字 id（`plugin/register`、日志、`robot/health` 里报的那个）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnId(pub u64);

/// 内核发起、等客户端答复的反向请求表。`Arc` 包一层，见 [`Conn::pending`] 上的注释。
type PendingMap = Arc<Mutex<HashMap<Id, oneshot::Sender<Result<Value, RpcError>>>>>;

/// 尝试对一条已经关闭（或正在关闭）的连接做 I/O。
#[derive(Debug)]
pub struct Closed;

/// 一次反向请求在 `pending` 表里的那一条，不管这次请求怎么结束（收到响应、超时、连接
/// 关了、或者调用方把 future 丢掉）都会被摘掉。收到响应时 `resolve_pending` 已经先
/// `remove` 过了，这里的第二次 `remove` 是无害的空操作。
struct PendingGuard {
    pending: PendingMap,
    id: Id,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        // 先落到局部变量再出作用域：摘下来的是一个 `oneshot::Sender`，丢掉它会唤醒
        // 等待者，那种事不该带着 `pending` 这把锁做（对称于 `resolve_pending`）。
        let tx = self.pending.lock().remove(&self.id);
        drop(tx);
    }
}

/// 一个脑在这条连接上的落脚点：撤销那次 `Brains::attach` 的 `Disposer`，加上这个脑还
/// 没被 `turn/done` 收走的 turn 句柄（`turn/propose`/`turn/say`/`turn/done` 按 turn 号
/// 查表）。
///
/// `Default`：`RpcBrainSink::on_turn` 和 `brain/attach` 的处理函数谁先跑不确定（attach
/// 一返回，路由任务就可能立刻给这个脑开出第一个 turn），两边都用 `entry().or_default()`
/// 落座，顺序就不再是正确性的一部分。
#[derive(Default)]
pub struct BrainSlot {
    pub detach: Option<Disposer>,
    pub turns: HashMap<u64, TurnHandle>,
}

/// 一个跨进程插件（`plugin/register`）在这条连接上的落脚点。
///
/// 自带 `kernel`：撤掉这个 Fiber 的两条路——断连钩子和心跳监视任务——都够不着
/// [`crate::router::Services`]（钩子只有 `Weak<Conn>`，监视任务扫的是整张连接表），
/// 而"用哪个内核句柄加进去的，就用哪个撤"本来就是这条记录自己的事。
pub struct RemoteFiber {
    /// `manifest.name`，也就是这个 Fiber 在树上的 key。
    pub plugin: String,
    /// 客户端承诺的心跳周期；`plugin/heartbeat` 按它续期。
    pub heartbeat: Duration,
    /// 过了这一刻还没等到心跳，监视任务就把这个 Fiber 撤掉。`tokio::time::Instant`
    /// 而不是内核时钟：判定它的是一个 tokio 定时器，两边得是同一把尺子。
    pub deadline: tokio::time::Instant,
    pub kernel: Arc<oeh_kernel::KernelHandle>,
}

impl RemoteFiber {
    /// 收到一次心跳：deadline 推到 `now + 3×heartbeat`。三倍是给"丢两拍"留的余量——
    /// 一次 GC 停顿或者一次调度打嗝不该让一个健康的插件被判死。
    pub fn refresh(&mut self) {
        self.deadline = tokio::time::Instant::now() + 3 * self.heartbeat;
    }
}

/// 一次正在跑的跨进程执行（`capability/execute`）在这条连接上的落脚点。
///
/// 存在的理由是执行期间的两条消息走的是**另一条路**：`capability/heartbeat` /
/// `capability/result` 是客户端发来的通知，由连接的读循环处理，而等结果的是
/// `RemoteExecutor::execute` 返回的那个 future。这张表就是两者之间的接头。
pub struct ExecSlot {
    /// 这条登记的世代号（[`Conn::next_exec_gen`]）。收尾的守卫据此只摘**自己**那一条，
    /// 见 `remote::ExecSlotGuard`。
    pub generation: u64,
    /// 收到 `capability/result` 时把结果送给等结果的那个 future。`oneshot::Sender`
    /// 是 `FnOnce` 式的：送一次之后这一条就该从表里摘掉。
    pub result: oneshot::Sender<oeh_proto::rpc::types::CapabilityResult>,
    /// 收到 `capability/heartbeat` 时盖戳。只能盖戳——取消端（`ExecControl`）留在准入
    /// 侧，见 `ExecCtx::heartbeat_handle` 上的注释。
    pub heartbeat: oeh_admission::capability::HeartbeatHandle,
    /// 这次执行是**谁**的执行器在跑，以及断连时该找谁把它撤掉，见
    /// [`crate::remote::ExecOwner`]。放在这一条登记上，是因为断连那一刻能拿到的就只有
    /// 这张表：`Conn` 不认识准入服务，也不知道自己上面注册过哪些插件的哪些能力。
    pub owner: Arc<crate::remote::ExecOwner>,
}

/// 这条连接持有的、断连时必须归还的东西。
///
/// 锁的纪律（这把锁是 [`Conn::state`]）：只在取/放这些句柄的一小段临界区里持有，绝不跨
/// `.await`、`WorldLog::append` 或者 `Disposer`——`Disposer` 会 append（`brain/detach`），
/// append 会同步跑观察者，观察者会 `conn.notify`，而背压路径上的 `notify` 会反过来跑
/// `on_close` 钩子，钩子要拿的正是这把锁。
#[derive(Default)]
pub struct ConnState {
    /// 这条连接 attach 过、还没 detach 的脑，按 `brain_id`。
    pub brains: HashMap<String, BrainSlot>,
    /// `log/tail` 挂在世界日志上的观察者；再调一次 `log/tail` 先拆旧的。
    pub tail: Option<Disposer>,
    /// 这条连接 `plugin/register` 过、还活着的 Fiber，按 `fiber_id`。
    pub fibers: HashMap<u32, RemoteFiber>,
    /// `approval/attach` 装上的那个审批服务的撤销句柄（至多一个）。
    pub approval: Option<Disposer>,
    /// 正在这条连接上跑的跨进程执行，按 `exec_id`（= `proposal_id`）。
    ///
    /// **断连时不清空这张表**（见 [`release_all`](crate::handlers::release_all)）：清空
    /// 会把 `oneshot::Sender` 丢掉，等结果的 future 于是以"通道关了"收场，而它真正该
    /// 报的是 `connection lost`——那两件事对脑来说不是一回事。等结果的 future 自己盯着
    /// `wait_closed()`，并在任何一条退出路径上把自己这条摘掉。
    pub execs: HashMap<String, ExecSlot>,
    /// [`crate::remote::release_executions_on_disconnect`] 扫过这条连接了没有。每条连接
    /// 只扫一次，见那个函数上的说明。
    pub(crate) execs_released: bool,
    /// 断连清理钩子登记过了没有。钩子一条就够（它把上面两样一起归还），而
    /// `brain/attach`/`log/tail` 可以被调用任意多次——每次都登记一条的话，
    /// `Conn::on_close` 那个 `Vec` 会随着调用次数无限变长。
    pub(crate) cleanup_hooked: bool,
}

/// 一条连接。`Arc` 共享：路由方法拿到的是 `Arc<Conn>`，反向请求/通知/关闭钩子都靠它。
pub struct Conn {
    id: u64,
    name: Mutex<String>,
    out: mpsc::Sender<String>,
    /// 取走给 `serve()` 驱动写任务；只应该被取走一次。
    out_rx: Mutex<Option<mpsc::Receiver<String>>>,
    /// 内核（这一侧）发起、等客户端答复的反向请求。`Arc` 包一层：`request_timeout` 的超时
    /// 分支要在一个 `'static` 的 future 里把自己那条摘掉，够不着 `&self`，只能揣一份这个
    /// `Arc` 的克隆过去（Fix round 2，REG-4）。
    pending: PendingMap,
    next_req: AtomicU64,
    /// `ConnState.execs` 里每条登记的世代号发号器，见 [`Conn::next_exec_gen`]。
    next_exec: AtomicU64,
    closed: AtomicBool,
    /// 正在处理、还没排出响应的 Request 数——`dispatch_line` 收到 Request 时 +1，负责
    /// 发响应的那个派生任务收尾（不管是正常发完、被拒收、还是自己被取消）时 -1。排水
    /// 循环靠它判断"出站队列看着空了"是不是真的空了，而不是"回复还在算/还没排上"
    /// （Task 3 Fix round 1，M2 第二轮：见 [`serve`] 写任务排水循环上的注释）。
    in_flight: AtomicU64,
    /// [`Conn::close_after_drain`] 置位：`try_enqueue`（`notify`/反向请求走的那条路）据此
    /// 拒绝任何新行入队，好让排水期间队列只能变短，不会被新工作延长 deadline——但
    /// `try_enqueue_response`（`send_response` 走的那条路）故意不看它，见那两个方法上的
    /// 注释。写任务据此把"收到关闭信号就立刻收工"换成"把已经在队列里/还在飞的响应排完
    /// 再收工"（Task 3 Fix round 1，H2/M2/答案 A）。
    draining: AtomicBool,
    /// 通知写任务"该排水了"；跟 `shutdown` 是两条独立的 watch——写任务的循环每轮先看
    /// `drain`、再看 `shutdown`，这样不管两个 watch 谁先触发 `changed()`（比如读任务
    /// EOF 造成的 `mark_closed()` 几乎同时把 `shutdown` 也翻了），下一轮循环重新读到的
    /// 都是"当前状态"而不是"当时是哪个分支被选中"，`draining==true` 永远赢，不会被
    /// `shutdown==true` 抢跑打断排水（`watch::Receiver::borrow()` 总是给最新值，跟
    /// 哪个 `changed()` 恰好唤醒了这一轮无关）。
    drain: watch::Sender<bool>,
    /// 读/写任务互相通知关闭，也是 `wait_closed()` 的底子——用 `watch` 而不是
    /// `Notify`：`Notify::notify_waiters()` 不存许可，"先关后等"的调用者会永远等下去
    /// （Fix round 1，M2）；`watch` 存最后一个值，`wait_for`/`changed` 配合先 `borrow()`
    /// 探一次的写法两头都不会漏。
    shutdown: watch::Sender<bool>,
    /// `serve()` 起的两个任务的 `AbortHandle`；`close()` 用它们强制收尾（Fix round 2，
    /// REG-1）。`serve()` 从没跑过（比如连接还没上过传输就被关了）时是空的。
    abort_handles: Mutex<Vec<tokio::task::AbortHandle>>,
    /// 断连清理钩子，逆序执行。
    on_close: Mutex<Vec<Disposer>>,
    pub state: Mutex<ConnState>,
    /// `log/tail` 的串行化闸门。**不是** `state` 那把锁：一次 `log/tail` 要跨
    /// `WorldLog::observe`、整段 `since` 补发和一串 `conn.notify` 才算办完，那些都不能
    /// 在一把 `parking_lot` 锁底下做（notify 会经背压路径重入 `append` → 观察者 →
    /// 回到这里）。用 `tokio::sync::Mutex` 是因为持有它的那一段本身是 `async` 的。
    ///
    /// 为什么非要一把锁：`dispatch_line` 给每个 Request 单独 `tokio::spawn`，所以一条
    /// 连接上两次 `log/tail` 是真并发。少了它，两边都会读到"还没有旧订阅"、各自
    /// `observe()` 一个观察者，然后后写的那个把前一个的 `Disposer` 覆盖掉——`Disposer`
    /// 是 `FnOnce`，丢掉不等于执行，于是那个观察者永远留在世界日志里（连接死了也留着，
    /// 只是 `Weak<Conn>` upgrade 不出来），每条事件推两遍，而且 `append` 每次提交都要
    /// 多克隆它一次。
    tail_lock: tokio::sync::Mutex<()>,
    /// `plugin/register` 的串行化闸门，跟 [`Conn::tail_lock`] 同一个道理（`dispatch_line`
    /// 给每个 Request 单独 `tokio::spawn`，所以一条连接上两次 `plugin/register` 是真
    /// 并发），但要挡的东西不同：一次注册要跨"查重名 → `add_fiber().await` → 记进
    /// `ConnState.fibers`"三步，中间那步是 `.await`。没有这把闸门，两个同名的注册会
    /// 双双查到"还没注册过"，然后在树上留下两个同名 Fiber——而重名检查的**全部**意义
    /// 就是不让这种事发生。
    ///
    /// `Arc` 包一层（于是能 `lock_owned`）：超预算的那条路要把闸门**交给一个清理任务**
    /// 带走——那一行幽灵 Fiber 撤干净之前，同名的下一次注册必须等着（见
    /// `handlers::plugin::dispose_over_budget`），而一个借着 `&self` 的 guard 出不了这个
    /// 函数。
    plugin_lock: Arc<tokio::sync::Mutex<()>>,
    /// 背压事件（`health/rpc_backpressure`）写去哪；测试/独立用法可以不传。
    log: Option<Arc<WorldLog>>,
}

impl Conn {
    pub fn new(id: u64, log: Option<Arc<WorldLog>>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(OUT_QUEUE_CAPACITY);
        let (shutdown, _) = watch::channel(false);
        let (drain, _) = watch::channel(false);
        Arc::new(Self {
            id,
            name: Mutex::new(format!("conn-{id}")),
            out: tx,
            out_rx: Mutex::new(Some(rx)),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_req: AtomicU64::new(0),
            next_exec: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            in_flight: AtomicU64::new(0),
            draining: AtomicBool::new(false),
            drain,
            shutdown,
            abort_handles: Mutex::new(Vec::new()),
            on_close: Mutex::new(Vec::new()),
            state: Mutex::new(ConnState::default()),
            tail_lock: tokio::sync::Mutex::new(()),
            plugin_lock: Arc::new(tokio::sync::Mutex::new(())),
            log,
        })
    }

    pub fn id(&self) -> ConnId {
        ConnId(self.id)
    }

    /// 拿 `log/tail` 的闸门；见 [`Conn::tail_lock`] 上的注释。
    pub async fn lock_tail(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.tail_lock.lock().await
    }

    /// 给一次跨进程执行的登记发一个世代号。单调、每条连接独立；`ExecSlot` 存着它，
    /// 收尾的守卫据此认出"表里现在这一条还是不是我放的那一条"。
    pub fn next_exec_gen(&self) -> u64 {
        self.next_exec.fetch_add(1, Ordering::SeqCst)
    }

    /// 拿 `plugin/register` 的闸门；见 [`Conn::plugin_lock`] 上的注释。`Owned` 的 guard：
    /// 调用方可以把它移进一个任务里（超预算的清理就是这么做的）。
    pub async fn lock_plugins(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.plugin_lock.clone().lock_owned().await
    }

    /// 测试用：某个脑在这条连接上还攥着几个 turn 句柄。`#[doc(hidden)]`——不是给正常
    /// 调用方用的 API，存在的理由是"句柄表只增不减"是一类从外面看不出症状的 bug
    /// （`TurnHandle::propose` 自己会认出陈旧的 turn，所以泄漏的句柄不会出错，只会积）。
    #[doc(hidden)]
    pub fn turn_count(&self, brain_id: &str) -> usize {
        self.state
            .lock()
            .brains
            .get(brain_id)
            .map(|s| s.turns.len())
            .unwrap_or(0)
    }

    /// 客户端自报名（`plugin/register`/`brain/attach` 之后 Task 4/5 会 `set_name`），
    /// 默认 `"conn-<id>"`。
    pub fn name(&self) -> String {
        self.name.lock().clone()
    }

    pub fn set_name(&self, s: &str) {
        *self.name.lock() = s.to_string();
    }

    /// 内核 → 客户端的通知；`try_send`，队列满就把连接判死。
    pub fn notify(&self, method: &str, params: Value) -> Result<(), Closed> {
        let line = encode(&Message::Notification(Notification::new(method, params)));
        self.try_enqueue(line)
    }

    /// 内核 → 客户端的反向请求（`approval/request`、`capability/execute`……），
    /// [`DEFAULT_REQUEST_TIMEOUT`]（60s）超时。
    pub fn request(&self, method: &str, params: Value) -> BoxFuture<Result<Value, RpcError>> {
        self.request_timeout(method, params, DEFAULT_REQUEST_TIMEOUT)
    }

    /// 内核 → 客户端的反向请求，带显式超时。断连（之前已经关闭，或者排队时发现队满/已关）
    /// 时 `code::REFUSED`，不是 `KERNEL_GONE`——那个码留给"内核 Op 通道关了"，这里关的只是
    /// 这一条连接；超时是 `code::TIMEOUT`（Fix round 1，M4）。
    ///
    /// **排水中（`draining`）或已关闭时立刻 `REFUSED`**，不等 `timeout`。排水的定义就是
    /// "不再收新工作"：`try_enqueue` 会拒掉这一行，对端因此永远看不到这条请求，也就永远
    /// 不会答复它——干等下去只是把默认 60 s 的超时加在一次早已注定的失败前面。而这条路
    /// 上的调用方（`approval/request` 的审批、`capability/execute` 的一次执行）等的是一个
    /// 机器人动作的裁决：把 60 s 的静默换成一个当场的拒绝，脑才能改走别的路。
    pub fn request_timeout(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> BoxFuture<Result<Value, RpcError>> {
        if self.is_closed() || self.draining.load(Ordering::SeqCst) {
            return Box::pin(async { Err(RpcError::new(code::REFUSED, "connection closed")) });
        }
        let idx = self.next_req.fetch_add(1, Ordering::SeqCst);
        let id = Id::Str(format!("k{idx}"));
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id.clone(), tx);
        // 在 `id` 被 `Request::new` 移走之前克隆一份留给下面那个守卫用（Fix round 2，
        // REG-4：不然超时之后 `pending` 里那条永远留着，直到整条连接关闭才被
        // `drain_pending` 捎带清掉）。
        let pending_guard = PendingGuard {
            pending: self.pending.clone(),
            id: id.clone(),
        };
        let line = encode(&Message::Request(Request::new(id, method, params)));
        if self.try_enqueue(line).is_err() {
            // 上面那道检查之后才关/才开始排水（或者出站队列此刻满了）：同样是"这条请求
            // 永远出不去"。守卫在这里 drop，`pending` 里刚插进去的那条跟着摘掉——别的
            // 反向请求不受牵连（`drain_pending` 会把它们一并拒了，而它们可能还有答复）。
            drop(pending_guard);
            return Box::pin(async { Err(RpcError::new(code::REFUSED, "connection closed")) });
        }
        if self.is_closed() {
            // enqueue 之后才发现（或者促成）连接已关：确保这次插入不会被落下，
            // 参见 `try_enqueue`/`mark_closed` 里对这条竞争的说明。
            self.drain_pending();
        }
        Box::pin(async move {
            // 守卫**移进** future 里：一次反向请求的调用方完全可能不等到底就把这个
            // future 丢掉（Task 5：`Admission` 的 `ask_timeout_ms` 到点就丢掉
            // `RpcApproval::decide`；一次被 abort 的执行会丢掉 `capability/execute`），
            // 而那条路上没有任何一个分支会跑到——只有 `Drop` 会。少了它，一条长命的
            // 控制台连接每被审批超时一次就在 `pending` 里留一条死项。
            let _pending_guard = pending_guard;
            match tokio::time::timeout(timeout, rx).await {
                Ok(Ok(v)) => v,
                Ok(Err(_)) => Err(RpcError::new(code::REFUSED, "connection closed")),
                Err(_) => Err(RpcError::new(code::TIMEOUT, "request timed out")),
            }
        })
    }

    /// 测试用：当前挂着的反向请求数。`#[doc(hidden)]`——不是给正常调用方用的 API
    /// （Fix round 2，REG-4 的鉴别测试）。
    #[doc(hidden)]
    pub fn pending_len(&self) -> usize {
        self.pending.lock().len()
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// 这条连接是否在排水（[`Conn::close_after_drain`] 置位、还没关完）。
    /// `#[doc(hidden)]`：给测试用——排水是一个中间状态，正常调用方只该关心
    /// `is_closed()` 和自己那次调用的返回值。
    #[doc(hidden)]
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// 强制关闭这条连接：`mark_closed()`（标记关闭、唤醒 `wait_closed`、`pending` 一律
    /// `REFUSED`、逆序跑 `on_close`）之外，再把 `serve()` 登记过的两个任务 `abort()`
    /// 掉。幂等，`serve()` 还没跑过也能调用（这时候没有任务可 abort，循环体是空的）。
    ///
    /// 光靠 `mark_closed()` 翻转 `shutdown` 这个 watch 本来也能让两个任务自己在下一次
    /// `select!` 醒来时退出——但那两个任务是 `tokio::spawn` 出来、脱离了 `serve()` 那个
    /// future 的：谁都不再 `.await` `serve()` 之后，它们不会凭空停下来（`JoinHandle`
    /// 被丢弃只是"取消关注"，不是"取消任务"，这正是 REG-1 那次连接泄漏的成因）。
    /// `abort()` 直接把它们从 tokio 的调度里摘掉，不依赖它们下一次被 poll 到。
    pub fn close(&self) {
        self.mark_closed();
        let handles = std::mem::take(&mut *self.abort_handles.lock());
        for h in handles {
            h.abort();
        }
    }

    /// 等到连接关闭。用 `wait_for`：即使调用的时候连接已经关了，也不会因为
    /// `Notify`-式"只唤醒已经在等的人"而永远等下去（Fix round 1，M2）。
    pub async fn wait_closed(&self) {
        let mut rx = self.shutdown.subscribe();
        let _ = rx.wait_for(|closed| *closed).await;
    }

    /// 排水关闭：先让写任务把已经在队列里、以及已经被 `dispatch_line` 记成
    /// [`Conn::in_flight`]（正在算、还没排上）的响应都发完，再关，而不是像
    /// [`Conn::close`] 那样直接 abort。用于：监听器 `Disposer` 通知 `shutdown` 之后关闭
    /// 每条连接（Task 3 H2），以及 [`serve`] 的读任务遇到 EOF/读错误时（Task 3
    /// M2）——两处都是"已经有一条回复排上或者正在算，不该被一次 `mark_closed()` 抢跑
    /// 吞掉"的场景。
    ///
    /// `grace` 是这一整套排水动作的预算：`draining` 置位之后 `try_enqueue`（不是
    /// `try_enqueue_response`）拒绝任何新行，写任务的排水循环靠 `in_flight` 精确判断
    /// "真的没有更多了"（见 [`serve`]），正常情况下远用不到 `grace` 就会自己收尾，
    /// `wait_closed()` 那一刻就会解出；`grace` 只在响应卡在 `router.call` 里出不来、或者
    /// 对端不读、写半卡在 `SEND_TIMEOUT` 里这类病态情况下才会真正触发——到点后无条件
    /// 调用硬 `close()`，保证这条连接终归会关，不会因为一个卡住的对端/handler 而永远
    /// 挂着。
    pub async fn close_after_drain(&self, grace: Duration) {
        self.draining.store(true, Ordering::SeqCst);
        self.drain.send_replace(true);
        let _ = tokio::time::timeout(grace, self.wait_closed()).await;
        // 无论上面是排水正常收尾解出的，还是 `grace` 到点了：`close()`/`mark_closed()`
        // 全都幂等，这里补一刀既不会重复产生副作用，也保证卡住的对端最终仍然被拆掉。
        self.close();
    }

    /// 断连清理钩子；`d` 是同步 `FnOnce`，绝不能在里面 `await`（需要 async 的清理自己
    /// `tokio::spawn`）。连接已经关了就立刻跑，不会因为注册晚了而永远不执行。
    pub fn on_close(&self, d: Disposer) {
        self.on_close.lock().push(d);
        if self.is_closed() {
            self.drain_hooks();
        }
    }

    /// 取走写任务用的接收端；只应该被 [`serve`] 调用一次。
    fn take_out_rx(&self) -> Option<mpsc::Receiver<String>> {
        self.out_rx.lock().take()
    }

    fn subscribe_shutdown(&self) -> watch::Receiver<bool> {
        self.shutdown.subscribe()
    }

    fn subscribe_drain(&self) -> watch::Receiver<bool> {
        self.drain.subscribe()
    }

    /// [`serve`] 调用一次，登记它刚 `tokio::spawn` 出来的读/写任务的 `AbortHandle`，供
    /// [`Conn::close`] 用。
    fn register_serve_tasks(&self, handles: [tokio::task::AbortHandle; 2]) {
        *self.abort_handles.lock() = handles.into();
    }

    /// 已关闭就直接拒绝，绝不再碰 `out`——这是 H2 的关键一环：`record_backpressure` 会
    /// `append` 到 `WorldLog`，`WorldLog::append` 同步跑观察者，观察者常见的写法是把事件
    /// 转发成 `event`（k→c）通知、再调回 `conn.notify`；如果这里不先挡住已关闭的连接，
    /// 那条转发出来的通知会再次撞上满队列、再记一次背压事件、再转发一次……直接栈溢出。
    ///
    /// `draining` 期间一律再多拒一条：`notify`/`request_timeout` 发起的都是排水开始
    /// *之后* 才有的新工作，不该延长排水窗口。响应走的是 [`Conn::send_response`]、
    /// [`Conn::try_enqueue_response`]——不经过这条检查，理由见那边。
    fn try_enqueue(&self, line: String) -> Result<(), Closed> {
        if self.draining.load(Ordering::SeqCst) {
            return Err(Closed);
        }
        self.try_enqueue_response(line)
    }

    /// [`Conn::send_response`] 专用：只看 `is_closed`，故意不看 `draining`。一次
    /// Response 对应的是 `draining` 置位*之前*就已经被 `dispatch_line` 记进
    /// [`Conn::in_flight`] 的一个 Request——它是排水本身要等的东西，不是排水期间新长出
    /// 来的工作。如果这里也拒 `draining`，`Conn::close_after_drain`/`in_flight` 那一整套
    /// 机制就是在等一样东西永远进不了队列：`in_flight` 归零了，但归零之前它想发的那条
    /// 响应早被这里连同"排水期间的新工作"一起拒了（Task 3 Fix round 1，M2 第三轮——
    /// 加了 `in_flight` 精确计数之后，栽在了`try_enqueue`自己身上：它认识不出"这条本来
    /// 就该被放行"，只看见"现在在排水，一律不收"）。
    fn try_enqueue_response(&self, line: String) -> Result<(), Closed> {
        if self.is_closed() {
            return Err(Closed);
        }
        match self.out.try_send(line) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                // 先关（把 `closed` 定住），再决定要不要记事件：`mark_closed` 返回
                // "是不是我把它从开变关的"，只有这次调用才记，保证背压事件全程只记一条。
                let first = self.mark_closed();
                if first {
                    self.record_backpressure();
                }
                Err(Closed)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.mark_closed();
                Err(Closed)
            }
        }
    }

    fn record_backpressure(&self) {
        if let Some(log) = &self.log {
            log.append(
                EventDraft::new(
                    EventKind::new("health/rpc_backpressure").expect("static kind"),
                    "rpc",
                    serde_json::json!({"conn": self.id, "dropped": true}),
                )
                .with_salience(200),
            );
        }
    }

    fn drain_pending(&self) {
        let pending: Vec<_> = std::mem::take(&mut *self.pending.lock())
            .into_iter()
            .collect();
        for (_, tx) in pending {
            let _ = tx.send(Err(RpcError::new(code::REFUSED, "connection closed")));
        }
    }

    fn drain_hooks(&self) {
        let hooks: Vec<Disposer> = std::mem::take(&mut *self.on_close.lock());
        for d in hooks.into_iter().rev() {
            d();
        }
    }

    /// 任一方向（读/写）结束时调用：标记关闭、通过 `shutdown` 唤醒另一个任务和
    /// `wait_closed`、让所有挂起的反向请求以 `REFUSED` 收场、逆序跑完 `on_close` 钩子。
    /// 幂等（`AtomicBool::swap`）——重复调用只重新扫一遍 `pending`/`on_close`（正常情况下
    /// 这两个此时都已经空了）。返回这次调用是不是让连接从"开"变成"关"。
    fn mark_closed(&self) -> bool {
        let first = !self.closed.swap(true, Ordering::SeqCst);
        if first {
            // `send_replace`, not `send`: `watch::Sender::send` is a documented no-op (drops
            // the value, returns `Err`) when there are currently zero receivers — which is
            // exactly the shape of a `Conn` that closes via backpressure before `serve()`
            // ever runs. `send_replace` always stores the value regardless of receiver count,
            // so a later `subscribe()` (from `wait_closed()` or a `serve()` started on an
            // already-closed `Conn`) still observes `true` (Fix round 2, REG-2).
            self.shutdown.send_replace(true);
        }
        self.drain_pending();
        self.drain_hooks();
        first
    }

    /// 把一次 Response 塞进出站队列；队满/已关就悄悄丢（连接本来就要关了）。用
    /// [`Conn::try_enqueue_response`]，不是 [`Conn::try_enqueue`]：见那边关于
    /// `draining` 的说明。
    fn send_response(&self, resp: Response) {
        let line = encode(&Message::Response(resp));
        let _ = self.try_enqueue_response(line);
    }

    fn resolve_pending(&self, resp: Response) {
        // 先把 guard 落到局部变量里再 drop：`tx.send` 会唤醒等待者的任务，不能带着
        // `pending` 这把锁跑过去（Fix round 1，L4）。
        let tx = self.pending.lock().remove(&resp.id);
        let Some(tx) = tx else { return };
        let result = match (resp.result, resp.error) {
            (_, Some(e)) => Err(e),
            (Some(v), None) => Ok(v),
            (None, None) => Err(RpcError::new(
                code::INTERNAL,
                "response without result or error",
            )),
        };
        let _ = tx.send(result);
    }
}

/// 读任务/写任务收工时的兜底：不管是正常 `break` 出循环，还是任务内部哪里意外
/// `panic`（`tokio::spawn` 会在任务边界拦下 panic 变成 `JoinError`，但 unwind 仍然会先
/// 经过这个栈帧——`Drop::drop` 照样跑），`mark_closed()` 都会执行到（Fix round 1，H3）。
struct CloseOnDrop(Arc<Conn>);
impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.mark_closed();
    }
}

/// 每个负责"等 `router.call` 结果、发 Response"的任务持有恰好一个：不管它正常结束
/// （发完 Response）、提前 `return`（handle 被取消），还是自己被外部 `abort()`，`Drop`
/// 都会跑，`in_flight` 保证净变化正好是 `dispatch_line` 那次 `+1` 对应的 `-1`
/// （Task 3 Fix round 1，M2 第二轮）。
struct InFlightGuard(Arc<Conn>);
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `serve()`'s own drop guard — distinct from [`CloseOnDrop`], which lives *inside* the two
/// spawned tasks. This one lives in `serve()`'s own async fn body: if the *caller* stops
/// polling `serve()` (races it in a `select!` against a shutdown signal, wraps it in
/// `tokio::time::timeout`, aborts a task that owns it — anything that drops the `serve()`
/// future without letting it run to completion), the two tasks it spawned keep running
/// detached — a dropped `JoinHandle` only stops you from awaiting/aborting a task, it does
/// not stop the task. `Conn::close()` (`mark_closed()` + `abort()` on both tasks) is what
/// actually tears the connection down in that case (Fix round 2, REG-1).
struct CloseServeOnDrop(Arc<Conn>);
impl Drop for CloseServeOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

fn dispatch_line(conn: &Arc<Conn>, router: &Arc<dyn Router>, raw: &str) {
    match decode(raw) {
        Ok(Message::Request(req)) => {
            let call_conn = conn.clone();
            let router = router.clone();
            let id = req.id;
            let method = req.method;
            let params = req.params;
            // +1 之前这个 Request 已经算"收到了"：排水循环靠 `in_flight` 判断"出站队列
            // 空了"是不是真的没有更多回复要来了（Task 3 Fix round 1，M2 第二轮）。
            // `InFlightGuard` 保证不管下面这个任务是正常发完、被拒收、还是自己被取消，
            // 都会有恰好一次 -1，`send_response` 之外的所有 `return` 路径都不会漏。
            conn.in_flight.fetch_add(1, Ordering::SeqCst);
            // The guard is built *here*, not inside the task below: a `tokio::spawn`ed
            // future that is dropped before it is ever polled (runtime shutting down, the
            // task aborted between spawn and first poll) never runs a line of its body, so
            // a guard constructed in there would never exist and the `+1` above would
            // never be paired with a `-1` — `in_flight` would sit at a non-zero value
            // forever and every later drain would spin until its grace ran out (Task 4
            // fix). Constructed before the spawn and *moved* into the future, it is owned
            // by the future itself, so dropping the future drops it either way.
            let in_flight_guard = InFlightGuard(conn.clone());
            // 单独一个任务跑 `router.call`：JoinError 能把一次 panic 变成
            // `Response::err(.., INTERNAL)`，而不是让请求方永远等不到答复（Fix round 1，
            // 半个 M4）。外层再套一个任务负责等结果、发 Response——这样 `dispatch_line`
            // 本身不 `await` 任何东西，读循环不会被一个慢方法（或者一个卡住的 handler）
            // 拖住。
            let handle = tokio::spawn(async move { router.call(call_conn, method, params).await });
            let conn = conn.clone();
            tokio::spawn(async move {
                let _in_flight_guard = in_flight_guard;
                let resp = match handle.await {
                    Ok(Ok(v)) => Response::ok(id, v),
                    Ok(Err(e)) => Response::err(id, e),
                    Err(join_err) if join_err.is_panic() => {
                        Response::err(id, RpcError::new(code::INTERNAL, "handler panicked"))
                    }
                    Err(_cancelled) => return,
                };
                conn.send_response(resp);
            });
        }
        Ok(Message::Notification(n)) => {
            // `router.notify` 是同步调用，跑在读任务自己的栈上；不像 `router.call` 那样
            // 有 `tokio::spawn` 天然挡住 panic，所以这里手动 `catch_unwind`——否则一次
            // panic 会直接 unwind 穿过 `mark_closed()`（它在循环之后才跑），连接的
            // `is_closed()` 永远是 false，`on_close` 钩子永远不执行（Fix round 1，H3）。
            let conn_for_call = conn.clone();
            let router = router.clone();
            let method = n.method;
            let params = n.params;
            let panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
                router.notify(conn_for_call, method, params);
            }))
            .is_err();
            if panicked {
                tracing::warn!(conn = conn.id, "router.notify panicked; closing connection");
                // `mark_closed` flips the shared `shutdown` watch; the read loop notices it
                // at the top of its next iteration and stops calling into this router.
                conn.mark_closed();
            }
        }
        Ok(Message::Response(resp)) => {
            conn.resolve_pending(resp);
        }
        Err(e) => {
            // 解不出合法信封就没有 id 可用；用 `Id::Num(0)` 顶上，连接本身不受影响。
            conn.send_response(Response::err(Id::Num(0), e));
        }
    }
}

/// 驱动一条连接：一个读任务 + 一个写任务，直到某一侧的传输关闭、超时或者被对方叫停。
///
/// - 读任务：收到 `Request` → 单独任务跑 `router.call`（慢方法/panic 都不挡路），完成后把
///   `Response` 排进出站队列；收到 `Notification` → 同步调 `router.notify`（`catch_unwind`
///   包一层）；收到 `Response` → 从 `pending` 摘一个 oneshot 回填（[`Conn::request`] 发起的
///   反向请求）；一行解不出合法信封 → 回一条 `Response::err(Id::Num(0), ..)`，连接不关；
///   `recv()` 返回 `None`/`Err`，或者收到关闭信号 → 收工。
/// - 写任务：从出站队列取一行，`send()` 套 [`SEND_TIMEOUT`] 超时；成功就继续，超时/出错/
///   队列关闭/收到关闭信号 → 收工，**不会**把队列剩下的行发完。
/// - 不管哪个任务先收工，都会 `mark_closed()`：置位关闭标志、把 `shutdown` watch 设成
///   `true`（另一个任务的 `select!` 立刻醒来退出并 `drop` 自己那一半传输，对端才会看到
///   EOF）、清空 `pending`（`REFUSED`）、逆序跑 `on_close`。两个任务各自套了
///   [`CloseOnDrop`]，就算任务体本身意外 panic，收工动作也不会被跳过。
pub async fn serve<T: Transport>(conn: Arc<Conn>, transport: T, router: Arc<dyn Router>) {
    let (mut read_half, mut write_half) = Box::new(transport).into_split();
    let mut out_rx = conn
        .take_out_rx()
        .expect("serve() must be called exactly once per Conn");

    let read_conn = conn.clone();
    let read_router = router.clone();
    let mut read_shutdown = conn.subscribe_shutdown();
    let read_task = tokio::spawn(async move {
        let _guard = CloseOnDrop(read_conn.clone());
        loop {
            if *read_shutdown.borrow() {
                break;
            }
            tokio::select! {
                line = read_half.recv() => {
                    match line {
                        None | Some(Err(_)) => {
                            // EOF / 读错误：不直接 `mark_closed()`——`dispatch_line` 给
                            // 上一条请求派生出去的响应任务可能还没把回复排进队列（甚至
                            // 还没算完），一断就硬关会把它悄悄吞掉（Task 3 Fix round
                            // 1，M2：`--rpc-stdio` 下 "写一行请求、关 stdin、等答案"这
                            // 个最常见的驱动方式，靠硬编码的 50% 赔率丢答案）。改成排水
                            // 关闭：给已经在飞、或者排水的短窗口内才姗姗排进队列的回复
                            // 一个机会，`EOF_DRAIN_GRACE` 兜底不会真的卡住。
                            read_conn.close_after_drain(EOF_DRAIN_GRACE).await;
                            break;
                        }
                        Some(Ok(raw)) => dispatch_line(&read_conn, &read_router, &raw),
                    }
                }
                _ = read_shutdown.changed() => break,
            }
        }
    });

    let write_conn = conn.clone();
    let mut write_shutdown = conn.subscribe_shutdown();
    let mut write_drain = conn.subscribe_drain();
    let write_task = tokio::spawn(async move {
        let guard = CloseOnDrop(write_conn);
        loop {
            // `draining` wins over `shutdown` even if both watches happen to have flipped
            // by the time this iteration runs (e.g. the read task's EOF path calls
            // `close_after_drain`, which sets `drain` before its own eventual
            // `mark_closed()` sets `shutdown`, all without this task ever being polled in
            // between) — checking it *first*, every iteration, rather than as another
            // `select!` arm that `break`s, is what makes that hold: `watch::Receiver::
            // borrow()` always returns the *current* value, regardless of which
            // `changed()` future is what woke this loop up (Task 3 Fix round 1, H2/M2).
            if *write_drain.borrow() {
                // Drain to "confirmed nothing more is coming", not "queue looked empty
                // once": `try_recv` finding the queue empty doesn't mean there isn't a
                // response still on its way — `dispatch_line` counts those in
                // `Conn::in_flight` (see `InFlightGuard`) precisely so this loop doesn't
                // have to *guess* a timeout for "probably done" (a first version did, with
                // a fixed idle window; under a real kernel boot's worth of competing tasks
                // that guess was reliably too short, dropping the very reply this exists to
                // protect — Task 3 Fix round 1, M2's second round). `DRAIN_POLL` only
                // controls how quickly a drop to zero is *noticed*, not whether draining is
                // correct; the actual deadline is the caller's `close_after_drain(grace)`
                // `timeout`, which hard-`close()`s (aborting this task outright) once it
                // elapses regardless of what this loop is doing.
                loop {
                    match out_rx.try_recv() {
                        Ok(line) => {
                            // Same `match` as the steady-state loop below, deliberately:
                            // the first version only broke on the *timeout* (`.is_err()`
                            // on the `timeout` result) and treated `Ok(Err(io))` — the
                            // peer already gone, every subsequent `send` certain to fail
                            // the same way — as "sent, keep draining", so a half-closed
                            // socket burned the caller's whole `close_after_drain(grace)`
                            // budget one dead write at a time (Task 4 fix).
                            match tokio::time::timeout(SEND_TIMEOUT, write_half.send(line)).await {
                                Ok(Ok(())) => {}
                                _ => break,
                            }
                        }
                        Err(mpsc::error::TryRecvError::Disconnected) => break,
                        Err(mpsc::error::TryRecvError::Empty) => {
                            if guard.0.in_flight.load(Ordering::SeqCst) == 0 {
                                break;
                            }
                            tokio::time::sleep(DRAIN_POLL).await;
                        }
                    }
                }
                break;
            }
            if *write_shutdown.borrow() {
                break;
            }
            tokio::select! {
                outgoing = out_rx.recv() => {
                    match outgoing {
                        Some(line) => {
                            match tokio::time::timeout(SEND_TIMEOUT, write_half.send(line)).await {
                                Ok(Ok(())) => {}
                                _ => break,
                            }
                        }
                        None => break,
                    }
                }
                // Neither arm `break`s directly: looping back to the top re-reads both
                // flags fresh, so `draining` (checked first, above) can't be pre-empted by
                // a `shutdown` that raced it into `true` at the same instant.
                _ = write_shutdown.changed() => {}
                _ = write_drain.changed() => {}
            }
        }
    });

    conn.register_serve_tasks([read_task.abort_handle(), write_task.abort_handle()]);
    // Must be created *after* `register_serve_tasks`: if `serve()`'s own future is dropped
    // while suspended at the `tokio::join!` below (the only await point from here on), this
    // guard's `Drop` — which runs `Conn::close()` — needs the handles already in place to
    // actually abort the two tasks (Fix round 2, REG-1).
    let _serve_guard = CloseServeOnDrop(conn.clone());

    let _ = tokio::join!(read_task, write_task);
}
