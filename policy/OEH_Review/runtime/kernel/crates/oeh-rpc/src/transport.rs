//! 监听器：把 UDS / WebSocket / stdio 三种传输接到同一个 [`Router`] 上，以及活跃连接表
//! （[`ConnRegistry`]）。每条新连接独立起一个 [`serve`] 任务；`robot/health` 的
//! `connections` 数直接读 `ConnRegistry::len`。
//!
//! [`spawn_listeners`] 返回的 [`Disposer`]：先把 `ConnRegistry` 标成"正在关"（新连接一律
//! 被拒），再 `abort` 全部 accept 循环和还在飞的 ws 握手任务，然后给当时还活着的每条连接
//! 发一次 `shutdown` 通知、`Conn::close_after_drain()`（Task 3 Fix round 1，H2/M1：既保证
//! 通知真的被写出去，也不给并发涌入的连接留一条谁都够不着的漏网之鱼），最后（有 UDS 的话）
//! 删掉 socket 文件。

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use oeh_kernel::{Disposer, WorldLog};
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use crate::codec::{
    encode, LineTransport, Transport, TransportFuture, TransportRead, TransportWrite, MAX_LINE,
};
use crate::conn::{serve, Conn};
use crate::router::Router;

/// 监听器 `Disposer` 通知每条连接 `shutdown` 之后，给它排水关闭的预算（控制器裁定：
/// "spawn close_after_drain(300 ms)"）。
const DISPOSE_GRACE: Duration = Duration::from_millis(300);
/// `close_all_connections` 在多线程 runtime 上，愿意为"所有连接都排水完"总共等多久（控制
/// 器裁定："recv_timeout(350 ms)"——比单条连接的 `DISPOSE_GRACE` 留一点调度余量，不是每条
/// 连接各自 350ms）。
const DISPOSE_WAIT_BUDGET: Duration = Duration::from_millis(350);

/// 默认的并发连接数上限（`rpc-server` 的 `max_connections` 配置项）。
///
/// 守的还是**内核任务**，和 `plugin/register` 那六道护栏同一个理由：每条连接可以注册
/// `max_fibers_per_conn`（默认 16）个远端 Fiber，所以没有这个上限的话，全局的 Fiber 数就是
/// `16 × 无穷`。而 `FiberTree::settle()` 是 O(行 × inject) 且跑在 `Kernel::run` 那一条任务上
/// （每次 `Op::AddFiber` 一遍），N 条连接各注册 16 个就是那条任务上的 Θ(N²)——它同时还在
/// 串行处理 `Op::IngestFrame`。此外每条连接本身就是一个 4096 格的出站队列加两个任务
/// （Plan 3 最终评审 I2）。
///
/// 256 对任何真实部署都绰绰有余（控制台 + 适配器 + 脑，个位数到几十），真需要更多的部署
/// 把它调大就是了。
const DEFAULT_MAX_CONNECTIONS: usize = 256;

/// `health/rpc_connection_refused` 的节流：每秒最多一条。
///
/// 和 `health/frame_egress_drop` 同一个形状，理由也一样——被拒的连接正是**成串**来的
/// （一个会重连的客户端、一次泛洪），而每条事件都要走一遍完整的观察者链。事件里带累计
/// 次数，所以节流掉的那些不是没了，只是合并了。
const REFUSAL_REPORT_PERIOD: Duration = Duration::from_secs(1);

/// 被 [`ConnRegistry::insert`] 拒掉的连接，写那一行 `shutdown` 通知的预算。
///
/// 只是一次 `write_all` + EOF，正常情况下是微秒级；这个超时防的是"对端连上就不读"——那种
/// 连接不该把一个 accept 循环派生出来的任务永远挂在那里。
const REFUSAL_NOTICE_TIMEOUT: Duration = Duration::from_secs(2);

/// [`ConnRegistry::insert`] 的结果：收下了，还是因为哪一种原因没收。
///
/// 两种拒绝的收尾方式不一样，所以不能是一个 `bool`：`Closing` 的时候监听器 `Disposer` 正在
/// 给每条**已经在表里**的连接发 `shutdown`，这一条还没进表、也永远不会被 `serve`，直接硬关
/// 就是对的；`Full` 是一个稳态的策略拒绝，客户端有权知道自己为什么刚连上就被断开
/// （见 [`spawn_conn`]）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// 收下了，连接已经在表里。
    Ok,
    /// 监听器 `Disposer` 已经开始收尾（[`ConnRegistry::set_closing`]）。
    Closing,
    /// 活跃连接数已经到了上限（见 [`DEFAULT_MAX_CONNECTIONS`]）。
    Full,
}

/// 活跃连接表：accept 时 [`ConnRegistry::insert`]，连接关闭时（`on_close` 钩子）
/// [`ConnRegistry::remove`] 自己。`robot/health` 的 `connections` 数、`Disposer` 广播
/// `shutdown` 都靠它。
pub struct ConnRegistry {
    conns: Mutex<HashMap<u64, Arc<Conn>>>,
    /// 监听器 `Disposer` 已经开始收尾：翻转之后 `insert` 一律拒绝——accept 循环里正在飞
    /// 的、ws 握手还没做完的连接，即便抢在被 `abort()` 之前多接了一条，也不会再留下一条
    /// 永远收不到 `shutdown`、永远不会被关掉的连接（Task 3 Fix round 1，M1）。
    closing: AtomicBool,
    /// 同时活着的连接数上限，见 [`DEFAULT_MAX_CONNECTIONS`]。**至少是 1**（见
    /// [`ConnRegistry::with_max`]）。
    max: usize,
    /// 累计拒了多少条（只增不减，进事件的 payload）。
    refused_total: AtomicU64,
    /// 上一条 `health/rpc_connection_refused` 的时刻。`Mutex` 而不是原子：这条路每秒最多
    /// 做一次真正的工作，而"这一秒该我记"不该有两个 accept 同时认为自己赢了。
    refused_report: Mutex<Option<Instant>>,
}

impl Default for ConnRegistry {
    fn default() -> Self {
        Self::with_max(DEFAULT_MAX_CONNECTIONS)
    }
}

impl ConnRegistry {
    /// `max` 是同时活着的连接数上限，**下限 1**：`max_connections: 0` 会把第一条连接就
    /// 拒掉，而在 `--rpc-stdio` 下那是这个进程唯一的通道——连那条拒绝通知都会写在 stdout
    /// 上，也就是 JSON-RPC 通道本身。一个显然写错的数字不该把内核变成一个谁也连不上、
    /// 还往协议流里吐垃圾的进程（再审 R4）。别的护栏也都有自己的下限语义（`clamp(1, ..)`、
    /// `max(8)`），这个补上。
    pub fn with_max(max: usize) -> Self {
        Self {
            conns: Mutex::new(HashMap::new()),
            closing: AtomicBool::new(false),
            max: max.max(1),
            refused_total: AtomicU64::new(0),
            refused_report: Mutex::new(None),
        }
    }

    /// 当前的连接数上限（进 `health/rpc_connection_refused` 的 payload）。
    pub fn max(&self) -> usize {
        self.max
    }

    /// 记一次拒绝；这一秒该记事件的话返回**累计**次数（见 [`REFUSAL_REPORT_PERIOD`]）。
    ///
    /// 只在这把小锁下判定，`append` 由调用方在锁外做：`WorldLog::append` 会同步跑完整条
    /// 观察者链（其中一个给每条连接投通知），那不是能带着锁做的事。
    fn refusal_due(&self) -> Option<u64> {
        let total = self.refused_total.fetch_add(1, Ordering::Relaxed) + 1;
        let mut last = self.refused_report.lock();
        let now = Instant::now();
        if last.is_none_or(|t| now.duration_since(t) >= REFUSAL_REPORT_PERIOD) {
            *last = Some(now);
            Some(total)
        } else {
            None
        }
    }

    /// 登记一条新连接；`Disposer` 已经在收尾（[`ConnRegistry::set_closing`]）或者连接数
    /// 已经到上限就拒绝，调用方（[`spawn_conn`]）据此关掉这条刚建好、还没来得及 `serve`
    /// 的连接（Task 3 Fix round 1，M1；上限见 [`DEFAULT_MAX_CONNECTIONS`]）。
    ///
    /// "看 `closing`" 和 "插进表里" 必须在**同一把** `conns` 锁下完成。分开做的话，两者
    /// 之间有一个窗口：这条连接读到 `closing == false`，`set_closing` 紧接着翻转标志、
    /// `close_all_connections` 拿着锁把当时表里的连接全取走并关掉，然后这条连接才拿到锁
    /// 把自己插进一张再也不会有人遍历的表里——一条谁都够不着、永远收不到 `shutdown`、
    /// 永远不会被关掉的连接，正是 M1 那条修复要消灭的东西（Task 4 fix）。
    ///
    /// 数量检查在**同一把锁**下，出于同一个理由：`len()` 读一次、`insert` 再拿一次锁的
    /// 话，N 条并发 accept 会全部读到"还没满"，然后一起插进去——一个只在没有并发时成立的
    /// 上限守不住任何东西。
    pub fn insert(&self, conn: Arc<Conn>) -> Admit {
        let mut g = self.conns.lock();
        if self.closing.load(Ordering::SeqCst) {
            return Admit::Closing;
        }
        if g.len() >= self.max {
            return Admit::Full;
        }
        g.insert(conn.id().0, conn);
        Admit::Ok
    }

    pub fn remove(&self, id: u64) {
        // 落到局部变量、让 `MutexGuard` 在这条语句结束时就释放，再在下一条语句里
        // `drop`：`HashMap::remove` 的返回值如果直接被丢在 `.lock().remove(&id);`
        // 这一整条语句里，temporaries 是逆序析构的——`Option<Arc<Conn>>`（含
        // `Arc<Conn>` 的 `Drop`，可能跑还没排干的 `on_close` 钩子）会先于
        // `MutexGuard` 析构，也就是**带着锁**跑。今天不会真的出问题（调用方总还
        // 攥着另一份 `Arc`），但 Task 4/5 一旦有钩子里握着真实资源，这就是一个悄
        // 无声息的"锁内跑回调"（Task 3 Fix round 1，L3）。
        let removed = self.conns.lock().remove(&id);
        drop(removed);
    }

    pub fn len(&self) -> usize {
        self.conns.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.conns.lock().is_empty()
    }

    /// 当前存活连接的一份快照（`Arc` 克隆，不持锁跑回调——`notify`/`close` 都可能触发
    /// 同步的下游钩子）。
    pub fn iter(&self) -> Vec<Arc<Conn>> {
        self.conns.lock().values().cloned().collect()
    }

    /// `Disposer` 收尾前调用：翻转之后 [`ConnRegistry::insert`] 一律拒绝（Task 3 Fix
    /// round 1，M1）。
    ///
    /// 翻转也在 `conns` 锁下：和 [`ConnRegistry::insert`] 同一把锁，两者之间就没有
    /// "读到 false 之后才插进去"的窗口（见那边的注释）。这把锁在这里只罩着一次原子写，
    /// 不罩后面的 `iter()`/`close`——那些会跑下游钩子，不能带着锁进去。
    pub fn set_closing(&self) {
        let _g = self.conns.lock();
        self.closing.store(true, Ordering::SeqCst);
    }

    pub fn is_closing(&self) -> bool {
        self.closing.load(Ordering::SeqCst)
    }
}

/// 要监听的传输集合；哪个是 `None`/`false` 就不起哪个。
#[derive(Default)]
pub struct Listeners {
    pub uds: Option<PathBuf>,
    pub ws: Option<SocketAddr>,
    pub stdio: bool,
    /// UDS 的本机鉴权，见 [`UdsAuth`]。ws/stdio 不受它管（见那边的说明）。
    pub uds_auth: UdsAuth,
}

/// UDS 监听器的本机鉴权：socket 文件的权限位 + 允许连进来的 uid 白名单。
///
/// 为什么是两道而不是一道。权限位（`mode`）挡的是"别的用户根本 connect(2) 不上"——
/// 判定在操作系统内核里，便宜、可靠、也管得住那些还没写一个字节就想连的对端；但它的
/// 粒度只到"文件的属主/组/其他人"，对**同一个用户下的另一个进程**（以及 root）无能为
/// 力，而且一个手滑的 `chmod` 就能把它整个拿掉。白名单（`peer_uids`）是这条边界自己
/// 的策略：accept 之后再问一次 `SO_PEERCRED`，这是唯一一道能在世界日志里留下
/// `safety/auth_refused` 的关卡——而"谁试过连这台机器人"正是事后要查的东西。
///
/// **只管 UDS**：ws 是一条 TCP 连接，`SO_PEERCRED` 在它上面没有意义（对端可能在另一台
/// 机器上），stdio 的"对端"就是起这个进程的那个人。这两条传输的边界由部署决定，不由
/// 这里——ws 默认不开，而且要 `ws` 和 `ws_allow_unauthenticated: true` 两个键一起写才
/// 开得起来（见 `rpc-server` 的配置），所以"这条传输上没有鉴权"是部署明说过的事。
pub struct UdsAuth {
    /// 绑完 socket 立刻 `chmod` 成这个权限位（`rpc-server` 的 `uds_mode`，默认 0600）。
    pub mode: u32,
    /// 允许连进来的 uid（`rpc-server` 的 `peer_uids`，默认只有当前 uid）。
    ///
    /// **空表示谁也不许**，不是"谁都行"：一份显式写成 `peer_uids: []` 的配置，唯一
    /// 说得通的读法就是"不放任何人进来"。同一个理由，[`current_uid`] 读不出来时默认
    /// 值也是空的——一个连自己是谁都不知道的进程，不该替运维猜一个白名单出来。
    pub peer_uids: Vec<u32>,
}

impl Default for UdsAuth {
    fn default() -> Self {
        Self {
            mode: DEFAULT_UDS_MODE,
            peer_uids: current_uid().into_iter().collect(),
        }
    }
}

/// 默认的 UDS socket 权限位：只有属主能读写。
pub const DEFAULT_UDS_MODE: u32 = 0o600;

/// 当前进程的**有效** uid，答不上来时 `None`。
///
/// 必须是 **effective** uid：[`authorize_peer`] 拿 `peer_cred()` 当尺子，而
/// `SO_PEERCRED`（Linux）/ `LOCAL_PEERCRED`（macOS、\*BSD）报的是对端 connect 那一刻
/// 的 euid。两边不是同一把尺子的话，一个 setuid 程序会在一边算自己人、在另一边算外人。
///
/// 返回 `Option` 不是因为 unix 上会失败——`geteuid(2)` 按 POSIX 不返回错误——而是因为
/// [`UdsAuth::peer_uids`] 的 fail-closed 语义（答不上来 ⇒ 空白名单 ⇒ 谁也不许）要有一
/// 个"答不上来"可表达。非 unix 平台走那一支。
///
/// **从前这里读 `/proc/self/status` 的 `Uid:` 行**，理由是不引 `libc`。那条路在没有
/// `/proc` 的 unix（macOS、\*BSD）上一律返回 `None`，于是默认白名单是空的，于是**每一
/// 条** UDS 连接都在 accept 之后被拒——一条本该只在配置显式写了 `peer_uids: []` 时才
/// 关上的门，在整个平台上默认关死，而症状是每个客户端都看到"connection closed"，跟
/// socket 路径写错、跟内核正在关，长得一模一样。`libc` 早就在这个工作区的依赖图里
/// （`tokio` 在 unix 上依赖它），所以换成 `geteuid()` 没有新依赖，也把"读得出自己是
/// 谁"从一件依赖 `/proc` 存在的事变成了平台无关的事实。
#[cfg(unix)]
pub fn current_uid() -> Option<u32> {
    // SAFETY: `geteuid(2)` 不接受参数、不解引用任何指针、不写调用方的内存，按 POSIX
    // 也永不失败——这次调用没有任何要维持的前置条件。
    Some(unsafe { libc::geteuid() })
}

/// 非 unix：没有 uid 这个概念，也就没有 `SO_PEERCRED` 这条边界。`None` 让
/// [`UdsAuth::peer_uids`] 保持 fail-closed。
#[cfg(not(unix))]
pub fn current_uid() -> Option<u32> {
    None
}

/// `spawn_listeners` 搭到一半的状态：两个监听器都已经绑好，但完整的 [`Disposer`] 还没能
/// 交给调用方之前，`Drop` 会把已经建好的东西拆掉——已经起的 accept 循环/握手任务
/// `abort()`，刚绑出来的 UDS socket 文件删掉。不这么做的话：uds 绑成功、ws 随后绑失败，
/// `?` 提前返回会让 uds 的 accept 循环成为一个再没有任何人持有 `AbortHandle` 的任务，
/// 也没有人删它的 socket 文件——对称于 `conn.rs` REG-1 那次"detached 任务没人管"的教训
/// （Task 3 Fix round 1，H1）。[`Building::into_disposer`] 把这个 guard 原地拆成真正
/// 长期持有的 `Disposer`：拆之前先置位 `armed`，`Drop` 看到就不再重复收尾。
struct Building {
    abort_handles: Vec<tokio::task::AbortHandle>,
    /// ws 每次握手都单独一个任务（见 `spawn_listeners`），数量随时间变化，用
    /// `Arc<Mutex<_>>` 让 accept 循环和这个 guard／最终的 `Disposer` 共享同一张表。
    handshake_tasks: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
    uds_path: Option<PathBuf>,
    registry: Arc<ConnRegistry>,
    armed: bool,
}

impl Building {
    fn new(registry: Arc<ConnRegistry>) -> Self {
        Self {
            abort_handles: Vec::new(),
            handshake_tasks: Arc::new(Mutex::new(Vec::new())),
            uds_path: None,
            registry,
            armed: false,
        }
    }

    /// 一切都绑好、也都 spawn 完了：把这份状态原地转成长期持有的 `Disposer`，并解除
    /// `Drop` 的收尾（那是留给"半路失败"的）。
    fn into_disposer(mut self) -> Disposer {
        self.armed = true;
        let abort_handles = std::mem::take(&mut self.abort_handles);
        let handshake_tasks = self.handshake_tasks.clone();
        let uds_path = self.uds_path.take();
        let registry = self.registry.clone();
        Box::new(move || {
            // (a) 先拦：翻转 `closing`，即便有连接抢在 (b) 的 `abort()` 之前从 accept
            // 循环/握手任务里跑出来，`spawn_conn` 也会直接把它关掉（Task 3 Fix round
            // 1，M1）。
            registry.set_closing();
            // (b) 再停：accept 循环 + 所有还在飞的 ws 握手任务——原来的实现只 abort 了
            // accept 循环，握手任务完全不受控，一次挂在半空的握手会在 `Disposer` 跑完
            // 之后才冒出一条全新的、没人会再通知/关闭的连接（Task 3 Fix round 1，M1）。
            for h in &abort_handles {
                h.abort();
            }
            for h in handshake_tasks.lock().iter() {
                h.abort();
            }
            // (c)(d) 通知 + 排水关闭每条连接，能等就等（Task 3 Fix round 1，H2）。
            close_all_connections(&registry);
            // 这次 run 绑出来的 UDS socket 文件；不存在（已经被清过，或者从来就没有）
            // 时是 no-op。
            if let Some(path) = uds_path {
                let _ = std::fs::remove_file(path);
            }
        })
    }
}

impl Drop for Building {
    fn drop(&mut self) {
        if self.armed {
            return;
        }
        for h in &self.abort_handles {
            h.abort();
        }
        for h in self.handshake_tasks.lock().iter() {
            h.abort();
        }
        if let Some(path) = &self.uds_path {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// 每条活着的连接：`notify("shutdown", ...)` **就在这里、同步地**入队（这时候
/// `draining` 还没置位，一定能排上），然后才各自起一个任务去 `close_after_drain`。
///
/// 顺序是这条函数的正确性所在，不是写法偏好：`close_after_drain` 一置位 `draining`，
/// `try_enqueue`（`notify` 走的那条路）就开始拒收一切新行。通知如果和排水关闭一起塞进
/// 那个 spawn 出去的任务里，它能不能排上就取决于任务体里两条语句的相对顺序——一旦有人
/// 把它们调个个儿，客户端就再也收不到 `shutdown`，只会看见 socket 无缘无故断了。
///
/// 能等就等：`Handle::try_current()` 报告一个多线程 runtime 时，用
/// `tokio::task::block_in_place` 把"等完成信号"挪到一个专门的线程上（多线程 runtime
/// 需要这个提示才会把别的任务挪去别的 worker，不然会跟这些通知任务抢同一个线程）、
/// `std::sync::mpsc::Receiver::recv_timeout` 卡一个 [`DISPOSE_WAIT_BUDGET`] 总预算——这样
/// `dispose()` 返回的时候，"shutdown" 那几个字节已经真的写进了对端的 socket，而不是还揣
/// 在一个刚 `tokio::spawn`、谁都没等过的任务手里（Task 3 Fix round 1，H2：原来的实现
/// `tokio::spawn` 完就撒手，`oeh run` 的调用链在它之后再没有别的 `.await`，
/// `#[tokio::main]` 一退出，运行时直接把这个任务连同还没来得及发出去的通知一起扔掉——
/// 复现 5/5）。
///
/// current-thread runtime（`block_in_place` 会直接 panic：没有别的 worker 可以把任务
/// 挪过去）或者压根没有 runtime（`tokio::spawn`/`close_after_drain` 内部的
/// `tokio::time::timeout` 都无从谈起）这两种情况做不到"等"，退化成尽力而为，并把这个
/// 限制写在这里而不是让调用方去猜（Task 3 Fix round 1，L2）。
fn close_all_connections(registry: &Arc<ConnRegistry>) {
    let conns = registry.iter();
    if conns.is_empty() {
        return;
    }
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            let n = conns.len();
            for conn in conns {
                let _ = conn.notify("shutdown", json!({"reason": "kernel stopping"}));
                let tx = tx.clone();
                handle.spawn(async move {
                    conn.close_after_drain(DISPOSE_GRACE).await;
                    let _ = tx.send(());
                });
            }
            drop(tx);
            // 借 `rx`、不夺走它（只捕获共享引用的闭包是 `Copy`）：下面那条兜底路径要能
            // 再跑一次同一件等待。
            let wait = || {
                let deadline = std::time::Instant::now() + DISPOSE_WAIT_BUDGET;
                for _ in 0..n {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() || rx.recv_timeout(remaining).is_err() {
                        break;
                    }
                }
            };
            // `runtime_flavor()` 说"多线程"还不足以保证 `block_in_place` 不 panic：在一个
            // `LocalSet` 里（`spawn_local` 的任务跑在 `block_on` 的那个线程上）它照样
            // panic，而这一次 panic 会从这里 unwind 穿过 `Disposer`、穿过 `FiberTree` 的
            // dispose，把一次正常的关机变成一次崩溃，连 UDS socket 文件都留在原地
            // （Task 4 fix）。接住它，再原地跑同一件等待——这一层本来就不是 async，
            // `recv_timeout` 只是阻塞当前线程，`block_in_place` 无非是告诉 tokio
            // "把别的任务挪去别的 worker" 的一个提示，拿不到这个提示只是慢一点，不是
            // 做不了。
            let unavailable =
                std::panic::catch_unwind(AssertUnwindSafe(|| tokio::task::block_in_place(wait)))
                    .is_err();
            if unavailable {
                tracing::debug!(
                    "block_in_place is unavailable here (LocalSet?); waiting inline instead"
                );
                wait();
            }
        }
        Ok(handle) => {
            // current-thread runtime：阻塞这唯一的线程等完成信号会把自己也搭进去（跑
            // 这些通知任务的正是同一个线程），`block_in_place` 对这种 runtime 直接
            // panic。退化成尽力而为：通知照样同步入队（那一步不需要 runtime），排水关闭
            // spawn 出去但**不等**——好过在这里 panic 掉整个 dispose 调用链。
            //
            // 代价说在这里，省得调用方去猜：在 current-thread runtime 上，`dispose()`
            // 返回的那一刻那几条 `shutdown` 通知只是进了出站队列，还没被写进 socket；
            // 真正写出去要等调用方下一次把控制权交还给 runtime（一个 `.await`）。调用链
            // 之后直接退出进程的话（`#[tokio::main]` 一返回），它们会连同任务一起被丢掉。
            for conn in conns {
                let _ = conn.notify("shutdown", json!({"reason": "kernel stopping"}));
                handle.spawn(async move {
                    conn.close_after_drain(DISPOSE_GRACE).await;
                });
            }
        }
        Err(_) => {
            // 压根没有 tokio runtime：同步地尽力而为。没有写任务在跑，也就没有"排水"
            // 这回事，`notify` 之后直接硬关。
            for conn in conns {
                let _ = conn.notify("shutdown", json!({"reason": "kernel stopping"}));
                conn.close();
            }
        }
    }
}

/// 监听 `l` 里启用的全部传输；每条新连接 `Conn::new(next_id, Some(log))` 后 `serve` 同一个
/// `router`。绑定失败（比如 ws 端口被占）直接返回 `Err`——调用方（`rpc-server` 插件）据此
/// fail-closed，且不会留下另一个已经绑定/正在监听的传输（Task 3 Fix round 1，H1：两个
/// 监听器都先绑好，一个都不 spawn，才开始起 accept 循环）。
pub fn spawn_listeners(
    l: Listeners,
    router: Arc<dyn Router>,
    log: Arc<WorldLog>,
    registry: Arc<ConnRegistry>,
) -> io::Result<Disposer> {
    let uds_listener = match &l.uds {
        Some(path) => Some(bind_uds(path, l.uds_auth.mode)?),
        None => None,
    };
    let mut building = Building::new(registry.clone());
    // `l.uds` 为 `None` 时这也是 `None`：只有真的绑出了监听器（上面那一步没有提前
    // `?` 返回），才需要在失败路径上把这个文件删掉。
    building.uds_path = l.uds.clone();
    let ws_listener = match l.ws {
        Some(addr) => Some(bind_tcp(addr)?),
        None => None,
    };

    let next_id = Arc::new(AtomicU64::new(1));

    if let Some(listener) = uds_listener {
        let router = router.clone();
        let log = log.clone();
        let registry = registry.clone();
        let next_id = next_id.clone();
        let peer_uids = l.uds_auth.peer_uids.clone();
        let h = tokio::spawn(async move {
            let mut refused = AuthRefusalLog::default();
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        // 鉴权在**建连接之前**：被拒的对端不该有 `Conn`、不该进
                        // `ConnRegistry`、也不该有一条 `serve` 任务——它在这个进程里
                        // 留下的唯一痕迹就是下面那条事件。`stream` 在这个分支里被丢掉，
                        // 对端于是读到 EOF。
                        if let Err(refusal) = authorize_peer(&stream, &peer_uids) {
                            refused.record(&log, &refusal);
                            continue;
                        }
                        let (r, w) = stream.into_split();
                        spawn_conn(LineTransport::new(r, w), &router, &log, &registry, &next_id);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "uds accept failed; stopping uds listener");
                        break;
                    }
                }
            }
        });
        building.abort_handles.push(h.abort_handle());
    }

    if let Some(listener) = ws_listener {
        let router = router.clone();
        let log = log.clone();
        let registry = registry.clone();
        let next_id = next_id.clone();
        let handshake_tasks = building.handshake_tasks.clone();
        let h = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let router = router.clone();
                        let log = log.clone();
                        let registry = registry.clone();
                        let next_id = next_id.clone();
                        let handshake_tasks = handshake_tasks.clone();
                        // 握手（`accept_async_with_config`）本身要读写字节，不能挡住后面
                        // 等待连接的 accept 循环，单独一个任务跑；`AbortHandle` 记进共享
                        // 表，`Disposer` 靠它才拆得掉一次还没握完手的连接（Task 3 Fix
                        // round 1，M1）。
                        let handshake = tokio::spawn(async move {
                            // `MAX_LINE` 跟 UDS/stdio 的行长度上限对齐：默认配置下
                            // `max_message_size` 是 64 MiB、`max_frame_size` 16
                            // MiB——一条拆成多帧、每帧都在 16 MiB 以内、但拼起来超过
                            // 16 MiB 的消息会被 tungstenite 完整攒好交给我们，比 UDS/
                            // stdio 那条路径宽松 4 倍（Task 3 Fix round 1，M3）。
                            let config = WebSocketConfig::default()
                                .max_message_size(Some(MAX_LINE))
                                .max_frame_size(Some(MAX_LINE));
                            match tokio_tungstenite::accept_async_with_config(stream, Some(config))
                                .await
                            {
                                Ok(ws) => spawn_conn(
                                    WsTransport::server(ws),
                                    &router,
                                    &log,
                                    &registry,
                                    &next_id,
                                ),
                                Err(e) => tracing::warn!(error = %e, "ws handshake failed"),
                            }
                        });
                        // 顺手筛掉已经跑完的老条目，这张表才不会随着连接总数无限变长
                        // （多数握手几十毫秒内就完成，真正"还在飞"的永远是极少数）。
                        let mut tasks = handshake_tasks.lock();
                        tasks.retain(|h| !h.is_finished());
                        tasks.push(handshake.abort_handle());
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "ws accept failed; stopping ws listener");
                        break;
                    }
                }
            }
        });
        building.abort_handles.push(h.abort_handle());
    }

    if l.stdio {
        // 只有一条连接，没有 accept 循环可言：直接 serve stdin/stdout。
        spawn_conn(
            LineTransport::new(tokio::io::stdin(), tokio::io::stdout()),
            &router,
            &log,
            &registry,
            &next_id,
        );
    }

    Ok(building.into_disposer())
}

/// 新连接的公共尾段：分配 id、登记进 `registry`、挂 `on_close` 自摘钩子、`tokio::spawn`
/// `serve`。UDS/ws/stdio 三条路径共用。
fn spawn_conn<T: Transport>(
    transport: T,
    router: &Arc<dyn Router>,
    log: &Arc<WorldLog>,
    registry: &Arc<ConnRegistry>,
    next_id: &Arc<AtomicU64>,
) {
    let id = next_id.fetch_add(1, Ordering::SeqCst);
    let conn = Conn::new(id, Some(log.clone()));
    match registry.insert(conn.clone()) {
        Admit::Ok => {}
        Admit::Closing => {
            // `Disposer` 已经在收尾（`ConnRegistry::closing`）：这条连接永远不会被
            // `serve()`，也就没有写任务可排水，直接硬关，不留一条谁都够不着、永远收不到
            // `shutdown` 的连接（Task 3 Fix round 1，M1）。此时监听器正在给每条**在表里**
            // 的连接发 `shutdown`，没必要也没地方再给这一条单独编一条理由。
            conn.close();
            return;
        }
        Admit::Full => {
            // 连接数到顶（`max_connections`）。告诉对端为什么，否则它看到的是"刚连上就被
            // 断开"，而那和"socket 路径写错了"、"内核正在关"长得一模一样——一个会重连的
            // 客户端会就这么一直重连下去。
            //
            // 这一行**直接写在传输上**，不走 `conn.notify`：出站队列要有 `serve()` 起的
            // 写任务才会被排出去，而这条连接正是那条不会被 `serve` 的。写完就把两半都丢掉，
            // 对端于是读到一行 `shutdown` 再读到 EOF。
            conn.close();
            // 再给运维留一条痕迹：光看 `robot/health.connections` 顶在上限上，事后读日志的
            // 人看不出"有人一直在被拒"（再审 R3）。节流到每秒一条，带累计次数；在写那一行
            // 通知**之前**记，所以"对端收到了拒绝"就意味着这条事件已经进了日志。
            if let Some(total) = registry.refusal_due() {
                log.append(
                    EventDraft::new(
                        EventKind::new("health/rpc_connection_refused").expect("static kind"),
                        "rpc",
                        json!({"limit": registry.max(), "refused": total}),
                    )
                    .with_collapse_key("rpc_connection_refused".to_string())
                    .with_salience(200),
                );
            }
            // 全限定：这个文件里的 `Message` 是 tungstenite 的 ws 帧类型。
            let line = encode(&oeh_proto::rpc::Message::Notification(
                oeh_proto::rpc::Notification::new(
                    "shutdown",
                    json!({"reason": "too many connections"}),
                ),
            ));
            tokio::spawn(async move {
                let (_read, mut write) = Box::new(transport).into_split();
                let _ = tokio::time::timeout(REFUSAL_NOTICE_TIMEOUT, write.send(line)).await;
            });
            return;
        }
    }
    // `Weak`，不是 `Arc`：`ConnRegistry` 持有 `Arc<Conn>`，`Conn` 的 `on_close` 钩子如果
    // 也持有 `Arc<ConnRegistry>`，两者就是一个引用环——一条逃过上面这次 `closing` 检查
    // 的连接（或者任何原因下没有正常走到 `mark_closed`）会让整条连接和它拉着的
    // `ConnRegistry`（进而是全部八个服务）永远不被回收（Task 3 Fix round 1，L4）。
    let reg = Arc::downgrade(registry);
    conn.on_close(Box::new(move || {
        if let Some(reg) = reg.upgrade() {
            reg.remove(id);
        }
    }));
    tokio::spawn(serve(conn, transport, router.clone()));
}

/// 一次被拒的 accept：写进 `safety/auth_refused` 的那两个字段。
///
/// 单独一个类型而不是就地 `append`：鉴权判定发生在 accept 循环里，而 `WorldLog::append`
/// 会同步跑完整条观察者链（其中一条会给每条活着的连接投通知）。把"判定"和"记账"分开，
/// 判定那一步就还是一次纯函数，测试也能不起内核就验它（见这个文件末尾的单元测试）。
#[derive(Debug, PartialEq, Eq)]
struct AuthRefusal {
    /// 对端的 uid；`SO_PEERCRED` 本身就失败时是 `None`（进事件时是 `null`）。
    uid: Option<u32>,
    reason: &'static str,
}

/// `safety/auth_refused` 的节流账本，每个 UDS accept 循环一份。
///
/// 和隔壁 `health/rpc_connection_refused` 同一个形状，理由也一样（见
/// [`REFUSAL_REPORT_PERIOD`]）：被拒的连接是**成串**来的——一个会重连的客户端、一次泛洪
/// ——而每条事件都要同步走完一遍观察者链，世界日志又是按**条数**保留的（4096 条），一个
/// 连接循环能把真正的历史挤干净。节流掉的不是没了，只是合并了：事件里带累计次数，
/// `collapse_key` 按 uid 分，控制台上"这个 uid 一直在敲门"仍然是一条看得见的线索。
///
/// 不是 `Mutex`，因为不需要：一个 accept 循环就是一个任务，这份账本只有它自己碰。
#[derive(Default)]
struct AuthRefusalLog {
    last_report: Option<Instant>,
    total: u64,
}

impl AuthRefusalLog {
    fn record(&mut self, log: &WorldLog, refusal: &AuthRefusal) {
        self.total += 1;
        let now = Instant::now();
        if self
            .last_report
            .is_some_and(|t| now.duration_since(t) < REFUSAL_REPORT_PERIOD)
        {
            return;
        }
        self.last_report = Some(now);
        // `uid: null` 的那一支（`peer_cred()` 自己失败）也得有一个稳定的 key，不然它的
        // 事件会一条条堆在控制台上，而那恰恰是最没有信息量的一种。
        let key = match refusal.uid {
            Some(uid) => format!("auth_refused:{uid}"),
            None => "auth_refused:unknown".to_string(),
        };
        log.append(
            EventDraft::new(
                EventKind::new("safety/auth_refused").expect("static kind"),
                "rpc",
                json!({"uid": refusal.uid, "reason": refusal.reason, "refused": self.total}),
            )
            .with_salience(200)
            .with_collapse_key(key),
        );
    }
}

/// 这个对端准不准连进来（`SO_PEERCRED`，见 [`UdsAuth`]）。
///
/// fail-closed 的两条边：`peer_cred()` 失败（对端在这一瞬间就没了，或者这个平台答不上
/// 来）一律拒——"问不出你是谁"不是"你是自己人"；白名单为空一律拒，见
/// [`UdsAuth::peer_uids`]。
fn authorize_peer(stream: &tokio::net::UnixStream, peer_uids: &[u32]) -> Result<(), AuthRefusal> {
    let uid = match stream.peer_cred() {
        Ok(cred) => cred.uid(),
        Err(e) => {
            tracing::warn!(error = %e, "uds peer credentials are unavailable; refusing");
            return Err(AuthRefusal {
                uid: None,
                reason: "peer credentials unavailable",
            });
        }
    };
    if peer_uids.contains(&uid) {
        return Ok(());
    }
    Err(AuthRefusal {
        uid: Some(uid),
        reason: "uid is not in peer_uids",
    })
}

/// UDS 监听：socket 文件已经存在但连不上（陈旧——上一次进程异常退出没清理干净）就删掉
/// 重建；连得上说明另一个进程正在监听，这是配置错误，直接失败（不删活人的 socket）。
/// 存在但根本不是 socket 类型（比如 `uds:` 配置项打错字指到了一个普通文件）：拒绝删除，
/// 直接报错——"陈旧 socket"的判定只该适用于真的曾经是 socket 的文件（Task 3 Fix round
/// 1，L6）。
///
/// `mode` 在**绑完立刻**打上（还没有人 accept 得到东西），见 [`UdsAuth`]。bind 与 chmod
/// 之间那一瞬间的权限由 umask 决定，关不掉——真正不依赖时序的那道关卡是 accept 之后的
/// `SO_PEERCRED` 检查，这一道是给"别的用户连 connect 都不该连上"用的。
fn bind_uds(path: &Path, mode: u32) -> io::Result<tokio::net::UnixListener> {
    if path.exists() {
        let is_socket = std::fs::symlink_metadata(path)
            .map(|m| {
                use std::os::unix::fs::FileTypeExt;
                m.file_type().is_socket()
            })
            .unwrap_or(false);
        if !is_socket {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} exists and is not a unix socket; refusing to delete it — fix the `uds` \
                     path or remove the file yourself",
                    path.display()
                ),
            ));
        }
        match std::os::unix::net::UnixStream::connect(path) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!("uds socket {} is already in use", path.display()),
                ))
            }
            Err(_) => {
                let _ = std::fs::remove_file(path);
            }
        }
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let std_listener = std::os::unix::net::UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    std_listener.set_nonblocking(true)?;
    tokio::net::UnixListener::from_std(std_listener)
}

fn bind_tcp(addr: SocketAddr) -> io::Result<tokio::net::TcpListener> {
    let std_listener = std::net::TcpListener::bind(addr)?;
    std_listener.set_nonblocking(true)?;
    tokio::net::TcpListener::from_std(std_listener)
}

/// 一条 WebSocket 连接：text frame ↔ 一行；`Message::Close`/流结束 → `None`；
/// binary/ping/pong 直接跳过（不是错误，也不是一行，继续等下一帧）；单条组装好的文本消息
/// 超过 [`MAX_LINE`] → 传输错误（关闭连接），跟 UDS/stdio 的 `LineReader` 同一个上限
/// （Task 3 Fix round 1，M3；`spawn_listeners` 里 `accept_async_with_config` 已经把
/// tungstenite 自己的 `max_message_size`/`max_frame_size` 也收紧到这个值，这里是双保险，
/// 不依赖调用方一定用了那个配置）。泛型 `S` 兼顾服务端（`WebSocketStream<TcpStream>`）和
/// 客户端（`WebSocketStream<MaybeTlsStream<TcpStream>>`）两种底层流。
pub struct WsTransport<S>(WebSocketStream<S>);

impl<S> WsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn new(ws: WebSocketStream<S>) -> Self {
        Self(ws)
    }

    /// 服务端：`tokio_tungstenite::accept_async`/`accept_async_with_config` 握手完成后
    /// 的流。
    pub fn server(ws: WebSocketStream<S>) -> Self {
        Self::new(ws)
    }

    /// 客户端：`tokio_tungstenite::connect_async` 握手完成后的流。跟 `server` 行为完全
    /// 一样——两个名字纯粹是调用点的可读性，服务端/客户端拿到的 `S` 通常不同
    /// （`TcpStream` vs `MaybeTlsStream<TcpStream>`）（Task 3 Fix round 1 nit）。
    pub fn client(ws: WebSocketStream<S>) -> Self {
        Self::new(ws)
    }
}

impl<S> Transport for WsTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn into_split(self: Box<Self>) -> (Box<dyn TransportRead>, Box<dyn TransportWrite>) {
        let (sink, stream) = self.0.split();
        (Box::new(WsRead(stream)), Box::new(WsWrite(sink)))
    }
}

struct WsRead<S>(SplitStream<WebSocketStream<S>>);

impl<S> TransportRead for WsRead<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn recv(&mut self) -> TransportFuture<'_, Option<io::Result<String>>> {
        Box::pin(async move {
            loop {
                match self.0.next().await {
                    None => return None,
                    Some(Ok(Message::Text(t))) => {
                        if t.len() > MAX_LINE {
                            return Some(Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("ws message exceeds MAX_LINE ({MAX_LINE} bytes)"),
                            )));
                        }
                        return Some(Ok(t.to_string()));
                    }
                    Some(Ok(Message::Close(_))) => return None,
                    // binary/ping/pong（以及原始 Frame，正常读路径不会出现）：不是一行，
                    // 也不是错误，接着等下一帧。
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => return Some(Err(io::Error::other(e))),
                }
            }
        })
    }
}

struct WsWrite<S>(SplitSink<WebSocketStream<S>, Message>);

impl<S> TransportWrite for WsWrite<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn send(&mut self, line: String) -> TransportFuture<'_, io::Result<()>> {
        Box::pin(async move {
            self.0
                .send(Message::Text(line.into()))
                .await
                .map_err(io::Error::other)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SO_PEERCRED` 的判定，三种情形。用的是一对真的 `UnixStream`（`pair()` 就是一次
    /// `socketpair(AF_UNIX)`，`peer_cred()` 在它上面照样答得上来），所以测的是**这个
    /// 函数本身**，不是一个改写过的判据。
    ///
    /// 第三种（白名单为空）是这里唯一一条不那么显然的：它必须是"谁也不许"，不是"谁都
    /// 行"——一份显式写成 `peer_uids: []` 的配置只有一种读法，而一个把空当成通配的实现
    /// 会在 `current_uid()` 读不出来的那一天把这条边界整个打开。
    #[tokio::test]
    async fn authorize_peer_admits_only_the_whitelisted_uid() {
        let (a, _b) = tokio::net::UnixStream::pair().unwrap();
        let me = a.peer_cred().unwrap().uid();

        assert_eq!(authorize_peer(&a, &[me]), Ok(()));
        assert_eq!(authorize_peer(&a, &[1, me, 2]), Ok(()));

        let missed = authorize_peer(&a, &[me.wrapping_add(1)]).unwrap_err();
        assert_eq!(missed.uid, Some(me));
        assert_eq!(missed.reason, "uid is not in peer_uids");

        let empty = authorize_peer(&a, &[]).unwrap_err();
        assert_eq!(empty.uid, Some(me));
        assert_eq!(empty.reason, "uid is not in peer_uids");
    }

    /// `current_uid()` 和 `peer_cred()` 必须是**同一把尺子**：默认白名单由前者产生，
    /// accept 之后的判定由后者做，两者不一致的后果是"默认配置下没有人连得上自己的
    /// 内核"，而客户端那一侧只看得见 `connection closed`。
    ///
    /// 回归点：从前 `current_uid()` 读 `/proc/self/status`，在 macOS / \*BSD 上一律
    /// `None` → 默认 `peer_uids` 为空 → [`UdsAuth::peer_uids`] 的"空 = 谁也不许"把整
    /// 个平台关在门外。socketpair 的对端就是这个进程自己，所以这条断言在任何 unix 上
    /// 都成立，而且测的是两条真实路径，不是一个改写过的判据。
    #[tokio::test]
    async fn current_uid_is_the_same_uid_peer_cred_reports_for_this_process() {
        let (a, _b) = tokio::net::UnixStream::pair().unwrap();
        assert_eq!(
            current_uid(),
            Some(a.peer_cred().unwrap().uid()),
            "the default whitelist and the accept-time check must agree"
        );
        // 而且默认白名单确实**装得下**这个进程自己：`UdsAuth::default()` 走的就是这条路。
        assert_eq!(authorize_peer(&a, &UdsAuth::default().peer_uids), Ok(()));
    }

    /// 节流：一秒之内的第二次拒绝不再写事件，但计数照加——合并掉的那些不是没了。
    /// 用 `WorldLog::next_seq()` 数写了几条（比翻日志便宜，也不依赖别人往里写什么）。
    #[tokio::test]
    async fn auth_refusals_are_throttled_but_counted() {
        let log = WorldLog::new(Arc::new(oeh_kernel::clock::MonotonicClock::default()));
        let mut book = AuthRefusalLog::default();
        let refusal = AuthRefusal {
            uid: Some(4_294_967_294),
            reason: "uid is not in peer_uids",
        };
        let before = log.next_seq();
        for _ in 0..5 {
            book.record(&log, &refusal);
        }
        assert_eq!(
            log.next_seq() - before,
            1,
            "five refusals in one window, one event"
        );
        assert_eq!(book.total, 5, "the merged ones are still counted");

        // 下一个窗口（把上一次上报的时刻拨回去，而不是真的睡一秒——这里要验的是"合并掉
        // 的那些会在下一条里现身"，不是定时器准不准）。
        book.last_report = Some(Instant::now() - REFUSAL_REPORT_PERIOD * 2);
        book.record(&log, &refusal);
        assert_eq!(log.next_seq() - before, 2);

        let written = log.since(before.saturating_sub(1)).unwrap();
        let refusals: Vec<_> = written
            .iter()
            .filter(|e| e.kind.as_str() == "safety/auth_refused")
            .collect();
        assert_eq!(refusals.len(), 2);
        assert_eq!(refusals[0].payload["uid"], json!(4_294_967_294u64));
        assert_eq!(refusals[0].payload["reason"], "uid is not in peer_uids");
        assert_eq!(refusals[0].payload["refused"], json!(1));
        assert_eq!(refusals[0].salience, 200);
        assert_eq!(
            refusals[0].collapse_key.as_deref(),
            Some("auth_refused:4294967294")
        );
        // 第二条带着累计数：中间那四次没有消失，只是合并了。
        assert_eq!(refusals[1].payload["refused"], json!(6));
    }
}
