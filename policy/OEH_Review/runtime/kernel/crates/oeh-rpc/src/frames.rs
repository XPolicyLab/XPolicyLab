//! 帧通道：内核的 [`FrameStore`] 和 iceoryx2 共享内存之间的那两条搬运线。
//!
//! 帧不走 JSON 那条路，一帧也不走。边界协议只负责**命名**一条通道（`plugin/register`
//! 的 result、`stream/subscribe` 的 result 里那个 `channel`），真正的字节从一块共享内存
//! 里过去，两边谁都不经过内核的连接队列——500 Hz 的 `lowstate` 要是每帧都编一次 JSON
//! 再挤过一条 UDS，控制面（提案、审批、日志）就会被传感器淹掉。
//!
//! 两个方向，两种线程：
//!
//! - **ingress**（[`Frames::start_ingress`]）：一个跨进程插件声明了一条流，内核这边起一个
//!   线程订阅它的通道，收到样本就 `FrameStore::ingest`。从内核往下看，这条流和一个进程内
//!   的相机驱动产出的流没有任何区别（蒸馏器、`robot/health`、`FrameRef` 全都照常）。
//! - **egress**（[`Frames::subscribe`]）：一条**本地**产生的流（`mock-stream`、进程内适配器）
//!   被进程外的人要走：懒起一个线程，`FrameStore::subscribe` → 有界 channel → 发布。懒，
//!   是因为没人看的时候不该有一个每秒 500 次往共享内存里拷贝的线程。
//!
//! **一条流上这两者互斥**，而且这不是一句约定，是 [`Frames::begin_remote`] 这把锁守着的
//! 不变式：远端在发的流永远不会被起 egress。两者同时存在的话，内核发出去的那一份会被自己
//! 的 ingress 再收回来（一个 iceoryx2 订阅者收所有发布者的样本），`ingest` 又喂给 egress
//! ——一个只受 CPU 限制、`frame_seq` 无限上涨的自激回路。标记分两步：`begin_remote` 在
//! `FrameStore::register` **之前**立（于是"正在注册"的那几毫秒里也没有窗口），但它只挡住
//! **新建**；等注册真的成功了，[`RemoteMark::commit`] 才去扫掉这条流上残留的 egress。
//! 反过来写——一上来就扫——意味着一次注册失败（流名撞上一个本地生产者）会顺手拆掉一个毫
//! 不相干的消费者的 egress，而它手里的通道名从此永远安静。
//!
//! 通道名带 run 标签（`oeh/<run_tag>/<stream_id>`，事件通道再加 `/ev`）：iceoryx2 的服务名
//! 是**机器全局**的，同一台机器上并排跑两个内核（一个真跑、一个回放）时，两边的 `lowstate`
//! 不能撞到同一块共享内存上去。没有持久化日志时的退路是 `local-<pid>` 而不是 `local`
//! （`rpc-server` 的 `run_tag` 配置项可以显式指定）——两个都叫 `local` 的内核就是同一条
//! 通道上的两个发布者，而且没有任何地方会报出来。
//!
//! 线程的纪律（`ipc::Service` 的 port 都不是 `Send`）：Node、service、publisher/subscriber、
//! notifier **全部**在用它们的那个线程里创建，线程之间只共享 `Arc<FrameStore>`、
//! `Arc<AtomicBool>` 和 channel 的两端。两条线程都在设置成功之后才回话（`ready`），所以
//! `plugin/register` / `stream/subscribe` 的 result 一发出去，通道就**确实**在那里了——
//! 客户端下一步就开始发/收帧，它没有第二次机会去发现"其实没起来"。
//!
//! ## 为什么 ingress 是轮询，而不是等事件
//!
//! iceoryx2 的事件 Listener 在 `bind()` 它那条 unix datagram socket 时会改**整个进程**的
//! `umask`（`iceoryx2-bb-posix` 的 `unix_datagram_socket.rs`：`umask(!0o600)` → bind →
//! 还原）。那个窗口里进程内**任何别的线程** `mkdir` 出来的目录都会少掉 `x` 位，于是紧接着
//! 在里面开文件就是 `EACCES`——症状落在受害者身上（会话日志、run 目录），原因在这里，中间
//! 没有任何线索。iceoryx2 自己知道这件事，但它只给**自己**的目录补权限。
//!
//! 所以 ingress 不建 Listener：循环轮询 `receive()`，收到就接着收，收空了才睡——[`TICK`]
//! （5 ms）起步，空转时退到 [`IDLE_TICK`]（50 ms），来帧立刻回到 5 ms。代价是一条**刚醒
//! 过来**的安静流最坏 50 ms 的延迟，换掉的是进程里唯一一个 `bind()`。事件服务照样开
//! （`open_or_create`，把 QoS 钉住），只是不建 Listener：开服务只碰共享内存和几个文件，
//! `bind` 只发生在 `listener_builder()` 里。egress 那一侧的 Notifier 走的是 `connect()`，
//! 也不碰 umask，所以照常通知——`<channel>/ev` 这个约定对进程外的订阅者（Python SDK 的
//! `FrameSubscriber`）完全不变。

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bytes::Bytes;
use oeh_kernel::{Disposer, FrameStore, WorldLog};
use oeh_proto::rpc::types::StreamChannel;
use oeh_proto::{EventDraft, EventKind, Frame};
use parking_lot::{Condvar, Mutex};
use serde_json::json;
use tokio::sync::mpsc;

/// 订阅者的缓冲深度。**两侧必须一致**（Python SDK 用同一个值）：iceoryx2 的
/// `open_or_create` 按这些参数开服务，先到的那一方决定了服务的配置。
pub const SUB_BUFFER: usize = 16;
/// 不保留历史：新订阅者拿到的是**当下**，不是一段回放。帧的历史在 `FrameStore` 的环形
/// 缓冲里（`FrameRef` 按 seq 取），不在通道里。
pub const HISTORY: usize = 0;
/// 一条通道上的发布者上限：正常只有一个（要么是远端适配器，要么是内核的 egress），
/// 留 4 是给"适配器重连、旧的还没被清掉"那一小段重叠期的。
pub const MAX_PUBLISHERS: usize = 4;
/// 订阅者上限：内核的 ingress 占一个，其余留给控制台/记录器这类旁观者。
pub const MAX_SUBSCRIBERS: usize = 8;
/// 事件服务的 QoS 也钉死，理由和上面四个一样：不钉的话它们是**建服务的那一方**的默认值
/// （16/16），于是任一侧换一份 `iceoryx2.toml` 就能悄悄改掉另一侧的队列形状。数字按
/// "每个订阅者一个 Listener、每个发布者一个 Notifier"对齐 [`MAX_SUBSCRIBERS`] /
/// [`MAX_PUBLISHERS`]。
pub const MAX_LISTENERS: usize = MAX_SUBSCRIBERS;
/// 见 [`MAX_LISTENERS`]。
pub const MAX_NOTIFIERS: usize = MAX_PUBLISHERS;

/// egress 的有界队列深度。满了就**丢**并计数，绝不阻塞——[`FrameStore::subscribe`] 的回调
/// 跑在产帧路径上（通常是内核任务），在那里等一个慢订阅者就是让整个内核跟着它慢下来。
const EGRESS_QUEUE: usize = 64;

/// ingress 轮询的起始间隔（也是每来一批帧之后回到的那一档）。
const TICK: Duration = Duration::from_millis(5);

/// 空转时退到的最大间隔。stop 标志最坏在这之后加一次在途 `ingest` 才被看见（≤ 60 ms），
/// 而一条安静的流的代价从每秒 200 次唤醒降到 20 次——内核可以有 64 条流（`plugin/register`
/// 的上限），每条一个线程。
const IDLE_TICK: Duration = Duration::from_millis(50);

/// 等一条搬运线程把端口开出来的上限。
///
/// 有上限这件事本身是承重的：`start_ingress` 被 `RemotePlugin::apply` 调用，而 `apply`
/// 跑在**内核任务**上（`Op::AddFiber` 那一条），同一条任务还在排着每个传感器的
/// `Op::IngestFrame`。iceoryx2 开服务最坏要等它自己的 `creation_timeout`（1 s），一份声明
/// 了 64 条流的 manifest 就是一分钟的内核停摆。这里取 2 s——iceoryx2 自己的
/// `creation_timeout` 是 1 s，而两个节点同时建/拆同一域里的服务时，后来的那个会等满
/// 这 1 s 再成功，所以门槛必须留出整整一个 creation_timeout 的余量。曾经是 250 ms，
/// GitHub 的冷 runner 上五条帧测试全在这道门上超时；改成 1 s 后还剩一条正好卡在 1 s
/// 的边上（本机几毫秒就过）。64 条流的最坏情况（128 s）由 `rpc-server.register_timeout_ms`
/// （默认 5 s）整体拒掉，内核任务的停摆则以"最慢的那一条流"为界；把开端口挪下内核
/// 任务是 Plan 5 的遗留项。
const READY_TIMEOUT: Duration = Duration::from_millis(2000);

/// 丢帧的报告节流：每条流每秒最多一条 `health/frame_egress_drop`。
const DROP_REPORT_PERIOD: Duration = Duration::from_secs(1);

/// 等别人把同一条流的 egress 建完的上限（见 [`Frames::ensure_egress`]）。
const EGRESS_WAIT: Duration = Duration::from_secs(2);

/// 一条流的通道名：`oeh/<run_tag>/<stream_id>`。
pub fn channel_name(run_tag: &str, stream_id: &str) -> String {
    format!("oeh/{run_tag}/{stream_id}")
}

/// 一条通道的事件（通知）服务名。发布者每发一个样本 `notify()` 一下，进程外的订阅者就不用
/// 忙等。内核这一侧只**发**通知，不收（见模块头上"为什么 ingress 是轮询"）。
pub fn event_name(channel: &str) -> String {
    format!("{channel}/ev")
}

/// 一条流的 schema 标识：`<schema>@<编码>`。里程碑 1 只有 JSON 一种编码；接上 flatbuffers
/// 之后这里会按 tier 分流。
///
/// 规则：**编码后缀归产帧的那一方所有**，`schema_id` 只在声明里没有后缀时补上里程碑 1 的
/// 默认值。`oeh_proto::payload::LowState::SCHEMA` 本身就是 `"oeh.LowState@json"`，再套一层
/// 就成了 `oeh.LowState@json@json`，而 `plugin/register` 和 `stream/subscribe` 必须对同一条
/// 流给出同一个标识。于是这个函数是**幂等**的，而且一个准备好发 flatbuffers 的适配器今天
/// 就能声明 `oeh.LowState@fb` 并被原样透传。
pub fn schema_id(schema: &str) -> String {
    if schema.contains('@') {
        schema.to_string()
    } else {
        format!("{schema}@json")
    }
}

/// iceoryx2 的错误类型五花八门（每种 port 一个枚举），但对调用方只有一件事要紧：通道没
/// 起来，以及原因的那句话。统一成 `io::Error`。
fn io_err<E: core::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

/// 一条流在 egress 上丢掉的帧：计数 + 限流的健康事件。
///
/// 为什么必须是**事件**而不只是一行日志：`robot/health` 里那个 `frames` 是
/// `FrameStore::stats().frames_total`，也就是**进来**的帧数——egress 丢帧的时候它一动不动。
/// 于是"进程外的消费者跟不上、正在丢帧"这件事，如果只写 stderr，就只有盯着终端的人知道，
/// 而运维看的是世界日志。
struct Drops {
    stream_id: String,
    /// 没有世界日志（测试里直接构造的 [`Frames`]）就只记数和打日志。
    log: Option<Arc<WorldLog>>,
    total: AtomicU64,
    /// 上一条事件的时刻。`Mutex` 而不是原子：这条路径每秒最多做一次真正的工作，而"一秒
    /// 一条"这个判定不该有两个线程同时认为自己赢了。
    last_report: Mutex<Option<Instant>>,
}

impl Drops {
    fn record(&self, reason: &'static str) {
        let total = self.total.fetch_add(1, Ordering::Relaxed) + 1;
        let due = {
            let mut last = self.last_report.lock();
            let now = Instant::now();
            let due = last.is_none_or(|t| now.duration_since(t) >= DROP_REPORT_PERIOD);
            if due {
                *last = Some(now);
            }
            due
        };
        if !due {
            return;
        }
        // 出锁之后才 append：`WorldLog::append` 会同步跑完所有观察者（其中一个会给每条
        // 连接投通知），那不是能带着锁做的事。
        tracing::warn!(stream = %self.stream_id, dropped = total, reason, "frame egress dropped frames");
        if let Some(log) = &self.log {
            log.append(
                EventDraft::new(
                    EventKind::new("health/frame_egress_drop").expect("static kind"),
                    "rpc",
                    json!({"stream_id": self.stream_id, "dropped": total}),
                )
                .with_collapse_key(format!("frame_egress_drop:{}", self.stream_id))
                .with_salience(100),
            );
        }
    }

    fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }
}

/// 一条流的 egress：`FrameStore` 的订阅 + 发布线程。
///
/// 两个 `Mutex<Option<..>>` 是为了让 [`Egress::stop`] 只要 `&self`（它被从一张
/// `HashMap<_, Arc<Egress>>` 里调用）。停的次序是固定的：**先**撤订阅（回调连同它手里的
/// `Sender` 一起消失 → 队列关闭）**再** join，反过来就是等一个永远收不到 `None` 的线程。
struct Egress {
    unsubscribe: Mutex<Option<Disposer>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    drops: Arc<Drops>,
}

impl Egress {
    fn stop(&self) {
        let d = self.unsubscribe.lock().take();
        if let Some(d) = d {
            d();
        }
        let h = self.thread.lock().take();
        if let Some(h) = h {
            let _ = h.join();
        }
        // 收摊的时候把总账报一次：中途那些 `warn!` 说的是"正在丢"，这一条说的是"这条流
        // 这一辈子一共丢了多少"。
        let dropped = self.drops.total();
        if dropped > 0 {
            tracing::info!(stream = %self.drops.stream_id, dropped, "frame egress closed after dropping frames");
        }
    }
}

/// 一条流的 egress 槽位。`Starting(gen)` 是 [`Frames::ensure_egress`] 的"占位"：建端口那
/// 几毫秒**不持锁**（锁纪律），但别的调用方也不该趁这个空当为同一条流再建一个发布者。
///
/// 占位带世代号，因为"这个占位是不是我的"必须是个**能回答**的问题：`shutdown()` 会把整张
/// 表清空，之后另一个调用方完全可能插进一个新的占位——没有世代号的话，先前那个建设者回来
/// 会把别人的占位（甚至别人刚落成的 `Live`）当成自己的摘走。
enum Slot {
    Starting(u64),
    Live(Arc<Egress>),
}

/// 一条流"由远端发布"这件事的两个阶段，各记一笔。
///
/// 两阶段是 R1：`begin`（`FrameStore::register` **之前**）只挡住"新建 egress"，
/// `commit`（注册成功**之后**）才是"这条流归远端了"，才去扫掉残留的 egress。一阶段的话，
/// 一个注册失败的 `plugin/register`（比如流名和本地生产者撞了）会在失败的路上把一个毫不
/// 相干的消费者的 egress 拆掉，而那个消费者手里的通道名从此永远安静。
///
/// 计数而不是布尔：同一条流上完全可以有两份凭据（第二个插件正在注册同名的流，注册马上会
/// 失败，但它的 `begin` 和第一个的 `commit` 在时间上重叠），而一份凭据被丢掉不该把另一份
/// 的标记也带走。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Marks {
    pending: u32,
    committed: u32,
}

impl Marks {
    /// 有任何一份凭据 = 不许为这条流新建 egress。
    fn any(&self) -> bool {
        self.pending + self.committed > 0
    }
}

#[derive(Default)]
struct Inner {
    egress: HashMap<String, Slot>,
    /// 由某个跨进程插件**发布**（或者正在声明要发布）的流。`stream/subscribe` 靠它分辨
    /// "这条流已经有人在往通道上发了（直接把通道名给你）"和"这条流只活在内核里（得起一条
    /// egress）"，而 `ensure_egress` 把它当成硬约束（见模块头上的自激回路）。
    remote: HashMap<String, Marks>,
    next_gen: u64,
}

impl Inner {
    fn is_marked(&self, stream_id: &str) -> bool {
        self.remote.get(stream_id).is_some_and(Marks::any)
    }

    /// 摘掉一条流已经建好的 egress（`Starting` 的摘不掉——建它的那个调用方回来发现流已经
    /// 变成远端发布的，会自己把它丢掉）。
    fn take_egress(&mut self, stream_id: &str) -> Option<Arc<Egress>> {
        match self.egress.get(stream_id) {
            Some(Slot::Live(_)) => match self.egress.remove(stream_id) {
                Some(Slot::Live(e)) => Some(e),
                _ => None,
            },
            _ => None,
        }
    }

    /// 撤掉**自己**那个占位（按世代号认领）。
    fn clear_starting(&mut self, stream_id: &str, generation: u64) {
        if matches!(self.egress.get(stream_id), Some(Slot::Starting(g)) if *g == generation) {
            self.egress.remove(stream_id);
        }
    }

    fn owns_starting(&self, stream_id: &str, generation: u64) -> bool {
        matches!(self.egress.get(stream_id), Some(Slot::Starting(g)) if *g == generation)
    }
}

/// 一条流"由远端发布"这件事的**凭据**，两个阶段（见 [`Marks`]）。
///
/// 是一个 guard 而不是几个函数，因为这个标记必须比 `FrameStore::register` 先立、比它后撤，
/// 而"先立后撤"写成两次调用，就是迟早有人会漏掉其中一次的那种代码。`RemotePlugin::apply`
/// 把一份克隆交给 `ctx.on_dispose`，于是它跟着 Fiber 走——包括 `apply` 自己中途失败、效果
/// 链被当场回滚的那条路，以及注册压根没成功、凭据随着栈一起消失的那条路。
///
/// `Clone` 是廉价的（一个 `Arc`）：标记在**最后**一份克隆消失时才交还。
#[derive(Clone)]
pub struct RemoteMark {
    state: Arc<MarkState>,
}

struct MarkState {
    inner: Arc<Mutex<Inner>>,
    stream_id: String,
    committed: AtomicBool,
}

impl RemoteMark {
    /// 注册成功了：把这份凭据从"正在声明"提升为"这条流归远端"，并扫掉这条流上残留的
    /// egress（F-2 的 path B：上一个本地生产者没了、它的 egress 还挂着，现在这条通道要
    /// 归远端发布者了）。幂等。
    pub fn commit(&self) {
        if self.state.committed.swap(true, Ordering::Relaxed) {
            return;
        }
        let egress = {
            let mut g = self.state.inner.lock();
            let m = g.remote.entry(self.state.stream_id.clone()).or_default();
            m.pending = m.pending.saturating_sub(1);
            m.committed += 1;
            g.take_egress(&self.state.stream_id)
        };
        if let Some(e) = egress {
            e.stop();
        }
    }
}

impl Drop for MarkState {
    /// 交还这一份凭据。**不**动 egress：标记还在的时候没人能新建一条（`ensure_egress` 在
    /// 锁里挡着），所以这里不可能有需要扫的东西；而"顺手扫一下"正是 R1 那个 bug 的形状。
    fn drop(&mut self) {
        let mut g = self.inner.lock();
        let committed = self.committed.load(Ordering::Relaxed);
        if let Some(m) = g.remote.get_mut(&self.stream_id) {
            if committed {
                m.committed = m.committed.saturating_sub(1);
            } else {
                m.pending = m.pending.saturating_sub(1);
            }
            if !m.any() {
                g.remote.remove(&self.stream_id);
            }
        }
    }
}

/// 这个内核的帧通道。`rpc-server` 插件建一个，[`crate::remote::RemotePlugin`] 和
/// `stream/subscribe` 共用它。
pub struct Frames {
    run_tag: String,
    /// **一帧**的字节上限（`rpc-server` 的 `frames.max_bytes`）：egress 的 publisher 按它
    /// 预留共享内存，而超过它的帧在进队列之前就被丢掉并计数（见 [`Frames::ensure_egress`]
    /// 的回调）。不这么做的话它只是个"初始预留"——`AllocationStrategy::PowerOfTwo` 会为一
    /// 帧 50 MiB 的东西把共享内存段翻倍，没有任何上界。
    max_bytes: usize,
    frames: Arc<FrameStore>,
    log: Option<Arc<WorldLog>>,
    inner: Arc<Mutex<Inner>>,
    /// 一个 `Starting` 槽位变成 `Live`（或者消失）时叫醒等在同一条流上的调用方。
    started: Condvar,
}

impl Drop for Frames {
    /// 兜底：`rpc-server` 正常撤销时走的是 [`Frames::shutdown`]（挂在 `ctx.on_dispose`
    /// 上），这里是为了一个没走插件路径构造出来的 `Frames`（测试）也不留下幽灵线程。
    /// ingress 线程不在这里——它们各自挂在注册它们的那个 Fiber 的 Effect 链上。
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Frames {
    /// 不带世界日志的构造（测试、嵌入用）：丢帧只进日志，不进 `health/frame_egress_drop`。
    pub fn new(run_tag: String, max_bytes: usize, frames: Arc<FrameStore>) -> Arc<Self> {
        Self::build(run_tag, max_bytes, frames, None)
    }

    /// `rpc-server` 用的构造：丢帧会变成世界日志里的 `health/frame_egress_drop` 事件。
    pub fn new_with_log(
        run_tag: String,
        max_bytes: usize,
        frames: Arc<FrameStore>,
        log: Arc<WorldLog>,
    ) -> Arc<Self> {
        Self::build(run_tag, max_bytes, frames, Some(log))
    }

    fn build(
        run_tag: String,
        max_bytes: usize,
        frames: Arc<FrameStore>,
        log: Option<Arc<WorldLog>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            run_tag,
            max_bytes,
            frames,
            log,
            inner: Arc::new(Mutex::new(Inner::default())),
            started: Condvar::new(),
        })
    }

    /// 这个内核里一条流的通道名。
    pub fn channel_of(&self, stream_id: &str) -> String {
        channel_name(&self.run_tag, stream_id)
    }

    /// 这条流是不是某个跨进程插件在往通道上发（或者正在声明要发）。
    ///
    /// 两个阶段都算"是"：`stream/subscribe` 在那几毫秒里拿到的是通道名而不是一条新起的
    /// egress，而这正是要的——注册要是成功了，起了 egress 就是一个自激回路；要是失败了，
    /// 这条流原有的 egress（如果有）一直在正常工作，从没被碰过。
    pub fn is_remote(&self, stream_id: &str) -> bool {
        self.inner.lock().is_marked(stream_id)
    }

    /// 这条流现在有没有一条活着的 egress。（诊断 + 测试用；`Starting` 不算。）
    pub fn has_egress(&self, stream_id: &str) -> bool {
        matches!(self.inner.lock().egress.get(stream_id), Some(Slot::Live(_)))
    }

    /// 这条流在 egress 上一共丢了多少帧（队列满 + 超过 `max_bytes` + 借不到样本）。
    pub fn egress_dropped(&self, stream_id: &str) -> u64 {
        match self.inner.lock().egress.get(stream_id) {
            Some(Slot::Live(e)) => e.drops.total(),
            _ => 0,
        }
    }

    /// 声明"这条流即将由远端发布"，返回持有这个标记的凭据（见 [`RemoteMark`]）。
    ///
    /// 必须在 `FrameStore::register` **之前**调用：`stream/subscribe` 只要看见流存在就会
    /// 考虑给它起 egress，而"流已经注册、标记还没打上"这几毫秒正是自激回路的入口。
    ///
    /// 这一步**只挡住新建**，什么都不拆：注册可能失败（流名和一个本地生产者撞了），而那
    /// 条路上这条流的 egress 一直是别人的、一直在正常工作。拆东西的是
    /// [`RemoteMark::commit`]，它只在注册**成功**之后被调用。
    pub fn begin_remote(&self, stream_id: &str) -> RemoteMark {
        let mut g = self.inner.lock();
        g.remote.entry(stream_id.to_string()).or_default().pending += 1;
        drop(g);
        RemoteMark {
            state: Arc::new(MarkState {
                inner: self.inner.clone(),
                stream_id: stream_id.to_string(),
                committed: AtomicBool::new(false),
            }),
        }
    }

    /// 起一条流的 ingress：一个线程订阅 `oeh/<run_tag>/<stream_id>`，收到的每个样本都
    /// `FrameStore::ingest` 进来。
    ///
    /// **同步地**等到端口开出来才返回（上限 [`READY_TIMEOUT`]，见那里的说明）：调用它的是
    /// `RemotePlugin::apply`，而 `apply` 一返回，`plugin/register` 就把通道名回给客户端了
    /// ——客户端下一句就是 publish。晚一步起订阅者，丢的是真实传感器的前几帧。
    ///
    /// 超时的话线程不 join（它可能还卡在 iceoryx2 的建服务里，join 就是把上限又变回无限）：
    /// 置 stop 标志、让它自己收摊，这次注册报错。
    ///
    /// 返回的 `Disposer` 置 stop 标志并 join（线程在 [`TICK`]–[`IDLE_TICK`] 之间醒一次，
    /// 所以最坏 ≤ 60 ms），由 `ctx.on_dispose` 挂到那个 Fiber 上：插件撤销 = 流从
    /// `stream/list` 消失 = ingress 线程结束，一件事的三种说法。
    pub fn start_ingress(&self, stream_id: &str) -> io::Result<Disposer> {
        let channel = self.channel_of(stream_id);
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<io::Result<()>>();
        let frames = self.frames.clone();
        let id = stream_id.to_string();
        let stop_thread = stop.clone();
        let handle = std::thread::Builder::new()
            .name(format!("oeh-ingress-{stream_id}"))
            .spawn(move || ingress(channel, id, frames, stop_thread, ready_tx))?;
        match wait_ready(&ready_rx, "frame ingress") {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                stop.store(true, Ordering::Relaxed);
                return Err(e);
            }
            Err(e) => {
                // 设置失败的线程已经自己退了；join 只是不留一个未回收的线程。
                let _ = handle.join();
                return Err(e);
            }
        }

        Ok(Box::new(move || {
            stop.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }))
    }

    /// 把一条流接到通道上，返回客户端该去读的 `{channel, schema_id}`。
    ///
    /// `remote_published`：这条流已经有一个跨进程的发布者（见 [`Frames::is_remote`]）→
    /// 直接把通道名给出去，多一个订阅者而已（同一个 iceoryx2 服务，[`MAX_SUBSCRIBERS`] 个
    /// 位置）。这个参数是**提示**，不是许可：真正说了算的是 [`Frames::ensure_egress`] 在锁
    /// 里重查的那一次——调用方读到 `false` 和它走到这里之间，一个远端插件完全可能刚好把流
    /// 注册上。
    ///
    /// 否则这条流只活在内核里，得起一条 egress 把它发出去——**懒**起，并且每条流只起一次：
    /// 第二个订阅者复用第一个订阅者起的那条线程。
    ///
    /// 会阻塞（建 Node、开共享内存段、起线程），所以调用方要么在 `spawn_blocking` 里，
    /// 要么在一条本来就同步的路径上。
    pub fn subscribe(&self, stream_id: &str, remote_published: bool) -> io::Result<StreamChannel> {
        let schema = self.frames.schema_of(stream_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("unknown stream `{stream_id}`"),
            )
        })?;
        let channel = self.channel_of(stream_id);
        if !remote_published {
            self.ensure_egress(stream_id, &channel)?;
        }
        Ok(StreamChannel {
            stream_id: stream_id.to_string(),
            channel,
            schema_id: schema_id(&schema),
        })
    }

    /// 懒起一条 egress（已经有了、或者这条流是远端发布的，就什么都不做）。
    ///
    /// 两段式，因为**锁不能跨过阻塞的 iceoryx2 调用**：锁里放一个 `Starting` 占位并出锁 →
    /// 锁外建端口、等就绪 → 回到锁里落成 `Live`。同一条流的第二个调用方在 condvar 上等那个
    /// 占位有个结果（[`EGRESS_WAIT`] 为上限，免得一个死掉的建设者把所有人吊死），**别的**
    /// 流的调用方一秒都不用等。
    ///
    /// 回到锁里那一次还要重查 `remote`：建的这几毫秒里一个跨进程插件可能刚好把这条流注册
    /// 走了，那就把刚建好的 egress 丢掉——宁可白建一次，也不能让两者共存。
    fn ensure_egress(&self, stream_id: &str, channel: &str) -> io::Result<()> {
        let deadline = Instant::now() + EGRESS_WAIT;
        loop {
            let generation = {
                let mut g = self.inner.lock();
                if g.is_marked(stream_id) {
                    return Ok(());
                }
                match g.egress.get(stream_id) {
                    Some(Slot::Live(_)) => return Ok(()),
                    Some(Slot::Starting(_)) => {
                        if Instant::now() >= deadline {
                            return Err(io::Error::new(
                                io::ErrorKind::TimedOut,
                                format!("another subscriber is still opening `{stream_id}`"),
                            ));
                        }
                        self.started.wait_for(&mut g, TICK);
                        continue;
                    }
                    None => {
                        let generation = g.next_gen;
                        g.next_gen += 1;
                        g.egress
                            .insert(stream_id.to_string(), Slot::Starting(generation));
                        generation
                    }
                }
            };
            // 锁外：建端口、起线程、等就绪、挂上 `FrameStore` 的订阅。
            let built = self.build_egress(stream_id, channel);
            let mut g = self.inner.lock();
            let result = match built {
                // 建好了，但这几毫秒里这条流成了远端发布的：把刚建好的东西丢掉（白建一次，
                // 好过一个自激回路），而调用方拿到的那个通道名是对的——远端在往上面发。
                Ok(e) if g.is_marked(stream_id) => {
                    g.clear_starting(stream_id, generation);
                    drop(g);
                    e.stop();
                    return finish(&self.started, Ok(()));
                }
                // 占位不见了（`shutdown()` 把表清了，`rpc-server` 正在撤）：同样丢掉，但
                // **报错**——回一个 `Ok` 就是告诉调用方"订阅好了"，而它拿到的是一条没有
                // 发布者的通道，它没有任何理由再来第二次。
                Ok(e) if !g.owns_starting(stream_id, generation) => {
                    drop(g);
                    e.stop();
                    return finish(
                        &self.started,
                        Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            format!("frame egress for `{stream_id}` was cancelled while opening"),
                        )),
                    );
                }
                Ok(e) => {
                    g.egress.insert(stream_id.to_string(), Slot::Live(e));
                    Ok(())
                }
                Err(e) => {
                    g.clear_starting(stream_id, generation);
                    Err(e)
                }
            };
            drop(g);
            return finish(&self.started, result);
        }
    }

    /// 建一条 egress：发布线程 + `FrameStore` 的订阅。不碰 `inner`（调用方在锁外调它）。
    fn build_egress(&self, stream_id: &str, channel: &str) -> io::Result<Arc<Egress>> {
        let (tx, rx) = mpsc::channel::<Bytes>(EGRESS_QUEUE);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<io::Result<()>>();
        let drops = Arc::new(Drops {
            stream_id: stream_id.to_string(),
            log: self.log.clone(),
            total: AtomicU64::new(0),
            last_report: Mutex::new(None),
        });
        let ch = channel.to_string();
        let max_bytes = self.max_bytes;
        let thread_drops = drops.clone();
        let handle = std::thread::Builder::new()
            .name(format!("oeh-egress-{stream_id}"))
            .spawn(move || egress(ch, max_bytes, rx, ready_tx, thread_drops))?;
        if let Err(e) = wait_ready(&ready_rx, "frame egress") {
            // 超时的那条也不用 join：`tx` 在这里被丢掉，线程一旦建完端口就会从
            // `blocking_recv()` 拿到 `None` 自己收摊。
            if e.kind() != io::ErrorKind::TimedOut {
                let _ = handle.join();
            }
            return Err(e);
        }

        let counter = drops.clone();
        // 回调跑在 ingest 线程上、同步：只 `try_send`，满了就丢。契约见
        // `FrameStore::subscribe`。
        let unsubscribe = self.frames.subscribe(
            stream_id,
            Arc::new(move |f: &Frame| {
                // `max_bytes` 是**一帧**的上限，在这里就拦住：让它进队列只是把同一个决定
                // 推迟到发布线程，代价是一次多余的拷贝和一个会自己长大的共享内存段。
                if f.payload.len() > max_bytes {
                    counter.record("oversize");
                    return;
                }
                // 队列满 = 发布线程跟不上，这是要记的账。队列**关了**（线程已经收摊、撤订阅
                // 和这一帧擦肩而过）不是：那是正常收尾，记进"丢帧"里只会让运维在每次关机
                // 日志上看见一条假警报。
                if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(f.payload.clone()) {
                    counter.record("queue full");
                }
            }),
        );
        // 订阅失败的话那条线程没人喂：队列的发送端随着上面那个闭包一起被丢掉，线程自己
        // 就收摊了，join 一下不留残骸。
        match unsubscribe {
            Ok(d) => Ok(Arc::new(Egress {
                unsubscribe: Mutex::new(Some(d)),
                thread: Mutex::new(Some(handle)),
                drops,
            })),
            Err(e) => {
                let _ = handle.join();
                Err(io_err(e))
            }
        }
    }

    /// 停掉所有 egress 线程。挂在 `rpc-server` 的 `ctx.on_dispose` 上（幂等）。
    ///
    /// 不能只靠 `Drop`：`Frames` 是被 `Arc<Services>` 攥着的，而每条活着的连接、每个在飞的
    /// 处理函数都各攥一份——一条迟迟不断的连接会让发布线程比"拥有"它们的那个 Fiber 活得久，
    /// 也就是撤销了 `rpc-server` 之后帧还在往共享内存里发。
    pub fn shutdown(&self) {
        let slots: Vec<Slot> = {
            let mut g = self.inner.lock();
            g.egress.drain().map(|(_, s)| s).collect()
        };
        for s in slots {
            if let Slot::Live(e) = s {
                e.stop();
            }
        }
        // 正在建的那些（`Starting`）没了槽位：建完之后的那次重查会发现槽位不是自己的，把
        // 刚建好的东西丢掉，并且**报错**而不是假装订阅成功了。
        self.started.notify_all();
    }
}

/// `ensure_egress` 的收尾：叫醒等在同一条流上的调用方，然后把结果交出去。
fn finish(started: &Condvar, r: io::Result<()>) -> io::Result<()> {
    started.notify_all();
    r
}

/// 等一条搬运线程把它的端口开出来。
///
/// - `Err(RecvTimeout::Timeout)` → `TimedOut`：调用方据此决定**不** join（线程可能还卡在
///   iceoryx2 里）。
/// - `Err(Disconnected)` = 线程连一句话都没说就没了（panic，`Sender` 随栈一起丢掉）：报成
///   一个明确的错误，而不是一直等下去。
fn wait_ready(rx: &std::sync::mpsc::Receiver<io::Result<()>>, what: &str) -> io::Result<()> {
    use std::sync::mpsc::RecvTimeoutError;
    match rx.recv_timeout(READY_TIMEOUT) {
        Ok(r) => r,
        Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("{what} did not open its ports within {READY_TIMEOUT:?}"),
        )),
        Err(RecvTimeoutError::Disconnected) => {
            Err(io::Error::other(format!("{what} died during setup")))
        }
    }
}

/// ingress 线程体：通道 → `FrameStore`。
fn ingress(
    channel: String,
    stream_id: String,
    frames: Arc<FrameStore>,
    stop: Arc<AtomicBool>,
    ready: std::sync::mpsc::Sender<io::Result<()>>,
) {
    let announce = ready.clone();
    let run = move || -> io::Result<()> {
        use iceoryx2::prelude::*;
        let node = NodeBuilder::new()
            .create::<ipc::Service>()
            .map_err(io_err)?;
        let service = node
            .service_builder(&ServiceName::new(&channel).map_err(io_err)?)
            .publish_subscribe::<[u8]>()
            .subscriber_max_buffer_size(SUB_BUFFER)
            .history_size(HISTORY)
            .max_publishers(MAX_PUBLISHERS)
            .max_subscribers(MAX_SUBSCRIBERS)
            .open_or_create()
            .map_err(io_err)?;
        // 事件服务**只开不用**：这一侧一个 Listener 都不建（那是进程里唯一会动 umask 的
        // 东西，见模块头），但把 QoS 钉在这里——一条远端发布的通道，事件服务的创建者本来
        // 会是对面的 Python 发布者，它按库的默认值（16/16）建；开一次就轮到内核这边说话，
        // 两侧于是对同一条通道给出同一份形状。`.event().open_or_create()` 只碰静态/动态
        // 配置存储（共享内存和文件），不碰任何 socket——bind 只发生在 `listener_builder()`
        // 里，而这里没有。
        let _ev = node
            .service_builder(&ServiceName::new(&event_name(&channel)).map_err(io_err)?)
            .event()
            .max_listeners(MAX_LISTENERS)
            .max_notifiers(MAX_NOTIFIERS)
            .open_or_create()
            .map_err(io_err)?;
        let subscriber = service.subscriber_builder().create().map_err(io_err)?;
        // 端口都开好了才回话。
        let _ = announce.send(Ok(()));
        // 没有 Listener（见模块头）：轮询，收到就接着收，收空了才睡。空转的周期从 `TICK`
        // 指数退到 `IDLE_TICK`，来帧就立刻回到 `TICK`——一条 500 Hz 的流永远在 5 ms 那一档，
        // 而一条安静的流（或者 64 条安静的流）不会每秒醒 200 次。stop 标志因此最坏
        // `IDLE_TICK`（50 ms）之后被看见，加上在途的一次 `ingest`。
        let mut last_err: Option<Instant> = None;
        let mut idle = TICK;
        while !stop.load(Ordering::Relaxed) {
            match subscriber.receive() {
                // 流没注册（Fiber 正在撤销，ingest 比 Effect 链慢了一步）是正常收尾，
                // 不是错误。
                Ok(Some(sample)) => {
                    idle = TICK;
                    let _ =
                        frames.ingest(&stream_id, Bytes::copy_from_slice(sample.payload()), None);
                }
                Ok(None) => {
                    std::thread::sleep(idle);
                    idle = (idle * 2).min(IDLE_TICK);
                }
                // 一个卡在错误状态里的订阅者会一直空转。限流地喊出来——不然这条流在
                // `stream/list` 里好端端地挂着，却再也不会有帧，而日志里一个字都没有。
                Err(e) => {
                    let now = Instant::now();
                    if last_err.is_none_or(|t| now.duration_since(t) >= DROP_REPORT_PERIOD) {
                        last_err = Some(now);
                        tracing::warn!(stream = %stream_id, error = %e, "frame ingress receive failed");
                    }
                    std::thread::sleep(TICK);
                }
            }
        }
        Ok(())
    };
    if let Err(e) = run() {
        let _ = ready.send(Err(e));
    }
}

/// egress 线程体：有界队列 → 通道。
fn egress(
    channel: String,
    max_bytes: usize,
    mut rx: mpsc::Receiver<Bytes>,
    ready: std::sync::mpsc::Sender<io::Result<()>>,
    drops: Arc<Drops>,
) {
    let announce = ready.clone();
    let mut run = move || -> io::Result<()> {
        use iceoryx2::prelude::*;
        let node = NodeBuilder::new()
            .create::<ipc::Service>()
            .map_err(io_err)?;
        let service = node
            .service_builder(&ServiceName::new(&channel).map_err(io_err)?)
            .publish_subscribe::<[u8]>()
            .subscriber_max_buffer_size(SUB_BUFFER)
            .history_size(HISTORY)
            .max_publishers(MAX_PUBLISHERS)
            .max_subscribers(MAX_SUBSCRIBERS)
            .open_or_create()
            .map_err(io_err)?;
        let ev = node
            .service_builder(&ServiceName::new(&event_name(&channel)).map_err(io_err)?)
            .event()
            .max_listeners(MAX_LISTENERS)
            .max_notifiers(MAX_NOTIFIERS)
            .open_or_create()
            .map_err(io_err)?;
        let publisher = service
            .publisher_builder()
            .initial_max_slice_len(max_bytes)
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .create()
            .map_err(io_err)?;
        // Notifier 是 `connect()`，不 `bind()`，所以它不碰进程的 umask（模块头上那条
        // 说明的另一半）。
        let notifier = ev.notifier_builder().create().map_err(io_err)?;
        let _ = announce.send(Ok(()));
        // `blocking_recv` 在这里是对的（这是一条 `std::thread`，不是 runtime 的 worker），
        // 也是必须的（iceoryx2 的 port 不是 `Send`，它们只能待在这个线程里）。队列的发送端
        // 是 `FrameStore` 的那个订阅回调，撤订阅 → `None` → 这个循环结束。
        while let Some(bytes) = rx.blocking_recv() {
            match publisher.loan_slice_uninit(bytes.len()) {
                Ok(sample) => {
                    let _ = sample.write_from_slice(&bytes).send();
                    let _ = notifier.notify();
                }
                // 借不到样本（段扩不动、借出的样本到上限）：丢掉这一帧，记同一笔账。帧的
                // 历史在 `FrameStore` 里，通道只负责"现在"。
                Err(e) => {
                    tracing::debug!(channel = %channel, error = %e, "frame egress loan failed");
                    drops.record("loan failed");
                }
            }
        }
        Ok(())
    };
    if let Err(e) = run() {
        let _ = ready.send(Err(e));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_carry_the_run_tag_and_the_encoding() {
        assert_eq!(channel_name("run-7", "lowstate"), "oeh/run-7/lowstate");
        assert_eq!(channel_name("local", "arm_state"), "oeh/local/arm_state");
        assert_eq!(event_name("oeh/local/arm_state"), "oeh/local/arm_state/ev");
        assert_eq!(schema_id("oeh.ArmState"), "oeh.ArmState@json");
    }

    /// 一个已经带编码后缀的 schema（`LowState::SCHEMA` 就是这样）再过一次这个函数必须
    /// 原样不动：两边对同一条流给出两个不同的标识，下游就得靠猜来解码。
    #[test]
    fn schema_id_is_idempotent() {
        assert_eq!(schema_id("oeh.LowState@json"), "oeh.LowState@json");
        assert_eq!(schema_id("oeh.LowState@fb"), "oeh.LowState@fb");
        assert_eq!(schema_id(&schema_id("oeh.ArmState")), "oeh.ArmState@json");
    }
}
