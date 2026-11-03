use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oeh_proto::{EventDraft, EventKind, ServiceName};
use tokio::sync::mpsc;

use crate::factories::find_factory;
use crate::fiber::{FiberState, FiberTree, SettleReport};
use crate::frame_store::FrameStore;
use crate::op::{FiberAdded, KernelHandle, Op};
use crate::plugin::Plugin;
use crate::profile::{EntryRow, ProfileError};
use crate::services::{FiberId, ServiceRegistry};
use crate::world_log::WorldLog;

#[derive(Debug, thiserror::Error)]
pub enum BootError {
    #[error("unknown plugin id `{0}`")]
    UnknownPlugin(String),
    #[error(transparent)]
    Profile(#[from] ProfileError),
}

// `ServiceName::from` allocates a `String`; these two names are looked up on every single Op
// (`run` resolves services per-Op, not once at boot), so intern them once instead of allocating
// on every iteration of the hot loop.
static WORLD_LOG: std::sync::OnceLock<ServiceName> = std::sync::OnceLock::new();
static FRAME_STORE: std::sync::OnceLock<ServiceName> = std::sync::OnceLock::new();
fn world_log_name() -> &'static ServiceName {
    WORLD_LOG.get_or_init(|| ServiceName::from("world-log"))
}
fn frame_store_name() -> &'static ServiceName {
    FRAME_STORE.get_or_init(|| ServiceName::from("frame-store"))
}

/// 内核的运行参数。`boot`/`boot_with` 用 [`KernelConfig::default`]；要改的用
/// [`Kernel::boot_with_config`]。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelConfig {
    /// Op 通道的深度。满了之后 [`KernelHandle::send`] 报 `Busy`，而不是让队列无界增长
    /// （见 [`KernelHandle`]）。
    pub op_queue: usize,
    /// 每隔这么久把"队列满而被丢弃的 Op 数"汇总成一条 `health/op_backpressure`
    /// （只在 > 0 时记）。做成参数是为了测试能把这个周期调短——否则一条关于背压的
    /// 测试就得真的等上一秒。
    pub backpressure_report_ms: u64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            op_queue: 4096,
            backpressure_report_ms: 1000,
        }
    }
}

pub struct Kernel {
    tree: FiberTree,
    registry: Arc<ServiceRegistry>,
    report: SettleReport,
    tx: mpsc::Sender<Op>,
    rx: mpsc::Receiver<Op>,
    /// `KernelHandle` 共享的那一份丢弃计数；`run` 定期取走并归零。
    dropped: Arc<AtomicU64>,
    cfg: KernelConfig,
    /// 上一次广播 `plugin/state` 时每个 Fiber 的状态；只有变化的才再发一遍。
    emitted: HashMap<FiberId, FiberState>,
}

impl std::fmt::Debug for Kernel {
    /// 手写而非 derive：`FiberTree` / `ServiceRegistry` 内部持有 `Box<dyn Plugin>` /
    /// `Arc<dyn Any>`，不 impl `Debug`。测试里的 `unwrap_err()` 需要 `Kernel: Debug`。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("states", &self.tree.states())
            .finish_non_exhaustive()
    }
}

/// 把"队列满而被拒的投递次数"记成一条 `health/op_backpressure`（只在 > 0 时），然后归零。
///
/// `dropped` 是**被拒次数**，不是丢失的 Op 数：会重试的调用方（`oeh-rpc` 的 3 × 10 ms）
/// 每被拒一次就计一次，而重试成功的那条 Op 一点没丢。它读作"这个周期内核让多少次投递
/// 吃了闭门羹"——一个背压强度指标。
///
/// 每条被拒的投递各记一条事件是不行的：发生背压的那一刻正是世界日志最不需要每秒几千条
/// 新事件的时候。汇总一条，`dropped` 就是这个周期里的次数。
///
/// 还没有 world-log 服务时**不归零**：那不是"没有背压"，是"还没有地方记"，计数留到
/// 下一个周期。
fn report_backpressure(registry: &Arc<ServiceRegistry>, dropped: &AtomicU64) {
    if dropped.load(Ordering::Relaxed) == 0 {
        return;
    }
    let Ok(log) = registry.get::<WorldLog>(world_log_name()) else {
        return;
    };
    let n = dropped.swap(0, Ordering::Relaxed);
    if n == 0 {
        return;
    }
    tracing::warn!(dropped = n, "kernel op queue overflowed");
    log.append(
        EventDraft::new(
            EventKind::new("health/op_backpressure").expect("static"),
            "kernel",
            serde_json::json!({"dropped": n}),
        )
        .with_salience(200),
    );
}

impl Kernel {
    /// 用编译期注册的工厂解析行。必须在 tokio runtime 内调用。
    pub fn boot(rows: Vec<EntryRow>) -> Result<Kernel, BootError> {
        Self::boot_with(rows, &|id| find_factory(id).map(|f| (f.build)()))
    }

    pub fn boot_with(
        rows: Vec<EntryRow>,
        resolver: &dyn Fn(&str) -> Option<Box<dyn Plugin>>,
    ) -> Result<Kernel, BootError> {
        Self::boot_with_config(rows, resolver, KernelConfig::default())
    }

    /// [`Kernel::boot_with`] + 显式的 [`KernelConfig`]。
    pub fn boot_with_config(
        rows: Vec<EntryRow>,
        resolver: &dyn Fn(&str) -> Option<Box<dyn Plugin>>,
        cfg: KernelConfig,
    ) -> Result<Kernel, BootError> {
        let registry = Arc::new(ServiceRegistry::default());
        let mut tree = FiberTree::new(registry.clone());
        // 在加载任何行之前把 `kernel` 服务挂上：跨进程插件（或任何解析行的过程）都能立刻
        // 拿到一个 KernelHandle。`FiberId(u32::MAX)` 是哨兵——这个"provider"不对应树里的
        // 任何一个真实 Fiber，谁都不能把它 dispose 掉。
        let (tx, rx) = mpsc::channel(cfg.op_queue.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        registry
            .provide(
                ServiceName::from("kernel"),
                FiberId(u32::MAX),
                Arc::new(KernelHandle::new(tx.clone(), dropped.clone())),
            )
            .expect("fresh registry");
        for row in rows {
            if row.disabled {
                continue;
            }
            let plugin =
                resolver(&row.id).ok_or_else(|| BootError::UnknownPlugin(row.id.clone()))?;
            tree.add(row, plugin);
        }
        let report = tree.settle();
        for (k, e) in &report.failed {
            tracing::error!(plugin = %k, error = %e, "plugin failed to activate");
        }
        for (k, missing) in &report.pending {
            tracing::warn!(plugin = %k, ?missing, "plugin pending: services unavailable");
        }
        Ok(Kernel {
            tree,
            registry,
            report,
            tx,
            rx,
            dropped,
            cfg,
            emitted: HashMap::new(),
        })
    }

    pub fn report(&self) -> &SettleReport {
        &self.report
    }
    pub fn states(&self) -> Vec<(String, FiberState)> {
        self.tree.states()
    }
    pub fn registry(&self) -> &Arc<ServiceRegistry> {
        &self.registry
    }
    pub fn handle(&self) -> KernelHandle {
        KernelHandle::new(self.tx.clone(), self.dropped.clone())
    }
    pub fn config(&self) -> &KernelConfig {
        &self.cfg
    }

    /// 把每个 Fiber 相对上一次广播变化了的状态记成一条 `plugin/state`（source `"kernel"`，
    /// salience 60）。没有 world-log 服务时静默跳过——boot 早期或它自己还没起来的窗口。
    ///
    /// `payload.plugin` 不是唯一键——`FiberTree::add` 允许同名多行（比如同一个 key 被
    /// dispose 后重新 add），所以两个 Fiber 完全可以共享同一个 `plugin` 字符串，产生两条
    /// 交错的 `plugin/state` 流。消费者必须按 `payload.fiber_id`（`FiberId`，稳定且每个
    /// Fiber 唯一）分流，把 `plugin` 只当展示用的名字。
    fn publish_states(&mut self) {
        let Ok(log) = self.registry.get::<WorldLog>(world_log_name()) else {
            return;
        };
        for f in self.tree.states_detailed() {
            if self.emitted.get(&f.id) == Some(&f.state) {
                continue;
            }
            self.emitted.insert(f.id, f.state);
            log.append(
                EventDraft::new(
                    EventKind::new("plugin/state").expect("static"),
                    "kernel",
                    serde_json::json!({
                        "fiber_id": f.id.0,
                        "plugin": f.key,
                        "state": f.state,
                        "error": f.error,
                    }),
                )
                .with_salience(60),
            );
        }
    }

    /// 消费 Op 直到 Shutdown；然后 flush 日志、逆序撤销全部插件。
    ///
    /// 服务是**每条 Op** 解析一次的：boot 时缺席的 provider（跨进程插件、后到的 bundle）
    /// 一旦出现就立刻生效；仍然缺席时 warn 并丢弃这条 Op。
    pub async fn run(mut self) -> Result<(), BootError> {
        self.publish_states();
        // 背压汇总的节拍。和 Op 一起 `select!`：内核循环是单任务的，这里不能起第二个
        // 任务去读同一个 `dropped`——它要写的是世界日志，而那正是这条循环在做的事。
        let mut report = tokio::time::interval(Duration::from_millis(
            self.cfg.backpressure_report_ms.max(1),
        ));
        report.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let op = tokio::select! {
                op = self.rx.recv() => match op {
                    Some(op) => op,
                    // 所有 `KernelHandle` 都没了：`Kernel` 自己留着一份 `tx`，所以这条
                    // 分支只在 `Kernel` 被消费之后的极端情况下出现。
                    None => break,
                },
                _ = report.tick() => {
                    // 关联函数而不是方法：`self.rx.recv()` 的 future 借着 `self.rx`，
                    // 一个 `&mut self` 的方法会和它撞上。
                    report_backpressure(&self.registry, &self.dropped);
                    continue;
                }
            };
            match op {
                Op::Emit(d) => match self.registry.get::<WorldLog>(world_log_name()) {
                    Ok(l) => {
                        l.append(d);
                    }
                    Err(_) => tracing::warn!("Op::Emit dropped: no world-log service"),
                },
                Op::IngestFrame {
                    stream_id,
                    payload,
                    t_source,
                } => match self.registry.get::<FrameStore>(frame_store_name()) {
                    Ok(f) => {
                        if let Err(e) = f.ingest(&stream_id, payload, t_source) {
                            tracing::warn!(error = %e, "ingest failed");
                        }
                    }
                    Err(_) => tracing::warn!("Op::IngestFrame dropped: no frame-store service"),
                },
                Op::AddFiber { row, plugin, reply } => {
                    let id = self.tree.add(row, plugin);
                    let _ = self.tree.settle();
                    let info = self.tree.states_detailed().into_iter().find(|f| f.id == id);
                    self.publish_states();
                    let _ = reply.send(FiberAdded {
                        id,
                        state: info.as_ref().map(|f| f.state).unwrap_or(FiberState::Failed),
                        error: info.and_then(|f| f.error),
                    });
                }
                Op::DisposeFiber { id, reply } => {
                    self.tree.dispose(id);
                    let _ = self.tree.settle();
                    self.publish_states();
                    // 剪枝在广播**之后**：正在退休的那一行的 `plugin/state{disposed}` 必须
                    // 先出去，行才能丢（见 `FiberTree::prune_disposed`）。跨进程插件把
                    // "加一行"变成了客户端能反复触发的动作，这一步是那条增长曲线的封顶。
                    self.tree.prune_disposed();
                    let kept: HashSet<FiberId> = self.tree.ids().into_iter().collect();
                    // 同一个集合：`emitted` 只是"上次广播过什么"的备忘，树上没有的行不会
                    // 再有状态变化。id 单调，所以一条陈旧记录也不可能被错认成新 Fiber。
                    self.emitted.retain(|id, _| kept.contains(id));
                    if let Some(r) = reply {
                        let _ = r.send(());
                    }
                }
                Op::Shutdown => break,
            }
        }
        // 关机前把最后一段丢弃计数落账：一次让内核落后到丢 Op 的过载，不能因为它发生在
        // 最后一个汇总周期里就在日志里查无此事。
        report_backpressure(&self.registry, &self.dropped);
        // dispose_all 会撤销 world-log 的注册，所以句柄要在那之前抓住。
        let log = self.registry.get::<WorldLog>(&"world-log".into()).ok();
        if let Some(l) = &log {
            let _ = l.flush();
        }
        // 路由是异步的：`dispose_all` 会 abort 路由任务，队列里还没处理的事件就此消失，
        // 关机前最后一批 turn/会话日志也跟着丢。先等它排空（卡住的话最多等 2 s）。
        if let Ok(b) = self.registry.get::<crate::brains::Brains>(&"brains".into()) {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(2), b.settle()).await;
            // 排空每个脑的会话日志（总预算 1 s），**在 `dispose_all` 之前**。
            //
            // 不能指望 `SessionLog::drop`：它只在撤销时**恰好是最后一个强引用**才排空，
            // 而关机这一刻 `Brains` 的 `Arc` 很可能还被别人攥着（RPC 服务器、还没归还的
            // 句柄）；而且它 flush 超时就不 join，进程随后退出会把尾巴一起带走。丢的正是
            // 关机前最后一段 turn / 提案 / 结局的记账。
            //
            // 这里是**阻塞**调用（最长 1 s），这一次是对的：Op 循环已经退出，这条任务此刻
            // 唯一的工作就是有序收尾。
            b.flush_sessions();
        }
        self.tree.dispose_all();
        // `dispose_all` tears fibers down in reverse registration order, so `world-log` can be
        // disposed (and its own sink flushed) before other plugins' spawned tasks (e.g.
        // `mock-stream` / `frame-counter`'s `spawn_interval`) are aborted. On a multi-threaded
        // runtime such a task can still be mid-`append()` on another OS thread when that happens,
        // landing a record in the sink's `BufWriter` after the flush above ran. Flush again here,
        // after every fiber (including any still-racing task) has been disposed, so nothing
        // committed to the in-memory log is left unflushed on disk.
        if let Some(l) = &log {
            let _ = l.flush();
        }
        Ok(())
    }
}
