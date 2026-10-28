use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use oeh_proto::EventDraft;
use tokio::sync::{mpsc, oneshot};

use crate::fiber::FiberState;
use crate::plugin::Plugin;
use crate::profile::EntryRow;
use crate::services::FiberId;

/// `Op::AddFiber` 的回执：加入 + settle 之后这个 Fiber 落在哪个状态。
#[derive(Debug, Clone)]
pub struct FiberAdded {
    pub id: FiberId,
    pub state: FiberState,
    pub error: Option<String>,
}

/// 进程内命令。Plan 2 起扩展 Propose / Steer / Interrupt / Approve；Plan 3 起扩展
/// AddFiber / DisposeFiber，让跨进程插件（RPC 服务器背后的客户端）能在运行时变成 Fiber。
pub enum Op {
    Emit(EventDraft),
    IngestFrame {
        stream_id: String,
        payload: Bytes,
        t_source: Option<u64>,
    },
    /// 跨进程插件：把一个 Plugin 实现加进树、settle、回报它的状态。
    AddFiber {
        row: EntryRow,
        plugin: Box<dyn Plugin>,
        reply: oneshot::Sender<FiberAdded>,
    },
    /// 显式移除（Disposed 是终态）。
    DisposeFiber {
        id: FiberId,
        reply: Option<oneshot::Sender<()>>,
    },
    Shutdown,
}

/// 一条 Op 投不进去的两种原因。调用方必须 fail-closed 处理：投不进去的 Op 就是没发生，
/// 不能当成成功。
///
/// 两者**必须分开**：`Gone` 是终局（内核停了或 `run` 已经返回，重试无意义），`Busy` 是
/// 此刻队列满了（内核正在排，10 ms 后大概率就有位置）。混成一个错误，调用方要么把一次
/// 背压当成内核死亡而放弃，要么对着一个已经没了的内核重试到天荒地老。
///
/// 这个类型取代了 Plan 1 起的 `KernelGone`（`Gone` 就是它）。那个类型在 Plan 4 Task 0
/// 之后全仓没有任何构造点，留着只会让人以为还有第二条错误通道。
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KernelSendError {
    #[error("kernel op channel is closed")]
    Gone,
    #[error("kernel op queue is full")]
    Busy,
}

/// 内核 Op 通道的**生产端**。克隆随便克隆：容量是共享的那一份。
///
/// 通道有界（[`crate::kernel::KernelConfig::op_queue`]，默认 4096）：满了就
/// [`KernelSendError::Busy`]，被拒的次数由内核每 `backpressure_report_ms` 汇总成一条
/// `health/op_backpressure`（见 [`crate::Kernel::run`]）。
///
/// 今天的生产代码里 Op 只有控制面的生产者——`oeh run` 的关机、`oeh-rpc` 的
/// `add_fiber`/`dispose_fiber`——全都低频、全都做了 `Busy` 重试。高频的那两条路**不经过
/// 这里**：`event/emit` 直接 `WorldLog::append`，iceoryx2 ingress 与 `mock-stream` 直接
/// 打 `FrameStore`。界限防的是**将来**：`Op::IngestFrame` 一旦被接上一条 500 Hz 的远端
/// 流，无界通道就是一条没有上限的内存曲线，而内核在 OOM 之前什么都不会说。到那时
/// `Busy`（丢最新的一帧）也正是合意的行为——好过让生产者阻塞。
#[derive(Clone)]
pub struct KernelHandle {
    tx: mpsc::Sender<Op>,
    /// 队列满而**被拒的投递次数**（不是"丢失的 Op 数"）：重试成功的那些也各计一次，
    /// 见 [`KernelHandle::dropped`]。内核循环定期取走并归零。
    dropped: Arc<AtomicU64>,
}

impl KernelHandle {
    pub(crate) fn new(tx: mpsc::Sender<Op>, dropped: Arc<AtomicU64>) -> Self {
        Self { tx, dropped }
    }
    /// 投一条 Op。**从不阻塞**（`try_send`）：调用方遍布异步任务、插件回调和别的
    /// 线程，任何一个在这里被按住都会把它自己那条链一起按住。
    pub fn send(&self, op: Op) -> Result<(), KernelSendError> {
        match self.tx.try_send(op) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Err(KernelSendError::Busy)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(KernelSendError::Gone),
        }
    }
    pub fn emit(&self, d: EventDraft) -> Result<(), KernelSendError> {
        self.send(Op::Emit(d))
    }
    pub fn ingest(
        &self,
        stream_id: &str,
        payload: Bytes,
        t_source: Option<u64>,
    ) -> Result<(), KernelSendError> {
        self.send(Op::IngestFrame {
            stream_id: stream_id.to_string(),
            payload,
            t_source,
        })
    }
    pub fn add_fiber(
        &self,
        row: EntryRow,
        plugin: Box<dyn Plugin>,
    ) -> Result<oneshot::Receiver<FiberAdded>, KernelSendError> {
        let (tx, rx) = oneshot::channel();
        self.send(Op::AddFiber {
            row,
            plugin,
            reply: tx,
        })?;
        Ok(rx)
    }
    pub fn dispose_fiber(&self, id: FiberId) -> Result<oneshot::Receiver<()>, KernelSendError> {
        let (tx, rx) = oneshot::channel();
        self.send(Op::DisposeFiber {
            id,
            reply: Some(tx),
        })?;
        Ok(rx)
    }
    /// 关机命令走同一条有界通道，也一样可能 `Busy`——它不能插队（前面排着的 Op 是
    /// 已经被接受的工作，关机不该把它们吞掉），也不能阻塞调用方。调用方重试即可。
    pub fn shutdown(&self) -> Result<(), KernelSendError> {
        self.send(Op::Shutdown)
    }
    /// 迄今为止因为队列满而**被拒的投递次数**（内核汇总时归零）。
    ///
    /// 不等于"丢失的 Op 数"：会重试的调用方（`oeh-rpc` 的 3 × 10 ms、`oeh run` 的关机
    /// 退避）每被拒一次就计一次，而最后那次成功的投递什么都没丢。它是一个**背压强度**
    /// 的指标——"内核这段时间里让多少次投递吃了闭门羹"——不是审计意义上的丢失计数。
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}
