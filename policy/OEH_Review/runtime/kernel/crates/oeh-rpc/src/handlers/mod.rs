//! 方法表的实现。每个子模块对应边界协议里的一组方法（`oeh_proto::rpc::method`），
//! [`crate::router::KernelRouter::call`] 只负责把方法名分发到这里。
//!
//! 三条贯穿所有处理函数的纪律：
//!
//! 1. **锁**：`Conn::state`（`parking_lot`）只在取/放句柄的一小段临界区里持有，绝不跨
//!    `.await`、`WorldLog::append` 或 `Disposer`。`turn/propose` 因此先把
//!    [`oeh_kernel::TurnHandle`] 克隆出来再出锁 `.await`；`turn/done` 先 `remove` 再出锁
//!    调 `done()`（`done` 会 append，append 会同步跑观察者，观察者会回头 `conn.notify`）。
//! 2. **身份**：写进世界日志的 `source` 永远由内核这一侧决定（`rpc:<conn 名>`），
//!    调用方说自己是谁一概不算数——见 [`event`] 模块的说明。
//! 3. **归属**：连接是资源的所有者。attach 的脑、挂上的 `log/tail` 观察者都记在
//!    [`crate::conn::ConnState`] 里，断连时由一条 `on_close` 钩子统一归还。

pub mod approval;
pub mod brain;
pub mod event;
pub mod log;
pub mod plugin;
pub mod robot;
pub mod stream;
pub mod turn;

use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::KernelSendError;
use oeh_proto::rpc::{code, RpcError};
use serde::Serialize;

use crate::conn::Conn;

/// Op 队列满时重试几次、每次隔多久。`Busy` 不是终局——内核正在排队，一次 10 ms 的
/// 退避足够它把队列挪动几千条；把一次瞬时背压直接报成 `KERNEL_GONE`，会让一个
/// 完全健康的适配器在内核最忙的那一刻放弃注册。
pub(crate) const OP_BUSY_RETRIES: usize = 3;
pub(crate) const OP_BUSY_BACKOFF: Duration = Duration::from_millis(10);

/// 投一条 Op，`Busy` 就退避重试；`Gone` 立刻放弃（重试没有意义）。
///
/// `f` 每次重试都重新调用：`Op::AddFiber` 这类带载荷的命令在被拒时载荷已经没了，
/// 调用方必须能重建它。
pub(crate) async fn send_op_retrying<T>(
    mut f: impl FnMut() -> Result<T, KernelSendError>,
) -> Result<T, KernelSendError> {
    for _ in 0..OP_BUSY_RETRIES {
        match f() {
            Err(KernelSendError::Busy) => tokio::time::sleep(OP_BUSY_BACKOFF).await,
            other => return other,
        }
    }
    f()
}

/// [`send_op_retrying`] 的同步版：调用方在一个**不能 `.await` 也不能阻塞**的地方
/// （`Conn::on_close` 钩子是同步 `FnOnce`，而且跑在连接的读任务上）。把重试甩给一个
/// tokio 任务；不在 runtime 里（进程正在拆）就只投一次，投不进去也就算了——内核没了
/// 或者忙到满队列时，那棵树连同这个 Fiber 一起没了。
pub(crate) fn send_op_retrying_detached<T: Send + 'static>(
    f: impl FnMut() -> Result<T, KernelSendError> + Send + 'static,
) {
    let mut f = f;
    match f() {
        Ok(_) | Err(KernelSendError::Gone) => {}
        Err(KernelSendError::Busy) => match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                rt.spawn(async move {
                    if send_op_retrying(f).await.is_err() {
                        tracing::warn!("kernel op dropped: queue stayed full");
                    }
                });
            }
            Err(_) => tracing::warn!("kernel op dropped: queue full and no runtime to retry on"),
        },
    }
}

/// 一个 serde 枚举的线格式名字（`CapabilityClass`/`RiskLevel` 这类 `rename_all` 过的
/// 单元枚举）。边界协议里它们是字符串，不是 Rust 的 `Debug` 名字——`format!("{:?}")`
/// 会写出 `Motion` 而线上约定是 `"motion"`。
pub(crate) fn wire_name<T: Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        other => {
            debug_assert!(false, "{other:?} is not a string-shaped enum");
            String::new()
        }
    }
}

/// 结果序列化失败只可能是编程错误（result 类型都是纯数据）；编成 `INTERNAL` 而不是
/// `unwrap()`，一条方法的笔误不该把整条连接的读循环炸掉。
pub(crate) fn ok<T: Serialize>(v: T) -> Result<serde_json::Value, RpcError> {
    serde_json::to_value(v).map_err(|e| {
        RpcError::new(code::INTERNAL, "failed to encode result")
            .with_data(serde_json::Value::String(e.to_string()))
    })
}

/// 登记这条连接的断连清理钩子（幂等）：把它 attach 过的脑 detach 掉、把 `log/tail` 的
/// 观察者撤掉。只登记一条——钩子本身把 `ConnState` 里的东西一次归还，而
/// `brain/attach`/`log/tail` 可以被调用任意多次。
///
/// 钩子持 `Weak<Conn>`：`Conn` 自己攥着这张钩子表，钩子里再攥一个 `Arc<Conn>` 就是一个
/// 自环，这条连接（以及它拉着的脑、观察者）永远不会被回收。
///
/// 连接在登记之前就已经关掉时 [`Conn::on_close`] 会立刻跑这条钩子；登记之后、放东西
/// 进 `ConnState` 之前关掉的情况由调用方各自的 `settle_if_closed` 兜住。
pub(crate) fn ensure_cleanup_hook(conn: &Arc<Conn>) {
    {
        let mut st = conn.state.lock();
        if st.cleanup_hooked {
            return;
        }
        st.cleanup_hooked = true;
    }
    let weak = Arc::downgrade(conn);
    conn.on_close(Box::new(move || {
        if let Some(conn) = weak.upgrade() {
            release_all(&conn);
        }
    }));
}

/// 归还这条连接持有的一切。出锁之后才跑 `Disposer`：`Brains::attach` 的那一个会 append
/// `brain/detach`，append 会同步跑观察者，观察者会回头 `conn.notify` ——带着
/// `ConnState` 的锁进去就是自己和自己死锁。
///
/// 这一次 `mem::take` **之后**才到手的 `Disposer` 不归这里管，也不会因此漏掉：交出
/// `Disposer` 的那些地方（目前只有 `brain/attach`）遵守"存不进去就当场执行"——它们在
/// `Conn::state` 的锁下检查槽位还在不在，不在就自己把它跑掉（出锁之后）。这条规则和这里
/// 的 `take` 是同一把锁串起来的，所以两边不会同时认为对方会处理。
///
/// 跨进程插件（`fibers`）的归还是**异步**的：`dispose_fiber` 只往内核的 Op 通道投一条
/// 命令，绝不等它被处理——这个函数会从 `on_close` 钩子里被调用，而钩子是同步 `FnOnce`
/// （`Conn::on_close` 的契约），在里面 `.await` 根本不可能，在里面阻塞等内核更是把
/// 连接的读任务押给了内核的调度。投不进去（内核已经停了）也不必兜底：内核停了，那棵树
/// 连同这个 Fiber 一起没了。
///
/// `execs` 不在这里**清**（见 [`crate::conn::ConnState::execs`] 上的注释），但在这里
/// **释放**：第一件事就是把这条连接上还在跑的跨进程执行当场撤掉
/// （[`crate::remote::release_executions_on_disconnect`]，Plan 3 遗留 I4）。次序是有意
/// 的——先撤执行、再撤 Fiber：撤执行要读 `ConnState.execs` 里每条登记的归属，而那份归属
/// 指向的准入注册正是 Fiber 的 dispose 要拆掉的东西。反过来写的话，一次跑得快的 dispose
/// 会让这条路看见一张认不出主人的表。
pub(crate) fn release_all(conn: &Arc<Conn>) {
    crate::remote::release_executions_on_disconnect(conn);
    let (brains, tail, fibers, approval) = {
        let mut st = conn.state.lock();
        (
            std::mem::take(&mut st.brains),
            st.tail.take(),
            std::mem::take(&mut st.fibers),
            st.approval.take(),
        )
    };
    for (_, slot) in brains {
        if let Some(d) = slot.detach {
            d();
        }
    }
    if let Some(d) = tail {
        d();
    }
    for (id, f) in fibers {
        // `Busy` 不能当成"撤掉了"：那一行会留在树上，而这条连接已经没了——没有人再会
        // 为它续心跳，也没有人再来 `plugin/unregister` 它。重试甩给一个任务（这里是
        // 同步钩子，不能 await）。
        let kernel = f.kernel.clone();
        send_op_retrying_detached(move || kernel.dispose_fiber(oeh_kernel::FiberId(id)));
    }
    if let Some(d) = approval {
        d();
    }
}

/// 刚往 `ConnState` 里放了东西之后调用：连接如果在"登记钩子"和"放进去"之间关掉了，
/// 那条钩子看到的是一个还空着的 `ConnState`，放进去的东西就再没人归还了。
pub(crate) fn release_if_closed(conn: &Arc<Conn>) {
    if conn.is_closed() {
        release_all(conn);
    }
}

pub(crate) fn not_attached(what: impl std::fmt::Display) -> RpcError {
    RpcError::new(code::NOT_ATTACHED, format!("not attached: {what}"))
}

pub(crate) fn refused(reason: impl Into<String>) -> RpcError {
    RpcError::new(code::REFUSED, reason)
}

pub(crate) fn invalid_params(reason: impl std::fmt::Display) -> RpcError {
    RpcError::new(code::INVALID_PARAMS, reason.to_string())
}

pub(crate) fn internal(reason: impl std::fmt::Display) -> RpcError {
    RpcError::new(code::INTERNAL, reason.to_string())
}
