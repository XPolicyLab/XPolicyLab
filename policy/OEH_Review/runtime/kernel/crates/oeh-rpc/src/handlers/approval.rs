//! `approval/attach` · `approval/detach`——把"谁来点头"这件事接到一条连接上。
//!
//! 同一时刻至多一个审批服务（`Admission::set_approval` 的语义），所以 attach 是**抢占
//! 式**的：后来的那个接管，前一个（可能在另一条连接上）从此不再被问。这是故意的——
//! 操作员接管一台机器人，不该先去问上一个控制台愿不愿意让出来。被顶掉的那条连接会收到
//! 一条 `approval/detached{reason:"replaced"}`：一个还在界面上显示"我是审批人"、却再也
//! 不会被问到的控制台，是这条接缝最不该产生的东西。
//!
//! attach **不叠加**：B 接管 A 之后 B 再 detach，系统回到"没有审批服务"（于是一切 `Ask`
//! 都是 `Deny`），不会回落到 A。fail-closed，而且"谁在把关"永远只有一个答案。
//!
//! 断连即 detach：审批服务归这条连接所有，记在 `ConnState.approval` 里，由
//! [`crate::handlers::release_all`] 统一归还。一个连着一条死连接的审批服务，效果等同于
//! "所有 Ask 都超时"——那是最坏的一种故障：机器人不动，而日志里只有一串审批超时。

use std::sync::Arc;

use oeh_proto::rpc::types::{ApprovalAttach, ApprovalDetached, Empty};
use oeh_proto::rpc::{method, RpcError};
use serde_json::Value;

use crate::conn::Conn;
use crate::handlers::{ensure_cleanup_hook, not_attached, ok, release_if_closed};
use crate::remote::RpcApproval;
use crate::router::{parse_params, Services};

/// 把这条连接装成审批服务。`name` 是写进 `act/decision{by}` 的那个名字（空了就用连接
/// 自己的名字）——审计里"谁批的"必须有个名字。
pub fn attach(s: &Services, conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: ApprovalAttach = parse_params(&params)?;
    let name = if p.name.trim().is_empty() {
        conn.name()
    } else {
        p.name.trim().to_string()
    };
    ensure_cleanup_hook(conn);
    // 整个"装上自己、收掉别人"锁在这把闸门下面（Task 5 re-review R1）：`set_approval` +
    // `ConnState.approval.replace` + `displace_others` 三步之间原来没有锁跨着，两条连接
    // 真并发 attach 时会在几条指令的窗口里插花——B 在 A 把 disposer 存进自己的
    // `ConnState` 之前就跑完了 `displace_others`（看到的还是 `None`，跳过 A），随后 A
    // 的 `displace_others` 却把刚接管的 B 收掉：两边都没了审批服务，而 A 从未被通知、
    // 自以为还在把关。函数体不 `.await`，锁不会跨等待点。
    //
    // 这把 `parking_lot` 锁确实跨了回调和一次 `append`——本 crate 的模块文档明文禁止的那
    // 个形状（"`parking_lot` 锁不跨 `append` / 回调"）。为什么**今天**它仍然是安全的，
    // 以及将来什么会让它不安全（最终评审 M5）：`displace_others` 里最长的一条链是
    // `other.notify` →（队列满）`mark_closed` → `drain_hooks` → `release_all` → 各个
    // `Disposer`（脑、远端 Fiber、tail）→ `WorldLog::append` → 整条观察者链。这条链上
    // **没有任何一环再去拿 `approval_gate`**：能拿它的只有这个文件里的 `attach`，而它只
    // 从方法分发进来（`Router::call`），不从任何 `Disposer` 或观察者里进来。所以不是"锁
    // 得很短所以没事"，是"这个环根本不存在"。
    //
    // 让它不再成立只需要一件事：某个 `Disposer` 或世界日志观察者里出现一次 attach/detach
    // （比如"审批端掉线就自动把审批交回某条控制台"）。真要加那种逻辑，先把这里改成"在闸门
    // 下**收集**要顶掉的名单，出锁之后再跑 `Disposer` 和 `notify`"——那是等价的、且不依赖
    // 调用图的写法。
    let _gate = s.approval_gate.lock();
    let disposer = s
        .admission
        .set_approval(Arc::new(RpcApproval::new(conn, &name)));
    let previous = conn.state.lock().approval.replace(disposer);
    // 出锁之后跑。这条连接**自己**之前那次 attach 的撤销句柄：`set_approval` 的
    // `Disposer` 只在装着的仍是自己那一个时才清除，所以这一下不会把刚装上的这个拆掉。
    // 自己顶掉自己不发通知——调用方刚发的就是这次 attach，它知道。
    if let Some(d) = previous {
        d();
    }
    displace_others(s, conn);
    drop(_gate);
    release_if_closed(conn);
    ok(Empty {})
}

/// 把**别的**连接上那份已经失效的审批登记收掉，并告诉它一声。
///
/// 为什么要收：`Admission` 那边早就只认最后装上的那一个了（`set_approval` 用 id 守着），
/// 所以别的连接手里的 `Disposer` 已经是个空操作。但它还躺在那条连接的 `ConnState.approval`
/// 里，于是 `approval/detach` 会回 `Ok`（听起来像"拆掉了"，其实什么都没拆），而那边的操作
/// 员界面还显示自己在把关。把它清掉，"`ConnState.approval` 是 `Some`" 就重新等于
/// "我现在真的是审批人"。
///
/// 锁的纪律：每条连接的句柄在它自己那把锁下取出，`Disposer` 和 `notify` 都在锁外跑。
///
/// 顺序（先装上自己、再收别人）是有意的。两条连接同时 attach 时，后收的那一下可能收掉的
/// 正是**刚装上**的那一个，于是两边都没了审批服务——一切 `Ask` 变成 `Deny`，而且两个控制
/// 台都收到了 `approval/detached`。反过来（先收后装）那个竞争会留下一个"有人把关、但被顶
/// 掉的那台什么都不知道"的状态，也就是这个函数要消灭的东西。一次需要重新 attach 的
/// fail-closed，好过一个沉默的假象。
fn displace_others(s: &Services, self_conn: &Arc<Conn>) {
    let detached = serde_json::to_value(ApprovalDetached {
        reason: "replaced".into(),
    })
    .unwrap_or(Value::Null);
    for other in s.conns.iter() {
        if other.id().0 == self_conn.id().0 {
            continue;
        }
        let stale = other.state.lock().approval.take();
        let Some(d) = stale else { continue };
        d();
        let _ = other.notify(method::APPROVAL_DETACHED, detached.clone());
    }
}

/// 拆掉**这条连接**装的那个。别人装的不在这张表里 → `NOT_ATTACHED`：一个能拆别人审批
/// 服务的 `approval/detach`，等于给了任何客户端一个"让机器人从此无人把关"的开关。
pub fn detach(conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let _: Empty = parse_params(&params).unwrap_or(Empty {});
    let disposer = conn.state.lock().approval.take();
    match disposer {
        Some(d) => {
            d();
            ok(Empty {})
        }
        None => Err(not_attached("approval")),
    }
}
