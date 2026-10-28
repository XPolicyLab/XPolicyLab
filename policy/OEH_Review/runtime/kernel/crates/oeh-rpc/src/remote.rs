//! 跨进程插件的内核侧实现：一条连接上的一次 `plugin/register` 在内核树上就是一个真正的
//! Fiber（[`RemotePlugin`]），它声明的能力由 [`RemoteExecutor`] 通过 `capability/execute`
//! 反向请求跑在客户端那一侧，`approval/attach` 装上的审批服务是 [`RpcApproval`]。
//!
//! 这个文件的中心思想只有一条：**跨进程的东西也必须是可逆的 Effect**。`RemotePlugin::
//! apply` 把 Manifest 里的声明（流、能力、provide）一条条变成 `ctx.on_dispose` 登记的
//! 撤销动作，于是"客户端断线"和"树上的一个 Fiber 被 dispose"变成同一件事——能力面缩回
//! 去、流从 `stream/list` 里消失、服务从注册表里撤掉，一件不落，不用任何一段专门的
//! "清理跨进程插件"的代码。
//!
//! 谁持有谁（回收链，这是几处 `Arc`/`Weak` 选择的依据）：
//!
//! - `FiberTree`（内核） → `RemotePlugin` → **`Weak<Conn>`**：`FiberTree` 从不真的丢掉
//!   一行——`dispose` 只把状态改成 `Disposed`，`Fiber`（连同它的 `Box<dyn Plugin>`）留在
//!   那个 `Vec` 里直到内核关机。这里要是 `Arc`，每一次"注册→断连"都会把一整条 `Conn`
//!   （连同它的出站队列和 `ConnState`）钉死到进程结束：一个反复重连的适配器就是一条
//!   稳定的内存泄漏。`apply` 只在激活的那一瞬间需要连接，`upgrade` 不出来就是
//!   fail-closed（给一条已经没了的连接注册能力毫无意义）。
//! - `CapabilityRegistry`（准入） → `RemoteExecutor` → `Arc<Conn>`：这里是 `Arc`，而且
//!   必须是——执行器要在**任意时刻**能往那条连接上发东西。它的生命期由注册本身的
//!   `Disposer` 兜住（Fiber 的 `Effects` 在 dispose 时逆序跑完），所以 Fiber 一撤，
//!   这个 `Arc` 就没了。
//! - `Admission` → [`RpcApproval`] → **`Weak<Conn>`**：这里如果是 `Arc` 就真成环了
//!   （`Conn` → `ConnState.approval` 的 `Disposer` → `Admission` 的 `Inner` →
//!   `AdmState.approval` → `RpcApproval` → `Conn`）。断连会跑那个 `Disposer` 把环拆开，
//!   但一个"只要清理路径跑到就不漏"的所有权图，和一个"本来就没有环"的所有权图，不是
//!   同一个安全等级。

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use std::time::Duration;

use oeh_admission::admission::Admission;
use oeh_admission::approval::{Approval, AskRequest, Decision};
use oeh_admission::capability::{
    CancelReason, CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor,
};
use oeh_kernel::plugin::{Ctx, Plugin, PluginError};
use oeh_kernel::proposal::{BoxFuture, ProposalId};
use oeh_kernel::{FrameStore, WorldLog};
use oeh_proto::rpc::types::{
    ApprovalDecision, ApprovalRequest, CapabilityAccepted, CapabilityCancel, CapabilityExecute,
    CapabilityResult, ExecRef,
};
use oeh_proto::rpc::{code, method};
use oeh_proto::{EventDraft, EventKind, Manifest, ServiceName};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::conn::{Conn, ConnId, ExecSlot};
use crate::frames::Frames;
use crate::router::RouterConfig;

/// 远端服务的占位值：`manifest.provide` 里的每个名字都由这个东西顶上。
///
/// 进程内依赖它的插件 `inject` 到的就是这个——它们关心的是"这个服务在不在"（不在的话
/// 它们自己就是 `Pending`，跨进程插件一上线就自动激活），真正的调用走各自的协议。
#[derive(Clone, Debug)]
pub struct RemoteService {
    pub conn: ConnId,
    pub name: ServiceName,
}

/// 一次 `plugin/register` 在内核树上的那个 Fiber。
pub struct RemotePlugin {
    manifest: Manifest,
    caps: Vec<CapabilitySpec>,
    /// `Weak`：见模块头上的回收链说明（`FiberTree` 永远不丢行）。
    conn: Weak<Conn>,
    cfg: Arc<RouterConfig>,
    /// 这个内核的帧通道（[`crate::frames`]）：声明的每条流都要在这上面起一条 ingress。
    /// `Arc` 而不是 `Weak`：它归 `rpc-server` 那个 Fiber 所有，而这个 Fiber 是它注册出来
    /// 的——`rpc-server` 撤了，所有连接连同它们的 Fiber 也就都没了。
    channels: Arc<Frames>,
}

impl RemotePlugin {
    pub fn new(
        manifest: Manifest,
        caps: Vec<CapabilitySpec>,
        conn: &Arc<Conn>,
        cfg: Arc<RouterConfig>,
        channels: Arc<Frames>,
    ) -> Self {
        Self {
            manifest,
            caps,
            conn: Arc::downgrade(conn),
            cfg,
            channels,
        }
    }
}

impl Plugin for RemotePlugin {
    /// 原样交出客户端声明的 Manifest：`inject` 决定这个 Fiber 什么时候能激活（声明的服务
    /// 一旦缺席，`settle()` 会把它退回 `Pending`），`provide` 决定它上线之后谁能被激活。
    /// 内核对一个跨进程插件的要求和对一个进程内插件**完全一样**——这正是"跨进程插件是
    /// 一等公民"的含义。
    fn manifest(&self) -> Manifest {
        self.manifest.clone()
    }

    /// 跑在内核任务上，同步、不阻塞：只做注册，一次 I/O 都没有（`ctx.get` 是读注册表，
    /// `register` 是改一张 `HashMap`）。真正会等的那些事（执行、审批）发生在别处。
    ///
    /// `frame-store` / `admission` 用 `ctx.get` 而不是写进 `manifest.inject`：inject 是
    /// **客户端**的声明，内核不该往里塞它没写过的东西（那会让 `robot/health` 里报的
    /// manifest 跟客户端自己说的不是一回事）。代价说清楚：这两个服务要是之后被撤掉，
    /// 这个 Fiber 不会自动退回 `Pending`。它们是内核骨架的一部分，真没了的话，一个
    /// 挂着的远端 Fiber 是最小的问题。
    fn apply(&self, ctx: &Ctx, _config: &Value) -> Result<(), PluginError> {
        // 连接已经没了（客户端在 `add_fiber` 和 settle 之间断线，或者这个 Fiber 是在一次
        // 重新 settle 里被激活的）：fail-closed。注册一个没人能执行的能力，比不注册糟得
        // 多——脑会提案它，然后每一次都撞在一个"连接丢了"的失败上。
        let conn = self
            .conn
            .upgrade()
            .ok_or_else(|| PluginError::Other("the registering connection is gone".into()))?;
        for spec in &self.manifest.streams {
            let store = ctx.get::<FrameStore>("frame-store")?;
            // 打标在 `register` **之前**，这一步的次序是承重的：`stream/subscribe` 只要看见
            // 流存在就会考虑给它起 egress，而内核自己的 egress 和这条流的 ingress 同时挂在
            // 一条通道上，就是一个自激的帧回路（见 [`crate::frames`] 模块头）。反过来写的话
            // ——先 register 再打标——中间那几毫秒就是那个回路的入口。
            //
            // 凭据先交给效果链（`apply` 中途失败、甚至 panic，`FiberTree::activate` 都会把
            // 效果逆序跑完），手里再留一份：`commit` 要等 `register` 的结果。
            let mark = self.channels.begin_remote(&spec.stream_id);
            let held = mark.clone();
            ctx.on_dispose(Box::new(move || drop(held)));
            // 注册失败（最常见的是流名撞上了一个本地生产者）：这一行 `?` 就地返回，标记随
            // 效果链一起交还，而这条流原有的 egress 一根汗毛都没动过——它属于别人，一次
            // 被拒绝的注册没有资格碰它。
            let d = store
                .register(spec.clone())
                .map_err(|e| PluginError::Other(e.to_string()))?;
            ctx.on_dispose(d);
            // 注册成功了，这条流从现在起归远端：扫掉残留的 egress（上一个本地生产者留下的
            // 那条），此后也不会再有新的。
            mark.commit();
            // 声明了一条流，就得有人真的把帧搬进来。起不来（共享内存开不出来、通道上的
            // 订阅者位置满了、建服务超时）是 fail-closed：一个"流在 `stream/list` 里、但
            // 永远没有帧"的插件，比一个明说自己 Failed 的插件难查一百倍。
            //
            // 次序：`FrameStore::register` 在前、ingress 在后，于是撤销时（Effects 逆序）
            // 先停线程、再注销流、最后才交还"远端发布"的标记——每一步都发生在下一步已经
            // 不可能被观察到之后。
            let d = self
                .channels
                .start_ingress(&spec.stream_id)
                .map_err(|e| PluginError::Other(format!("stream `{}`: {e}", spec.stream_id)))?;
            ctx.on_dispose(d);
        }
        if !self.caps.is_empty() {
            let admission = ctx.get::<Admission>("admission")?;
            // 这个插件所有执行器共享的一份归属，见 [`ExecOwner`]。`world-log` 和
            // `admission` 一样用 `ctx.get` 而不是写进 `manifest.inject`（见这个函数
            // 头上的说明）。
            let owner = Arc::new(ExecOwner {
                plugin: self.manifest.name.clone(),
                admission: Arc::downgrade(&admission),
                log: ctx.get::<WorldLog>("world-log")?,
            });
            for spec in &self.caps {
                let exec = Arc::new(RemoteExecutor {
                    conn: conn.clone(),
                    accept: Duration::from_millis(self.cfg.execute_accept_ms),
                    cancel_grace: Duration::from_millis(self.cfg.cancel_grace_ms),
                    lease_ttl_s: spec.lease_ttl_s,
                    owner: owner.clone(),
                });
                // 以**这个插件的名字**注册：`robot/capabilities` 的 `provider` 报的就是
                // 它，而一次对内置 `say` 的顶替（以及断连时的恢复）靠它说清楚是谁干的。
                let d = admission
                    .capabilities()
                    .register_from(&self.manifest.name, spec.clone(), exec)
                    .map_err(|e| PluginError::Other(e.to_string()))?;
                ctx.on_dispose(d);
            }
        }
        for name in &self.manifest.provide {
            // `provide` 自己就登记了撤销动作（`Ctx::provide`），这里不必再补一条。
            ctx.provide(
                name.0.as_str(),
                Arc::new(RemoteService {
                    conn: conn.id(),
                    name: name.clone(),
                }),
            )?;
        }
        Ok(())
    }
}

/// 一个跑在连接那一头的执行器。
pub struct RemoteExecutor {
    conn: Arc<Conn>,
    /// 等 `capability/execute` 那条**回执**（`{accepted}`）的上限。不是等结果的上限——
    /// 结果什么时候回来由租约（`lease_ttl_s` + 心跳）管。
    accept: Duration,
    /// 发出 `capability/cancel` 之后，还愿意等客户端把结果补回来多久。
    cancel_grace: Duration,
    lease_ttl_s: f64,
    /// 断连即释放要用的东西，见 [`ExecOwner`]。
    owner: Arc<ExecOwner>,
}

/// 一次跨进程执行的归属：跑它的是**谁**的执行器，以及这条连接断了之后该找谁把它撤掉、
/// 把"适配器没了"这件事记去哪。
///
/// 一个 `RemotePlugin` 的所有执行器共享同一份（`Arc`），每个 [`ExecSlot`] 各攥一个
/// 克隆——于是断连那一刻，光看 `ConnState.execs` 就够了：谁在跑、是谁的、该撤到哪里去，
/// 三个问题都在这张表里，不必再回头去问内核树。
pub struct ExecOwner {
    /// `manifest.name`：`safety/adapter_lost{plugin}` 里的那个名字，也是分组的键（一条
    /// 连接上可以注册好几个插件）。
    plugin: String,
    /// `Weak`，而且必须是：`Admission` → `CapabilityRegistry` → `RemoteExecutor` →
    /// `Arc<Conn>` → `ConnState` → `ExecSlot` → 这里，攥一个 `Arc` 就把这条链合成了环。
    /// upgrade 不出来说明准入服务本身已经撤了——那时候一次执行都不在跑，没有什么要释放。
    admission: Weak<Admission>,
    /// `safety/adapter_lost` 写去哪。
    log: Arc<WorldLog>,
}

/// 断连即释放（Plan 3 遗留 I4）：把这条连接上还在跑的跨进程执行**当场**撤掉，并给运维
/// 留一条 `safety/adapter_lost`。
///
/// 为什么不能等租约：租约那条路的前提是"执行器可能只是失联，机器人还在动"，所以它保留
/// motion 的位子、过了 `lease_grace_ms` 才强行摘掉登记，中间还要再喊一声停。断连不是
/// 那种情况——那条连接上再也收不到、也发不出任何东西，`Executor::stop` 没有对象，等宽
/// 限期等的是一条永远不会来的消息。而代价是实打实的：motion 的位子只有一个，白占着的
/// 那半秒里，下一个 `walk` 只能靠抢占进场，一次 `stop`/`recover` 会被 `rule-fsm` 拒掉。
///
/// 每条连接**只扫一次**（`ConnState::execs_released`）。这不是优化：两条路都会走到这里
/// ——断连钩子（[`crate::handlers::release_all`]）和还等在 `wait_closed` 上的执行 future
/// ——而多线程 runtime 上它们是真并发。少了这道闸门，两边会各撤下一半执行、各写一条只
/// 列了一半的 `safety/adapter_lost`。
///
/// 闸门**不保证**每次执行都被这一扫覆盖到，也不打算保证：`Conn::mark_closed` 先唤醒
/// `wait_closed`、再跑 `on_close` 钩子，所以在多核上完全可能是钩子拿到闸门之后被调度走，
/// 而那次执行的 future 先跑完——它看到闸门已经被人拿了就直接返回，钩子回来时登记已经被
/// `finish()` 摘掉，于是两边都没撤成，结局退回成 `Failed{"connection lost"}`、
/// `safety/adapter_lost` 一条都没有（Fix round 1，Important #1）。所以每个执行 future 在
/// 自己那条退出路径上**无条件**再撤自己一次（[`release_one_on_disconnect`]），谁先谁后
/// 都得到同一个结局。这一扫负责的是"没有 future 正等着的那些"和把它们合成一条事件。
pub(crate) fn release_executions_on_disconnect(conn: &Arc<Conn>) {
    let victims: Vec<(ProposalId, Arc<ExecOwner>)> = {
        let mut st = conn.state.lock();
        if st.execs_released {
            return;
        }
        st.execs_released = true;
        // 只取快照，**不清表**：摘登记是每次执行自己那个守卫的事（见 `ExecSlotGuard`），
        // 在这里清掉会把 `oneshot::Sender` 一起丢掉，等结果的 future 于是以"通道关了"
        // 收场——而它该看到的是撤销。
        st.execs
            .iter()
            .map(|(id, slot)| (ProposalId(id.clone()), slot.owner.clone()))
            .collect()
    };
    // 按插件分组；`BTreeMap` 只是为了让事件的顺序稳定（同一条连接上的两个插件不该因为
    // 哈希顺序而每次换个先后）。
    let mut lost: BTreeMap<String, (Arc<ExecOwner>, Vec<String>)> = BTreeMap::new();
    for (pid, owner) in victims {
        let Some(admission) = owner.admission.upgrade() else {
            continue;
        };
        // 认领是幂等的：这次执行要是在同一瞬间自己收了尾（结果刚到）、或者已经被别人
        // 撤了（急停、抢占），`cancel_exec` 返回 `false`，这里就当它不存在——一次提案
        // 只能有一个终局。所以 `executions` 列的是**真的被这条路撤掉的**那些。
        if admission.cancel_exec(&pid, CancelReason::Disconnected) {
            lost.entry(owner.plugin.clone())
                .or_insert_with(|| (owner.clone(), Vec::new()))
                .1
                .push(pid.0);
        }
    }
    for (_, (owner, executions)) in lost {
        record_adapter_lost(&owner, executions);
    }
}

/// 撤掉**一次**执行（因为它的适配器断了），真撤下来了就记一条只列它自己的
/// `safety/adapter_lost`。返回"这一下是不是我撤的"。
///
/// 每个跨进程执行的 future 在断连退出路径上都调它，而且**无条件**调——`claim` 本身是
/// 幂等的（已经有终局、或者登记已经不在 → `false`），所以"扫描已经撤过它了"和"它自己
/// 刚收了尾"都只是一次空操作。它存在的理由见 [`release_executions_on_disconnect`] 上的
/// 第二段：结局不能取决于闸门被谁先拿到。
///
/// 为什么这里也写事件：`safety/adapter_lost` 是"一台机器人动到一半，看着它的那一头没了"
/// 这件事在审计里**唯一**的记录——事后没有任何办法把它重建出来。宁可多一条（同一个
/// `collapse_key`，控制台上是同一件事），不能少一条。
fn release_one_on_disconnect(owner: &Arc<ExecOwner>, pid: &ProposalId) -> bool {
    let Some(admission) = owner.admission.upgrade() else {
        return false;
    };
    if !admission.cancel_exec(pid, CancelReason::Disconnected) {
        return false;
    }
    record_adapter_lost(owner, vec![pid.0.clone()]);
    true
}

/// 给运维的那一条：这个插件的这些执行跟着它的连接一起没了。
///
/// `collapse_key`：一个反复重连的适配器会反复走这条路，而控制台上那是同一件事。
fn record_adapter_lost(owner: &Arc<ExecOwner>, executions: Vec<String>) {
    owner.log.append(
        EventDraft::new(
            EventKind::new("safety/adapter_lost").expect("static kind"),
            "rpc",
            json!({"plugin": owner.plugin, "executions": executions}),
        )
        .with_salience(200)
        .with_collapse_key(format!("adapter_lost:{}", owner.plugin)),
    );
}

/// 从 `ConnState.execs` 里摘掉自己那一条——不管这次执行是怎么结束的（收到结果、被取消、
/// 连接断了，还是整个 future 被丢掉）。`Drop` 而不是几条退出路径各写一遍：漏掉任何一条
/// 都是一条永远留在表里的死登记。
///
/// 只摘**自己**那一条：表按 `exec_id` 索引，而 `generation` 是这条登记的世代号。今天
/// `exec_id` 就是 `ProposalId`，而它是 `{trace_id}-p{n}`——`trace_id` 每个 turn 一个、
/// `n` 是 turn 内的计数器（`oeh_kernel::brains`），所以两次执行不可能撞号，无条件
/// `remove` 也是对的。世代号存在的理由正是不让这条结论成为**别处**的不变式：哪天
/// `ProposalId` 的构造换了、或者有人复用一个 id，一个收尾晚了的守卫会把后来者的登记
/// 摘走，而症状是"结果送不到，执行一直等到租约过期"——那种 bug 从现场完全看不出根因。
struct ExecSlotGuard {
    conn: Weak<Conn>,
    exec_id: String,
    generation: u64,
}

impl Drop for ExecSlotGuard {
    fn drop(&mut self) {
        let Some(conn) = self.conn.upgrade() else {
            return;
        };
        // 先出锁再 drop：摘下来的 `ExecSlot` 里有一个 `oneshot::Sender`，丢掉它会唤醒
        // 等在另一头的任务——那种事不该带着 `ConnState` 的锁做。
        let slot = {
            let mut st = conn.state.lock();
            let mine = st
                .execs
                .get(&self.exec_id)
                .is_some_and(|s| s.generation == self.generation);
            if mine {
                st.execs.remove(&self.exec_id)
            } else {
                None
            }
        };
        drop(slot);
    }
}

fn outcome_of(r: CapabilityResult) -> ExecOutcome {
    ExecOutcome {
        ok: r.ok,
        result: r.result,
        error: r.error,
    }
}

fn failed(error: impl Into<String>) -> ExecOutcome {
    ExecOutcome {
        ok: false,
        result: Value::Null,
        error: Some(error.into()),
    }
}

impl Executor for RemoteExecutor {
    /// 租约过期时准入服务叫的那一下：客户端可能只是**失联**（心跳断了），而机器人还在
    /// 动——`ExecCtx` 的取消信号它已经看不见了，所以另外明确地喊一声停。不等回复：一个
    /// 已经失联的执行器不值得内核再等（`Executor::stop` 的契约就是立刻返回）。
    fn stop(&self, pid: &ProposalId) -> BoxFuture<()> {
        let _ = self.conn.notify(
            method::CAPABILITY_CANCEL,
            json!(CapabilityCancel {
                exec_id: pid.0.clone(),
                reason: "lease expired".into(),
            }),
        );
        Box::pin(async {})
    }

    /// 立刻返回一个 future（契约要求），真正的往返都在 future 里。
    ///
    /// 登记 `ExecSlot` **早于**发出请求：客户端完全可以在还没回 `{accepted:true}` 之前
    /// 就把 `capability/result` 发过来（它自己的执行可能比一次往返还快）。先发后登记的
    /// 话，那条结果会撞上一张还没有这次执行的表，被当成陈旧通知丢掉，然后这次执行一直
    /// 等到租约过期——一个只在"客户端太快"时出现的挂起。
    fn execute(&self, call: ExecutionCall, mut ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let conn = self.conn.clone();
        let accept = self.accept;
        let cancel_grace = self.cancel_grace;
        let owner = self.owner.clone();
        let exec_id = call.proposal.proposal_id.0.clone();
        let params = json!(CapabilityExecute {
            exec_id: exec_id.clone(),
            capability: call.capability,
            // 规则链夹过之后的那一份（准入服务保证 `params == proposal.params`）。
            params: call.params,
            proposal: serde_json::to_value(&call.proposal).unwrap_or(Value::Null),
            lease_ttl_s: self.lease_ttl_s,
        });
        Box::pin(async move {
            let (tx, mut rx) = oneshot::channel::<CapabilityResult>();
            let generation = conn.next_exec_gen();
            let _guard = ExecSlotGuard {
                conn: Arc::downgrade(&conn),
                exec_id: exec_id.clone(),
                generation,
            };
            conn.state.lock().execs.insert(
                exec_id.clone(),
                ExecSlot {
                    generation,
                    result: tx,
                    heartbeat: ctx.heartbeat_handle(),
                    owner: owner.clone(),
                },
            );
            // 连接在登记之前就关了：`_guard` 会把它摘掉，但没人会来送结果，所以这里
            // 直接收场，不必去发一条注定发不出去的请求。
            //
            // 断连即释放照走（Fix round 1，Important #1）：这次执行是在那一扫**之后**才
            // 登记的，扫描不可能覆盖到它，而它和别的执行一样是"适配器没了"的受害者——
            // 结局必须是 `Cancelled{"adapter disconnected"}`，不能因为它晚了几微秒就退回
            // 成一次 `Failed`。
            if conn.is_closed() {
                release_one_on_disconnect(&owner, &ProposalId(exec_id.clone()));
                return failed("connection lost");
            }

            match conn
                .request_timeout(method::CAPABILITY_EXECUTE, params, accept)
                .await
            {
                Ok(v) => match serde_json::from_value::<CapabilityAccepted>(v) {
                    Ok(a) if a.accepted => {}
                    Ok(a) => return failed(a.reason.unwrap_or_else(|| "executor refused".into())),
                    Err(e) => return failed(format!("malformed capability/execute reply: {e}")),
                },
                // 断连期间发出去的请求会以 `REFUSED{"connection closed"}` 收场
                // （`Conn::request_timeout` 的语义）。对脑来说那和"执行到一半插件没了"
                // 是同一件事，所以报同一句话：`connection lost` 是这条边界上"客户端
                // 不在了"的**唯一**说法，调用方不该为了同一个原因去匹配两种文本。
                Err(e) if conn.is_closed() => {
                    tracing::debug!(error = %e.message, "capability/execute on a closed connection");
                    return failed("connection lost");
                }
                Err(e) => return failed(format!("capability/execute: {}", e.message)),
            }

            // `biased`（按写下的顺序轮询）不是风格，是判定：一条**已经到手**的结果必须
            // 赢过同一刻才发生的断连/取消，否则一次干净收尾的执行会被报成
            // `connection lost`，而机器人那边其实已经做完了。
            tokio::select! {
                biased;
                r = &mut rx => match r {
                    Ok(r) => outcome_of(r),
                    // 送结果的那一端只可能被 `_guard` 摘走，而它和这个 future 是同生
                    // 共死的——真走到这里说明有人从外面动了 `ConnState.execs`。
                    Err(_) => failed("capability result channel closed"),
                },
                _ = ctx.cancelled() => {
                    let _ = conn.notify(
                        method::CAPABILITY_CANCEL,
                        json!(CapabilityCancel { exec_id, reason: "cancelled".into() }),
                    );
                    // 喊了停还给它一点时间把结果补回来：一个守规矩的执行器会回一条
                    // `{ok:false, error:"cancelled"}`，那比内核自己编一个结局准确。
                    //
                    // 但连接一旦关了就别再等：等的是一条永远不会来的消息，而这段等待
                    // **挂着 FSM**——准入侧要等执行器的 future 解出来才跑 `finish()`，
                    // 而 `finish()` 才是把 ACTING 交还给 STANDING 的那一步。白等
                    // `cancel_grace_ms`，就是让机器人在"没有任何东西在动"的情况下多报
                    // 半秒 ACTING，而那半秒里 `rule-fsm` 会拒掉一次 `recover`/`stop`。
                    tokio::select! {
                        biased;
                        r = &mut rx => match r {
                            Ok(r) => outcome_of(r),
                            Err(_) => failed("cancelled"),
                        },
                        _ = conn.wait_closed() => failed("cancelled"),
                        _ = tokio::time::sleep(cancel_grace) => failed("cancelled"),
                    }
                }
                _ = conn.wait_closed() => {
                    // 断连即释放（I4）。两步，缺一不可：
                    //
                    // 1. 扫一遍这条连接上**所有**在跑的执行——多线程 runtime 上这个
                    //    future 和断连钩子是真并发，谁先到都得把别人的那些也撤掉，否则
                    //    一次执行的结局会取决于调度。幂等，见那个函数。
                    // 2. **无条件**再撤自己一次。第 1 步可能什么都没做（闸门已经被并发的
                    //    钩子拿走了），而那个钩子的快照完全可能是在自己登记之前取的、或者
                    //    它正被调度走——这一下才是"这次执行一定以 `Cancelled` 收场"的保证
                    //    （Fix round 1，Important #1）。
                    release_executions_on_disconnect(&conn);
                    release_one_on_disconnect(&owner, &ProposalId(exec_id.clone()));
                    // 到这里这次执行已经带上 `cancel_reason` 了，结局由它定
                    // （`finish()` 读登记上的那个，不是这里的返回值）——这句话只在
                    // "准入服务本身已经没了"时才会被人看见。
                    failed("connection lost")
                }
            }
        })
    }
}

/// 把 `capability/heartbeat` 盖到对应的那次执行上。认不出 `exec_id` 就是一条陈旧通知
/// （执行已经收尾了）：丢掉，不是错误。
pub fn on_heartbeat(conn: &Arc<Conn>, params: Value) {
    let Ok(p) = serde_json::from_value::<ExecRef>(params) else {
        tracing::debug!("malformed capability/heartbeat");
        return;
    };
    let hb = conn
        .state
        .lock()
        .execs
        .get(&p.exec_id)
        .map(|s| s.heartbeat.clone());
    // 出锁之后再盖戳：`beat()` 只动一个原子，但它是准入侧的代码，不在这把锁的保护范围内。
    if let Some(hb) = hb {
        hb.beat();
    }
}

/// 把 `capability/result` 送给等它的那次执行。同一个 `exec_id` 送第二次就没人接了
/// （第一次已经把登记摘走）——重复的结果被丢掉，而不是覆盖一次已经定下的结局。
pub fn on_result(conn: &Arc<Conn>, params: Value) {
    let Ok(p) = serde_json::from_value::<CapabilityResult>(params) else {
        tracing::debug!("malformed capability/result");
        return;
    };
    let slot = conn.state.lock().execs.remove(&p.exec_id);
    match slot {
        // 出锁之后才 `send`：它会唤醒等结果的那个任务。
        Some(slot) => {
            let _ = slot.result.send(p);
        }
        None => tracing::debug!(exec = %p.exec_id, "capability/result for an unknown execution"),
    }
}

/// `approval/attach` 装上的审批服务：把 `Ask` 判定变成一次 `approval/request` 反向请求。
pub struct RpcApproval {
    /// `Weak`：见模块头上的回收链说明。
    conn: Weak<Conn>,
    name: String,
}

impl RpcApproval {
    pub fn new(conn: &Arc<Conn>, name: &str) -> Self {
        Self {
            conn: Arc::downgrade(conn),
            name: name.to_string(),
        }
    }
}

/// 审批端出任何岔子都是 `Deny`，绝不是"放行"：一个够不着的审批服务等于没有审批服务。
fn deny(by: &str, reason: &str) -> Decision {
    Decision::Deny {
        by: by.to_string(),
        reason: reason.to_string(),
    }
}

impl Approval for RpcApproval {
    fn decide(&self, req: AskRequest) -> BoxFuture<Decision> {
        let name = self.name.clone();
        let Some(conn) = self.conn.upgrade() else {
            return Box::pin(async move { deny(&name, "approval channel closed") });
        };
        let params = json!(ApprovalRequest {
            ask_id: req.proposal.proposal_id.0.clone(),
            proposal: serde_json::to_value(&req.proposal).unwrap_or(Value::Null),
            spec: serde_json::to_value(&req.spec).unwrap_or(Value::Null),
            reason: req.reason,
            trace: serde_json::to_value(&req.trace).unwrap_or(Value::Null),
        });
        Box::pin(async move {
            // 请求在 future **里面**发出，不在 `decide()` 里：一个没被 poll 就被丢掉的
            // `decide()`（调用方自己改主意了）不该已经在操作员的控制台上弹出一个对话框。
            // 和 `RemoteExecutor::execute` 同一个约定——`decide` 本身只做常数级的准备
            // 工作，一次 I/O 都不做。
            //
            // 这次请求的超时由**准入侧**管（`AdmissionConfig::ask_timeout_ms`，它
            // `timeout` 着整个 `decide()`），所以这里用连接自己的默认超时兜底就够了。
            let v = match conn.request(method::APPROVAL_REQUEST, params).await {
                Ok(v) => v,
                // 连接层自己超时了（配置把 `ask_timeout_ms` 调得比连接的默认超时还长时
                // 才会发生）：这是**内核**放弃了等待，不是那个人拒绝了。署上审批者的名字
                // 就是把一次内核侧的超时写成"某某操作员否决了它"——审计里最不该有的那种
                // 句子。理由文本跟 `decide_with_timeout` 的那条保持一个形状。
                Err(e) if e.code == code::TIMEOUT => return deny("kernel", "approval timed out"),
                Err(_) => return deny(&name, "approval channel closed"),
            };
            match serde_json::from_value::<ApprovalDecision>(v) {
                Ok(d) => {
                    // `by` 由**审批端**说了算（控制台里点头的是某个具体的人），空了才退回
                    // attach 时登记的名字；理由原样透传，它会出现在脑拿到的
                    // `Outcome::Refused{reason}` 里。
                    let by = if d.by.trim().is_empty() {
                        name.clone()
                    } else {
                        d.by
                    };
                    match d.decision.as_str() {
                        "allow" => Decision::Allow {
                            by,
                            reason: d.reason,
                        },
                        "deny" => Decision::Deny {
                            by,
                            reason: d.reason,
                        },
                        // 认不出的判定不是"放行"，也不是一个可以转述的拒绝理由。
                        _ => deny(&name, "malformed decision"),
                    }
                }
                Err(_) => deny(&name, "malformed decision"),
            }
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}
