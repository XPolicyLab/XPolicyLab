//! `brain/attach` · `brain/detach` · `brain/cursor` · `brain/wake_policy/update` ·
//! `brain/inject`，以及把内核的 turn 变成通知的 [`RpcBrainSink`]。
//!
//! 一个跨进程的脑就是"一条连接 + 一次 `Brains::attach`"：attach 成功之后这条连接**拥有**
//! 那个脑，断连即 detach（见 [`crate::handlers::ensure_cleanup_hook`]）。这条归属关系是
//! `brain/detach` 只允许拆自己那一份的理由，也是重连之后同名 attach 必然能成功的理由。

use std::collections::hash_map::Entry;
use std::sync::{Arc, Weak};

use oeh_kernel::observation::Observation;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::{BrainSink, BrainStatus, BrainsError, TurnHandle};
use oeh_proto::rpc::types::{
    BrainAttach, BrainAttachResult, BrainCursorResult, BrainInject, BrainInjectResult, BrainRef,
    BrainSteerNotice, BrainTurn, Empty, WakePolicyUpdate,
};
use oeh_proto::rpc::{code, method, RpcError};
use serde_json::Value;

use crate::conn::{BrainSlot, Conn};
use crate::handlers::{
    ensure_cleanup_hook, internal, invalid_params, not_attached, ok, refused, release_if_closed,
};
use crate::router::{parse_params, Services};

/// 把内核发起的 turn/steer/interrupt 回调变成 k→c 通知。
///
/// 持 `Weak<Conn>`，不是 `Arc`：`Brains` 攥着 `Arc<dyn BrainSink>`，而这条连接的
/// `ConnState` 里又攥着撤销那次 attach 的 `Disposer`——两边都用强引用就是一个环，连接
/// 关了也回收不掉。`Weak` 断了（连接已经彻底没了）就什么也不做：真正的清理由 `on_close`
/// 钩子负责。
pub struct RpcBrainSink {
    conn: Weak<Conn>,
    brain_id: String,
}

impl RpcBrainSink {
    pub fn new(conn: &Arc<Conn>, brain_id: &str) -> Self {
        Self {
            conn: Arc::downgrade(conn),
            brain_id: brain_id.to_string(),
        }
    }
}

impl BrainSink for RpcBrainSink {
    /// 回调跑在内核的路由任务上，必须非阻塞：只做"把句柄放进 `ConnState`"（一小段
    /// `parking_lot` 临界区）和一次 `conn.notify`（`try_send`），中间不 `.await`、不
    /// append、不做 I/O。
    ///
    /// 顺序是先存句柄、后发通知，不能反过来：客户端收到 `brain/turn` 之后随时可能回一条
    /// `turn/propose`，那条请求在另一个任务上查的正是这张表，通知先出去就会撞上一个
    /// "turn 不存在" 的窗口（`NOT_ATTACHED`）。
    ///
    /// 存新句柄的同时把这个脑**旧**的句柄清掉（`retain`）。一个脑同一时刻只有一个
    /// turn，所以内核开出第 `n` 个 turn 就意味着前一个已经结束了——问题在于"结束"有三条
    /// 路，只有一条会经过 `turn/done`：被 `turn/interrupt` 打断，或者被内核的 turn 超时
    /// 扫描强行收掉（`sweep_timeouts`，根本不通知 sink）。少了这一行，一个跑一整班的
    /// 跨进程脑每被打断/超时一次就往这张表里留一个死句柄，只增不减，而且是连接自己的
    /// 状态，别人回收不了。
    ///
    /// 脑已经不在表里（`brain/detach` 刚把它摘掉，而这次回调是在那之前就从内核的路由
    /// 任务里发出来的）：丢掉句柄，什么也不做。用 `get_mut` 而不是 `entry().or_default()`
    /// 正是为此——把槽位重新变出来会留下一个 `detach: None` 的孤儿，"槽位存在"从此不再
    /// 等于"这条连接拥有这个脑"，而模块里别处正是这么读这张表的。没 `done` 的那个 turn
    /// 由内核的超时机制收尾。
    fn on_turn(&self, obs: Observation, turn: TurnHandle) {
        let Some(conn) = self.conn.upgrade() else {
            return;
        };
        let n = turn.turn;
        let trace_id = turn.trace_id.clone();
        {
            let mut st = conn.state.lock();
            let Some(slot) = st.brains.get_mut(&self.brain_id) else {
                tracing::debug!(
                    brain = %self.brain_id,
                    turn = n,
                    "turn for a brain this connection no longer holds; dropping the handle"
                );
                return;
            };
            slot.turns.retain(|k, _| *k == n);
            slot.turns.insert(n, turn);
        }
        let params = serde_json::to_value(BrainTurn {
            brain_id: self.brain_id.clone(),
            turn: n,
            trace_id,
            observation: serde_json::to_value(&obs).unwrap_or(Value::Null),
        })
        .unwrap_or(Value::Null);
        let _ = conn.notify(method::BRAIN_TURN, params);
    }

    /// Steer 投进**当前** turn，不开新 turn——句柄表不变，只发一条通知。
    fn steer(&self, obs: Observation) {
        let Some(conn) = self.conn.upgrade() else {
            return;
        };
        let params = serde_json::to_value(BrainSteerNotice {
            brain_id: self.brain_id.clone(),
            observation: serde_json::to_value(&obs).unwrap_or(Value::Null),
        })
        .unwrap_or(Value::Null);
        let _ = conn.notify(method::BRAIN_STEER, params);
    }

    /// 打断只是一个信号：被打断的那个 turn 由内核 `finish_turn` 收尾，紧接着的
    /// `on_turn` 会带来新的句柄，并在那时把这个死句柄清掉（见 `on_turn` 的 `retain`）。
    /// 这里不动句柄表：这一刻还不知道内核会不会重新唤醒，而陈旧的句柄留着也无害
    /// （`TurnHandle::propose` 自己会认出 turn 已经不是当前的那个）。
    fn interrupt(&self) {
        let Some(conn) = self.conn.upgrade() else {
            return;
        };
        let params = serde_json::to_value(BrainRef {
            brain_id: self.brain_id.clone(),
        })
        .unwrap_or(Value::Null);
        let _ = conn.notify(method::BRAIN_INTERRUPT, params);
    }
}

/// `wake_policy` 缺省（字段没给或者是 `null`）→ [`WakePolicy::default`]；给了但解不出来
/// → `INVALID_PARAMS`（`WakePolicy` 是 `deny_unknown_fields`，写错一个键会在这里响，
/// 而不是变成一条悄悄被忽略的规则）。
fn policy_of(v: Value) -> Result<WakePolicy, RpcError> {
    if v.is_null() {
        return Ok(WakePolicy::default());
    }
    serde_json::from_value(v).map_err(|e| invalid_params(format_args!("wake_policy: {e}")))
}

fn attach_error(e: BrainsError) -> RpcError {
    match e {
        // 一个 brain_id 同一时刻只能有一个 Provider：抢一个已经 attach 的脑是策略性
        // 拒绝，不是参数错误。
        BrainsError::AlreadyAttached(id) => refused(format!("brain `{id}` is already attached")),
        BrainsError::InvalidBrainId(id) => invalid_params(format_args!("invalid brain_id `{id}`")),
        other => internal(other),
    }
}

fn status_name(s: BrainStatus) -> &'static str {
    match s {
        BrainStatus::Idle => "idle",
        BrainStatus::Running => "running",
        BrainStatus::Maintenance => "maintenance",
    }
}

/// attach 一个脑：这条连接从此是它的 Provider，内核的 turn 从这一刻起以 `brain/turn`
/// 通知的形式到达。
///
/// `BrainSlot` 的位子在 `attach` **之前**就留好：attach 一返回，内核的路由任务就可能立刻
/// 开出第一个 turn 并回调 `on_turn`，而 `on_turn` 只认已经存在的槽位（它不能凭空建一个，
/// 否则 `brain/detach` 刚摘掉的槽位会被一次在途的回调重新变出来）。attach 失败时把这个
/// 占位撤掉——但只撤我们自己放的那一个（`detach` 仍是 `None`），不能误伤这条连接本来就
/// 有的那份（`AlreadyAttached` 完全可能就是它自己）。
///
/// **绝不丢掉一个存不进去的 `Disposer`。** `Brains::attach` 要开会话日志文件，慢到足够让
/// 一条并发的 `brain/detach`、或者一次断连清理，在它返回之前把占位撤走；这时候
/// `get_mut` 是 `None`，而 `Disposer` 是 `FnOnce`——丢掉它不等于执行它，于是 `Brains` 里
/// 那份 attach 永远留着：这个 `brain_id` 在整个进程生命期内都再也 attach 不上
/// （全是 `AlreadyAttached`），它的 `SessionLog` 句柄留着，路由照样把 turn 派给一个指向
/// 死连接的 sink，而 `robot/health` 一切正常。所以存不进去就**当场执行**（出锁之后），
/// 并且如实告诉调用方这次 attach 没有留下任何东西。
///
/// result 的 `cursor` 是一次**实时**读取，不是 attach 那一刻定下的挂载点：
/// `Brains::attach` 把挂载点定在 `log.next_seq() - 1` 但只返回 `Disposer`，所以这里只能
/// 回头问。两者之间路由任务完全可能已经把这个脑叫醒并推进过游标，于是客户端拿到的是一个
/// **不早于**挂载点的值。对"我从哪儿开始看"这个用途它仍然是安全的方向（只会漏掉刚刚已经
/// 以 `brain/turn` 送到手上的事件，不会漏掉没送过的），但它不是挂载点本身。要精确值得让
/// `Brains::attach` 一并返回 `start`。
pub fn attach(s: &Services, conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: BrainAttach = parse_params(&params)?;
    let policy = policy_of(p.wake_policy)?;
    ensure_cleanup_hook(conn);
    let placed = match conn.state.lock().brains.entry(p.brain_id.clone()) {
        Entry::Occupied(_) => false,
        Entry::Vacant(v) => {
            v.insert(BrainSlot::default());
            true
        }
    };
    let sink = Arc::new(RpcBrainSink::new(conn, &p.brain_id));
    match s.brains.attach(&p.brain_id, policy, sink) {
        Ok(disposer) => {
            let orphan = {
                let mut st = conn.state.lock();
                match st.brains.get_mut(&p.brain_id) {
                    Some(slot) => {
                        slot.detach = Some(disposer);
                        None
                    }
                    // 占位在 attach 途中被撤走了：连接正在关，或者客户端在不等回复的
                    // 情况下发了 `brain/detach`。
                    None => Some(disposer),
                }
            };
            if let Some(d) = orphan {
                // 出锁之后跑：它会 append `brain/detach`，append 会同步跑观察者。
                d();
                return Err(refused("detached during attach"));
            }
        }
        Err(e) => {
            if placed {
                let mut st = conn.state.lock();
                if st
                    .brains
                    .get(&p.brain_id)
                    .is_some_and(|s| s.detach.is_none())
                {
                    st.brains.remove(&p.brain_id);
                }
            }
            return Err(attach_error(e));
        }
    }
    release_if_closed(conn);
    // 连接从此按脑的名字出现在日志、`event/emit` 的 `source`（`rpc:<name>`）和
    // `plugin/register` 里——一条"conn-3"的记录没人看得出是谁。
    conn.set_name(&p.brain_id);
    let cursor = s.brains.cursor(&p.brain_id).unwrap_or(0);
    ok(BrainAttachResult { cursor })
}

/// 只拆**这条连接**attach 的脑。别的连接的脑不在这张表里 → `NOT_ATTACHED`：一个能拆
/// 别人的脑的 `brain/detach` 等于把任何客户端都变成了操作员。
///
/// 槽位在、但 `detach` 还是 `None`（一次**占位**，也就是同一条连接上还没返回的
/// `brain/attach`）→ `REFUSED`，而且**不撤**这个槽位。把占位当成"已经 attach 了"来拆是
/// 两个错一起犯：它什么也没拆（没有 `Disposer` 可跑），却回了 `Ok`，同时把那次在途的
/// attach 的落脚点抽掉——于是 attach 拿着一个存不进去的 `Disposer`。客户端不等回复就
/// 连发 attach + detach（一个再正常不过的关机路径）就会走到这里。
pub fn detach(conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: BrainRef = parse_params(&params)?;
    let slot = {
        let mut st = conn.state.lock();
        match st.brains.get(&p.brain_id) {
            None => None,
            // 占位：留着，让在途的那次 attach 有地方落。
            Some(slot) if slot.detach.is_none() => return Err(refused("attach in progress")),
            Some(_) => st.brains.remove(&p.brain_id),
        }
    };
    let Some(slot) = slot else {
        return Err(not_attached(format_args!("brain `{}`", p.brain_id)));
    };
    // 出锁之后才跑：`Disposer` 会 append `brain/detach`，append 会同步跑观察者。
    if let Some(d) = slot.detach {
        d();
    }
    ok(Empty {})
}

/// 游标和状态。读的是 `Brains`（不是这条连接的表）：控制台要看的是"那个脑现在到哪儿了"，
/// 不论它挂在谁的连接上。
pub fn cursor(s: &Services, params: Value) -> Result<Value, RpcError> {
    let p: BrainRef = parse_params(&params)?;
    let (cursor, status) = (s.brains.cursor(&p.brain_id), s.brains.status(&p.brain_id));
    match (cursor, status) {
        (Some(cursor), Some(status)) => ok(BrainCursorResult {
            cursor,
            status: status_name(status).to_string(),
        }),
        _ => Err(not_attached(format_args!("brain `{}`", p.brain_id))),
    }
}

/// 就地换掉一个脑的 `WakePolicy`。只影响这次调用之后**路由**的事件：已经在路由队列里
/// 的那些按旧 policy 走。
pub fn update_wake_policy(s: &Services, params: Value) -> Result<Value, RpcError> {
    let p: WakePolicyUpdate = parse_params(&params)?;
    let policy = policy_of(p.wake_policy)?;
    s.brains
        .update_policy(&p.brain_id, policy)
        .map_err(brains_op_error)?;
    ok(Empty {})
}

/// 往一个脑的上下文里塞一句话：`brain/inject{brain_id, text}` → `{seq}`。
///
/// 和 `turn/steer` 是一对——都是定向事件，走同一条路（见 [`oeh_kernel::Brains::inject`]）
/// ——差别只在脑那一侧：注入进的是 `Lane::Inject`（捎带上的上下文），steer 进的是
/// `Lane::Steer`（下一步照它办）。
///
/// **任何连接都能调**，和 `turn/steer`/`turn/interrupt` 一样：注入的典型调用方是控制台或
/// 运维脚本，它们自己不 attach 任何脑。身份不由调用方决定：payload 里的 `source` 是
/// `rpc:<这条连接的名字>`，和 `event/emit` 同一条规矩。
///
/// 错误分两类，和边上的 `turn/steer` 逐字相同：点不到那个脑 → `NOT_ATTACHED`（通过
/// [`brains_op_error`]），文本空或超限 → `INVALID_PARAMS`。同一个问题在两个相邻的方法上
/// 不能有两种答案。
pub fn inject(s: &Services, conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: BrainInject = parse_params(&params)?;
    let source = format!("rpc:{}", conn.name());
    let seq = s
        .brains
        .inject(&p.brain_id, &p.text, &source)
        .map_err(|e| match e {
            // 只有这一种是"调用方写错了"；剩下的（`NotAttached`，以及今天不会从 `inject`
            // 出来但明天可能会的别的变体）走和 `turn/steer` 同一条映射。
            BrainsError::InvalidParams(_) => invalid_params(format_args!("{e}")),
            other => brains_op_error(other),
        })?;
    ok(BrainInjectResult { seq })
}

/// `steer`/`interrupt`/`update_policy` 共用：这三个都只会因为"脑没 attach"而失败。
pub(crate) fn brains_op_error(e: BrainsError) -> RpcError {
    match e {
        BrainsError::NotAttached(id) => not_attached(format_args!("brain `{id}`")),
        other => RpcError::new(code::INTERNAL, other.to_string()),
    }
}
