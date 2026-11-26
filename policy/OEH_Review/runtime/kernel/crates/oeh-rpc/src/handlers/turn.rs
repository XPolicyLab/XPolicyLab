//! `turn/steer` · `turn/interrupt` · `turn/propose` · `turn/say` · `turn/ask_human` ·
//! `turn/done`。
//!
//! 两半：`steer`/`interrupt` 是**冲着一个脑**去的定向操作（控制台也会用，所以任何连接
//! 都能调）；`propose`/`say`/`ask_human`/`done` 是**一个 turn 之内**的动作，只有持有那个
//! turn 句柄的连接（也就是这个脑的 Provider）才做得了。
//!
//! `say` / `ask_human` 没有旁路：它们和 `walk` 一样是提案，走同一条准入链
//! （规格 §5.1）。这里唯一的差别是能力名和 params 的形状是写死的。

use std::sync::Arc;

use oeh_kernel::TurnHandle;
use oeh_proto::rpc::types::{
    BrainRef, Empty, TurnAskHuman, TurnDone, TurnPropose, TurnProposeResult, TurnSay, TurnSteer,
};
use oeh_proto::rpc::RpcError;
use serde_json::{json, Value};

use crate::conn::Conn;
use crate::handlers::brain::brains_op_error;
use crate::handlers::{not_attached, ok};
use crate::router::{parse_params, Services};

/// 定向 steer。内核记一条 `brain/steer{brain_id,text}`（source `"brains"`），那条事件
/// 是唯一被当作定向的形状——`event/emit` 伪造不出来（它的 source 永远是 `rpc:<name>`）。
pub fn steer(s: &Services, params: Value) -> Result<Value, RpcError> {
    let p: TurnSteer = parse_params(&params)?;
    s.brains
        .steer(&p.brain_id, &p.text)
        .map_err(brains_op_error)?;
    ok(Empty {})
}

pub fn interrupt(s: &Services, params: Value) -> Result<Value, RpcError> {
    let p: BrainRef = parse_params(&params)?;
    s.brains.interrupt(&p.brain_id).map_err(brains_op_error)?;
    ok(Empty {})
}

/// 取出这条连接上 `brain_id` 的第 `turn` 个 turn 的句柄（克隆一份）。
///
/// **克隆而不是借**：接下来要 `.await` 一次完整的准入链（规则、审批、执行，可以是秒级），
/// 而这张表的锁是 `parking_lot` 的——带着它跨 `.await` 会把同一条连接上所有别的方法、
/// 以及内核路由任务里的 `RpcBrainSink::on_turn` 一起堵住。克隆不放大任何权力：句柄自己
/// 带 epoch+turn 双重检查。
fn handle_of(conn: &Arc<Conn>, brain_id: &str, turn: u64) -> Result<TurnHandle, RpcError> {
    conn.state
        .lock()
        .brains
        .get(brain_id)
        .and_then(|slot| slot.turns.get(&turn).cloned())
        .ok_or_else(|| not_attached(format_args!("turn {turn} of brain `{brain_id}`")))
}

/// 一次提案的公共尾段。`proposal_id` 在提案发出的那一刻就拿到（`propose_with_id`），
/// 不用等结局：调用方据此把自己这次调用和世界日志里的 `act/*` 对上号。
async fn propose_with(
    conn: &Arc<Conn>,
    brain_id: &str,
    turn: u64,
    capability: &str,
    params: Value,
) -> Result<Value, RpcError> {
    let handle = handle_of(conn, brain_id, turn)?;
    let (proposal_id, fut) = handle.propose_with_id(capability, params);
    let outcome = fut.await;
    ok(TurnProposeResult {
        proposal_id: proposal_id.0,
        outcome: serde_json::to_value(&outcome).unwrap_or(Value::Null),
    })
}

pub async fn propose(conn: Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: TurnPropose = parse_params(&params)?;
    propose_with(&conn, &p.brain_id, p.turn, &p.capability, p.params).await
}

pub async fn say(conn: Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: TurnSay = parse_params(&params)?;
    propose_with(&conn, &p.brain_id, p.turn, "say", json!({ "text": p.text })).await
}

pub async fn ask_human(conn: Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: TurnAskHuman = parse_params(&params)?;
    propose_with(
        &conn,
        &p.brain_id,
        p.turn,
        "ask_human",
        json!({ "question": p.question }),
    )
    .await
}

/// 结束一个 turn。句柄被**取走**（`remove`）：`TurnHandle::done` 消费 `self`，而再调一次
/// 也不该凭空多出一条 `brain/turn_end`——第二次调用因此是 `NOT_ATTACHED`，这正是调用方
/// 分得清"我刚关掉了它"和"它早就不在了"的方式。
///
/// 出锁之后才 `done()`：它会 append `brain/turn_end`，append 会同步跑观察者，观察者会
/// 回头 `conn.notify`。
pub fn done(conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: TurnDone = parse_params(&params)?;
    let handle = {
        let mut st = conn.state.lock();
        st.brains
            .get_mut(&p.brain_id)
            .and_then(|slot| slot.turns.remove(&p.turn))
    };
    let Some(handle) = handle else {
        return Err(not_attached(format_args!(
            "turn {} of brain `{}`",
            p.turn, p.brain_id
        )));
    };
    handle.done(&p.reason);
    ok(Empty {})
}
