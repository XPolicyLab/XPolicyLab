//! 提案接缝：脑想做什么 → 内核转交给准入服务 → 结果回到脑，两端都在会话日志里留痕。
//!
//! 内核本身**不**判断能不能做（那是 Plan 2b 的准入链的事），它只保证：提案带着发起它的
//! turn 的身份（`brain_id`/`turn`/`trace_id`），陈旧的 turn 提不出提案，以及每一次提案和
//! 它的结局都能在会话日志里对上号。

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 一次提案的标识：`{trace_id}-p{n}`，`n` 是该 turn 内的序号（从 1 起）。
/// 把 trace_id 编进来，读日志的人不用查表就知道它属于哪个 turn。
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProposalId(pub String);

/// 脑交给准入服务的一次请求。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub proposal_id: ProposalId,
    pub brain_id: String,
    pub turn: u64,
    pub trace_id: String,
    pub capability: String,
    pub params: Value,
}

/// 一次提案的结局。`Refused` 是还没开始就被挡下（含内核自己挡下的陈旧 turn / 无服务），
/// `Failed` 是执行中出错，`Preempted` 是被更高优先级抢走，`Cancelled` 是主动撤销。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Refused { reason: String },
    Executed { result: Value },
    Failed { error: String },
    Preempted,
    Cancelled { reason: String },
}

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// 准入服务实现的接口。`propose` 必须立刻返回一个 future（不阻塞调用它的 sink 回调），
/// 真正的准入判定和执行发生在 future 里。
pub trait Proposer: Send + Sync {
    fn propose(&self, p: Proposal) -> BoxFuture<Outcome>;
}
