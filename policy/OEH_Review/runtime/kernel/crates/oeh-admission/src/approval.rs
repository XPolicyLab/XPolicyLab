//! 审批服务：规则链把判定收紧到 `Ask` 之后，谁来把 `Ask` 变成最终的 `Allow`/`Deny`。
//!
//! `Approval` 是这件事的合约：给一次请求（`AskRequest`），异步产出一个 `Decision`。
//! 这个模块提供两个实现——`AutoApproval`（测试/无人值守场景，固定放行或拒绝）和
//! `ChannelApproval`（把请求转发给外部消费者：控制台、UI，通过 `mpsc` + `oneshot`
//! 应答）——以及 `decide_with_timeout`，给"没有配置审批服务"和"审批超时"两种边界
//! 情况一个统一的、内核自己给出的 `Deny`。

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::proposal::{BoxFuture, Proposal};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::capability::CapabilitySpec;
use crate::verdict::RuleTrace;

/// 一次审批请求：触发它的提案、被提案能力的静态描述、规则链给出的原因，以及完整的
/// 规则 trace（审批方可能想看到"为什么会走到这一步"）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AskRequest {
    pub proposal: Proposal,
    pub spec: CapabilitySpec,
    pub reason: String,
    pub trace: Vec<RuleTrace>,
}

/// 审批结果。带标签序列化成 `{"decision": "allow", "by": .., "reason": ..}` 这样的形式，
/// `by` 记录是谁给出的这个决定（比如某个 operator 的名字，或者 `"kernel"`/`"approval-auto"`
/// 这样的服务自己）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "lowercase")]
pub enum Decision {
    Allow { by: String, reason: String },
    Deny { by: String, reason: String },
}

/// 审批服务实现的合约：给定一次请求，产出一个 future。`decide` 本身必须立刻返回
/// （不阻塞调用它的准入循环），真正的等待发生在返回的 future 里。
pub trait Approval: Send + Sync {
    fn decide(&self, req: AskRequest) -> BoxFuture<Decision>;
    fn name(&self) -> &str;
}

/// 无人值守的审批：固定放行或拒绝，不看请求内容。用于测试，或者"暂时没有人工审批,
/// 先都放行/都拒绝"这样的配置。
pub struct AutoApproval {
    pub allow: bool,
}

impl Approval for AutoApproval {
    fn decide(&self, _req: AskRequest) -> BoxFuture<Decision> {
        let by = self.name().to_string();
        let allow = self.allow;
        Box::pin(async move {
            if allow {
                Decision::Allow {
                    by,
                    reason: "auto-approved".into(),
                }
            } else {
                Decision::Deny {
                    by,
                    reason: "auto-denied".into(),
                }
            }
        })
    }

    fn name(&self) -> &str {
        "approval-auto"
    }
}

/// 把审批请求转发给外部消费者的审批服务：`decide` 把 `(AskRequest, oneshot::Sender<Decision>)`
/// 发到内部的 `mpsc` 通道，`ChannelApproval::new` 返回的接收端交给测试/控制台，它们
/// `recv()` 拿到请求，处理完之后用带的 `oneshot::Sender` 送回答案。
///
/// 如果消费者已经把接收端丢了（没人再读这个通道），或者拿到请求之后把 `oneshot::Sender`
/// 丢了没应答（比如控制台崩了），`decide` 不会挂起——两种情况都解析成
/// `Deny{by: "approval-channel", reason: "no answer"}`，因为准入服务不能因为审批端消失
/// 就永远卡住。
pub struct ChannelApproval {
    tx: mpsc::Sender<(AskRequest, oneshot::Sender<Decision>)>,
}

impl ChannelApproval {
    pub fn new() -> (
        Arc<Self>,
        mpsc::Receiver<(AskRequest, oneshot::Sender<Decision>)>,
    ) {
        let (tx, rx) = mpsc::channel(16);
        (Arc::new(Self { tx }), rx)
    }
}

impl Approval for ChannelApproval {
    fn decide(&self, req: AskRequest) -> BoxFuture<Decision> {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (answer_tx, answer_rx) = oneshot::channel();
            if tx.send((req, answer_tx)).await.is_err() {
                // 没人在另一头读这个通道了：不会有人应答，立刻拒绝。
                return Decision::Deny {
                    by: "approval-channel".into(),
                    reason: "no answer".into(),
                };
            }
            answer_rx.await.unwrap_or(Decision::Deny {
                by: "approval-channel".into(),
                reason: "no answer".into(),
            })
        })
    }

    fn name(&self) -> &str {
        "approval-channel"
    }
}

/// 给一次审批请求一个带超时的、内核自己兜底的结果：
/// - 没有配置审批服务（`approval` 是 `None`）：直接 `Deny{by: "kernel", reason: "no approval provider"}`；
/// - `decide()` 这次调用本身 panic 了：`Deny{by: <审批服务名>, reason: "approval panicked"}`；
/// - 配置了，但在 `timeout` 之内没等到结果：`Deny{by: "kernel", reason: "approval timeout"}`；
/// - 否则原样返回审批服务给出的 `Decision`。
pub async fn decide_with_timeout(
    approval: Option<Arc<dyn Approval>>,
    req: AskRequest,
    timeout: Duration,
) -> Decision {
    let Some(approval) = approval else {
        return Decision::Deny {
            by: "kernel".into(),
            reason: "no approval provider".into(),
        };
    };
    // `decide()` 返回的是 future，但**这次调用本身**也是第三方代码：它可能在造出
    // future 之前就 panic。那一下不在 `timeout` 的 future 里，unwind 会直接穿出整条
    // 准入流水线。审批炸了 = 拒绝，和"没人应答"一个待遇。
    let Ok(fut) = catch_unwind(AssertUnwindSafe(|| approval.decide(req))) else {
        return Decision::Deny {
            by: approval.name().to_string(),
            reason: "approval panicked".into(),
        };
    };
    match tokio::time::timeout(timeout, fut).await {
        Ok(decision) => decision,
        Err(_) => Decision::Deny {
            by: "kernel".into(),
            reason: "approval timeout".into(),
        },
    }
}
