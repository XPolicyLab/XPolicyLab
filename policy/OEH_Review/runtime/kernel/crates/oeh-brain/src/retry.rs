//! 重试与 fallback 瀑布：一次 step 的模型调用失败之后发生什么。
//!
//! ```text
//! runtime 0（主）：试 1 → 失败 → 等 250 ms → 试 2 → 失败 → 等 500 ms → 试 3 → 失败
//!   └─ fallback ─→ runtime 1：试 1 → …
//!                              └─ 都用尽 → 把最后一个错还给 turn 循环（fail closed）
//! ```
//!
//! 三条规则，每条都有一个反例在背后：
//!
//! 1. **不是所有失败都值得重试。** `MissingKey` 重试多少次都还是没有 key，4xx 重试多少
//!    次都还是同一个被拒的请求，解不开的回答重试多少次都还是解不开——这三种直接换
//!    runtime。5xx、超时、连接错才退避重试。
//! 2. **退避是指数的、有上限的**（250 ms → 500 ms → … → 4 s）。上限存在的理由是 turn
//!    有超时：等到第 5 次重试时 turn 早就被内核收掉了，那次等待纯属浪费。
//! 3. **链不记忆。** 每个 step 都从主 runtime 重新开始——降级是这一次调用的事，不是这
//!    个脑的新身份。主 provider 恢复了就该立刻用回去（规格 §7.2："下一 turn 尝试恢复
//!    主 provider"）。
//!
//! `sleep` 是注入的：测试把它换成"立刻返回"，于是退避的**时长**是被断言的数字而不是被
//! 等出来的墙钟。生产路径传的是 `tokio::time::sleep`。

use std::sync::Arc;
use std::time::Duration;

use crate::model::{BoxFuture, EventStream, Model, ModelError, ModelRequest};

/// 退避参数。默认值是规格的数：最多 3 次、250 ms 起步、封顶 4 s。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_ms: u64,
    pub cap_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_attempts: 3,
            base_ms: 250,
            cap_ms: 4_000,
        }
    }
}

impl RetryPolicy {
    /// 第 `attempt` 次失败之后等多久（`attempt` 从 1 起）。
    fn backoff(&self, attempt: u32) -> u64 {
        let shift = attempt.saturating_sub(1).min(31);
        let d = self.base_ms.saturating_mul(1u64 << shift);
        d.min(self.cap_ms)
    }
}

/// 瀑布上发生的一件事。调用方把它翻成世界日志里的 `brain/request-error` /
/// `brain/retry` / `brain/fallback`（见 [`crate::native`]）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetryEvent {
    /// 一次失败。`model` 是 [`Model::name`]（`<provider>/<model>`）。
    Error {
        model: String,
        attempt: u32,
        reason: String,
    },
    /// 决定重试，并且要等 `delay_ms`。
    Retry { attempt: u32, delay_ms: u64 },
    /// 换 runtime。
    Fallback { from: String, to: String },
}

/// 一个 step 的模型调用：重试 + fallback，成功就把流交出去。
///
/// `sleep` 与 `on_event` 要 `Send + Sync`：这个 future 活在 turn 任务里（`tokio::spawn`），
/// 而一个持着 `&dyn Fn` 的 future 只有在那个 `Fn` 是 `Sync` 时才 `Send`。
pub async fn complete_with_retry(
    chain: &[Arc<dyn Model>],
    req: ModelRequest,
    policy: &RetryPolicy,
    sleep: &(dyn Fn(Duration) -> BoxFuture<()> + Send + Sync),
    on_event: &(dyn Fn(RetryEvent) + Send + Sync),
) -> Result<EventStream, ModelError> {
    complete_with_retry_observed(chain, req, policy, sleep, on_event, &|_, _| {}).await
}

/// The observer runs immediately before each provider invocation, never when a
/// retry is merely planned (a deadline can expire during backoff).
pub async fn complete_with_retry_observed(
    chain: &[Arc<dyn Model>],
    req: ModelRequest,
    policy: &RetryPolicy,
    sleep: &(dyn Fn(Duration) -> BoxFuture<()> + Send + Sync),
    on_event: &(dyn Fn(RetryEvent) + Send + Sync),
    on_attempt: &(dyn Fn(&str, u32) + Send + Sync),
) -> Result<EventStream, ModelError> {
    // 0 当 1：`max_attempts: 0` 是配置写坏了，不是"一次都不试"——那样这个脑每次唤醒都
    // 只会记一条 request-error，而日志里没有任何线索说是配置的问题。
    let max_attempts = policy.max_attempts.max(1);
    let mut last = ModelError::Transport("no model in the fallback chain".into());
    for (i, model) in chain.iter().enumerate() {
        for attempt in 1..=max_attempts {
            on_attempt(&model.name(), attempt);
            let err = match model.complete(req.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(e) => e,
            };
            on_event(RetryEvent::Error {
                model: model.name(),
                attempt,
                reason: err.to_string(),
            });
            let retryable = retryable(&err);
            last = err;
            if !retryable || attempt == max_attempts {
                break;
            }
            let delay_ms = policy.backoff(attempt);
            on_event(RetryEvent::Retry { attempt, delay_ms });
            sleep(Duration::from_millis(delay_ms)).await;
        }
        if let Some(next) = chain.get(i + 1) {
            on_event(RetryEvent::Fallback {
                from: model.name(),
                to: next.name(),
            });
        }
    }
    Err(last)
}

/// 再试一次有没有可能不一样。
fn retryable(e: &ModelError) -> bool {
    match e {
        // 5xx = 对面的问题，通常是暂时的。4xx 一般是我们的请求，重试也还是同一个请求——
        // 除了两个和时间有关的：429（限流，真实环境里最典型的"再等等就好"）和 408
        // （对面自己说超时了）。`Retry-After` 暂不读，退避照旧。
        ModelError::Http { status, .. } => *status >= 500 || *status == 429 || *status == 408,
        ModelError::Timeout | ModelError::Transport(_) => true,
        ModelError::MissingKey(_) | ModelError::Protocol(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_backoff_doubles_until_the_cap() {
        let p = RetryPolicy::default();
        assert_eq!(
            (1..=6).map(|a| p.backoff(a)).collect::<Vec<_>>(),
            vec![250, 500, 1000, 2000, 4000, 4000]
        );
    }

    #[test]
    fn a_huge_attempt_number_does_not_overflow() {
        let p = RetryPolicy::default();
        assert_eq!(p.backoff(u32::MAX), p.cap_ms);
    }

    #[test]
    fn only_the_failures_that_could_go_differently_are_retried() {
        assert!(retryable(&ModelError::Http {
            status: 503,
            body: String::new()
        }));
        assert!(retryable(&ModelError::Timeout));
        assert!(retryable(&ModelError::Transport("reset".into())));
        // 两个和时间有关的 4xx 也重试。
        assert!(retryable(&ModelError::Http {
            status: 429,
            body: String::new()
        }));
        assert!(retryable(&ModelError::Http {
            status: 408,
            body: String::new()
        }));
        // 别的 4xx 不：重试多少次都还是同一个被拒的请求。
        assert!(!retryable(&ModelError::Http {
            status: 400,
            body: String::new()
        }));
        assert!(!retryable(&ModelError::Http {
            status: 404,
            body: String::new()
        }));
        assert!(!retryable(&ModelError::MissingKey("K".into())));
        assert!(!retryable(&ModelError::Protocol("half".into())));
    }
}
