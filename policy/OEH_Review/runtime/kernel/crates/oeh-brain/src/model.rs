//! 模型合约：四种 wire（`chat_completions` / `anthropic_messages` / `responses` /
//! `realtime`，规格 §7.2）共用的同一套请求、事件与错误。
//!
//! 这里只定义"一次模型调用长什么样、回来的是什么"，不含任何 HTTP：传输实现（Task 3）
//! 各自把这套结构翻译成厂商的线格式，再把厂商的流翻回 [`ModelEvent`]。原生 turn 循环
//! （Task 2）因此只认识这个模块，换厂商不改循环。
//!
//! 事件是**流式**的：`complete` 先还一个 [`EventStream`]，文本增量、工具调用、用量
//! 依次到达，最后一定是一条 [`ModelEvent::Done`]。

use std::pin::Pin;

use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use oeh_kernel::proposal::BoxFuture;

/// 模型看到的一个工具。`parameters` 是 JSON schema（能力投影时直接用能力的
/// `params_schema`）。
///
/// `concurrency_safe` 不进线格式——它是**本地**调度用的：一批工具调用里连续的安全调用
/// 并行跑，不安全的是屏障（[`crate::tools::ToolRegistry::run_batch`]）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON schema（对象）。
    pub parameters: Value,
    #[serde(default)]
    pub concurrency_safe: bool,
}

/// transcript 里的一条消息。内部 tag `role`，和三家厂商的线格式同名，翻译时少一层映射。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    /// Explicit image attachments; never encode image bytes as ordinary text.
    UserMultimodal {
        content: String,
        images: Vec<ImageInput>,
    },
    Assistant {
        content: String,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        call_id: String,
        content: String,
    },
}

/// Bounded inline camera image. External URLs and filesystem paths are excluded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageInput {
    pub camera: String,
    pub data_url: String,
}

impl ImageInput {
    pub fn validate(&self) -> Result<(), ModelError> {
        let encoded = self
            .data_url
            .strip_prefix("data:image/jpeg;base64,")
            .or_else(|| self.data_url.strip_prefix("data:image/png;base64,"));
        if self.camera.is_empty() || self.camera.len() > 64 || encoded.is_none() {
            return Err(ModelError::Protocol("invalid inline camera image".into()));
        }
        let encoded = encoded.expect("checked");
        if encoded.is_empty()
            || encoded.len() > 2 * 1024 * 1024
            || !encoded.len().is_multiple_of(4)
            || !encoded
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        {
            return Err(ModelError::Protocol(
                "invalid or oversized camera image encoding".into(),
            ));
        }
        Ok(())
    }
}

/// 模型要求调用的一个工具。`id` 由模型给出，结果必须带着同一个 id 回去。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

/// 一次模型调用的输入。`system` 单独一项而不是塞进 `messages`：Anthropic 的线格式就是
/// 单独字段，而 prompt 缓存要求 system 前缀**字节级稳定**，分开放最不容易被动态内容污染。
#[derive(Clone, Debug)]
pub struct ModelRequest {
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
    /// 采样温度。`None` = **这个字段根本不上线**，用厂商自己的默认值。
    ///
    /// 不是 `f32` 加一个"中性值"：没有那样一个值。OpenAI 的推理模型（`gpt-5` 那一档）
    /// 对 `temperature` 一律回 400，而 realtime 只接受 0.6–1.2——同一个数在两个 wire 上
    /// 一个是非法、一个是越界。唯一对所有 wire 都成立的"别管它"就是不发。
    pub temperature: Option<f32>,
}

/// 流上的一个事件。
#[derive(Clone, Debug, PartialEq)]
pub enum ModelEvent {
    /// 文本增量（可能很碎）。
    TextDelta(String),
    /// 一次完整的工具调用（传输负责把增量的参数片段拼完整再发）。
    ToolCall(ToolCall),
    /// 用量。`cached` 是命中 prompt 缓存的输入 token 数；`reasoning` 是 `output` 里推理用掉的
    /// 那一部分，`None` = 这条 wire 不单独报它。
    Usage {
        input: u32,
        output: u32,
        cached: u32,
        reasoning: Option<u32>,
    },
    /// 厂商给这次调用附带的一条不致命告警（Codex 的 `error` item：模型元数据缺失、改路由、
    /// 配置告警……）。不影响答案，但要进证据：调用方把它记下来，不当成失败。
    Warning(String),
    /// 流的最后一条。
    Done { stop: StopReason },
}

/// 模型为什么停。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other,
}

/// 一次模型调用的失败。重试/fallback（Task 3）按种类决定要不要重试：
/// `MissingKey` 永远不重试（重试多少次都还是没有 key），`Http`/`Transport`/`Timeout` 重试。
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("transport: {0}")]
    Transport(String),
    /// `body` 是厂商回的正文（传输读它时已经有上限）。**Display 只放一段摘录**：这串
    /// 字会经 `brain/request-error.reason` 进世界日志，而厂商的 4xx 常常把被拒的请求片段
    /// 回显回来（OpenAI 的 `Invalid 'input[3].content': …` 就是），那既是模型输出又可能
    /// 很长。状态码 + 摘录 + 原长足够定位，全文留在错误值里给会话日志/调试。
    #[error("http {status}: {} ({} bytes)", excerpt(.body), .body.len())]
    Http { status: u16, body: String },
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("timeout")]
    Timeout,
    #[error("missing api key {0}")]
    MissingKey(String),
}

/// [`ModelError::Http`] 的 Display 最多放这么多字节的正文。
pub const MAX_REASON_EXCERPT: usize = 256;

/// 一段能安全进日志的摘录：折掉所有空白（错误页常常是多行 HTML），截到
/// [`MAX_REASON_EXCERPT`] 字节**且落在字符边界上**，截过就加省略号。
fn excerpt(body: &str) -> String {
    let flat = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.len() <= MAX_REASON_EXCERPT {
        return flat;
    }
    let mut end = MAX_REASON_EXCERPT;
    while end > 0 && !flat.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &flat[..end])
}

/// 模型事件流。流本身也可能出错（连接中途断、SSE 解析失败）。
pub type EventStream = Pin<Box<dyn Stream<Item = Result<ModelEvent, ModelError>> + Send>>;

/// 一个可以被原生 turn 循环驱动的模型。
///
/// 不是 `async fn`：`Model` 要能 `dyn`（fallback 链是 `Vec<Arc<dyn Model>>`），所以手写
/// 返回 [`BoxFuture`]。实现必须立刻返回 future，别在 `complete` 里同步发请求。
pub trait Model: Send + Sync {
    /// 进 `brain/request` 的名字，形如 `openai/gpt-5`。
    fn name(&self) -> String;
    /// 这个模型说哪种 wire，snake_case，和 `providers.yaml` 里的 `api_mode` 是同一个词
    /// （`chat_completions` / `anthropic_messages` / `responses` / `realtime`）。
    ///
    /// 有默认值是因为**不是每个 `Model` 都有 wire**：剧本模型、测试替身、将来包在别的
    /// harness 外面的脑都没有。它们报 `"unknown"` 比逼着每个实现编一个词诚实。
    fn api_mode(&self) -> String {
        "unknown".into()
    }
    fn complete(&self, req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_round_trips_with_role_tag() {
        let msgs = vec![
            Message::System {
                content: "you are a robot".into(),
            },
            Message::User {
                content: "walk".into(),
            },
            Message::Assistant {
                content: "ok".into(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "robot_walk".into(),
                    args: json!({"vx": 0.1}),
                }],
            },
            Message::Tool {
                call_id: "c1".into(),
                content: "{\"outcome\":\"executed\"}".into(),
            },
        ];
        let wire = serde_json::to_value(&msgs).unwrap();
        // 线格式钉住：`role` 是 tag，名字小写。
        assert_eq!(wire[0]["role"], "system");
        assert_eq!(wire[2]["role"], "assistant");
        assert_eq!(wire[2]["tool_calls"][0]["name"], "robot_walk");
        assert_eq!(wire[3]["call_id"], "c1");
        let back: Vec<Message> = serde_json::from_value(wire).unwrap();
        assert_eq!(back, msgs);
    }

    #[test]
    fn a_long_error_body_is_excerpted_and_flattened_in_the_message() {
        let body = format!(
            "<html>\n  <body>\n    {}\n  </body>\n</html>",
            "x".repeat(4096)
        );
        let msg = ModelError::Http {
            status: 502,
            body: body.clone(),
        }
        .to_string();
        assert!(msg.starts_with("http 502: <html> <body> xxx"), "{msg}");
        assert!(msg.ends_with(&format!("({} bytes)", body.len())), "{msg}");
        // 摘录 + "http 502: " + " (N bytes)" 的量级，不是 4 KB。
        assert!(msg.len() < MAX_REASON_EXCERPT + 64, "{}", msg.len());
    }

    #[test]
    fn a_short_error_body_survives_whole() {
        let msg = ModelError::Http {
            status: 400,
            body: "bad request".into(),
        }
        .to_string();
        assert_eq!(msg, "http 400: bad request (11 bytes)");
    }

    #[test]
    fn an_excerpt_never_splits_a_character() {
        // 截点正好落在一个三字节字符中间。
        let body = "\u{4e00}".repeat(MAX_REASON_EXCERPT);
        let e = excerpt(&body);
        assert!(e.ends_with('…'));
        assert!(e.len() <= MAX_REASON_EXCERPT + 3);
    }

    #[test]
    fn assistant_tool_calls_default_to_empty() {
        let m: Message =
            serde_json::from_value(json!({"role":"assistant","content":"hi"})).unwrap();
        assert_eq!(
            m,
            Message::Assistant {
                content: "hi".into(),
                tool_calls: vec![]
            }
        );
    }
}
