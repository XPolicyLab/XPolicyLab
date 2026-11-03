//! 各种 wire 的实现（规格 §7.2）。
//!
//! | 模块 | wire | 端点 / 凭据 |
//! |---|---|---|
//! | [`chat_completions`] | OpenAI 兼容的 chat | `POST {base_url}/chat/completions` |
//! | [`anthropic_messages`] | Anthropic Messages | `POST {base_url}/v1/messages` |
//! | [`responses`] | OpenAI Responses | `POST {base_url}/responses` |
//! | [`realtime`] | OpenAI Realtime | WebSocket `{base_url}?model=…` |
//! | [`codex_exec`] | Codex CLI 子进程 | ChatGPT 订阅登录（`CODEX_HOME`） |
//! | [`claude_code_exec`] | Claude Code CLI 子进程 | Claude 订阅 token（`claude setup-token`） |
//!
//! 前四个按 API key 计费，后两个用账号订阅（不读任何 API key）。所有传输对外只有一张脸：[`Model`]。翻译发生在两头——请求体各按各家的形状拼，流回来
//! 之后各按各家的事件翻回同一串 [`ModelEvent`]。turn 循环因此不认识任何一家厂商。
//!
//! 三个 HTTP 传输共用这里的三件东西：[`client`]（一个 `reqwest::Client`，超时就是 step
//! 的超时）、[`post`]（发请求 + 把非 2xx 变成 [`ModelError::Http`]）、[`stream_of`]（把
//! 字节流经 [`sse::SseDecoder`] 喂给各家的 [`Decode`]）。realtime 不走 SSE，自己有一套。

pub mod anthropic_messages;
pub mod chat_completions;
pub mod claude_code_exec;
pub mod codex_exec;
pub mod realtime;
pub mod responses;
pub mod sse;

pub use anthropic_messages::AnthropicMessagesModel;
pub use chat_completions::ChatCompletionsModel;
pub use claude_code_exec::ClaudeCodeExecModel;
pub use codex_exec::CodexExecModel;
pub use realtime::{Device, NullDevice, RealtimeModel};
pub use responses::ResponsesModel;

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::pin::Pin;
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

use crate::model::{EventStream, ModelError, ModelEvent, ToolCall};
use crate::providers::Runtime;
use sse::{SseDecoder, SseEvent};

/// 起一个订阅 CLI 子进程（`codex_exec` / `claude_code_exec`）。
///
/// `ExecutableFileBusy`（ETXTBSY：可执行文件此刻还被某个进程开着写）重试几次再放弃：CLI 的自动更新
/// 正在替换文件时会撞上；测试里刚写好的假 CLI 被并发 fork 出去的子进程短暂继承了写句柄时也会。
pub(crate) async fn spawn_cli(
    cmd: &mut tokio::process::Command,
) -> std::io::Result<tokio::process::Child> {
    let mut attempt = 0u64;
    loop {
        match cmd.spawn() {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && attempt < 5 => {
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(20 * attempt)).await;
            }
            other => return other,
        }
    }
}

/// 一家厂商的流解码器：一条 SSE 事件 → 0..n 条 [`ModelEvent`]。
///
/// 有状态（工具调用的参数是一片一片来的、用量和停止理由在别的事件上），所以是一个
/// `&mut self` 的对象而不是一个纯函数。
pub(crate) trait Decode: Send + 'static {
    fn on_event(&mut self, ev: SseEvent) -> Result<Vec<ModelEvent>, ModelError>;
    /// 流没了（正常收尾或者连接断掉）：还没发出去的东西在这里补。
    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelError>;
}

/// 造这个 runtime 的 HTTP 客户端。
///
/// 超时用的就是 step 的超时（`timeout_ms`，0 = 不设限）：对一个 turn 来说，"服务器不
/// 回话"和"流读不完"是同一件事，没有理由给它们两个不同的预算。
pub(crate) fn client(rt: &Runtime) -> Result<reqwest::Client, String> {
    let mut b = reqwest::Client::builder();
    if rt.timeout_ms > 0 {
        b = b.timeout(Duration::from_millis(rt.timeout_ms));
    }
    b.build().map_err(|e| format!("http client: {e}"))
}

/// 一次 POST：非 2xx 直接变成 [`ModelError::Http`]（body 原样带上——厂商把"为什么被拒"
/// 写在那里，而重试瀑布只看状态码，人看的是 body）。
pub(crate) async fn post(
    http: &reqwest::Client,
    url: &str,
    headers: &BTreeMap<String, String>,
    body: &Value,
) -> Result<reqwest::Response, ModelError> {
    let mut rb = http.post(url).json(body);
    for (k, v) in headers {
        rb = rb.header(k, v);
    }
    let resp = rb.send().await.map_err(map_reqwest)?;
    let status = resp.status();
    if !status.is_success() {
        return Err(ModelError::Http {
            status: status.as_u16(),
            body: error_body(resp).await,
        });
    }
    Ok(resp)
}

/// 非 2xx 的正文最多读这么多。对面可能回一整页 HTML（中间代理的错误页），而这段文本
/// 会跟着 [`ModelError::Http`] 一路走到世界日志的 `reason` 上——那里既没有 payload 上限，
/// 也不该装厂商回显回来的东西。
pub(crate) const MAX_ERROR_BODY: usize = 64 * 1024;

/// 读错误正文，最多 [`MAX_ERROR_BODY`] 字节；读到上限就停（剩下的连着连接一起丢），
/// 末尾补一个记号让读日志的人知道后面还有。
async fn error_body(resp: reqwest::Response) -> String {
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(Ok(chunk)) = stream.next().await {
        let room = MAX_ERROR_BODY - buf.len();
        if chunk.len() >= room {
            buf.extend_from_slice(&chunk[..room]);
            let mut s = String::from_utf8_lossy(&buf).into_owned();
            s.push_str(" …[truncated]");
            return s;
        }
        buf.extend_from_slice(&chunk);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

pub(crate) fn map_reqwest(e: reqwest::Error) -> ModelError {
    if e.is_timeout() {
        ModelError::Timeout
    } else if e.is_decode() {
        ModelError::Protocol(e.to_string())
    } else {
        ModelError::Transport(e.to_string())
    }
}

/// 认证头 + 声明里的额外头。
pub(crate) fn headers_with(rt: &Runtime, auth: &[(&str, String)]) -> BTreeMap<String, String> {
    let mut h: BTreeMap<String, String> = auth
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    // 声明里的头在后：profile 能覆盖默认值（比如换一个 `anthropic-version`）。
    h.extend(rt.headers.clone());
    h
}

/// 响应体 → [`EventStream`]。
///
/// 用 `unfold` 手写而不是 `map`/`scan` 串起来：一条 SSE 事件可能翻出 0 条、也可能翻出
/// 3 条 [`ModelEvent`]，而流要一条一条往外吐——中间必须有一个队列。
pub(crate) fn stream_of<D: Decode>(resp: reqwest::Response, dec: D) -> EventStream {
    let bytes: ByteStream = Box::pin(resp.bytes_stream().map(|r| match r {
        Ok(b) => Ok(b.to_vec()),
        Err(e) => Err(map_reqwest(e)),
    }));
    stream_of_bytes(bytes, dec)
}

/// [`stream_of`] 的内脏，脱开 `reqwest`：字节流 → [`EventStream`]。分出来是为了能用一条
/// 假的分片流测它（`reqwest::Response` 造不出来）。
pub(crate) fn stream_of_bytes<D: Decode>(bytes: ByteStream, dec: D) -> EventStream {
    Box::pin(futures_util::stream::unfold(
        Decoding {
            bytes,
            sse: SseDecoder::new(),
            dec,
            queue: VecDeque::new(),
            ended: false,
        },
        |mut s| async move {
            loop {
                if let Some(item) = s.queue.pop_front() {
                    return Some((item, s));
                }
                if s.ended {
                    return None;
                }
                match s.bytes.next().await {
                    Some(Ok(chunk)) => {
                        // 切分本身也会失败：一条超过 `sse::MAX_SSE_LINE` /
                        // `MAX_SSE_EVENT` 的流是协议错，流到此为止（不再喂下一片，
                        // 缓冲不再涨）。
                        let evs = match s.sse.push(&chunk) {
                            Ok(evs) => evs,
                            Err(e) => {
                                s.ended = true;
                                s.queue.push_back(Err(e));
                                continue;
                            }
                        };
                        for ev in evs {
                            match s.dec.on_event(ev) {
                                Ok(evs) => s.queue.extend(evs.into_iter().map(Ok)),
                                Err(e) => {
                                    s.ended = true;
                                    s.queue.push_back(Err(e));
                                    break;
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        s.ended = true;
                        s.queue.push_back(Err(e));
                    }
                    None => {
                        s.ended = true;
                        // 缓冲里剩下的半条事件先交出去（连接被中途关掉时最后一条常常就
                        // 差那个空行），再让解码器收尾。收尾出过错就不再收尾——`Done`
                        // 跟在一条 `Err` 后面只会让"这一轮说完了"和"这一轮断了"混在一起。
                        let (tail, tail_err): (Vec<SseEvent>, Option<ModelError>) =
                            match s.sse.finish() {
                                Ok(evs) => (evs, None),
                                Err(e) => (Vec::new(), Some(e)),
                            };
                        let mut failed = tail_err.is_some();
                        if let Some(e) = tail_err {
                            s.queue.push_back(Err(e));
                        }
                        for ev in tail {
                            match s.dec.on_event(ev) {
                                Ok(evs) => s.queue.extend(evs.into_iter().map(Ok)),
                                Err(e) => {
                                    s.queue.push_back(Err(e));
                                    failed = true;
                                    break;
                                }
                            }
                        }
                        if !failed {
                            match s.dec.finish() {
                                Ok(evs) => s.queue.extend(evs.into_iter().map(Ok)),
                                Err(e) => s.queue.push_back(Err(e)),
                            }
                        }
                    }
                }
            }
        },
    ))
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, ModelError>> + Send>>;

struct Decoding<D> {
    bytes: ByteStream,
    sse: SseDecoder,
    dec: D,
    queue: VecDeque<Result<ModelEvent, ModelError>>,
    ended: bool,
}

/// 工具调用的参数：厂商发的是一串 JSON **文本片段**，拼完了才解析。
///
/// 空串当空对象——没有参数的工具，三家都可能一个片段都不发。解不开就是
/// [`ModelError::Protocol`]：这次调用没法执行，而把一个半截的 JSON 当成 `{}` 交给能力
/// 执行器是最坏的那种"继续跑"。
pub(crate) fn parse_args(name: &str, raw: &str) -> Result<Value, ModelError> {
    let t = raw.trim();
    if t.is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    // 错误串里**没有那段文本**：它会原样走到 `brain/request-error.reason` 和
    // `brain/turn_end.reason`，而"模型输出全文不进世界日志"是一条纪律（工具参数里可能
    // 有地址、人名）。名字 + 长度 + serde 报的位置足够定位，而 serde 解到 `Value` 时
    // 只会报语法错（"EOF while parsing …"），不回显内容。
    serde_json::from_str(t).map_err(|e| {
        ModelError::Protocol(format!(
            "tool call `{name}` arguments: {e} ({} bytes)",
            t.len()
        ))
    })
}

/// 线上**必须**有的一个字符串字段。
///
/// 不用 `unwrap_or_default()`：那会把"读不到"降级成"读到了空的"，而一次名字为空的工具
/// 调用会被当成"模型要求了一个不存在的工具"——一条根本不是模型犯的错，还查不出来源。
/// 厂商换字段名时要的是一条明确的协议错。
pub(crate) fn required_str<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a str, ModelError> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ModelError::Protocol(format!("{what}: missing `{key}`")))
}

pub(crate) fn tool_call(id: &str, name: &str, raw: &str) -> Result<ToolCall, ModelError> {
    Ok(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        args: parse_args(name, raw)?,
    })
}

/// 线上的 JSON：解不开就是协议错，不是 panic。
pub(crate) fn parse_json(data: &str) -> Result<Value, ModelError> {
    serde_json::from_str(data).map_err(|e| ModelError::Protocol(format!("sse data: {e}")))
}

/// `f32` 的配置进 JSON。
///
/// 直接 `as f64` 会把 `0.2` 写成 `0.20000000298023224`（f32 那个值的十进制全展开）：
/// 厂商都收，但请求体是要进人眼、也要进 prompt 缓存哈希的。留三位小数——采样参数从来
/// 不需要更细。
pub(crate) fn temperature(t: f32) -> f64 {
    (f64::from(t) * 1000.0).round() / 1000.0
}

/// 把 `temperature` 放进请求体——**只在配了的时候**。
///
/// `None` 就一个键都不加，而不是加一个"中性"的数：没有那样一个数（推理模型对任何
/// `temperature` 都回 400），而一个 400 在瀑布眼里不可重试，于是整条链走完、每一个 turn
/// 都以 `done("error: http 400 …")` 收尾。见 [`crate::model::ModelRequest::temperature`]。
pub(crate) fn put_temperature(body: &mut Value, t: Option<f32>) {
    if let (Value::Object(m), Some(t)) = (&mut *body, t) {
        m.insert("temperature".into(), json!(temperature(t)));
    }
}

/// 工具参数的确定性编码（prompt 缓存要求请求体字节级稳定）。
pub(crate) fn compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
}

/// `usage` 里的一个计数。厂商的字段名各不相同，取第一个存在的。
pub(crate) fn count(v: &Value, keys: &[&str]) -> u32 {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(Value::as_u64) {
            return n as u32;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_arguments_mean_an_empty_object() {
        assert_eq!(parse_args("t", "  ").unwrap(), json!({}));
    }

    #[test]
    fn half_a_json_object_is_a_protocol_error_not_an_empty_call() {
        let e = parse_args(
            "robot_walk",
            "{\"vx\":0.1,\"where\":\"\u{5317}\u{4eac}\u{5e02}\u{671d}\u{9633}\u{533a}\"",
        )
        .unwrap_err();
        assert!(matches!(e, ModelError::Protocol(_)), "{e:?}");
        let msg = e.to_string();
        assert!(msg.contains("robot_walk"), "{msg}");
        // 名字、长度、位置——**没有那段文本**（它会一路走到世界日志）。
        assert!(msg.contains("bytes"), "{msg}");
        assert!(
            !msg.contains("\u{671d}\u{9633}"),
            "参数全文漏进了错误串: {msg}"
        );
        assert!(!msg.contains("vx"), "参数全文漏进了错误串: {msg}");
    }

    #[test]
    fn a_missing_wire_field_is_a_protocol_error_not_an_empty_string() {
        let v = json!({"name": "", "call_id": "c1"});
        assert_eq!(required_str(&v, "call_id", "x").unwrap(), "c1");
        for key in ["name", "nope"] {
            let e = required_str(&v, key, "realtime function_call").unwrap_err();
            assert!(matches!(e, ModelError::Protocol(_)), "{e:?}");
            assert!(e.to_string().contains(key), "{e}");
        }
    }

    #[test]
    fn a_temperature_does_not_carry_f32_noise_into_the_body() {
        assert_eq!(json!(temperature(0.2)).to_string(), "0.2");
        assert_eq!(json!(temperature(1.0)).to_string(), "1.0");
    }

    #[test]
    fn a_count_takes_the_first_name_that_exists() {
        let u = json!({"output_tokens": 5});
        assert_eq!(count(&u, &["completion_tokens", "output_tokens"]), 5);
        assert_eq!(count(&u, &["nope"]), 0);
    }

    /// 一个解码器：每条 SSE 事件变成一条同名的 `TextDelta`。
    struct Echo;
    impl Decode for Echo {
        fn on_event(&mut self, ev: SseEvent) -> Result<Vec<ModelEvent>, ModelError> {
            Ok(vec![ModelEvent::TextDelta(ev.data)])
        }
        fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelError> {
            Ok(Vec::new())
        }
    }

    /// I1：一条**永不结束**的分片流。没有上限时这里会一直攒到进程被 OOM 杀掉；有上限
    /// 时流以一条 `Protocol` 收尾，然后**真的结束**（下一次 `next()` 是 `None`）——
    /// 瀑布看见协议错不重试，turn 以 `done("error: …")` 收尾。
    #[tokio::test]
    async fn an_unbounded_sse_line_ends_the_stream_with_a_protocol_error() {
        // 无限流：每次都给一片不含换行的字节。取够条数就一定越界。
        let bytes: ByteStream = Box::pin(futures_util::stream::repeat_with(|| {
            Ok(vec![b'x'; 64 * 1024])
        }));
        let mut s = stream_of_bytes(bytes, Echo);
        let first = s.next().await.expect("流上必须有一条错").unwrap_err();
        assert!(matches!(first, ModelError::Protocol(_)), "{first:?}");
        assert!(first.to_string().contains("too large"), "{first}");
        assert!(s.next().await.is_none(), "出错之后流必须结束");
    }

    /// 正常的一段分片照常翻成事件——上限不改成功路径的行为。
    #[tokio::test]
    async fn a_bounded_stream_still_decodes() {
        let chunks: Vec<Result<Vec<u8>, ModelError>> =
            vec![Ok(b"data: a\n\ndata: b".to_vec()), Ok(b"\n\n".to_vec())];
        let bytes: ByteStream = Box::pin(futures_util::stream::iter(chunks));
        let mut s = stream_of_bytes(bytes, Echo);
        let mut got = Vec::new();
        while let Some(ev) = s.next().await {
            got.push(ev.unwrap());
        }
        assert_eq!(
            got,
            vec![
                ModelEvent::TextDelta("a".into()),
                ModelEvent::TextDelta("b".into())
            ]
        );
    }
}
