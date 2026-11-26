//! `realtime`：OpenAI Realtime 的 WebSocket wire（快脑的语音通道）。
//!
//! 和三个 HTTP 传输不同的是**连接是有状态的**：连上之后先 `session.update`（工具表 +
//! instructions），再把 transcript 逐条 `conversation.item.create` 推过去，最后一条
//! `response.create` 才让模型开口。回来的事件里 `response.text.delta` 是文本、
//! `response.function_call_arguments.done` 是一次完整的工具调用、`response.done` 收尾。
//!
//! 一次 [`Model::complete`] = 一条连接。里程碑 1 不做"一条长连接跨 turn 复用"：turn 循环
//! 的每个 step 都是一次独立的调用，而复用连接意味着要在脑这一侧维护会话状态机（谁清
//! transcript、断线重连之后模型记得多少）——那是语音落地时才有足够信息决定的事。
//!
//! **语音转写不是模型事件，是一条观测。** 收到
//! `conversation.item.input_audio_transcription.completed` 时这里往世界日志写一条
//! `obs/speech{text}`：别的脑、回放和蒸馏器因此都看得见"有人说了这句话"，而不是只有这
//! 一个 turn 的 transcript 里有（规格 §7.2 的音频行）。

use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use oeh_kernel::world_log::WorldLog;
use oeh_proto::{EventDraft, EventKind};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use super::{count, put_temperature, required_str, tool_call};
use crate::model::{
    BoxFuture, EventStream, Message, Model, ModelError, ModelEvent, ModelRequest, StopReason,
};
use crate::providers::Runtime;

/// `obs/speech` 的 `source`。
const SOURCE: &str = "brain-realtime";

/// 音频设备的接缝。
///
/// 里程碑 1 只有 [`NullDevice`]：realtime 传输跑纯文本，麦克风与扬声器是**设备插件**的
/// 事（规格 §7.2），接进来的时候实现这个 trait 而不是改传输。留在这里是为了让"音频往
/// 哪接"在代码里有一个位置，而不是只在计划文档里。
pub trait Device: Send + Sync {
    /// 下一段要发给模型的 PCM。`None` = 这个设备不录音。
    fn next_input(&self) -> Option<Vec<u8>> {
        None
    }
    /// 模型回来的 PCM。默认丢掉。
    fn play(&self, _pcm: &[u8]) {}
}

/// 不录也不放。
pub struct NullDevice;

impl Device for NullDevice {}

pub struct RealtimeModel {
    rt: Runtime,
    /// `obs/speech` 的去处。`None` = 没人接（直连测试、或者建模型时还没有日志）：转写
    /// 就只影响不了世界，而不是让整条流失败。
    log: Option<Arc<WorldLog>>,
}

impl RealtimeModel {
    pub fn new(rt: Runtime) -> Result<Self, String> {
        if !rt.base_url.starts_with("ws://") && !rt.base_url.starts_with("wss://") {
            return Err(format!(
                "provider `{}`: realtime needs a ws:// or wss:// base_url, got {}",
                rt.provider, rt.base_url
            ));
        }
        Ok(RealtimeModel { rt, log: None })
    }

    pub fn with_world_log(mut self, log: Arc<WorldLog>) -> Self {
        self.log = Some(log);
        self
    }
}

impl Model for RealtimeModel {
    fn name(&self) -> String {
        self.rt.name()
    }

    fn api_mode(&self) -> String {
        self.rt.api_mode.as_str().to_string()
    }

    fn complete(&self, req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
        let rt = self.rt.clone();
        let log = self.log.clone();
        Box::pin(async move {
            if req
                .messages
                .iter()
                .any(|m| matches!(m, Message::UserMultimodal { .. }))
            {
                return Err(ModelError::Protocol(
                    "inline vision currently requires the responses transport".into(),
                ));
            }

            let Some(key) = rt.api_key.clone() else {
                return Err(ModelError::MissingKey(rt.env.clone()));
            };
            let base = with_path(&rt.base_url);
            let sep = if base.contains('?') { '&' } else { '?' };
            let url = format!("{base}{sep}model={}", rt.model);
            let mut request = url
                .into_client_request()
                .map_err(|e| ModelError::Transport(format!("realtime url: {e}")))?;
            let h = request.headers_mut();
            let mut set = |k: &str, v: String| -> Result<(), ModelError> {
                let name = HeaderName::from_bytes(k.as_bytes())
                    .map_err(|e| ModelError::Transport(format!("header {k}: {e}")))?;
                let value = HeaderValue::from_str(&v)
                    .map_err(|e| ModelError::Transport(format!("header {k}: {e}")))?;
                h.insert(name, value);
                Ok(())
            };
            set("authorization", format!("Bearer {key}"))?;
            set("openai-beta", "realtime=v1".into())?;
            for (k, v) in &rt.headers {
                set(k, v.clone())?;
            }
            let (mut ws, _) = tokio_tungstenite::connect_async(request)
                .await
                .map_err(|e| ModelError::Transport(format!("realtime connect: {e}")))?;
            for m in outgoing(&req) {
                send(&mut ws, m).await?;
            }
            Ok(events(ws, log))
        })
    }
}

/// `ws://host:port` 这样没有路径的 URL 拼出来的请求行是 `GET  HTTP/1.1`，握手当场失败。
/// 补一个 `/`——一个自建网关的声明里完全可能只写到端口。
fn with_path(url: &str) -> String {
    match url.split_once("//") {
        Some((_, rest)) if !rest.contains('/') && !rest.contains('?') => format!("{url}/"),
        _ => url.to_string(),
    }
}

/// 连上之后要发出去的那一串：session.update → 每条消息一个 item → response.create。
fn outgoing(req: &ModelRequest) -> Vec<Value> {
    // `temperature` 在 realtime 是**会话**级的（不是每次 response 的参数）。厂商侧还有
    // 一条下界（beta 的 realtime 只接受 0.6–1.2），配得更低会被 400 拒掉——那是 provider
    // 侧的约束，不该由传输偷偷改写脑的配置。
    // 没配就不发：厂商的默认值一定在它自己的区间里，而我们编的任何一个数都可能不在。
    let mut session = json!({"modalities": ["text"]});
    put_temperature(&mut session, req.temperature);
    if !req.system.is_empty() {
        session["instructions"] = json!(req.system);
    }
    if !req.tools.is_empty() {
        session["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect(),
        );
    }
    let mut out = vec![json!({"type": "session.update", "session": session})];
    for m in &req.messages {
        let item = match m {
            Message::System { content } | Message::User { content } => json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            }),
            Message::UserMultimodal { .. } => unreachable!("validated transport modality"),
            Message::Assistant { content, .. } => json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
            }),
            Message::Tool { call_id, content } => json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            }),
        };
        out.push(json!({"type": "conversation.item.create", "item": item}));
    }
    // `max_output_tokens` 是 response 级的：`NativeConfig.max_tokens` 在这条 wire 上只有
    // 放在这里才生效。
    out.push(json!({
        "type": "response.create",
        "response": {"max_output_tokens": req.max_tokens},
    }));
    out
}

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

async fn send(ws: &mut Ws, v: Value) -> Result<(), ModelError> {
    ws.send(WsMessage::text(v.to_string()))
        .await
        .map_err(|e| ModelError::Transport(format!("realtime send: {e}")))
}

/// 服务器事件 → [`ModelEvent`] 流。
fn events(ws: Ws, log: Option<Arc<WorldLog>>) -> EventStream {
    Box::pin(futures_util::stream::unfold(
        State {
            ws,
            log,
            dec: Decoder::default(),
            queue: std::collections::VecDeque::new(),
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
                match s.ws.next().await {
                    Some(Ok(WsMessage::Text(t))) => match serde_json::from_str::<Value>(&t) {
                        Ok(v) => {
                            if let Some(text) = transcript(&v) {
                                if let Some(log) = &s.log {
                                    log.append(EventDraft::new(
                                        EventKind::new("obs/speech").expect("static kind"),
                                        SOURCE,
                                        json!({"text": text}),
                                    ));
                                }
                            }
                            match s.dec.on_message(&v) {
                                Ok(evs) => s.queue.extend(evs.into_iter().map(Ok)),
                                Err(e) => {
                                    s.ended = true;
                                    s.queue.push_back(Err(e));
                                }
                            }
                        }
                        Err(e) => {
                            s.ended = true;
                            s.queue
                                .push_back(Err(ModelError::Protocol(format!("realtime: {e}"))));
                        }
                    },
                    // 二进制（音频）与控制帧：里程碑 1 不处理。
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        s.ended = true;
                        s.queue
                            .push_back(Err(ModelError::Transport(format!("realtime: {e}"))));
                    }
                    None => {
                        s.ended = true;
                        s.queue
                            .extend(s.dec.end(StopReason::Other).into_iter().map(Ok));
                    }
                }
            }
        },
    ))
}

struct State {
    ws: Ws,
    log: Option<Arc<WorldLog>>,
    dec: Decoder,
    queue: std::collections::VecDeque<Result<ModelEvent, ModelError>>,
    ended: bool,
}

/// 这条事件是不是一句被转写出来的话。
fn transcript(v: &Value) -> Option<&str> {
    (v["type"] == "conversation.item.input_audio_transcription.completed")
        .then(|| v["transcript"].as_str())
        .flatten()
        .filter(|t| !t.is_empty())
}

/// 一个已经宣布、还没发出去的函数调用。
///
/// **工具名只在 item 上**：`response.function_call_arguments.delta/done` 的载荷是
/// `{event_id, type, response_id, item_id, output_index, call_id, arguments}`——没有
/// `name`。名字来自 `response.output_item.added`（调用一开始就发，早于参数）或
/// `.done` 的 `item{type:"function_call", id, call_id, name}`。
struct Pending {
    call_id: String,
    name: String,
    emitted: bool,
}

#[derive(Default)]
struct Decoder {
    /// `item_id` → 那个 item 宣布的调用。
    items: std::collections::BTreeMap<String, Pending>,
    calls: u32,
    ended: bool,
}

impl Decoder {
    /// 按 `item_id` 找那条 item；找不到就按 `call_id` 找（两个 id 在参数事件上都带着，
    /// 而代理/新版本最可能换掉的是其中一个）。
    fn pending(&mut self, item_id: &str, call_id: &str) -> Option<&mut Pending> {
        if self.items.contains_key(item_id) {
            return self.items.get_mut(item_id);
        }
        if call_id.is_empty() {
            return None;
        }
        self.items.values_mut().find(|p| p.call_id == call_id)
    }

    fn on_message(&mut self, v: &Value) -> Result<Vec<ModelEvent>, ModelError> {
        let mut out = Vec::new();
        let kind = v["type"].as_str().unwrap_or_default();
        match kind {
            "response.text.delta" => {
                if let Some(t) = v["delta"].as_str() {
                    if !t.is_empty() {
                        out.push(ModelEvent::TextDelta(t.to_string()));
                    }
                }
            }
            // 调用的**名字**在这里。`added` 在参数之前发（它的 `arguments` 是占位的
            // 空串，不能当真）；`done` 带着最终参数，只有在参数事件没来过时才由它补发
            // 这次调用——免得同一次调用发两遍。
            "response.output_item.added" | "response.output_item.done" => {
                let item = &v["item"];
                if item["type"] != "function_call" {
                    return Ok(out);
                }
                let id = item["id"].as_str().unwrap_or_default().to_string();
                let call_id = required_str(item, "call_id", "realtime function_call")?.to_string();
                let name = required_str(item, "name", "realtime function_call")?.to_string();
                let p = self.items.entry(id).or_insert_with(|| Pending {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    emitted: false,
                });
                p.call_id = call_id;
                p.name = name;
                if kind == "response.output_item.done" && !p.emitted {
                    p.emitted = true;
                    let args = item["arguments"].as_str().unwrap_or_default();
                    let call = tool_call(&p.call_id, &p.name, args)?;
                    self.calls += 1;
                    out.push(ModelEvent::ToolCall(call));
                }
            }
            "response.function_call_arguments.done" => {
                let item_id = v["item_id"].as_str().unwrap_or_default();
                let call_id = v["call_id"].as_str().unwrap_or_default();
                let args = v["arguments"].as_str().unwrap_or_default().to_string();
                // 名字取不到就是协议错，**不发一次名字为空的调用**：那会变成"模型要求了
                // 一个不存在的工具"，一条根本不是模型犯的错。
                let Some(p) = self.pending(item_id, call_id) else {
                    return Err(ModelError::Protocol(format!(
                        "realtime function call {item_id}/{call_id}: no output_item announced its name"
                    )));
                };
                if p.emitted {
                    return Ok(out);
                }
                p.emitted = true;
                let call = tool_call(&p.call_id, &p.name, &args)?;
                self.calls += 1;
                out.push(ModelEvent::ToolCall(call));
            }
            "response.done" => {
                let u = &v["response"]["usage"];
                if u.is_object() {
                    out.push(ModelEvent::Usage {
                        input: count(u, &["input_tokens"]),
                        output: count(u, &["output_tokens"]),
                        // realtime 的字段名是单数的 `input_token_details`，Responses 那边
                        // 是复数——两个都认，省得一次改名让用量静悄悄变成 0。
                        cached: count(&u["input_token_details"], &["cached_tokens"])
                            .max(count(&u["input_tokens_details"], &["cached_tokens"])),
                        reasoning: None,
                    });
                }
                let stop = if self.calls > 0 {
                    StopReason::ToolUse
                } else {
                    StopReason::EndTurn
                };
                out.append(&mut self.end(stop));
            }
            "error" => {
                let msg = v["error"]["message"].as_str().unwrap_or("realtime error");
                return Err(ModelError::Transport(msg.to_string()));
            }
            _ => {}
        }
        Ok(out)
    }

    fn end(&mut self, stop: StopReason) -> Vec<ModelEvent> {
        if self.ended {
            return Vec::new();
        }
        self.ended = true;
        vec![ModelEvent::Done { stop }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ToolSpec;
    use crate::providers::ApiMode;
    use std::collections::BTreeMap;

    fn rt(base_url: &str) -> Runtime {
        Runtime {
            provider: "openai-realtime".into(),
            model: "gpt-realtime".into(),
            api_mode: ApiMode::Realtime,
            base_url: base_url.into(),
            env: "K".into(),
            api_key: Some("k".into()),
            cache: None,
            headers: BTreeMap::new(),
            timeout_ms: 0,
            program: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn an_http_base_url_is_rejected_at_build_time() {
        let Err(err) = RealtimeModel::new(rt("http://x")) else {
            panic!("http:// 不是一条 WebSocket")
        };
        assert!(err.contains("ws://"), "{err}");
    }

    #[test]
    fn a_base_url_without_a_path_gets_one() {
        assert_eq!(with_path("ws://127.0.0.1:9"), "ws://127.0.0.1:9/");
        assert_eq!(
            with_path("wss://api.openai.com/v1/realtime"),
            "wss://api.openai.com/v1/realtime"
        );
    }

    #[test]
    fn the_first_message_carries_the_tools_and_the_last_one_asks_for_a_response() {
        let msgs = outgoing(&ModelRequest {
            system: "S".into(),
            messages: vec![Message::User {
                content: "hi".into(),
            }],
            tools: vec![ToolSpec {
                name: "robot_walk".into(),
                description: "d".into(),
                parameters: json!({"type": "object"}),
                concurrency_safe: false,
            }],
            max_tokens: 8,
            temperature: Some(0.0),
        });
        assert_eq!(msgs[0]["type"], "session.update");
        assert_eq!(msgs[0]["session"]["instructions"], "S");
        assert_eq!(msgs[0]["session"]["tools"][0]["name"], "robot_walk");
        assert_eq!(msgs[0]["session"]["temperature"], 0.0);
        assert_eq!(msgs[1]["type"], "conversation.item.create");
        assert_eq!(msgs[2]["type"], "response.create");
        assert_eq!(msgs[2]["response"]["max_output_tokens"], 8);
    }

    /// I2：realtime 的 `temperature` 是**会话**级的，而厂商只接受 0.6–1.2。没配就不发，
    /// 让厂商用自己的默认值——传输不替脑编一个落在区间里的数。
    #[test]
    fn temperature_is_sent_only_when_it_is_configured() {
        let mut r = ModelRequest {
            system: String::new(),
            messages: vec![Message::User {
                content: "hi".into(),
            }],
            tools: vec![],
            max_tokens: 8,
            temperature: None,
        };
        assert!(outgoing(&r)[0]["session"].get("temperature").is_none());
        r.temperature = Some(0.8);
        assert_eq!(outgoing(&r)[0]["session"]["temperature"], 0.8);
    }

    /// 参数事件上没有名字：没人宣布过这个 item 就报协议错，而不是发一次空名字的调用。
    #[test]
    fn arguments_without_an_announced_item_are_a_protocol_error() {
        let mut d = Decoder::default();
        let e = d
            .on_message(&json!({
                "type": "response.function_call_arguments.done",
                "item_id": "item_1",
                "call_id": "call_1",
                "arguments": "{}"
            }))
            .unwrap_err();
        assert!(matches!(e, ModelError::Protocol(_)), "{e:?}");
        assert!(e.to_string().contains("item_1"), "{e}");
    }

    /// 参数事件一条都没来（有些网关只发 item 事件）：由 `output_item.done` 补发。
    #[test]
    fn an_item_done_emits_the_call_when_no_arguments_event_arrived() {
        let mut d = Decoder::default();
        let out = d
            .on_message(&json!({
                "type": "response.output_item.done",
                "item": {"id": "item_1", "type": "function_call", "call_id": "call_1",
                         "name": "robot_walk", "arguments": "{\"vx\":0.1}"}
            }))
            .unwrap();
        assert_eq!(
            out,
            vec![ModelEvent::ToolCall(crate::model::ToolCall {
                id: "call_1".into(),
                name: "robot_walk".into(),
                args: json!({"vx": 0.1}),
            })]
        );
    }

    /// 名字来自 `response.output_item.added`；参数事件只带 id。
    #[test]
    fn the_name_comes_from_the_item_and_the_arguments_from_their_own_event() {
        let mut d = Decoder::default();
        assert!(d
            .on_message(&json!({
                "type": "response.output_item.added",
                "item": {"id": "item_1", "type": "function_call", "call_id": "call_1", "name": "robot_walk"}
            }))
            .unwrap()
            .is_empty());
        let out = d
            .on_message(&json!({
                "type": "response.function_call_arguments.done",
                "item_id": "item_1",
                "call_id": "call_1",
                "arguments": "{\"vx\":0.1}"
            }))
            .unwrap();
        assert_eq!(
            out,
            vec![ModelEvent::ToolCall(crate::model::ToolCall {
                id: "call_1".into(),
                name: "robot_walk".into(),
                args: json!({"vx": 0.1}),
            })]
        );
        // `response.output_item.done` 跟在后面（带最终参数）不会把同一次调用发第二遍。
        let again = d
            .on_message(&json!({
                "type": "response.output_item.done",
                "item": {"id": "item_1", "type": "function_call", "call_id": "call_1",
                         "name": "robot_walk", "arguments": "{\"vx\":0.1}"}
            }))
            .unwrap();
        assert!(again.is_empty(), "{again:?}");
    }

    #[test]
    fn only_a_completed_transcription_is_an_observation() {
        assert_eq!(
            transcript(&json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "hi"
            })),
            Some("hi")
        );
        assert_eq!(transcript(&json!({"type": "response.done"})), None);
        // 空转写不写事件：那是一段没听清的音频，不是一句话。
        assert_eq!(
            transcript(&json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": ""
            })),
            None
        );
    }
}
