//! `anthropic_messages`：Anthropic 的 Messages API。
//!
//! 和 OpenAI 那套差在三个地方，每一个都影响请求体的形状：
//!
//! - **system 是独立字段**，不是 `messages` 的第一条（[`ModelRequest::system`] 因此也是
//!   独立字段——两边同形，翻译少一层）；
//! - **消息只有 user / assistant 两种角色**：上一轮的工具调用是 assistant 里的
//!   `tool_use` 块，工具结果是**下一条 user** 里的 `tool_result` 块；
//! - **工具的 schema 叫 `input_schema`**。
//!
//! prompt 缓存（`cache: system_and_3`）：`cache_control` 打在 system 块和**最后三条
//! user** 上。Anthropic 最多认 4 个缓存点，而"system 前缀 + 最近几轮"正好是那条稳定
//! 前缀的边界——system 按纪律字节级稳定，最近三条 user 是滚动的窗口。

use serde_json::{json, Value};

use super::{
    client, count, headers_with, parse_json, post, put_temperature, required_str, stream_of,
    tool_call, Decode,
};
use crate::model::{
    BoxFuture, EventStream, Message, Model, ModelError, ModelEvent, ModelRequest, StopReason,
};
use crate::providers::Runtime;
use crate::transport::sse::SseEvent;

/// 厂商要求的版本头。声明里的 `headers` 可以覆盖它（新版本出来不必改代码）。
const VERSION: &str = "2023-06-01";

/// `cache: system_and_3` 里的 3。
const CACHED_USER_TURNS: usize = 3;

pub struct AnthropicMessagesModel {
    rt: Runtime,
    http: reqwest::Client,
}

impl AnthropicMessagesModel {
    pub fn new(rt: Runtime) -> Result<Self, String> {
        let http = client(&rt)?;
        Ok(AnthropicMessagesModel { rt, http })
    }
}

impl Model for AnthropicMessagesModel {
    fn name(&self) -> String {
        self.rt.name()
    }

    fn api_mode(&self) -> String {
        self.rt.api_mode.as_str().to_string()
    }

    fn complete(&self, req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
        let rt = self.rt.clone();
        let http = self.http.clone();
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
            let url = format!("{}/v1/messages", rt.base_url);
            let headers = headers_with(
                &rt,
                &[
                    ("x-api-key", key),
                    ("anthropic-version", VERSION.to_string()),
                ],
            );
            let resp = post(&http, &url, &headers, &body(&rt, &req)).await?;
            Ok(stream_of(resp, Decoder::default()))
        })
    }
}

fn body(rt: &Runtime, req: &ModelRequest) -> Value {
    let cache = rt.caches();
    let mut messages: Vec<Value> = Vec::new();
    for m in &req.messages {
        match m {
            // system 消息混进 transcript 时当 user：Anthropic 没有第三种角色，而丢掉它
            // 比错放更糟（那通常是一段被追加的指令）。
            Message::System { content } | Message::User { content } => {
                push_user(&mut messages, json!({"type": "text", "text": content}));
            }
            Message::UserMultimodal { .. } => unreachable!("validated transport modality"),
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut blocks = Vec::new();
                if !content.is_empty() {
                    blocks.push(json!({"type": "text", "text": content}));
                }
                for c in tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.args,
                    }));
                }
                messages.push(json!({"role": "assistant", "content": blocks}));
            }
            // 工具结果是 user 的一个块。连着的几条结果要并进**同一条** user 消息：
            // Anthropic 不接受两条相邻的 user。
            Message::Tool { call_id, content } => push_user(
                &mut messages,
                json!({"type": "tool_result", "tool_use_id": call_id, "content": content}),
            ),
        }
    }
    if cache {
        mark_last_user_turns(&mut messages);
    }
    let mut body = json!({
        "model": rt.model,
        "stream": true,
        "max_tokens": req.max_tokens,
        "messages": messages,
    });
    put_temperature(&mut body, req.temperature);
    if !req.system.is_empty() {
        // 打缓存点时 system 必须是块数组（`cache_control` 挂在块上）；不打就是一个
        // 字符串——多余的结构只会让"前缀字节级稳定"多一处可变的地方。
        body["system"] = if cache {
            json!([{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}])
        } else {
            json!(req.system)
        };
    }
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect(),
        );
    }
    body
}

/// 往最后一条 user 里追加一个块；最后一条不是 user 就新开一条。
fn push_user(messages: &mut Vec<Value>, block: Value) {
    if let Some(last) = messages.last_mut() {
        if last["role"] == "user" {
            if let Some(a) = last["content"].as_array_mut() {
                a.push(block);
                return;
            }
        }
    }
    messages.push(json!({"role": "user", "content": [block]}));
}

/// 最后 [`CACHED_USER_TURNS`] 条 user 的**最后一个块**打 `cache_control`。
fn mark_last_user_turns(messages: &mut [Value]) {
    let mut left = CACHED_USER_TURNS;
    for m in messages.iter_mut().rev() {
        if left == 0 {
            return;
        }
        if m["role"] != "user" {
            continue;
        }
        left -= 1;
        if let Some(last) = m["content"].as_array_mut().and_then(|a| a.last_mut()) {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
    }
}

/// 一个还开着的 content block。
enum Block {
    Text,
    Tool {
        id: String,
        name: String,
        json: String,
    },
}

#[derive(Default)]
struct Decoder {
    /// `index` → 这个块是什么。Anthropic 的增量全部按 index 定位。
    blocks: std::collections::BTreeMap<u64, Block>,
    input: u32,
    cached: u32,
    output: u32,
    /// `None` = 厂商还没说为什么停（流断在 `message_delta` 之前）。
    stop: Option<StopReason>,
    ended: bool,
}

impl Decode for Decoder {
    fn on_event(&mut self, ev: SseEvent) -> Result<Vec<ModelEvent>, ModelError> {
        let v = parse_json(&ev.data)?;
        // 以正文里的 `type` 为准：`event:` 行和它永远一致，而重连/代理最容易弄丢的是
        // header 那一行。
        let kind = v["type"].as_str().unwrap_or_default();
        let mut out = Vec::new();
        match kind {
            "message_start" => {
                let u = &v["message"]["usage"];
                self.input = count(u, &["input_tokens"]);
                self.cached = count(u, &["cache_read_input_tokens"]);
                self.output = count(u, &["output_tokens"]);
            }
            "content_block_start" => {
                let idx = v["index"].as_u64().unwrap_or(0);
                let b = &v["content_block"];
                let block = match b["type"].as_str() {
                    // 缺字段是协议错，不是一次名字为空的调用（见 `required_str`）。
                    Some("tool_use") => Block::Tool {
                        id: required_str(b, "id", "anthropic tool_use")?.to_string(),
                        name: required_str(b, "name", "anthropic tool_use")?.to_string(),
                        json: String::new(),
                    },
                    _ => Block::Text,
                };
                self.blocks.insert(idx, block);
            }
            "content_block_delta" => {
                let idx = v["index"].as_u64().unwrap_or(0);
                let d = &v["delta"];
                match d["type"].as_str() {
                    Some("text_delta") => {
                        if let Some(t) = d["text"].as_str() {
                            if !t.is_empty() {
                                out.push(ModelEvent::TextDelta(t.to_string()));
                            }
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(Block::Tool { json, .. }) = self.blocks.get_mut(&idx) {
                            json.push_str(d["partial_json"].as_str().unwrap_or_default());
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let idx = v["index"].as_u64().unwrap_or(0);
                if let Some(Block::Tool { id, name, json }) = self.blocks.remove(&idx) {
                    out.push(ModelEvent::ToolCall(tool_call(&id, &name, &json)?));
                }
            }
            "message_delta" => {
                if let Some(r) = v["delta"]["stop_reason"].as_str() {
                    self.stop = Some(stop_of(r));
                }
                // 用量在这里发（`message_stop` 之前、工具调用之后）：这条事件是厂商唯一
                // 报 output token 的地方。**没带 `usage` 就不发**——发一条三个 0 的用量
                // 和"这次没报用量"是两件事。
                let u = &v["usage"];
                if u.is_object() {
                    self.output = count(u, &["output_tokens"]).max(self.output);
                    out.push(ModelEvent::Usage {
                        input: self.input,
                        output: self.output,
                        cached: self.cached,
                        reasoning: None,
                    });
                }
            }
            "message_stop" => out.append(&mut self.end()),
            // 厂商自己报的错（overloaded / api_error）：按传输错误处理，重试瀑布会重试它。
            "error" => {
                let msg = v["error"]["message"].as_str().unwrap_or("anthropic error");
                let t = v["error"]["type"].as_str().unwrap_or("error");
                return Err(ModelError::Transport(format!("{t}: {msg}")));
            }
            // `ping` 与将来新增的事件：忽略。
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(self.end())
    }
}

impl Decoder {
    fn end(&mut self) -> Vec<ModelEvent> {
        if self.ended {
            return Vec::new();
        }
        self.ended = true;
        vec![ModelEvent::Done {
            stop: self.stop.unwrap_or(StopReason::Other),
        }]
    }
}

fn stop_of(r: &str) -> StopReason {
    match r {
        "tool_use" => StopReason::ToolUse,
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ToolCall;
    use crate::providers::ApiMode;
    use std::collections::BTreeMap;

    fn rt(cache: bool) -> Runtime {
        Runtime {
            provider: "anthropic".into(),
            model: "claude".into(),
            api_mode: ApiMode::AnthropicMessages,
            base_url: "http://x".into(),
            env: "K".into(),
            api_key: Some("k".into()),
            cache: cache.then(|| "system_and_3".to_string()),
            headers: BTreeMap::new(),
            timeout_ms: 0,
            program: None,
            reasoning_effort: None,
        }
    }

    fn req(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            system: "S".into(),
            messages,
            tools: vec![],
            max_tokens: 8,
            temperature: Some(0.0),
        }
    }

    /// I2：配了就发，没配就**一个键都没有**。
    #[test]
    fn temperature_is_sent_only_when_it_is_configured() {
        let msgs = vec![Message::User {
            content: "hi".into(),
        }];
        assert_eq!(body(&rt(false), &req(msgs.clone()))["temperature"], 0.0);
        let mut r = req(msgs);
        r.temperature = None;
        assert!(body(&rt(false), &r).get("temperature").is_none());
    }

    #[test]
    fn two_tool_results_in_a_row_become_one_user_message() {
        let b = body(
            &rt(false),
            &req(vec![
                Message::Assistant {
                    content: "ok".into(),
                    tool_calls: vec![
                        ToolCall {
                            id: "a".into(),
                            name: "t".into(),
                            args: json!({}),
                        },
                        ToolCall {
                            id: "b".into(),
                            name: "t".into(),
                            args: json!({}),
                        },
                    ],
                },
                Message::Tool {
                    call_id: "a".into(),
                    content: "1".into(),
                },
                Message::Tool {
                    call_id: "b".into(),
                    content: "2".into(),
                },
            ]),
        );
        // assistant 一条 + user 一条（两个 tool_result 块）：相邻的 user 会被厂商拒掉。
        assert_eq!(b["messages"].as_array().unwrap().len(), 2);
        assert_eq!(b["messages"][1]["content"].as_array().unwrap().len(), 2);
        assert_eq!(b["messages"][1]["content"][1]["tool_use_id"], "b");
        // 不打缓存时 system 是字符串。
        assert_eq!(b["system"], "S");
    }

    #[test]
    fn the_cache_marks_at_most_three_user_turns() {
        let msgs: Vec<Message> = (0..5)
            .flat_map(|i| {
                vec![
                    Message::User {
                        content: format!("u{i}"),
                    },
                    Message::Assistant {
                        content: format!("a{i}"),
                        tool_calls: vec![],
                    },
                ]
            })
            .collect();
        let b = body(&rt(true), &req(msgs));
        let marked: Vec<String> = b["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["content"][0].get("cache_control").is_some())
            .map(|m| m["content"][0]["text"].as_str().unwrap_or_default().into())
            .collect();
        assert_eq!(marked, vec!["u2", "u3", "u4"]);
        assert_eq!(b["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn a_message_delta_without_usage_does_not_invent_one() {
        let mut d = Decoder::default();
        let out = d
            .on_event(SseEvent {
                event: None,
                data: "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}"
                    .into(),
            })
            .unwrap();
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn a_tool_use_block_without_a_name_is_a_protocol_error() {
        let mut d = Decoder::default();
        let e = d
            .on_event(SseEvent {
                event: None,
                data: "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\"}}".into(),
            })
            .unwrap_err();
        assert!(matches!(e, ModelError::Protocol(_)), "{e:?}");
        assert!(e.to_string().contains("name"), "{e}");
    }

    #[test]
    fn an_error_event_is_a_transport_error_so_the_waterfall_retries_it() {
        let mut d = Decoder::default();
        let e = d
            .on_event(SseEvent {
                event: Some("error".into()),
                data: "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}".into(),
            })
            .unwrap_err();
        assert!(matches!(e, ModelError::Transport(_)), "{e:?}");
        assert!(e.to_string().contains("overloaded_error"), "{e}");
    }
}
