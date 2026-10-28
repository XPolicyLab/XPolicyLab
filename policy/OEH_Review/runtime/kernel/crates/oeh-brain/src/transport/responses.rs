//! `responses`：OpenAI 的 Responses API。
//!
//! `POST {base_url}/responses`。transcript 不是"消息数组"而是一串 **item**：一条 user
//! 消息是一个 `message` item，assistant 要求的一次工具调用是一个 `function_call` item，
//! 工具结果是一个 `function_call_output` item——也就是说 assistant 的一段话和它要求的
//! 调用是**两个** item，而不是一条消息的两个字段。system 走 `instructions`。
//!
//! 流回来的是有名字的事件：`response.output_text.delta`（文本）、
//! `response.output_item.done`（一个 item 完成——工具调用的参数在这里已经是完整的一段
//! JSON 文本了，不用自己拼）、`response.completed`（用量 + 收尾）。

use serde_json::{json, Value};

use super::{
    client, compact, count, headers_with, parse_json, post, put_temperature, required_str,
    stream_of, tool_call, Decode,
};
use crate::model::{
    BoxFuture, EventStream, Message, Model, ModelError, ModelEvent, ModelRequest, StopReason,
};
use crate::providers::Runtime;
use crate::transport::sse::SseEvent;

pub struct ResponsesModel {
    rt: Runtime,
    http: reqwest::Client,
}

impl ResponsesModel {
    pub fn new(rt: Runtime) -> Result<Self, String> {
        let http = client(&rt)?;
        Ok(ResponsesModel { rt, http })
    }
}

impl Model for ResponsesModel {
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
            for message in &req.messages {
                if let Message::UserMultimodal { images, .. } = message {
                    if images.is_empty() || images.len() > 3 {
                        return Err(ModelError::Protocol(
                            "expected one to three camera images".into(),
                        ));
                    }
                    for image in images {
                        image.validate()?;
                    }
                }
            }

            let Some(key) = rt.api_key.clone() else {
                return Err(ModelError::MissingKey(rt.env.clone()));
            };
            let url = format!("{}/responses", rt.base_url);
            let headers = headers_with(&rt, &[("authorization", format!("Bearer {key}"))]);
            let resp = post(&http, &url, &headers, &body(&rt, &req)).await?;
            Ok(stream_of(resp, Decoder::default()))
        })
    }
}

fn body(rt: &Runtime, req: &ModelRequest) -> Value {
    let mut input: Vec<Value> = Vec::new();
    for m in &req.messages {
        match m {
            Message::System { content } => input.push(text_item("system", "input_text", content)),
            Message::User { content } => input.push(text_item("user", "input_text", content)),
            Message::UserMultimodal { content, images } => {
                let mut parts = vec![json!({"type": "input_text", "text": content})];
                for image in images {
                    parts.push(
                        json!({"type": "input_text", "text": format!("Camera: {}", image.camera)}),
                    );
                    parts.push(json!({"type": "input_image", "image_url": image.data_url, "detail": "auto"}));
                }
                input.push(json!({"type": "message", "role": "user", "content": parts}));
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                if !content.is_empty() {
                    input.push(text_item("assistant", "output_text", content));
                }
                for c in tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": c.id,
                        "name": c.name,
                        "arguments": compact(&c.args),
                    }));
                }
            }
            Message::Tool { call_id, content } => input.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            })),
        }
    }
    let mut body = json!({
        "model": rt.model,
        "stream": true,
        "max_output_tokens": req.max_tokens,
        "input": input,
    });
    // `None` 就不发这个键。OpenAI 的推理模型（README 里的例子 `gpt-5` 就是）对
    // `temperature` 一律回 400，而 400 不可重试——瀑布会一路走完整条链，每个 turn 都以
    // `done("error: http 400 …")` 收尾。一份 `temperature: null` 的声明因此不是"随便设个
    // 数"，而是这个 wire 唯一能用的配法。
    put_temperature(&mut body, req.temperature);
    if !req.system.is_empty() {
        body["instructions"] = json!(req.system);
    }
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(
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
    body
}

fn text_item(role: &str, kind: &str, text: &str) -> Value {
    json!({"type": "message", "role": role, "content": [{"type": kind, "text": text}]})
}

#[derive(Default)]
struct Decoder {
    calls: u32,
    ended: bool,
}

impl Decode for Decoder {
    fn on_event(&mut self, ev: SseEvent) -> Result<Vec<ModelEvent>, ModelError> {
        let v = parse_json(&ev.data)?;
        let kind = v["type"].as_str().unwrap_or_default();
        let mut out = Vec::new();
        match kind {
            "response.output_text.delta" => {
                if let Some(t) = v["delta"].as_str() {
                    if !t.is_empty() {
                        out.push(ModelEvent::TextDelta(t.to_string()));
                    }
                }
            }
            "response.output_item.done" => {
                let item = &v["item"];
                // 只有 function_call 是这一层关心的 item：文本的 item 已经按增量发过了，
                // 再发一次就是重复。
                if item["type"] == "function_call" {
                    // 缺字段是协议错，不是一次名字为空的调用（见 `required_str`）。
                    let call_id = required_str(item, "call_id", "responses function_call")?;
                    let name = required_str(item, "name", "responses function_call")?;
                    let args = item["arguments"].as_str().unwrap_or_default();
                    let call = tool_call(call_id, name, args)?;
                    self.calls += 1;
                    out.push(ModelEvent::ToolCall(call));
                }
            }
            "response.completed" | "response.incomplete" => {
                out.append(&mut self.usage(&v["response"]["usage"]));
                let stop = if kind == "response.incomplete" {
                    // `incomplete` 的理由只有一种要区分：截断。
                    match v["response"]["incomplete_details"]["reason"].as_str() {
                        Some("max_output_tokens") => StopReason::MaxTokens,
                        _ => StopReason::Other,
                    }
                } else if self.calls > 0 {
                    StopReason::ToolUse
                } else {
                    StopReason::EndTurn
                };
                out.append(&mut self.end(stop));
            }
            // 厂商报的失败：按传输错误处理，重试瀑布会重试它。
            "response.failed" | "error" => {
                let msg = v["response"]["error"]["message"]
                    .as_str()
                    .or_else(|| v["message"].as_str())
                    .unwrap_or("responses error");
                return Err(ModelError::Transport(msg.to_string()));
            }
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(self.end(StopReason::Other))
    }
}

impl Decoder {
    fn usage(&self, u: &Value) -> Vec<ModelEvent> {
        if !u.is_object() {
            return Vec::new();
        }
        vec![ModelEvent::Usage {
            input: count(u, &["input_tokens"]),
            output: count(u, &["output_tokens"]),
            cached: count(&u["input_tokens_details"], &["cached_tokens"]),
            reasoning: None,
        }]
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
    use crate::providers::ApiMode;
    use std::collections::BTreeMap;

    fn rt() -> Runtime {
        Runtime {
            provider: "openai".into(),
            model: "gpt-5".into(),
            api_mode: ApiMode::Responses,
            base_url: "http://x".into(),
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
    fn camera_bytes_are_image_parts_with_explicit_camera_labels() {
        let request = ModelRequest {
            system: "robot".into(),
            tools: vec![],
            max_tokens: 128,
            temperature: None,
            messages: vec![Message::UserMultimodal {
                content: "observation-1".into(),
                images: vec![crate::model::ImageInput {
                    camera: "head".into(),
                    data_url: "data:image/png;base64,aGVsbG8=".into(),
                }],
            }],
        };
        let body = body(&rt(), &request);
        let parts = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "observation-1");
        assert_eq!(parts[1]["text"], "Camera: head");
        assert_eq!(parts[2]["type"], "input_image");
        assert_eq!(parts[2]["image_url"], "data:image/png;base64,aGVsbG8=");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn an_assistant_turn_with_a_call_is_two_items() {
        let b = body(
            &rt(),
            &ModelRequest {
                system: String::new(),
                messages: vec![Message::Assistant {
                    content: "ok".into(),
                    tool_calls: vec![crate::model::ToolCall {
                        id: "c1".into(),
                        name: "t".into(),
                        args: json!({"a": 1}),
                    }],
                }],
                tools: vec![],
                max_tokens: 8,
                temperature: Some(0.0),
            },
        );
        assert_eq!(b["input"].as_array().unwrap().len(), 2);
        assert_eq!(b["input"][0]["content"][0]["type"], "output_text");
        assert_eq!(b["input"][1]["type"], "function_call");
        // system 为空时不发 `instructions`（空字符串也是一次缓存前缀的变化）。
        assert!(b.get("instructions").is_none(), "{b}");
    }

    /// I2：这是那条 Task-3 遗留的裁决落地的地方。`gpt-5` 那一档的推理模型对
    /// `temperature` 一律回 400，而 `providers/default.yaml` 里 `openai` 就是
    /// `api_mode: responses`——发不发这个字段决定了那份声明能不能用。
    #[test]
    fn temperature_is_sent_only_when_it_is_configured() {
        let mut r = ModelRequest {
            system: String::new(),
            messages: vec![Message::User {
                content: "hi".into(),
            }],
            tools: vec![],
            max_tokens: 8,
            temperature: Some(0.2),
        };
        assert_eq!(body(&rt(), &r)["temperature"], 0.2);
        r.temperature = None;
        assert!(body(&rt(), &r).get("temperature").is_none());
    }

    #[test]
    fn a_function_call_item_without_a_name_is_a_protocol_error() {
        let mut d = Decoder::default();
        let e = d
            .on_event(SseEvent {
                event: None,
                data: "{\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"fc_1\",\"arguments\":\"{}\"}}".into(),
            })
            .unwrap_err();
        assert!(matches!(e, ModelError::Protocol(_)), "{e:?}");
        assert!(e.to_string().contains("name"), "{e}");
    }

    #[test]
    fn an_incomplete_response_stops_on_max_tokens() {
        let mut d = Decoder::default();
        let out = d
            .on_event(SseEvent {
                event: None,
                data: "{\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}".into(),
            })
            .unwrap();
        assert_eq!(
            out,
            vec![ModelEvent::Done {
                stop: StopReason::MaxTokens
            }]
        );
    }
}
