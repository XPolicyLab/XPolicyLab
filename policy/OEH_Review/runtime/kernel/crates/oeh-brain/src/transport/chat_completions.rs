//! `chat_completions`：OpenAI 兼容的那套（DeepSeek、vLLM、绝大多数本地网关都说它）。
//!
//! `POST {base_url}/chat/completions`，`stream: true`，工具是 `type: "function"` 的函数
//! 声明。回来的 SSE 每条是一个 `choices[0].delta`：
//!
//! - 文本在 `delta.content`；
//! - 工具调用是**按 `index` 增量拼**的——`id` 与 `name` 只在第一片里出现，`arguments`
//!   是一串 JSON 文本片段（`{"vx":` + `0.1}`）。拼完之前它不是一次合法的调用，所以这里
//!   等到 `finish_reason` 才把整批发出去；
//! - 用量在 `choices` 为空的那条上（要 `stream_options.include_usage`）；
//! - 最后一条是 `data: [DONE]`，不是 JSON。

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::{
    client, compact, count, headers_with, parse_json, post, put_temperature, stream_of, tool_call,
    Decode,
};
use crate::model::{
    BoxFuture, EventStream, Message, Model, ModelError, ModelEvent, ModelRequest, StopReason,
};
use crate::providers::Runtime;
use crate::transport::sse::SseEvent;

pub struct ChatCompletionsModel {
    rt: Runtime,
    http: reqwest::Client,
}

impl ChatCompletionsModel {
    pub fn new(rt: Runtime) -> Result<Self, String> {
        let http = client(&rt)?;
        Ok(ChatCompletionsModel { rt, http })
    }
}

impl Model for ChatCompletionsModel {
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
            let url = format!("{}/chat/completions", rt.base_url);
            let headers = headers_with(&rt, &[("authorization", format!("Bearer {key}"))]);
            let resp = post(&http, &url, &headers, &body(&rt, &req)).await?;
            Ok(stream_of(resp, Decoder::default()))
        })
    }
}

fn body(rt: &Runtime, req: &ModelRequest) -> Value {
    let mut messages = Vec::new();
    if !req.system.is_empty() {
        messages.push(json!({"role": "system", "content": req.system}));
    }
    for m in &req.messages {
        messages.push(match m {
            Message::System { content } => json!({"role": "system", "content": content}),
            Message::User { content } => json!({"role": "user", "content": content}),
            Message::UserMultimodal { .. } => unreachable!("validated transport modality"),
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut v = json!({"role": "assistant", "content": content});
                if !tool_calls.is_empty() {
                    v["tool_calls"] = Value::Array(
                        tool_calls
                            .iter()
                            .map(|c| {
                                json!({
                                    "id": c.id,
                                    "type": "function",
                                    "function": {"name": c.name, "arguments": compact(&c.args)},
                                })
                            })
                            .collect(),
                    );
                }
                v
            }
            Message::Tool { call_id, content } => {
                json!({"role": "tool", "tool_call_id": call_id, "content": content})
            }
        });
    }
    let mut body = json!({
        "model": rt.model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": req.max_tokens,
        "messages": messages,
    });
    put_temperature(&mut body, req.temperature);
    // 没有工具就不发 `tools` 字段：一部分网关会拒一个空数组。
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    })
                })
                .collect(),
        );
    }
    body
}

/// 一个还没拼完的工具调用。
#[derive(Default)]
struct Partial {
    id: String,
    name: String,
    args: String,
}

#[derive(Default)]
struct Decoder {
    /// 按 `index` 攒——同一条 delta 里可以有好几个调用的片段。
    calls: BTreeMap<u64, Partial>,
    stop: Option<StopReason>,
    ended: bool,
}

impl Decoder {
    /// 把攒到的调用一次性发出去（`finish_reason` 那一刻，或者流断了）。
    fn flush(&mut self) -> Result<Vec<ModelEvent>, ModelError> {
        let mut out = Vec::new();
        for (_, p) in std::mem::take(&mut self.calls) {
            out.push(ModelEvent::ToolCall(tool_call(&p.id, &p.name, &p.args)?));
        }
        Ok(out)
    }

    fn end(&mut self) -> Result<Vec<ModelEvent>, ModelError> {
        if self.ended {
            return Ok(Vec::new());
        }
        self.ended = true;
        let mut out = self.flush()?;
        out.push(ModelEvent::Done {
            stop: self.stop.unwrap_or(StopReason::Other),
        });
        Ok(out)
    }
}

impl Decode for Decoder {
    fn on_event(&mut self, ev: SseEvent) -> Result<Vec<ModelEvent>, ModelError> {
        if ev.data.trim() == "[DONE]" {
            return self.end();
        }
        let v = parse_json(&ev.data)?;
        let mut out = Vec::new();
        let choice = &v["choices"][0];
        if let Some(t) = choice["delta"]["content"].as_str() {
            if !t.is_empty() {
                out.push(ModelEvent::TextDelta(t.to_string()));
            }
        }
        if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
            for c in calls {
                let idx = c["index"].as_u64().unwrap_or(0);
                let p = self.calls.entry(idx).or_default();
                if let Some(id) = c["id"].as_str() {
                    p.id = id.to_string();
                }
                if let Some(n) = c["function"]["name"].as_str() {
                    p.name = n.to_string();
                }
                if let Some(a) = c["function"]["arguments"].as_str() {
                    p.args.push_str(a);
                }
            }
        }
        if let Some(r) = choice["finish_reason"].as_str() {
            self.stop = Some(match r {
                "tool_calls" | "function_call" => StopReason::ToolUse,
                "stop" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                _ => StopReason::Other,
            });
            // 工具调用在这里就拼完了：晚发一条事件就是晚一步执行。
            out.append(&mut self.flush()?);
        }
        if v["usage"].is_object() {
            let u = &v["usage"];
            out.push(ModelEvent::Usage {
                input: count(u, &["prompt_tokens"]),
                output: count(u, &["completion_tokens"]),
                cached: count(&u["prompt_tokens_details"], &["cached_tokens"]),
                reasoning: None,
            });
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<ModelEvent>, ModelError> {
        // 没等到 `[DONE]` 就断了：已经拼完的调用和一条 `Done` 还是要有——上层靠 `Done`
        // 判断"这一轮说完了"，而拼完的调用是这次 step 唯一的产出。
        self.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ToolCall, ToolSpec};
    use serde_json::json;

    fn req() -> ModelRequest {
        ModelRequest {
            system: "S".into(),
            messages: vec![Message::Assistant {
                content: "ok".into(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "robot_walk".into(),
                    args: json!({"vx": 0.1}),
                }],
            }],
            tools: vec![],
            max_tokens: 8,
            temperature: Some(0.0),
        }
    }

    fn rt() -> Runtime {
        Runtime {
            provider: "p".into(),
            model: "m".into(),
            api_mode: crate::providers::ApiMode::ChatCompletions,
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
    fn an_empty_tool_table_leaves_the_field_out() {
        let b = body(&rt(), &req());
        assert!(b.get("tools").is_none(), "{b}");
        // 上一轮的工具调用：参数是**字符串**，不是对象。
        assert_eq!(
            b["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"vx\":0.1}"
        );
    }

    #[test]
    fn tools_become_openai_functions() {
        let mut r = req();
        r.tools = vec![ToolSpec {
            name: "robot_walk".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
            concurrency_safe: true,
        }];
        let b = body(&rt(), &r);
        assert_eq!(b["tools"][0]["function"]["name"], "robot_walk");
        // `concurrency_safe` 是本地调度用的，不上线。
        assert!(b["tools"][0]["function"].get("concurrency_safe").is_none());
    }

    /// I2：配了就发，没配就**一个键都没有**——不是发一个编出来的默认值。
    #[test]
    fn temperature_is_sent_only_when_it_is_configured() {
        assert_eq!(body(&rt(), &req())["temperature"], 0.0);
        let mut r = req();
        r.temperature = None;
        assert!(body(&rt(), &r).get("temperature").is_none());
    }

    #[test]
    fn a_stream_that_stops_mid_arguments_is_a_protocol_error() {
        let mut d = Decoder::default();
        d.on_event(SseEvent {
            event: None,
            data: "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"t\",\"arguments\":\"{\"}}]}}]}".into(),
        })
        .unwrap();
        assert!(matches!(d.finish(), Err(ModelError::Protocol(_))));
    }
}
