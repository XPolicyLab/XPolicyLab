//! 四种 wire（规格 §7.2）：**发出去的请求长什么样**，以及**回来的流翻成了哪串
//! [`ModelEvent`]**。
//!
//! 每个用例都打本地 mock（`http://127.0.0.1:<内核分配的端口>` / `ws://…`），不碰外网、
//! 不需要任何 API key，也就不需要 TLS——这台机器上 rustls 编不过（见
//! `crates/oeh-brain/Cargo.toml` 的 `tls` feature）。
//!
//! 四个传输的期望事件序列**是同一串**：`TextDelta("Hel")`, `TextDelta("lo")`,
//! `ToolCall{robot_walk,{vx:0.1}}`, `Usage{10,5,3}`, `Done{ToolUse}`。厂商的线格式差得
//! 很远（增量拼接的位置、用量在哪条事件上、停止理由怎么说），而循环那一侧只该看见这
//! 一串——所以断言的是整串，不是"包含某几条"。

mod mock_http;

use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::StreamExt;
use mock_http::ws::MockWs;
use mock_http::{MockHttp, Reply};
use oeh_brain::model::{
    Message, Model, ModelError, ModelEvent, ModelRequest, StopReason, ToolCall, ToolSpec,
};
use oeh_brain::providers::{ApiMode, Runtime};
use oeh_brain::transport::{
    AnthropicMessagesModel, ChatCompletionsModel, RealtimeModel, ResponsesModel,
};
use oeh_kernel::{ManualClock, WorldLog};
use serde_json::{json, Value};

// ---------------------------------------------------------------- 公共夹具

fn rt(api_mode: ApiMode, base_url: String) -> Runtime {
    Runtime {
        provider: "mock".into(),
        model: "m1".into(),
        api_mode,
        base_url,
        env: "OEH_TEST_UNUSED_KEY".into(),
        api_key: Some("k".into()),
        cache: None,
        headers: BTreeMap::new(),
        timeout_ms: 5_000,
        program: None,
        reasoning_effort: None,
    }
}

/// 一次带着 transcript（user → assistant+tool_call → tool 结果 → user）的请求：四种
/// wire 对"上一轮的工具调用与结果"的表示法各不相同，只有真的带上它们才测得到。
fn req() -> ModelRequest {
    ModelRequest {
        system: "SYSTEM".into(),
        messages: vec![
            Message::User {
                content: "walk".into(),
            },
            Message::Assistant {
                content: "ok".into(),
                tool_calls: vec![ToolCall {
                    id: "prev1".into(),
                    name: "robot_walk".into(),
                    args: json!({"vx": 0.2}),
                }],
            },
            Message::Tool {
                call_id: "prev1".into(),
                content: "{\"outcome\":\"executed\"}".into(),
            },
            Message::User {
                content: "again".into(),
            },
        ],
        tools: vec![ToolSpec {
            name: "robot_walk".into(),
            description: "走两步".into(),
            parameters: json!({"type":"object","properties":{"vx":{"type":"number"}}}),
            concurrency_safe: false,
        }],
        max_tokens: 256,
        temperature: Some(0.2),
    }
}

/// 四个传输共同的期望。`call_id` 是各家自己给的 id——它必须原样传出来（工具结果要带着
/// 同一个 id 回去）。
fn expected(call_id: &str) -> Vec<ModelEvent> {
    vec![
        ModelEvent::TextDelta("Hel".into()),
        ModelEvent::TextDelta("lo".into()),
        ModelEvent::ToolCall(ToolCall {
            id: call_id.into(),
            name: "robot_walk".into(),
            args: json!({"vx": 0.1}),
        }),
        ModelEvent::Usage {
            input: 10,
            output: 5,
            cached: 3,
            reasoning: None,
        },
        ModelEvent::Done {
            stop: StopReason::ToolUse,
        },
    ]
}

async fn drain(m: &dyn Model) -> Vec<ModelEvent> {
    let mut s = m.complete(req()).await.expect("complete");
    let mut out = Vec::new();
    while let Some(e) = s.next().await {
        out.push(e.expect("流上不该有错"));
    }
    out
}

#[tokio::test]
async fn unsupported_image_transports_reject_before_sending_text_only_requests() {
    let models: Vec<Box<dyn Model>> = vec![
        Box::new(
            ChatCompletionsModel::new(rt(ApiMode::ChatCompletions, "http://127.0.0.1:9".into()))
                .unwrap(),
        ),
        Box::new(
            AnthropicMessagesModel::new(rt(
                ApiMode::AnthropicMessages,
                "http://127.0.0.1:9".into(),
            ))
            .unwrap(),
        ),
        Box::new(RealtimeModel::new(rt(ApiMode::Realtime, "ws://127.0.0.1:9".into())).unwrap()),
    ];
    for model in models {
        let mut request = req();
        request.messages = vec![Message::UserMultimodal {
            content: "current camera".into(),
            images: vec![oeh_brain::model::ImageInput {
                camera: "head".into(),
                data_url: "data:image/png;base64,aGVsbG8=".into(),
            }],
        }];
        assert!(matches!(
            model.complete(request).await,
            Err(ModelError::Protocol(_))
        ));
    }
}

// ---------------------------------------------------------------- chat_completions

const CHAT_SSE: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"robot_walk\",\"arguments\":\"{\\\"vx\\\":\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"0.1}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn chat_completions_joins_tool_call_fragments_and_reports_usage() {
    let srv = MockHttp::start(vec![Reply::sse(CHAT_SSE)]).await;
    let m = ChatCompletionsModel::new(rt(ApiMode::ChatCompletions, srv.base_url())).unwrap();
    assert_eq!(m.name(), "mock/m1");
    assert_eq!(m.api_mode(), "chat_completions");
    assert_eq!(drain(&m).await, expected("call_1"));

    let r = srv.last();
    assert_eq!(r.path, "/chat/completions");
    assert_eq!(
        r.headers.get("authorization").map(String::as_str),
        Some("Bearer k")
    );
    let b = &r.body;
    assert_eq!(b["model"], "m1");
    assert_eq!(b["stream"], true);
    assert_eq!(b["max_tokens"], 256);
    // 工具按 OpenAI 的 function 格式。
    assert_eq!(b["tools"][0]["type"], "function");
    assert_eq!(b["tools"][0]["function"]["name"], "robot_walk");
    assert_eq!(b["tools"][0]["function"]["parameters"]["type"], "object");
    // system 是 messages 的第一条；上一轮的工具调用与结果各有各的位置。
    assert_eq!(b["messages"][0]["role"], "system");
    assert_eq!(b["messages"][0]["content"], "SYSTEM");
    assert_eq!(b["messages"][2]["role"], "assistant");
    assert_eq!(b["messages"][2]["tool_calls"][0]["id"], "prev1");
    assert_eq!(
        b["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{\"vx\":0.2}"
    );
    assert_eq!(b["messages"][3]["role"], "tool");
    assert_eq!(b["messages"][3]["tool_call_id"], "prev1");
}

#[tokio::test]
async fn an_sse_event_split_across_chunks_is_still_one_event() {
    // 分片边界落在一行的中间（`{"vx":` 与 `0.1}` 之间、甚至 `data:` 与 JSON 之间）。
    let mut chunks = Vec::new();
    let mut rest = CHAT_SSE;
    while rest.len() > 17 {
        let (a, b) = rest.split_at(17);
        chunks.push(a.to_string());
        rest = b;
    }
    chunks.push(rest.to_string());
    let srv = MockHttp::start(vec![Reply::Sse(chunks)]).await;
    let m = ChatCompletionsModel::new(rt(ApiMode::ChatCompletions, srv.base_url())).unwrap();
    assert_eq!(drain(&m).await, expected("call_1"));
}

#[tokio::test]
async fn a_5xx_is_an_http_error_that_carries_the_body() {
    let srv = MockHttp::start(vec![Reply::Status(500, "upstream boom".into())]).await;
    let m = ChatCompletionsModel::new(rt(ApiMode::ChatCompletions, srv.base_url())).unwrap();
    match m.complete(req()).await {
        Err(ModelError::Http { status, body }) => {
            assert_eq!(status, 500);
            assert!(body.contains("upstream boom"), "{body}");
        }
        Err(other) => panic!("期望 Http{{500}}，拿到 {other:?}"),
        Ok(_) => panic!("500 不该还一条流"),
    }
}

#[tokio::test]
async fn a_huge_error_body_is_capped_before_it_can_reach_the_log() {
    // 一页 1 MB 的代理错误页。整段进 `reason` 的话，世界日志（没有 payload 上限）会
    // 按每次尝试一条地吃下它。
    let page = format!(
        "<html>\n<body>\n{}\n</body>\n</html>",
        "boom ".repeat(200_000)
    );
    let srv = MockHttp::start(vec![Reply::Status(502, page.clone())]).await;
    let m = ChatCompletionsModel::new(rt(ApiMode::ChatCompletions, srv.base_url())).unwrap();
    let Err(e) = m.complete(req()).await else {
        panic!("502 不该还一条流")
    };
    let ModelError::Http { status, body } = &e else {
        panic!("期望 Http，拿到 {e:?}")
    };
    assert_eq!(*status, 502);
    // 读进来的有上限，进 `reason` 的那一串还要小得多。
    assert!(body.len() <= 64 * 1024 + 32, "body {} 字节", body.len());
    assert!(body.len() < page.len());
    let reason = e.to_string();
    assert!(reason.len() < 400, "reason {} 字节: {reason}", reason.len());
    assert!(
        reason.starts_with("http 502: <html> <body> boom"),
        "{reason}"
    );
}

// ---------------------------------------------------------------- anthropic_messages

const ANTHROPIC_SSE: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":3,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"robot_walk\",\"input\":{}}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"vx\\\":\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"0.1}\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn anthropic_messages_puts_system_in_its_own_field_and_tools_take_input_schema() {
    let srv = MockHttp::start(vec![Reply::sse(ANTHROPIC_SSE)]).await;
    let m = AnthropicMessagesModel::new(rt(ApiMode::AnthropicMessages, srv.base_url())).unwrap();
    assert_eq!(m.api_mode(), "anthropic_messages");
    assert_eq!(drain(&m).await, expected("toolu_1"));

    let r = srv.last();
    assert_eq!(r.path, "/v1/messages");
    // Anthropic 用 `x-api-key`，不是 Bearer；版本头是必填的。
    assert_eq!(r.headers.get("x-api-key").map(String::as_str), Some("k"));
    assert!(
        r.headers.contains_key("anthropic-version"),
        "{:?}",
        r.headers
    );
    let b = &r.body;
    assert_eq!(b["system"], "SYSTEM");
    assert_eq!(b["stream"], true);
    assert_eq!(b["tools"][0]["name"], "robot_walk");
    assert_eq!(b["tools"][0]["input_schema"]["type"], "object");
    // 上一轮：assistant 的 tool_use 块 + 下一条 user 里的 tool_result 块。
    assert_eq!(b["messages"][1]["role"], "assistant");
    assert_eq!(b["messages"][1]["content"][1]["type"], "tool_use");
    assert_eq!(b["messages"][1]["content"][1]["id"], "prev1");
    assert_eq!(b["messages"][1]["content"][1]["input"], json!({"vx": 0.2}));
    assert_eq!(b["messages"][2]["role"], "user");
    assert_eq!(b["messages"][2]["content"][0]["type"], "tool_result");
    assert_eq!(b["messages"][2]["content"][0]["tool_use_id"], "prev1");
}

#[tokio::test]
async fn the_cache_mode_marks_the_system_prefix_and_the_last_user_turns() {
    let srv = MockHttp::start(vec![Reply::sse(ANTHROPIC_SSE)]).await;
    let mut r = rt(ApiMode::AnthropicMessages, srv.base_url());
    r.cache = Some("system_and_3".into());
    let m = AnthropicMessagesModel::new(r).unwrap();
    drain(&m).await;

    let b = srv.last().body;
    // 打了缓存点的 system 必须是块数组（cache_control 挂在块上）。
    assert_eq!(b["system"][0]["type"], "text");
    assert_eq!(b["system"][0]["text"], "SYSTEM");
    assert_eq!(b["system"][0]["cache_control"]["type"], "ephemeral");
    // 这份 transcript 里只有两条 user 消息：`walk`，以及 `tool_result` + `again`
    // 合并成的那一条（相邻的 user 会被厂商拒掉，见 `push_user`）。两条都在最后三条
    // 之内，所以两条都打了点。
    assert_eq!(b["messages"].as_array().unwrap().len(), 3);
    for i in [0, 2] {
        let msg = &b["messages"][i];
        assert_eq!(msg["role"], "user", "messages[{i}] = {msg}");
        let last = msg["content"].as_array().expect("content 是块数组").len() - 1;
        assert_eq!(
            msg["content"][last]["cache_control"]["type"], "ephemeral",
            "messages[{i}] = {msg}"
        );
    }
}

// ---------------------------------------------------------------- responses

const RESPONSES_SSE: &str = concat!(
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n",
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\n",
    "event: response.output_item.done\n",
    "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"fc_1\",\"name\":\"robot_walk\",\"arguments\":\"{\\\"vx\\\":0.1}\"}}\n\n",
    "event: response.completed\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"input_tokens_details\":{\"cached_tokens\":3}}}}\n\n",
);

#[tokio::test]
async fn responses_turns_the_transcript_into_input_items() {
    let srv = MockHttp::start(vec![Reply::sse(RESPONSES_SSE)]).await;
    let m = ResponsesModel::new(rt(ApiMode::Responses, srv.base_url())).unwrap();
    assert_eq!(m.api_mode(), "responses");
    assert_eq!(drain(&m).await, expected("fc_1"));

    let r = srv.last();
    assert_eq!(r.path, "/responses");
    let b = &r.body;
    assert_eq!(b["instructions"], "SYSTEM");
    assert_eq!(b["stream"], true);
    assert_eq!(b["tools"][0]["type"], "function");
    assert_eq!(b["tools"][0]["name"], "robot_walk");
    // input items：user message → assistant message → function_call → function_call_output
    // → user message。assistant 的文本和它要求的调用是**两个** item。
    assert_eq!(b["input"][0]["type"], "message");
    assert_eq!(b["input"][0]["role"], "user");
    assert_eq!(b["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(b["input"][1]["role"], "assistant");
    assert_eq!(b["input"][1]["content"][0]["type"], "output_text");
    assert_eq!(b["input"][2]["type"], "function_call");
    assert_eq!(b["input"][2]["call_id"], "prev1");
    assert_eq!(b["input"][2]["arguments"], "{\"vx\":0.2}");
    assert_eq!(b["input"][3]["type"], "function_call_output");
    assert_eq!(b["input"][3]["call_id"], "prev1");
    assert_eq!(b["input"][4]["role"], "user");
}

// ---------------------------------------------------------------- realtime

/// 真实的 OpenAI Realtime（beta v1）载荷形状。
///
/// **工具名只在 `response.output_item.added` 的 item 上**：
/// `response.function_call_arguments.delta/done` 带的是 `{item_id, output_index, call_id,
/// delta|arguments}`，没有 `name`。替身照这个形状发，测试才是在给对的东西背书。
fn realtime_script() -> Vec<Value> {
    vec![
        json!({"type": "response.text.delta", "response_id": "resp_1", "item_id": "item_0", "output_index": 0, "delta": "Hel"}),
        json!({"type": "response.text.delta", "response_id": "resp_1", "item_id": "item_0", "output_index": 0, "delta": "lo"}),
        // 语音转写：不是模型事件，而是一条世界日志里的观测。
        json!({"type": "conversation.item.input_audio_transcription.completed", "item_id": "item_u", "content_index": 0, "transcript": "hi"}),
        json!({"type": "response.output_item.added", "response_id": "resp_1", "output_index": 1,
               "item": {"id": "item_1", "object": "realtime.item", "type": "function_call",
                        "status": "in_progress", "name": "robot_walk", "call_id": "call_1", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "response_id": "resp_1", "item_id": "item_1", "output_index": 1, "call_id": "call_1", "delta": "{\"vx\":"}),
        json!({"type": "response.function_call_arguments.done", "response_id": "resp_1", "item_id": "item_1", "output_index": 1, "call_id": "call_1", "arguments": "{\"vx\":0.1}"}),
        json!({"type": "response.output_item.done", "response_id": "resp_1", "output_index": 1,
               "item": {"id": "item_1", "object": "realtime.item", "type": "function_call",
                        "status": "completed", "name": "robot_walk", "call_id": "call_1", "arguments": "{\"vx\":0.1}"}}),
        json!({"type": "response.done", "response": {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5, "input_token_details": {"cached_tokens": 3}}}}),
    ]
}

#[tokio::test]
async fn realtime_sends_session_update_then_streams_and_writes_obs_speech() {
    let srv = MockWs::start(realtime_script()).await;
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let m = RealtimeModel::new(rt(ApiMode::Realtime, srv.url()))
        .unwrap()
        .with_world_log(log.clone());
    assert_eq!(m.api_mode(), "realtime");
    assert_eq!(drain(&m).await, expected("call_1"));

    // 客户端这一侧：session.update（工具 + instructions）→ 每条消息一个
    // conversation.item.create → response.create。
    let got = srv.received();
    assert_eq!(got[0]["type"], "session.update");
    assert_eq!(got[0]["session"]["instructions"], "SYSTEM");
    assert_eq!(got[0]["session"]["tools"][0]["name"], "robot_walk");
    assert_eq!(got[0]["session"]["tools"][0]["type"], "function");
    // 采样参数：realtime 的 temperature 是会话级的，max_output_tokens 是 response 级的。
    assert_eq!(got[0]["session"]["temperature"], 0.2);
    assert_eq!(got[1]["type"], "conversation.item.create");
    assert_eq!(got[1]["item"]["content"][0]["text"], "walk");
    assert_eq!(
        got.iter()
            .find(|m| m["item"]["type"] == "function_call_output")
            .map(|m| m["item"]["call_id"].clone()),
        Some(json!("prev1"))
    );
    let last = got.last().expect("至少有一条");
    assert_eq!(last["type"], "response.create");
    assert_eq!(last["response"]["max_output_tokens"], 256);
    // 握手带着 Authorization。
    assert_eq!(
        srv.handshake_headers()
            .get("authorization")
            .map(String::as_str),
        Some("Bearer k")
    );

    // 转写进了世界日志——别的脑与回放也因此看得见（规格 §7.2 的音频行）。
    let speech: Vec<Value> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "obs/speech")
        .map(|e| e.payload)
        .collect();
    assert_eq!(speech, vec![json!({"text": "hi"})]);
}
