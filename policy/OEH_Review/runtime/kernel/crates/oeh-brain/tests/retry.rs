//! 重试与 fallback 瀑布：一次 step 的模型调用失败之后，谁重试、等多久、什么时候换
//! runtime，以及这些决定在世界日志里长什么样。
//!
//! 两层：
//!
//! 1. [`complete_with_retry`] 自己——`sleep` 注入成"立刻返回"，所以退避的**时长**是被
//!    断言的数字（`Retry{delay_ms}` 与传给 `sleep` 的 `Duration`），不是被等出来的墙钟；
//! 2. 接进 turn 循环之后——真内核、真 `Brains`、真 `TurnHandle`，断言 `brain/request-error`
//!    / `brain/retry` / `brain/fallback` 三条事件带着这个 turn 的 `trace_id`。

mod mock_http;

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use mock_http::{MockHttp, Reply};
use oeh_brain::model::{
    BoxFuture, EventStream, Model, ModelError, ModelEvent, ModelRequest, StopReason, ToolCall,
};
use oeh_brain::native::{NativeBrain, NativeConfig};
use oeh_brain::providers::{build_model, ApiMode, Providers, Runtime};
use oeh_brain::retry::{
    complete_with_retry, complete_with_retry_observed, RetryEvent, RetryPolicy,
};
use oeh_brain::tools::testing::recording_registry;
use oeh_brain::tools::ToolRegistry;
use oeh_brain::transport::ChatCompletionsModel;
use oeh_kernel::brains::{Brains, BrainsConfig};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::{ManualClock, WorldLog};
use oeh_proto::{EventDraft, EventKind, WorldEvent};
use parking_lot::Mutex;
use serde_json::{json, Value};

// ---------------------------------------------------------------- 假模型

/// 按剧本失败的模型：`plan` 里每一项是这一次调用的结局（`Some(err)` = 失败，`None` =
/// 成功），用完之后一律成功。
struct Flaky {
    name: String,
    plan: Mutex<VecDeque<Option<ModelError>>>,
    /// 第一次**成功**时要求的工具调用（只要求一次，之后还纯文本）。多 step 的用例靠它
    /// 让 turn 走到下一步。
    tool: Mutex<Option<String>>,
    calls: AtomicU64,
}

impl Flaky {
    fn new(name: &str, plan: Vec<Option<ModelError>>) -> Arc<Flaky> {
        Arc::new(Flaky {
            name: name.into(),
            plan: Mutex::new(plan.into()),
            tool: Mutex::new(None),
            calls: AtomicU64::new(0),
        })
    }
    fn ok(name: &str) -> Arc<Flaky> {
        Flaky::new(name, vec![])
    }
    /// 一次就成功，并且第一次成功时要求调一次 `tool`。
    fn calling(name: &str, tool: &str) -> Arc<Flaky> {
        let m = Flaky::new(name, vec![]);
        *m.tool.lock() = Some(tool.to_string());
        m
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Model for Flaky {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn complete(&self, _req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // 弹出发生在同步的那一刻：并发调用时"第 n 次调用拿第 n 步"必须由调用顺序决定。
        let step = self.plan.lock().pop_front().flatten();
        let tool = if step.is_none() {
            self.tool.lock().take()
        } else {
            None
        };
        let name = self.name.clone();
        Box::pin(async move {
            match (step, tool) {
                (Some(e), _) => Err(e),
                (None, Some(t)) => Ok(call_stream(&t)),
                (None, None) => Ok(text_stream(&name)),
            }
        })
    }
}

fn text_stream(name: &str) -> EventStream {
    let evs = vec![
        Ok(ModelEvent::TextDelta(format!("hi from {name}"))),
        Ok(ModelEvent::Done {
            stop: oeh_brain::model::StopReason::EndTurn,
        }),
    ];
    Box::pin(futures_util::stream::iter(evs))
}

/// 一次工具调用 + `Done{ToolUse}`：turn 因此会走到下一个 step。
fn call_stream(tool: &str) -> EventStream {
    let evs = vec![
        Ok(ModelEvent::ToolCall(ToolCall {
            id: "c1".into(),
            name: tool.into(),
            args: serde_json::json!({}),
        })),
        Ok(ModelEvent::Done {
            stop: StopReason::ToolUse,
        }),
    ];
    Box::pin(futures_util::stream::iter(evs))
}

fn http(status: u16) -> Option<ModelError> {
    Some(ModelError::Http {
        status,
        body: "boom".into(),
    })
}

fn req() -> ModelRequest {
    ModelRequest {
        system: "s".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 16,
        temperature: None,
    }
}

#[tokio::test]
async fn provider_observer_counts_dispatch_not_a_retry_cancelled_in_backoff() {
    let model = Flaky::new("fixture/model", vec![http(503)]);
    let chain: Vec<Arc<dyn Model>> = vec![model.clone()];
    let attempts = Mutex::new(Vec::new());
    let events = Mutex::new(Vec::new());
    let observe = |name: &str, attempt| attempts.lock().push((name.to_string(), attempt));
    let event = |value| events.lock().push(value);
    let sleep = |_duration: Duration| -> BoxFuture<()> { Box::pin(std::future::pending()) };
    let policy = RetryPolicy {
        max_attempts: 3,
        base_ms: 1,
        cap_ms: 1,
    };
    let result = tokio::time::timeout(
        Duration::from_millis(10),
        complete_with_retry_observed(&chain, req(), &policy, &sleep, &event, &observe),
    )
    .await;
    assert!(result.is_err());
    assert_eq!(model.calls(), 1);
    assert_eq!(*attempts.lock(), vec![("fixture/model".to_string(), 1)]);
    assert!(events
        .lock()
        .iter()
        .any(|value| matches!(value, RetryEvent::Retry { .. })));
}

/// 跑一次 `complete_with_retry`，收下事件与每次 `sleep` 被要求等的毫秒数。
async fn run(
    chain: &[Arc<dyn Model>],
    policy: &RetryPolicy,
) -> (Vec<RetryEvent>, Vec<u64>, Result<(), ModelError>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let slept = Arc::new(Mutex::new(Vec::new()));
    let e = events.clone();
    let s = slept.clone();
    // 注入的 sleep **立刻返回**：退避的时长是被断言的数字，不是被等出来的。
    let sleep = move |d: Duration| -> BoxFuture<()> {
        s.lock().push(d.as_millis() as u64);
        Box::pin(async {})
    };
    let on_event = move |ev: RetryEvent| e.lock().push(ev);
    let out = complete_with_retry(chain, req(), policy, &sleep, &on_event).await;
    let evs = events.lock().clone();
    let ms = slept.lock().clone();
    (evs, ms, out.map(|_| ()))
}

fn err(model: &str, attempt: u32, status: u16) -> RetryEvent {
    RetryEvent::Error {
        model: model.into(),
        attempt,
        // `ModelError::Http` 的 Display 只放一段摘录 + 原长（全文不进世界日志）。
        reason: format!("http {status}: boom (4 bytes)"),
    }
}

// ---------------------------------------------------------------- 用例

#[tokio::test]
async fn two_5xx_then_success_backs_off_exponentially() {
    let m = Flaky::new("a/1", vec![http(500), http(500)]);
    let chain: Vec<Arc<dyn Model>> = vec![m.clone()];
    let (evs, slept, out) = run(&chain, &RetryPolicy::default()).await;
    assert!(out.is_ok(), "第三次该成功：{out:?}");
    assert_eq!(
        evs,
        vec![
            err("a/1", 1, 500),
            RetryEvent::Retry {
                attempt: 1,
                delay_ms: 250
            },
            err("a/1", 2, 500),
            RetryEvent::Retry {
                attempt: 2,
                delay_ms: 500
            },
        ]
    );
    assert_eq!(slept, vec![250, 500]);
    assert_eq!(m.calls(), 3);
}

#[tokio::test]
async fn a_runtime_out_of_attempts_falls_back_to_the_next_one() {
    let a = Flaky::new("a/1", vec![http(500), http(503), http(500)]);
    let b = Flaky::ok("b/2");
    let chain: Vec<Arc<dyn Model>> = vec![a.clone(), b.clone()];
    let (evs, slept, out) = run(&chain, &RetryPolicy::default()).await;
    assert!(out.is_ok(), "{out:?}");
    assert_eq!(
        evs,
        vec![
            err("a/1", 1, 500),
            RetryEvent::Retry {
                attempt: 1,
                delay_ms: 250
            },
            err("a/1", 2, 503),
            RetryEvent::Retry {
                attempt: 2,
                delay_ms: 500
            },
            err("a/1", 3, 500),
            // 第三次失败之后不再 Retry（尝试用尽），直接换 runtime。
            RetryEvent::Fallback {
                from: "a/1".into(),
                to: "b/2".into()
            },
        ]
    );
    assert_eq!(slept, vec![250, 500]);
    assert_eq!((a.calls(), b.calls()), (3, 1));
}

#[tokio::test]
async fn a_missing_key_does_not_retry_it_falls_back_at_once() {
    let a = Flaky::new("a/1", vec![Some(ModelError::MissingKey("A_KEY".into()))]);
    let b = Flaky::ok("b/2");
    let chain: Vec<Arc<dyn Model>> = vec![a.clone(), b.clone()];
    let (evs, slept, out) = run(&chain, &RetryPolicy::default()).await;
    assert!(out.is_ok(), "{out:?}");
    assert_eq!(
        evs,
        vec![
            RetryEvent::Error {
                model: "a/1".into(),
                attempt: 1,
                reason: "missing api key A_KEY".into()
            },
            RetryEvent::Fallback {
                from: "a/1".into(),
                to: "b/2".into()
            },
        ]
    );
    assert!(slept.is_empty(), "没有 key 的时候等多久都还是没有 key");
    assert_eq!((a.calls(), b.calls()), (1, 1));
}

#[tokio::test]
async fn a_4xx_does_not_retry_and_the_last_error_is_what_comes_back() {
    let a = Flaky::new("a/1", vec![http(400)]);
    let b = Flaky::new("b/2", vec![http(404)]);
    let chain: Vec<Arc<dyn Model>> = vec![a.clone(), b.clone()];
    let (evs, slept, out) = run(&chain, &RetryPolicy::default()).await;
    match out {
        Err(ModelError::Http { status, .. }) => assert_eq!(status, 404),
        other => panic!("期望最后一个 runtime 的错误，拿到 {other:?}"),
    }
    assert_eq!(
        evs,
        vec![
            err("a/1", 1, 400),
            RetryEvent::Fallback {
                from: "a/1".into(),
                to: "b/2".into()
            },
            err("b/2", 1, 404),
        ]
    );
    assert!(slept.is_empty());
    assert_eq!((a.calls(), b.calls()), (1, 1));
}

#[tokio::test]
async fn the_backoff_is_capped() {
    let a = Flaky::new("a/1", vec![http(500), http(500), http(500), http(500)]);
    let chain: Vec<Arc<dyn Model>> = vec![a.clone()];
    let policy = RetryPolicy {
        max_attempts: 5,
        base_ms: 250,
        cap_ms: 600,
    };
    let (_, slept, out) = run(&chain, &policy).await;
    assert!(out.is_ok(), "{out:?}");
    assert_eq!(slept, vec![250, 500, 600, 600]);
}

#[tokio::test]
async fn transport_errors_and_timeouts_retry_but_a_protocol_error_does_not() {
    let a = Flaky::new(
        "a/1",
        vec![
            Some(ModelError::Transport("reset".into())),
            Some(ModelError::Timeout),
        ],
    );
    let chain: Vec<Arc<dyn Model>> = vec![a.clone()];
    let (_, slept, out) = run(&chain, &RetryPolicy::default()).await;
    assert!(out.is_ok(), "{out:?}");
    assert_eq!(slept, vec![250, 500]);

    let b = Flaky::new(
        "b/2",
        vec![Some(ModelError::Protocol("half an event".into()))],
    );
    let chain: Vec<Arc<dyn Model>> = vec![b.clone()];
    let (_, slept, out) = run(&chain, &RetryPolicy::default()).await;
    assert!(out.is_err(), "解不开的回答重试多少次都还是解不开");
    assert!(slept.is_empty());
    assert_eq!(b.calls(), 1);
}

// ---------------------------------------------------------------- 接进 turn 循环

/// 一个只有脑的最小内核：真 `Brains`、真 `TurnHandle`、真 `WorldLog`，没有准入链
/// （这些用例的模型要么失败要么只说话，一个提案都不发）。
struct Harness {
    log: Arc<WorldLog>,
    brains: Brains,
    _dir: tempfile::TempDir,
}

impl Harness {
    fn new() -> Harness {
        let clock = Arc::new(ManualClock::new(0));
        let log = Arc::new(WorldLog::new(clock.clone()));
        let state = Arc::new(StateRegistry::new(clock.clone()));
        let dir = tempfile::tempdir().unwrap();
        let brains = Brains::new(
            clock,
            log.clone(),
            state,
            dir.path().to_path_buf(),
            BrainsConfig::default(),
        )
        .unwrap();
        Harness {
            log,
            brains,
            _dir: dir,
        }
    }

    /// 挂一个带 fallback 链的脑，叫醒它，等 turn 收尾，把**这个 turn 的**事件按序交出来。
    ///
    /// 退避用的是真 `tokio::time::sleep`（循环里就该是真的），所以策略配成 1 ms / 2 ms——
    /// 判别信号是**事件出现**，墙钟只是那两次退避本身。
    async fn run_turn(
        &self,
        primary: Arc<dyn Model>,
        fallbacks: Vec<Arc<dyn Model>>,
        tools: ToolRegistry,
        max_steps: u32,
    ) -> Vec<(String, Value)> {
        let sink = NativeBrain::with_chain(
            NativeConfig {
                brain_id: "fast".into(),
                max_steps,
                ..NativeConfig::default()
            },
            primary,
            fallbacks,
            RetryPolicy {
                max_attempts: 3,
                base_ms: 1,
                cap_ms: 2,
            },
            tools,
            self.log.clone(),
        );
        let policy = WakePolicy::from_yaml(
            "rules: { obs/person: wake, brain/*: ignore }\nbudget: { max_consecutive_wakes: 5, min_interval_ms: 0 }",
        )
        .unwrap();
        let d = self.brains.attach("fast", policy, sink).unwrap();
        self.log.append(EventDraft::new(
            EventKind::new("obs/person").unwrap(),
            "test",
            json!({"n": 1}),
        ));
        let end = wait_for(&self.log, |e| e.kind.as_str() == "brain/turn_end").await;
        let trace = end.trace_id.clone().expect("turn_end 带 trace_id");
        let mine = self
            .log
            .since(0)
            .unwrap()
            .into_iter()
            .filter(|e| e.trace_id.as_deref() == Some(trace.as_str()))
            .map(|e| (e.kind.as_str().to_string(), e.payload))
            .collect();
        d();
        mine
    }

    /// 世界日志此刻的全文。"某个秘密有没有漏进去"只能这么问。
    fn dump(&self) -> String {
        serde_json::to_string(&self.log.since(0).unwrap()).unwrap()
    }
}

fn kinds_of(evs: &[(String, Value)]) -> Vec<&str> {
    evs.iter().map(|(k, _)| k.as_str()).collect()
}

fn payloads_of<'a>(evs: &'a [(String, Value)], kind: &str) -> Vec<&'a Value> {
    evs.iter()
        .filter(|(k, _)| k == kind)
        .map(|(_, p)| p)
        .collect()
}

/// 一个 step 的模型调用先 500 三次再换 runtime，世界日志里必须留下三种痕迹，且都带这个
/// turn 的 `trace_id`。
#[tokio::test]
async fn the_turn_logs_request_error_retry_and_fallback_with_the_trace_id() {
    let h = Harness::new();
    let a = Flaky::new("openai/gpt-5", vec![http(500), http(500), http(500)]);
    let b = Flaky::ok("anthropic/claude");
    let mine = h
        .run_turn(a.clone(), vec![b.clone()], ToolRegistry::new(), 1)
        .await;
    let kinds = kinds_of(&mine);
    // 前三条是内核的记账，最后一条是 `done` 的结果；中间那一段是这次瀑布。
    assert_eq!(
        kinds,
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/request-error",
            "brain/retry",
            "brain/request-error",
            "brain/retry",
            "brain/request-error",
            "brain/fallback",
            "brain/model_output",
            "brain/turn_end",
        ]
    );
    let by = |kind: &str| -> Vec<Value> { payloads_of(&mine, kind).into_iter().cloned().collect() };
    // `brain/request` 报的是**主** runtime 的身份（换没换 runtime 看 `brain/fallback`）。
    let request = &by("brain/request")[0];
    assert_eq!(request["brain_id"], "fast");
    assert_eq!(request["provider"], "openai");
    assert_eq!(request["model"], "gpt-5");
    assert_eq!(request["api_mode"], "unknown");
    let errors = by("brain/request-error");
    assert_eq!(
        errors[0],
        json!({"brain_id":"fast","provider":"openai","model":"gpt-5","attempt":1,
               "reason":"http 500: boom (4 bytes)","step":0})
    );
    assert_eq!(errors[2]["attempt"], 3);
    assert_eq!(
        by("brain/retry"),
        vec![
            json!({"brain_id":"fast","attempt":1,"delay_ms":1}),
            json!({"brain_id":"fast","attempt":2,"delay_ms":2}),
        ]
    );
    assert_eq!(
        by("brain/fallback"),
        vec![json!({"brain_id":"fast","from":"openai/gpt-5","to":"anthropic/claude"})]
    );
    assert_eq!(by("brain/turn_end")[0]["reason"], "end_turn");
    assert_eq!((a.calls(), b.calls()), (3, 1));
}

/// 一个只有文本的 chat_completions 回放：`ok` + `finish_reason: stop`。
const OK_SSE: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// 没有 API key 的 provider：**attach 照样成功**（计划的全局约束：key 是运行期的东西，
/// 不是配置错），turn 里第一次要发请求时才 `MissingKey`，不重试、直接换下一个 runtime。
///
/// 顺带钉住一条更要紧的事：第二顺位**有** key，而那把 key 只该出现在 Authorization 头
/// 上——世界日志里一个字都不该有。
#[tokio::test]
async fn a_provider_without_its_key_still_attaches_and_falls_back_without_leaking_the_next_one() {
    const SECRET: &str = "sk-test-not-in-the-log-9f3a";
    // 这个名字进程里不会有人设。声明解析、`Runtime` 构造、`build_model` 全都成功。
    let declared = Providers::from_yaml(
        "providers:\n  needs_key: { api_mode: chat_completions, base_url: http://127.0.0.1:1, env: OEH_TEST_UNSET_KEY }\n",
    )
    .unwrap();
    let a = build_model(&declared, "needs_key", "m", 1_000).expect("缺 key 不该挡住 boot");

    // 第二顺位：一个真的会回话的 mock 服务器，key 是上面那串。
    let srv = MockHttp::start(vec![Reply::sse(OK_SSE)]).await;
    let b: Arc<dyn Model> = Arc::new(
        ChatCompletionsModel::new(Runtime {
            provider: "mock".into(),
            model: "m2".into(),
            api_mode: ApiMode::ChatCompletions,
            base_url: srv.base_url(),
            env: "OEH_TEST_UNSET_KEY".into(),
            api_key: Some(SECRET.into()),
            cache: None,
            headers: BTreeMap::new(),
            timeout_ms: 5_000,
            program: None,
            reasoning_effort: None,
        })
        .unwrap(),
    );

    let h = Harness::new();
    let mine = h.run_turn(a, vec![b], ToolRegistry::new(), 1).await;
    assert_eq!(
        kinds_of(&mine),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            "brain/request",
            "brain/request-error",
            // 没有 `brain/retry`：重试多少次都还是没有 key。
            "brain/fallback",
            "brain/model_output",
            "brain/turn_end",
        ]
    );
    let err = payloads_of(&mine, "brain/request-error")[0];
    assert_eq!(err["reason"], "missing api key OEH_TEST_UNSET_KEY");
    assert_eq!(err["provider"], "needs_key");
    assert_eq!(err["attempt"], 1);
    assert_eq!(err["step"], 0);
    assert_eq!(
        payloads_of(&mine, "brain/turn_end")[0]["reason"],
        "end_turn"
    );

    // key 真的被用上了……
    assert_eq!(
        srv.last().headers.get("authorization").map(String::as_str),
        Some(format!("Bearer {SECRET}").as_str())
    );
    // ……而世界日志里一个字都没有。
    assert!(!h.dump().contains(SECRET), "API key 漏进了世界日志");
}

/// 链不记忆：第 0 步降级到了第二顺位，第 1 步还是从主 runtime 重新开始。
///
/// 降级是这一次调用的事，不是这个脑的新身份——主 provider 恢复了就该立刻用回去
/// （规格 §7.2）。
#[tokio::test]
async fn the_chain_starts_over_from_the_primary_on_the_next_step() {
    let h = Harness::new();
    // 第 0 步：主 runtime 三次 500 → 降级；第二顺位要求调一次工具（于是有第 1 步）。
    // 第 1 步：剧本已经空了，主 runtime 这次成功，说完话结束 turn。
    let a = Flaky::new("openai/gpt-5", vec![http(500), http(500), http(500)]);
    let b = Flaky::calling("anthropic/claude", "probe");
    let (tools, _calls) = recording_registry(&["probe:safe"]);
    let mine = h.run_turn(a.clone(), vec![b.clone()], tools, 4).await;

    assert_eq!(
        kinds_of(&mine),
        vec![
            "brain/wake",
            "brain/turn_start",
            "brain/cursor",
            // ---- step 0：主 runtime 用尽 → 降级 → 第二顺位要求一次工具
            "brain/request",
            "brain/request-error",
            "brain/retry",
            "brain/request-error",
            "brain/retry",
            "brain/request-error",
            "brain/fallback",
            "brain/model_output",
            "brain/tool_call",
            "brain/tool_result",
            // ---- step 1：回到主 runtime，一次就成
            "brain/request",
            "brain/model_output",
            "brain/turn_end",
        ]
    );
    // 第 0 步的三条错都带 step 0；第 1 步一条错都没有——也就是说它没有接着用第二顺位。
    for e in payloads_of(&mine, "brain/request-error") {
        assert_eq!(e["step"], 0, "{e}");
    }
    assert_eq!(payloads_of(&mine, "brain/fallback").len(), 1);
    assert_eq!(payloads_of(&mine, "brain/model_output")[1]["step"], 1);
    // 主 runtime 被叫了 4 次（第 0 步三次 + 第 1 步一次），第二顺位只有第 0 步那一次。
    assert_eq!((a.calls(), b.calls()), (4, 1));
    assert_eq!(
        payloads_of(&mine, "brain/turn_end")[0]["reason"],
        "end_turn"
    );
}

/// 等世界日志里出现第一条满足 `pred` 的事件（注册观察者**之前**已经进去的也算）。
async fn wait_for(
    log: &Arc<WorldLog>,
    pred: impl Fn(&WorldEvent) -> bool + Send + Sync + 'static,
) -> WorldEvent {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let pred = Arc::new(pred);
    let p = pred.clone();
    let dispose = log.observe(Arc::new(move |e: &WorldEvent| {
        if p(e) {
            let _ = tx.send(e.clone());
        }
    }));
    let found = match log
        .since(0)
        .unwrap_or_default()
        .into_iter()
        .find(|e| pred(e))
    {
        Some(e) => e,
        None => tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("等世界事件超时")
            .expect("世界日志观察者通道关了"),
    };
    dispose();
    found
}
