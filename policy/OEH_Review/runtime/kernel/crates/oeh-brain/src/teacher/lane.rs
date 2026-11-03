//! 复审老师的脑内通道（`review.teacher.channel: brain`）：`TeacherBrain`（teacher.py:371-549）的
//! 模型循环搬进内核进程里的慢脑模块，Python 只留下它独有的那几样——运行时状态与决策校验。
//!
//! ```text
//!   obs/teacher_packet ──wake──▶ TeacherLane::on_turn
//!                                  │ begin      ─▶ Python：当前包（state.packet）的文本与图片路径
//!                                  │ 循环（teacher.py:425 `_decide`）：
//!                                  │   state    ─▶ Python：state.closed(rid) / runtime.closed
//!                                  │   Codex 线程上跑一个 turn（[`CodexClient::create`]，阻塞线程里）
//!                                  │   validate ─▶ Python：validate_teacher_response + intent_from_decision
//!                                  │   window   ─▶ Python：state.wait_window，然后记 teacher_decision
//!                                  │   turn.propose(oeh_manipulate, intent) ─▶ 准入 ─▶ MotionPlugin._act_review
//!                                  │   unwait   ─▶ Python：state.waiting.discard
//!                                  │ emit / fatal / failed ─▶ Python 照原样记进 events.jsonl
//!                                  ▼ turn.done("decided")
//! ```
//!
//! **桥**是两种世界日志事件：本通道写 `brain/teacher_bridge`（Python 的 watcher 用 `log/tail`
//! 收，`brain` 前缀本来就在它的订阅里），Python 用 `event/emit` 写 `obs/teacher_bridge`（本通道
//! 的世界日志观察者收）。请求带 `id`，回复是 `{"op": "reply", "id"}`。本通道的事件都从同一个
//! turn 任务里**依次** append，所以 Python 按发出的顺序处理它们——`emit` 进 events.jsonl 的顺序
//! 就是 Python `TeacherBrain` 记它们的顺序。
//!
//! **为什么问而不是镜像。** Python 的 `TeacherBrain` 在几个精确的时刻读运行时状态（调用前、调用
//! 后、提案前等窗口）。把这些状态推给本通道再在这边判断，会让每次判断都晚一拍、而且要追踪
//! `ReviewState` 的每一处改动；在同一时刻**问** Python，用的就是 Python 自己的那几行代码
//! （`state.closed`、`wait_window`、`validate_teacher_response`），一次往返是亚毫秒级的。
//!
//! 提案仍然是这个脑自己的：`TurnHandle::propose` 走准入，`brain_id` 就是 `teacher`，参数就是
//! Python 那条通道会提的那份 intent，所以内核里的 `act/*` 记录和 `MotionPlugin._act_review`
//! 收到的东西与 Python 通道一样。
//!
//! 调用记录（`teacher/call-NNNN.json`、`-output.json`）由这里直接写，字节与 Python 相同
//! （[`pyjson`](super::pyjson)）；`model_call` / `usage` / `teacher_call_error` 等事件由 Python 照原样
//! 记（`emit`），计数器随每条消息带过去，`teacher_summary` 由 Python 在关闭时从最后一份计数器拼。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use oeh_kernel::observation::Observation;
use oeh_kernel::proposal::Outcome;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{BrainSink, TurnHandle};
use oeh_proto::{EventDraft, EventKind, WorldEvent};
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use tokio::sync::{oneshot, watch};

use super::codex_app_server::{
    truthy, CloseHandle, CodexClient, CodexConversation, TeacherConfig, TeacherError,
};
use super::pyjson::{b64encode, digest, dumps_default, encoded, py_number, redact, truncate};

/// Match Python's time.monotonic_ns/time.time audit clocks (not the kernel-relative clock).
fn audit_time() -> Result<(u64, f64), Stop> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: ts is a valid writable timespec; CLOCK_MONOTONIC requires no other state.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } != 0 {
        return Err(Stop::failed(
            "OSError",
            std::io::Error::last_os_error().to_string(),
        ));
    }
    let utc = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Stop::failed("OSError", e.to_string()))?
        .as_secs_f64();
    Ok((ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64, utc))
}

/// 本通道写给 Python 的事件种类。
pub const TO_PYTHON: &str = "brain/teacher_bridge";
/// Python 写给本通道的事件种类。
pub const FROM_PYTHON: &str = "obs/teacher_bridge";
const SOURCE: &str = "brain-teacher";
/// teacher.py:54。
pub const WINDOW_CLOSED: &str = "window closed";
/// teacher.py:56：被保持动作抢走这一拍的提案，下一拍原样再提。
const RETRY_OUTCOMES: &[&str] = &["preempted", "cancelled"];
/// teacher.py:436 / 454。
const NO_CALL: &str = "answer with exactly one robodojo_execute call";
const CORRECTION: &str = "Submit a corrected robodojo_execute call for the same packet.";
/// oeh_sdk.protocol.code.TIMEOUT：Python 通道里 `ctx.propose(timeout=…)` 超时抛的 RpcError 码。
const RPC_TIMEOUT: i64 = -32003;

/// 本通道的配置（`brain-teacher` 那一行）。
#[derive(Debug, Clone)]
pub struct LaneConfig {
    /// 本次尝试的输出目录（runtime.output）。调用记录写在 `<output>/teacher/`。
    pub output: PathBuf,
    /// 冻结的 `review.teacher` 原样（模型、`max_invalid`、价格）。
    pub teacher: Value,
    pub capability: String,
    /// `review.turn_seconds`：每次提案的等待上限（Python `ctx.propose(timeout=…)`）。
    pub turn_seconds: Value,
    /// `model_call.method`（`teacher_codex`）。
    pub method: String,
}

impl LaneConfig {
    fn model(&self) -> &str {
        self.teacher
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    fn max_invalid(&self) -> u64 {
        self.teacher
            .get("max_invalid")
            .and_then(Value::as_u64)
            .unwrap_or(6)
    }
}

/// `TeacherBrain` 的计数器（teacher.py:382-384），随消息带给 Python 拼 `teacher_summary`。
#[derive(Debug, Default, Clone)]
struct Counters {
    calls: u64,
    failed_calls: u64,
    invalid: u64,
    input: i64,
    cached_input: i64,
    output: i64,
    reasoning: i64,
    cost: f64,
}

impl Counters {
    fn to_value(&self) -> Value {
        json!({
            "calls": self.calls, "failed_calls": self.failed_calls, "invalid_or_refused": self.invalid,
            "tokens": {"input": self.input, "cached_input": self.cached_input, "output": self.output,
                       "reasoning": self.reasoning},
            "cost": self.cost,
        })
    }
}

/// 循环为什么停下：Python `on_turn` 的三条 except（teacher.py:409-418）。
#[derive(Debug)]
enum Stop {
    /// `TeacherFatal`：provider 拒绝了登录、模型或额度。
    Fatal { message: String, errors: Vec<Value> },
    /// 别的异常：`f"{type(exc).__name__}: {exc}"`，`give_up(rid, type)`。
    Failed { type_name: String, message: String },
    /// Python 那一半不在了（运行时关了、内核在撤这一行）：什么都不记。
    Gone,
}

impl Stop {
    fn failed(type_name: &str, message: impl Into<String>) -> Self {
        Stop::Failed {
            type_name: type_name.into(),
            message: message.into(),
        }
    }

    fn from_teacher(error: TeacherError) -> Self {
        match error {
            TeacherError::Fatal { message, errors } => Stop::Fatal { message, errors },
            TeacherError::Unavailable { message, .. } => {
                Stop::failed("TeacherUnavailable", message)
            }
            // 阶段 1 把 Python 的 KeyError 写成 "KeyError: 'id'"。
            TeacherError::Unexpected(text) => match text.split_once(": ") {
                Some((name, rest)) => Stop::failed(name, rest),
                None => Stop::failed("RuntimeError", text),
            },
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 桥
// ---------------------------------------------------------------------------------------------

/// 和 Python 那一半说话：发事件、等回复。
struct Bridge {
    log: Arc<WorldLog>,
    next: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Value>>>,
    gone: AtomicBool,
}

impl Bridge {
    fn send(&self, payload: Value, trace: Option<&str>) {
        let mut draft = EventDraft::new(
            EventKind::new(TO_PYTHON).expect("static kind"),
            SOURCE,
            payload,
        );
        if let Some(trace) = trace {
            draft = draft.with_trace_id(trace);
        }
        self.log.append(draft);
    }

    async fn request(&self, op: &str, fields: Value, trace: Option<&str>) -> Result<Value, Stop> {
        if self.gone.load(Ordering::SeqCst) {
            return Err(Stop::Gone);
        }
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id, tx);
        let mut payload = match fields {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        payload.insert("op".into(), json!(op));
        payload.insert("id".into(), json!(id));
        self.send(Value::Object(payload), trace);
        rx.await.map_err(|_| Stop::Gone)
    }

    fn resolve(&self, payload: &Value) {
        let Some(id) = payload.get("id").and_then(Value::as_u64) else {
            return;
        };
        let tx = self.pending.lock().remove(&id);
        if let Some(tx) = tx {
            let _ = tx.send(payload.clone());
        }
    }

    /// 撤这一行：在等的请求全部以 [`Stop::Gone`] 结束。
    fn shut(&self) {
        self.gone.store(true, Ordering::SeqCst);
        self.pending.lock().clear();
    }
}

// ---------------------------------------------------------------------------------------------
// 通道
// ---------------------------------------------------------------------------------------------

/// 关闭时连临时目录一起收拾（Python `CodexClient.close`）；`Drop` 只杀子进程。
struct OwnedClient(CodexClient);

impl Drop for OwnedClient {
    fn drop(&mut self) {
        self.0.close();
    }
}

struct LaneState {
    conversation: CodexConversation,
    counters: Counters,
}

struct LaneInner {
    cfg: LaneConfig,
    bridge: Bridge,
    client: Arc<Mutex<Option<OwnedClient>>>,
    close: CloseHandle,
    preflight: watch::Receiver<Option<Value>>,
    state: tokio::sync::Mutex<LaneState>,
    rt: tokio::runtime::Handle,
}

/// `brain_id: teacher` 的那个 [`BrainSink`]。克隆共享同一份状态。
#[derive(Clone)]
pub struct TeacherLane {
    inner: Arc<LaneInner>,
}

impl TeacherLane {
    /// 造通道并立刻在阻塞线程里跑预检（Python 通道在 `Runtime.start` 开头跑它；这里是内核起来、
    /// 这一行被应用的那一刻）。结果留着，等 Python 来要（`{"op": "preflight"}`）。
    pub fn new(
        cfg: LaneConfig,
        teacher: &TeacherConfig,
        episode_id: &str,
        output_schema: Value,
        client: CodexClient,
        log: Arc<WorldLog>,
        rt: tokio::runtime::Handle,
    ) -> Self {
        let close = client.close_handle();
        let client = Arc::new(Mutex::new(Some(OwnedClient(client))));
        let (tx, rx) = watch::channel(None);
        {
            let client = client.clone();
            rt.spawn_blocking(move || {
                let checked = match client.lock().as_mut() {
                    Some(c) => c.0.preflight(),
                    None => Err(TeacherError::Unavailable {
                        message: "teacher client closed".into(),
                        errors: Vec::new(),
                    }),
                };
                let message = match checked {
                    Ok(result) => json!({"op": "preflight", "ok": true, "result": result}),
                    Err(error) => json!({"op": "preflight", "ok": false, "error": raised(&error)}),
                };
                let _ = tx.send(Some(message));
            });
        }
        TeacherLane {
            inner: Arc::new(LaneInner {
                state: tokio::sync::Mutex::new(LaneState {
                    conversation: CodexConversation::new(teacher, episode_id, output_schema),
                    counters: Counters::default(),
                }),
                cfg,
                bridge: Bridge {
                    log,
                    next: AtomicU64::new(1),
                    pending: Mutex::default(),
                    gone: AtomicBool::new(false),
                },
                client,
                close,
                preflight: rx,
                rt,
            }),
        }
    }

    /// 世界日志观察者：只看 Python 写给本通道的那一种事件。观察者跑在 append 的调用方上，
    /// 所以这里只送 oneshot 或者 spawn。
    pub fn on_world_event(&self, ev: &WorldEvent) {
        if ev.kind.as_str() != FROM_PYTHON {
            return;
        }
        let payload = &ev.payload;
        match payload.get("op").and_then(Value::as_str) {
            Some("reply") => self.inner.bridge.resolve(payload),
            Some("preflight") => {
                let inner = self.inner.clone();
                self.inner.rt.spawn(async move {
                    let mut rx = inner.preflight.clone();
                    let Ok(message) = rx.wait_for(Option::is_some).await.map(|m| m.clone()) else {
                        return;
                    };
                    if let Some(message) = message {
                        inner.bridge.send(message, None);
                    }
                });
            }
            Some("close") => self.close_client(),
            _ => {}
        }
    }

    /// `TeacherBrain.close()`：在飞的调用不再重试；app-server 停掉，临时目录删掉。
    fn close_client(&self) {
        self.inner.close.close();
        let client = self.inner.client.clone();
        self.inner.rt.spawn_blocking(move || {
            // 在飞的 create 看到关闭标志会很快返回，这把锁随即可得。
            drop(client.lock().take());
        });
    }

    /// 内核撤这一行：在等 Python 的请求全部放弃，app-server 停掉。
    pub fn shutdown(&self) {
        self.inner.bridge.shut();
        self.close_client();
    }
}

/// 预检失败交给 Python 的形状：`class` 决定 Python 那边重建哪种异常。
fn raised(error: &TeacherError) -> Value {
    match error {
        TeacherError::Fatal { message, errors } => {
            json!({"class": "TeacherFatal", "message": message, "errors": errors})
        }
        TeacherError::Unavailable { message, errors } => {
            json!({"class": "TeacherUnavailable", "message": message, "errors": errors})
        }
        TeacherError::Unexpected(message) => json!({"class": "Unexpected", "message": message}),
    }
}

impl BrainSink for TeacherLane {
    fn on_turn(&self, _obs: Observation, turn: TurnHandle) {
        let inner = self.inner.clone();
        self.inner.rt.spawn(async move {
            // 一次一个 turn：对话是这个 episode 的，不能两个 turn 交错着往里写。
            let mut st = inner.state.lock().await;
            inner.turn(&mut st, &turn).await;
            drop(st);
            turn.done("decided");
        });
    }

    /// Python `TeacherBrain` 不处理 steer。
    fn steer(&self, _obs: Observation) {}

    /// Python `TeacherBrain` 不处理 interrupt（`on_interrupt` 的默认是忽略）：在跑的循环照常
    /// 跑完，它之后的提案由内核按"turn 不是当前的"拒掉。
    fn interrupt(&self) {}
}

impl LaneInner {
    fn emit(&self, kind: &str, payload: Value, counters: &Counters, trace: Option<&str>) {
        self.bridge.send(
            json!({"op": "emit", "kind": kind, "payload": payload, "counters": counters.to_value()}),
            trace,
        );
    }

    fn counters(&self, counters: &Counters, trace: Option<&str>) {
        self.bridge.send(
            json!({"op": "counters", "counters": counters.to_value()}),
            trace,
        );
    }

    /// `state.closed(rid)` 与 `runtime.closed`，在 Python 里同一时刻读。
    async fn state(&self, rid: &str, trace: Option<&str>) -> Result<(bool, bool), Stop> {
        let reply = self
            .bridge
            .request("state", json!({"request_id": rid}), trace)
            .await?;
        Ok((
            reply.get("closed").and_then(Value::as_bool).unwrap_or(true),
            reply
                .get("runtime_closed")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        ))
    }

    /// teacher.py:401 `on_turn`。
    async fn turn(&self, st: &mut LaneState, turn: &TurnHandle) {
        let trace = Some(turn.trace_id.as_str());
        let Ok(begin) = self.bridge.request("begin", json!({}), trace).await else {
            return;
        };
        // 当前包为空、是 OpenWAM 自己跑的 `auto` 包、或者已经关了：不是老师的。
        let Some(rid) = begin
            .get("request_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            return;
        };
        let outcome = self.decide(st, turn, &rid, &begin).await;
        let counters = st.counters.to_value();
        match outcome {
            Ok(()) | Err(Stop::Gone) => {}
            Err(Stop::Fatal { message, errors }) => self.bridge.send(
                json!({"op": "fatal", "request_id": rid, "error": truncate(&message, 800),
                       "fatal_error": errors.last().cloned().unwrap_or(Value::Null),
                       "counters": counters}),
                trace,
            ),
            Err(Stop::Failed { type_name, message }) => self.bridge.send(
                json!({"op": "failed", "request_id": rid,
                       "error": truncate(&format!("{type_name}: {message}"), 800),
                       "reason": type_name, "counters": counters}),
                trace,
            ),
        }
    }

    /// teacher.py:425 `_decide`。
    async fn decide(
        &self,
        st: &mut LaneState,
        turn: &TurnHandle,
        rid: &str,
        begin: &Value,
    ) -> Result<(), Stop> {
        let trace = Some(turn.trace_id.as_str());
        let text = begin.get("text").and_then(Value::as_str).unwrap_or("");
        let mut images = Vec::new();
        for path in begin
            .get("images")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let path = path.as_str().unwrap_or("");
            // 和运行时交给 Python 老师的 `state.images` 是同一份字节：image_items 把它们原样写在包旁边。
            let data = std::fs::read(path).map_err(|e| Stop::failed("OSError", e.to_string()))?;
            images.push(json!({"type": "input_image",
                               "image_url": format!("data:image/png;base64,{}", b64encode(&data))}));
        }
        let first = begin.get("decision").and_then(Value::as_i64) == Some(0);
        st.conversation.add_packet(text, &images, first);
        let mut refusals: u64 = 0;
        loop {
            if self.state(rid, trace).await?.0 {
                return Ok(());
            }
            let call = self.call(st, rid, trace).await?;
            if self.state(rid, trace).await?.0 {
                return Ok(()); // 答得太晚：运行时已经往前走了，下一个包回答这一次调用
            }
            let mut problem: Vec<(&str, Value)> = match call {
                None => vec![("error", json!(NO_CALL)), ("no_execution", json!(true))],
                Some(call) => {
                    let arguments = call.get("arguments").cloned().unwrap_or(Value::Null);
                    let verdict = self
                        .bridge
                        .request(
                            "validate",
                            json!({"request_id": rid, "arguments": arguments}),
                            trace,
                        )
                        .await?;
                    if let Some(raised) = verdict.get("exception").filter(|v| truthy(v)) {
                        return Err(Stop::failed(
                            raised
                                .get("type")
                                .and_then(Value::as_str)
                                .unwrap_or("Exception"),
                            raised.get("message").and_then(Value::as_str).unwrap_or(""),
                        ));
                    }
                    if verdict.get("valid").and_then(Value::as_bool) == Some(true) {
                        let intent = verdict.get("intent").cloned().unwrap_or(Value::Null);
                        match self.propose(turn, rid, intent).await? {
                            None => return Ok(()),
                            Some(reason) => vec![
                                ("error", json!(reason)),
                                ("error_type", json!("oeh_admission_refused")),
                                ("no_execution", json!(true)),
                            ],
                        }
                    } else {
                        vec![
                            (
                                "error",
                                verdict.get("error").cloned().unwrap_or(Value::Null),
                            ),
                            // Python 那一半给的 error_type（review guard 的拒绝是 `review_guard`，
                            // teacher.py `_decide`），没给就是校验失败的 recoverable_tool_input。
                            (
                                "error_type",
                                verdict
                                    .get("error_type")
                                    .filter(|v| v.is_string())
                                    .cloned()
                                    .unwrap_or_else(|| json!("recoverable_tool_input")),
                            ),
                            ("no_execution", json!(true)),
                        ]
                    }
                }
            };
            refusals += 1;
            st.counters.invalid += 1;
            self.counters(&st.counters, trace);
            if refusals >= self.cfg.max_invalid() {
                return Err(Stop::failed(
                    "RuntimeError",
                    format!("{refusals} invalid or refused decisions for one packet"),
                ));
            }
            problem.push(("request_id", json!(rid)));
            problem.push(("correction", json!(CORRECTION)));
            let text = dumps_default(&problem).map_err(|e| Stop::failed("ValueError", e))?;
            st.conversation.add_error(&text);
        }
    }

    /// teacher.py:457 `_call`：一次模型调用，连同它的调用记录与事件。
    async fn call(
        &self,
        st: &mut LaneState,
        rid: &str,
        trace: Option<&str>,
    ) -> Result<Option<Value>, Stop> {
        let model = self.cfg.model().to_owned();
        st.counters.calls += 1;
        let calls = st.counters.calls;
        let request_id = format!("teacher:{rid}:{calls}");
        let body = st.conversation.body();
        let directory = self.cfg.output.join("teacher");
        let name = format!("call-{calls:04}.json");
        let bad = |e: String| Stop::failed("ValueError", e);
        let new_items_from = st.conversation.recorded;
        let new_items = redact(&Value::Array(st.conversation.unrecorded())).map_err(bad)?;
        let (started_monotonic_ns, started_utc) = audit_time()?;
        let artifact = json!({
            "request_id": request_id, "packet_request_id": rid,
            "started_monotonic_ns": started_monotonic_ns, "started_utc": started_utc,
            "full_input_sha256": digest(&body).map_err(bad)?,
            "input_items": body["input"].as_array().map_or(0, Vec::len),
            "new_items_from": new_items_from, "new_items": new_items, "model": body["model"],
            "reasoning": body["reasoning"], "prompt_cache_key": body["prompt_cache_key"],
            "tool_sha256": digest(&body["tools"]).map_err(bad)?,
        });
        std::fs::write(directory.join(&name), encoded(&artifact).map_err(bad)?)
            .map_err(|e| Stop::failed("OSError", e.to_string()))?;
        self.emit(
            "model_call",
            json!({"model_id": model, "request_id": request_id,
                   "input_sha256": digest(&artifact).map_err(bad)?,
                   "input_artifact": format!("teacher/{name}"),
                   "full_input_sha256": artifact["full_input_sha256"], "method": self.cfg.method}),
            &st.counters,
            trace,
        );
        let started = Instant::now();
        let client = self.client.clone();
        let sent = body.clone();
        let result = self
            .rt
            .spawn_blocking(move || match client.lock().as_mut() {
                Some(c) => c.0.create(&sent),
                None => Err(TeacherError::Unavailable {
                    message: "teacher client closed".into(),
                    errors: Vec::new(),
                }),
            })
            .await
            .unwrap_or_else(|e| Err(TeacherError::Unexpected(format!("RuntimeError: {e}"))));
        let elapsed_ms = || started.elapsed().as_nanos() as f64 / 1e6;
        let (data, errors) = match result {
            Ok(v) => v,
            Err(TeacherError::Unexpected(text)) => {
                return Err(Stop::from_teacher(TeacherError::Unexpected(text)))
            }
            Err(error) => {
                st.counters.failed_calls += 1;
                let (message, errors) = match &error {
                    TeacherError::Fatal { message, errors }
                    | TeacherError::Unavailable { message, errors } => (message.clone(), errors),
                    TeacherError::Unexpected(_) => unreachable!("handled above"),
                };
                if !self.state(rid, trace).await?.1 {
                    for e in errors {
                        self.emit(
                            "teacher_call_error",
                            merged(&request_id, e),
                            &st.counters,
                            trace,
                        );
                    }
                    self.emit(
                        "model_failure",
                        json!({"model_id": model, "request_id": request_id,
                               "reason": truncate(&message, 800)}),
                        &st.counters,
                        trace,
                    );
                    self.emit(
                        "model_timing",
                        json!({"request_id": request_id, "duration_ms": elapsed_ms()}),
                        &st.counters,
                        trace,
                    );
                } else {
                    self.counters(&st.counters, trace);
                }
                return Err(Stop::from_teacher(error));
            }
        };
        if self.state(rid, trace).await?.1 {
            return Err(Stop::failed(
                "TeacherUnavailable",
                "episode closed during the call",
            ));
        }
        for e in &errors {
            self.emit(
                "teacher_call_error",
                merged(&request_id, e),
                &st.counters,
                trace,
            );
        }
        self.emit(
            "model_timing",
            json!({"request_id": request_id, "duration_ms": elapsed_ms()}),
            &st.counters,
            trace,
        );
        let usage = data
            .get("usage")
            .filter(|u| truthy(u))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let details = |key: &str, field: &str| {
            usage
                .get(key)
                .filter(|d| truthy(d))
                .and_then(|d| d.get(field))
                .cloned()
                .unwrap_or(Value::Null)
        };
        let cached = or_zero(&details("input_tokens_details", "cached_tokens"));
        let reasoning = details("output_tokens_details", "reasoning_tokens");
        let cost = self.cost(&usage, &cached);
        if let Some(cost) = cost {
            st.counters.cost += cost;
        }
        let field = |key: &str| usage.get(key).cloned().unwrap_or(Value::Null);
        st.counters.input += as_int(&or_zero(&field("input_tokens")));
        st.counters.cached_input += as_int(&cached);
        st.counters.output += as_int(&or_zero(&field("output_tokens")));
        st.counters.reasoning += as_int(&or_zero(&reasoning));
        self.emit(
            "usage",
            json!({"model_id": model, "request_id": request_id,
                   "input_tokens": field("input_tokens"), "output_tokens": field("output_tokens"),
                   "cached_input_tokens": cached, "reasoning_output_tokens": reasoning,
                   "call_complete": true, "total_tokens": field("total_tokens"), "cost_usd": cost,
                   "cost_source": cost.map(|_| "list-price estimate"),
                   "source": "provider_reported"}),
            &st.counters,
            trace,
        );
        let output: Vec<Value> = data
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let call = st.conversation.add_output(&output);
        st.conversation.recorded = st.conversation.items.len();
        let (completed_monotonic_ns, completed_utc) = audit_time()?;
        let mut record = json!({
            "request_id": request_id, "response_id": data.get("id").cloned().unwrap_or(Value::Null),
            "usage": usage, "output": redact(&Value::Array(output)).map_err(bad)?,
            "completed_monotonic_ns": completed_monotonic_ns, "completed_utc": completed_utc,
        });
        for key in ["stop", "codex", "claude_code"] {
            if let Some(v) = data.get(key).filter(|v| truthy(v)) {
                record[key] = v.clone();
            }
        }
        std::fs::write(
            directory.join(format!("call-{calls:04}-output.json")),
            encoded(&record).map_err(bad)?,
        )
        .map_err(|e| Stop::failed("OSError", e.to_string()))?;
        Ok(call)
    }

    /// teacher.py:501-509：价目表上的估价。运算顺序照 Python，浮点结果逐位相同。
    fn cost(&self, usage: &Value, cached: &Value) -> Option<f64> {
        let prices = self
            .cfg
            .teacher
            .get("prices_usd_per_million")
            .filter(|p| truthy(p))?;
        if !truthy(usage) {
            return None;
        }
        let price = |k: &str| prices.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let written = usage
            .get("input_tokens_details")
            .filter(|d| truthy(d))
            .and_then(|d| d.get("cache_creation_tokens"))
            .map(or_zero)
            .map_or(0.0, |v| as_f64(&v));
        let write_price = prices
            .get("cache_write")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| price("input"));
        let num = |k: &str| usage.get(k).map_or(0.0, as_f64);
        let cached = as_f64(cached);
        Some(
            ((num("input_tokens") - cached - written) * price("input")
                + written * write_price
                + cached * price("cached_input")
                + num("output_tokens") * price("output"))
                / 1e6,
        )
    }

    /// teacher.py:528 `_propose`：等窗口、提案；被保持动作抢走的那一拍原样再提。`None` = 已执行
    /// 或已无意义，否则是拒绝的理由（回给模型）。
    async fn propose(
        &self,
        turn: &TurnHandle,
        rid: &str,
        intent: Value,
    ) -> Result<Option<String>, Stop> {
        let trace = Some(turn.trace_id.as_str());
        let result = self.propose_loop(turn, rid, intent).await;
        // `finally: state.waiting.discard(rid)`。
        if !matches!(result, Err(Stop::Gone)) {
            self.bridge
                .send(json!({"op": "unwait", "request_id": rid}), trace);
        }
        result
    }

    async fn propose_loop(
        &self,
        turn: &TurnHandle,
        rid: &str,
        intent: Value,
    ) -> Result<Option<String>, Stop> {
        let trace = Some(turn.trace_id.as_str());
        let limit = self.cfg.turn_seconds.as_f64().unwrap_or(1800.0).max(0.0);
        loop {
            // Python 在这里 `await state.wait_window(rid)`、再查一次 closed、再记 teacher_decision。
            let window = self
                .bridge
                .request("window", json!({"request_id": rid}), trace)
                .await?;
            if window.get("moot").and_then(Value::as_bool) != Some(false) {
                return Ok(None);
            }
            let pending = turn.propose(&self.cfg.capability, intent.clone());
            let outcome = match tokio::time::timeout(Duration::from_secs_f64(limit), pending).await
            {
                Ok(outcome) => outcome,
                Err(_) => {
                    let t = match &self.cfg.turn_seconds {
                        Value::Number(n) => py_number(n).unwrap_or_else(|_| n.to_string()),
                        other => other.to_string(),
                    };
                    return Err(Stop::failed(
                        "RpcError",
                        format!("[{RPC_TIMEOUT}] `turn/propose` timed out after {t}s"),
                    ));
                }
            };
            let (tag, reason) = outcome_reason(&outcome);
            if tag == "executed" {
                return Ok(None);
            }
            if reason.contains(WINDOW_CLOSED) || RETRY_OUTCOMES.contains(&tag) {
                continue; // 保持动作占了这一拍：下一拍原样再提
            }
            return Ok(Some(reason));
        }
    }
}

/// Python `str(outcome.get("reason") or outcome.get("error") or outcome.get("outcome"))`，
/// 外加结局的标签。
fn outcome_reason(outcome: &Outcome) -> (&'static str, String) {
    let pick = |text: &str, tag: &'static str| {
        (
            tag,
            if text.is_empty() {
                tag.to_owned()
            } else {
                text.to_owned()
            },
        )
    };
    match outcome {
        Outcome::Executed { .. } => ("executed", "executed".into()),
        Outcome::Refused { reason } => pick(reason, "refused"),
        Outcome::Failed { error } => pick(error, "failed"),
        Outcome::Preempted => ("preempted", "preempted".into()),
        Outcome::Cancelled { reason } => pick(reason, "cancelled"),
    }
}

/// `{"request_id": request_id, **error}`。
fn merged(request_id: &str, error: &Value) -> Value {
    let mut out = Map::new();
    out.insert("request_id".into(), json!(request_id));
    if let Value::Object(map) = error {
        for (k, v) in map {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// `x or 0`。
fn or_zero(v: &Value) -> Value {
    if truthy(v) {
        v.clone()
    } else {
        json!(0)
    }
}

fn as_int(v: &Value) -> i64 {
    v.as_i64()
        .unwrap_or_else(|| v.as_f64().map_or(0, |f| f as i64))
}

fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_reasons_follow_python_or_chains() {
        assert_eq!(
            outcome_reason(&Outcome::Failed {
                error: "ValueError('window closed: x')".into()
            }),
            ("failed", "ValueError('window closed: x')".into())
        );
        assert_eq!(
            outcome_reason(&Outcome::Refused { reason: "".into() }),
            ("refused", "refused".into())
        );
        assert_eq!(
            outcome_reason(&Outcome::Preempted),
            ("preempted", "preempted".into())
        );
        assert_eq!(
            outcome_reason(&Outcome::Cancelled {
                reason: "rules".into()
            }),
            ("cancelled", "rules".into())
        );
    }

    #[test]
    fn errors_merge_after_the_request_id() {
        let v = merged("teacher:r:1", &json!({"status": null, "attempt": 1}));
        assert_eq!(
            v,
            json!({"request_id": "teacher:r:1", "status": null, "attempt": 1})
        );
    }

    #[test]
    fn unexpected_errors_keep_their_python_type() {
        match Stop::from_teacher(TeacherError::Unexpected("KeyError: 'id'".into())) {
            Stop::Failed { type_name, message } => {
                assert_eq!(type_name, "KeyError");
                assert_eq!(message, "'id'");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn counters_carry_the_summary_fields() {
        let c = Counters {
            calls: 2,
            invalid: 1,
            input: 10,
            cost: 0.5,
            ..Counters::default()
        };
        let v = c.to_value();
        assert_eq!(v["invalid_or_refused"], 1);
        assert_eq!(v["tokens"]["input"], 10);
        assert_eq!(v["cost"], 0.5);
    }
}
