//! 复审老师的 Codex app-server 客户端：`python/oeh-policy/oeh_policy/codex_teacher.py` 的移植。
//!
//! **目标是线上逐字段一致**，而不是"同一个意思"：同一串调用下，本模块发给 app-server 的每一条
//! JSON-RPC 请求（方法、参数、id 序号、`initialized` 通知、对服务端请求的 -32601 拒绝）、
//! 子进程的 argv（`-c` 覆盖项）与环境（擦掉的 API key、`CODEX_HOME`）都和 Python 的
//! `CodexClient` 一样。证明在 `python/oeh-policy/tests/test_codex_teacher_parity.py`：两个客户端
//! 对着同一个 `fake_codex_app_server.py` 跑同一串包，比较它记下的线上消息。
//!
//! 控制流也照抄 Python（同步、阻塞，和 Python 一样由调用方放进工作线程）：
//!
//! ```text
//! preflight: _launch(initialize + initialized，最多 LAUNCH_ATTEMPTS 次) → model/list → account/rateLimits/read
//!            额度用完 → 换下一个账号重来；没有账号了 → Fatal
//! create:    _ensure(必要时 _launch，再 thread/start 或 thread/resume) → turn/start → 收通知到 turn/completed
//!            超过 request_timeout_s → turn/interrupt，再等 30 s
//!            按 codexErrorInfo 分类：换账号 / 重试 / 放弃（Unavailable）/ 没账号可换（Fatal）
//! ```
//!
//! 和 Python 的已知差别（都不在线上）：
//! - JSON 用 serde_json 的紧凑格式写出，且对象键按字母序（Python `json.dumps` 是 `", "` 分隔、
//!   保留插入序、非 ASCII 转义）。解析后的值相同，字节不同。
//! - 错误消息里 Python 的 `repr(dict)`（[`py_repr`]）对来自服务端的对象按键的字母序排，
//!   Python 按服务端给的顺序排。
//! - Python 里会以 `KeyError` 冲出 `create` 的情形（`thread/start` 回复没有 `thread.id`、
//!   `turn/start` 回复没有 `turn.id`）在这里是 [`TeacherError::Unexpected`]。

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Condvar, Mutex};
use serde_json::{json, Map, Value};

/// 第一包前面那段说明，逐字照抄 codex_teacher.py:30。
pub const CODEX_FIRST_INTRO: &str = "Act as the autonomous policy agent for this single simulation rollout. This message is the first observation packet. In this session a robodojo_execute call is written as your final message: exactly the JSON arguments of one robodojo_execute call (this turn's output schema), with no tool use and no other text. The result of the call arrives as the next message. The user authorizes sending this episode's three RGB views, end-effector state, the OpenWAM proposal, the task text and same-episode history to the model for online decisions.";

/// codexErrorInfo 里换个账号可能有用的码（codex_teacher.py:37）。
pub const ACCOUNT_CODES: &[&str] = &["usageLimitExceeded", "unauthorized"];
/// 晚点重试可能有用的码（codex_teacher.py:38）。
pub const RETRY_CODES: &[&str] = &[
    "rateLimitExceeded",
    "serverOverloaded",
    "internalServerError",
    "other",
    "httpConnectionFailed",
    "responseStreamConnectionFailed",
    "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
];
/// 老师用不着的 Codex 特性，按 Python 的顺序逐个 `-c features.X=false`（codex_teacher.py:41）。
pub const FEATURES_OFF: &[&str] = &[
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "plugins",
    "remote_plugin",
    "goals",
    "skill_search",
    "tool_suggest",
    "in_app_browser",
    "hooks",
    "multi_agent",
    "shell_tool",
    "unified_exec",
    "sleep_tool",
    "memories",
];
/// 子进程环境里擦掉的变量：登录只从 `CODEX_HOME` 来（codex_teacher.py:44）。
pub const SCRUBBED_ENV: &[&str] = &[
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
];
const CLOSED_METHOD: &str = "__closed__";

// ---------------------------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------------------------

/// `review.teacher` 里本客户端读的那几项（Python 的 `cfg` dict）。
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherConfig {
    pub model: String,
    pub reasoning_effort: String,
    pub request_timeout_s: f64,
    /// `cfg.get("max_attempts", 4)`（codex_teacher.py:372）。
    pub max_attempts: u32,
    /// `cfg.get("retry_delays_s") or [5]`（codex_teacher.py:373）：空表也按 `[5]`。
    pub retry_delays_s: Vec<f64>,
    /// `cfg.get("codex_program", "codex")`（codex_teacher.py:275）。
    pub codex_program: String,
}

impl TeacherConfig {
    /// 从和 Python 同一份 dict（JSON）读出来，缺省值与 Python 相同。
    pub fn from_json(cfg: &Value) -> Result<Self, String> {
        let text = |key: &str| {
            cfg.get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("teacher config: {key} must be a string"))
        };
        let delays: Vec<f64> = match cfg.get("retry_delays_s") {
            Some(Value::Array(values)) if !values.is_empty() => values
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or("teacher config: retry_delays_s must be numbers")
                })
                .collect::<Result<_, _>>()?,
            Some(Value::Null) | None => vec![5.0],
            Some(Value::Array(_)) => vec![5.0],
            Some(_) => return Err("teacher config: retry_delays_s must be a list".into()),
        };
        Ok(Self {
            model: text("model")?,
            reasoning_effort: text("reasoning_effort")?,
            request_timeout_s: cfg
                .get("request_timeout_s")
                .and_then(Value::as_f64)
                .ok_or("teacher config: request_timeout_s must be a number")?,
            max_attempts: match cfg.get("max_attempts") {
                None => 4,
                Some(v) => v
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or("teacher config: max_attempts must be a non-negative integer")?,
            },
            retry_delays_s: delays,
            codex_program: match cfg.get("codex_program") {
                None => "codex".into(),
                Some(_) => text("codex_program")?,
            },
        })
    }
}

/// 启动重试的三个常量（codex_teacher.py:46-48）。Python 的测试用 monkeypatch 改它们，
/// 这里是字段。
#[derive(Debug, Clone, PartialEq)]
pub struct LaunchPolicy {
    pub attempts: u32,
    pub timeout: Duration,
    pub backoff: Vec<Duration>,
}

impl Default for LaunchPolicy {
    fn default() -> Self {
        Self {
            attempts: 3,
            timeout: Duration::from_secs(120),
            backoff: vec![Duration::from_secs(10), Duration::from_secs(30)],
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 错误
// ---------------------------------------------------------------------------------------------

/// Python 在 `create` / `preflight` 里接住的三类异常（`OSError, RuntimeError, TimeoutError`），
/// 外加它接不住的 `KeyError`。`type_name` 就是 Python 的 `type(exc).__name__`，进错误记录。
#[derive(Debug, Clone, PartialEq)]
pub enum RpcError {
    Runtime(String),
    Timeout(String),
    /// `OSError` 及其子类；第一个字段是 Python 会报的类名（`FileNotFoundError` 等）。
    Os(String, String),
    /// Python 的 `KeyError`：不被 `create` 接住，直接冲出去。
    Key(String),
}

impl RpcError {
    pub fn type_name(&self) -> &str {
        match self {
            RpcError::Runtime(_) => "RuntimeError",
            RpcError::Timeout(_) => "TimeoutError",
            RpcError::Os(name, _) => name,
            RpcError::Key(_) => "KeyError",
        }
    }

    fn os(err: &std::io::Error, filename: Option<&str>) -> Self {
        use std::io::ErrorKind as K;
        let name = match err.kind() {
            K::NotFound => "FileNotFoundError",
            K::PermissionDenied => "PermissionError",
            K::BrokenPipe => "BrokenPipeError",
            K::AlreadyExists => "FileExistsError",
            _ => "OSError",
        };
        RpcError::Os(name.into(), py_os_message(err, filename))
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpcError::Runtime(m) | RpcError::Timeout(m) | RpcError::Os(_, m) => f.write_str(m),
            // str(KeyError('id')) == "'id'"
            RpcError::Key(k) => f.write_str(&py_str_repr(k)),
        }
    }
}

/// Python 的 `TeacherFatal` / `TeacherUnavailable`（teacher.py:76-85），各带 `errors` 列表。
#[derive(Debug, Clone, PartialEq)]
pub enum TeacherError {
    /// 换账号、重试都帮不上：额度、登录、模型被拒。
    Fatal { message: String, errors: Vec<Value> },
    /// 在给定次数内没够到模型。
    Unavailable { message: String, errors: Vec<Value> },
    /// Python 会以 `KeyError` 冲出去的情形（不在它的三类异常里）。
    Unexpected(String),
}

impl fmt::Display for TeacherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TeacherError::Fatal { message, .. } | TeacherError::Unavailable { message, .. } => {
                f.write_str(message)
            }
            TeacherError::Unexpected(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for TeacherError {}

// ---------------------------------------------------------------------------------------------
// 纯函数：与 Python 同名
// ---------------------------------------------------------------------------------------------

/// 登录目录表：一行一个绝对路径，`#` 起注释（codex_teacher.py:51）。
pub fn read_accounts(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut homes = Vec::new();
    // Python 的 str.splitlines()：\n、\r\n、\r 都断行（还有几个少见的分隔符，这里不管）。
    for line in text.split(['\n', '\r']) {
        let line = line.split('#').next().unwrap_or("").trim();
        if !line.is_empty() {
            if !Path::new(line).is_absolute() {
                return Err(format!(
                    "codex account list {}: {} is not an absolute path",
                    path.display(),
                    py_str_repr(line)
                ));
            }
            homes.push(line.to_owned());
        }
    }
    if homes.is_empty() {
        return Err(format!(
            "codex account list {} names no login directory",
            path.display()
        ));
    }
    Ok(homes)
}

/// TurnError 的 codexErrorInfo 码：字符串，或以码为键的对象（codex_teacher.py:65）。
pub fn error_code(error: Option<&Value>) -> Option<String> {
    let info = error.and_then(Value::as_object)?.get("codexErrorInfo")?;
    match info {
        Value::String(s) => Some(s.clone()),
        // Python 取 dict 的第一个键；serde_json 的 Map 是有序（字母序）的，单键对象两边一样。
        Value::Object(map) => map.keys().next().cloned(),
        _ => None,
    }
}

/// 账号被拒，翻成运行时与评测器已经认识的 provider 错误（codex_teacher.py:75）。
pub fn refusal(code: Option<&str>) -> Value {
    if code == Some("unauthorized") {
        return json!({"status": 401, "type": "authentication_error", "code": "codex_unauthorized"});
    }
    json!({"status": 429, "type": "insufficient_quota", "code": "codex_usage_limit_exceeded"})
}

/// 本 turn 的 token：线程累计值的差（codex_teacher.py:483）。
pub fn usage_delta(before: &Value, after: &Value) -> Value {
    let d = |key: &str| {
        let get = |v: &Value| py_int(v.get(key));
        (get(after) - get(before)).max(0)
    };
    json!({
        "input_tokens": d("inputTokens"),
        "input_tokens_details": {"cached_tokens": d("cachedInputTokens")},
        "output_tokens": d("outputTokens"),
        "output_tokens_details": {"reasoning_tokens": d("reasoningOutputTokens")},
        "total_tokens": d("totalTokens"),
    })
}

/// 额度快照的摘要（codex_teacher.py:492）。
pub fn limit_summary(snapshot: Option<&Value>) -> Value {
    let Some(snapshot) = snapshot.filter(|s| truthy(s)) else {
        return Value::Null;
    };
    let get = |v: &Value, k: &str| v.get(k).cloned().unwrap_or(Value::Null);
    let mut out = Map::new();
    out.insert("limit_id".into(), get(snapshot, "limitId"));
    out.insert("plan".into(), get(snapshot, "planType"));
    out.insert("reached".into(), get(snapshot, "rateLimitReachedType"));
    for key in ["primary", "secondary"] {
        if let Some(window) = snapshot.get(key).filter(|w| truthy(w)) {
            out.insert(
                key.into(),
                json!({
                    "used_percent": get(window, "usedPercent"),
                    "window_minutes": get(window, "windowDurationMins"),
                    "resets_at": get(window, "resetsAt"),
                }),
            );
        }
    }
    out.insert("credits".into(), credit_summary(snapshot.get("credits")));
    Value::Object(out)
}

/// 读数里的 credits 是否可用（`credits.hasCredits` 或 `credits.unlimited`；codex_teacher.py has_credits）。
pub fn has_credits(snapshot: &Value) -> bool {
    snapshot
        .get("credits")
        .filter(|c| c.is_object())
        .is_some_and(|c| {
            c.get("hasCredits").is_some_and(truthy) || c.get("unlimited").is_some_and(truthy)
        })
}

/// credits 的摘要（codex_teacher.py credit_summary）。
fn credit_summary(credits: Option<&Value>) -> Value {
    match credits.filter(|c| c.is_object()) {
        Some(c) => {
            let get = |k: &str| c.get(k).cloned().unwrap_or(Value::Null);
            json!({"has_credits": get("hasCredits"), "unlimited": get("unlimited"), "balance": get("balance")})
        }
        None => Value::Null,
    }
}

/// 一个码该怎么办（codex_teacher.py:402-405）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeClass {
    /// 换下一个账号（`ACCOUNT_CODES`）。
    Account,
    /// 重试（`RETRY_CODES`，或者根本没有码：超时、中断、进程退出）。
    Retry,
    /// 别的码：不重试。
    GiveUp,
}

pub fn classify(code: Option<&str>) -> CodeClass {
    match code {
        Some(c) if ACCOUNT_CODES.contains(&c) => CodeClass::Account,
        Some(c) if RETRY_CODES.contains(&c) => CodeClass::Retry,
        None => CodeClass::Retry,
        Some(_) => CodeClass::GiveUp,
    }
}

// ---------------------------------------------------------------------------------------------
// 对话（Python CodexConversation）
// ---------------------------------------------------------------------------------------------

/// 每个 episode 一份：下一个 turn 的输入（历史在 Codex 线程里）。`items` 是调用记录要的
/// 输入/输出日志，这里从不裁剪（codex_teacher.py:159）。
#[derive(Debug, Clone)]
pub struct CodexConversation {
    model: String,
    reasoning_effort: String,
    output_schema: Value,
    pub items: Vec<Value>,
    pub pending: Vec<Value>,
    pub cache_key: String,
    pub recorded: usize,
}

impl CodexConversation {
    /// `output_schema` 是 Python `review.response_schema()` 的值；Rust 侧没有那份 schema 的
    /// 源头，所以由调用方给。
    pub fn new(cfg: &TeacherConfig, episode_id: &str, output_schema: Value) -> Self {
        Self {
            model: cfg.model.clone(),
            reasoning_effort: cfg.reasoning_effort.clone(),
            output_schema,
            items: Vec::new(),
            pending: Vec::new(),
            cache_key: format!("oeh-review-{episode_id}"),
            recorded: 0,
        }
    }

    /// codex_teacher.py:173：`create` 只读 `input` 和 `output_schema`，其余是给调用记录的。
    pub fn body(&self) -> Value {
        json!({
            "model": self.model,
            "input": self.pending,
            "transport": "codex_app_server",
            "reasoning": {"effort": self.reasoning_effort},
            "prompt_cache_key": self.cache_key,
            "tools": [],
            "output_schema": self.output_schema,
        })
    }

    fn add(&mut self, content: Vec<Value>) {
        self.pending.extend(content.iter().cloned());
        self.items.extend(content);
    }

    /// 一个观测包：可选的第一包说明、包的 JSON 文本、每张图一个 `{"type":"image","url":…}`。
    /// `images` 的每一项带 `image_url`（Responses 的 `input_image` 形状），同 codex_teacher.py:182。
    pub fn add_packet(&mut self, text: &str, images: &[Value], intro: bool) {
        let mut content = Vec::new();
        if intro {
            content.push(json!({"type": "text", "text": CODEX_FIRST_INTRO}));
        }
        content.push(json!({"type": "text", "text": text}));
        for image in images {
            content.push(json!({"type": "image", "url": image.get("image_url").cloned().unwrap_or(Value::Null)}));
        }
        self.add(content);
    }

    /// 一次校验失败或准入拒绝的反馈文本（重问）。
    pub fn add_error(&mut self, text: &str) {
        self.add(vec![json!({"type": "text", "text": text})]);
    }

    /// 记下输出、清空待发输入，返回其中的 `function_call`（codex_teacher.py:190）。
    pub fn add_output(&mut self, output: &[Value]) -> Option<Value> {
        let mut call = None;
        for item in output {
            self.items.push(item.clone());
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                call = Some(item.clone());
            }
        }
        self.pending.clear();
        call
    }

    pub fn unrecorded(&mut self) -> Vec<Value> {
        let items = self.items[self.recorded..].to_vec();
        self.recorded = self.items.len();
        items
    }
}

// ---------------------------------------------------------------------------------------------
// 关闭信号（Python 的 threading.Event）
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Event {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Event {
    fn set(&self) {
        *self.flag.lock() = true;
        self.cv.notify_all();
    }

    fn is_set(&self) -> bool {
        *self.flag.lock()
    }

    /// `Event.wait(timeout)`：置位了就是 true。
    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut flag = self.flag.lock();
        while !*flag {
            if self.cv.wait_until(&mut flag, deadline).timed_out() {
                break;
            }
        }
        *flag
    }
}

// ---------------------------------------------------------------------------------------------
// JSON-RPC over stdio（Python Rpc）
// ---------------------------------------------------------------------------------------------

/// app-server 的 stdio 上的 JSON-RPC 2.0：回复按 id 交付，其余（通知、服务端请求）按到达顺序
/// 排队。stderr 写本地文件（codex_teacher.py:82）。
struct Rpc {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, Sender<Value>>>>,
    incoming: Mutex<Receiver<Value>>,
    next: Mutex<u64>,
    closed: AtomicBool,
}

impl Rpc {
    fn spawn(
        argv: &[String],
        env: &EnvPlan,
        cwd: Option<&Path>,
        stderr_path: &Path,
    ) -> Result<Arc<Self>, RpcError> {
        let stderr = File::create(stderr_path)
            .map_err(|e| RpcError::os(&e, Some(&stderr_path.display().to_string())))?;
        let mut command = Command::new(&argv[0]);
        command.args(&argv[1..]);
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        for key in &env.remove {
            command.env_remove(key);
        }
        for (key, value) in &env.set {
            command.env(key, value);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|e| RpcError::os(&e, Some(&argv[0])))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout was piped");
        let pending: Arc<Mutex<HashMap<u64, Sender<Value>>>> = Arc::default();
        let (tx, rx) = mpsc::channel();
        let replies = pending.clone();
        std::thread::Builder::new()
            .name("oeh-codex-teacher-read".into())
            .spawn(move || read_loop(stdout, replies, tx))
            .map_err(|e| RpcError::os(&e, None))?;
        Ok(Arc::new(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending,
            incoming: Mutex::new(rx),
            next: Mutex::new(1),
            closed: AtomicBool::new(false),
        }))
    }

    /// `process.poll()`：还在跑是 None，否则是 Python 的 returncode。
    fn poll(&self) -> Option<i32> {
        self.child.lock().try_wait().ok().flatten().map(returncode)
    }

    fn send(&self, message: &Value) -> Result<(), RpcError> {
        let mut stdin = self.stdin.lock();
        let result = match stdin.as_mut() {
            None => Err("I/O operation on closed file.".to_owned()),
            Some(pipe) => {
                let mut line = serde_json::to_string(message).expect("JSON values serialise");
                line.push('\n');
                pipe.write_all(line.as_bytes())
                    .and_then(|_| pipe.flush())
                    .map_err(|e| py_os_message(&e, None))
            }
        };
        result.map_err(|exc| {
            let code = self.poll().map_or("None".to_owned(), |c| c.to_string());
            RpcError::Runtime(format!("the app-server exited (code {code}): {exc}"))
        })
    }

    fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, RpcError> {
        let identifier = {
            let mut next = self.next.lock();
            let id = *next;
            *next += 1;
            id
        };
        let (tx, rx) = mpsc::channel();
        self.pending.lock().insert(identifier, tx);
        let message =
            json!({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params});
        if let Err(e) = self.send(&message) {
            self.pending.lock().remove(&identifier);
            return Err(e);
        }
        let deadline = Instant::now() + timeout;
        // codex_teacher.py:125：每秒醒一次看进程还在不在，免得一个死掉的进程让人等满超时。
        let reply = loop {
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_secs(1));
            match rx.recv_timeout(wait) {
                Ok(reply) => break reply,
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                    if let Some(code) = self.poll() {
                        self.pending.lock().remove(&identifier);
                        return Err(RpcError::Runtime(format!(
                            "app-server {method}: the app-server exited (code {code})"
                        )));
                    }
                    if Instant::now() >= deadline {
                        self.pending.lock().remove(&identifier);
                        return Err(RpcError::Timeout(format!(
                            "app-server {method}: no reply in {} s",
                            py_g(timeout.as_secs_f64())
                        )));
                    }
                }
            }
        };
        if let Some(error) = reply.get("error") {
            return Err(RpcError::Runtime(format!(
                "app-server {method}: {}",
                py_repr(error)
            )));
        }
        // `reply.get("result") or {}`
        Ok(reply
            .get("result")
            .filter(|r| truthy(r))
            .cloned()
            .unwrap_or_else(|| json!({})))
    }

    /// 服务端请求（审批、工具调用）一律拒绝（codex_teacher.py:141）。
    fn refuse(&self, message: &Value) -> Result<(), RpcError> {
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": message.get("id").cloned().unwrap_or(Value::Null),
            "error": {"code": -32601, "message": "the review teacher executes no tools or approvals"},
        }))
    }

    fn recv(&self, timeout: Duration) -> Option<Value> {
        self.incoming.lock().recv_timeout(timeout).ok()
    }

    /// 关 stdin → SIGTERM → 等 10 s → SIGKILL（codex_teacher.py:145）。可重入。
    fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        drop(self.stdin.lock().take());
        let mut child = self.child.lock();
        if child.try_wait().ok().flatten().is_none() {
            if let Ok(pid) = libc::pid_t::try_from(child.id()) {
                // SAFETY: kill(2) on our own child's pid; it is not reaped yet, so the pid is still ours.
                unsafe {
                    libc::kill(pid, libc::SIGTERM);
                }
            }
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) if Instant::now() >= deadline => break,
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn read_loop(
    stdout: std::process::ChildStdout,
    pending: Arc<Mutex<HashMap<u64, Sender<Value>>>>,
    incoming: Sender<Value>,
) {
    let mut reader = BufReader::new(stdout);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Ok(message) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        // codex_teacher.py:102：有 id 没 method 的是回复，按 id 交给等它的人；没人等就丢掉。
        if message.get("id").is_some() && message.get("method").is_none() {
            let id = message.get("id").and_then(Value::as_u64);
            if let Some(box_) = id.and_then(|id| pending.lock().remove(&id)) {
                let _ = box_.send(message);
            }
            continue;
        }
        if incoming.send(message).is_err() {
            return;
        }
    }
    let _ = incoming.send(json!({"method": CLOSED_METHOD}));
}

fn returncode(status: ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| -status.signal().unwrap_or(0))
}

/// 子进程环境的改动：在继承的环境上去掉 [`SCRUBBED_ENV`]、设 `CODEX_HOME`（codex_teacher.py:290）。
#[derive(Debug, Clone, PartialEq)]
pub struct EnvPlan {
    pub remove: Vec<String>,
    pub set: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------------------------
// 客户端（Python CodexClient）
// ---------------------------------------------------------------------------------------------

/// 从别的线程关掉客户端：置关闭标志，并关掉正在跑的 app-server（Python `close()` 跨线程调用的效果）。
#[derive(Clone)]
pub struct CloseHandle {
    closed: Arc<Event>,
    live: Arc<Mutex<Option<Arc<Rpc>>>>,
}

impl CloseHandle {
    pub fn close(&self) {
        self.closed.set();
        let live = self.live.lock().take();
        if let Some(rpc) = live {
            rpc.close();
        }
    }

    pub fn is_closed(&self) -> bool {
        self.closed.is_set()
    }
}

/// 一个 app-server、每个 episode 一条线程。`create` 跑一个 turn，按 Responses 的形状返回
/// （最终消息作为一次 `robodojo_execute` 调用的参数）。阻塞；自己不记任何东西（codex_teacher.py:248）。
pub struct CodexClient {
    cfg: TeacherConfig,
    launch: LaunchPolicy,
    closed: Arc<Event>,
    live: Arc<Mutex<Option<Arc<Rpc>>>>,
    workdir: PathBuf,
    prompt: String,
    accounts: Vec<String>,
    argv_prefix: Option<Vec<String>>,
    server: Option<Arc<Rpc>>,
    pub account: usize,
    pub thread_id: Option<Value>,
    pub thread_path: Option<Value>,
    pub total: Option<Value>,
    pub rate_limits: Option<Value>,
    pub switches: Vec<Value>,
    local: PathBuf,
    pub launches: u32,
    /// 每次起 app-server：到握手完成的秒数，或者怎么失败的（codex_teacher.py `launch_log`）。
    pub launch_log: Vec<Value>,
    /// 关闭时把 app-server 的 stderr 拷进这个目录（只为诊断；codex_teacher.py `keep_logs`）。
    logs_dir: Option<PathBuf>,
    thread_ready: bool,
    /// app-server 子进程的工作目录与额外要擦掉的环境变量。Python 的 `CodexClient` 在策略进程里
    /// 起子进程（继承那个进程的 cwd 与 `os.environ`）；脑内通道在内核里起，用这两项把子进程放回
    /// 同一个位置（[`CodexClient::with_process`]）。
    process_cwd: Option<PathBuf>,
    env_remove: Vec<String>,
    /// 接手策略进程预启动的 app-server（[`CodexClient::with_adopted`]）：第一次 `launch` 起的是这条
    /// 中继，不握手；`preflight` 直接给预启动时的那份结果。
    adopt: Option<(Vec<String>, u64)>,
    adopted_preflight: Option<Value>,
}

/// 策略进程预启动、已经握手和预检过的 app-server（codex_teacher.py `CodexClient.hand_over`）。
#[derive(Debug, Clone)]
pub struct Adopted {
    /// 中继命令：它的 stdio 就是那个 app-server 的 stdio。
    pub argv: Vec<String>,
    /// 预检时用的登录（预检可能已经换过账号）。
    pub account: usize,
    /// 预检读到的额度快照。
    pub rate_limits: Option<Value>,
    /// 那个连接上下一个请求 id（Python 的 `Rpc._next`）。
    pub next_id: u64,
    /// 预检结果（不含 prestarted / start_seconds，那两项 Python 自己加）。
    pub preflight: Value,
}

/// 一次 turn 的结果：`turn/completed` 的 turn、最终消息、原生条目、用量。
type TurnOutcome = (Value, Option<Value>, Vec<Value>, Value);

impl CodexClient {
    /// `accounts` 为 None 时读 `codex_accounts_file`（Python 的 `read_accounts(cfg[...])`）——
    /// 这里由调用方先读好再给，`argv_prefix` 只给测试用（替身 app-server）。
    pub fn new(
        cfg: TeacherConfig,
        workdir: impl Into<PathBuf>,
        prompt: impl Into<String>,
        accounts: Vec<String>,
        argv_prefix: Option<Vec<String>>,
    ) -> std::io::Result<Self> {
        // Codex 的 sqlite 状态放本地盘，不放卷（codex_teacher.py:268）。
        let local = tempfile::Builder::new()
            .prefix("oeh-codex-")
            .tempdir()?
            .keep();
        Ok(Self {
            cfg,
            launch: LaunchPolicy::default(),
            closed: Arc::default(),
            live: Arc::default(),
            workdir: workdir.into(),
            prompt: prompt.into(),
            accounts,
            argv_prefix,
            server: None,
            account: 0,
            thread_id: None,
            thread_path: None,
            total: None,
            rate_limits: None,
            switches: Vec::new(),
            local,
            launches: 0,
            launch_log: Vec::new(),
            logs_dir: None,
            thread_ready: false,
            process_cwd: None,
            env_remove: Vec::new(),
            adopt: None,
            adopted_preflight: None,
        })
    }

    /// 接手一个预启动的 app-server（TIME-1：它在 OpenWAM 加载时就起好了，不占第一次 reset 的时限）。
    /// 之后的状态和 Python 通道里那个预启动客户端一模一样：同一个进程、同一个账号、同一份额度、
    /// 同一串请求 id，第一个包来时才 `thread/start`。进程没了或换账号时，照常自己起。
    pub fn with_adopted(mut self, adopted: Adopted) -> Self {
        self.account = adopted.account;
        self.rate_limits = adopted.rate_limits;
        // 预启动的启动记录（含失败重试）接着记，后面再起的编号也接着排（Python 的 launches / launch_log）。
        self.launch_log = adopted
            .preflight
            .get("launches")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        self.launches = u32::try_from(self.launch_log.len()).unwrap_or(u32::MAX);
        self.adopt = Some((adopted.argv, adopted.next_id));
        self.adopted_preflight = Some(adopted.preflight);
        self
    }

    /// 子进程在 `cwd` 里起，并且（在 [`SCRUBBED_ENV`] 之外）不继承 `env_remove` 里的变量。
    pub fn with_process(mut self, cwd: Option<PathBuf>, env_remove: Vec<String>) -> Self {
        self.process_cwd = cwd;
        self.env_remove = env_remove;
        self
    }

    /// 关闭时 app-server 的 stderr 留到 `dir`（codex_teacher.py `logs_dir`）。
    pub fn with_logs_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.logs_dir = dir;
        self
    }

    /// codex_teacher.py `keep_logs`：只为诊断，拷不了就算了。
    fn keep_logs(&self) {
        let Some(dir) = &self.logs_dir else {
            return;
        };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        let Ok(entries) = std::fs::read_dir(&self.local) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("app-server-") && name.ends_with(".stderr") {
                let _ = std::fs::copy(entry.path(), dir.join(&*name));
            }
        }
    }

    pub fn with_launch_policy(mut self, launch: LaunchPolicy) -> Self {
        self.launch = launch;
        self
    }

    pub fn close_handle(&self) -> CloseHandle {
        CloseHandle {
            closed: self.closed.clone(),
            live: self.live.clone(),
        }
    }

    /// 本客户端的 sqlite / log 目录（argv 里的 `sqlite_home` / `log_dir` 在它下面）。
    pub fn local_dir(&self) -> &Path {
        &self.local
    }

    pub fn has_server(&self) -> bool {
        self.server.is_some()
    }

    fn set_server(&mut self, server: Option<Arc<Rpc>>) {
        *self.live.lock() = server.clone();
        self.server = server;
    }

    fn drop_server(&mut self) {
        if let Some(server) = self.server.take() {
            server.close();
        }
        *self.live.lock() = None;
    }

    // -- 进程与线程 ------------------------------------------------------------------------------

    /// codex_teacher.py:273。
    pub fn argv(&self) -> Vec<String> {
        let effort = &self.cfg.reasoning_effort;
        let mut argv = self
            .argv_prefix
            .clone()
            .unwrap_or_else(|| vec![self.cfg.codex_program.clone(), "app-server".into()]);
        let mut overrides = vec![
            format!("model=\"{}\"", self.cfg.model),
            format!("model_reasoning_effort=\"{effort}\""),
            "analytics.enabled=false".to_owned(),
            format!("sqlite_home=\"{}\"", self.local.join("state").display()),
            format!("log_dir=\"{}\"", self.local.join("log").display()),
        ];
        overrides.extend(
            FEATURES_OFF
                .iter()
                .map(|name| format!("features.{name}=false")),
        );
        for value in overrides {
            argv.push("-c".into());
            argv.push(value);
        }
        argv
    }

    /// codex_teacher.py:290-291。
    pub fn env_plan(&self) -> EnvPlan {
        EnvPlan {
            remove: SCRUBBED_ENV
                .iter()
                .map(|k| (*k).to_owned())
                .chain(self.env_remove.iter().cloned())
                .collect(),
            set: vec![("CODEX_HOME".into(), self.accounts[self.account].clone())],
        }
    }

    /// 起 app-server、握手；起不来或不答的再试（codex_teacher.py:283）。
    fn launch(&mut self) -> Result<(), RpcError> {
        if let Some((argv, next_id)) = self.adopt.take() {
            if !self.closed.is_set() {
                // 预启动的那个已经握过手：只接上中继，不再 initialize。
                let env = self.env_plan();
                self.thread_ready = false;
                // 中继本身不是一次启动：编号不加，stderr 也不和 app-server 的混在一起。
                let stderr_path = self.local.join("relay.stderr");
                let server = Rpc::spawn(&argv, &env, self.process_cwd.as_deref(), &stderr_path)?;
                *server.next.lock() = next_id;
                self.set_server(Some(server));
                return Ok(());
            }
        }
        let mut errors: Vec<String> = Vec::new();
        for attempt in 0..self.launch.attempts {
            if self.closed.is_set() {
                break;
            }
            let env = self.env_plan();
            self.launches += 1;
            self.thread_ready = false;
            let started = Instant::now();
            let stderr_path = self
                .local
                .join(format!("app-server-{}.stderr", self.launches));
            // Python 的 Rpc(...) 在 try 之外（codex_teacher.py:294）：起不了进程是 OSError，直接冲出去、不重试。
            let server = Rpc::spawn(
                &self.argv(),
                &env,
                self.process_cwd.as_deref(),
                &stderr_path,
            )?;
            self.set_server(Some(server.clone()));
            let handshake = server
                .request(
                    "initialize",
                    json!({"clientInfo": {"name": "oeh-review-teacher", "version": "1"},
                           "capabilities": {"experimentalApi": true}}),
                    self.launch.timeout,
                )
                .and_then(|_| {
                    server.send(&json!({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
                });
            let seconds = round1(started.elapsed().as_secs_f64());
            match handshake {
                Ok(()) => {
                    self.launch_log
                        .push(json!({"launch": self.launches, "seconds": seconds}));
                    return Ok(());
                }
                Err(RpcError::Key(k)) => return Err(RpcError::Key(k)),
                Err(exc) => {
                    errors.push(format!("{}: {exc}", exc.type_name()));
                    let cut: String = errors
                        .last()
                        .map_or("", String::as_str)
                        .chars()
                        .take(200)
                        .collect();
                    self.launch_log
                        .push(json!({"launch": self.launches, "seconds": seconds, "error": cut}));
                    self.drop_server();
                }
            }
            if attempt + 1 < self.launch.attempts {
                let backoff = &self.launch.backoff;
                let delay = backoff
                    .get((attempt as usize).min(backoff.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or_default();
                if self.closed.wait(delay) {
                    break;
                }
            }
        }
        Err(RpcError::Runtime(format!(
            "the codex app-server did not start in {} attempt(s): {}; launches {}",
            errors.len(),
            errors.last().map_or("client closed", String::as_str),
            py_repr(&Value::Array(self.launch_log.clone()))
        )))
    }

    /// codex_teacher.py:309。
    fn thread_params(&self) -> Map<String, Value> {
        let mut params = Map::new();
        params.insert("model".into(), json!(self.cfg.model));
        params.insert("cwd".into(), json!(self.workdir.display().to_string()));
        params.insert("approvalPolicy".into(), json!("never"));
        params.insert("sandbox".into(), json!("read-only"));
        params.insert("baseInstructions".into(), json!(self.prompt));
        params.insert(
            "config".into(),
            json!({"model_reasoning_effort": self.cfg.reasoning_effort}),
        );
        params
    }

    /// 新线程，或者（换账号之后）从 rollout 文件续上同一条线程（codex_teacher.py:314）。
    fn open_thread(&mut self) -> Result<(), RpcError> {
        let server = self.server.clone().expect("open_thread runs after launch");
        let mut params = self.thread_params();
        let response = match &self.thread_id {
            None => {
                params.insert("environments".into(), json!([]));
                params.insert("ephemeral".into(), json!(false));
                server.request(
                    "thread/start",
                    Value::Object(params),
                    Duration::from_secs(120),
                )?
            }
            Some(thread_id) => {
                params.insert("threadId".into(), thread_id.clone());
                params.insert(
                    "path".into(),
                    self.thread_path.clone().unwrap_or(Value::Null),
                );
                server.request(
                    "thread/resume",
                    Value::Object(params),
                    Duration::from_secs(120),
                )?
            }
        };
        let model = response.get("model").cloned().unwrap_or(Value::Null);
        let effort = response
            .get("reasoningEffort")
            .cloned()
            .unwrap_or(Value::Null);
        if model.as_str() != Some(&self.cfg.model)
            || effort.as_str() != Some(&self.cfg.reasoning_effort)
        {
            return Err(RpcError::Runtime(format!(
                "app-server runs {}/{}, not the frozen {}/{}",
                py_str(&model),
                py_str(&effort),
                self.cfg.model,
                self.cfg.reasoning_effort
            )));
        }
        let thread = response
            .get("thread")
            .filter(|t| truthy(t))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let id = thread
            .get("id")
            .cloned()
            .ok_or_else(|| RpcError::Key("id".into()))?;
        self.thread_id = Some(id);
        if let Some(path) = thread.get("path").filter(|p| truthy(p)) {
            self.thread_path = Some(path.clone());
        }
        Ok(())
    }

    /// codex_teacher.py:328。
    fn ensure(&mut self) -> Result<(), RpcError> {
        let alive = self.server.as_ref().is_some_and(|s| s.poll().is_none());
        if !alive {
            self.launch()?;
        }
        if !self.thread_ready {
            self.open_thread()?;
            self.thread_ready = true;
        }
        Ok(())
    }

    /// 换下一个登录；没有了返回 false（codex_teacher.py:335）。
    fn switch_account(&mut self, code: Option<&str>) -> bool {
        if self.account + 1 >= self.accounts.len() {
            return false;
        }
        self.switches.push(json!({
            "from_account": self.account, "to_account": self.account + 1, "code": code, "at": unix_now(),
        }));
        self.drop_server();
        self.account += 1;
        true
    }

    // -- 脑的接口 -------------------------------------------------------------------------------

    /// 登录能跑、能用冻结的模型；额度满了的账号交给下一个（codex_teacher.py:348）。
    pub fn preflight(&mut self) -> Result<Value, TeacherError> {
        if let Some(result) = self.adopted_preflight.take() {
            // 预检已经在策略进程里跑过（同一个进程、同样的请求）：接上它就是这次的预检。
            return match self.launch() {
                Ok(()) => Ok(result),
                Err(RpcError::Key(k)) => Err(TeacherError::Unexpected(format!("KeyError: '{k}'"))),
                Err(exc) => Err(TeacherError::Unavailable {
                    message: format!("codex app-server preflight failed: {exc}"),
                    errors: Vec::new(),
                }),
            };
        }
        // 都到额度时，第一个读数里有可用 credits 的账号用 credits 跑（先用各计划窗口，再用 Pro 账号的 credits）。
        let mut credit_account: Option<usize> = None;
        let mut on_credits = false;
        loop {
            let probed = self.launch().and_then(|()| {
                let server = self.server.clone().expect("launch leaves a server");
                let started = Instant::now();
                let models = server.request("model/list", json!({}), Duration::from_secs(60))?;
                let limits = server.request(
                    "account/rateLimits/read",
                    json!({}),
                    Duration::from_secs(60),
                )?;
                Ok((models, limits, round1(started.elapsed().as_secs_f64())))
            });
            let (models, limits, requests_seconds) = match probed {
                Ok(v) => v,
                Err(RpcError::Key(k)) => {
                    return Err(TeacherError::Unexpected(format!("KeyError: '{k}'")))
                }
                Err(exc) => {
                    return Err(TeacherError::Unavailable {
                        message: format!("codex app-server preflight failed: {exc}"),
                        errors: Vec::new(),
                    })
                }
            };
            let served = models
                .get("data")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|m| {
                    m.get("id")
                        .filter(|v| truthy(v))
                        .or_else(|| m.get("model"))
                        .and_then(Value::as_str)
                })
                .any(|name| name == Some(self.cfg.model.as_str()));
            if !served {
                return Err(TeacherError::Fatal {
                    message: format!("codex does not serve {}", self.cfg.model),
                    errors: vec![
                        json!({"status": 404, "type": "invalid_request_error", "code": "model_not_found"}),
                    ],
                });
            }
            let snapshot = limits
                .get("rateLimits")
                .filter(|v| truthy(v))
                .cloned()
                .unwrap_or(limits);
            self.rate_limits = Some(snapshot.clone());
            let reached = snapshot.get("rateLimitReachedType").is_some_and(truthy);
            let result = |client: &Self| {
                let mut out = json!({
                    "model": client.cfg.model, "transport": "codex_app_server", "account": client.account,
                    "accounts": client.accounts.len(), "rate_limits": limit_summary(Some(&snapshot)),
                    "launches": client.launch_log.clone(), "requests_seconds": requests_seconds,
                });
                if reached {
                    out["on_credits"] = Value::Bool(true);
                }
                out
            };
            if !reached || (on_credits && has_credits(&snapshot)) {
                return Ok(result(self));
            }
            if credit_account.is_none() && has_credits(&snapshot) {
                credit_account = Some(self.account);
            }
            if self.switch_account(Some("usageLimitExceeded")) {
                continue;
            }
            let Some(credit) = credit_account.filter(|_| !on_credits) else {
                return Err(TeacherError::Fatal {
                    message: "every codex account is at its usage limit".into(),
                    errors: vec![refusal(Some("usageLimitExceeded"))],
                });
            };
            on_credits = true;
            if credit != self.account {
                self.switches.push(json!({
                    "from_account": self.account, "to_account": credit, "code": "credits", "at": unix_now(),
                }));
                self.drop_server();
                self.account = credit;
                continue;
            }
            return Ok(result(self));
        }
    }

    /// 跑一个 turn（必要时重试、换账号），返回 `(data, errors)`，同 codex_teacher.py:370。
    /// `body` 是 [`CodexConversation::body`]：只读 `input` 和 `output_schema`。
    pub fn create(&mut self, body: &Value) -> Result<(Value, Vec<Value>), TeacherError> {
        let attempts = self.cfg.max_attempts;
        let delays = self.cfg.retry_delays_s.clone();
        let mut errors: Vec<ErrorRecord> = Vec::new();
        let mut tries: u32 = 0;
        while tries < attempts {
            if self.closed.is_set() {
                return Err(TeacherError::Unavailable {
                    message: "teacher client closed".into(),
                    errors: errors.iter().map(ErrorRecord::to_value).collect(),
                });
            }
            let started = Instant::now();
            let outcome = self.ensure().and_then(|()| self.turn(body));
            let (code, mut error) = match outcome {
                Err(RpcError::Key(k)) => {
                    return Err(TeacherError::Unexpected(format!("KeyError: '{k}'")))
                }
                Err(exc) => (
                    None,
                    ErrorRecord::new(exc.type_name(), None, false, &exc.to_string()),
                ),
                Ok((turn, final_, natives, usage)) => {
                    let status = turn.get("status").cloned().unwrap_or(Value::Null);
                    if status.as_str() == Some("completed") {
                        let mut output = Vec::new();
                        if let Some(final_) = final_.filter(truthy) {
                            output.push(json!({"type": "function_call", "call_id": turn["id"],
                                               "name": "robodojo_execute", "arguments": final_}));
                        }
                        let data = json!({
                            "id": turn["id"], "status": "completed", "output": output, "usage": usage,
                            "codex": {
                                "account": self.account,
                                "thread_id": self.thread_id.clone().unwrap_or(Value::Null),
                                "duration_ms": turn.get("durationMs").cloned().unwrap_or(Value::Null),
                                "native_items": natives,
                                "compaction_telemetry_enabled": true,
                                "rate_limits": limit_summary(self.rate_limits.as_ref()),
                                "switches": self.switches,
                            },
                        });
                        return Ok((data, errors.iter().map(ErrorRecord::to_value).collect()));
                    }
                    let turn_error = turn.get("error");
                    let code = error_code(turn_error);
                    let message = turn_error
                        .filter(|e| truthy(e))
                        .and_then(|e| e.get("message"))
                        .filter(|m| truthy(m))
                        .map(py_str)
                        .unwrap_or_default();
                    let record = ErrorRecord::new(
                        &format!("codex_turn_{}", py_str(&status)),
                        code.clone(),
                        true,
                        &message,
                    );
                    (code, record)
                }
            };
            tries += 1;
            let class = classify(code.as_deref());
            let retry = class == CodeClass::Retry;
            error.attempt = tries;
            error.seconds = round3(started.elapsed().as_secs_f64());
            error.account = self.account;
            error.retry = retry && tries < attempts;
            error.fatal = false;
            errors.push(error);
            if class == CodeClass::Account {
                if self.switch_account(code.as_deref()) {
                    tries -= 1; // 新账号给这个包完整的重试预算（codex_teacher.py:407）
                    continue;
                }
                let last = errors.last_mut().expect("just pushed");
                last.fatal = true;
                let mut values: Vec<Value> = errors.iter().map(ErrorRecord::to_value).collect();
                let tail = values.last_mut().expect("just pushed");
                if let (Value::Object(tail), Value::Object(r)) = (tail, refusal(code.as_deref())) {
                    tail.extend(r);
                }
                return Err(TeacherError::Fatal {
                    message: format!(
                        "every codex account refused: {}",
                        code.as_deref().unwrap_or("None")
                    ),
                    errors: values,
                });
            }
            if !retry || tries >= attempts {
                break;
            }
            let delay = delays[((tries - 1) as usize).min(delays.len() - 1)];
            if self.closed.wait(Duration::from_secs_f64(delay.max(0.0))) {
                break;
            }
        }
        Err(TeacherError::Unavailable {
            message: format!(
                "teacher unavailable after {} attempt(s): {}",
                errors.len(),
                errors.last().map(ErrorRecord::py_repr).unwrap_or_default()
            ),
            errors: errors.iter().map(ErrorRecord::to_value).collect(),
        })
    }

    /// codex_teacher.py:419。
    fn turn(&mut self, body: &Value) -> Result<TurnOutcome, RpcError> {
        let server = self.server.clone().expect("ensure leaves a server");
        let before = self.total.clone().unwrap_or_else(|| json!({}));
        let result = server.request(
            "turn/start",
            json!({
                "threadId": self.thread_id.clone().unwrap_or(Value::Null),
                "input": body.get("input").cloned().unwrap_or(Value::Null),
                "outputSchema": body.get("output_schema").cloned().unwrap_or(Value::Null),
                "effort": self.cfg.reasoning_effort,
            }),
            Duration::from_secs(60),
        )?;
        let turn_id = result
            .get("turn")
            .filter(|t| truthy(t))
            .and_then(|t| t.get("id"))
            .cloned()
            .ok_or_else(|| RpcError::Key("id".into()))?;
        let mut deadline =
            Instant::now() + Duration::from_secs_f64(self.cfg.request_timeout_s.max(0.0));
        let mut final_: Option<Value> = None;
        let mut natives: Vec<Value> = Vec::new();
        let mut interrupted = false;
        loop {
            if self.closed.is_set() && !interrupted {
                self.interrupt(&server, &turn_id);
                interrupted = true;
                deadline = deadline.min(Instant::now() + Duration::from_secs(30));
            }
            let mut remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() && !interrupted {
                self.interrupt(&server, &turn_id);
                interrupted = true;
                // 被中断的 turn 还会报它的结束（codex_teacher.py:436）。
                deadline = Instant::now() + Duration::from_secs(30);
                remaining = Duration::from_secs(30);
            } else if remaining.is_zero() {
                return Err(RpcError::Timeout(
                    "codex turn neither completed nor ended after an interrupt".into(),
                ));
            }
            let Some(message) = server.recv(remaining.min(Duration::from_secs(1))) else {
                continue;
            };
            let method = message.get("method").cloned().unwrap_or(Value::Null);
            let params = message
                .get("params")
                .filter(|p| truthy(p))
                .cloned()
                .unwrap_or_else(|| json!({}));
            if method.as_str() == Some(CLOSED_METHOD) {
                return Err(RpcError::Runtime(
                    "codex app-server exited during a turn".into(),
                ));
            }
            if message.get("id").is_some() {
                // 服务端请求（审批、工具调用）：老师什么都不执行（codex_teacher.py:447）。
                natives.push(json!({"request": method}));
                server.refuse(&message)?;
                continue;
            }
            match params.get("threadId") {
                None | Some(Value::Null) => {}
                Some(t) if Some(t) == self.thread_id.as_ref() => {}
                Some(_) => continue,
            }
            match method.as_str() {
                Some("item/completed") => {
                    let item = params
                        .get("item")
                        .filter(|i| truthy(i))
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let kind = item.get("type").cloned().unwrap_or(Value::Null);
                    match kind.as_str() {
                        Some("agentMessage") => {
                            if let Some(text) = item.get("text").filter(|t| truthy(t)) {
                                final_ = Some(text.clone());
                            }
                        }
                        Some("reasoning" | "userMessage") => {}
                        _ => {
                            natives.push(json!({
                                "item": kind, "status": item.get("status").cloned().unwrap_or(Value::Null),
                            }));
                            if kind.as_str() == Some("contextCompaction") {
                                natives.push(json!({
                                    "event": "thread/compacted", "turn_id": turn_id,
                                    "source": "contextCompaction",
                                }));
                            }
                        }
                    }
                }
                Some("thread/compacted") => {
                    natives.push(json!({"event": "thread/compacted", "turn_id": turn_id}));
                }
                Some("thread/tokenUsage/updated") => {
                    let total = params
                        .get("tokenUsage")
                        .filter(|u| truthy(u))
                        .and_then(|u| u.get("total"))
                        .filter(|t| truthy(t))
                        .cloned();
                    if total.is_some() {
                        self.total = total;
                    }
                }
                Some("account/rateLimits/updated") => {
                    let mut merged = match self.rate_limits.take() {
                        Some(Value::Object(m)) => m,
                        _ => Map::new(),
                    };
                    if let Some(Value::Object(update)) = params.get("rateLimits") {
                        for (k, v) in update {
                            if !v.is_null() {
                                merged.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    self.rate_limits = Some(Value::Object(merged));
                }
                Some("turn/completed") => {
                    let turn = params.get("turn").filter(|t| truthy(t));
                    if turn.and_then(|t| t.get("id")) == Some(&turn_id) {
                        let usage = usage_delta(&before, self.total.as_ref().unwrap_or(&json!({})));
                        return Ok((params["turn"].clone(), final_, natives, usage));
                    }
                }
                _ => {}
            }
        }
    }

    fn interrupt(&self, server: &Rpc, turn_id: &Value) {
        let _ = server.request(
            "turn/interrupt",
            json!({"threadId": self.thread_id.clone().unwrap_or(Value::Null), "turnId": turn_id}),
            Duration::from_secs(30),
        );
    }

    /// codex_teacher.py:475。
    pub fn close(&mut self) {
        self.closed.set();
        self.drop_server();
        self.keep_logs();
        let _ = std::fs::remove_dir_all(&self.local);
    }
}

impl Drop for CodexClient {
    fn drop(&mut self) {
        self.drop_server();
    }
}

/// `create` 的一条错误记录，字段与顺序同 codex_teacher.py:385/399/403（顺序只对 `repr` 有意义）。
#[derive(Debug, Clone)]
struct ErrorRecord {
    type_: String,
    has_code: bool,
    code: Option<String>,
    message: String,
    attempt: u32,
    seconds: f64,
    account: usize,
    retry: bool,
    fatal: bool,
}

impl ErrorRecord {
    fn new(type_: &str, code: Option<String>, has_code: bool, message: &str) -> Self {
        Self {
            type_: type_.to_owned(),
            has_code,
            code,
            // `str(...)[:400]`：按字符截。
            message: message.chars().take(400).collect(),
            attempt: 0,
            seconds: 0.0,
            account: 0,
            retry: false,
            fatal: false,
        }
    }

    fn fields(&self) -> Vec<(&'static str, Value)> {
        let mut fields = vec![("status", Value::Null), ("type", json!(self.type_))];
        if self.has_code {
            fields.push(("code", json!(self.code)));
        }
        fields.extend([
            ("message", json!(self.message)),
            ("attempt", json!(self.attempt)),
            ("seconds", json!(self.seconds)),
            ("account", json!(self.account)),
            ("retry", json!(self.retry)),
            ("fatal", json!(self.fatal)),
        ]);
        fields
    }

    fn to_value(&self) -> Value {
        Value::Object(
            self.fields()
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        )
    }

    fn py_repr(&self) -> String {
        let body: Vec<String> = self
            .fields()
            .iter()
            .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
            .collect();
        format!("{{{}}}", body.join(", "))
    }
}

// ---------------------------------------------------------------------------------------------
// Python 语义的小工具
// ---------------------------------------------------------------------------------------------

/// Python 的真值：None、False、0、""、[]、{} 为假。
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `int(value or 0)`：数字截断成整数，数字串照 `int()` 解析，别的当 0。
fn py_int(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(n)) => n
            .as_i64()
            .unwrap_or_else(|| n.as_f64().map_or(0, |f| f.trunc() as i64)),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0),
        Some(Value::Bool(true)) => 1,
        _ => 0,
    }
}

/// Python 的 `round(x, 1)`。
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `f"{x:g}"`，只覆盖本模块会用到的超时值。
fn py_g(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// `str(value)`：字符串原样，其余同 `repr`。
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python 的 `repr`，用于错误消息里的 dict（`f"... {reply['error']}"`）。
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_str_repr(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// `repr(str)`：默认单引号，含单引号而不含双引号时用双引号。
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `str(OSError)`：`[Errno 2] No such file or directory: 'codex'`。
fn py_os_message(err: &std::io::Error, filename: Option<&str>) -> String {
    match err.raw_os_error() {
        Some(code) => {
            let text = err.to_string();
            let strerror = text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text);
            match filename {
                Some(name) => format!("[Errno {code}] {strerror}: {}", py_str_repr(name)),
                None => format!("[Errno {code}] {strerror}"),
            }
        }
        None => err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TeacherConfig {
        TeacherConfig::from_json(&json!({
            "api": "codex_app_server", "model": "gpt-6-luna", "reasoning_effort": "xhigh",
            "request_timeout_s": 10, "max_attempts": 2, "retry_delays_s": [0],
        }))
        .unwrap()
    }

    #[test]
    fn config_defaults_follow_python() {
        let c = TeacherConfig::from_json(&json!({
            "model": "m", "reasoning_effort": "high", "request_timeout_s": 1.5, "retry_delays_s": [],
        }))
        .unwrap();
        assert_eq!(c.max_attempts, 4);
        assert_eq!(c.retry_delays_s, vec![5.0]); // `or [5]`
        assert_eq!(c.codex_program, "codex");
        assert!(TeacherConfig::from_json(&json!({"model": "m"})).is_err());
    }

    #[test]
    fn error_codes_are_strings_or_single_key_objects() {
        assert_eq!(
            error_code(Some(&json!({"codexErrorInfo": "unauthorized"}))).as_deref(),
            Some("unauthorized")
        );
        assert_eq!(
            error_code(Some(
                &json!({"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 502}}})
            ))
            .as_deref(),
            Some("httpConnectionFailed")
        );
        assert_eq!(error_code(Some(&json!({"codexErrorInfo": {}}))), None);
        assert_eq!(error_code(Some(&json!({"message": "x"}))), None);
        assert_eq!(error_code(Some(&json!("not a dict"))), None);
        assert_eq!(error_code(None), None);
    }

    #[test]
    fn codes_are_classified_like_python() {
        assert_eq!(classify(Some("usageLimitExceeded")), CodeClass::Account);
        assert_eq!(classify(Some("unauthorized")), CodeClass::Account);
        for code in RETRY_CODES {
            assert_eq!(classify(Some(code)), CodeClass::Retry);
        }
        assert_eq!(classify(None), CodeClass::Retry);
        assert_eq!(classify(Some("contextWindowExceeded")), CodeClass::GiveUp);
        assert_eq!(refusal(Some("unauthorized"))["status"], 401);
        assert_eq!(
            refusal(Some("usageLimitExceeded"))["type"],
            "insufficient_quota"
        );
        assert_eq!(refusal(None)["code"], "codex_usage_limit_exceeded");
    }

    #[test]
    fn usage_is_the_delta_of_the_thread_total() {
        let before = json!({"inputTokens": 1000, "cachedInputTokens": 600, "outputTokens": 50,
                            "reasoningOutputTokens": 20, "totalTokens": 1050});
        let after = json!({"inputTokens": 2500, "cachedInputTokens": 1000.7, "outputTokens": 40,
                           "reasoningOutputTokens": null, "totalTokens": "3000"});
        assert_eq!(
            usage_delta(&before, &after),
            json!({"input_tokens": 1500, "input_tokens_details": {"cached_tokens": 400},
                   "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1950})
        );
        assert_eq!(usage_delta(&json!({}), &json!({}))["total_tokens"], 0);
    }

    #[test]
    fn limit_summaries_skip_empty_windows() {
        assert_eq!(limit_summary(None), Value::Null);
        assert_eq!(limit_summary(Some(&json!({}))), Value::Null);
        let s = limit_summary(Some(&json!({
            "limitId": "codex", "planType": "pro", "rateLimitReachedType": null,
            "primary": {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 1}, "secondary": null,
        })));
        assert_eq!(
            s,
            json!({"limit_id": "codex", "plan": "pro", "reached": null,
                   "primary": {"used_percent": 40, "window_minutes": 10080, "resets_at": 1}, "credits": null})
        );
    }

    #[test]
    fn credits_count_only_when_usable() {
        let reached = |credits: Value| json!({"rateLimitReachedType": "usageLimitExceeded", "credits": credits});
        assert!(has_credits(&reached(
            json!({"hasCredits": true, "unlimited": false, "balance": "62239.00"})
        )));
        assert!(has_credits(&reached(
            json!({"hasCredits": false, "unlimited": true, "balance": null})
        )));
        assert!(!has_credits(&reached(
            json!({"hasCredits": false, "unlimited": false, "balance": null})
        )));
        assert!(!has_credits(&reached(Value::Null)));
        assert!(!has_credits(
            &json!({"rateLimitReachedType": "usageLimitExceeded"})
        ));
        assert_eq!(
            limit_summary(Some(&reached(
                json!({"hasCredits": true, "unlimited": false, "balance": "1.50"})
            )))["credits"],
            json!({"has_credits": true, "unlimited": false, "balance": "1.50"})
        );
    }

    #[test]
    fn conversation_items_match_python() {
        let mut c = CodexConversation::new(&cfg(), "ep", json!({"type": "object"}));
        let images = [json!({"type": "input_image", "image_url": "data:image/png;base64,AAAA"})];
        c.add_packet("{\"a\":1}", &images, true);
        let body = c.body();
        assert_eq!(body["input"][0]["text"], CODEX_FIRST_INTRO);
        assert_eq!(
            body["input"][2],
            json!({"type": "image", "url": "data:image/png;base64,AAAA"})
        );
        assert_eq!(body["prompt_cache_key"], "oeh-review-ep");
        assert_eq!(body["reasoning"]["effort"], "xhigh");
        let call = c.add_output(&[json!({"type": "function_call", "arguments": "{}"})]);
        assert!(call.is_some() && c.pending.is_empty() && c.items.len() == 4);
        c.add_error("{\"error\":\"x\"}");
        assert_eq!(c.body()["input"].as_array().unwrap().len(), 1);
        assert_eq!(c.unrecorded().len(), 5);
        assert!(c.unrecorded().is_empty());
    }

    #[test]
    fn argv_and_env_carry_the_overrides_and_scrub() {
        let client = CodexClient::new(cfg(), "/w", "p", vec!["/acct/a".into()], None).unwrap();
        let argv = client.argv();
        assert_eq!(&argv[..2], ["codex", "app-server"]);
        let overrides: Vec<&str> = argv[2..]
            .iter()
            .skip(1)
            .step_by(2)
            .map(String::as_str)
            .collect();
        assert_eq!(overrides[0], "model=\"gpt-6-luna\"");
        assert_eq!(overrides[1], "model_reasoning_effort=\"xhigh\"");
        assert_eq!(overrides[2], "analytics.enabled=false");
        assert!(overrides[3].starts_with("sqlite_home=\"") && overrides[3].contains("oeh-codex-"));
        assert_eq!(overrides.len(), 5 + FEATURES_OFF.len());
        assert!(argv[2..].iter().step_by(2).all(|a| a == "-c"));
        let env = client.env_plan();
        assert_eq!(
            env.set,
            vec![("CODEX_HOME".to_owned(), "/acct/a".to_owned())]
        );
        assert!(env.remove.contains(&"OPENAI_API_KEY".to_owned()));
        let local = client.local_dir().to_owned();
        drop(client);
        let _ = std::fs::remove_dir_all(local);
    }

    #[test]
    fn python_reprs() {
        assert_eq!(
            py_repr(&json!({"code": -32601, "message": "no x"})),
            "{'code': -32601, 'message': 'no x'}"
        );
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_str_repr("a\"b'c"), "'a\"b\\'c'");
        assert_eq!(py_repr(&json!([null, true, 1.5])), "[None, True, 1.5]");
        assert_eq!(py_g(120.0), "120");
        assert_eq!(py_g(0.5), "0.5");
        let r = ErrorRecord::new("codex_turn_failed", Some("x".into()), true, "m");
        assert_eq!(
            r.py_repr(),
            "{'status': None, 'type': 'codex_turn_failed', 'code': 'x', 'message': 'm', 'attempt': 0, \
             'seconds': 0.0, 'account': 0, 'retry': False, 'fatal': False}"
        );
    }

    #[test]
    fn accounts_file_is_absolute_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(
            &path,
            "# logins\n/workspace/codex/a\n\n/workspace/codex/b  # second\n",
        )
        .unwrap();
        assert_eq!(
            read_accounts(&path).unwrap(),
            vec!["/workspace/codex/a", "/workspace/codex/b"]
        );
        std::fs::write(&path, "relative/home\n").unwrap();
        assert!(read_accounts(&path)
            .unwrap_err()
            .contains("'relative/home' is not an absolute path"));
        std::fs::write(&path, "# none\n").unwrap();
        assert!(read_accounts(&path).is_err());
    }

    #[test]
    fn event_wait_returns_when_set() {
        let e = Arc::new(Event::default());
        assert!(!e.wait(Duration::from_millis(10)));
        let e2 = e.clone();
        let t = std::thread::spawn(move || e2.set());
        assert!(e.wait(Duration::from_secs(5)));
        t.join().unwrap();
    }
}
