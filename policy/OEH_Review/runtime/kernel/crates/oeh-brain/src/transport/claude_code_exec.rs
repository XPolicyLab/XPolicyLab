//! `claude_code_exec`：把 Claude Code CLI 当成一条请求/响应的 wire（`claude -p`，stream-json 进出）。
//!
//! 和 [`codex_exec`](super::codex_exec) 同一个思路：Claude Code 在这里**只是一个带订阅登录（Pro /
//! Max）的模型通道**，不是一个 agent。每次 `complete` 起一个新的 `claude -p` 子进程：
//!
//! - system prompt = 声明的 system + 答案约定 + 工具表，从 `--system-prompt-file` 进去，替换掉 Claude
//!   Code 自己的 agent 提示词。它只随工具表变，订阅下的 prompt 缓存照样命中；
//! - stdin 上一条 stream-json 的 user 消息：`# Conversation` 那段文本，外加相机图（base64 image 块）；
//! - Claude Code 的工具、MCP、技能、设置、CLAUDE.md、钩子在 argv 与环境里全部关掉（见 [`exec_args`]、
//!   [`command`]）。
//!
//! 然后读它的 stream-json 事件：
//!
//! ```text
//! system{subtype=init}             → tools / mcp_servers 必须是空的，否则协议错
//! system{其他 subtype}              → 忽略（重试、思考计数）
//! assistant{content}               → text / thinking 忽略（答案以 result 为准）；其他任何块（tool_use、
//!                                    server_tool_use……）→ 协议错
//! user{content 里有 tool_result}    → 协议错（有工具真的跑了）
//! rate_limit_event{status≠allowed} → ModelEvent::Warning（订阅额度告警，原文进证据）
//! result{is_error=false}           → 答案 + ModelEvent::Usage（含 thinking_tokens）
//! result{is_error=true}            → 有 api_error_status 就是 ModelError::Http（401 = token 失效、
//!                                    429 = 额度用完，瀑布照状态码决定重试不重试），否则 Transport
//! 没有 result                       → Transport（带 stderr 尾巴）
//! ```
//!
//! 答案约定与 `codex_exec` 完全相同（[`decide`]）：`{"tool": <名字>, "arguments": {...}}` 或
//! `{"tool": null}`。在 `result` 之后才失败时事件流同样是 `[Warning…, Usage, Err]`。
//!
//! "key" 是 `claude setup-token` 给的长期 OAuth token（要求 Claude 订阅）：provider 声明的 `env` 指向
//! 存它的变量，这里只把它作为 `CLAUDE_CODE_OAUTH_TOKEN` 交给子进程，不进 argv。子进程的环境是**清空后**
//! 只放回 [`PASSTHROUGH_ENV`]：`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、Bedrock / Vertex 开关这类
//! 变量一个都带不进去，调用不会悄悄改走 API key 计费；`CLAUDE_CONFIG_DIR` 指向这次调用自己的空目录，
//! 用户 `~/.claude` 里的登录、设置、插件与会话记录都不读也不写。
//!
//! argv 与事件形状在 Claude Code 2.1.288 上核过。

use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use super::codex_exec::{bounded, contract, conversation, decide, MAX_STDERR_TAIL, MAX_STDOUT};
use crate::model::{
    BoxFuture, EventStream, Model, ModelError, ModelEvent, ModelRequest, StopReason, ToolSpec,
};
use crate::providers::Runtime;

/// 从父进程原样带给子进程的变量：找得到 `claude`、Node 起得来、走得出代理、认得自签 CA。别的一概不带。
const PASSTHROUGH_ENV: [&str; 16] = [
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "NODE_EXTRA_CA_CERTS",
];

static CALLS: AtomicU64 = AtomicU64::new(0);

pub struct ClaudeCodeExecModel {
    rt: Runtime,
}

impl ClaudeCodeExecModel {
    pub fn new(rt: Runtime) -> Result<Self, String> {
        if rt.program.as_deref().unwrap_or("claude").is_empty() {
            return Err(format!("provider `{}`: empty claude program", rt.provider));
        }
        Ok(Self { rt })
    }
}

impl Model for ClaudeCodeExecModel {
    fn name(&self) -> String {
        self.rt.name()
    }

    fn api_mode(&self) -> String {
        "claude_code_exec".into()
    }

    fn complete(&self, req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
        let rt = self.rt.clone();
        Box::pin(async move {
            let events = run(&rt, &req).await?;
            Ok(Box::pin(futures_util::stream::iter(events)) as EventStream)
        })
    }
}

/// 同 `codex_exec`：要么整条失败，要么一串事件（最后一条可能是 `Err`，用量在它前面）。
type Events = Vec<Result<ModelEvent, ModelError>>;

async fn run(rt: &Runtime, req: &ModelRequest) -> Result<Events, ModelError> {
    let token = rt
        .api_key
        .clone()
        .ok_or_else(|| ModelError::MissingKey(rt.env.clone()))?;
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    let work = std::env::temp_dir().join(format!("oeh-claude-{}-{call}", std::process::id()));
    std::fs::create_dir_all(work.join("config"))
        .map_err(|e| ModelError::Transport(format!("workdir: {e}")))?;
    let result = run_in(rt, req, &token, &work).await;
    let _ = std::fs::remove_dir_all(&work);
    result
}

async fn run_in(
    rt: &Runtime,
    req: &ModelRequest,
    token: &str,
    work: &Path,
) -> Result<Events, ModelError> {
    let (line, system) = render(req)?;
    std::fs::write(work.join("system.md"), system)
        .map_err(|e| ModelError::Transport(format!("system prompt: {e}")))?;
    let program = rt.program.clone().unwrap_or_else(|| "claude".into());
    let mut child = super::spawn_cli(&mut command(rt, token, work))
        .await
        .map_err(|e| ModelError::Transport(format!("spawn {program}: {e}")))?;
    let mut stdin = child.stdin.take().expect("piped");
    let mut stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let io = async {
        if let Err(e) = stdin.write_all(line.as_bytes()).await {
            // The CLI exited before it read its prompt (an option it does not know, say): its exit status and
            // stderr say why, so read those rather than report the broken pipe.
            if e.kind() != std::io::ErrorKind::BrokenPipe {
                return Err(e);
            }
        }
        drop(stdin);
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut limited = (&mut stdout).take(MAX_STDOUT as u64 + 1);
        let (a, b) = tokio::join!(limited.read_to_end(&mut out), stderr.read_to_end(&mut err));
        a?;
        b?;
        let status = child.wait().await?;
        Ok::<_, std::io::Error>((status, out, err))
    };
    let (status, out, err) = if rt.timeout_ms > 0 {
        tokio::time::timeout(Duration::from_millis(rt.timeout_ms), io)
            .await
            .map_err(|_| ModelError::Timeout)?
    } else {
        io.await
    }
    .map_err(|e| ModelError::Transport(format!("claude io: {e}")))?;
    if out.len() > MAX_STDOUT {
        return Err(ModelError::Protocol("claude output exceeds limit".into()));
    }
    let Parsed {
        usage,
        warnings,
        finished,
        result,
    } = parse(&out, &req.tools);
    let error = match result {
        Ok(events) => return Ok(events.into_iter().map(Ok).collect()),
        // 没有 result 事件：CLI 自己没起来（参数不认识、Node 崩了），原因只在 stderr 里。
        Err(e) if !finished => {
            let tail = String::from_utf8_lossy(&err);
            let tail = &tail[tail.len().saturating_sub(MAX_STDERR_TAIL)..];
            ModelError::Transport(format!(
                "claude exited {}: {} ({e})",
                status.code().map_or("by signal".into(), |c| c.to_string()),
                tail.trim()
            ))
        }
        Err(e) => e,
    };
    match usage {
        Some(usage) => Ok(warnings
            .into_iter()
            .map(|w| Ok(ModelEvent::Warning(w)))
            .chain([Ok(usage), Err(error)])
            .collect()),
        None if warnings.is_empty() => Err(error),
        None => Err(with_warnings(error, &warnings)),
    }
}

/// 没有流可交的失败：告警原文拼进错误信息，照样进 `brain/request-error`。
fn with_warnings(error: ModelError, warnings: &[String]) -> ModelError {
    let note = format!(" (claude warnings: {})", warnings.join(" | "));
    match error {
        ModelError::Transport(m) => ModelError::Transport(m + &note),
        ModelError::Protocol(m) => ModelError::Protocol(m + &note),
        other => other,
    }
}

/// `claude` 的完整 argv（不含程序名），逐字钉在测试里。
///
/// - `--safe-mode`：CLAUDE.md、技能、插件、钩子、MCP、自定义 agent 一律不加载（登录照旧）；
/// - `--tools ""`：一个内建工具都没有；`--strict-mcp-config` 且不给 `--mcp-config`：没有 MCP；
/// - `--disable-slash-commands`、`--no-session-persistence`（会话不落盘）、`--permission-mode dontAsk`
///   （万一有东西要权限，一律拒，不等人）；
/// - `--system-prompt-file`：替换默认提示词，不是追加。
fn exec_args(rt: &Runtime, system_file: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--safe-mode",
        "--tools",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--system-prompt-file",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    args.push(system_file.into());
    args.push("--model".into());
    args.push(rt.model.clone().into());
    if let Some(effort) = &rt.reasoning_effort {
        args.push("--effort".into());
        args.push(effort.into());
    }
    args
}

/// 一次调用的子进程：环境清空，只放回 [`PASSTHROUGH_ENV`]、token 与这次调用自己的配置目录。
fn command(rt: &Runtime, token: &str, work: &Path) -> Command {
    let mut cmd = Command::new(rt.program.as_deref().unwrap_or("claude"));
    cmd.args(exec_args(rt, &work.join("system.md"))).env_clear();
    for name in PASSTHROUGH_ENV {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    cmd.env("CLAUDE_CONFIG_DIR", work.join("config"))
        .env("CLAUDE_CODE_OAUTH_TOKEN", token)
        // 遥测、错误上报、自动更新都不发：一次决策只该有那一个请求。
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .current_dir(work)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd
}

/// 请求 → （stdin 上的那一行 stream-json，system prompt 文件的内容）。
fn render(req: &ModelRequest) -> Result<(String, String), ModelError> {
    let (text, images) = conversation(&req.messages)?;
    let mut content = vec![json!({"type": "text", "text": text})];
    for image in images {
        let (media_type, data) = match image.data_url.strip_prefix("data:image/png;base64,") {
            Some(data) => ("image/png", data),
            None => (
                "image/jpeg",
                image.data_url.trim_start_matches("data:image/jpeg;base64,"),
            ),
        };
        content.push(json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }));
    }
    let line = json!({"type": "user", "message": {"role": "user", "content": content}});
    let system = format!("{}\n\n{}", req.system, contract(&req.tools));
    Ok((format!("{line}\n"), system))
}

/// 一次 `claude -p` 的 stream-json 读完之后。
struct Parsed {
    /// 成功的 `result` 才有。
    usage: Option<ModelEvent>,
    warnings: Vec<String>,
    /// 见到了 `result` 事件（CLI 跑完了一个 turn，不管成没成）。
    finished: bool,
    result: Result<Vec<ModelEvent>, ModelError>,
}

/// stream-json → 事件。读完整条流再下结论：前面有不许有的块，后面成功的 `result` 的用量也要收下。
fn parse(out: &[u8], tools: &[ToolSpec]) -> Parsed {
    let mut session = String::from("claude");
    let mut failure: Option<ModelError> = None;
    let mut warnings: Vec<String> = Vec::new();
    let mut usage: Option<ModelEvent> = None;
    let mut answer: Option<Result<String, ModelError>> = None;
    for line in out.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let event: Value = match serde_json::from_slice(line) {
            Ok(event) => event,
            Err(e) => {
                failure.get_or_insert(ModelError::Protocol(format!(
                    "claude event is not JSON: {e}"
                )));
                continue;
            }
        };
        match event["type"].as_str() {
            Some("system") if event["subtype"] == "init" => {
                if let Some(id) = event["session_id"].as_str() {
                    session = id.to_string();
                }
                for key in ["tools", "mcp_servers"] {
                    if event[key].as_array().is_some_and(|a| !a.is_empty()) {
                        failure.get_or_insert_with(|| {
                            ModelError::Protocol(format!(
                                "claude session started with {key} {}; this wire runs with none",
                                event[key]
                            ))
                        });
                    }
                }
            }
            Some("assistant") => {
                for block in event["message"]["content"].as_array().into_iter().flatten() {
                    match block["type"].as_str() {
                        Some("text") | Some("thinking") | Some("redacted_thinking") => {}
                        other => {
                            failure.get_or_insert_with(|| ModelError::Protocol(refused(other)));
                        }
                    }
                }
            }
            Some("user") => {
                let ran = event["message"]["content"]
                    .as_array()
                    .is_some_and(|c| c.iter().any(|b| b["type"] == "tool_result"));
                if ran {
                    failure
                        .get_or_insert_with(|| ModelError::Protocol(refused(Some("tool_result"))));
                }
            }
            Some("rate_limit_event") => {
                let info = &event["rate_limit_info"];
                if let Some(status) = info["status"].as_str().filter(|s| *s != "allowed") {
                    warnings.push(bounded(&format!(
                        "claude rate limit {status} ({})",
                        info["rateLimitType"].as_str().unwrap_or("unknown")
                    )));
                }
            }
            Some("result") => {
                let text = event["result"].as_str().unwrap_or("");
                answer = Some(
                    if event["subtype"] == "success" && event["is_error"] == false {
                        usage = Some(usage_of(&event["usage"]));
                        Ok(text.to_string())
                    } else {
                        Err(
                            match event["api_error_status"]
                                .as_u64()
                                .and_then(|s| u16::try_from(s).ok())
                            {
                                Some(status) => ModelError::Http {
                                    status,
                                    body: text.to_string(),
                                },
                                // "Not logged in"（token 不对）、`error_max_turns`、`error_during_execution`……
                                None => ModelError::Transport(format!(
                                    "claude turn failed ({}): {}",
                                    event["terminal_reason"]
                                        .as_str()
                                        .or(event["subtype"].as_str())
                                        .unwrap_or("unknown"),
                                    bounded(text)
                                )),
                            },
                        )
                    },
                );
            }
            _ => {}
        }
    }
    let finished = answer.is_some();
    let result = outcome(failure, answer, usage.clone(), &warnings, tools, &session);
    Parsed {
        usage,
        warnings,
        finished,
        result,
    }
}

fn outcome(
    failure: Option<ModelError>,
    answer: Option<Result<String, ModelError>>,
    usage: Option<ModelEvent>,
    warnings: &[String],
    tools: &[ToolSpec],
    session: &str,
) -> Result<Vec<ModelEvent>, ModelError> {
    if let Some(e) = failure {
        return Err(e);
    }
    let text =
        answer.ok_or_else(|| ModelError::Transport("claude produced no result".into()))??;
    let usage = usage.expect("a successful result carries its usage");
    let (call, correction) = decide("claude", &text, tools, session)?;
    let mut events = Vec::new();
    if call.is_none() {
        events.push(ModelEvent::TextDelta(text));
    }
    events.extend(warnings.iter().cloned().map(ModelEvent::Warning));
    events.extend(correction.map(ModelEvent::Warning));
    events.push(usage);
    let stop = match call {
        Some(call) => {
            events.push(ModelEvent::ToolCall(call));
            StopReason::ToolUse
        }
        None => StopReason::EndTurn,
    };
    events.push(ModelEvent::Done { stop });
    Ok(events)
}

fn usage_of(u: &Value) -> ModelEvent {
    let n = |v: &Value| v.as_u64().map(|n| n.min(u32::MAX as u64) as u32);
    ModelEvent::Usage {
        input: n(&u["input_tokens"]).unwrap_or(0),
        output: n(&u["output_tokens"]).unwrap_or(0),
        cached: n(&u["cache_read_input_tokens"]).unwrap_or(0),
        reasoning: n(&u["output_tokens_details"]["thinking_tokens"]),
    }
}

/// 一个工具块 → 错误信息（只报类型）。
fn refused(kind: Option<&str>) -> String {
    format!(
        "claude emitted a `{}` block; this wire accepts text and thinking only",
        kind.unwrap_or("untyped")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageInput, Message, ToolCall};
    use crate::providers::ApiMode;
    use std::collections::BTreeMap;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "robot_oeh_manipulate".into(),
            description: "move".into(),
            parameters: json!({"type": "object"}),
            concurrency_safe: false,
        }
    }

    fn runtime(program: &Path, token: Option<&str>) -> Runtime {
        Runtime {
            provider: "claude-code".into(),
            model: "claude-opus-5-5".into(),
            api_mode: ApiMode::ClaudeCodeExec,
            base_url: String::new(),
            env: "CLAUDE_CODE_OAUTH_TOKEN".into(),
            api_key: token.map(str::to_string),
            cache: None,
            headers: BTreeMap::new(),
            timeout_ms: 10_000,
            program: Some(program.display().to_string()),
            reasoning_effort: Some("high".into()),
        }
    }

    /// 一个假的 claude：把 argv、system prompt 文件、环境与 stdin 记下来，再吐出给定的事件。
    fn fake(dir: &Path, events: &str) -> PathBuf {
        let script = dir.join("claude");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\" >> {d}/argv; done\n\
                 while [ $# -gt 0 ]; do [ \"$1\" = --system-prompt-file ] && cp \"$2\" {d}/system; shift; done\n\
                 env > {d}/env\ncat > {d}/stdin\ncat <<'EOF'\n{events}\nEOF\n",
                d = dir.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    fn request() -> ModelRequest {
        ModelRequest {
            system: "You are the slow brain.".into(),
            messages: vec![Message::UserMultimodal {
                content: "escalation: stall".into(),
                images: vec![ImageInput {
                    camera: "cam_head".into(),
                    data_url: "data:image/png;base64,iVBORw0KGgo=".into(),
                }],
            }],
            tools: vec![tool()],
            max_tokens: 2048,
            temperature: None,
        }
    }

    async fn items(
        model: &ClaudeCodeExecModel,
    ) -> Result<Vec<Result<ModelEvent, ModelError>>, ModelError> {
        use futures_util::StreamExt;
        Ok(model.complete(request()).await?.collect::<Vec<_>>().await)
    }

    async fn collect(model: &ClaudeCodeExecModel) -> Result<Vec<ModelEvent>, ModelError> {
        items(model).await?.into_iter().collect()
    }

    fn model(dir: &Path, events: &str) -> ClaudeCodeExecModel {
        ClaudeCodeExecModel::new(runtime(&fake(dir, events), Some("sk-ant-oat01-test"))).unwrap()
    }

    const INIT: &str = r#"{"type":"system","subtype":"init","session_id":"s1","tools":[],"mcp_servers":[],"model":"claude-opus-5-5","apiKeySource":"none"}"#;
    const ANSWER: &str = r#"{\"tool\":\"robot_oeh_manipulate\",\"arguments\":{\"mode\":\"vla\"}}"#;

    /// Claude Code 2.1.288 的事件形状（只留这条 wire 读的字段）。
    fn ok(answer: &str) -> String {
        format!(
            r#"{INIT}
{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"thinking","thinking":"","signature":"x"}}]}}}}
{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"{answer}"}}]}}}}
{{"type":"rate_limit_event","rate_limit_info":{{"status":"allowed","rateLimitType":"five_hour"}}}}
{{"type":"result","subtype":"success","is_error":false,"result":"{answer}","session_id":"s1","usage":{{"input_tokens":1200,"cache_read_input_tokens":800,"output_tokens":90,"output_tokens_details":{{"thinking_tokens":60}}}}}}"#
        )
    }

    const USAGE: ModelEvent = ModelEvent::Usage {
        input: 1200,
        output: 90,
        cached: 800,
        reasoning: Some(60),
    };

    #[tokio::test]
    async fn an_answer_becomes_one_tool_call_with_subscription_usage() {
        let dir = tempfile::tempdir().unwrap();
        let events = collect(&model(dir.path(), &ok(ANSWER))).await.unwrap();
        assert_eq!(
            events,
            vec![
                USAGE,
                ModelEvent::ToolCall(ToolCall {
                    id: "s1:0".into(),
                    name: "robot_oeh_manipulate".into(),
                    args: json!({"mode": "vla"})
                }),
                ModelEvent::Done {
                    stop: StopReason::ToolUse
                }
            ]
        );
        // The whole argv, in order; `--tools` takes an empty value (no built-in tool at all).
        let argv = std::fs::read_to_string(dir.path().join("argv")).unwrap();
        let argv: Vec<&str> = argv.lines().collect();
        let system_file = argv[argv
            .iter()
            .position(|a| *a == "--system-prompt-file")
            .unwrap()
            + 1];
        assert_eq!(
            argv,
            [
                "-p",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                "--verbose",
                "--safe-mode",
                "--tools",
                "",
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--permission-mode",
                "dontAsk",
                "--system-prompt-file",
                system_file,
                "--model",
                "claude-opus-5-5",
                "--effort",
                "high",
            ]
        );
        // System prompt: the declared system, then the contract and the tool table.
        let system = std::fs::read_to_string(dir.path().join("system")).unwrap();
        assert!(system.starts_with("You are the slow brain.\n\n# Output contract\n"));
        assert!(system.contains("## robot_oeh_manipulate"));
        // Stdin: one stream-json user message, the conversation then the camera image.
        let stdin = std::fs::read_to_string(dir.path().join("stdin")).unwrap();
        assert_eq!(stdin.lines().count(), 1);
        let line: Value = serde_json::from_str(&stdin).unwrap();
        let content = &line["message"]["content"];
        assert!(content[0]["text"]
            .as_str()
            .unwrap()
            .contains("escalation: stall"));
        assert_eq!(
            content[1]["source"],
            json!({"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="})
        );
    }

    /// Only the pass-through variables, the token and the call's own config directory reach the child: whatever
    /// the parent holds (an `ANTHROPIC_API_KEY`, a Bedrock switch, a Claude Code session) never does.
    #[tokio::test]
    async fn the_child_environment_holds_the_token_and_nothing_else_it_could_bill_through() {
        let dir = tempfile::tempdir().unwrap();
        collect(&model(dir.path(), &ok(ANSWER))).await.unwrap();
        let env = std::fs::read_to_string(dir.path().join("env")).unwrap();
        let mut ours = Vec::new();
        for line in env.lines() {
            let (name, value) = line.split_once('=').unwrap_or((line, ""));
            match name {
                "CLAUDE_CODE_OAUTH_TOKEN" => assert_eq!(value, "sk-ant-oat01-test"),
                "CLAUDE_CONFIG_DIR" => assert!(value.contains("oeh-claude-"), "{value}"),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" => assert_eq!(value, "1"),
                // set by the shell itself
                "PWD" | "OLDPWD" | "SHLVL" | "_" => continue,
                other => {
                    assert!(
                        PASSTHROUGH_ENV.contains(&other),
                        "{other} leaked into the child"
                    );
                    continue;
                }
            }
            ours.push(name);
        }
        ours.sort();
        assert_eq!(
            ours,
            [
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CLAUDE_CONFIG_DIR"
            ]
        );
    }

    #[tokio::test]
    async fn tool_null_is_an_explicit_no_action_that_ends_the_turn() {
        let dir = tempfile::tempdir().unwrap();
        let events = collect(&model(dir.path(), &ok(r#"```json\n{\"tool\": null}\n```"#)))
            .await
            .unwrap();
        assert!(matches!(&events[0], ModelEvent::TextDelta(t) if t.contains("null")));
        assert_eq!(events[1], USAGE);
        assert_eq!(
            events[2],
            ModelEvent::Done {
                stop: StopReason::EndTurn
            }
        );
    }

    /// Anything that is not the answer contract, or any sign of a tool, fails the call; usage is still reported.
    #[tokio::test]
    async fn a_bad_answer_or_a_tool_block_is_a_protocol_error_that_keeps_its_usage() {
        let tool_use = ok(ANSWER).replace(
            r#"[{"type":"thinking","thinking":"","signature":"x"}]"#,
            r#"[{"type":"tool_use","id":"t","name":"Bash","input":{}}]"#,
        );
        let tool_result = ok(ANSWER).replace(
            r#"{"type":"rate_limit_event""#,
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\"}]}}\n{\"type\":\"rate_limit_event\"",
        );
        for (events, expected) in [
            (ok("I would grasp the cup."), "neither a declared tool call"),
            (
                ok(r#"{\"tool\":\"robot_other\",\"arguments\":{}}"#),
                "neither a declared tool call",
            ),
            (tool_use, "`tool_use` block"),
            (tool_result, "`tool_result` block"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut items = items(&model(dir.path(), &events))
                .await
                .unwrap()
                .into_iter();
            assert_eq!(items.next().unwrap().unwrap(), USAGE);
            match items.next().unwrap() {
                Err(ModelError::Protocol(m)) => assert!(m.contains(expected), "{m}"),
                other => panic!("{expected}: {other:?}"),
            }
            assert!(items.next().is_none());
        }
    }

    #[tokio::test]
    async fn a_session_that_starts_with_tools_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let events = ok(ANSWER).replace(r#""tools":[]"#, r#""tools":["Bash"]"#);
        match collect(&model(dir.path(), &events)).await {
            Err(ModelError::Protocol(m)) => assert!(m.contains("started with tools"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    /// A refused account comes back with its HTTP status (401 is not retried, 429 is); a missing login and a CLI
    /// that never produced a result are transport errors that say why; no token means no child at all.
    #[tokio::test]
    async fn account_failures_keep_their_status_and_reason() {
        let failed = |status: &str, text: &str| {
            format!(
                "{INIT}\n{{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":true,\"api_error_status\":{status},\"result\":\"{text}\",\"usage\":{{\"input_tokens\":0,\"output_tokens\":0}}}}"
            )
        };
        let dir = tempfile::tempdir().unwrap();
        let limited = failed("429", "Claude usage limit reached");
        match collect(&model(dir.path(), &limited)).await {
            Err(ModelError::Http { status: 429, body }) => assert!(body.contains("usage limit")),
            other => panic!("{other:?}"),
        }
        let refused = failed(
            "401",
            "Failed to authenticate. API Error: 401 OAuth access token is invalid.",
        );
        assert!(matches!(
            collect(&model(dir.path(), &refused)).await,
            Err(ModelError::Http { status: 401, .. })
        ));
        let logged_out = failed("null", "Not logged in · Please run /login");
        match collect(&model(dir.path(), &logged_out)).await {
            Err(ModelError::Transport(m)) => assert!(m.contains("Not logged in"), "{m}"),
            other => panic!("{other:?}"),
        }
        let script = dir.path().join("broken");
        std::fs::write(
            &script,
            "#!/bin/sh\necho \"error: unknown option '--safe-mode'\" >&2\nexit 1\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let broken = ClaudeCodeExecModel::new(runtime(&script, Some("t"))).unwrap();
        match collect(&broken).await {
            Err(ModelError::Transport(m)) => {
                assert!(m.contains("exited 1") && m.contains("--safe-mode"), "{m}")
            }
            other => panic!("{other:?}"),
        }
        let no_token = ClaudeCodeExecModel::new(runtime(&fake(dir.path(), INIT), None)).unwrap();
        assert!(
            matches!(collect(&no_token).await, Err(ModelError::MissingKey(ref v)) if v == "CLAUDE_CODE_OAUTH_TOKEN")
        );
    }

    #[tokio::test]
    async fn a_rate_limit_warning_goes_into_the_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let events = ok(ANSWER).replace(r#""status":"allowed""#, r#""status":"allowed_warning""#);
        let events = collect(&model(dir.path(), &events)).await.unwrap();
        assert_eq!(
            events[0],
            ModelEvent::Warning("claude rate limit allowed_warning (five_hour)".into())
        );
        assert_eq!(events[1], USAGE);
    }
}
