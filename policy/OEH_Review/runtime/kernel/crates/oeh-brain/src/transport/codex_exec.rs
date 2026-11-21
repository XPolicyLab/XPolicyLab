//! `codex_exec`：把 Codex CLI 当成一条请求/响应的 wire（`codex exec --json`）。
//!
//! Codex 在这里**只是一个带账号登录的模型通道**，不是一个 agent：每次 `complete` 起一个
//! 新的 `codex exec` 子进程，把 system + transcript + 工具说明拼成一段 prompt 从 stdin
//! 喂进去，相机图用 `--image` 挂上，Codex 自己的工具在 argv 里全部关掉（见 [`exec_args`]），
//! 然后读它的 JSONL 事件流：
//!
//! ```text
//! item.completed{item.type=agent_message, text}  → 最后一条就是答案
//! item.*{item.type=reasoning}                    → 忽略（推理摘要）
//! item.completed{item.type=error}                → ModelEvent::Warning（不致命的告警：模型元数据缺失、
//!                                                  改路由、配置告警；原文进证据，调用照常）
//! item.*{其他任何 item.type}                      → ModelError::Protocol（命令、改文件、MCP、搜索、
//!                                                  协作、计划……任何工具 item 都不许出现在这条 wire 上）
//! turn.completed{usage}                          → ModelEvent::Usage（含 reasoning_output_tokens）
//! turn.failed / 没有 turn.completed               → 这次调用失败
//! 顶层 error                                     → 不致命：0.155.1 把可重试的流错误（"Reconnecting...
//!                                                  n/5"，will_retry）也打成这一行，turn 照常继续
//! ```
//!
//! 答案约定是一个 JSON 对象：`{"tool": <名字>, "arguments": {...}}` 翻成一条
//! [`ModelEvent::ToolCall`]，交给 turn 循环照常走准入（参数是否合法由内核里的能力 schema 与准入
//! 决定，这里不替它判断）；`{"tool": null}` 是明确的"不干预"，turn 以 `end_turn` 收尾；别的答案
//! （不是 JSON、没声明过的工具名、参数不是对象）是协议错。
//!
//! 调用在 `turn.completed` 之后才失败时（多了一个不许有的 item、答案不合约），事件流是
//! `[Warning…, Usage, Err]`：这次调用的用量和告警照样记账（`native::collect` 先收下再报错）；
//! 没有 `turn.completed` 就失败时告警原文拼进错误信息。
//!
//! "key" 是登录目录：provider 声明的 `env` 指向的环境变量存的是 `CODEX_HOME` 路径（里面
//! 有 device 登录留下的 `auth.json`）。这个模块只把路径交给子进程，自己从不读那个目录。
//! 子进程的环境里去掉 [`API_KEY_ENV`]：`OPENAI_API_KEY` 这类变量在场时 Codex 可能改走按量
//! 计费的 API key，而这条 wire 说好了只用订阅登录。
//!
//! 答案约定（[`contract`] / [`conversation`] / [`decide`]）和 `claude_code_exec` 共用。

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::model::{
    BoxFuture, EventStream, ImageInput, Message, Model, ModelError, ModelEvent, ModelRequest,
    StopReason, ToolCall, ToolSpec,
};
use crate::providers::Runtime;

/// stdout 的上限：一个决策的 JSONL 不该有这么大；超了就是协议错，不无限缓冲。
pub(super) const MAX_STDOUT: usize = 8 * 1024 * 1024;
/// stderr 只留尾巴进错误信息。
pub(super) const MAX_STDERR_TAIL: usize = 4 * 1024;

/// 不交给 `codex` 子进程的变量（`oeh_policy.codex_teacher.SCRUBBED_ENV` 同一张表）：有它们
/// 在，调用可能从订阅登录换成 API key 计费。
const API_KEY_ENV: [&str; 5] = [
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
];

static CALLS: AtomicU64 = AtomicU64::new(0);

pub struct CodexExecModel {
    rt: Runtime,
}

impl CodexExecModel {
    pub fn new(rt: Runtime) -> Result<Self, String> {
        if rt.program.as_deref().unwrap_or("codex").is_empty() {
            return Err(format!("provider `{}`: empty codex program", rt.provider));
        }
        Ok(Self { rt })
    }
}

impl Model for CodexExecModel {
    fn name(&self) -> String {
        self.rt.name()
    }

    fn api_mode(&self) -> String {
        "codex_exec".into()
    }

    fn complete(&self, req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
        let rt = self.rt.clone();
        Box::pin(async move {
            let events = run(&rt, &req).await?;
            Ok(Box::pin(futures_util::stream::iter(events)) as EventStream)
        })
    }
}

/// 一次调用的事件：要么整条失败（`Err`，还没有任何用量），要么一串事件——最后一条可能是
/// `Err`（`turn.completed` 之后才失败，用量在它前面）。
type Events = Vec<Result<ModelEvent, ModelError>>;

async fn run(rt: &Runtime, req: &ModelRequest) -> Result<Events, ModelError> {
    let home = rt
        .api_key
        .clone()
        .ok_or_else(|| ModelError::MissingKey(rt.env.clone()))?;
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    let work = std::env::temp_dir().join(format!("oeh-codex-{}-{call}", std::process::id()));
    std::fs::create_dir_all(&work).map_err(|e| ModelError::Transport(format!("workdir: {e}")))?;
    let result = run_in(rt, req, Path::new(&home), &work).await;
    let _ = std::fs::remove_dir_all(&work);
    result
}

async fn run_in(
    rt: &Runtime,
    req: &ModelRequest,
    home: &Path,
    work: &Path,
) -> Result<Events, ModelError> {
    let (prompt, images) = render(req)?;
    let mut image_paths: Vec<PathBuf> = Vec::new();
    for (i, (camera, bytes, ext)) in images.into_iter().enumerate() {
        let path = work.join(format!("{i:02}-{camera}.{ext}"));
        std::fs::write(&path, bytes).map_err(|e| ModelError::Transport(format!("image: {e}")))?;
        image_paths.push(path);
    }
    let program = rt.program.clone().unwrap_or_else(|| "codex".into());
    let mut child = super::spawn_cli(&mut command(rt, home, work, &image_paths))
        .await
        .map_err(|e| ModelError::Transport(format!("spawn {program}: {e}")))?;
    let mut stdin = child.stdin.take().expect("piped");
    let mut stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let io = async {
        stdin.write_all(prompt.as_bytes()).await?;
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
    .map_err(|e| ModelError::Transport(format!("codex io: {e}")))?;
    if out.len() > MAX_STDOUT {
        return Err(ModelError::Protocol("codex output exceeds limit".into()));
    }
    let Parsed {
        usage,
        warnings,
        result,
    } = parse(&out, &req.tools);
    let error = match result {
        Ok(events) => return Ok(events.into_iter().map(Ok).collect()),
        Err(e) if !status.success() => {
            let tail = String::from_utf8_lossy(&err);
            let tail = &tail[tail.len().saturating_sub(MAX_STDERR_TAIL)..];
            ModelError::Transport(format!(
                "codex exited {}: {} ({e})",
                status.code().map_or("by signal".into(), |c| c.to_string()),
                tail.trim()
            ))
        }
        Err(e) => e,
    };
    match usage {
        // 失败在 `turn.completed` 之后：这次调用的告警和花了多少照样报上去。
        Some(usage) => Ok(warnings
            .into_iter()
            .map(|w| Ok(ModelEvent::Warning(w)))
            .chain([Ok(usage), Err(error)])
            .collect()),
        None if warnings.is_empty() => Err(error),
        None => Err(with_warnings(error, &warnings)),
    }
}

/// 一次调用的子进程：登录目录进 `CODEX_HOME`，[`API_KEY_ENV`] 不进。
fn command(rt: &Runtime, home: &Path, work: &Path, images: &[PathBuf]) -> Command {
    let mut cmd = Command::new(rt.program.as_deref().unwrap_or("codex"));
    cmd.args(exec_args(rt, work, images))
        .env("CODEX_HOME", home)
        .current_dir(work)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for name in API_KEY_ENV {
        cmd.env_remove(name);
    }
    cmd
}

/// 没有流可交的失败：告警原文拼进错误信息，照样进 `brain/request-error`。
fn with_warnings(error: ModelError, warnings: &[String]) -> ModelError {
    let note = format!(" (codex warnings: {})", warnings.join(" | "));
    match error {
        ModelError::Transport(m) => ModelError::Transport(m + &note),
        ModelError::Protocol(m) => ModelError::Protocol(m + &note),
        other => other,
    }
}

/// `codex exec` 的完整 argv（不含程序名），逐字钉在测试里。
///
/// `--image` 在 0.155.1 里是 `num_args = 1..`（`codex-rs/utils/cli/src/shared_options.rs`），
/// 会把后面所有不像 flag 的词都吞成图片；所以 stdin 哨兵 `-` 前面必须有 `--`，否则 clap
/// 把 `-` 当成第四张图，prompt 为空，Codex 每次都多报一条"读不了 `-` 这张图"。
///
/// 计划 §5："no shell tools"。下面每一项都在 0.155.1（tag `rust-v0.155.1`）的源码里核过：
/// - `--ignore-user-config` / `--ignore-rules`：`codex-rs/exec/src/cli.rs:39-45`。不读
///   `$CODEX_HOME/config.toml`（那里的 MCP server、插件、模型设置都不进来；登录照旧），也不读
///   execpolicy `.rules`。
/// - `features.shell_tool` / `unified_exec` / `view_image` / `apps` / `plugins` /
///   `image_generation`：`codex-rs/features/src/lib.rs` 的 `FEATURES` 表，都是 Stable、默认开。
///   `shell_tool` 关掉时 `codex-rs/core/src/tools/spec_plan.rs::add_shell_tools` 直接返回（只关
///   `unified_exec` 还会留一个一次性 shell）；`view_image` 是读本地图片文件的工具；`apps` 是
///   ChatGPT 登录下的连接器工具（`Features::apps_enabled_for_auth`）。
/// - `agents.enabled=false`：`codex-rs/config/src/config_toml.rs` `AgentsToml.enabled`；
///   `codex-rs/core/src/config/mod.rs::multi_agent_version_override` 因此给 `Disabled`，不管模型
///   目录怎么说都没有 spawn_agent 这类协作工具。
/// - `web_search="disabled"`：`config_toml.rs` 的 `web_search`，取值见
///   `codex-rs/protocol/src/config_types.rs` `WebSearchMode`（默认是 `cached`，即开着）。
///
/// `apply_patch` 在 0.155.1 里没有配置开关（跟着模型目录走）：只读沙箱拒绝它写盘，而它的
/// `file_change` item 在 [`parse`] 里就是协议错。
fn exec_args(rt: &Runtime, work: &Path, images: &[PathBuf]) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = [
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "-c",
        "approval_policy=\"never\"",
        "-c",
        "features.shell_tool=false",
        "-c",
        "features.unified_exec=false",
        "-c",
        "features.view_image=false",
        "-c",
        "features.apps=false",
        "-c",
        "features.plugins=false",
        "-c",
        "features.image_generation=false",
        "-c",
        "agents.enabled=false",
        "-c",
        "web_search=\"disabled\"",
        "--cd",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    args.push(work.as_os_str().into());
    args.push("--model".into());
    args.push(rt.model.clone().into());
    if let Some(effort) = &rt.reasoning_effort {
        args.push("-c".into());
        args.push(format!("model_reasoning_effort=\"{effort}\"").into());
    }
    for path in images {
        args.push("--image".into());
        args.push(path.as_os_str().into());
    }
    args.push("--".into());
    args.push("-".into());
    args
}

/// 一张随 prompt 附上的图片：文件名、字节、扩展名。
type RenderedImage = (String, Vec<u8>, &'static str);

/// transcript → 一段 prompt + 图片。图片从 data URL 里解出来（传输前 [`ImageInput::validate`]
/// 过的同一套限制）。
fn render(req: &ModelRequest) -> Result<(String, Vec<RenderedImage>), ModelError> {
    let (conversation, attached) = conversation(&req.messages)?;
    let mut images = Vec::new();
    for image in attached {
        let (ext, data) = if let Some(d) = image.data_url.strip_prefix("data:image/png;base64,") {
            ("png", d)
        } else {
            (
                "jpg",
                image.data_url.trim_start_matches("data:image/jpeg;base64,"),
            )
        };
        let bytes = decode_base64(data)
            .ok_or_else(|| ModelError::Protocol("invalid camera image base64".into()))?;
        images.push((sanitize(&image.camera), bytes, ext));
    }
    let text = format!(
        "# Instructions\n{}\n\n{}{conversation}",
        req.system,
        contract(&req.tools)
    );
    Ok((text, images))
}

/// 答案约定 + 工具表：只随工具表变，不随 turn 变。
pub(super) fn contract(tools: &[ToolSpec]) -> String {
    let mut text = String::from("# Output contract\nDo not run commands or edit files. Reply with exactly one JSON object and nothing else:\n");
    text.push_str("{\"tool\": \"<one tool name below>\", \"arguments\": { ... }}\n");
    text.push_str("The arguments must satisfy that tool's JSON schema. To take no action, reply with {\"tool\": null}.\n\n# Tools\n");
    for tool in tools {
        text.push_str(&format!(
            "## {}\n{}\nSchema: {}\n\n",
            tool.name, tool.description, tool.parameters
        ));
    }
    text
}

/// transcript → `# Conversation` 一段文本，外加按出现顺序附上的相机图（都 [`ImageInput::validate`] 过）。
pub(super) fn conversation(messages: &[Message]) -> Result<(String, Vec<&ImageInput>), ModelError> {
    let mut images = Vec::new();
    let mut text = String::from("# Conversation\n");
    for message in messages {
        match message {
            Message::System { content } => text.push_str(&format!("[system]\n{content}\n\n")),
            Message::User { content } => text.push_str(&format!("[user]\n{content}\n\n")),
            Message::UserMultimodal {
                content,
                images: attached,
            } => {
                for image in attached {
                    image.validate()?;
                    images.push(image);
                }
                let cameras: Vec<&str> = attached.iter().map(|i| i.camera.as_str()).collect();
                text.push_str(&format!(
                    "[user; attached cameras in order: {cameras:?}]\n{content}\n\n"
                ));
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                text.push_str(&format!("[assistant]\n{content}\n"));
                for call in tool_calls {
                    text.push_str(&format!("(called {} with {})\n", call.name, call.args));
                }
                text.push('\n');
            }
            Message::Tool { call_id, content } => {
                text.push_str(&format!("[tool result {call_id}]\n{content}\n\n"))
            }
        }
    }
    Ok((text, images))
}

fn sanitize(camera: &str) -> String {
    camera
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// 一次 `codex exec` 的 JSONL 读完之后：用量（有 `turn.completed` 才有）与结局。
struct Parsed {
    usage: Option<ModelEvent>,
    /// `error` item 的原文（截到 [`MAX_ITEM_NOTE`]），按出现顺序。
    warnings: Vec<String>,
    result: Result<Vec<ModelEvent>, ModelError>,
}

/// 告警 item 的原文最多带这么多字节进证据（世界日志的 `brain/model_warning` / 错误信息）。
const MAX_ITEM_NOTE: usize = 240;

/// JSONL → 事件。最后一条 agent_message 是答案；用量来自 `turn.completed`。
///
/// 读完整条流再下结论：一个不许有的 item 出现在前面，后面的 `turn.completed` 用量也要收下。
fn parse(out: &[u8], tools: &[ToolSpec]) -> Parsed {
    let mut answer: Option<String> = None;
    let mut usage: Option<ModelEvent> = None;
    let mut thread = String::from("codex");
    let mut failure: Option<ModelError> = None;
    let mut last_error: Option<String> = None;
    let mut warnings: Vec<String> = Vec::new();
    for line in out.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let event: Value = match serde_json::from_slice(line) {
            Ok(event) => event,
            Err(e) => {
                failure.get_or_insert(ModelError::Protocol(format!(
                    "codex event is not JSON: {e}"
                )));
                continue;
            }
        };
        match event["type"].as_str() {
            Some("thread.started") => {
                if let Some(id) = event["thread_id"].as_str() {
                    thread = id.to_string();
                }
            }
            Some("item.started") | Some("item.updated") | Some("item.completed") => {
                let item = &event["item"];
                match item["type"].as_str() {
                    Some("agent_message") => {
                        if event["type"] == "item.completed" {
                            answer = item["text"].as_str().map(str::to_string);
                        }
                    }
                    Some("reasoning") => {}
                    // Codex surfaces its warnings as `error` items: recorded, never fatal.
                    Some("error") => {
                        if event["type"] == "item.completed" {
                            warnings.push(bounded(item["message"].as_str().unwrap_or("")));
                        }
                    }
                    other => {
                        failure.get_or_insert_with(|| ModelError::Protocol(refused_item(other)));
                    }
                }
            }
            Some("turn.completed") => {
                let u = &event["usage"];
                let n = |k: &str| u[k].as_u64().unwrap_or(0).min(u32::MAX as u64) as u32;
                usage = Some(ModelEvent::Usage {
                    input: n("input_tokens"),
                    output: n("output_tokens"),
                    cached: n("cached_input_tokens"),
                    reasoning: Some(n("reasoning_output_tokens")),
                });
            }
            Some("turn.failed") => {
                failure.get_or_insert_with(|| {
                    ModelError::Transport(format!(
                        "codex turn failed: {}",
                        event["error"]["message"].as_str().unwrap_or("unknown")
                    ))
                });
            }
            // 不致命：可重试的流错误（"Reconnecting... n/5"）也是这一行，turn 还在跑。
            // 真失败会有 `turn.failed`，或者根本没有 `turn.completed`。
            Some("error") => last_error = event["message"].as_str().map(str::to_string),
            _ => {}
        }
    }
    let result = (|| {
        if let Some(e) = failure {
            return Err(e);
        }
        let Some(usage) = usage.clone() else {
            return Err(ModelError::Transport(match &last_error {
                Some(m) => format!("codex turn did not complete; last error: {m}"),
                None => "codex turn did not complete".into(),
            }));
        };
        let Some(text) = answer else {
            return Err(ModelError::Protocol(
                "codex produced no agent message".into(),
            ));
        };
        let mut events = Vec::new();
        let (call, correction) = decide("codex", &text, tools, &thread)?;
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
    })();
    Parsed {
        usage,
        warnings,
        result,
    }
}

/// 一个工具 item → 错误信息（只报类型）。
fn refused_item(kind: Option<&str>) -> String {
    format!(
        "codex emitted a `{}` item; this wire accepts reasoning, agent_message and warning items only",
        kind.unwrap_or("untyped")
    )
}

/// 告警原文截到 [`MAX_ITEM_NOTE`] 字节，落在字符边界上。
pub(super) fn bounded(note: &str) -> String {
    let mut end = note.len().min(MAX_ITEM_NOTE);
    while !note.is_char_boundary(end) {
        end -= 1;
    }
    note[..end].to_string()
}

/// How many edits (Levenshtein) an answer's tool name may be from the one declared tool it then means. Dev smoke
/// 2026-09-27: gpt-6-sol wrote `robot_oe_manipulate` for `robot_oeh_manipulate` in 3 of 17 memory-hybrid calls.
const TOOL_NAME_EDITS: usize = 2;

fn edits(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (diagonal + usize::from(ca != *cb))
                .min(row[j] + 1)
                .min(above + 1);
            diagonal = above;
        }
    }
    row[b.len()]
}

/// The declared tool an answer names: the exact name, or else the only declared tool within [`TOOL_NAME_EDITS`]
/// edits (of names long enough that so few edits cannot make another word); `true` when the name was corrected.
fn declared<'a>(name: &str, tools: &'a [ToolSpec]) -> Option<(&'a ToolSpec, bool)> {
    if let Some(tool) = tools.iter().find(|t| t.name == name) {
        return Some((tool, false));
    }
    let mut near = tools.iter().filter(|t| {
        t.name.chars().count() > 4 * TOOL_NAME_EDITS && edits(&t.name, name) <= TOOL_NAME_EDITS
    });
    match (near.next(), near.next()) {
        (Some(tool), None) => Some((tool, true)),
        _ => None,
    }
}

/// 答案 → 一条工具调用（`Some`）、明确的不干预 `{"tool": null}`（`None`），或协议错。
/// 只接受声明过的工具名（或只差 [`TOOL_NAME_EDITS`] 处编辑、且只可能是它的那一个；这时附一条告警进证据）
/// 与对象形的参数。`who` 是错误信息里的 wire 名（`codex` / `claude`），`thread` 给工具调用的 id。
pub(super) fn decide(
    who: &str,
    text: &str,
    tools: &[ToolSpec],
    thread: &str,
) -> Result<(Option<ToolCall>, Option<String>), ModelError> {
    let trimmed = text.trim();
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(trimmed)
        .trim();
    // The refusal keeps why and how the answer began (the model's own text, never evaluator data), so a protocol
    // failure can be diagnosed from the evidence (analysis finding F7: the text used to be discarded).
    let refused = |why: &str| {
        let excerpt: String = body.chars().take(ANSWER_EXCERPT_CHARS).collect();
        ModelError::Protocol(format!(
            "{who} answer is neither a declared tool call nor {{\"tool\": null}} ({why}); answer begins {excerpt:?}"
        ))
    };
    let value: Value =
        serde_json::from_str(body).map_err(|e| refused(&format!("not one JSON value: {e}")))?;
    match value.get("tool") {
        Some(Value::Null) => Ok((None, None)),
        Some(Value::String(name)) => {
            let (tool, corrected) =
                declared(name, tools).ok_or_else(|| refused("no declared tool name"))?;
            let args = value
                .get("arguments")
                .filter(|a| a.is_object())
                .ok_or_else(|| refused("arguments is not an object"))?;
            let correction = corrected.then(|| {
                format!("tool name {name:?} read as the declared {:?} ({TOOL_NAME_EDITS} edits at most)", tool.name)
            });
            Ok((
                Some(ToolCall {
                    id: format!("{thread}:0"),
                    name: tool.name.clone(),
                    args: args.clone(),
                }),
                correction,
            ))
        }
        _ => Err(refused("no declared tool name")),
    }
}

/// How much of a refused answer the protocol error keeps.
const ANSWER_EXCERPT_CHARS: usize = 400;

fn decode_base64(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    }
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 {
            return None;
        }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            let v = if i >= 4 - pad { 0 } else { val(c)? };
            n = (n << 6) | v;
        }
        let triple = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&triple[..3 - pad]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImageInput, ToolSpec};
    use crate::providers::ApiMode;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::os::unix::fs::PermissionsExt;

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "robot_oeh_manipulate".into(),
            description: "move".into(),
            parameters: json!({"type": "object"}),
            concurrency_safe: false,
        }
    }

    fn runtime(program: &Path, home: Option<&Path>) -> Runtime {
        Runtime {
            provider: "codex".into(),
            model: "gpt-6-sol".into(),
            api_mode: ApiMode::CodexExec,
            base_url: String::new(),
            env: "OEH_CODEX_HOME".into(),
            api_key: home.map(|h| h.display().to_string()),
            cache: None,
            headers: BTreeMap::new(),
            timeout_ms: 10_000,
            program: Some(program.display().to_string()),
            reasoning_effort: Some("high".into()),
        }
    }

    /// 一个假的 codex：把 argv、CODEX_HOME 与 stdin 记下来，再吐出 `$FAKE_EVENTS` 的内容。
    fn fake(dir: &Path, events: &str) -> PathBuf {
        let script = dir.join("codex");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > {d}/argv\nprintf '%s' \"$CODEX_HOME\" > {d}/home\ncat > {d}/stdin\ncat <<'EOF'\n{events}\nEOF\n",
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

    async fn collect(model: &CodexExecModel) -> Result<Vec<ModelEvent>, ModelError> {
        use futures_util::StreamExt;
        let stream = model.complete(request()).await?;
        stream.collect::<Vec<_>>().await.into_iter().collect()
    }

    /// The stream item by item, so a usage event before an error is visible.
    async fn items(
        model: &CodexExecModel,
    ) -> Result<Vec<Result<ModelEvent, ModelError>>, ModelError> {
        use futures_util::StreamExt;
        Ok(model.complete(request()).await?.collect::<Vec<_>>().await)
    }

    /// The flags every call carries before `--cd` (see [`exec_args`]).
    const FIXED: [&str; 24] = [
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "-c",
        "approval_policy=\"never\"",
        "-c",
        "features.shell_tool=false",
        "-c",
        "features.unified_exec=false",
        "-c",
        "features.view_image=false",
        "-c",
        "features.apps=false",
        "-c",
        "features.plugins=false",
        "-c",
        "features.image_generation=false",
        "-c",
        "agents.enabled=false",
    ];

    /// A call that fails after `turn.completed` still reports its usage, then the error.
    fn usage_then(items: Vec<Result<ModelEvent, ModelError>>) -> ModelError {
        let mut items = items.into_iter();
        assert_eq!(
            items.next().unwrap().unwrap(),
            ModelEvent::Usage {
                input: 1200,
                output: 90,
                cached: 800,
                reasoning: Some(60)
            }
        );
        let error = items.next().unwrap().unwrap_err();
        assert!(items.next().is_none());
        error
    }

    const OK: &str = r#"{"type":"thread.started","thread_id":"th1"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"i0","type":"reasoning","text":"thinking"}}
{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"{\"tool\":\"robot_oeh_manipulate\",\"arguments\":{\"mode\":\"vla\"}}"}}
{"type":"turn.completed","usage":{"input_tokens":1200,"cached_input_tokens":800,"output_tokens":90,"reasoning_output_tokens":60}}"#;

    #[tokio::test]
    async fn an_answer_becomes_one_tool_call_with_provider_usage() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("codex_home");
        let model = CodexExecModel::new(runtime(&fake(dir.path(), OK), Some(&home))).unwrap();
        let events = collect(&model).await.unwrap();
        assert_eq!(
            events[0],
            ModelEvent::Usage {
                input: 1200,
                output: 90,
                cached: 800,
                reasoning: Some(60)
            }
        );
        assert_eq!(
            events[1],
            ModelEvent::ToolCall(ToolCall {
                id: "th1:0".into(),
                name: "robot_oeh_manipulate".into(),
                args: json!({"mode": "vla"})
            })
        );
        assert_eq!(
            events[2],
            ModelEvent::Done {
                stop: StopReason::ToolUse
            }
        );
        // The whole argv, in order: the camera image goes in with `--image`, then `--` ends the
        // options so the stdin sentinel `-` is the prompt, not a fourth `--image` value.
        let argv = std::fs::read_to_string(dir.path().join("argv")).unwrap();
        let argv: Vec<&str> = argv.lines().collect();
        let work = argv[argv.iter().position(|a| *a == "--cd").unwrap() + 1];
        let image = format!("{work}/00-cam_head.png");
        let mut expected = FIXED.to_vec();
        expected.extend([
            "-c",
            "web_search=\"disabled\"",
            "--cd",
            work,
            "--model",
            "gpt-6-sol",
            "-c",
            "model_reasoning_effort=\"high\"",
            "--image",
            &image,
            "--",
            "-",
        ]);
        assert_eq!(argv, expected);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("home")).unwrap(),
            home.display().to_string()
        );
        let stdin = std::fs::read_to_string(dir.path().join("stdin")).unwrap();
        assert!(stdin.contains("robot_oeh_manipulate") && stdin.contains("escalation: stall"));
        assert_eq!(model.api_mode(), "codex_exec");
    }

    #[tokio::test]
    async fn tool_null_is_an_explicit_no_action_that_ends_the_turn() {
        let dir = tempfile::tempdir().unwrap();
        let body = OK.replace(
            r#"{\"tool\":\"robot_oeh_manipulate\",\"arguments\":{\"mode\":\"vla\"}}"#,
            r#"```json\n{\"tool\": null}\n```"#,
        );
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        let events = collect(&model).await.unwrap();
        assert!(matches!(&events[0], ModelEvent::TextDelta(t) if t.contains("null")));
        assert!(matches!(
            events[1],
            ModelEvent::Usage {
                reasoning: Some(60),
                ..
            }
        ));
        assert_eq!(
            events[2],
            ModelEvent::Done {
                stop: StopReason::EndTurn
            }
        );
        assert_eq!(events.len(), 3);
    }

    #[tokio::test]
    async fn a_near_miss_of_the_declared_tool_name_is_read_as_it_with_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let body = OK.replace(
            r#"\"tool\":\"robot_oeh_manipulate\""#,
            r#"\"tool\":\"robot_oe_manipulate\""#,
        );
        assert!(body.contains("robot_oe_manipulate"));
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        let events = collect(&model).await.unwrap();
        assert!(events
            .iter()
            .any(|e| matches!(e, ModelEvent::ToolCall(c) if c.name == "robot_oeh_manipulate")));
        assert!(events.iter().any(|e| matches!(e, ModelEvent::Warning(w)
            if w.contains("\"robot_oe_manipulate\" read as the declared \"robot_oeh_manipulate\""))));
        assert_eq!(edits("robot_oe_manipulate", "robot_oeh_manipulate"), 1);
        assert_eq!(edits("kitten", "sitting"), 3);
        assert!(edits("robot_other", "robot_oeh_manipulate") > TOOL_NAME_EDITS);
    }

    #[tokio::test]
    async fn any_other_answer_is_a_protocol_error_that_keeps_its_usage() {
        for answer in [
            r#"{\"tool\":\"robot_other\",\"arguments\":{}}"#,
            "I would grasp the cup.",
            r#"{\"tool\":\"robot_oeh_manipulate\",\"arguments\":[1]}"#,
            r#"{\"arguments\":{}}"#,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let body = OK.replace(
                r#"{\"tool\":\"robot_oeh_manipulate\",\"arguments\":{\"mode\":\"vla\"}}"#,
                answer,
            );
            let model =
                CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
            match usage_then(items(&model).await.unwrap()) {
                ModelError::Protocol(m) => assert!(
                    m.contains("neither a declared tool call") && m.contains("answer begins"),
                    "{m}"
                ),
                other => panic!("{answer}: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn every_tool_item_is_a_protocol_error_that_keeps_its_usage() {
        let refused = [
            (
                r#"{"type":"item.started","item":{"id":"c","type":"command_execution","command":"ls","aggregated_output":"","exit_code":null,"status":"in_progress"}}"#,
                "command_execution",
            ),
            (
                r#"{"type":"item.completed","item":{"id":"f","type":"file_change","changes":[],"status":"failed"}}"#,
                "file_change",
            ),
            (
                r#"{"type":"item.started","item":{"id":"m","type":"mcp_tool_call","server":"s","tool":"t","arguments":{},"result":null,"error":null,"status":"in_progress"}}"#,
                "mcp_tool_call",
            ),
            (
                r#"{"type":"item.started","item":{"id":"a","type":"collab_tool_call","tool":"spawn_agent","sender_thread_id":"x","receiver_thread_ids":[],"prompt":null,"agents_states":{},"status":"in_progress"}}"#,
                "collab_tool_call",
            ),
            (
                r#"{"type":"item.completed","item":{"id":"w","type":"web_search","query":"q","action":{"type":"other"}}}"#,
                "web_search",
            ),
            (
                r#"{"type":"item.started","item":{"id":"p","type":"todo_list","items":[]}}"#,
                "todo_list",
            ),
        ];
        for (line, expected) in refused {
            let dir = tempfile::tempdir().unwrap();
            let body = OK.replace(
                r#"{"type":"turn.started"}"#,
                &format!("{{\"type\":\"turn.started\"}}\n{line}"),
            );
            let model =
                CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
            match usage_then(items(&model).await.unwrap()) {
                ModelError::Protocol(m) => assert!(m.contains(expected), "{m}"),
                other => panic!("{expected}: {other:?}"),
            }
        }
    }

    const METADATA: &str = "Model metadata for `gpt-6-sol` not found. Defaulting to fallback metadata; this can degrade performance and cause issues.";

    fn warning_item(message: &str) -> String {
        json!({"type": "item.completed", "item": {"id": "e", "type": "error", "message": message}})
            .to_string()
    }

    #[tokio::test]
    async fn a_warning_item_is_recorded_and_the_call_kept() {
        let dir = tempfile::tempdir().unwrap();
        let body = OK.replace(
            r#"{"type":"turn.started"}"#,
            &format!("{{\"type\":\"turn.started\"}}\n{}", warning_item(METADATA)),
        );
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        let events = collect(&model).await.unwrap();
        assert_eq!(events[0], ModelEvent::Warning(METADATA.into()));
        assert!(matches!(
            events[1],
            ModelEvent::Usage {
                reasoning: Some(60),
                ..
            }
        ));
        assert!(matches!(events[2], ModelEvent::ToolCall(_)));
        assert_eq!(
            events[3],
            ModelEvent::Done {
                stop: StopReason::ToolUse
            }
        );
        // Bounded: a long warning keeps only its first MAX_ITEM_NOTE bytes.
        let long = "w".repeat(1000);
        let body = OK.replace(
            r#"{"type":"turn.started"}"#,
            &format!("{{\"type\":\"turn.started\"}}\n{}", warning_item(&long)),
        );
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        assert_eq!(
            collect(&model).await.unwrap()[0],
            ModelEvent::Warning("w".repeat(MAX_ITEM_NOTE))
        );
    }

    #[tokio::test]
    async fn warnings_survive_a_failed_call() {
        // After turn.completed: [Warning, Usage, Err].
        let dir = tempfile::tempdir().unwrap();
        let body = OK
            .replace(
                r#"{"type":"turn.started"}"#,
                &format!("{{\"type\":\"turn.started\"}}\n{}", warning_item("reroute")),
            )
            .replace(
                r#"{\"tool\":\"robot_oeh_manipulate\",\"arguments\":{\"mode\":\"vla\"}}"#,
                "not an answer",
            );
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        let mut items = items(&model).await.unwrap().into_iter();
        assert_eq!(
            items.next().unwrap().unwrap(),
            ModelEvent::Warning("reroute".into())
        );
        assert!(matches!(
            items.next().unwrap(),
            Ok(ModelEvent::Usage { .. })
        ));
        assert!(matches!(
            items.next().unwrap(),
            Err(ModelError::Protocol(_))
        ));
        assert!(items.next().is_none());
        // Without turn.completed there is no stream: the warning goes into the error itself.
        let body = format!(
            "{{\"type\":\"thread.started\",\"thread_id\":\"t\"}}\n{}",
            warning_item(METADATA)
        );
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        match collect(&model).await {
            Err(ModelError::Transport(m)) => assert!(
                m.contains("did not complete") && m.contains(METADATA),
                "{m}"
            ),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_reconnect_notice_is_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let body = OK.replace(
            r#"{"type":"turn.started"}"#,
            "{\"type\":\"turn.started\"}\n{\"type\":\"error\",\"message\":\"Reconnecting... 1/5 (stream disconnected before completion)\"}",
        );
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), &body), Some(dir.path()))).unwrap();
        let events = collect(&model).await.unwrap();
        assert!(matches!(events[1], ModelEvent::ToolCall(_)));
        assert_eq!(
            events[2],
            ModelEvent::Done {
                stop: StopReason::ToolUse
            }
        );
    }

    #[tokio::test]
    async fn a_turn_that_never_completes_fails_with_the_last_error() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{"type":"thread.started","thread_id":"th1"}
{"type":"turn.started"}
{"type":"error","message":"Reconnecting... 5/5 (stream disconnected before completion)"}"#;
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), body), Some(dir.path()))).unwrap();
        match collect(&model).await {
            Err(ModelError::Transport(m)) => {
                assert!(m.contains("did not complete") && m.contains("5/5"), "{m}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_failed_turn_and_a_missing_login_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{"type":"thread.started","thread_id":"th1"}
{"type":"turn.failed","error":{"message":"usage limit reached"}}"#;
        let model =
            CodexExecModel::new(runtime(&fake(dir.path(), body), Some(dir.path()))).unwrap();
        match collect(&model).await {
            Err(ModelError::Transport(m)) => assert!(m.contains("usage limit"), "{m}"),
            other => panic!("{other:?}"),
        }
        let model = CodexExecModel::new(runtime(&fake(dir.path(), OK), None)).unwrap();
        assert!(
            matches!(collect(&model).await, Err(ModelError::MissingKey(ref v)) if v == "OEH_CODEX_HOME")
        );
    }

    #[test]
    fn the_stdin_sentinel_follows_an_end_of_options_marker() {
        let rt = runtime(Path::new("codex"), None);
        let images = [
            PathBuf::from("/w/00-cam_head.png"),
            PathBuf::from("/w/01-cam_left_wrist.png"),
            PathBuf::from("/w/02-cam_right_wrist.png"),
        ];
        let args = exec_args(&rt, Path::new("/w"), &images);
        let args: Vec<&str> = args.iter().map(|a| a.to_str().unwrap()).collect();
        let mut expected = FIXED.to_vec();
        expected.extend([
            "-c",
            "web_search=\"disabled\"",
            "--cd",
            "/w",
            "--model",
            "gpt-6-sol",
            "-c",
            "model_reasoning_effort=\"high\"",
            "--image",
            "/w/00-cam_head.png",
            "--image",
            "/w/01-cam_left_wrist.png",
            "--image",
            "/w/02-cam_right_wrist.png",
            "--",
            "-",
        ]);
        assert_eq!(args, expected);
    }

    /// The subscription login is the only credential the child sees: no API key can switch it to API billing.
    #[test]
    fn api_key_variables_are_removed_from_the_child() {
        let rt = runtime(Path::new("codex"), None);
        let cmd = command(&rt, Path::new("/home"), Path::new("/w"), &[]);
        let envs: Vec<_> = cmd.as_std().get_envs().collect();
        assert!(envs.contains(&("CODEX_HOME".as_ref(), Some("/home".as_ref()))));
        for name in API_KEY_ENV {
            assert!(envs.contains(&(name.as_ref(), None)), "{name}");
        }
    }

    #[test]
    fn base64_round_trips_standard_padding() {
        assert_eq!(decode_base64("aGk=").unwrap(), b"hi");
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("YWJj").unwrap(), b"abc");
        assert!(decode_base64("a*c=").is_none());
    }
}
