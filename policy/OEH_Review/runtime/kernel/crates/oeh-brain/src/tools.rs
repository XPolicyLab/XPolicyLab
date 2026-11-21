//! 工具注册表：模型看到的那张工具表，以及怎么执行一批工具调用。
//!
//! **工具是能力的投影，不是能力本身**（规格 §7.2）。每个能力投成一个 `robot_<name>`
//! 工具，调用它就是往内核提一次提案（`TurnHandle::propose`）——准入链、规则、审批、
//! 租约全都还在原来的位置，脑这边没有旁路。除此之外只有两类工具：
//!
//! - `recall_grep` / `recall_read` / `recall_glob`：沙箱在一个根目录下的**只读**检索
//!   （skills/、memory/）。它们不经过内核，因为读自己的笔记不是对世界的动作。
//! - `ask_slow_brain`：向慢脑提问，也是一次提案（能力同名，执行器记一条 `brain/ask`）。
//!   它是唯一一个不带 `robot_` 前缀的能力工具——问一个脑不是对机器人的动作。
//! - `skill_manage` / `memory_manage` / `read`：RSI 第一级的知识读写
//!   （[`ToolRegistry::add_knowledge`]）。**只注册进 review fork 自己那张表**，普通
//!   turn 的工具表里没有它们——一个能在任意 turn 里改自己 skill 的脑，就是一个能在被
//!   steer 的半路上改自己行为的脑。
//!
//! 批执行的规则同样来自 §7.2：连续的 `concurrency_safe` 调用并行，exclusive 调用是
//! 屏障。motion 类能力天然互斥（同一时刻只能有一条腿的命令），speech/query 可以并行。

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Read};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::FutureExt;
use oeh_admission::capability::{CapabilityClass, CapabilitySpec};
use oeh_kernel::proposal::{BoxFuture, Outcome};
use oeh_kernel::TurnHandle;
use serde::Serialize;
use serde_json::{json, Value};

use crate::model::{ToolCall, ToolSpec};
use crate::rsi::{KnowledgeStore, RsiError};

/// 一次工具调用回给模型的内容上限（16 KiB）。超出就截断并带 `truncated: true`——模型的
/// 上下文是预算，一次 grep 不能把它吃光。
pub const MAX_TOOL_OUTPUT: usize = 16 * 1024;

/// grep 里单行的上限：一行长到离谱（压缩过的 JSON、base64）时只留开头。
const MAX_LINE: usize = 1024;

/// 一次 recall 遍历最多看的文件数与目录深度。沙箱里本该只有笔记，但别让一个奇怪的树
/// 把一次 grep 变成一次全盘扫描。
const MAX_WALK_FILES: usize = 4096;
const MAX_WALK_DEPTH: usize = 16;

/// 一个 glob 模式最多这么长（字节），一次 `recall_glob` 最多回这么多条路径。
///
/// 模式是**模型给的**一个参数，而它会被拿去比对每一个走到的文件（最多
/// [`MAX_WALK_FILES`] 个）。匹配器本身是线性的（见 [`seg_match`]），所以这两个上限不是
/// 在补一个算法漏洞，而是和走文件数、输出预算同一类的界：一次奇怪的调用的代价必须是
/// 常数级的，模型才可以自由地试。
const MAX_GLOB_PATTERN: usize = 256;
const MAX_GLOB_MATCHES: usize = 1024;

/// 一次工具调用的结果。`ok` 决定 transcript 里这条 `tool` 消息算成功还是失败；
/// `result` 是回给模型的 JSON（能力工具放的是 `Outcome` 本身）。
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutcome {
    pub ok: bool,
    pub result: Value,
}

impl ToolOutcome {
    pub fn ok(result: Value) -> Self {
        ToolOutcome { ok: true, result }
    }

    /// 失败一律是 `{"error": "..."}`：模型只要看一个字段就知道出了什么事。
    pub fn err(msg: impl Into<String>) -> Self {
        ToolOutcome {
            ok: false,
            result: json!({ "error": msg.into() }),
        }
    }
}

pub type ToolFuture = BoxFuture<ToolOutcome>;

/// 一个工具的实现。`call` 必须立刻返回 future（不在调用线程上干活）：一批安全调用是
/// 先把 future 全造出来再一起 poll 的，在 `call` 里同步阻塞就等于把并行变回串行。
///
/// `turn` 是 **`Option`**：有些调用方根本不在一个 turn 里。RSI 的 review fork 就是这样
/// （维护相位跑在 turn 的终态之后，会话末那一次干脆跑在 detach 之后），而它那张表里的
/// 三个工具一个都不提案。把"没有 turn"写进类型，比传一个假的句柄进来诚实：一个需要
/// turn 的工具拿到 `None` 时会明明白白地回一条 `ok:false`，而不是把提案递给一个
/// 早就没人接的 `Weak` 然后收到一句 "brains gone"。
///
/// 这同时是"review 提不了案"的第三重结构性保证（前两重：review 用的是另一张表，那张
/// 表里根本没有提案类工具）。
pub trait ToolHandler: Send + Sync {
    fn call(&self, args: Value, turn: Option<&TurnHandle>) -> ToolFuture;
}

/// 一个脑这一轮看到的全部工具。按名字有序（`BTreeMap`）：prompt 缓存要求工具 schema
/// 的顺序固定，排序是最省事的固定法。
///
/// `Clone` 是浅的（每个 handler 都是一个 `Arc`），存在的理由是重投影：
/// [`crate::native::NativeBrain`] 留着一张**底表**（`recall_*` / `ask_slow_brain`），每次
/// 能力面变动就克隆它再叠一遍 [`Self::project_capabilities`]。在一张已经投过的表上做
/// 增量（"减掉上次多的、加上这次少的"）是一次可以写错的差分，克隆重建不可能错。
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, (ToolSpec, Arc<dyn ToolHandler>)>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个工具；同名覆盖。
    pub fn add(&mut self, spec: ToolSpec, handler: Arc<dyn ToolHandler>) {
        self.tools.insert(spec.name.clone(), (spec, handler));
    }

    /// 能力面 → 工具面。每个能力变成 `robot_<name>`：
    ///
    /// - schema 用能力自己的 `params_schema`（不是对象 schema 的话兜底成
    ///   `{"type":"object"}`——厂商的工具定义都要求顶层是对象）；
    /// - description 是 `"<class>/<risk_level> capability <name>"`，让模型看得见风险等级；
    /// - `concurrency_safe = class != Motion`。
    ///
    /// handler 就是 `turn.propose(<能力名>, args)`：工具名带 `robot_` 前缀只是给模型看的，
    /// 提案用的始终是能力的真名。
    ///
    /// 名字不能直接当工具名（`[a-z0-9_]` 之外的字符）的能力会被**跳过**，并作为返回值
    /// 报给调用方：三家厂商对工具名的字符集都有要求，一个 `robot_walk.fast` 换来的是整
    /// 个请求的 HTTP 400——那时候**所有**工具都用不了，而日志里没有任何线索。
    ///
    /// 为什么是返回值而不是就地 `warn!`：这个函数现在**每个 turn 都可能被调一次**
    /// （能力面是活的，见 [`crate::native::NativeBrain::with_knowledge`]），而一条每个
    /// turn 都重复一遍的告警等于没有告警。谁调用谁知道"和上次比变了没有"，所以那个判断
    /// 归调用方。
    #[must_use = "the skipped capability names have to be reported somewhere"]
    pub fn project_capabilities(&mut self, specs: &[CapabilitySpec]) -> Vec<String> {
        let mut skipped = Vec::new();
        for spec in specs {
            if spec.name == ASK_SLOW_BRAIN {
                // 它确实是一个能力（提问也要过准入），但它的工具面由
                // [`ToolRegistry::add_ask_slow_brain`] 手写，名字不带 `robot_` 前缀——
                // 问一个脑不是对机器人的动作。投影再造一个 `robot_ask_slow_brain` 只会
                // 让模型面对两个同义工具，而两个里挑哪个都对。
                continue;
            }
            if !is_projectable_name(&spec.name) {
                skipped.push(spec.name.clone());
                continue;
            }
            let parameters = if spec.params_schema.is_object() {
                spec.params_schema.clone()
            } else {
                json!({ "type": "object" })
            };
            let tool = ToolSpec {
                name: format!("robot_{}", spec.name),
                description: format!(
                    "{}/{} capability {}",
                    serde_name(&spec.class),
                    serde_name(&spec.risk_level),
                    spec.name
                ),
                parameters,
                concurrency_safe: spec.class != CapabilityClass::Motion,
            };
            self.add(
                tool,
                Arc::new(CapabilityTool {
                    capability: spec.name.clone(),
                }),
            );
        }
        skipped
    }

    /// `recall_read` / `recall_grep` / `recall_glob`：沙箱在 `root` 下的只读检索。
    ///
    /// `root` 在这里 canonicalize 一次（跟随符号链接、去掉 `..`），之后每个路径参数都
    /// 要 join + canonicalize + `starts_with(root)` 才算数：`../../etc/passwd` 会被挡下，
    /// 指向沙箱外的符号链接也会（canonicalize 先把它解开）。
    ///
    /// **一张表只有一个 recall 根**：三个工具名是固定的，第二次调用只会把第一个根悄悄
    /// 顶掉（`r.add_recall(skills); r.add_recall(memory)` 会换来一张对 skills 失明的表，
    /// 没有任何提示）。所以第二次调用不改任何东西，只记一条 `error`——签名是 brief 钉
    /// 死的，不能改成 `Result`，那就至少别静默。
    pub fn add_recall(&mut self, root: PathBuf) {
        if self.tools.contains_key("recall_read") {
            tracing::error!(
                root = %root.display(),
                "recall tools are already registered; keeping the first root (one recall root per registry)"
            );
            return;
        }
        let sb = Arc::new(Sandbox::new(root));
        self.add(
            ToolSpec {
                name: "recall_read".into(),
                description: format!(
                    "read a text file from the recall root (read-only, first {MAX_TOOL_OUTPUT} bytes)"
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "path relative to the recall root" }
                    },
                    "required": ["path"]
                }),
                concurrency_safe: true,
            },
            Arc::new(RecallTool {
                sb: sb.clone(),
                kind: RecallKind::Read,
            }),
        );
        self.add(
            ToolSpec {
                name: "recall_grep".into(),
                description: "search the recall root for a literal substring; returns {path, line, text} matches".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "literal substring to look for" },
                        "path": { "type": "string", "description": "optional file or directory to search, relative to the recall root" }
                    },
                    "required": ["pattern"]
                }),
                concurrency_safe: true,
            },
            Arc::new(RecallTool {
                sb: sb.clone(),
                kind: RecallKind::Grep,
            }),
        );
        self.add(
            ToolSpec {
                name: "recall_glob".into(),
                description: "list files under the recall root matching a glob (`*` within a path segment, `**` across segments)".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "for example `**/*.md`" }
                    },
                    "required": ["pattern"]
                }),
                concurrency_safe: true,
            },
            Arc::new(RecallTool {
                sb,
                kind: RecallKind::Glob,
            }),
        );
    }

    /// `skill_manage` / `memory_manage` / `read`：RSI 第一级（[`crate::rsi`]）的三个
    /// 工具，全部绑在**同一个** store 和**同一个** actor 上。
    ///
    /// `actor` 是账本和 `created_by` 里那个名字（review fork 用 `review:<brain_id>`）。
    /// 它由这里绑死而不是当成一个工具参数：参数是模型给的，而"我是谁"这件事不能由被
    /// 检查的一方填——否则所有权护栏（`skill` 只有创建者能改）一句 `actor: "seed"` 就
    /// 绕过去了。
    ///
    /// 路径合法性**一点都不在这里判**，全部交给 [`KnowledgeStore`]：白名单形状、`..`、
    /// 符号链接、受保护前缀是一整套互相咬合的判定，抄一半到这里只会多出一份会漂移的
    /// 真相；更要紧的是，store 在拒绝一次越界写的同时会记一条 `safety/tamper_attempt`，
    /// 而在这里先拦下来等于把一次篡改尝试降级成一条悄无声息的工具错误。
    ///
    /// 两个写工具是 `concurrency_safe: false`（批执行里的屏障）。它们**不是**为了线程
    /// 安全——store 自己有写锁——而是为了语义：同一批里并行的两次写会让第二次拿到一个
    /// 它没读过的哈希，于是 read-before-write 变成一次随机的失败。
    pub fn add_knowledge(&mut self, store: Arc<KnowledgeStore>, actor: String) {
        let hash = "sha256 of the version you last read; required for `update`";
        self.add(
            ToolSpec {
                name: "skill_manage".into(),
                description: "create or update one of this robot's skills at skills/<name>/SKILL.md (Markdown; the first line is the title)".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "op": { "type": "string", "enum": ["create", "update"] },
                        "path": { "type": "string", "description": "skills/<name>/SKILL.md" },
                        "content": { "type": "string", "description": "the whole file, not a patch" },
                        "expect_sha256": { "type": "string", "description": hash }
                    },
                    "required": ["op", "path", "content"]
                }),
                concurrency_safe: false,
            },
            Arc::new(KnowledgeTool {
                store: store.clone(),
                actor: actor.clone(),
                kind: KnowledgeKind::Skill,
            }),
        );
        self.add(
            ToolSpec {
                name: "memory_manage".into(),
                description: "update or append to this robot's memory at memory/MEMORY.md or memory/topics/<name>.md".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "op": { "type": "string", "enum": ["update", "append"] },
                        "path": { "type": "string", "description": "memory/MEMORY.md or memory/topics/<name>.md" },
                        "content": { "type": "string", "description": "the whole file for `update`, the new text for `append`" },
                        "expect_sha256": { "type": "string", "description": hash }
                    },
                    "required": ["op", "path", "content"]
                }),
                concurrency_safe: false,
            },
            Arc::new(KnowledgeTool {
                store: store.clone(),
                actor: actor.clone(),
                kind: KnowledgeKind::Memory,
            }),
        );
        self.add(
            ToolSpec {
                name: "read".into(),
                description: "read one skill or memory file; returns {text, sha256} and that sha256 is what a later update must carry".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "skills/<name>/SKILL.md, memory/MEMORY.md or memory/topics/<name>.md" }
                    },
                    "required": ["path"]
                }),
                concurrency_safe: true,
            },
            Arc::new(KnowledgeTool {
                store,
                actor,
                kind: KnowledgeKind::Read,
            }),
        );
    }

    /// `ask_slow_brain{q}`：向慢脑提问。和能力工具一样是一次提案（能力名
    /// `ask_slow_brain`），执行器记一条 `brain/ask`，慢脑的 wake 策略负责醒来。
    ///
    /// 工具名不带 `robot_` 前缀（[`ToolRegistry::project_capabilities`] 因此跳过这个
    /// 能力）：问一个脑不是对机器人的动作。能力没注册时提案会被准入以"没有这个能力"
    /// 挡下，工具照常返回一条 `ok:false` 的结果，脑不会挂。
    pub fn add_ask_slow_brain(&mut self) {
        self.add(
            ToolSpec {
                name: ASK_SLOW_BRAIN.into(),
                // 描述必须说清楚它**不阻塞**：调用当场还 `{ask_id}`，回答是之后一条
                // 单独的 `brain/answer`（提问者的 wake 策略把它变成一次 steer 或一个新
                // turn）。以为自己会等到答案的模型会把这一个 turn 的计划做错。
                description:
                    "query capability ask_slow_brain: ask the slow brain a question. Returns {ask_id} immediately and does NOT wait: the answer arrives later as a brain/answer event that wakes or steers you"
                        .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string", "description": "the question, in one sentence" }
                    },
                    "required": ["q"]
                }),
                concurrency_safe: true,
            },
            Arc::new(AskSlowBrain),
        );
    }

    /// 全部工具的 spec，按名字排序（prompt 前缀要字节级稳定）。
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|(s, _)| s.clone()).collect()
    }

    /// Restrict both advertised tools and executable handlers for a frozen policy.
    pub fn retain_names(&mut self, names: &[String]) {
        self.tools.retain(|name, _| names.contains(name));
    }

    pub fn get(&self, name: &str) -> Option<(&ToolSpec, &Arc<dyn ToolHandler>)> {
        self.tools.get(name).map(|(s, h)| (s, h))
    }

    /// 跑一批模型给出的工具调用。
    ///
    /// 连续的 `concurrency_safe` 调用并行（一组一起 `join_all`），exclusive 调用是屏障：
    /// 它前面那一组全部结束之后它才开始，它结束之后后面的才开始。返回**按模型给出的
    /// 顺序**——`tool/result` 必须按模型顺序回到 transcript 里，否则下一个 step 的
    /// call_id 对不上。
    ///
    /// handler 也按模型顺序被调用（并行组是先按顺序把 future 造出来，再一起 poll）。
    /// 未知工具立刻还一条 `ok:false`，不当屏障——它没有副作用，没什么好等的。
    pub async fn run_batch(
        &self,
        calls: Vec<ToolCall>,
        turn: Option<&TurnHandle>,
    ) -> Vec<(ToolCall, ToolOutcome)> {
        let mut out: Vec<(ToolCall, ToolOutcome)> = Vec::with_capacity(calls.len());
        let mut group: Vec<(ToolCall, ToolFuture)> = Vec::new();
        for call in calls {
            let safe = self
                .get(&call.name)
                .map(|(s, _)| s.concurrency_safe)
                .unwrap_or(true);
            if !safe {
                // 屏障：先把在飞的安全调用等完，再单独起这一条。dispatch 放在等待
                // 之后——handler 在 `call` 里就可能把提案递出去了，那也得排在屏障后面。
                run_group(&mut group, &mut out).await;
                let fut = self.dispatch(&call, turn);
                let outcome = fut.await;
                out.push((call, outcome));
            } else {
                let fut = self.dispatch(&call, turn);
                group.push((call, fut));
            }
        }
        run_group(&mut group, &mut out).await;
        out
    }

    /// 找到 handler 并把它包成"panic 也只是一条失败结果"的 future。
    ///
    /// `ToolHandler` 是这个工作区的一条扩展边界（规则、执行器、RPC handler、fiber apply
    /// 都包了 `catch_unwind`，Task 2 还会从 [`ToolRegistry::add`] 注入脚本工具），所以这
    /// 里也包：一个工具 panic 不该把 turn 带走，更不该让**同一批里别的工具**的结果跟着
    /// 消失——那样模型下一步会看到有 `tool_call` 却没有对应的 `tool` 消息，请求直接被
    /// 厂商拒掉。同步的那一截（`call` 自己，参数解析就在里面）和它返回的 future 都包。
    fn dispatch(&self, call: &ToolCall, turn: Option<&TurnHandle>) -> ToolFuture {
        let Some((_, h)) = self.tools.get(&call.name) else {
            let name = call.name.clone();
            return Box::pin(async move { ToolOutcome::err(format!("unknown tool: {name}")) });
        };
        let args = call.args.clone();
        match std::panic::catch_unwind(AssertUnwindSafe(|| h.call(args, turn))) {
            Ok(fut) => Box::pin(async move {
                match AssertUnwindSafe(fut).catch_unwind().await {
                    Ok(outcome) => outcome,
                    Err(_) => ToolOutcome::err(PANICKED),
                }
            }),
            Err(_) => Box::pin(async { ToolOutcome::err(PANICKED) }),
        }
    }
}

/// handler panic 之后回给模型的那句话。
const PANICKED: &str = "tool panicked";

/// 把攒下的一组安全调用一起跑完，结果按组内顺序追加。
async fn run_group(
    group: &mut Vec<(ToolCall, ToolFuture)>,
    out: &mut Vec<(ToolCall, ToolOutcome)>,
) {
    if group.is_empty() {
        return;
    }
    let (calls, futs): (Vec<ToolCall>, Vec<ToolFuture>) = group.drain(..).unzip();
    let outcomes = futures_util::future::join_all(futs).await;
    out.extend(calls.into_iter().zip(outcomes));
}

/// 慢脑提问用的能力名（工具名同名，没有 `robot_` 前缀：它不是机器人的动作）。
pub(crate) const ASK_SLOW_BRAIN: &str = "ask_slow_brain";

/// 一个能力投影出来的工具：调用 = 提一次提案。
struct CapabilityTool {
    capability: String,
}

impl ToolHandler for CapabilityTool {
    fn call(&self, args: Value, turn: Option<&TurnHandle>) -> ToolFuture {
        let Some(turn) = turn else { return no_turn() };
        propose(turn, &self.capability, args)
    }
}

struct AskSlowBrain;

impl ToolHandler for AskSlowBrain {
    fn call(&self, args: Value, turn: Option<&TurnHandle>) -> ToolFuture {
        let Some(turn) = turn else { return no_turn() };
        let Some(q) = args.get("q").and_then(Value::as_str) else {
            return Box::pin(async { ToolOutcome::err("ask_slow_brain requires a string `q`") });
        };
        propose(turn, ASK_SLOW_BRAIN, json!({ "q": q }))
    }
}

/// 一个需要 turn 的工具被在 turn 之外调用了。说清楚是"没有 turn"而不是"提案被拒"：
/// 这两者要查的地方完全不同。
fn no_turn() -> ToolFuture {
    Box::pin(async { ToolOutcome::err("no live turn: this tool can only be called inside a turn") })
}

/// 提案 → 工具结果。`ok` 只在 `Executed` 时为真；`result` 是 `Outcome` 的 JSON（内部
/// tag `outcome`），所以被拒/失败的理由原样进 transcript，模型能据此改主意。
fn propose(turn: &TurnHandle, capability: &str, args: Value) -> ToolFuture {
    let fut = turn.propose(capability, args);
    Box::pin(async move {
        let outcome = fut.await;
        ToolOutcome {
            ok: matches!(outcome, Outcome::Executed { .. }),
            result: serde_json::to_value(&outcome)
                .unwrap_or_else(|e| json!({ "error": format!("outcome not serialisable: {e}") })),
        }
    })
}

/// 枚举的 serde 名字（两个能力枚举都是 `rename_all = "lowercase"`）。从 serde 取而不是
/// 另写一份 match：description 里的 class/risk 名字必须和日志、规则链里的是同一套。
fn serde_name<T: Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(Value::String(s)) => s,
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// 知识：RSI 第一级的读写
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum KnowledgeKind {
    Skill,
    Memory,
    Read,
}

struct KnowledgeTool {
    store: Arc<KnowledgeStore>,
    actor: String,
    kind: KnowledgeKind,
}

impl ToolHandler for KnowledgeTool {
    fn call(&self, args: Value, _turn: Option<&TurnHandle>) -> ToolFuture {
        // `TurnHandle` 用不上：这三个工具**不提案**。改自己的笔记不是对世界的动作，所以
        // 它们和 `recall_*` 一样绕过内核——区别只在这三个会写。
        let (store, actor, kind) = (self.store.clone(), self.actor.clone(), self.kind);
        // store 全是同步文件操作（还带 fsync），不能占着 runtime 的工作线程。
        Box::pin(async move {
            match tokio::task::spawn_blocking(move || knowledge(&store, &actor, kind, &args)).await
            {
                Ok(out) => out,
                Err(e) => ToolOutcome::err(format!("knowledge task failed: {e}")),
            }
        })
    }
}

fn knowledge(
    store: &KnowledgeStore,
    actor: &str,
    kind: KnowledgeKind,
    args: &Value,
) -> ToolOutcome {
    match kind {
        KnowledgeKind::Read => read_doc(store, args),
        KnowledgeKind::Skill => write_doc(store, actor, args, &["create", "update"]),
        KnowledgeKind::Memory => write_doc(store, actor, args, &["update", "append"]),
    }
}

fn read_doc(store: &KnowledgeStore, args: &Value) -> ToolOutcome {
    let Some(path) = str_arg(args, "path") else {
        return ToolOutcome::err("read requires a string `path`");
    };
    match store.read(path) {
        Ok(doc) => {
            let (text, cut) = cap(&doc.text, MAX_TOOL_OUTPUT);
            // `sha256` 始终是**全文**的哈希，所以截断过的一份不能被当成"我读过它了"
            // 再原样写回去——那会把后半个文件悄悄删掉。说出来，由模型决定怎么办。
            let mut v = json!({ "rel": doc.rel, "text": text, "sha256": doc.sha256 });
            if cut {
                v["truncated"] = Value::Bool(true);
                v["note"] = Value::String(
                    "text was cut to fit; sha256 is of the whole file, do not write this back as a full update"
                        .into(),
                );
            }
            ToolOutcome::ok(v)
        }
        Err(e) => ToolOutcome::err(e.to_string()),
    }
}

/// 一次写：`create` / `update` / `append` 三种 op 都落到 [`KnowledgeStore::write`] 上。
///
/// `update`（和 `append`）**没带哈希时传 `Some("")` 而不是 `None`**。`write` 把
/// `None` 读成"新建"，于是一次打错文件名的 `update` 会安静地建出一个新文件——这正是
/// read-before-write 要挡的那一类错误，只是换了个面目。空串永远对不上任何内容的
/// sha256，所以模型拿到的是一条 `StaleRead`，正是它该看到的东西。
fn write_doc(store: &KnowledgeStore, actor: &str, args: &Value, allowed: &[&str]) -> ToolOutcome {
    let (Some(op), Some(path), Some(content)) = (
        str_arg(args, "op"),
        str_arg(args, "path"),
        str_arg(args, "content"),
    ) else {
        return ToolOutcome::err("this tool requires string `op`, `path` and `content`");
    };
    if !allowed.contains(&op) {
        return ToolOutcome::err(format!("`op` must be one of {allowed:?}, got {op:?}"));
    }
    let expect = str_arg(args, "expect_sha256");
    let written = match op {
        "append" => append_doc(store, actor, path, content, expect),
        "create" => store.write(path, content, actor, expect),
        _ => store.write(path, content, actor, Some(expect.unwrap_or(""))),
    };
    match written {
        // `rel` 也回给调用方：review 的 `writes` 列表要的是"哪几个文件真的变了"，从
        // 参数里回读等于相信模型填对了，从结果里读的是 store 真的提交了什么。
        Ok(doc) => ToolOutcome::ok(json!({ "rel": doc.rel, "sha256": doc.sha256 })),
        Err(e) => ToolOutcome::err(e.to_string()),
    }
}

/// `append` = 读 + 拼 + 用**刚读到的**哈希写回去。
///
/// 文件还不存在就等于从空开始（于是 `append` 顺带把 `memory/MEMORY.md` 建出来）。
/// 调用方给了 `expect_sha256` 就先核对一遍：它说的是"我以为现在是这一版"，如果那一版
/// 已经不是当前版本，静悄悄地追加在一份它没读过的内容后面比报错糟得多。
fn append_doc(
    store: &KnowledgeStore,
    actor: &str,
    path: &str,
    content: &str,
    expect: Option<&str>,
) -> Result<crate::rsi::Doc, RsiError> {
    let cur = match store.read(path) {
        Ok(d) => Some(d),
        Err(RsiError::NotFound(_)) => None,
        Err(e) => return Err(e),
    };
    let actual = cur.as_ref().map(|d| d.sha256.clone()).unwrap_or_default();
    if let Some(want) = expect {
        if want != actual {
            return Err(RsiError::StaleRead {
                expected: want.to_string(),
                actual,
            });
        }
    }
    let mut text = cur.as_ref().map(|d| d.text.clone()).unwrap_or_default();
    // 追加的内容自成一行：没有这一下，两条记录会粘成一行，而这些文件是 Markdown。
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(content);
    store.write(path, &text, actor, cur.as_ref().map(|d| d.sha256.as_str()))
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

// ---------------------------------------------------------------------------
// recall：沙箱化的只读检索
// ---------------------------------------------------------------------------

/// 一个 canonicalize 过的根目录，加上"这个路径还在里面吗"的判定。
struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(root: PathBuf) -> Self {
        // 根目录不存在时 canonicalize 会失败：留着原路径，后面每个 join 的
        // canonicalize 一样会失败，于是所有调用都被拒——比 panic 或静默放行都好。
        let root = root.canonicalize().unwrap_or(root);
        Sandbox { root }
    }

    /// 模型给的相对路径 → 沙箱内的真实路径。canonicalize 会解开符号链接，所以"链接到
    /// 沙箱外"和"`..` 爬出去"是同一条检查挡下的。
    fn resolve(&self, rel: &str) -> Result<PathBuf, String> {
        let real = self
            .root
            .join(rel)
            .canonicalize()
            .map_err(|e| format!("cannot resolve {rel:?} under the recall root: {e}"))?;
        if !real.starts_with(&self.root) {
            return Err(format!("path escapes the recall root: {rel:?}"));
        }
        Ok(real)
    }

    /// 真实路径 → 回给模型的相对路径（模型只该看见沙箱内的名字）。
    fn rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned()
    }
}

#[derive(Clone, Copy)]
enum RecallKind {
    Read,
    Grep,
    Glob,
}

struct RecallTool {
    sb: Arc<Sandbox>,
    kind: RecallKind,
}

impl ToolHandler for RecallTool {
    fn call(&self, args: Value, _turn: Option<&TurnHandle>) -> ToolFuture {
        let (sb, kind) = (self.sb.clone(), self.kind);
        // 文件系统是阻塞的：一次 grep 可能读几百个文件，不能占着 runtime 的工作线程。
        Box::pin(async move {
            let join = tokio::task::spawn_blocking(move || match kind {
                RecallKind::Read => recall_read(&sb, &args),
                RecallKind::Grep => recall_grep(&sb, &args),
                RecallKind::Glob => recall_glob(&sb, &args),
            })
            .await;
            match join {
                Ok(out) => out,
                Err(e) => ToolOutcome::err(format!("recall task failed: {e}")),
            }
        })
    }
}

fn recall_read(sb: &Sandbox, args: &Value) -> ToolOutcome {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return ToolOutcome::err("recall_read requires a string `path`");
    };
    let real = match sb.resolve(path) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    if !real.is_file() {
        return ToolOutcome::err(format!("not a file: {path:?}"));
    }
    let file = match fs::File::open(&real) {
        Ok(f) => f,
        Err(e) => return ToolOutcome::err(format!("cannot read {path:?}: {e}")),
    };
    // 最多读 cap + 1 字节，多出来的那一个字节只用来回答"是不是还有更多"。上限是在
    // **读之前**生效的：一个只读、声称自己最多回 16 KiB 的工具，不能因为 memory/ 里
    // 躺着一个 3 GB 的 session log 就把进程撑爆。
    let mut bytes = Vec::new();
    if let Err(e) = file
        .take(MAX_TOOL_OUTPUT as u64 + 1)
        .read_to_end(&mut bytes)
    {
        return ToolOutcome::err(format!("cannot read {path:?}: {e}"));
    }
    let more = bytes.len() > MAX_TOOL_OUTPUT;
    let (content, cut) = cap(&String::from_utf8_lossy(&bytes), MAX_TOOL_OUTPUT);
    ToolOutcome::ok(json!({
        "path": sb.rel(&real),
        "content": content,
        "truncated": more || cut,
    }))
}

fn recall_grep(sb: &Sandbox, args: &Value) -> ToolOutcome {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return ToolOutcome::err("recall_grep requires a string `pattern`");
    };
    let base = match args.get("path").and_then(Value::as_str) {
        Some(p) => match sb.resolve(p) {
            Ok(p) => p,
            Err(e) => return ToolOutcome::err(e),
        },
        None => sb.root.clone(),
    };
    let mut matches = Vec::new();
    let mut budget = MAX_TOOL_OUTPUT;
    let found = walk(sb, &base);
    // 遍历被上限砍掉过也算截断：`{"matches": [], "truncated": false}` 会被模型读成
    // 权威的"没有这条笔记"，而那是 recall 最不能有的失败方式。
    let mut truncated = found.truncated;
    for file in found.files {
        let rel = sb.rel(&file);
        if !grep_file(&file, &rel, pattern, &mut budget, &mut matches) {
            truncated = true;
            break;
        }
    }
    ToolOutcome::ok(json!({ "matches": matches, "truncated": truncated }))
}

/// 在一个文件里找 `pattern`（字面子串），**逐行**读。
///
/// 整个文件从不一次性进内存：每次最多持有一行，而且一行超过 [`MAX_LINE`] 就到此为止、
/// 余下的直接丢到行尾（`skip_to_eol`，只过 `BufReader` 的缓冲区）。所以一个 3 GB 的
/// 单行文件也只花 1 KiB。返回 `false` 表示输出预算用尽，外面该停了。
fn grep_file(
    path: &Path,
    rel: &str,
    pattern: &str,
    budget: &mut usize,
    matches: &mut Vec<Value>,
) -> bool {
    let Ok(file) = fs::File::open(path) else {
        return true;
    };
    let mut reader = io::BufReader::new(file);
    let mut buf = Vec::new();
    let mut lineno = 0usize;
    loop {
        buf.clear();
        let n = match (&mut reader)
            .take(MAX_LINE as u64 + 1)
            .read_until(b'\n', &mut buf)
        {
            Ok(n) => n,
            Err(_) => return true,
        };
        if n == 0 {
            return true; // EOF
        }
        lineno += 1;
        // 没读到换行且正好读满上限 → 这一行还没完。剩下的不进内存，但**继续找**：
        // 丢掉的部分要是不算数，一条 3 GB 单行日志里的匹配就成了"没找到"。
        let mut line_cut = false;
        let mut hit_in_tail = false;
        if n > MAX_LINE && !buf.ends_with(b"\n") {
            line_cut = true;
            match scan_to_eol(&mut reader, pattern, &buf) {
                Ok(hit) => hit_in_tail = hit,
                Err(_) => return true,
            }
        }
        while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
            buf.pop();
        }
        let line = String::from_utf8_lossy(&buf);
        if !line.contains(pattern) && !hit_in_tail {
            continue;
        }
        let (text, text_cut) = cap(&line, MAX_LINE);
        let cost = text.len() + rel.len() + 24; // 加上 JSON 结构的粗略开销
        if cost > *budget {
            return false;
        }
        *budget -= cost;
        matches.push(json!({
            "path": rel,
            "line": lineno,             // 1 起，和编辑器/grep 一致
            "text": text,
            "truncated": line_cut || text_cut,
        }));
    }
}

/// 把当前行剩下的部分读掉，同时**继续找** `needle`。只经过 `BufReader` 的缓冲区，外加
/// 一小段跨块用的尾巴（`needle.len() - 1` 字节），所以这一行有多长都不占内存。
///
/// `head` 是这一行已经读到的开头：拿它的尾巴给 `carry` 起头，**跨越截断点**的匹配才不
/// 会漏。返回"被丢掉的那一段里出现过 needle 没有"。
fn scan_to_eol<R: BufRead>(reader: &mut R, needle: &str, head: &[u8]) -> io::Result<bool> {
    let needle = needle.as_bytes();
    if needle.is_empty() {
        // 空 pattern 匹配一切，行首已经算命中了，剩下的只要跳过。
        return skip_to_eol(reader).map(|()| true);
    }
    let overlap = needle.len() - 1;
    let mut carry: Vec<u8> = head[head.len().saturating_sub(overlap)..].to_vec();
    let mut found = false;
    loop {
        let (done, used, hit) = {
            let buf = reader.fill_buf()?;
            if buf.is_empty() {
                return Ok(found); // EOF
            }
            let eol = buf.iter().position(|b| *b == b'\n');
            let slice = eol.map_or(buf, |i| &buf[..i]);
            let mut probe = std::mem::take(&mut carry);
            probe.extend_from_slice(slice);
            let hit = probe.windows(needle.len()).any(|w| w == needle);
            let keep = probe.len().saturating_sub(overlap);
            carry = probe[keep..].to_vec();
            (eol.is_some(), eol.map_or(buf.len(), |i| i + 1), hit)
        };
        found |= hit;
        reader.consume(used);
        if done {
            return Ok(found);
        }
    }
}

/// 把当前行剩下的部分读掉扔掉，只经过缓冲区，不攒内存。
fn skip_to_eol<R: BufRead>(reader: &mut R) -> io::Result<()> {
    loop {
        let (done, used) = {
            let buf = reader.fill_buf()?;
            if buf.is_empty() {
                return Ok(()); // EOF
            }
            match buf.iter().position(|b| *b == b'\n') {
                Some(i) => (true, i + 1),
                None => (false, buf.len()),
            }
        };
        reader.consume(used);
        if done {
            return Ok(());
        }
    }
}

fn recall_glob(sb: &Sandbox, args: &Value) -> ToolOutcome {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return ToolOutcome::err("recall_glob requires a string `pattern`");
    };
    // 和"缺 `pattern`"同一种拒法：一条 `ok:false` 回给模型，它自己改短再试。
    if pattern.len() > MAX_GLOB_PATTERN {
        return ToolOutcome::err(format!(
            "recall_glob `pattern` is too long ({} bytes, max {MAX_GLOB_PATTERN})",
            pattern.len()
        ));
    }
    let mut paths = Vec::new();
    let mut budget = MAX_TOOL_OUTPUT;
    let found = walk(sb, &sb.root);
    let mut truncated = found.truncated; // 遍历上限咬住了也要说（见 recall_grep）
    for file in found.files {
        let rel = sb.rel(&file);
        if !glob_match(pattern, &rel) {
            continue;
        }
        if paths.len() >= MAX_GLOB_MATCHES || rel.len() + 4 > budget {
            truncated = true;
            break;
        }
        budget -= rel.len() + 4;
        paths.push(Value::String(rel));
    }
    ToolOutcome::ok(json!({ "paths": paths, "truncated": truncated }))
}

/// 一次遍历的结果：沙箱内的文件（每个目录内按路径排序，所以同一次检索两次跑一样），
/// 外加"遍历有没有被上限砍掉"——砍掉了结果就是不完整的，调用方必须把这件事告诉模型。
struct Walk {
    files: Vec<PathBuf>,
    truncated: bool,
}

/// `base` 是文件时就是它自己。
fn walk(sb: &Sandbox, base: &Path) -> Walk {
    let mut w = Walk {
        files: Vec::new(),
        truncated: false,
    };
    if base.is_file() {
        w.files.push(base.to_path_buf());
        return w;
    }
    // 走过的规范路径。沙箱**内部**指向祖先的符号链接（`memory/latest -> .`）会让同一
    // 批文件被反复走到深度上限为止：模型于是把同一条笔记读成十几条互相独立的证据，
    // 预算也全花在重复上。规范路径去重一次就没了。
    let mut seen: HashSet<PathBuf> = HashSet::new();
    seen.insert(base.to_path_buf());
    walk_dir(sb, base, 0, &mut seen, &mut w);
    w
}

fn walk_dir(sb: &Sandbox, dir: &Path, depth: usize, seen: &mut HashSet<PathBuf>, w: &mut Walk) {
    if depth > MAX_WALK_DEPTH || w.files.len() >= MAX_WALK_FILES {
        w.truncated = true;
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for entry in entries {
        // 每个条目单独 canonicalize：沙箱里的一个符号链接可能指到外面去，遍历也要挡。
        let Ok(real) = entry.canonicalize() else {
            continue;
        };
        if !real.starts_with(&sb.root) {
            continue;
        }
        if !seen.insert(real.clone()) {
            continue;
        }
        if real.is_dir() {
            walk_dir(sb, &real, depth + 1, seen, w);
        } else if real.is_file() {
            w.files.push(real);
        }
        if w.files.len() >= MAX_WALK_FILES {
            w.truncated = true;
            return;
        }
    }
}

/// 能力名能不能直接拼成工具名：非空且只有 `[a-z0-9_]`。
fn is_projectable_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// 截断到 `limit` 字节，且不切坏 UTF-8。返回 `(内容, 是否截断)`。
pub(crate) fn cap(text: &str, limit: usize) -> (String, bool) {
    if text.len() <= limit {
        return (text.to_string(), false);
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

/// 极小的 glob：`*` 在一个路径段内匹配任意（可空）字符串，`**` 匹配任意多个段。
/// 够 `**/*.md`、`notes/*.txt` 这类用法；再复杂的用 `recall_grep`。
fn glob_match(pattern: &str, path: &str) -> bool {
    let pat: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let seg: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    seg_match(&pat, &seg)
}

/// 段一级的通配匹配，**线性**：两个游标 + 最后一个 `**` 的回退点，代价是
/// O(模式段数 × 路径段数)，和 `**` 有几个无关。
///
/// 这里原来是"`**` 吃掉 0..=n 个段，剩下的递归匹配"。那一版对连着的 `**` 会把同一个
/// 状态算无数遍——代价是 `C(n+m, m)`，18 个 `**` 对一条 13 段的路径要 6 秒，而
/// `recall_glob` 对**每一个**走到的文件都调一次它（最多 [`MAX_WALK_FILES`] 个）。模式
/// 是模型给的参数，而 `recall_*` 跑在 `spawn_blocking` 上——被 `spawn_blocking` 占住的
/// 线程**谁也 abort 不了**：interrupt、detach、`Drop`、内核的 turn 超时扫描全都只能看着
/// 它转。一个畸形的工具参数因此能在一个机器人控制进程里造出一次无法取消的挂起。
///
/// 这一版是经典的两游标通配匹配：遇到 `**` 就记下"最后一个星号在哪、它当时吃到哪"，
/// 后面失配时让它多吃一段再重来。每个 `(模式位置, 路径位置)` 至多访问一次。
fn seg_match(pat: &[&str], seg: &[&str]) -> bool {
    let (mut p, mut s) = (0usize, 0usize);
    // `star`：最后一个 `**` 在模式里的下标，以及它当时吃到的路径位置。
    let mut star: Option<(usize, usize)> = None;
    while s < seg.len() {
        if p < pat.len() && pat[p] == "**" {
            // 先让它吃 0 段；不够再回来多吃。连着的 `**` 因此只是把这个记号往后挪，
            // 不会各自展开一遍。
            star = Some((p, s));
            p += 1;
        } else if p < pat.len() && pat[p] != "**" && star_match(pat[p], seg[s]) {
            p += 1;
            s += 1;
        } else if let Some((sp, ss)) = star {
            p = sp + 1;
            s = ss + 1;
            star = Some((sp, s));
        } else {
            return false;
        }
    }
    // 路径吃完了：模式剩下的只能是 `**`（吃 0 段）。
    pat[p..].iter().all(|x| *x == "**")
}

/// 单个路径段的匹配：`*` 是任意（可空）子串，不跨 `/`。
fn star_match(pat: &str, s: &str) -> bool {
    let parts: Vec<&str> = pat.split('*').collect();
    if parts.len() == 1 {
        return pat == s;
    }
    let Some(rest) = s.strip_prefix(parts[0]) else {
        return false;
    };
    let mut rest = rest;
    for mid in &parts[1..parts.len() - 1] {
        if mid.is_empty() {
            continue;
        }
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    let last = parts[parts.len() - 1];
    rest.len() >= last.len() && rest.ends_with(last)
}

/// 下游 crate 的测试要一个 `TurnHandle` 和一张会记账的工具表。不是公开合约。
#[doc(hidden)]
pub mod testing {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use parking_lot::Mutex;

    /// 一个不连着内核的 `TurnHandle`：提案一律 `Refused { reason: "brains gone" }`。
    pub fn dummy_turn() -> TurnHandle {
        TurnHandle::detached_for_tests("test-brain")
    }

    /// `(调用号, "start" | "end")` 的记账本。
    pub type CallLog = Arc<Mutex<Vec<(String, String)>>>;

    /// 一张只有记账处理器的工具表。`decls` 每项是 `"<name>:safe"` 或 `"<name>:exclusive"`。
    ///
    /// 调用号 = 这张表被调用的次序（1 起）。`run_batch` 按模型顺序调用 handler，所以
    /// 它就等于那一批里 `ToolCall` 的下标 + 1——测试把调用编号成 "1","2",… 就能直接对上。
    pub fn recording_registry(decls: &[&str]) -> (ToolRegistry, CallLog) {
        let log: CallLog = Arc::new(Mutex::new(Vec::new()));
        let seq = Arc::new(AtomicUsize::new(0));
        let mut reg = ToolRegistry::new();
        for decl in decls {
            let (name, mode) = decl
                .split_once(':')
                .unwrap_or_else(|| panic!("decl must be `<name>:safe|exclusive`, got {decl:?}"));
            let concurrency_safe = match mode {
                "safe" => true,
                "exclusive" => false,
                other => panic!("unknown mode {other:?} in {decl:?}"),
            };
            reg.add(
                ToolSpec {
                    name: (*name).to_string(),
                    description: format!("recording {mode} tool {name}"),
                    parameters: json!({ "type": "object" }),
                    concurrency_safe,
                },
                Arc::new(Recorder {
                    log: log.clone(),
                    seq: seq.clone(),
                }),
            );
        }
        (reg, log)
    }

    struct Recorder {
        log: CallLog,
        seq: Arc<AtomicUsize>,
    }

    impl ToolHandler for Recorder {
        fn call(&self, args: Value, _turn: Option<&TurnHandle>) -> ToolFuture {
            // 编号在 `call` 里取（同步的那一刻），不是在 future 里：一组并行调用的
            // future 是按模型顺序造出来的，poll 的顺序则不保证。
            let id = (self.seq.fetch_add(1, Ordering::SeqCst) + 1).to_string();
            let log = self.log.clone();
            Box::pin(async move {
                log.lock().push((id.clone(), "start".into()));
                // 只为制造重叠：让出一次，同组的下一个 future 就能跑起来。不睡墙钟——
                // 顺序断言不该由时间长短决定。
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                log.lock().push((id.clone(), "end".into()));
                ToolOutcome::ok(json!({ "call": id, "args": args }))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oeh_admission::capability::RiskLevel;

    fn spec(name: &str, class: CapabilityClass, risk: RiskLevel, schema: Value) -> CapabilitySpec {
        CapabilitySpec {
            name: name.into(),
            class,
            risk_level: risk,
            params_schema: schema,
            lease_ttl_s: 1.0,
            overridable: false,
        }
    }

    #[test]
    fn projection_keeps_the_schema_and_spells_out_class_and_risk() {
        let mut r = ToolRegistry::new();
        let _ = r.project_capabilities(&[spec(
            "walk",
            CapabilityClass::Motion,
            RiskLevel::High,
            json!({"type":"object","properties":{"vx":{"type":"number"}}}),
        )]);
        let (s, _) = r.get("robot_walk").unwrap();
        assert_eq!(s.description, "motion/high capability walk");
        assert_eq!(s.parameters["properties"]["vx"]["type"], "number");
    }

    #[test]
    fn a_non_object_schema_becomes_an_empty_object_schema() {
        let mut r = ToolRegistry::new();
        let _ = r.project_capabilities(&[spec(
            "stop",
            CapabilityClass::Control,
            RiskLevel::None,
            Value::Null,
        )]);
        let (s, _) = r.get("robot_stop").unwrap();
        assert_eq!(s.parameters, json!({"type":"object"}));
        assert!(s.concurrency_safe);
    }

    #[tokio::test]
    async fn a_capability_tool_reports_the_refusal_verbatim() {
        let mut r = ToolRegistry::new();
        let _ = r.project_capabilities(&[spec(
            "say",
            CapabilityClass::Speech,
            RiskLevel::Low,
            json!({"type":"object"}),
        )]);
        let turn = testing::dummy_turn();
        let (_, h) = r.get("robot_say").unwrap();
        let out = h.call(json!({"text": "hi"}), Some(&turn)).await;
        assert!(!out.ok);
        // 结局原样回给模型：内部 tag `outcome` + 理由。
        assert_eq!(out.result["outcome"], "refused");
        assert_eq!(out.result["reason"], "brains gone");
    }

    #[tokio::test]
    async fn ask_slow_brain_needs_a_question() {
        let mut r = ToolRegistry::new();
        r.add_ask_slow_brain();
        let turn = testing::dummy_turn();
        let (_, h) = r.get("ask_slow_brain").unwrap();
        assert!(!h.call(json!({}), Some(&turn)).await.ok);
    }

    #[tokio::test]
    async fn an_unknown_tool_is_an_error_not_a_panic() {
        let r = ToolRegistry::new();
        let turn = testing::dummy_turn();
        let outs = r
            .run_batch(
                vec![ToolCall {
                    id: "1".into(),
                    name: "robot_nope".into(),
                    args: json!({}),
                }],
                Some(&turn),
            )
            .await;
        assert!(!outs[0].1.ok);
        assert_eq!(outs[0].1.result["error"], "unknown tool: robot_nope");
    }

    #[test]
    fn glob_matches_stars_within_a_segment_and_double_stars_across() {
        assert!(glob_match("*.md", "notes.md"));
        assert!(!glob_match("*.md", "sub/notes.md"));
        assert!(glob_match("**/*.md", "sub/deep/notes.md"));
        assert!(glob_match("**/*.md", "notes.md"));
        assert!(glob_match("sub/*/notes.md", "sub/deep/notes.md"));
        assert!(!glob_match("sub/*.md", "other/notes.md"));
        assert!(glob_match("**", "a/b/c"));
        assert!(glob_match("n*t*s.md", "notes.md"));
        assert!(!glob_match("n*t*z.md", "notes.md"));
    }

    /// C1：18 个连着的 `**`。旧的递归匹配器在这条路径上要 6 秒（`C(n+m, m)` 次调用），
    /// 而 `recall_glob` 会对每一个走到的文件都调一次——一次模型给的模式就能把一个
    /// **不可 abort** 的 `spawn_blocking` 线程占到天荒地老。
    ///
    /// 判别信号只能是耗时，所以余量给到四个数量级：线性匹配器在这里是微秒级。
    #[test]
    fn a_pathological_double_star_pattern_returns_promptly() {
        let pattern = format!("{}/notes.md", vec!["**"; 18].join("/"));
        let deep = "a/b/c/d/e/f/g/h/i/j/k/l/notes.md";
        let t = std::time::Instant::now();
        // 命中与不命中两条路都走一遍：指数爆炸最狠的恰恰是**失配**那一条。
        assert!(glob_match(&pattern, deep));
        assert!(!glob_match(&pattern, "a/b/c/d/e/f/g/h/i/j/k/l/notes.txt"));
        let dt = t.elapsed();
        assert!(dt < std::time::Duration::from_millis(100), "{dt:?}");
    }

    #[test]
    fn cap_never_splits_a_utf8_char() {
        let (s, cut) = cap("日本語", 4);
        assert!(cut);
        assert_eq!(s, "日");
        assert_eq!(cap("abc", 8), ("abc".to_string(), false));
    }

    #[tokio::test]
    async fn recall_output_is_capped_and_flagged() {
        let dir = tempfile::tempdir().unwrap();
        // 1 MiB：上限必须在**读之前**生效，返回的内容正好 16 KiB。
        let big = "x".repeat(1024 * 1024);
        std::fs::write(dir.path().join("big.txt"), &big).unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(dir.path().to_path_buf());
        let turn = testing::dummy_turn();
        let (_, h) = r.get("recall_read").unwrap();
        let out = h.call(json!({"path": "big.txt"}), Some(&turn)).await;
        assert!(out.ok);
        assert_eq!(out.result["truncated"], true);
        assert_eq!(
            out.result["content"].as_str().unwrap().len(),
            MAX_TOOL_OUTPUT
        );
    }

    #[tokio::test]
    async fn a_symlink_out_of_the_sandbox_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "shh").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            dir.path().join("link.txt"),
        )
        .unwrap();
        std::fs::write(dir.path().join("own.txt"), "mine").unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(dir.path().to_path_buf());
        let turn = testing::dummy_turn();
        let (_, h) = r.get("recall_read").unwrap();
        assert!(!h.call(json!({"path": "link.txt"}), Some(&turn)).await.ok);
        // 遍历也不跟出去：grep 只能看见沙箱里自己的文件。
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "shh"}), Some(&turn)).await;
        assert_eq!(out.result["matches"].as_array().unwrap().len(), 0);
        let (_, gl) = r.get("recall_glob").unwrap();
        let out = gl.call(json!({"pattern": "*.txt"}), Some(&turn)).await;
        assert_eq!(out.result["paths"], json!(["own.txt"]));
    }

    // ---- 复查第 1 轮 ----------------------------------------------------

    #[tokio::test]
    async fn recall_grep_caps_a_huge_file_without_slurping_it() {
        let dir = tempfile::tempdir().unwrap();
        // ~1 MiB，每一行都命中：输出预算必须在 16 KiB 处停下并说自己截断了。
        let mut text = String::new();
        for i in 0..30_000 {
            text.push_str(&format!("needle on line {i} with a bit of padding\n"));
        }
        assert!(text.len() > 1024 * 1024);
        std::fs::write(dir.path().join("big.log"), &text).unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(dir.path().to_path_buf());
        let turn = testing::dummy_turn();
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "needle"}), Some(&turn)).await;
        assert!(out.ok);
        assert_eq!(out.result["truncated"], true);
        let matches = out.result["matches"].as_array().unwrap();
        assert!(matches.len() < 30_000, "budget must bite");
        assert_eq!(matches[0]["line"], 1);
        let text_bytes: usize = matches
            .iter()
            .map(|m| m["text"].as_str().unwrap().len())
            .sum();
        assert!(text_bytes <= MAX_TOOL_OUTPUT, "{text_bytes} > cap");
    }

    #[tokio::test]
    async fn a_match_past_the_line_cap_is_still_reported() {
        let dir = tempfile::tempdir().unwrap();
        // 一条 1 MiB 的单行，needle 在最后：行首之外的部分不进内存，但**照样要算**。
        let mut line = "z".repeat(1024 * 1024);
        line.push_str("needle\n");
        std::fs::write(dir.path().join("one.log"), &line).unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(dir.path().to_path_buf());
        let turn = testing::dummy_turn();
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "needle"}), Some(&turn)).await;
        let matches = out.result["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["line"], 1);
        assert_eq!(matches[0]["truncated"], true);
        assert_eq!(matches[0]["text"].as_str().unwrap().len(), MAX_LINE);
    }

    #[tokio::test]
    async fn a_walk_cut_short_by_the_depth_limit_says_truncated() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("top.md"), "shallow").unwrap();
        let mut deep = dir.path().to_path_buf();
        for i in 0..(MAX_WALK_DEPTH + 2) {
            deep = deep.join(format!("d{i}"));
        }
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("deep.md"), "buried").unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(dir.path().to_path_buf());
        let turn = testing::dummy_turn();
        let (_, gl) = r.get("recall_glob").unwrap();
        let out = gl.call(json!({"pattern": "**/*.md"}), Some(&turn)).await;
        // 浅的那个看得见，深的那个没走到——所以结果**不完整**，必须说出来。
        assert_eq!(out.result["paths"], json!(["top.md"]));
        assert_eq!(out.result["truncated"], true);
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "buried"}), Some(&turn)).await;
        assert_eq!(out.result["matches"].as_array().unwrap().len(), 0);
        assert_eq!(out.result["truncated"], true);
    }

    #[tokio::test]
    async fn an_in_sandbox_symlink_loop_does_not_multiply_results() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/note.md"), "one note\n").unwrap();
        // `sub/latest -> .`：沙箱内的环，去重之前会把 note.md 走上十几遍。
        std::os::unix::fs::symlink(dir.path().join("sub"), dir.path().join("sub/latest")).unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(dir.path().to_path_buf());
        let turn = testing::dummy_turn();
        let (_, gl) = r.get("recall_glob").unwrap();
        let out = gl.call(json!({"pattern": "**/*.md"}), Some(&turn)).await;
        assert_eq!(out.result["paths"], json!(["sub/note.md"]));
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "one note"}), Some(&turn)).await;
        assert_eq!(out.result["matches"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_second_add_recall_keeps_the_first_root() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        std::fs::write(first.path().join("a.md"), "from the first root").unwrap();
        std::fs::write(second.path().join("b.md"), "from the second root").unwrap();
        let mut r = ToolRegistry::new();
        r.add_recall(first.path().to_path_buf());
        r.add_recall(second.path().to_path_buf()); // 记一条 error，什么都不改
        let turn = testing::dummy_turn();
        let (_, h) = r.get("recall_read").unwrap();
        assert!(h.call(json!({"path": "a.md"}), Some(&turn)).await.ok);
        assert!(!h.call(json!({"path": "b.md"}), Some(&turn)).await.ok);
    }

    #[test]
    fn a_capability_name_that_cannot_be_a_tool_name_is_skipped() {
        let mut r = ToolRegistry::new();
        let skipped = r.project_capabilities(&[
            spec(
                "walk.fast",
                CapabilityClass::Motion,
                RiskLevel::Low,
                json!({"type":"object"}),
            ),
            spec(
                "walk",
                CapabilityClass::Motion,
                RiskLevel::Low,
                json!({"type":"object"}),
            ),
        ]);
        // 带点的那个整批请求都会被厂商 400 掉，所以它根本不上桌——而且它的名字要被
        // **报出来**：投影现在每个 turn 都可能重来一次，谁调用谁决定这条告警说几遍。
        assert_eq!(skipped, vec!["walk.fast".to_string()]);
        assert_eq!(
            r.specs().iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            vec!["robot_walk"]
        );
        assert!(is_projectable_name("static_pose"));
        assert!(is_projectable_name("look_at2"));
        assert!(!is_projectable_name(""));
        assert!(!is_projectable_name("Walk"));
        assert!(!is_projectable_name("walk fast"));
        assert!(!is_projectable_name("walk-fast"));
    }

    struct PanicInFuture;
    impl ToolHandler for PanicInFuture {
        fn call(&self, _args: Value, _turn: Option<&TurnHandle>) -> ToolFuture {
            Box::pin(async { panic!("boom in the future") })
        }
    }

    struct PanicInCall;
    impl ToolHandler for PanicInCall {
        fn call(&self, _args: Value, _turn: Option<&TurnHandle>) -> ToolFuture {
            panic!("boom in call")
        }
    }

    /// 注意：这个测试会在 stderr 上打两条 panic 记录，那是预期的——被接住不等于不记账。
    #[tokio::test]
    async fn a_panicking_tool_does_not_take_down_the_batch() {
        fn tool(name: &str, safe: bool) -> ToolSpec {
            ToolSpec {
                name: name.into(),
                description: "test".into(),
                parameters: json!({"type":"object"}),
                concurrency_safe: safe,
            }
        }
        let (mut r, _log) = testing::recording_registry(&["robot_say:safe"]);
        r.add(tool("boom_future", true), Arc::new(PanicInFuture));
        r.add(tool("boom_call", true), Arc::new(PanicInCall));
        r.add(tool("boom_exclusive", false), Arc::new(PanicInFuture));
        let call = |id: &str, name: &str| ToolCall {
            id: id.into(),
            name: name.into(),
            args: json!({}),
        };
        let outs = r
            .run_batch(
                vec![
                    call("1", "boom_future"),
                    call("2", "robot_say"),
                    call("3", "boom_call"),
                    call("4", "boom_exclusive"),
                    call("5", "robot_say"),
                ],
                Some(&testing::dummy_turn()),
            )
            .await;
        assert_eq!(
            outs.iter().map(|(c, _)| c.id.as_str()).collect::<Vec<_>>(),
            ["1", "2", "3", "4", "5"]
        );
        for i in [0, 2, 3] {
            assert!(!outs[i].1.ok);
            assert_eq!(outs[i].1.result["error"], "tool panicked");
        }
        // 同一批里的好工具照常有结果——模型不会看到有 tool_call 没有 tool 消息。
        assert!(outs[1].1.ok && outs[4].1.ok);
    }
}
