//! review fork：一个 turn 结束之后（或者会话结束时）跑的维护相位。
//!
//! 它是这个脑唯一能**改自己**的地方，所以它的形状全部由"改不动什么"决定：
//!
//! - **它没有世界**。review 用的是一张**自己的** [`ToolRegistry`]，里面只有
//!   `skill_manage` / `memory_manage` / `read`。没有 `robot_*`、没有 `say`、没有
//!   `ask_*`、连 `recall_*` 都没有。规格里"review 提的动作一律 auto-deny"这条约束在这
//!   里不是一条审批规则，而是一张**根本没有那些工具的工具表**——审批规则可以被配置改
//!   掉，一张不存在的工具表不行。
//! - **它不递归**。review 里不会再触发 review：触发点只有 `run_turn` 的收尾和 detach
//!   （见 [`super::Shared::after_turn`] / [`super::NativeBrain`] 的 `Drop`），而这两处
//!   都不在 review 里面。
//! - **它出错不带走 turn**。任何一种失败（模型流断、工具超时、panic）都只变成
//!   `brain/review{error}`，触发它的那个 turn 照常 `done`。维护是锦上添花，不是交付
//!   物；让一次记笔记的失败把一个已经做完的 turn 记成失败，是把因果关系写反了。
//!
//! 跑在 `Maintenance` 相位里。进出由 [`super::MaintenanceGuard`] 这一个 RAII 守卫配
//! 对——**两边都由它管**：只进不出会让这个脑永远只 latch 不开 turn，只出不进（守卫写
//! 成一半时真的发生过）会让整段 review 留在 `Running` 里，于是
//! `Brains::sweep_timeouts` 够得着它，一个已经做完 step 循环、只是在记笔记的 turn 会被
//! 强行关掉，留下一条假的 `brain/error{reason:"turn_timeout"}`。
//!
//! 对唤醒来说 `Maintenance` 等价于 Running（只 latch，不重复开 turn），但**不受内核的
//! turn 超时约束**——review 的几步模型调用不该让一个已经做完的 turn 被超时扫描记成
//! `brain/error`。这条性质有 e2e 钉着（`tests/rsi_review.rs` 里那个 `turn_timeout_ms`
//! 非 0 的用例），不是只写在注释里。
//!
//! 代价是这一段必须**自己有界**，而且那道界不能是可选的：内核的 `turn_timeout_ms`
//! 兜底只扫 `Running` 的脑，所以一次永远不返回的模型调用会把 turn 挂在
//! `Maintenance` 上——没有 `done`、没有 `brain/error`、此后所有唤醒只 latch 不开
//! turn，只有一次定向 `interrupt` 救得回来。因此 [`ReviewConfig::timeout_ms`] 是
//! **整段 review 的硬上限**，不允许配成 0，也不看 `step_timeout_ms` 的脸色
//! （后者只有非 0 时才额外管住单步）。
//!
//! 同样的理由，这一段也必须**自己接住 panic**：它跑在 `run_turn` 那层
//! `catch_unwind` 的外面（那层只包 step 循环），一个漏出去的 panic 会同时吃掉
//! `leave_maintenance()` 和 `done`。见 [`super::Shared::review`]。
//!
//! 写完之后 system 前缀重建（`# Skills` 段变化），于是下一个 turn 的 prompt 缓存自然
//! 失效——这是**有意**的一次失效，不是漏掉的一次稳定：模型必须看得见它刚给自己写下的
//! 东西。

use std::sync::Arc;

use oeh_kernel::session_log::SessionEntry;
use parking_lot::Mutex;
use serde_json::{json, Value};

use super::{args_bytes, by, prompt, review_prompt, Shared, StepOut};
use crate::model::{Message, ModelRequest, StopReason};
use crate::rsi::{Kind, KnowledgeStore};

/// review 的触发与预算。
///
/// `every_turns: 0` = 关掉周期触发（`on_session_end` 不受影响）。这是"关掉"的唯一写
/// 法：一个 `Option<u32>` 会让配置里出现 `null` 和 `0` 两种等价写法，而它们在 YAML 里
/// 看起来完全不像同一件事。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewConfig {
    /// 每这么多个 turn 跑一次 review（在**第 N 个** turn 的 `done` 之前）。0 = 关。
    pub every_turns: u32,
    /// detach 时，如果距上次 review 还有 ≥ 1 个 turn，再补一次。
    pub on_session_end: bool,
    /// 一次 review 最多几个 step（一个 step = 一次模型调用 + 它要求的那一批工具）。
    pub max_steps: u32,
    /// **整段** review 的墙上时限。到点就把整个 review 丢掉，记一条
    /// `brain/review{error: "timeout"}`，然后 `leave_maintenance()` + turn 照常 `done`。
    ///
    /// **不能是 0**（配置层直接拒绝激活，见 `builtin::native_brain`）。这里没有"不设
    /// 限"这个选项，理由见模块头：维护相位不在内核超时扫描的范围里，所以"不设限"的真
    /// 实含义是"一次卡住的模型调用可以永久冻住这个脑"，那不是任何人想配的东西。
    /// `max_steps` 和 `step_timeout_ms` 仍然各管各的（步数、单步），这一条管总量。
    pub timeout_ms: u64,
}

impl Default for ReviewConfig {
    fn default() -> Self {
        ReviewConfig {
            every_turns: 10,
            on_session_end: true,
            max_steps: 4,
            timeout_ms: 30_000,
        }
    }
}

/// 一次 review 的结局。`error` 有值不代表什么都没写成——流可能断在第三步，而前两步的写
/// 已经落盘入账了。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReviewReport {
    pub steps: u32,
    /// 真正写成功的 rel，去重、按首次出现排序。
    pub writes: Vec<String>,
    pub error: Option<String>,
}

/// 两个写工具的名字。`writes` 只数它们——`read` 成功也会回一个 `rel`。
const WRITE_TOOLS: [&str; 2] = ["skill_manage", "memory_manage"];

/// 这一段的模型调用在世界日志里的相位标记。普通 step 不带这个字段，所以
/// `brain/request-error{step: 0}` 到底来自 turn 的第一步还是维护相位的第一步，看一眼
/// 就知道——两者要查的地方完全不同。
pub(super) const PHASE: &str = "review";

/// 跑一次 review。
///
/// **没有 `TurnHandle`**。维护相位跑在 turn 的终态之后，会话末那一次干脆跑在 detach
/// 之后，而这张表里的三个工具一个都不提案——所以 `run_batch` 收到的是 `None`
/// （见 [`crate::tools::ToolHandler`]）。这同时是"review 提不了案"的第三重结构性保证。
///
/// `trace` 是**调用方给的**：会话末那一次没有句柄可读，但这次 review 在日志上仍然应该
/// 挂在它总结的那些 turn 的最后一个 trace 上（见 [`super::NativeBrain`] 的 `Drop`）。
///
/// `report` 是**一边跑一边填的**，不是返回值：整段 review 外面还套着一个
/// [`ReviewConfig::timeout_ms`] 和一个 `catch_unwind`（见 [`super::Shared::review`]），
/// 而这两种收场都会把这个 future 整个丢掉。已经落盘入账的那几次写必须还能被报出来——
/// 否则一次超时会让 `brain/review.writes` 谎称什么都没写，而且 `# Skills` 前缀不会重
/// 建，模型下一个 turn 看不见自己刚写下的 skill。
///
/// **不写会话日志**：`SessionEntry` 的 `step` 是 turn 内的步号，review 不是 turn 的
/// step，混进去之后没人分得清第 0 步是这个 turn 的第一步还是那次 review 的第一步。更
/// 要紧的是 `recent` 正是从会话记录来的——review 记下自己的输出，等于让下一次 review
/// 把自己上一次的思考当成"最近发生的事"。review 留下的痕迹在账本和 `brain/review` 里。
pub(super) async fn run_review(
    s: &Shared,
    store: Arc<KnowledgeStore>,
    cfg: &ReviewConfig,
    trace: &str,
    recent: &[SessionEntry],
    report: &Mutex<ReviewReport>,
) {
    // 前缀在整个 review 里固定：几个 step 之间缓存纪律和普通 turn 一样适用，而 skill
    // 索引在这中间变化恰恰是最贵的一种变化（每一步都重新计费）。模型知道自己写了什么
    // ——工具的返回值说得清清楚楚。
    let skills = skill_index(&store);
    let system = prompt::system_prefix(review_prompt::SYSTEM, &s.review_specs, &skills);
    let mut messages = vec![Message::User {
        content: review_prompt::user(recent),
    }];
    let max_steps = u64::from(cfg.max_steps.max(1));
    let mut step: u64 = 0;
    loop {
        let req = ModelRequest {
            system: system.clone(),
            messages: messages.clone(),
            tools: s.review_specs.clone(),
            max_tokens: s.cfg.max_tokens,
            temperature: s.cfg.temperature,
        };
        let deadline = s.deadline();
        let StepOut { out, err, logged } = s.complete_in(PHASE, req, trace, step, deadline).await;
        report.lock().steps = (step + 1) as u32;
        if let Some(e) = err {
            // 流已经在吐东西然后断掉的那种失败，瀑布不会记（它只管"调用根本没成功"）。
            // 和 step 循环里逐字同形的一条，外加 `phase`：维护相位不记
            // `brain/request`，少了这条断流记录，`brain/review{error}` 就是这次失败在日
            // 志里的全部，而那句话说不出"断之前模型已经说了 3 KB、还要求了两次写"。
            if !logged {
                s.emit_at(
                    trace,
                    "brain/request-error",
                    0,
                    json!({
                        "reason": e.to_string(),
                        "phase": PHASE,
                        "step": step,
                        "partial_chars": out.text.chars().count(),
                        "partial_tool_calls": out.calls.len(),
                    }),
                );
            }
            // 整条 fallback 链都失败（或者流断在半路）。已经写成的那几条留在 `writes`
            // 里：它们确实落盘了，账本上也有。
            report.lock().error = Some(e.to_string());
            return;
        }
        if out.calls.is_empty() || out.stop == StopReason::EndTurn {
            return;
        }
        // 记账在执行之前，和 step 循环同一条理由：一次卡住的写之后，日志里必须留得下
        // "模型确实要求过这次调用"。
        for c in &out.calls {
            s.emit_at(
                trace,
                "brain/tool_call",
                0,
                json!({ "name": c.name, "args_bytes": args_bytes(&c.args) }),
            );
        }
        let pending: Vec<String> = out.calls.iter().map(|c| c.name.clone()).collect();
        let results = match by(deadline, s.review_tools.run_batch(out.calls, None)).await {
            Ok(r) => r,
            Err(_) => {
                for name in &pending {
                    s.emit_at(
                        trace,
                        "brain/tool_result",
                        0,
                        json!({ "name": name, "ok": false, "error": "timeout" }),
                    );
                }
                report.lock().error = Some("step timeout: tools".into());
                return;
            }
        };
        for (c, o) in &results {
            let mut payload = json!({ "name": c.name, "ok": o.ok });
            if !o.ok {
                // 失败的理由进世界日志（这里的失败全是 `RsiError` 的 `Display`：路径
                // 越界、陈旧读、不是你建的）。普通 step 的 `brain/tool_result` 不带它，
                // 因为那边的失败理由已经在 `act/refused` 里有一条自己的链；这边没有。
                payload["error"] = Value::String(err_text(&o.result));
            }
            s.emit_at(trace, "brain/tool_result", 0, payload);
            if o.ok && WRITE_TOOLS.contains(&c.name.as_str()) {
                if let Some(rel) = o.result.get("rel").and_then(Value::as_str) {
                    let mut g = report.lock();
                    if !g.writes.iter().any(|w| w == rel) {
                        g.writes.push(rel.to_string());
                    }
                }
            }
        }
        messages.push(Message::Assistant {
            content: out.text,
            tool_calls: results.iter().map(|(c, _)| c.clone()).collect(),
        });
        for (c, o) in results {
            messages.push(Message::Tool {
                call_id: c.id,
                content: serde_json::to_string(&o.result)
                    .unwrap_or_else(|e| format!("{{\"error\":\"unserialisable result: {e}\"}}")),
            });
        }
        step += 1;
        if step >= max_steps {
            return;
        }
    }
}

/// `# Skills` 段的素材：每个 `skills/**/SKILL.md` 的 rel 和它的第一行。
///
/// 读不动的文件标题留空而不是整条丢掉：一个存在但读不出来的 skill，模型至少该知道它
/// 在那里（否则它会新建一个同名的，然后撞上 `NotOwner`）。
///
/// **同步 I/O**：调用方负责把它放进 `spawn_blocking`（见
/// [`super::Shared::finish_review`]），或者在构造期调用（那时还没有 runtime）。
pub(crate) fn skill_index(store: &KnowledgeStore) -> Vec<(String, String)> {
    let Ok(rels) = store.list(Kind::Skill) else {
        return Vec::new();
    };
    rels.into_iter()
        .map(|rel| {
            let title = store
                .read(&rel)
                .ok()
                .and_then(|d| d.text.lines().next().map(str::to_string))
                .unwrap_or_default();
            (rel, title)
        })
        .collect()
}

/// 一条失败结果里的那句话。[`crate::tools::ToolOutcome::err`] 的形状永远是
/// `{"error": "…"}`。
fn err_text(v: &Value) -> String {
    v.get("error")
        .and_then(Value::as_str)
        .unwrap_or("unknown error")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_matches_the_documented_numbers() {
        let d = ReviewConfig::default();
        assert_eq!(d.every_turns, 10);
        assert!(d.on_session_end);
        assert_eq!(d.max_steps, 4);
    }

    #[test]
    fn only_the_two_write_tools_count_as_writes() {
        assert!(WRITE_TOOLS.contains(&"skill_manage"));
        assert!(WRITE_TOOLS.contains(&"memory_manage"));
        assert!(!WRITE_TOOLS.contains(&"read"));
    }

    #[test]
    fn a_result_without_an_error_field_still_says_something() {
        assert_eq!(err_text(&json!({"error": "bad path"})), "bad path");
        assert_eq!(err_text(&json!({})), "unknown error");
    }

    /// 空 store 的索引是空的；写进去一个 skill 之后它带着第一行出现。
    #[test]
    fn the_skill_index_carries_each_skills_first_line() {
        let tmp = tempfile::tempdir().unwrap();
        let store =
            KnowledgeStore::open(tmp.path(), crate::rsi::ProtectedPaths::default(), None).unwrap();
        assert!(skill_index(&store).is_empty());
        store
            .write(
                "skills/nav/SKILL.md",
                "# nav\n\nbody\n",
                "review:fast",
                None,
            )
            .unwrap();
        assert_eq!(
            skill_index(&store),
            vec![("skills/nav/SKILL.md".to_string(), "# nav".to_string())]
        );
    }
}
