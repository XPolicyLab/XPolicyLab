//! review fork 看到的两段文字：一段固定的 system 指令，和一段"这几个 turn 发生了什么"
//! 的 user message。
//!
//! 为什么指令写成**偏好顺序**而不是"请把有用的东西记下来"：一个只被告知"记下有用的
//! 东西"的模型会不停地新建 skill，因为新建永远比"找到那条已经存在的、把它改对"容易。
//! 三个月以后那摞 `skills/` 里会有四十个互相覆盖的文件，而检索它们的代价（以及它们在
//! 每个 turn 的 system 前缀里占的位置，见 [`super::prompt::system_prefix`]）都由**此后
//! 每一次**模型调用买单。所以顺序是硬性的：先改你刚用过的那一个，再改已有的伞，再加
//! 支持文件，实在没有才新建。
//!
//! 为什么有一张 DO-NOT-CAPTURE 清单：RSI 的失败模式不是"学不到东西"，是**学到错的
//! 东西并从此相信它**。三类噪音最容易被误当成知识：
//!
//! 1. 环境相关的失败（这台机器上的一条路径、一次网络抖动、一个没插好的设备）——它们
//!    在下一次运行里根本不成立，写下来只会让模型绕开一条其实通的路；
//! 2. "某能力不能用"这种负面断言——能力面是动态的（执行器插件随时能注册/注销），一条
//!    写死的"X 用不了"会在 X 重新挂上之后继续生效，而没有任何东西会去撤销它；
//! 3. 未解决的尝试链——"我试了 A、又试了 B、还是不行"不是知识，是一份没有结论的日志；
//!    下一个 turn 读到它只会把同样的弯路再走一遍。
//!
//! **prompt 全文不进世界日志**（和 [`super::prompt`] 同一条纪律）：这里的文本只出现在
//! 模型请求里和会话日志里。
use oeh_kernel::session_log::SessionEntry;
use serde_json::Value;

/// review fork 的 system 指令（人格位）。它会被
/// [`super::prompt::system_prefix`] 接上工具索引与 `# Skills` 段——review 用的是和普通
/// turn **同一个**前缀渲染器，因为缓存纪律对它一样适用：一次 review 里的几个 step 之间，
/// 前缀必须逐字节相同。
pub const SYSTEM: &str = "\
You are the maintenance pass of a robot's brain. The turns you are shown have already \
happened; you cannot act on the world from here. You have exactly three tools: `read`, \
`skill_manage` and `memory_manage`. There is no way to move, speak, or ask anyone \
anything, and nothing you do here starts another maintenance pass.

Your one job: decide whether anything in the recent activity is worth writing down so \
that a future turn does better, and if so write it down.

# Where to put it

Follow this order and stop at the first option that fits:

1. Update the skill that was already loaded for this work. Sharpening the file that was \
   actually in play is almost always the right move.
2. Otherwise, update an existing umbrella skill that already covers this area. Look at \
   the skill index above and `read` the candidate before you decide.
3. Otherwise, add a supporting file under an existing skill \
   (`skills/<existing>/<topic>/SKILL.md`), so the umbrella keeps its shape.
4. Only if none of the above fits, create a new class-level skill \
   (`skills/<class>/SKILL.md`). A new top-level skill is a claim that a whole class of \
   work was missing; make that claim rarely.

Durable facts about this robot, its people and its place go in memory instead: \
`memory/MEMORY.md` for the short standing list, `memory/topics/<name>.md` for one \
subject that has outgrown it.

# DO NOT capture

- Environment-specific failures: a path that only exists on this machine, one flaky \
  device, a single network error, anything that would not be true on the next boot.
- Negative assertions of the form \"capability X does not work\" or \"X is not \
  supported\". The capability surface changes between runs; a written-down `no` \
  outlives the reason for it and nothing will ever come back to retract it.
- Unresolved attempt chains: \"I tried A, then B, then C and it still failed\". A list \
  of dead ends without a working answer is a log, not knowledge.

If nothing clears that bar, write nothing and say so in one line. Writing nothing is a \
good outcome.

# How to write

- Every file is Markdown. The **first line is the title** and it is the only line other \
  turns see in their skill index, so make it say what the skill is for.
- Be concrete and reusable: the shape of the situation, what to do, what to watch out \
  for. No timestamps, no run-specific ids, no transcripts.
- Read before you write. `skill_manage{op: \"update\"}` and \
  `memory_manage{op: \"update\"}` require the `expect_sha256` you got from `read` (or \
  from the write that created the file); without it the write is refused as a stale \
  read. `memory_manage{op: \"append\"}` re-reads for you.
- You can only write `skills/**/SKILL.md`, `memory/MEMORY.md` and \
  `memory/topics/*.md`. Everything else is refused in code, not by politeness.";

/// 一次 review 的 user message：上次 review 之后这几个 turn 里发生了什么，**最新的在
/// 最前面**。
///
/// 为什么倒序：这段文字有预算（[`MAX_RECENT_BYTES`]），超了就从**尾巴**（也就是最旧的
/// 那几条）开始丢。最近发生的事情是最可能值得记下来的，让它们去竞争一个可能被截断的
/// 位置是把优先级写反了。
///
/// 为什么只有三种条目：`ModelOutput` / `ToolCall` / `ToolResult` 就是"这个脑做了什么、
/// 世界回了什么"。观测（`SessionEntry::Observation`）不在里面——它是唤醒的原始素材，
/// 一份就能占满整个预算，而 review 要看的是**行为**，不是当时的传感器读数。
pub fn user(recent: &[SessionEntry]) -> String {
    const HEADER: &str = "# Recent activity (newest first)\n";
    if recent.is_empty() {
        return format!("{HEADER}(nothing happened since the last maintenance pass)\n");
    }
    // 先渲染再裁：`render` 对它不关心的 `SessionEntry` 变体回 `None`，而 "omitted" 那个
    // 数必须只数**渲染得出来的**那些——把模型从来看不见的变体也算进去，等于告诉它有一
    // 段它够不着的历史。今天 `push_recent` 只存三种可渲染的变体，所以先前那个
    // `recent.len() - lines.len()` 恰好是对的，但那是巧合，不是不变量。
    //
    // 条数上限是 [`super::MAX_RECENT_ENTRIES`]（256），单条上限 1 KiB，所以这一遍最多
    // 是几百 KB 的临时字符串。
    let rendered: Vec<String> = recent.iter().rev().filter_map(render).collect();
    let mut used = HEADER.len();
    let mut kept = 0usize;
    for l in &rendered {
        // 至少留一条：一个只有标题的 review 提示比没有还糟。
        // `break` 而不是 `continue`：这是一个**连续**的窗口（最近的 N 条），跳过一条大
        // 的再收一条更老的小的，会在时间线中间留一个模型看不见的洞。
        if kept > 0 && used + l.len() > MAX_RECENT_BYTES {
            break;
        }
        used += l.len();
        kept += 1;
    }
    let dropped = rendered.len() - kept;
    let mut s = String::with_capacity(used + 48);
    s.push_str(HEADER);
    for l in &rendered[..kept] {
        s.push_str(l);
    }
    if dropped > 0 {
        s.push_str(&format!("[… {dropped} older entries omitted]\n"));
    }
    s
}

/// 一次 review 的 user message 的字节上限（16 KiB）。
///
/// 和 [`super::prompt::MAX_INBOX_BYTES`] 同一个理由：这段文字会在 review 的**每一个
/// step** 上重复计费，而它的长度由"上次 review 之后这个脑有多忙"决定——也就是完全不
/// 受这里控制。
pub const MAX_RECENT_BYTES: usize = 16 * 1024;

/// 单条记录的字节上限（1 KiB）。一次 `recall_grep` 的结果或者一段长输出不该独占整个
/// 预算，把别的 turn 全挤出去。
const MAX_ENTRY_BYTES: usize = 1024;

/// 一条会话记录渲染成一行。三种之外的返回 `None`（不是这个 review 关心的东西）。
fn render(e: &SessionEntry) -> Option<String> {
    let line = match e {
        SessionEntry::ModelOutput { turn, step, text } => {
            format!("[t{turn} s{step}] said: {}", one_line(text))
        }
        SessionEntry::ToolCall {
            turn,
            step,
            call_id,
            name,
            args,
        } => format!("[t{turn} s{step}] call {call_id} {name} {}", compact(args)),
        SessionEntry::ToolResult {
            turn,
            step,
            call_id,
            ok,
            result,
        } => format!(
            "[t{turn} s{step}] result {call_id} ok={ok} {}",
            compact(result)
        ),
        _ => return None,
    };
    let mut line = clip(&line, MAX_ENTRY_BYTES);
    line.push('\n');
    Some(line)
}

/// 换行压成 `⏎`：一条记录占一行是这段渲染唯一的结构，模型输出里的换行会把它打碎。
fn one_line(s: &str) -> String {
    s.replace('\n', " ⏎ ")
}

fn compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "<unserialisable>".into())
}

/// 按字节截断，切点落在字符边界上。
fn clip(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let cut = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|i| *i <= limit.saturating_sub(1))
        .last()
        .unwrap_or(0);
    format!("{}…", &s[..cut])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn out(turn: u64, step: u64, text: &str) -> SessionEntry {
        SessionEntry::ModelOutput {
            turn,
            step,
            text: text.into(),
        }
    }

    /// "省略了 N 条"只数**渲染得出来的**那些。`render` 不关心的变体（`Say`、`Steer`、
    /// `TurnEnd` …）不该被算进去——模型从来看不到它们，告诉它有一段够不着的历史只会让
    /// 它去猜那里面有什么。
    #[test]
    fn the_omitted_count_ignores_entries_that_were_never_renderable() {
        let recent = vec![
            SessionEntry::Say {
                turn: 1,
                text: "hi".into(),
            },
            out(1, 0, "thinking"),
            SessionEntry::TurnEnd {
                turn: 1,
                t_kernel: 0,
                reason: "end_turn".into(),
            },
        ];
        let text = user(&recent);
        assert!(text.contains("thinking"), "{text}");
        assert!(
            !text.contains("omitted"),
            "两条不可渲染的记录不是被省略的历史: {text}"
        );
    }

    /// 真的因为预算被挤掉的那些**要**数进去，而且窗口是连续的（最新的若干条），不会跳
    /// 过一条大的再收一条更老的小的——时间线中间留洞比少给几条更难读。
    #[test]
    fn entries_dropped_for_budget_are_counted_and_the_window_stays_contiguous() {
        let big = "x".repeat(900);
        let mut recent: Vec<SessionEntry> = (0..40).map(|i| out(1, i, &big)).collect();
        // 最老的那条很短：如果实现是 `continue` 而不是 `break`，它会从洞里钻进来。
        recent.insert(0, out(1, 99, "OLDEST-TINY"));
        let text = user(&recent);
        assert!(text.len() <= MAX_RECENT_BYTES + 1024, "{}", text.len());
        assert!(text.contains("omitted"), "{text}");
        assert!(
            !text.contains("OLDEST-TINY"),
            "窗口必须是连续的一段，不该跳过大的去捡小的"
        );
    }

    #[test]
    fn the_system_instructions_carry_the_preference_order_and_the_do_not_list() {
        for needle in [
            "Update the skill that was already loaded",
            "update an existing umbrella skill",
            "add a supporting file under an existing skill",
            "create a new class-level skill",
            "DO NOT capture",
            "Environment-specific failures",
            "does not work",
            "Unresolved attempt chains",
        ] {
            assert!(SYSTEM.contains(needle), "system 指令里少了 {needle:?}");
        }
        // review 够不到世界：三个工具之外一个字都不该提。
        for forbidden in ["robot_", "ask_slow_brain", "ask_human"] {
            assert!(!SYSTEM.contains(forbidden), "{forbidden} 不该出现在指令里");
        }
    }

    #[test]
    fn an_empty_history_says_so_instead_of_rendering_an_empty_block() {
        let text = user(&[]);
        assert!(text.contains("nothing happened"), "{text}");
    }

    #[test]
    fn entries_render_newest_first_and_a_flood_drops_the_oldest() {
        let text = user(&[out(1, 0, "first"), out(2, 0, "second")]);
        assert_eq!(
            text,
            "# Recent activity (newest first)\n[t2 s0] said: second\n[t1 s0] said: first\n"
        );

        let many: Vec<SessionEntry> = (0..100)
            .map(|n| out(n, 0, &format!("entry-{n:03} {}", "x".repeat(500))))
            .collect();
        let text = user(&many);
        assert!(text.len() <= MAX_RECENT_BYTES + 64, "{}", text.len());
        assert!(text.contains("entry-099"), "最新的那条必须在");
        assert!(!text.contains("entry-000"), "最旧的那条该被丢掉");
        assert!(text.contains("older entries omitted"), "{text}");
    }

    #[test]
    fn a_multi_line_model_output_still_occupies_exactly_one_line() {
        let text = user(&[out(1, 0, "a\nb\nc")]);
        assert_eq!(text.lines().count(), 2, "{text}");
        assert!(text.contains("said: a ⏎ b ⏎ c"), "{text}");
    }

    #[test]
    fn tool_calls_and_results_are_paired_by_call_id() {
        let text = user(&[
            SessionEntry::ToolCall {
                turn: 3,
                step: 1,
                call_id: "c1".into(),
                name: "robot_walk".into(),
                args: json!({"vx": 0.1}),
            },
            SessionEntry::ToolResult {
                turn: 3,
                step: 1,
                call_id: "c1".into(),
                ok: false,
                result: json!({"error": "refused"}),
            },
        ]);
        assert!(text.contains("[t3 s1] result c1 ok=false {\"error\":\"refused\"}"));
        assert!(text.contains("[t3 s1] call c1 robot_walk {\"vx\":0.1}"));
    }

    /// 一条超长的记录被截断，而且切点不会落在半个 UTF-8 序列上。
    #[test]
    fn one_oversized_entry_is_clipped_on_a_char_boundary() {
        let text = user(&[out(1, 0, &"路".repeat(MAX_ENTRY_BYTES))]);
        assert!(text.contains('…'), "{}", &text[..40]);
        assert!(text.len() < MAX_ENTRY_BYTES + 64);
    }
}
