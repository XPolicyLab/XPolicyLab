//! 模型看到的两段文字：**稳定**的 system 前缀，和**每 turn 变化**的观测渲染。
//!
//! 这条分界线是 prompt 缓存的全部：三家厂商都按"前缀逐字节相同"命中缓存，所以人格 +
//! 工具索引必须字节级稳定（工具按名字排序、不含任何时间戳/序号/计数），而观测、收件箱、
//! steer 只能出现在 user message 里。把一条"当前 t_kernel"塞进 system 前缀，代价是此后
//! 每个 step 的输入 token 全额计费——而且没有任何报错告诉你发生了什么。
//!
//! 渲染刻意是**行式**而不是 JSON：观测本身是给人看的摘要（折叠过的事件 + 状态寄存器
//! 快照），一行一条比一坨嵌套 JSON 省 token，也更难让模型把结构当成可以照抄的格式。
//! 每条事件的 payload 仍然是紧凑 JSON——那是原始事实，重新措辞就等于篡改。

use oeh_kernel::observation::Observation;
use serde_json::Value;

use crate::model::ToolSpec;

/// system 前缀：人格 + 工具索引 + skill 索引。**字节级稳定**——同一份人格、同一张工具表
/// 和同一摞 skill 永远渲染成同一个字符串。
///
/// 工具在这里只有一行 `name: description`：完整的 JSON schema 走
/// [`crate::model::ModelRequest::tools`]（厂商的工具定义字段），在 system 里再抄一遍
/// 既浪费 token 又给了模型两份可能不一致的真相。
///
/// 排序在这里做而不是指望调用方：`ToolRegistry` 用的是 `BTreeMap`，但这个函数的签名
/// 收的是一个切片，谁都能传一个乱序的 `Vec` 进来，而缓存失效是静默的。
///
/// `skills` 是 `(rel, 第一行)` 的有序表（`KnowledgeStore::list(Kind::Skill)` 已经排过
/// 序）。它**只在非空时**渲染成一段 `# Skills`：一个还没有任何 skill 的机器人不该为一个
/// 空小节付 token，而"第一个 skill 被写下来"那一刻前缀从没有这一段变成有，正是 review
/// 写完之后我们要的那次缓存失效。
///
/// 这一段是前缀里唯一**会变**的东西，而且只在一次成功的 review 写入之后变（见
/// [`crate::native::review`]）。这是有意的代价：换来的是模型每个 turn 都看得见自己有
/// 哪些 skill，而不必先花一次工具调用去 list。
pub fn system_prefix(persona: &str, tools: &[ToolSpec], skills: &[(String, String)]) -> String {
    let mut sorted: Vec<&ToolSpec> = tools.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut s = String::with_capacity(persona.len() + 64 + sorted.len() * 64);
    s.push_str(persona.trim_end());
    s.push_str("\n\n# Tools\n");
    for t in sorted {
        s.push_str(&t.name);
        s.push_str(": ");
        s.push_str(t.description.trim());
        s.push('\n');
    }
    push_skills(&mut s, skills);
    s
}

/// `# Skills` 段。空表就整段不渲染（见 [`system_prefix`]）。
///
/// 每条一行 `<rel>: <SKILL.md 的第一行>`，标题按字符截到 [`MAX_SKILL_TITLE`]，整段按
/// 字节截到 [`MAX_SKILLS_BYTES`]。两道闸都是必要的：这段文字挂在**每一个 step 的输入
/// token 上**，而它的内容是模型自己写进磁盘的——没有上限的话，一次 review 写下一个
/// 40 KB 的第一行就等于给这个脑此后的每一次调用都加价。截断是**确定性**的（`skills`
/// 已排序），所以前缀仍然字节稳定。
fn push_skills(s: &mut String, skills: &[(String, String)]) {
    if skills.is_empty() {
        return;
    }
    const HEADER: &str = "\n# Skills\n";
    let mut body = String::with_capacity(skills.len() * 64);
    let mut kept = 0usize;
    for (rel, title) in skills {
        let line = format!("{rel}: {}\n", clip(title.trim(), MAX_SKILL_TITLE));
        if !body.is_empty() && body.len() + line.len() > MAX_SKILLS_BYTES {
            break;
        }
        body.push_str(&line);
        kept += 1;
    }
    s.push_str(HEADER);
    s.push_str(&body);
    if kept < skills.len() {
        s.push_str(&format!(
            "[… {} more skills omitted]\n",
            skills.len() - kept
        ));
    }
}

/// 一条 skill 标题的字符上限。
const MAX_SKILL_TITLE: usize = 160;
/// 整个 `# Skills` 段的字节上限（4 KiB）。
const MAX_SKILLS_BYTES: usize = 4 * 1024;

/// 按**字符**截断（切点落在字符边界上，一个中文标题不该在半个 UTF-8 序列上切开）。
fn clip(s: &str, chars: usize) -> String {
    match s.char_indices().nth(chars) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

/// 一份观测渲染成 user message 的正文。
///
/// 四段，缺哪段就写 `(none)`——固定的骨架比"有内容才出现的小节"好读，模型也不用猜
/// "没有 Events 小节"是没事发生还是渲染坏了：
///
/// ```text
/// # Observation
/// wake: obs/gesture seq=41 action=wake
/// t_kernel: 1234567
/// seq_range: 38..=41
///
/// ## State
/// fresh.lowstate: true
///
/// ## Events
/// obs/gesture x1: {"gesture":"wave"}
///
/// (2 more events were dropped by the observation budget)
/// ```
///
/// 每条折叠事件只渲染 `last`：`first` 和帧引用留在会话日志里的那份完整观测中
/// （`SessionEntry::Observation`），模型要的是"最后一次看到的是什么"。
pub fn render_observation(obs: &Observation) -> String {
    cap(observation_text(obs), MAX_OBSERVATION_BYTES)
}

/// Opt-in evaluation path: never turn a complete policy packet into a partial one.
pub fn render_complete_observation(obs: &Observation) -> Result<String, String> {
    let text = observation_text(obs);
    if text.len() > MAX_OBSERVATION_BYTES {
        return Err(format!(
            "policy observation exceeds complete-observation limit: {} > {} bytes",
            text.len(),
            MAX_OBSERVATION_BYTES
        ));
    }
    Ok(text)
}

fn observation_text(obs: &Observation) -> String {
    let mut s = String::with_capacity(256);
    s.push_str("# Observation\n");
    s.push_str(&format!(
        "wake: {} seq={} action={}\n",
        obs.wake_reason.kind,
        obs.wake_reason.seq,
        action_name(&obs.wake_reason.action)
    ));
    s.push_str(&format!("t_kernel: {}\n", obs.t_kernel));
    s.push_str(&format!(
        "seq_range: {}..={}\n",
        obs.source_seq_range.0, obs.source_seq_range.1
    ));

    s.push_str("\n## State\n");
    if obs.state.is_empty() {
        s.push_str("(none)\n");
    } else {
        for (k, v) in &obs.state {
            s.push_str(&format!("{k}: {}\n", compact(v)));
        }
    }

    s.push_str("\n## Events\n");
    if obs.events.is_empty() {
        s.push_str("(none)\n");
    } else {
        for e in &obs.events {
            let payload = if e.kind == super::vision::CAMERA_EVENT {
                super::vision::text_payload(&e.last)
            } else {
                e.last.clone()
            };
            s.push_str(&format!("{} x{}: {}\n", e.kind, e.count, compact(&payload)));
        }
    }
    if obs.truncated > 0 {
        s.push_str(&format!(
            "\n({} more events were dropped by the observation budget)\n",
            obs.truncated
        ));
    }
    s
}

/// 一次观测渲染的字节上限。
///
/// 内核的观测预算（默认 32 条 / 8 KiB）已经把常规情况压住了，但它的**字节**那一遍在
/// `events.len() <= 1` 时就停了——一条超大的 payload 会整条穿过来。而每条 steer 又会把
/// 一整份观测渲染推进收件箱（[`crate::native`]），一个把高频 kind 映射成 `steer` 的唤醒
/// 策略就够堆出一条任意大的 user message，**而且它会在这个 turn 剩下的每个 step 上重复
/// 计费**。所以脑这边自己也设一道闸。
pub const MAX_OBSERVATION_BYTES: usize = 32 * 1024;

/// 超长就截断并留下痕迹。切点落在字符边界上（`char_indices`）——一个中文 payload 不该
/// 让这里在半个 UTF-8 序列上 panic。
fn cap(s: String, limit: usize) -> String {
    if s.len() <= limit {
        return s;
    }
    const MARK: &str = "\n[truncated]\n";
    let room = limit.saturating_sub(MARK.len());
    let cut = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|i| *i <= room)
        .last()
        .unwrap_or(0);
    let mut out = s[..cut].to_string();
    out.push_str(MARK);
    out
}

/// 一次 `# Inbox` 块的字节上限。
///
/// [`MAX_OBSERVATION_BYTES`] 管的是**单条**，这里管的是**总量**：`Inbox` 本身没有容量
/// 上限，而一个把高频 kind 映射成 `steer` 的唤醒策略会在一个慢 step 里塞进任意多条，每条
/// 都是一整份观测渲染。没有这道闸，一个 step 的 user message 可以是 N × 32 KiB，而且
/// 这个 turn 剩下的每个 step 都要重新为它付一次钱。
pub const MAX_INBOX_BYTES: usize = 16 * 1024;

/// 认领到的收件箱条目渲染成一段 user message；没有条目就是 `None`（别推一条空 message
/// 进 transcript——厂商对空 content 的反应各不相同）。
///
/// 条目正文**多于一行时，后续行缩进四格**。一条 steer 的正文就是一整份
/// [`render_observation`]（见 [`crate::native`] 的 `BrainSink::steer`），不缩进的话只有
/// 第一行带着 `[steer]`，剩下的——连同它自己的 `# Observation` / `## Events` 标题——会
/// 平铺在 `# Inbox` 底下，模型完全可以把那一截读成第二份观测。
///
/// 总量超过 [`MAX_INBOX_BYTES`] 时**留新丢旧**：收件箱是按投递序排的，最后那几条是最
/// 接近"现在"的指令，而最旧的那条 steer 很可能已经被后来的覆盖了。丢掉的条数写成一行
/// `[… N older items omitted]`——模型得知道自己看到的是一截而不是全部；日志那一侧由
/// `brain/inbox_dropped` 负责（见 [`crate::native`]）。
pub fn render_inbox(items: &[crate::inbox::InboxItem]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    const HEADER: &str = "# Inbox\n";
    // 从最新往回收，预算用光就停。至少留一条：一个只有标题的收件箱比没有还糟——模型会
    // 以为"有东西进来但内容是空的"。那一条要是自己就超预算，下面的 `cap` 会把它截断。
    let mut blocks: Vec<String> = Vec::new();
    let mut used = HEADER.len();
    for i in items.iter().rev() {
        let b = render_item(i);
        if !blocks.is_empty() && used + b.len() > MAX_INBOX_BYTES {
            break;
        }
        used += b.len();
        blocks.push(b);
    }
    blocks.reverse();
    let dropped = items.len() - blocks.len();
    let mut s = String::with_capacity(used + 48);
    s.push_str(HEADER);
    if dropped > 0 {
        s.push_str(&format!("[… {dropped} older items omitted]\n"));
    }
    for b in blocks {
        s.push_str(&b);
    }
    // 硬上限：留下来的那一条自己超预算时，这一刀是最后的保证。
    Some(cap(s, MAX_INBOX_BYTES))
}

/// The complete policy path refuses an inbox that the legacy renderer would trim.
pub fn render_complete_inbox(items: &[crate::inbox::InboxItem]) -> Result<Option<String>, String> {
    if items.is_empty() {
        return Ok(None);
    }
    let mut text = String::from("# Inbox\n");
    for item in items {
        text.push_str(&render_item(item));
    }
    if text.len() > MAX_INBOX_BYTES {
        return Err(format!(
            "policy inbox exceeds complete-observation limit: {} > {} bytes",
            text.len(),
            MAX_INBOX_BYTES
        ));
    }
    Ok(Some(text))
}

/// 一条收件箱条目：`[lane] 第一行`，其余行缩进四格。
fn render_item(i: &crate::inbox::InboxItem) -> String {
    let mut lines = i.text.lines();
    let mut s = format!("[{}] {}\n", lane_name(i.lane), lines.next().unwrap_or(""));
    for l in lines {
        s.push_str("    ");
        s.push_str(l);
        s.push('\n');
    }
    s
}

fn lane_name(l: crate::inbox::Lane) -> &'static str {
    match l {
        crate::inbox::Lane::Followup => "followup",
        crate::inbox::Lane::Steer => "steer",
        crate::inbox::Lane::Inject => "inject",
    }
}

/// `WakeAction` 的 serde 名字（小写），和世界日志、唤醒策略 YAML 里的是同一套。
fn action_name(a: &oeh_kernel::wake::WakeAction) -> String {
    match serde_json::to_value(a) {
        Ok(Value::String(s)) => s,
        _ => String::new(),
    }
}

/// 紧凑 JSON；序列化不该失败（`Value` 总能序列化），真失败了也只损失这一行。
fn compact(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "<unserialisable>".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inbox::{InboxItem, Lane};
    use oeh_kernel::observation::{CollapsedEvent, WakeReason};
    use oeh_kernel::wake::WakeAction;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn tool(name: &str, desc: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: desc.into(),
            parameters: json!({"type": "object"}),
            concurrency_safe: true,
        }
    }

    #[test]
    fn the_system_prefix_sorts_tools_and_is_byte_stable() {
        let a = system_prefix(
            "you are a robot",
            &[tool("robot_walk", "walk"), tool("recall_read", "read")],
            &[],
        );
        let b = system_prefix(
            "you are a robot",
            &[tool("recall_read", "read"), tool("robot_walk", "walk")],
            &[],
        );
        assert_eq!(a, b, "工具顺序不同不该改变前缀");
        assert_eq!(
            a,
            "you are a robot\n\n# Tools\nrecall_read: read\nrobot_walk: walk\n"
        );
    }

    #[test]
    fn an_empty_tool_table_still_renders_the_header() {
        assert_eq!(system_prefix("p", &[], &[]), "p\n\n# Tools\n");
    }

    fn skill(rel: &str, title: &str) -> (String, String) {
        (rel.to_string(), title.to_string())
    }

    /// 没有 skill 就整段不渲染；有了就多一段——那一下**正是**我们要的缓存失效。
    #[test]
    fn the_skills_section_appears_only_once_there_is_a_skill() {
        let none = system_prefix("p", &[], &[]);
        let one = system_prefix("p", &[], &[skill("skills/nav/SKILL.md", "# nav")]);
        assert_eq!(
            one,
            "p\n\n# Tools\n\n# Skills\nskills/nav/SKILL.md: # nav\n"
        );
        assert_ne!(none, one);
        // 同一摞 skill 渲染两次逐字节相同。
        assert_eq!(
            one,
            system_prefix("p", &[], &[skill("skills/nav/SKILL.md", "# nav")])
        );
    }

    /// 标题是模型自己写进磁盘的，所以它受两道确定性的闸：单条按字符截、整段按字节截。
    #[test]
    fn an_oversized_skill_index_is_clipped_deterministically() {
        let long = skill("skills/a/SKILL.md", &"标".repeat(MAX_SKILL_TITLE * 2));
        let text = system_prefix("p", &[], std::slice::from_ref(&long));
        assert!(text.contains("…\n"), "{text}");
        assert_eq!(text, system_prefix("p", &[], &[long]));

        let many: Vec<(String, String)> = (0..400)
            .map(|n| skill(&format!("skills/s{n:03}/SKILL.md"), &"x".repeat(100)))
            .collect();
        let text = system_prefix("p", &[], &many);
        let body = text.split("\n# Skills\n").nth(1).unwrap();
        assert!(body.len() <= MAX_SKILLS_BYTES + 64, "{}", body.len());
        assert!(body.contains("more skills omitted"), "{body}");
        assert_eq!(text, system_prefix("p", &[], &many), "截断也必须是确定的");
    }

    fn obs(events: Vec<CollapsedEvent>, truncated: usize) -> Observation {
        Observation {
            t_kernel: 77,
            wake_reason: WakeReason {
                kind: "obs/gesture".into(),
                seq: 41,
                action: WakeAction::Wake,
            },
            state: BTreeMap::from([("fresh.lowstate".to_string(), json!(true))]),
            events,
            source_seq_range: (38, 41),
            truncated,
        }
    }

    fn collapsed(kind: &str, count: u64, last: Value) -> CollapsedEvent {
        CollapsedEvent {
            kind: kind.into(),
            collapse_key: kind.into(),
            first_seq: 38,
            last_seq: 41,
            count,
            salience: 128,
            first: json!({}),
            last,
            first_frame_refs: vec![],
            last_frame_refs: vec![],
        }
    }

    #[test]
    fn the_observation_renders_wake_state_and_collapsed_events() {
        let text = render_observation(&obs(
            vec![collapsed("obs/gesture", 2, json!({"gesture": "wave"}))],
            3,
        ));
        assert_eq!(
            text,
            "# Observation\n\
             wake: obs/gesture seq=41 action=wake\n\
             t_kernel: 77\n\
             seq_range: 38..=41\n\
             \n## State\n\
             fresh.lowstate: true\n\
             \n## Events\n\
             obs/gesture x2: {\"gesture\":\"wave\"}\n\
             \n(3 more events were dropped by the observation budget)\n"
        );
    }

    #[test]
    fn an_observation_without_events_says_so_instead_of_dropping_the_section() {
        let text = render_observation(&obs(vec![], 0));
        assert!(text.contains("\n## Events\n(none)\n"), "{text}");
        assert!(
            !text.contains("dropped by the observation budget"),
            "{text}"
        );
    }

    #[test]
    fn an_empty_inbox_renders_to_nothing() {
        assert_eq!(render_inbox(&[]), None);
    }

    #[test]
    fn a_huge_event_payload_is_capped_with_a_marker() {
        let big = "x".repeat(MAX_OBSERVATION_BYTES * 2);
        let text = render_observation(&obs(vec![collapsed("obs/blob", 1, json!(big))], 0));
        assert!(
            text.len() <= MAX_OBSERVATION_BYTES,
            "{} > {MAX_OBSERVATION_BYTES}",
            text.len()
        );
        assert!(text.ends_with("\n[truncated]\n"), "{}", &text[..80]);
        // 开头照常可读：截断不该把 wake 那几行也吃掉。
        assert!(
            text.starts_with("# Observation\nwake: obs/gesture"),
            "{}",
            &text[..80]
        );
    }

    #[test]
    fn complete_policy_observation_refuses_only_rendering_loss() {
        let small = obs(
            vec![collapsed("obs/blob", 1, json!({"note": "[truncated]"}))],
            3,
        );
        // A literal marker and unrelated events already dropped by the kernel do not
        // imply this policy packet was cut by the renderer.
        assert_eq!(
            render_complete_observation(&small).unwrap(),
            render_observation(&small)
        );
        let large = obs(
            vec![collapsed(
                "obs/blob",
                1,
                json!("x".repeat(MAX_OBSERVATION_BYTES)),
            )],
            0,
        );
        assert!(render_complete_observation(&large)
            .unwrap_err()
            .contains("complete-observation limit"));
        assert!(render_observation(&large).ends_with("\n[truncated]\n"));
    }

    #[test]
    fn complete_policy_inbox_refuses_omission_and_truncation() {
        use crate::inbox::{InboxItem, Lane};
        let items = vec![InboxItem {
            lane: Lane::Steer,
            text: "x".repeat(MAX_INBOX_BYTES),
            seq: 1,
        }];
        assert!(render_complete_inbox(&items)
            .unwrap_err()
            .contains("policy inbox"));
        assert!(render_inbox(&items).unwrap().ends_with("\n[truncated]\n"));
        let small = vec![InboxItem {
            lane: Lane::Steer,
            text: "complete".into(),
            seq: 2,
        }];
        assert_eq!(render_complete_inbox(&small).unwrap(), render_inbox(&small));
    }

    #[test]
    fn capping_never_splits_a_utf8_char() {
        // 每个字符 3 字节，上限刻意落在字符中间。
        let s = "路".repeat(40);
        let out = cap(s, 20);
        assert!(out.ends_with("\n[truncated]\n"));
        assert!(out.len() <= 20);
        // 能解回字符串就说明没切坏（`String` 本身保证 UTF-8，切坏会 panic 在 `cap` 里）。
        assert!(out
            .trim_end_matches("\n[truncated]\n")
            .chars()
            .all(|c| c == '路'));
    }

    #[test]
    fn a_flooded_inbox_keeps_the_newest_items_and_says_how_many_it_dropped() {
        // 40 条 × ~1 KiB 远超 16 KiB 的总预算。
        let items: Vec<InboxItem> = (0..40)
            .map(|n| InboxItem {
                lane: Lane::Steer,
                text: format!("item-{n:02} {}", "x".repeat(1000)),
                seq: n,
            })
            .collect();
        let text = render_inbox(&items).unwrap();
        assert!(
            text.len() <= MAX_INBOX_BYTES,
            "{} > {MAX_INBOX_BYTES}",
            text.len()
        );
        let kept = text.matches("[steer] item-").count();
        assert!(kept > 0 && kept < 40, "留下了 {kept} 条");
        // 留新丢旧：最后一条在，第一条不在。
        assert!(text.contains("[steer] item-39 "), "{}", &text[..120]);
        assert!(!text.contains("item-00 "), "最旧的那条该被丢掉");
        // 丢了几条要说出来，而且数目对得上。
        assert!(
            text.contains(&format!("[… {} older items omitted]\n", 40 - kept)),
            "{}",
            &text[..120]
        );
    }

    #[test]
    fn one_oversized_inbox_item_is_still_rendered_but_capped() {
        // 唯一的一条自己就超预算：不能渲染成一个空收件箱，但也不能突破上限。
        let text = render_inbox(&[InboxItem {
            lane: Lane::Inject,
            text: "y".repeat(MAX_INBOX_BYTES * 3),
            seq: 1,
        }])
        .unwrap();
        assert!(text.len() <= MAX_INBOX_BYTES);
        assert!(text.starts_with("# Inbox\n[inject] yyy"), "{}", &text[..40]);
        assert!(text.ends_with("\n[truncated]\n"));
    }

    #[test]
    fn a_multi_line_inbox_item_is_indented_under_its_lane() {
        let text = render_inbox(&[InboxItem {
            lane: Lane::Steer,
            text: "# Observation\nwake: obs/gesture seq=1 action=steer\n".into(),
            seq: 1,
        }])
        .unwrap();
        assert_eq!(
            text,
            "# Inbox\n[steer] # Observation\n    wake: obs/gesture seq=1 action=steer\n"
        );
    }

    #[test]
    fn inbox_items_keep_their_lane_visible() {
        let text = render_inbox(&[
            InboxItem {
                lane: Lane::Steer,
                text: "stop talking".into(),
                seq: 1,
            },
            InboxItem {
                lane: Lane::Followup,
                text: "finish the walk".into(),
                seq: 2,
            },
        ])
        .unwrap();
        assert_eq!(
            text,
            "# Inbox\n[steer] stop talking\n[followup] finish the walk\n"
        );
    }
}
