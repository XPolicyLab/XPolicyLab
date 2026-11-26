//! `ScriptedModel`：一个从 YAML 读剧本的 [`Model`]。
//!
//! 它存在的理由是"没有 API key 也要能把整条链跑一遍"——e2e 测试和 `dev-native` profile
//! 里的脑用的都是它。剧本是一串 step，每次 [`Model::complete`] 弹出下一条：
//!
//! ```yaml
//! - text: "路看起来是空的，我走两步"
//!   tool_calls:
//!     - { name: robot_walk, args: { vx: 0.1, duration_s: 0.2 } }
//! - text: "到了"
//! - missing_key: OPENAI_API_KEY      # 这一步不产出流，直接失败
//! ```
//!
//! 剧本也可以分成两段，给 RSI review fork（[`crate::native::review`]）用：
//!
//! ```yaml
//! turns:
//!   - text: "走两步"
//! review:
//!   - tool_calls: [{ name: skill_manage, args: { op: create, path: ..., content: ... } }]
//! ```
//!
//! 选哪一段看**这次请求的工具表里有没有 `skill_manage`**——那张表是 review 独有的
//! （普通 turn 的工具表里根本没有这三个工具），所以这个判据不会错认。用工具表而不是一个
//! 显式的开关：剧本模型是个假模型，它只该看得见真模型看得见的东西，而"我这次被叫来干
//! 什么"对真模型来说也正是从系统提示和工具表里读出来的。
//!
//! 裸列表（没有 `turns:` / `review:` 两个键）等价于"全是 turn，没有 review"——所有既有
//! 剧本按原样继续工作。
//!
//! 剧本用完之后每次调用都还一条空文本的 `Done{EndTurn}`：一个跑过头的 turn 会干净地
//! 收尾，而不是卡住或者 panic——脚本的长度不该变成一条隐式的断言。
//!
//! 文本按**两段**增量发：真实传输的文本是碎的，一次性发完的假模型会让"增量拼接"这段
//! 代码在测试里永远走不到。

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::model::{
    BoxFuture, EventStream, Model, ModelError, ModelEvent, ModelRequest, StopReason, ToolCall,
};

/// [`Model::name`] 的返回值，也是 `brain/request` 里的 provider / model / api_mode。
pub const NAME: &str = "scripted";

/// 剧本里的一步。
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScriptedStep {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub tool_calls: Vec<ScriptedCall>,
    /// 这一步不产出流，而是让 `complete` 还一个 [`ModelError::MissingKey`]。用来演练
    /// "attach 成功但没有 key"那条路（规格 §11 / Plan 4 的 fail-closed 约束）。
    #[serde(default)]
    pub missing_key: Option<String>,
    /// 这一步的流在文本增量与工具调用**之后**断掉：`Err(ModelError::Transport(<这里的
    /// 字>))`，没有 `Done`。
    ///
    /// 和 `missing_key` 的区别就是"断在哪里"：`missing_key` 发生在 `complete` 返回之前
    /// （一个字都还没出去），这个发生在流上（已经吐了半句话、也可能已经拼完了两次工具
    /// 调用）。真实传输的 SSE 中途重置就是后者，而那半句话和那两次调用是不是被留下来
    /// 是两种完全不同的行为。
    #[serde(default)]
    pub stream_error: Option<String>,
    /// 这一步的流在文本增量之后 **panic**。用来验"turn 里的 panic 也有终态"
    /// （`done("error: panicked")` + `brain/turn_end`）——没有这条，那五行隔离代码
    /// 就是没被验证过的，而它的症状恰恰是"什么都没发生"。
    #[serde(default)]
    pub panic: Option<String>,
    /// 这一步在产出任何东西**之前**先等这么久。
    ///
    /// 演的是"一次合法但慢的模型调用"——和 `hang` 的区别是它会正常跑完。维护相位那条
    /// 用例要的正是这个：一段比内核 `turn_timeout_ms` 还长、但在 `review.timeout_ms`
    /// 之内、并且**干净收尾**的 review。这是被测系统的**刺激**，不是判别信号（判别信
    /// 号始终是世界日志里的事件）。
    #[serde(default)]
    pub delay_ms: u64,
    /// 这一步的流**永远不产出任何东西，也永远不结束**。
    ///
    /// 一个真正挂住的厂商调用（TCP 连着、字节不来）在这一层看起来就是这样。它是唯一能
    /// 演练"整段有时限"这类兜底的形状——`stream_error` 是立刻失败，跟超时不是一回事。
    /// 用它的测试自己必须有界（`timeout_ms` / `step_timeout_ms`），否则测试也会挂住。
    #[serde(default)]
    pub hang: bool,
}

/// 剧本里的一次工具调用。`id` 不在剧本里：它由模型分配（真实厂商也是如此），这里按
/// `c1`, `c2`, … 递增，整个 `ScriptedModel` 的生命周期内唯一。
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScriptedCall {
    pub name: String,
    #[serde(default = "empty_object")]
    pub args: Value,
}

fn empty_object() -> Value {
    json!({})
}

/// 一份分段剧本。两段都可省。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Script {
    #[serde(default)]
    turns: Vec<ScriptedStep>,
    #[serde(default)]
    review: Vec<ScriptedStep>,
}

/// 这次请求是不是 review 的：只有 review 那张工具表里有它。
const REVIEW_MARKER: &str = "skill_manage";

pub struct ScriptedModel {
    steps: Mutex<VecDeque<ScriptedStep>>,
    /// review fork 的那一段。和 `steps` 各走各的队列：一次 review 插在两个 turn 中间，
    /// 共用一个队列的话 turn 的剧本会被 review 吃掉几步，而那种错位在断言失败时看起来
    /// 像是"第三个 turn 说错了话"。
    review: Mutex<VecDeque<ScriptedStep>>,
    /// 收到过的每一次请求，按调用顺序。测试靠它断言"steer 的文本真的进了下一个 step 的
    /// user message"——那件事在世界日志里看不见（prompt 全文不进日志）。
    requests: Mutex<Vec<ModelRequest>>,
    next_call_id: AtomicU64,
}

impl ScriptedModel {
    /// 裸列表或者 `{turns, review}` 两段式都收。
    ///
    /// 先试裸列表：它是绝大多数剧本的形状，而且先试两段式的话，一个把 `text` 打成
    /// `txt` 的裸列表会先以"不是一个 map"失败，错误信息指不到那个错别字上。两种都解不出
    /// 来时报**裸列表**那一条错，理由相同。
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        let flat = match serde_yaml_ng::from_str::<Vec<ScriptedStep>>(text) {
            Ok(steps) => return Ok(Self::from_steps(steps)),
            Err(e) => e,
        };
        match serde_yaml_ng::from_str::<Script>(text) {
            Ok(s) => Ok(Self::from_sections(s.turns, s.review)),
            Err(_) => Err(flat.to_string()),
        }
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("script {}: {e}", path.display()))?;
        Self::from_yaml(&text).map_err(|e| format!("script {}: {e}", path.display()))
    }

    pub fn from_steps(steps: Vec<ScriptedStep>) -> Self {
        Self::from_sections(steps, Vec::new())
    }

    /// 两段式：普通 turn 的步子和 review 的步子。
    pub fn from_sections(turns: Vec<ScriptedStep>, review: Vec<ScriptedStep>) -> Self {
        Self {
            steps: Mutex::new(turns.into()),
            review: Mutex::new(review.into()),
            requests: Mutex::new(Vec::new()),
            next_call_id: AtomicU64::new(0),
        }
    }

    /// 剧本里还剩几步（两段合计）。
    pub fn remaining(&self) -> usize {
        self.steps.lock().len() + self.review.lock().len()
    }

    /// 收到过的全部请求，按调用顺序。**不是公开合约**：只有测试与 dev 诊断用得上它，
    /// 生产路径没有理由读模型收到过什么（那份记录在会话日志里）。
    #[doc(hidden)]
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().clone()
    }
}

impl Model for ScriptedModel {
    fn name(&self) -> String {
        NAME.to_string()
    }

    /// 剧本模型没有 wire。报自己的名字而不是默认的 `"unknown"`：`brain/request` 的三个
    /// 身份字段因此全是 `scripted`，一眼看得出这个 turn 根本没出过网。
    fn api_mode(&self) -> String {
        NAME.to_string()
    }

    fn complete(&self, req: ModelRequest) -> BoxFuture<Result<EventStream, ModelError>> {
        // 记账与弹出都在这里做（同步的那一刻），不在 future 里：并发调用时"第 n 次调用
        // 拿到第 n 步"必须由调用顺序决定，而 future 被 poll 的顺序不保证。两把锁各自
        // 只按一下，中间不 await。
        let is_review = req.tools.iter().any(|t| t.name == REVIEW_MARKER);
        self.requests.lock().push(req);
        let step = if is_review {
            self.review.lock().pop_front()
        } else {
            self.steps.lock().pop_front()
        };
        let Some(step) = step else {
            // 剧本用完：空文本的 `Done{EndTurn}`，turn 干净收尾。
            return Box::pin(async { Ok(stream(vec![done(StopReason::EndTurn)])) });
        };
        if let Some(env) = step.missing_key {
            return Box::pin(async move { Err(ModelError::MissingKey(env)) });
        }
        if step.delay_ms > 0 {
            // 等在 `complete` 返回的那个 future 里（不是在这里同步睡）：调用方仍然用
            // `by(deadline, ..)` 包着它，所以 step / review 的时限照常管得住它。
            let d = Duration::from_millis(step.delay_ms);
            let rest = ScriptedStep {
                delay_ms: 0,
                ..step
            };
            let ids = self
                .next_call_id
                .fetch_add(rest.tool_calls.len() as u64, Ordering::SeqCst);
            return Box::pin(async move {
                tokio::time::sleep(d).await;
                Ok(render(rest, ids))
            });
        }
        if step.hang {
            // 一条永远不产出、也永远不结束的流。`pending()` 不占 CPU，被丢掉时也干净。
            return Box::pin(async {
                Ok(Box::pin(futures_util::stream::pending()) as EventStream)
            });
        }
        let ids = self
            .next_call_id
            .fetch_add(step.tool_calls.len() as u64, Ordering::SeqCst);
        Box::pin(async move { Ok(render(step, ids)) })
    }
}

/// 一步的事件流。`first_call_id` 是这一步第一次工具调用的编号基数——编号在
/// [`Model::complete`] 的同步那一刻就领走了（和弹出剧本同一个理由：并发调用时编号必须
/// 由调用顺序决定，而不是 future 被 poll 的顺序）。
fn render(step: ScriptedStep, first_call_id: u64) -> EventStream {
    let mut events = Vec::new();
    for chunk in split_in_two(&step.text) {
        events.push(Ok(ModelEvent::TextDelta(chunk)));
    }
    if let Some(msg) = step.panic {
        // 增量先出去，然后这一条在被 poll 的那一刻炸掉——和一个在半个 SSE 事件上
        // panic 的解析器同形。
        return panicking_stream(events, msg);
    }
    let has_calls = !step.tool_calls.is_empty();
    for (i, c) in step.tool_calls.into_iter().enumerate() {
        let n = first_call_id + i as u64 + 1;
        events.push(Ok(ModelEvent::ToolCall(ToolCall {
            id: format!("c{n}"),
            name: c.name,
            args: c.args,
        })));
    }
    if let Some(msg) = step.stream_error {
        // 增量和工具调用都已经出去了，然后流断掉：没有 `Done`。
        events.push(Err(ModelError::Transport(msg)));
        return stream(events);
    }
    events.push(done(if has_calls {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    }));
    stream(events)
}

fn done(stop: StopReason) -> Result<ModelEvent, ModelError> {
    Ok(ModelEvent::Done { stop })
}

fn stream(events: Vec<Result<ModelEvent, ModelError>>) -> EventStream {
    Box::pin(futures_util::stream::iter(events))
}

/// `events`，然后一条 poll 就 panic 的尾巴。
fn panicking_stream(events: Vec<Result<ModelEvent, ModelError>>, msg: String) -> EventStream {
    use futures_util::StreamExt;
    let tail = futures_util::stream::once(async move { boom(msg) });
    Box::pin(futures_util::stream::iter(events).chain(tail))
}

/// 返回类型写实（不是 `!`）好让 `chain` 的 item 类型推得出来。
fn boom(msg: String) -> Result<ModelEvent, ModelError> {
    panic!("scripted model panic: {msg}");
}

/// 把一段文本切成至多两个增量。切点落在**字符边界**上（`char_indices` 找的），不然
/// 一个中文人格会让这个假模型在半个 UTF-8 序列上 panic。
fn split_in_two(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mid = text
        .char_indices()
        .nth(text.chars().count() / 2)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    if mid == 0 || mid >= text.len() {
        return vec![text.to_string()];
    }
    vec![text[..mid].to_string(), text[mid..].to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{FutureExt, StreamExt};
    use std::panic::AssertUnwindSafe;

    fn req() -> ModelRequest {
        ModelRequest {
            system: "s".into(),
            messages: vec![],
            tools: vec![],
            max_tokens: 16,
            temperature: Some(0.0),
        }
    }

    async fn drain(m: &ScriptedModel) -> Result<Vec<ModelEvent>, ModelError> {
        let mut s = m.complete(req()).await?;
        let mut out = Vec::new();
        while let Some(e) = s.next().await {
            out.push(e?);
        }
        Ok(out)
    }

    #[tokio::test]
    async fn a_script_step_streams_text_then_tool_calls_then_done() {
        let m = ScriptedModel::from_yaml(
            "- text: hello\n  tool_calls:\n    - { name: robot_walk, args: { vx: 0.1 } }\n",
        )
        .unwrap();
        let evs = drain(&m).await.unwrap();
        assert_eq!(
            evs,
            vec![
                ModelEvent::TextDelta("he".into()),
                ModelEvent::TextDelta("llo".into()),
                ModelEvent::ToolCall(ToolCall {
                    id: "c1".into(),
                    name: "robot_walk".into(),
                    args: json!({"vx": 0.1}),
                }),
                ModelEvent::Done {
                    stop: StopReason::ToolUse
                },
            ]
        );
        assert_eq!(m.remaining(), 0);
    }

    #[tokio::test]
    async fn a_step_without_tool_calls_ends_the_turn() {
        let m = ScriptedModel::from_yaml("- text: done\n").unwrap();
        let evs = drain(&m).await.unwrap();
        assert_eq!(
            evs.last(),
            Some(&ModelEvent::Done {
                stop: StopReason::EndTurn
            })
        );
    }

    #[tokio::test]
    async fn an_exhausted_script_keeps_answering_with_an_empty_end_turn() {
        let m = ScriptedModel::from_steps(vec![]);
        for _ in 0..2 {
            assert_eq!(
                drain(&m).await.unwrap(),
                vec![ModelEvent::Done {
                    stop: StopReason::EndTurn
                }]
            );
        }
    }

    #[tokio::test]
    async fn a_missing_key_step_fails_without_a_stream() {
        let m = ScriptedModel::from_yaml("- missing_key: OPENAI_API_KEY\n").unwrap();
        // `EventStream` 不是 `Debug`，所以不能 `unwrap_err`。
        let Err(err) = m.complete(req()).await else {
            panic!("missing_key 这一步不该产出流");
        };
        assert!(
            matches!(&err, ModelError::MissingKey(e) if e == "OPENAI_API_KEY"),
            "{err:?}"
        );
        assert_eq!(err.to_string(), "missing api key OPENAI_API_KEY");
    }

    #[tokio::test]
    async fn every_request_is_recorded_in_call_order() {
        let m = ScriptedModel::from_yaml("- text: a\n- text: b\n").unwrap();
        let mut first = req();
        first.system = "one".into();
        let mut second = req();
        second.system = "two".into();
        let _ = m.complete(first).await.unwrap();
        let _ = m.complete(second).await.unwrap();
        let got: Vec<String> = m.requests().into_iter().map(|r| r.system).collect();
        assert_eq!(got, vec!["one", "two"]);
    }

    #[tokio::test]
    async fn a_stream_error_arrives_after_the_deltas_and_calls_that_preceded_it() {
        let m = ScriptedModel::from_yaml(
            "- text: abcd\n  tool_calls: [{ name: robot_walk }]\n  stream_error: connection reset\n",
        )
        .unwrap();
        let mut s = m.complete(req()).await.unwrap();
        assert_eq!(
            s.next().await.unwrap().unwrap(),
            ModelEvent::TextDelta("ab".into())
        );
        assert_eq!(
            s.next().await.unwrap().unwrap(),
            ModelEvent::TextDelta("cd".into())
        );
        // 工具调用在断流之前就拼完了——`partial_tool_calls` 数的就是这些。
        assert_eq!(
            s.next().await.unwrap().unwrap(),
            ModelEvent::ToolCall(ToolCall {
                id: "c1".into(),
                name: "robot_walk".into(),
                args: json!({}),
            })
        );
        let Some(Err(e)) = s.next().await else {
            panic!("最后一条该是 Err");
        };
        assert_eq!(e.to_string(), "transport: connection reset");
        assert!(s.next().await.is_none(), "断掉的流之后没有 Done");
    }

    #[tokio::test]
    async fn a_panicking_step_panics_while_the_stream_is_polled_not_before() {
        let m = ScriptedModel::from_yaml("- text: ab\n  panic: boom\n").unwrap();
        let mut s = m.complete(req()).await.expect("complete 本身不 panic");
        assert_eq!(
            s.next().await.unwrap().unwrap(),
            ModelEvent::TextDelta("a".into())
        );
        assert_eq!(
            s.next().await.unwrap().unwrap(),
            ModelEvent::TextDelta("b".into())
        );
        let caught = AssertUnwindSafe(s.next()).catch_unwind().await;
        assert!(caught.is_err(), "第三次 poll 该炸");
    }

    #[test]
    fn tool_call_args_default_to_an_empty_object() {
        let m = ScriptedModel::from_yaml("- tool_calls: [{ name: stop }]\n").unwrap();
        assert_eq!(m.remaining(), 1);
        let step = m.steps.lock()[0].clone();
        assert_eq!(step.tool_calls[0].args, json!({}));
        assert_eq!(step.text, "");
    }

    /// 两段式剧本：按请求的工具表选段，两个队列互不相干。
    #[tokio::test]
    async fn a_two_section_script_serves_review_steps_from_its_own_queue() {
        let m = ScriptedModel::from_yaml(
            "turns:\n  - text: turn-one\n  - text: turn-two\nreview:\n  - text: review-one\n",
        )
        .unwrap();
        assert_eq!(m.remaining(), 3);
        let review_req = || ModelRequest {
            tools: vec![crate::model::ToolSpec {
                name: REVIEW_MARKER.into(),
                description: String::new(),
                parameters: json!({"type": "object"}),
                concurrency_safe: false,
            }],
            ..req()
        };
        // 一次 review 调用只吃 review 那一段。
        let _ = m.complete(review_req()).await.unwrap();
        assert_eq!(m.steps.lock().len(), 2, "turn 那一段没被动过");
        assert_eq!(m.review.lock().len(), 0);
        // 普通调用继续从 turn 那一段拿。
        let _ = m.complete(req()).await.unwrap();
        assert_eq!(m.steps.lock().len(), 1);
    }

    /// 裸列表照旧（所有既有剧本），而且它等价于"没有 review 段"。
    #[test]
    fn a_flat_list_is_still_a_script_with_no_review_section() {
        let m = ScriptedModel::from_yaml("- text: a\n- text: b\n").unwrap();
        assert_eq!(m.steps.lock().len(), 2);
        assert!(m.review.lock().is_empty());
    }

    #[test]
    fn an_unknown_key_in_the_script_is_an_error() {
        let Err(err) = ScriptedModel::from_yaml("- txt: oops\n") else {
            panic!("剧本里的错别字必须报错，不能被忽略");
        };
        assert!(err.contains("txt"), "{err}");
        // 两段式里的错别字同样不能被忽略（`turns` 打成 `turn`）。
        let Err(err) = ScriptedModel::from_yaml("turn:\n  - text: a\n") else {
            panic!("两段式里的错别字必须报错");
        };
        assert!(!err.is_empty(), "{err}");
    }

    #[test]
    fn splitting_never_cuts_a_utf8_char() {
        assert_eq!(split_in_two(""), Vec::<String>::new());
        assert_eq!(split_in_two("a"), vec!["a".to_string()]);
        assert_eq!(
            split_in_two("路是空的"),
            vec!["路是".to_string(), "空的".to_string()]
        );
    }
}
