//! Codex app-server 线格式里我们用得到的那一小块：方法名 + 每个方法的 params/result。
//!
//! **方法名逐字照抄 Codex**，不按这个仓库的习惯改写：段是 `/` 分开的，段内是
//! camelCase（`item/agentMessage/delta`、`item/commandExecution/requestApproval`）。
//! 字段名则是 snake_case（`thread_id`、`turn_id`、`call_id`），所以结构体字段直接用
//! Rust 的写法就对得上，不需要 `rename`。
//!
//! 信封是 JSON-RPC 2.0 行分隔——和这个仓库的边界协议同一套 codec
//! （[`oeh_rpc::codec`]），所以子进程的 stdin/stdout 可以直接交给
//! [`oeh_rpc::client::Client`]。
//!
//! 这个文件**只有类型**：谁在什么时候发什么，全在 [`super`]。反序列化一律给可选/默认
//! 值，wire 上少一个字段不该让一次 turn 变成 panic——这些字节来自另一个进程。

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ------------------------------------------------------------ 方法名

/// client → codex：开一条线程（`{}` → `{thread_id}`）。
pub const THREAD_START: &str = "thread/start";
/// client → codex：开一个 turn（[`TurnStart`] → [`TurnStarted`]）。
pub const TURN_START: &str = "turn/start";
/// client → codex：往进行中的 turn 里插一句话（[`TurnSteer`]）。
pub const TURN_STEER: &str = "turn/steer";
/// client → codex：打断进行中的 turn（[`TurnRef`]）。
pub const TURN_INTERRUPT: &str = "turn/interrupt";

/// codex → client 通知：一段 agent 文本增量（[`AgentMessageDelta`]）。
pub const AGENT_MESSAGE_DELTA: &str = "item/agentMessage/delta";
/// codex → client 通知：一个 item 结束了（[`ItemCompleted`]）。
pub const ITEM_COMPLETED: &str = "item/completed";
/// codex → client 通知：turn 结束了（[`TurnCompleted`]）。
pub const TURN_COMPLETED: &str = "turn/completed";

/// codex → client 请求：要执行一条命令，问能不能（[`ApprovalRequest`] → [`ApprovalReply`]）。
pub const REQUEST_APPROVAL: &str = "item/commandExecution/requestApproval";
/// codex → client 请求：调一个 DynamicTool（[`ToolCall`] → [`ToolReply`]）。
pub const TOOL_CALL: &str = "item/tool/call";

/// [`ItemCompleted::item`] 里 agent 文本那一类的 `type`。camelCase，照抄。
pub const ITEM_AGENT_MESSAGE: &str = "agentMessage";

/// 能力工具在 codex 那边的名字前缀。`robot_walk` → 能力 `walk`。
pub const ROBOT_PREFIX: &str = "robot_";

// ------------------------------------------------------------ client → codex

/// `turn/start` 的一条输入。里程碑 1 只有文本（图片/音频是 Codex 的别的 variant）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Input {
    Text { text: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnStart {
    pub thread_id: String,
    pub input: Vec<Input>,
    /// 能力面摘要。可省——省了就是"这个 turn 没有工具可用"。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_context: Option<String>,
}

/// `turn/interrupt` 的 params，也是任何"指名道姓某个 turn"的方法的公共部分。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRef {
    pub thread_id: String,
    pub turn_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnSteer {
    pub thread_id: String,
    pub turn_id: String,
    pub text: String,
}

// ------------------------------------------------------------ codex → client（结果）

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadStarted {
    pub thread_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnStarted {
    pub turn_id: String,
}

// ------------------------------------------------------------ codex → client（通知）

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentMessageDelta {
    #[serde(default)]
    pub thread_id: String,
    /// 哪个 turn 的增量。**`Option` 而不是"默认空串"**：这两件事必须分得开——
    /// "codex 没说这条属于谁"（`None`，按当前 turn 算）和"codex 说了，但不是这个 turn"
    /// （丢掉并计数）。默认成空串再拿它做严格相等的话，对端哪天换个字段名，结果是**全部**
    /// 模型输出被静默丢掉：turn 照常 `end_turn`，会话日志空白，日志里一条线索都没有。
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub delta: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Item {
    /// `agentMessage` / `functionCall` / …。不认识的一律忽略。
    #[serde(rename = "type", default)]
    pub item_type: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ItemCompleted {
    #[serde(default)]
    pub item: Item,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnCompleted {
    #[serde(default)]
    pub thread_id: String,
    #[serde(default)]
    pub turn_id: String,
    /// `completed` / `interrupted` / …。它决定 `done` 的理由：`completed` → `end_turn`，
    /// `interrupted` → 内核已关的 turn 不再 `done`，其余 → `error: codex turn <status>`。
    #[serde(default)]
    pub status: String,
}

// ------------------------------------------------------------ codex → client（请求）

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    #[serde(default)]
    pub item_id: String,
    /// argv 数组或一整行字符串——Codex 两种都发得出来，原样转给审批服务去看。
    #[serde(default)]
    pub command: Value,
    #[serde(default)]
    pub cwd: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Accept,
    Decline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalReply {
    pub decision: ApprovalDecision,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
    #[serde(default)]
    pub call_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolReply {
    /// 工具的输出。Codex 这条线上是**字符串**，所以结构化的结局（`Outcome`）要先
    /// 序列化成 JSON 文本再放进来。
    pub output: String,
    pub success: bool,
}

impl ToolReply {
    /// 一次没跑起来的调用：给一句短话 + `success: false`。
    pub fn failed(output: &str) -> Self {
        ToolReply {
            output: output.to_string(),
            success: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_method_names_are_codex_s_own_spelling() {
        // camelCase 的段不能被这个仓库的 snake_case 习惯改写掉——改了就谁也调不通。
        assert_eq!(AGENT_MESSAGE_DELTA, "item/agentMessage/delta");
        assert_eq!(REQUEST_APPROVAL, "item/commandExecution/requestApproval");
        assert_eq!(ITEM_AGENT_MESSAGE, "agentMessage");
    }

    #[test]
    fn turn_start_serialises_to_codex_s_shape() {
        let v = serde_json::to_value(TurnStart {
            thread_id: "t1".into(),
            input: vec![Input::Text {
                text: "hello".into(),
            }],
            additional_context: None,
        })
        .unwrap();
        assert_eq!(
            v,
            json!({"thread_id": "t1", "input": [{"type": "text", "text": "hello"}]})
        );
    }

    #[test]
    fn a_decision_is_accept_or_decline_on_the_wire() {
        let v = serde_json::to_value(ApprovalReply {
            decision: ApprovalDecision::Decline,
        })
        .unwrap();
        assert_eq!(v, json!({"decision": "decline"}));
    }

    #[test]
    fn a_notification_missing_fields_still_parses() {
        // 对端少发一个字段不该让一次 turn 变成 panic。
        let d: AgentMessageDelta = serde_json::from_value(json!({"delta": "hi"})).unwrap();
        assert_eq!(d.delta, "hi");
        // 缺字段 ≠ 空串：`None` 是"没说属于谁"，由脑按当前 turn 算。
        assert_eq!(d.turn_id, None);
        let c: ItemCompleted = serde_json::from_value(json!({})).unwrap();
        assert_eq!(c.item.item_type, "");
    }

    #[test]
    fn a_delta_that_names_a_turn_keeps_the_name() {
        let d: AgentMessageDelta =
            serde_json::from_value(json!({"turn_id": "u2", "delta": "x"})).unwrap();
        assert_eq!(d.turn_id.as_deref(), Some("u2"));
    }
}
