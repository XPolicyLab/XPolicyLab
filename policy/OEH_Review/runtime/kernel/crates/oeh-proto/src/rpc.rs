//! 边界协议的线格式：一个 JSON-RPC 2.0 信封（`Request`/`Response`/`Notification`），
//! 一张方法名常量表（`method`），以及每个方法 params/result 的具体类型（`types`）。
//!
//! 这是 Plan 3 全部 Rust 与 Python 代码的线格式依据：Rust 侧的 RPC server crate 与
//! Python SDK 都对着这份表和 `fixtures/rpc/*.json` 序列化，不各自发明字段。
//!
//! # 方法表
//!
//! | 方法 | params | result / 说明 |
//! |---|---|---|
//! | robot/capabilities | Empty | RobotCapabilitiesResult |
//! | robot/state | Empty | RobotStateResult |
//! | robot/health | Empty | RobotHealthResult |
//! | brain/attach | BrainAttach | BrainAttachResult |
//! | brain/detach | BrainRef | Empty |
//! | brain/cursor | BrainRef | BrainCursorResult |
//! | brain/wake_policy/update | WakePolicyUpdate | Empty |
//! | brain/inject | BrainInject | BrainInjectResult |
//! | turn/steer | TurnSteer | Empty |
//! | turn/interrupt | BrainRef | Empty |
//! | turn/propose | TurnPropose | TurnProposeResult |
//! | turn/say | TurnSay | TurnProposeResult |
//! | turn/ask_human | TurnAskHuman | TurnProposeResult |
//! | turn/done | TurnDone | Empty |
//! | stream/list | Empty | StreamListResult |
//! | stream/subscribe | StreamSubscribe | StreamChannel |
//! | plugin/register | PluginRegister | PluginRegisterResult |
//! | plugin/heartbeat | FiberRef | Empty |
//! | plugin/unregister | FiberRef | Empty |
//! | event/emit | EventEmit | EventEmitResult |
//! | log/tail | LogTail | LogTailResult |
//! | log/replay | LogReplay | LogReplayResult |
//! | log/flush | Empty | Empty |
//! | approval/attach | ApprovalAttach | Empty |
//! | approval/detach | Empty | Empty |
//! | approval/request（k→c） | ApprovalRequest | ApprovalDecision |
//! | approval/detached（k→c 通知） | ApprovalDetached | — |
//! | capability/execute（k→c） | CapabilityExecute | CapabilityAccepted |
//! | capability/heartbeat（c→k 通知） | ExecRef | — |
//! | capability/result（c→k 通知） | CapabilityResult | — |
//! | capability/cancel（k→c 通知） | CapabilityCancel | — |
//! | brain/turn（k→c 通知） | BrainTurn | — |
//! | brain/steer（k→c 通知） | BrainSteerNotice | — |
//! | brain/interrupt（k→c 通知） | BrainRef | — |
//! | event（k→c 通知） | WorldEvent | — |
//! | plugin/state（k→c 通知） | FiberStateInfo | — |
//! | shutdown（k→c 通知） | ShutdownNotice | — |
//!
//! 规格 §6.2 里列而这张表里**没有**的方法只有一个：`turn/start` 不实现——turn 由内核
//! 推送（`brain/turn` 带着 Observation 和 `wake_reason` 过来），客户端用 `turn/*` 回应，
//! 所以不存在"客户端发起一个 turn"这件事；§7.3 里的 `turn/start` 是 Codex app-server
//! 那一侧的方法，不是这条边界上的。
//!
//! `robot/state` 的 `fsm`、`brain/turn`/`brain/steer` 里 `observation.state["fsm"]`
//! 的取值是 `FsmState`（`oeh-kernel`）的线格式：`SCREAMING_SNAKE_CASE`（如
//! `"STANDING"`/`"ACTING"`/`"EMERGENCY_STOP"`），不是小写。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Manifest, WorldEvent};

pub const JSONRPC: &str = "2.0";

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Id {
    Num(u64),
    Str(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    pub id: Id,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Notification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Id,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// 顺序有讲究：Request 要 id+method；Response 要 id 且没 method；Notification 只有 method。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Message {
    Request(Request),
    Response(Response),
    Notification(Notification),
}

impl Request {
    pub fn new(id: Id, method: &str, params: Value) -> Self {
        Self {
            jsonrpc: JSONRPC.into(),
            id,
            method: method.into(),
            params,
        }
    }
}
impl Notification {
    pub fn new(method: &str, params: Value) -> Self {
        Self {
            jsonrpc: JSONRPC.into(),
            method: method.into(),
            params,
        }
    }
}
impl Response {
    pub fn ok(id: Id, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC.into(),
            id,
            result: Some(result),
            error: None,
        }
    }
    pub fn err(id: Id, e: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC.into(),
            id,
            result: None,
            error: Some(e),
        }
    }
}
impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
    pub fn with_data(mut self, d: Value) -> Self {
        self.data = Some(d);
        self
    }
}

pub mod code {
    pub const PARSE: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL: i64 = -32603;
    /// 内核 Op 通道已关：fail-closed。
    pub const KERNEL_GONE: i64 = -32000;
    /// 这条连接上没有这个脑 / turn / fiber。
    pub const NOT_ATTACHED: i64 = -32001;
    /// 保留的事件种类、重复注册之类的策略性拒绝。
    pub const REFUSED: i64 = -32002;
    pub const TIMEOUT: i64 = -32003;
}

/// 方法名常量：Rust 与 Python 的路由表都从这份表来（Python 侧手抄同名字符串，fixtures 保证一致）。
pub mod method {
    pub const ROBOT_CAPABILITIES: &str = "robot/capabilities";
    pub const ROBOT_STATE: &str = "robot/state";
    pub const ROBOT_HEALTH: &str = "robot/health";
    pub const BRAIN_ATTACH: &str = "brain/attach";
    pub const BRAIN_DETACH: &str = "brain/detach";
    pub const BRAIN_CURSOR: &str = "brain/cursor";
    pub const BRAIN_WAKE_POLICY_UPDATE: &str = "brain/wake_policy/update";
    pub const BRAIN_INJECT: &str = "brain/inject";
    pub const TURN_STEER: &str = "turn/steer";
    pub const TURN_INTERRUPT: &str = "turn/interrupt";
    pub const TURN_PROPOSE: &str = "turn/propose";
    pub const TURN_SAY: &str = "turn/say";
    pub const TURN_ASK_HUMAN: &str = "turn/ask_human";
    pub const TURN_DONE: &str = "turn/done";
    pub const STREAM_LIST: &str = "stream/list";
    pub const STREAM_SUBSCRIBE: &str = "stream/subscribe";
    pub const PLUGIN_REGISTER: &str = "plugin/register";
    pub const PLUGIN_HEARTBEAT: &str = "plugin/heartbeat";
    pub const PLUGIN_UNREGISTER: &str = "plugin/unregister";
    pub const EVENT_EMIT: &str = "event/emit";
    pub const LOG_TAIL: &str = "log/tail";
    pub const LOG_REPLAY: &str = "log/replay";
    pub const LOG_FLUSH: &str = "log/flush";
    pub const APPROVAL_ATTACH: &str = "approval/attach";
    pub const APPROVAL_DETACH: &str = "approval/detach";
    /// k→c 通知：这条连接装的审批服务被别人接管了（`approval/attach` 是抢占式的）。
    pub const APPROVAL_DETACHED: &str = "approval/detached";
    // kernel → client 请求
    pub const APPROVAL_REQUEST: &str = "approval/request";
    pub const CAPABILITY_EXECUTE: &str = "capability/execute";
    // client → kernel 通知（执行期间）
    pub const CAPABILITY_HEARTBEAT: &str = "capability/heartbeat";
    pub const CAPABILITY_RESULT: &str = "capability/result";
    // kernel → client 通知
    pub const CAPABILITY_CANCEL: &str = "capability/cancel";
    pub const BRAIN_TURN: &str = "brain/turn";
    pub const BRAIN_STEER: &str = "brain/steer";
    pub const BRAIN_INTERRUPT: &str = "brain/interrupt";
    pub const EVENT: &str = "event";
    pub const PLUGIN_STATE: &str = "plugin/state";
    pub const SHUTDOWN: &str = "shutdown";
    /// 全表（fixtures 测试用它保证每个方法都有 fixture）。
    pub const ALL: &[&str] = &[
        ROBOT_CAPABILITIES,
        ROBOT_STATE,
        ROBOT_HEALTH,
        BRAIN_ATTACH,
        BRAIN_DETACH,
        BRAIN_CURSOR,
        BRAIN_WAKE_POLICY_UPDATE,
        BRAIN_INJECT,
        TURN_STEER,
        TURN_INTERRUPT,
        TURN_PROPOSE,
        TURN_SAY,
        TURN_ASK_HUMAN,
        TURN_DONE,
        STREAM_LIST,
        STREAM_SUBSCRIBE,
        PLUGIN_REGISTER,
        PLUGIN_HEARTBEAT,
        PLUGIN_UNREGISTER,
        EVENT_EMIT,
        LOG_TAIL,
        LOG_REPLAY,
        LOG_FLUSH,
        APPROVAL_ATTACH,
        APPROVAL_DETACH,
        APPROVAL_DETACHED,
        APPROVAL_REQUEST,
        CAPABILITY_EXECUTE,
        CAPABILITY_HEARTBEAT,
        CAPABILITY_RESULT,
        CAPABILITY_CANCEL,
        BRAIN_TURN,
        BRAIN_STEER,
        BRAIN_INTERRUPT,
        EVENT,
        PLUGIN_STATE,
        SHUTDOWN,
    ];
}

/// 每个方法的 params / result。`Value` 字段是别的 crate 的类型（Proposal / Outcome /
/// CapabilitySpec / Observation / WakePolicy）——proto 不能依赖它们，线格式就是它们的 serde JSON。
pub mod types {
    use super::*;
    use crate::StreamSpec;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct Empty {}

    /// 一条能力，两个方向：`plugin/register` 里是客户端的**声明**，`robot/capabilities`
    /// 里是内核的**快照**。最后两个字段只在后一个方向上有意义。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct CapabilityDecl {
        pub name: String,
        /// "motion" | "speech" | "query" | "control"
        pub class: String,
        /// "none" | "low" | "medium" | "high"
        pub risk_level: String,
        #[serde(default)]
        pub params_schema: Value,
        #[serde(default)]
        pub lease_ttl_s: f64,
        /// 谁提供它：内置是 `"kernel"`，跨进程插件是它的 `manifest.name`。一条
        /// `say` 到底是从扬声器出来还是只进了世界日志，答案就在这个字段里。
        ///
        /// 注册方向上写什么都不算数——归属由内核决定，不由声明它的人决定。
        #[serde(default)]
        pub provider: String,
        /// 这一项允许被一个适配器的同类（speech）能力顶替吗。只有内置 `say` 是 true：
        /// 它是"还没有机器人接上来"时的兜底。同样，注册方向上写什么都不算数。
        #[serde(default)]
        pub overridable: bool,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct RobotCapabilitiesResult {
        pub capabilities: Vec<CapabilityDecl>,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct RobotStateResult {
        pub fsm: String,
        pub state: Value,
        pub t_kernel: u64,
    }

    /// `state` 是 `FiberState`（`oeh-kernel`）的线格式：`"pending"` | `"active"` |
    /// `"failed"` | `"disposed"`（`"loading"` 在枚举里存在，但只在一次 `settle()`
    /// 内部短暂出现，`plugin/state`/`robot/health` 永远不会广播它——消费者不应该等它
    /// 出现）。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct FiberStateInfo {
        pub fiber_id: u32,
        pub plugin: String,
        pub state: String,
        #[serde(default)]
        pub error: Option<String>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct StreamHealth {
        pub stream_id: String,
        pub schema: String,
        pub nominal_hz: f64,
        pub frames: u64,
        pub age_ms: Option<u64>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct RobotHealthResult {
        pub fibers: Vec<FiberStateInfo>,
        pub streams: Vec<StreamHealth>,
        pub log_sink_errors: u64,
        pub brains_dropped_events: u64,
        pub connections: u32,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainAttach {
        pub brain_id: String,
        #[serde(default)]
        pub wake_policy: Value,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainAttachResult {
        pub cursor: u64,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainRef {
        pub brain_id: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainCursorResult {
        pub cursor: u64,
        pub status: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct WakePolicyUpdate {
        pub brain_id: String,
        pub wake_policy: Value,
    }
    /// `brain/inject` 的 params。注入方的身份**不在**这里：内核按连接的名字填
    /// （`rpc:<name>`），和 `event/emit` 同一条规矩——调用方说了不算。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainInject {
        pub brain_id: String,
        pub text: String,
    }
    /// 那条 `brain/inject` 事件的 seq：调用方拿它去和世界日志（或 `event` 通知流）对上
    /// 号，而不是靠事后按 kind + 文本去猜是哪一条。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainInjectResult {
        pub seq: u64,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct TurnSteer {
        pub brain_id: String,
        pub text: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct TurnPropose {
        pub brain_id: String,
        pub turn: u64,
        pub capability: String,
        #[serde(default)]
        pub params: Value,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct TurnSay {
        pub brain_id: String,
        pub turn: u64,
        pub text: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct TurnAskHuman {
        pub brain_id: String,
        pub turn: u64,
        pub question: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct TurnDone {
        pub brain_id: String,
        pub turn: u64,
        #[serde(default)]
        pub reason: String,
    }
    /// `turn/propose` / `turn/say` / `turn/ask_human` 的 result：`Outcome` 的 serde JSON。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct TurnProposeResult {
        pub proposal_id: String,
        pub outcome: Value,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct StreamChannel {
        pub stream_id: String,
        pub channel: String,
        pub schema_id: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct StreamListResult {
        pub streams: Vec<StreamSpec>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct StreamSubscribe {
        pub stream_id: String,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct PluginRegister {
        pub manifest: Manifest,
        #[serde(default)]
        pub capabilities: Vec<CapabilityDecl>,
        #[serde(default)]
        pub config: Value,
        /// 客户端承诺的心跳周期；0 = 用内核默认（1000）。
        #[serde(default)]
        pub heartbeat_ms: u64,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct PluginRegisterResult {
        pub fiber_id: u32,
        pub state: String,
        #[serde(default)]
        pub error: Option<String>,
        pub streams: Vec<StreamChannel>,
        pub heartbeat_ms: u64,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct FiberRef {
        pub fiber_id: u32,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct EventEmit {
        pub kind: String,
        #[serde(default)]
        pub payload: Value,
        #[serde(default)]
        pub salience: Option<u8>,
        #[serde(default)]
        pub collapse_key: Option<String>,
        #[serde(default)]
        pub trace_id: Option<String>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct EventEmitResult {
        pub seq: u64,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct LogTail {
        #[serde(default)]
        pub after: Option<u64>,
        #[serde(default)]
        pub prefixes: Vec<String>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct LogTailResult {
        pub from: u64,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct LogReplay {
        #[serde(default)]
        pub dir: Option<String>,
        #[serde(default)]
        pub after: u64,
        #[serde(default = "d_limit")]
        pub limit: usize,
    }
    fn d_limit() -> usize {
        1000
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct LogReplayResult {
        pub events: Vec<WorldEvent>,
        pub done: bool,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ApprovalAttach {
        #[serde(default)]
        pub name: String,
    }
    /// `approval/detached` 的 params：这条连接不再是审批服务了，以及为什么
    /// （目前只有 `"replaced"`——另一条连接接管了）。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ApprovalDetached {
        pub reason: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ApprovalRequest {
        pub ask_id: String,
        pub proposal: Value,
        pub spec: Value,
        pub reason: String,
        #[serde(default)]
        pub trace: Value,
    }
    /// `approval/request` 的响应：`{"decision":"allow"|"deny","by":..,"reason":..}`（Admission 的 `Decision`）。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ApprovalDecision {
        pub decision: String,
        pub by: String,
        pub reason: String,
    }

    /// 不变量：`params == proposal.params`（`Proposal` 的 `params` 字段）——两者都是
    /// 规则链跑完之后（夹过限幅/限速的）那一份。读 `proposal.params` 而不是顶层
    /// `params` 的执行器会悄悄绕过规则链——这正是这个不变量要防的事故
    /// （见 `oeh-admission` 的 `ExecutionCall` 文档）。
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct CapabilityExecute {
        pub exec_id: String,
        pub capability: String,
        pub params: Value,
        pub proposal: Value,
        pub lease_ttl_s: f64,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct CapabilityAccepted {
        pub accepted: bool,
        #[serde(default)]
        pub reason: Option<String>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ExecRef {
        pub exec_id: String,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct CapabilityResult {
        pub exec_id: String,
        pub ok: bool,
        #[serde(default)]
        pub result: Value,
        #[serde(default)]
        pub error: Option<String>,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct CapabilityCancel {
        pub exec_id: String,
        pub reason: String,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainTurn {
        pub brain_id: String,
        pub turn: u64,
        pub trace_id: String,
        pub observation: Value,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct BrainSteerNotice {
        pub brain_id: String,
        pub observation: Value,
    }
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ShutdownNotice {
        pub reason: String,
    }
}
