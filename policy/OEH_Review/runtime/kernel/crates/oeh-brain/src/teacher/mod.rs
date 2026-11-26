//! 复审老师（review teacher）的 Rust 侧。
//!
//! - [`codex_app_server`]：`python/oeh-policy/oeh_policy/codex_teacher.py` 里 `CodexClient` /
//!   `CodexConversation` 的逐行移植，对真实 `codex app-server` 说一模一样的 JSON-RPC。它和
//!   [`crate::codex::CodexBrain`] 是两回事，互不影响。
//! - [`lane`]：`review.teacher.channel: brain` 时替代 Python `TeacherBrain` 的脑（`brain-teacher`
//!   插件，[`crate::builtin::teacher_brain`]）：模型循环在这里，运行时状态与决策校验仍在 Python。
//! - [`pyjson`]：调用记录要的 Python 字节（`encoded` / `digest` / `json.dumps` / `redact`）。

pub mod codex_app_server;
pub mod lane;
pub mod pyjson;

pub use codex_app_server::{
    CloseHandle, CodexClient, CodexConversation, LaunchPolicy, TeacherConfig, TeacherError,
};
pub use lane::{LaneConfig, TeacherLane};
