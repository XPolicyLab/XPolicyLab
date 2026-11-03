//! `oeh-brain`：脑这一侧。
//!
//! 内核只知道"脑是一个 [`BrainSink`](oeh_kernel::BrainSink)：给它一个 observation 和一个
//! `TurnHandle`，它自己去想"。这个 crate 装的就是"自己去想"的那部分：
//!
//! - [`model`]：四种 LLM wire 共用的请求/事件/错误合约（Task 3 的传输实现它）；
//! - [`inbox`]：规格 §7.2 的三 lane 收件箱与 claim 规则；
//! - [`tools`]：模型看到的工具表——能力面的投影、沙箱化的 recall、`ask_slow_brain`，
//!   以及"安全调用并行、exclusive 调用是屏障"的批执行。
//!
//! 这三块都不认识具体的厂商，也不认识具体的机器人：能力从
//! `Admission::capabilities().list()` 投进来，模型从 provider 声明里造出来。把它们串
//! 起来的是：
//!
//! - [`transport`]：四种 wire 的实现（`chat_completions` / `anthropic_messages` /
//!   `responses` / `realtime`）；
//! - [`providers`]：`providers.yaml`——厂商是声明，不是代码；
//! - [`retry`]：重试与 fallback 瀑布；
//! - [`native`]：原生 turn 循环——`BrainSink` 的实现、字节级稳定的 prompt 前缀、
//!   step 步进与工具批；
//! - [`codex`]：另一种脑——一个 `codex app-server` 子进程，走 JSON-RPC 而不是 LLM wire；
//! - [`builtin`]：`brain-native` / `brain-codex` 插件与剧本模型；
//! - [`teacher`]：复审老师的 Codex app-server 客户端（Python `codex_teacher.py` 的移植）与它的脑内通道
//!   （`review.teacher.channel: brain`，`brain-teacher` 插件）。
//!
//! 最后还有一块不属于"想"而属于"改自己"的：[`rsi`]——规格 §10.1 的第一级自改进，
//! skills / memory 的受护栏写入（read-before-write、所有权、账本、回滚、受保护路径）。

pub mod builtin;
pub mod codex;
pub mod inbox;
pub mod model;
pub mod native;
pub mod providers;
pub mod retry;
pub mod rsi;
pub mod teacher;
pub mod tools;
pub mod transport;

pub use codex::{CodexBrain, CodexConfig, CodexError};
pub use inbox::{Inbox, InboxItem, Lane};
pub use model::{
    EventStream, Message, Model, ModelError, ModelEvent, ModelRequest, StopReason, ToolCall,
    ToolSpec,
};
pub use native::{NativeBrain, NativeConfig};
pub use providers::{ApiMode, ProviderDecl, Providers, Runtime};
pub use retry::{RetryEvent, RetryPolicy};
pub use rsi::{Doc, KnowledgeStore, ProtectedPaths, RsiError};
pub use tools::{ToolFuture, ToolHandler, ToolOutcome, ToolRegistry};
