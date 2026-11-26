//! 脑这一侧的内置插件。
//!
//! - [`native_brain`]：`brain-native`——原生 turn 循环接到内核的 `brains` 服务上；
//! - [`codex_brain`]：`brain-codex`——一个 `codex app-server` 子进程接到同一个服务上；
//! - [`teacher_brain`]：`brain-teacher`——复审老师的脑内通道（`review.teacher.channel: brain`）；
//! - [`scripted_model`]：剧本模型，测试与 `dev-native` profile 用（没有 API key 也能把
//!   整条链跑一遍）。

pub mod codex_brain;
pub mod native_brain;
pub mod scripted_model;
pub mod teacher_brain;
