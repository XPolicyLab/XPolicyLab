//! RSI 第一级：知识（skills / memory）的受护栏存储。
//!
//! 规格 §10.1 给这一级定了两件事：**存什么**（`$OEH_HOME/robots/<id>` 下的
//! `skills/**/SKILL.md`、`memory/MEMORY.md`、`memory/topics/*.md`，Markdown，grep 可读）
//! 和**护栏在哪**——"写入护栏（代码而非 prompt）"。后半句是这个模块存在的全部理由。
//!
//! 一个能改自己 skill 的脑，就是一个能改自己行为的脑。指望 prompt 里写一句"不要动
//! profile"来兜住它，等于把安全边界交给一个采样过程：它可以被说服、被越狱、被一次
//! 不走运的采样绕过。所以这里的每一条规则都落在 Rust 里，模型只能看见 `Err`：
//!
//! | 护栏 | 挡住的是 |
//! |---|---|
//! | read-before-write（[`KnowledgeStore::write`] 必须带上次读到的 sha256） | 基于陈旧内容的覆盖——review fork 和人可能同时在改同一个文件 |
//! | `created_by` 所有权（只管 skill） | 一个 fork 改另一个 fork（或人）写下的 skill |
//! | [`ledger`] + `.blobs/<sha256>` + [`KnowledgeStore::rollback`] | "改坏了但没人知道改了什么、也回不去" |
//! | 原子写（临时文件 + rename） | 崩在写一半时留下半个 SKILL.md，下次启动喂给模型 |
//! | [`protected`] 路径 + `safety/tamper_attempt` | 拿 skill 的写入口去改 profile / bundle / patch / wake 策略 / 规则 / prompt 模板 |
//!
//! 所有权那一条只管 skill：`memory/` 是这台机器人的记忆，不是某一个 fork 的私产，快脑记
//! 下的一条偏好慢脑必须能接着改；skill 是行为本身，§10.1 要求"fork 只能改自己创建的"。
//! 两边都照样入账。
//!
//! 路径检查分两层，顺序是固定的：先**句法**（白名单形状、没有 `..`、不是绝对路径），
//! 后**物理**（解析出来的父目录必须还在根里、路径上不能有符号链接）。句法层在任何一次
//! `stat` 之前就挡住了绝大多数越界写，物理层管的是"根目录里被人塞了一条链接"这种只有
//! 摸过磁盘才看得出来的情况。两层的失败都是 [`RsiError::BadPath`]，两层都不落盘。

pub mod ledger;
pub mod protected;
pub mod store;

pub use ledger::{LedgerEntry, Op, LEDGER_FILE};
pub use protected::ProtectedPaths;
#[cfg(feature = "test-hooks")]
pub use store::FaultPoint;
pub use store::{sha256_hex, Doc, Kind, KnowledgeStore, RsiError};
