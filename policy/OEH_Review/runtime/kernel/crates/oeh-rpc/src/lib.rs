//! 边界协议的 Rust 实现：行编解码（[`codec`]）、连接 actor（[`conn`]）、方法路由
//! （[`router`] 的表 + [`handlers`] 的实现）、最小 Rust 客户端（[`client`]）、UDS/ws/stdio
//! 监听器与活跃连接表（[`transport`]），以及把它们接到内核服务上的 `rpc-server` 插件
//! （[`plugin`]）。
//!
//! 边界协议的方法全部接上了：`robot/*`、`brain/*`、`turn/*`、`stream/*`（`subscribe` 的
//! 帧走 iceoryx2 共享内存，[`frames`]）、`event/emit`、`log/*`、`plugin/*` 与
//! `capability/*`（跨进程插件与执行器，[`remote`]）、`approval/*`。

pub mod client;
pub mod codec;
pub mod conn;
pub mod frames;
pub mod handlers;
pub mod plugin;
pub mod remote;
pub mod router;
pub mod transport;
