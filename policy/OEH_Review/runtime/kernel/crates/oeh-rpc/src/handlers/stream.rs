//! `stream/list` · `stream/subscribe`——流的声明面，和把一条流接到共享内存通道上的那一步。
//!
//! 帧本身一帧都不走这条 JSON 路：`stream/subscribe` 回的是一个**通道名**，客户端拿着它
//! 自己去 iceoryx2 上读（[`crate::frames`] 说明了两个方向各自是怎么搬的）。

use std::sync::Arc;

use oeh_proto::rpc::types::{StreamListResult, StreamSubscribe};
use oeh_proto::rpc::RpcError;
use serde_json::Value;

use crate::handlers::{internal, not_attached, ok};
use crate::router::{parse_params, Services};

/// 当前注册着的流，按 `stream_id` 排序（`FrameStore::streams` 已经排好）。只有声明
/// （`StreamSpec`），没有帧：帧不走 JSON 这条路。
pub fn list(s: &Services) -> Result<Value, RpcError> {
    ok(StreamListResult {
        streams: s.frames.streams(),
    })
}

/// 把一条流接到通道上，回 `{stream_id, channel, schema_id}`。
///
/// 三种情况：
///
/// - 流不存在（`schema_of` 是 `None`）→ `NOT_ATTACHED`。和"脑没 attach"同一个编码：
///   调用方要的那个东西现在不在这个内核上。给一个通道名反而更糟——客户端会拿着它去等
///   一条永远不会有帧的通道。
/// - 流由某个跨进程插件发布（[`crate::frames::Frames::is_remote`]）→ 直接把它的通道名
///   给出去。那条通道上已经有一个发布者（那个插件）和一个订阅者（内核的 ingress），
///   再多一个订阅者就是了，内核不该把帧收进来再原样发出去一遍。
/// - 其余（`mock-stream`、进程内适配器产的流）→ 懒起一条 egress。
///
/// `spawn_blocking`：第一次订阅一条本地流要建一个 iceoryx2 Node、开共享内存段、起一条
/// 发布线程并等它就绪——毫秒级，但那是**阻塞**的毫秒。放在 tokio 的 worker 上就是让同一
/// 个 worker 上排队的其他请求（心跳、结果）跟着等。
pub async fn subscribe(s: Arc<Services>, params: Value) -> Result<Value, RpcError> {
    let p: StreamSubscribe = parse_params(&params)?;
    if s.frames.schema_of(&p.stream_id).is_none() {
        return Err(not_attached(format_args!("stream `{}`", p.stream_id)));
    }
    let remote = s.channels.is_remote(&p.stream_id);
    let channels = s.channels.clone();
    let stream_id = p.stream_id.clone();
    let ch = tokio::task::spawn_blocking(move || channels.subscribe(&stream_id, remote))
        .await
        .map_err(internal)?;
    match ch {
        Ok(ch) => ok(ch),
        // 流在这两步之间没了（发布它的插件正好断线）：那还是"你要的东西不在这个内核上"，
        // 和上面那次检查是同一件事，不该因为它是在另一个线程里发现的就变成 `INTERNAL`。
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(not_attached(format_args!("stream `{}`", p.stream_id)))
        }
        // 通道起不来（共享内存开不出来、发布者位置满了、建服务超时）是 `INTERNAL`：
        // 客户端没做错任何事，重试也许会成，但这台机器上确实出了问题。
        Err(e) => Err(internal(e)),
    }
}
