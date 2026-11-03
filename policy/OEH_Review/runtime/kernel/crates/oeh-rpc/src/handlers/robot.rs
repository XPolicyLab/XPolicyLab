//! `robot/capabilities` · `robot/state` · `robot/health`——一台机器人"现在能做什么、
//! 正处在什么状态、哪里不对劲"的三个只读入口。全部同步：读一把锁、拷一份快照，不
//! `.await`，也不碰内核 Op 通道。

use oeh_proto::rpc::types::{
    CapabilityDecl, RobotCapabilitiesResult, RobotHealthResult, RobotStateResult, StreamHealth,
};
use oeh_proto::rpc::RpcError;
use serde_json::Value;

use crate::handlers::{ok, wire_name};
use crate::router::{FiberStates, Services};

/// 能力注册表的一份快照。`class`/`risk_level` 用 serde 的线格式名（`"motion"` /
/// `"medium"`），不是 Rust 的 `Debug` 名；按名字排序，这样两次调用之间顺序稳定
/// （注册表底下是 `HashMap`，迭代顺序每进程一换）。
///
/// `provider` / `overridable` 是这份快照里**只有内核说得出**的两件事：谁在执行这个名字，
/// 以及它让不让位。一个正要注册的适配器读的就是它们——内核自己兜底的 `say` 可以顶替，
/// `stop` 不行，而另一台机器人占着的名字既不能顶替也不该让出。
pub fn capabilities(s: &Services) -> Result<Value, RpcError> {
    let mut entries = s.admission.capabilities().list_entries();
    entries.sort_by(|a, b| a.spec.name.cmp(&b.spec.name));
    let capabilities = entries
        .into_iter()
        .map(|e| CapabilityDecl {
            name: e.spec.name,
            class: wire_name(&e.spec.class),
            risk_level: wire_name(&e.spec.risk_level),
            params_schema: e.spec.params_schema,
            lease_ttl_s: e.spec.lease_ttl_s,
            provider: e.provider,
            overridable: e.spec.overridable,
        })
        .collect();
    ok(RobotCapabilitiesResult { capabilities })
}

/// `fsm` 是 [`oeh_kernel::FsmState`] 的线格式（`SCREAMING_SNAKE_CASE`，如 `"STANDING"`），
/// `state` 是蒸馏状态寄存器的一次**整体**快照（`StateRegistry::snapshot` 一次加锁取齐，
/// 跨键一致），`t_kernel` 是取这份快照时的内核单调时间。
pub fn state(s: &Services) -> Result<Value, RpcError> {
    ok(RobotStateResult {
        fsm: s.fsm.state().to_string(),
        state: serde_json::to_value(s.state.snapshot()).unwrap_or(Value::Null),
        t_kernel: s.clock.0.now_ns(),
    })
}

/// 四路健康度：
///
/// - `fibers`：路由自己维护的 `plugin/state` 快照（见 [`FiberStates`]），不直接碰
///   `FiberTree`——那棵树归内核的 `run` 循环，跨线程读它需要一把这里不该有的锁。
/// - `streams`：`frames.streams()` 的每条流配上 `stats()` 的累计帧数，`age_ms` 是
///   **最后一帧**的年龄（`latest()` 的 `t_kernel` 和现在的差），一帧都没来过是 `null`
///   ——不是 0：`0` 会被读成"刚刚才来过一帧"，正好和真相相反。
/// - `log_sink_errors` / `brains_dropped_events`：两个背压/丢弃计数器。
/// - `connections`：此刻活着的边界连接数。
pub fn health(s: &Services, fibers: &FiberStates) -> Result<Value, RpcError> {
    let now = s.clock.0.now_ns();
    let streams = s
        .frames
        .streams()
        .into_iter()
        .map(|spec| {
            let stats = s.frames.stats(&spec.stream_id);
            let age_ms = s
                .frames
                .latest(&spec.stream_id)
                .map(|f| now.saturating_sub(f.t_kernel) / 1_000_000);
            StreamHealth {
                stream_id: spec.stream_id,
                schema: spec.schema,
                nominal_hz: spec.nominal_hz,
                frames: stats.map(|s| s.frames_total).unwrap_or(0),
                age_ms,
            }
        })
        .collect();
    ok(RobotHealthResult {
        fibers: fibers.snapshot(),
        streams,
        log_sink_errors: s.log.sink_errors(),
        brains_dropped_events: s.brains.dropped_events(),
        connections: s.conns.len() as u32,
    })
}
