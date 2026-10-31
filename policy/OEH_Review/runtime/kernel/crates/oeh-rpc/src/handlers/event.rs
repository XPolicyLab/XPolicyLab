//! `event/emit`：让跨进程的客户端往世界日志里写一条事件。
//!
//! 两道闸门，都是安全属性而不是便利性检查：
//!
//! 1. **身份不由调用方决定**。`source` 永远是 `rpc:<这条连接的名字>`，调用方在 params 里
//!    写什么都不算数。这不只是审计洁癖：`Brains` 把 `source == "brains"` 的
//!    `brain/steer`/`brain/interrupt` 当作**定向**命令——绕过目标脑的 `WakePolicy` 和唤醒
//!    预算直接操纵它。一个能自报 source 的 `event/emit` 就是这条策略的后门：任何客户端
//!    都能伪造一条"来自内核"的定向命令去驱动别人的脑。`EventDraft::new` 在这里拿到的
//!    source 只有一个来源，就是连接自己。
//! 2. **保留的种类不许伪造**。`act/*` 是准入流水线的记账（提案、判定、执行的全过程），
//!    `safety/fsm` 是状态机迁移的唯一凭据，`safety/tamper_attempt` 是"有规则想放水"的
//!    告警——三者都是别的子系统**据以做判断**的事实，不是观测。伪造它们能让规则链、
//!    FSM 对账和审计同时失真，所以是 `REFUSED`（策略拒绝），不是参数错误。
//!
//! 写入直接走 [`oeh_kernel::WorldLog::append`]，不经 `Op::Emit`：调用方要的是这条事件的
//! `seq`，而 Op 通道是单向的（投进去就没有回执）。代价是这次 `append` 的观察者跑在当前
//! 任务上——它们全都只做非阻塞的 `try_send` / `conn.notify`，这一点由观察者那一侧的契约
//! 保证（见 `WorldLog::observe`）。

use std::sync::Arc;

use oeh_proto::rpc::types::{EventEmit, EventEmitResult};
use oeh_proto::rpc::RpcError;
use oeh_proto::{EventDraft, EventKind};
use serde_json::Value;

use crate::conn::Conn;
use crate::handlers::{invalid_params, ok, refused};
use crate::router::{parse_params, RouterConfig, Services};

/// 没给 `salience` 时的默认值：中位。太高会挤掉真正的告警，太低会在观测预算紧张时被
/// 第一个丢掉。
const DEFAULT_SALIENCE: u8 = 100;

/// `kind` / `collapse_key` / `trace_id` 每个的字节上限。
///
/// 这三个字符串和 `payload` 一样是**被留下来**的：它们一起进 `WorldEvent`，进那个按条数
/// （4096 条）保留的环，进每个 `log/tail` 观察者的通知，进每个 attach 着的脑的收件箱。
/// `EventKind::new` 只验 `<已知前缀>/<非空>`，`collapse_key`/`trace_id` 连格式都不验，所以
/// 在补上 payload 的上限之后，同一个放大还能从这三个字段里走一遍（最终评审再审 R1，
/// 实测：100 KiB 的 `collapse_key` 和 100 KiB 的 `kind` 都能越过 64 KiB 的 payload 上限）。
///
/// 256 字节：一条 kind 是 `obs/person` 这样的东西，一个折叠键是 `stale:lowstate`，一个
/// trace id 是一个 uuid——都远在这之内。单独设一个短上限而不是只靠总量，是因为总量允许
/// 一条"payload 很小、kind 有 64 KiB"的事件，而那种事件只可能是错的。
const MAX_EVENT_STRING_BYTES: usize = 256;

/// 这条 kind 是不是别的子系统的记账，客户端不许自己写。
///
/// `capability/*` 整个前缀都在里面（`overridden` / `restored` / `registered` /
/// `unregistered`，以及以后的）：能力面易主是**注册表自己**在改表的那个临界区里记下的
/// 事实，一条伪造的 `capability/unregistered` 会让读日志的人以为某台适配器走了，而一条
/// 伪造的 `capability/registered` 会让一个进程外的脑去调一件根本不在台面上的事。
/// `safety/adapter_lost` 与 `safety/auth_refused` 同理：前者是内核判定的失联，后者是
/// 边界拒绝了一个 uid——两条都是"关于连接的事实"，而伪造它们的正是连接自己。
fn is_reserved(kind: &str) -> bool {
    kind.starts_with("act/")
        || kind.starts_with("capability/")
        || matches!(
            kind,
            "safety/fsm" | "safety/tamper_attempt" | "safety/adapter_lost" | "safety/auth_refused"
        )
}

pub fn emit(
    s: &Services,
    cfg: &RouterConfig,
    conn: &Arc<Conn>,
    params: Value,
) -> Result<Value, RpcError> {
    let p: EventEmit = parse_params(&params)?;
    // 种类检查在合法性检查之前：`act/proposed` 本身是一条合法的 kind，它被挡下来的理由
    // 是策略（REFUSED），不是拼写（INVALID_PARAMS），两者的 code 不该混。
    if is_reserved(&p.kind) {
        return Err(refused(format!("`{}` is written by the kernel", p.kind)));
    }
    // 三个会被留下来的字符串各自的上限，见 [`MAX_EVENT_STRING_BYTES`]。
    for (what, value) in [
        ("kind", Some(p.kind.as_str())),
        ("collapse_key", p.collapse_key.as_deref()),
        ("trace_id", p.trace_id.as_deref()),
    ] {
        if let Some(v) = value {
            if v.len() > MAX_EVENT_STRING_BYTES {
                return Err(invalid_params(format_args!(
                    "`{what}` is {} bytes, over the {MAX_EVENT_STRING_BYTES} byte limit",
                    v.len()
                )));
            }
        }
    }
    // 字节上限（见 `RouterConfig::max_event_bytes`）。世界日志按**条数**保留，不按字节：
    // 4096 条各带一个 16 MiB 的载荷就是 64 GiB 常驻，外加每个 tail 观察者、每个 attach 着
    // 的脑各一份克隆（Plan 3 最终评审 I3）。`plugin/register` 早就有同一形状的护栏。
    //
    // 量的是这条连接决定的、**会被留下来**的那些字节——payload 加上那三个字符串——而不是
    // 整个 params（`salience` 是一个字节的枚举，不留在这里算）：第一版只量了 payload，于是
    // 同一个放大从 `kind` 里原样走了一遍（再审 R1）。
    let payload_bytes = serde_json::to_vec(&p.payload).map(|v| v.len()).unwrap_or(0);
    let retained = payload_bytes
        + p.kind.len()
        + p.collapse_key.as_deref().map_or(0, str::len)
        + p.trace_id.as_deref().map_or(0, str::len);
    if retained > cfg.max_event_bytes {
        return Err(invalid_params(format_args!(
            "event retains {retained} bytes (payload {payload_bytes} + kind/collapse_key/trace_id), \
             over the {} byte limit",
            cfg.max_event_bytes
        )));
    }
    let kind = EventKind::new(&p.kind).map_err(invalid_params)?;
    // `format!` 出来的这个 source 是唯一的一份：params 里就算带了 `source`，也在
    // `EventEmit` 反序列化的那一刻就被丢掉了（它没有这个字段），这里再覆盖一次是为了
    // 让"谁决定 source"这件事在代码里看得见，而不是依赖一个结构体碰巧少了个字段。
    let source = format!("rpc:{}", conn.name());
    let mut draft = EventDraft::new(kind, source, p.payload)
        .with_salience(p.salience.unwrap_or(DEFAULT_SALIENCE));
    if let Some(k) = p.collapse_key {
        draft = draft.with_collapse_key(k);
    }
    if let Some(t) = p.trace_id {
        draft = draft.with_trace_id(t);
    }
    let ev = s.log.append(draft);
    ok(EventEmitResult { seq: ev.seq })
}

#[cfg(test)]
mod tests {
    use super::is_reserved;

    #[test]
    fn reserved_kinds_are_the_kernels_own_bookkeeping() {
        for k in [
            "act/proposed",
            "act/result",
            "act/",
            "safety/fsm",
            "safety/tamper_attempt",
            "safety/adapter_lost",
            "safety/auth_refused",
            "capability/overridden",
            "capability/restored",
            "capability/registered",
            "capability/unregistered",
            // 整个前缀，不是一张名单：下一条 `capability/*` 也该是内核自己写的，而一
            // 张名单要靠加它的人记得回来改这里。
            "capability/whatever_comes_next",
        ] {
            assert!(is_reserved(k), "{k}");
        }
        // 观测、健康度和脑自己的事件都是客户端该写的。`safety/estop` 也是：按下急停的
        // 人不一定是内核（外部急停按钮的监听器就是一个 RPC 客户端）。
        for k in [
            "obs/person",
            "health/stale",
            "brain/mock_turn",
            "safety/estop",
            "patch/apply",
            // 前缀比的是**段**，不是字符串：一个叫 `capabilities/...` 的种类不该被
            // `capability/` 顺手挡掉。
            "capabilities/list",
        ] {
            assert!(!is_reserved(k), "{k}");
        }
    }
}
