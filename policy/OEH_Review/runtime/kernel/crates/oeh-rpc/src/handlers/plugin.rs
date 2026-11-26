//! `plugin/register` · `plugin/heartbeat` · `plugin/unregister`——一个跨进程插件的一生。
//!
//! `plugin/register` 是整个边界协议里唯一一个**改变内核树**的方法：它把客户端的
//! Manifest 交给 `Op::AddFiber`，从那一刻起这个客户端就是树上的一个 Fiber，和一个进程
//! 内插件同权同责（`robot/health` 里有它，`plugin/state` 会广播它的状态变化，撤销它就
//! 撤销它注册的一切）。
//!
//! 归属：Fiber 归**这条连接**。断连 → [`crate::handlers::release_all`] → `dispose_fiber`；
//! 心跳超时 → 路由的监视任务 → 同一条路。客户端自己也能走 `plugin/unregister`。三条路
//! 通向同一个终点，因为"谁来撤"从来不该取决于"怎么没的"。

use std::sync::Arc;
use std::time::Duration;

use oeh_admission::capability::CapabilitySpec;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::FiberId;
use oeh_proto::rpc::types::{
    CapabilityDecl, Empty, FiberRef, PluginRegister, PluginRegisterResult, StreamChannel,
};
use oeh_proto::rpc::{code, RpcError};
use serde_json::Value;
use tokio::sync::oneshot::error::RecvError;

use crate::conn::{Conn, RemoteFiber};
use crate::frames;
use crate::handlers::{
    ensure_cleanup_hook, invalid_params, not_attached, ok, refused, release_if_closed,
    send_op_retrying, wire_name,
};
use crate::remote::RemotePlugin;
use crate::router::{parse_params, RouterConfig, Services};

/// 内核的 Op 通道已经关了。给 `KERNEL_GONE` 而不是 `INTERNAL`：调用方据此知道"再试也
/// 没用"，而不是"再试一次也许就好了"。
fn kernel_gone() -> RpcError {
    RpcError::new(code::KERNEL_GONE, "kernel is gone")
}

/// 内核骨架自己的服务名：一个跨进程插件的 `manifest.provide` 里出现它们中的任何一个，
/// 都是 `INVALID_PARAMS`（Plan 3 遗留 M12）。
///
/// 为什么必须在**注册这一步**挡，而不是让 `Ctx::provide` 去撞车：`provide` 的冲突今天
/// 会把这个 Fiber 变成 `Failed`——也就是说客户端得到的是一次"成功的调用 + 一个失败的
/// Fiber"，树上真的多了一行，而运维看到的是"某个适配器加载失败了"，不是"它想冒充时钟"。
/// 更要紧的是这些名字是**别的插件 inject 的东西**：`inject: ["clock"]` 的每一个进程内
/// 插件都在下一次 settle 里跟着这行走，一个抢到名字的远端服务就是把内核骨架的一块换成
/// 了一条连接。这条边界上，这必须是一个当场的、不留痕迹的拒绝。
pub const RESERVED_SERVICES: &[&str] = &[
    "kernel",
    "clock",
    "world-log",
    "frame-store",
    "state",
    "brains",
    "admission",
    "predictor",
    "approval",
    "rpc-server",
];

/// 超预算的注册（见 [`RouterConfig::register_timeout_ms`]）。`REFUSED` 而不是
/// `INVALID_PARAMS`：同一份 manifest 在一台不忙的机器上完全可能注册得上。
fn budget_exceeded() -> RpcError {
    refused("register budget exceeded")
}

/// 把一行已经进了树、但已经没人要的 Fiber 撤掉，**撤干净之前不放开注册闸门**。
///
/// 两条超预算的路都用它，`receipt` 是那条 `Op::AddFiber` 的收据：一条路上它早就好了
/// （拿到了 `FiberAdded` 才发现超预算），另一条路上它还没好（预算先到，命令还在内核的
/// 队列里排着）。两种都得等——收据被丢掉不等于命令没投出去，内核照样会把那一行加上。
///
/// 应答（`REFUSED`）已经先发出去了，所以清理只能在一个任务里做。**闸门跟着进那个任务**
/// 是这条函数的要害：树上这一行的名字要到 dispose 真的跑完才腾出来，而重名检查读的是
/// `ConnState.fibers`——这个幽灵从来没进过那张表。客户端收到 `REFUSED` 之后立刻重试是
/// 最自然的反应，闸门就是那次重试要等的东西：等到了，它看到的是一棵干净的树。
///
/// 等 dispose **完成**（第二个 `.await`），不是投出去就算：那一行的效果链（ingress 线程、
/// 流注册、能力注册、`provide` 的条目）是在内核任务上逆序跑完的，投递成功只说明命令进了
/// 队列。重试要等的是效果，不是命令。
///
/// 清理要是永远完不成（内核彻底卡住），闸门就一直攥着——而那不会把调用方挂死：下一次
/// 注册等闸门的时间**算在它自己的预算里**（见 `register` 里那个 `timeout_at`），到点
/// 就是一句 `REFUSED{"register budget exceeded"}`。fail closed，而且是同一句话。
fn dispose_over_budget(
    kernel: Arc<oeh_kernel::KernelHandle>,
    gate: tokio::sync::OwnedMutexGuard<()>,
    plugin: String,
    receipt: impl std::future::Future<Output = Result<oeh_kernel::FiberAdded, RecvError>>
        + Send
        + 'static,
) {
    tokio::spawn(async move {
        // 撤干净之前不放开闸门。
        let _gate = gate;
        let Ok(added) = receipt.await else {
            // 内核没了：那棵树连同这一行一起没了。
            return;
        };
        tracing::warn!(
            %plugin,
            fiber = added.id.0,
            "register budget exceeded; disposing the fiber the kernel added anyway"
        );
        let done = send_op_retrying(|| kernel.dispose_fiber(added.id)).await;
        match done {
            Ok(rx) => {
                let _ = rx.await;
            }
            Err(_) => tracing::warn!(%plugin, "kernel is gone; the over-budget fiber stays"),
        }
    });
}

/// 线格式的能力声明 → 准入侧的 `CapabilitySpec`。`class`/`risk_level` 是小写的枚举名，
/// 写错一个字必须在**注册**这一步响（`INVALID_PARAMS`），而不是变成一个注册失败的
/// Fiber：那两件事对客户端是完全不同的两种错。
///
/// `provider` 与 `overridable` **不**从这里进来，哪怕客户端发了：前者由 `RemotePlugin`
/// 按 `manifest.name` 填（自报归属等于可以冒充内核），后者永远是 `false`（自封"可顶替"
/// 等于邀请下一个连上来的进程接管自己的能力面）。这两件事都只有内核说了算。
fn spec_of(d: &CapabilityDecl) -> Result<CapabilitySpec, RpcError> {
    serde_json::from_value(serde_json::json!({
        "name": d.name,
        "class": d.class,
        "risk_level": d.risk_level,
        "params_schema": d.params_schema,
        "lease_ttl_s": d.lease_ttl_s,
    }))
    .map_err(|e| invalid_params(format_args!("capability `{}`: {e}", d.name)))
}

/// 注册一个跨进程插件。
///
/// 全程在这条连接的 `plugin` 闸门下（见 [`Conn::lock_plugins`]）：查重名和落表之间夹着
/// 一次 `.await`（内核 settle），没有闸门的话两个同名的并发注册会双双通过重名检查。
///
/// `state == "failed"` 仍然是一次**成功**的调用（200 + result）：内核确实收下了这一行、
/// 试着激活了它、并且给出了原因（`error`）。把它编码成 RPC 错误会丢掉 `fiber_id`，而
/// 客户端正需要它去 `plugin/unregister`——一个 Failed 的 Fiber 也是树上的一行。
///
/// 整条路由一个**总预算**罩着（[`RouterConfig::register_timeout_ms`]，Plan 3 遗留 I6）。
/// 预算不是一个包在外面的 `timeout(...)`，而是三个等待点各自的截止时刻，因为三个点上
/// "被取消"意味着完全不同的三件事：闸门和 Op 队列退避上取消掉什么都没发生过（那一刻还
/// 没有任何东西被造出来），而 `add_fiber` 那个等待点上取消掉，**内核照样会把那一行加上**
/// ——收据被丢掉不等于命令没投出去。所以那一个点不能只是"不等了"，得把收据交给一个清理
/// 任务：等那行真的出现，再把它撤掉。一个包在外面的 `timeout` 做不到这件事，它只会在
/// 树上留下一个没人续心跳、也没人认领的 Fiber。
pub async fn register(
    s: Arc<Services>,
    cfg: Arc<RouterConfig>,
    conn: Arc<Conn>,
    params: Value,
) -> Result<Value, RpcError> {
    // 预算从**收到这一条**起算（下面的检查都是纯计算，不进预算也无所谓——它们加起来是
    // 微秒级，而预算的默认值是 5 秒）。
    let deadline = tokio::time::Instant::now() + Duration::from_millis(cfg.register_timeout_ms);
    // 字节上限先查，在**解析之前**：它是唯一一个把 `config`、每份 `params_schema` 和整个
    // manifest 一次罩住的数字，而后面每一步（解析、克隆进 `EntryRow`、克隆进
    // `RemotePlugin`）都会把这些字节再拷一遍。一次序列化就够了，别的检查都比它便宜。
    let bytes = serde_json::to_vec(&params).map(|v| v.len()).unwrap_or(0);
    if bytes > cfg.max_register_bytes {
        return Err(invalid_params(format_args!(
            "params is {bytes} bytes, over the {} byte limit",
            cfg.max_register_bytes
        )));
    }
    let p: PluginRegister = parse_params(&params)?;
    let name = p.manifest.name.trim().to_string();
    if name.is_empty() {
        return Err(invalid_params("manifest.name must not be empty"));
    }
    // 数量上限：`RemotePlugin::apply` 跑在内核任务上，每条流/每个能力都是那里的一次
    // 同步注册。全部编成 `INVALID_PARAMS`（不是一个 `Failed` 的 Fiber）——超限是调用方
    // 的参数错，和写错一个 `class` 同一类，客户端不该为了分辨它去读 `state`。
    for (what, len, max) in [
        (
            "streams",
            p.manifest.streams.len(),
            cfg.max_streams_per_register,
        ),
        (
            "capabilities",
            p.capabilities.len(),
            cfg.max_capabilities_per_register,
        ),
        (
            "inject",
            p.manifest.inject.len(),
            cfg.max_inject_per_register,
        ),
        (
            "provide",
            p.manifest.provide.len(),
            cfg.max_provide_per_register,
        ),
    ] {
        if len > max {
            return Err(invalid_params(format_args!(
                "manifest declares {len} {what}, over the limit of {max}"
            )));
        }
    }
    // 数量上限数的是"几条流"，从来不看一条流**里面**的数字——而内核正是按里面那个数字
    // 预分配环形缓冲的（`FrameStore::register`：`nominal_hz × window_s` 帧）。
    // `nominal_hz: 1e12` 因此是一次 176 TB 的分配请求，而分配失败走 `handle_alloc_error`：
    // 那是 abort，不是 panic，`FiberTree::activate` 的 `catch_unwind` 接不住，内核进程当场
    // 没（Plan 3 最终评审 C1，实测 SIGABRT）。这条检查在 `Op::AddFiber` 之前，所以被拒的
    // 注册连一行都不会进树。
    //
    // `INVALID_PARAMS` 而不是 `REFUSED`：和写错一个 `class` 同一类——同样的 params 重发
    // 多少次都是错的。非有限值（JSON 本身不表示 NaN/Infinity，但别的编码可能）和 ≤ 0 一
    // 并挡掉：一条"零赫兹"的流不是一个能声明出来的东西，而它会让内核那边的容量算式落在
    // 下界上，把一个明显写错的 manifest 变成一条静悄悄工作不正常的流。
    for sp in &p.manifest.streams {
        let hz = sp.nominal_hz;
        if !hz.is_finite() || hz <= 0.0 || hz > cfg.max_nominal_hz {
            return Err(invalid_params(format_args!(
                "stream `{}`: nominal_hz must be a finite number in (0, {}], got {hz}",
                sp.stream_id, cfg.max_nominal_hz
            )));
        }
    }
    // 保留服务名（M12）：在造出任何东西之前。`provide` 的顺序由客户端决定，所以报的是
    // 撞上的**那一个**名字——客户端要改的就是它。
    for name in &p.manifest.provide {
        if RESERVED_SERVICES.contains(&name.0.as_str()) {
            return Err(invalid_params(format_args!(
                "reserved service name: {}",
                name.0
            )));
        }
    }
    let caps = p
        .capabilities
        .iter()
        .map(spec_of)
        .collect::<Result<Vec<_>, _>>()?;
    let heartbeat_ms = if p.heartbeat_ms == 0 {
        cfg.heartbeat_ms
    } else {
        p.heartbeat_ms
    };
    // 客户端报 0 以外的心跳周期，但内核默认也是 0（配置里写了 `heartbeat_ms: 0`）：那就
    // 没有心跳监视可言。挡住它——一个永不超时的远端插件正是心跳要防的那种东西。
    if heartbeat_ms == 0 {
        return Err(invalid_params("heartbeat_ms must be > 0"));
    }
    let streams: Vec<StreamChannel> = p
        .manifest
        .streams
        .iter()
        .map(|sp| StreamChannel {
            stream_id: sp.stream_id.clone(),
            channel: frames::channel_name(&cfg.run_tag, &sp.stream_id),
            schema_id: frames::schema_id(&sp.schema),
        })
        .collect();

    // 闸门也在预算里：一条连接上另一次慢注册正攥着它的时候，这一次的等待时间完全由
    // **别人**决定——不算进预算的话，那就是一个绕开预算的办法。这个等待点上取消掉是
    // 干净的：还什么都没造出来。
    let Ok(gate) = tokio::time::timeout_at(deadline, conn.lock_plugins()).await else {
        return Err(budget_exceeded());
    };
    {
        let st = conn.state.lock();
        if st.fibers.values().any(|f| f.plugin == name) {
            return Err(refused(format!(
                "plugin `{name}` is already registered on this connection"
            )));
        }
        // 活着的 Fiber 数：重名检查挡不住这条路——`a1`/`a2`/`a3`… 都是合法的不同名字，
        // 而每一个都是内核树上真实的一行。这是一个**状态**条件，不是参数错误（Task 5
        // re-review R2）：同样的 params 在一次 `plugin/unregister` 之后原样重发就能成功，
        // 这正是 `REFUSED`（"现在不行"）和 `INVALID_PARAMS`（"这次调用就是错的，别重试"）
        // 的分野——旁边的重名检查已经在用 `REFUSED`，这里该跟它一致。
        if st.fibers.len() >= cfg.max_fibers_per_conn {
            return Err(refused(format!(
                "this connection already holds {} plugins, the limit is {}",
                st.fibers.len(),
                cfg.max_fibers_per_conn
            )));
        }
    }

    // 行的 `id` 是 `remote:<name>`，`name` 是 `manifest.name`——树上的 key 用后者
    // （`EntryRow::key()`），所以 `plugin/state` / `robot/health` 里它就叫客户端自己报的
    // 那个名字。`id` 永远解析不出工厂，也不需要：`Op::AddFiber` 带着实现一起进来。
    let row = EntryRow::new(format!("remote:{name}"))
        .with_name(&name)
        .with_config(p.config);
    // 载荷（Manifest + 能力表）每次重试都重建：一条被拒的 `Op::AddFiber` 把它的
    // `Box<dyn Plugin>` 一起带走了。构造是纯数据克隆，重试三次的代价可以忽略。
    //
    // 这个等待点上取消掉同样是干净的：`send_op_retrying` 唯一的等待是一次 `Busy` 之后的
    // 退避，也就是说被取消的那一刻，最后一次尝试是**被拒**的——命令没进队列。
    let receipt = tokio::time::timeout_at(deadline, {
        send_op_retrying(|| {
            let plugin = RemotePlugin::new(
                p.manifest.clone(),
                caps.clone(),
                &conn,
                cfg.clone(),
                s.channels.clone(),
            );
            s.kernel.add_fiber(row.clone(), Box::new(plugin))
        })
    })
    .await
    .map_err(|_| budget_exceeded())?
    .map_err(|_| kernel_gone())?;

    // **测试钩子**（`test-hooks` feature，见 [`RouterConfig::stream_setup_delay_ms`]）：
    // 假装内核那一侧给每条声明的流建通道各要这么久。放在命令已经投出去、收据还没到手的
    // 这一段，因为这正是一次慢注册的形状——那一行会被加上，而调用方还在等。
    #[cfg(feature = "test-hooks")]
    if cfg.stream_setup_delay_ms > 0 {
        let per = cfg
            .stream_setup_delay_ms
            .min(crate::router::MAX_STREAM_SETUP_DELAY_MS);
        let total = per.saturating_mul(p.manifest.streams.len() as u64);
        tokio::time::sleep(Duration::from_millis(total)).await;
    }

    // 从这里起命令已经在内核的队列里了：超预算不能只是"不等了"（见函数头上的说明）。
    // `biased` 让判定确定——超预算的那一刻，收据和定时器完全可能同时就绪，而一个随机挑
    // 分支的 `select!` 会把这条测不准的边界留在生产代码里。判定先落成一个值、出了
    // `select!` 再动手：超时那一支要把 `receipt` **交出去**，而 `&mut receipt` 这个借用
    // 要到整个 `select!` 结束才还。
    let mut receipt = receipt;
    let verdict = tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline) => None,
        added = &mut receipt => Some(added.map_err(|_| kernel_gone())?),
    };
    let Some(added) = verdict else {
        // 收据（连同闸门）交给清理任务：内核会把那一行加上，只是没人要了。
        dispose_over_budget(s.kernel.clone(), gate, name, receipt);
        return Err(budget_exceeded());
    };
    // 上面那个 `biased` 管的是"两边同时就绪"，还有一种情况它管不到：`RemotePlugin::apply`
    // 跑在内核任务上，一次真正慢下来的注册会把那条**线程**占住——tokio 的时间驱动要靠
    // 运行时让出控制权才推进，于是这条 future 直到内核忙完才被 poll，而那一刻
    // `sleep_until` 手里的"现在"还是进入等待之前的那个：它没就绪，收据却早就好了。预算是
    // 按墙钟算的，所以拿到收据之后再按墙钟判一次。
    //
    // （测试钩子走不到这一支——它是异步 sleep，时间驱动一直在推进。这一支守的是真实世界
    // 里那种把线程占住的慢，测试造不出来也不该去造：`std::thread::sleep` 一个内核任务，
    // 造出来的是"整个进程卡住"，那是另一回事。）
    if tokio::time::Instant::now() >= deadline {
        // 这一支的"收据"已经在手里了，用一个立刻就绪的 future 顶上：清理要等的只是
        // "那一行的 id 什么时候到手"。
        let ready = std::future::ready(Ok(added));
        dispose_over_budget(s.kernel.clone(), gate, name, ready);
        return Err(budget_exceeded());
    }

    // 钩子先登记、再落表，和 `brain/attach` 同一个次序与同一个兜底（`release_if_closed`）：
    // 连接要是在 `add_fiber` 这段 `.await` 里关掉了，钩子看到的是一张还空着的表。
    ensure_cleanup_hook(&conn);
    let mut fiber = RemoteFiber {
        plugin: name.clone(),
        heartbeat: Duration::from_millis(heartbeat_ms),
        deadline: tokio::time::Instant::now(),
        kernel: s.kernel.clone(),
    };
    fiber.refresh();
    conn.state.lock().fibers.insert(added.id.0, fiber);
    release_if_closed(&conn);
    // 从此这条连接在日志、`event/emit` 的 source、`robot/health` 里都按插件名出现。
    conn.set_name(&name);

    ok(PluginRegisterResult {
        fiber_id: added.id.0,
        state: wire_name(&added.state),
        error: added.error,
        streams,
        heartbeat_ms,
    })
}

/// 续一次心跳。只认**这条连接**注册的 Fiber：别人的 Fiber 不在这张表里 → `NOT_ATTACHED`，
/// 否则任何一个客户端都能替一个已经失联的插件续命。
pub fn heartbeat(conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: FiberRef = parse_params(&params)?;
    let mut st = conn.state.lock();
    match st.fibers.get_mut(&p.fiber_id) {
        Some(f) => {
            f.refresh();
            drop(st);
            ok(Empty {})
        }
        None => Err(not_attached(format_args!("fiber {}", p.fiber_id))),
    }
}

/// 主动注销。等内核真的撤完再回复：调用方据此知道"我的能力已经从能力面上下去了"，
/// 而不是"命令投出去了"——一个正要退出的适配器就是靠这个差别决定什么时候能断电。
pub async fn unregister(conn: &Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: FiberRef = parse_params(&params)?;
    let fiber = conn.state.lock().fibers.remove(&p.fiber_id);
    let Some(fiber) = fiber else {
        return Err(not_attached(format_args!("fiber {}", p.fiber_id)));
    };
    send_op_retrying(|| fiber.kernel.dispose_fiber(FiberId(p.fiber_id)))
        .await
        .map_err(|_| kernel_gone())?
        .await
        .map_err(|_| kernel_gone())?;
    ok(Empty {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use oeh_admission::capability::{CapabilityClass, RiskLevel};

    fn decl(class: &str, risk: &str) -> CapabilityDecl {
        CapabilityDecl {
            name: "wave".into(),
            class: class.into(),
            risk_level: risk.into(),
            params_schema: serde_json::json!({"type": "object"}),
            lease_ttl_s: 1.5,
            provider: String::new(),
            overridable: false,
        }
    }

    /// 线格式的名字是**小写的 serde 名**，不是 Rust 的 `Debug` 名。这条测试钉住的是
    /// 边界协议和准入侧枚举之间的那次转换。
    #[test]
    fn capability_decl_parses_into_a_spec() {
        let s = spec_of(&decl("motion", "low")).unwrap();
        assert_eq!(s.name, "wave");
        assert_eq!(s.class, CapabilityClass::Motion);
        assert_eq!(s.risk_level, RiskLevel::Low);
        assert_eq!(s.lease_ttl_s, 1.5);
    }

    /// 客户端自封的归属一概不算数。`provider` 是内核按 `manifest.name` 填的，
    /// `overridable` 只有内置能力有——一个能自称可顶替的插件，就是在邀请下一个连上来
    /// 的进程接管自己的能力面，而它连"下一个是谁"都不知道。
    #[test]
    fn a_client_cannot_declare_its_own_provider_or_overridability() {
        let mut d = decl("speech", "none");
        d.provider = "kernel".into();
        d.overridable = true;
        let s = spec_of(&d).unwrap();
        assert!(!s.overridable);
    }

    /// 写错 class/risk_level 是**调用方**的参数错误，必须在注册这一步就响：编成一个
    /// Failed 的 Fiber 的话，客户端看到的是"内核加载我失败了"，而真相是"你那个字写错了"。
    #[test]
    fn a_misspelled_class_is_invalid_params_not_a_failed_fiber() {
        for d in [decl("Motion", "low"), decl("motion", "lowish")] {
            let e = spec_of(&d).unwrap_err();
            assert_eq!(e.code, code::INVALID_PARAMS, "{e:?}");
            assert!(e.message.contains("wave"), "{e:?}");
        }
    }
}
