//! `rpc-server` 内核插件：把 [`crate::transport::spawn_listeners`] 接到从 `Ctx` 拿到的
//! 服务上，供跨进程客户端（`oeh rpc` CLI、Python SDK……）通过 UDS/ws/stdio 连内核。
//!
//! 这个文件只负责"把 `Ctx` 里的服务攒成 [`crate::router::Services`]、按配置起监听器"；
//! 方法表本身在 [`crate::router`] / [`crate::handlers`]。

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use oeh_admission::admission::Admission;
use oeh_kernel::factories::PluginFactory;
use oeh_kernel::plugin::{Ctx, Plugin, PluginError};
use oeh_kernel::{
    Brains, ClockService, FrameStore, KernelHandle, RobotFsm, StateRegistry, WorldLog,
};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::router::{KernelRouter, Router, RouterConfig, Services};
use crate::transport::{spawn_listeners, ConnRegistry, Listeners, UdsAuth};

fn default_uds() -> Option<PathBuf> {
    Some(PathBuf::from("./run/oeh.sock"))
}
/// UDS 的本机鉴权，见 [`crate::transport::UdsAuth`]。
fn default_uds_mode() -> u32 {
    crate::transport::DEFAULT_UDS_MODE
}
fn default_peer_uids() -> Vec<u32> {
    crate::transport::current_uid().into_iter().collect()
}
/// 一次 `plugin/register` 从收到到应答的**总预算**，见 [`RouterConfig::register_timeout_ms`]。
fn default_register_timeout_ms() -> u64 {
    5000
}
fn default_heartbeat_ms() -> u64 {
    1000
}
fn default_execute_accept_ms() -> u64 {
    2000
}
fn default_cancel_grace_ms() -> u64 {
    500
}
fn default_frames_max_bytes() -> usize {
    1024 * 1024
}
/// `plugin/register` 的护栏，见 [`RouterConfig`] 上的说明。
fn default_max_per_register() -> usize {
    64
}
fn default_max_fibers_per_conn() -> usize {
    16
}
fn default_max_register_bytes() -> usize {
    256 * 1024
}
/// 见 [`RouterConfig::max_nominal_hz`]：规格里最快的一条流是 500 Hz，40 倍余量。
fn default_max_nominal_hz() -> f64 {
    20_000.0
}
/// 见 [`RouterConfig::max_replay_page`]。
fn default_max_replay_page() -> usize {
    4096
}
/// 见 [`RouterConfig::max_event_bytes`]。
fn default_max_event_bytes() -> usize {
    64 * 1024
}
/// 见 `crate::transport::DEFAULT_MAX_CONNECTIONS`。下限 1 由
/// `ConnRegistry::with_max` 兜着（`max_connections: 0` 会把唯一那条 stdio 连接也拒掉）。
fn default_max_connections() -> usize {
    256
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FramesConfig {
    #[serde(default = "default_frames_max_bytes")]
    max_bytes: usize,
}
impl Default for FramesConfig {
    fn default() -> Self {
        Self {
            max_bytes: default_frames_max_bytes(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "default_uds")]
    uds: Option<PathBuf>,
    /// 这次 run 的帧通道命名空间（`oeh/<run_tag>/<stream_id>`）。不写就按世界日志的 run
    /// 目录名，再退到 `local-<pid>`——见 [`resolve_run_tag`]。
    #[serde(default)]
    run_tag: Option<String>,
    /// WebSocket 监听地址。**默认不开**，而且开它要两个键一起写（见
    /// [`Config::ws_allow_unauthenticated`]）：ws 上没有 `SO_PEERCRED` 可言（对端可能在
    /// 另一台机器上），而这条边界后面是一台会动的机器人。要开就得是部署里写下来的一个
    /// 决定，不是一个默认值（Plan 3 遗留 I5）。
    #[serde(default)]
    ws: Option<String>,
    /// 明写"我知道 ws 上没有任何鉴权"。
    ///
    /// 为什么要第二个键：`ws` 本身长得像一个纯粹的"监听在哪"的旋钮，一个为了调试顺手
    /// 填上去的地址会悄悄把这台机器人的控制面开到网络上——UDS 那一侧的两道关卡
    /// （0600 + uid 白名单）在 TCP 上一道都不成立。两个键一起写，第二个键的名字本身就是
    /// 那句话的内容，谁也没法说自己不知道。只写 `ws` 是**启动期配置错误**（fail closed，
    /// `oeh run` 会因为这个 Fiber `Failed` 而拒绝启动），不是一条被忽略的配置。
    #[serde(default)]
    ws_allow_unauthenticated: bool,
    #[serde(default)]
    stdio: bool,
    #[serde(default = "default_heartbeat_ms")]
    heartbeat_ms: u64,
    #[serde(default = "default_execute_accept_ms")]
    execute_accept_ms: u64,
    #[serde(default = "default_cancel_grace_ms")]
    cancel_grace_ms: u64,
    #[serde(default)]
    frames: FramesConfig,
    #[serde(default = "default_max_per_register")]
    max_streams_per_register: usize,
    #[serde(default = "default_max_per_register")]
    max_capabilities_per_register: usize,
    #[serde(default = "default_max_per_register")]
    max_inject_per_register: usize,
    #[serde(default = "default_max_per_register")]
    max_provide_per_register: usize,
    #[serde(default = "default_max_fibers_per_conn")]
    max_fibers_per_conn: usize,
    #[serde(default = "default_max_register_bytes")]
    max_register_bytes: usize,
    #[serde(default = "default_max_nominal_hz")]
    max_nominal_hz: f64,
    #[serde(default = "default_max_replay_page")]
    max_replay_page: usize,
    #[serde(default = "default_max_event_bytes")]
    max_event_bytes: usize,
    #[serde(default = "default_max_connections")]
    max_connections: usize,
    /// UDS socket 文件绑完立刻打上的权限位（默认 0600），见 [`crate::transport::UdsAuth`]。
    #[serde(default = "default_uds_mode")]
    uds_mode: u32,
    /// 允许连进 UDS 的 uid（默认只有当前 uid），见 [`crate::transport::UdsAuth`]。
    #[serde(default = "default_peer_uids")]
    peer_uids: Vec<u32>,
    /// 一次 `plugin/register` 的总预算，见 [`RouterConfig::register_timeout_ms`]。
    #[serde(default = "default_register_timeout_ms")]
    register_timeout_ms: u64,
    /// **测试钩子**，见 [`RouterConfig::stream_setup_delay_ms`]。只在 `test-hooks`
    /// feature 下**存在**——产品构建里这个字段根本不在结构体上，profile 里写它会被
    /// `deny_unknown_fields` 拒成一个启动错误。
    #[cfg(feature = "test-hooks")]
    #[serde(default)]
    stream_setup_delay_ms: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            uds: default_uds(),
            run_tag: None,
            ws: None,
            stdio: false,
            heartbeat_ms: default_heartbeat_ms(),
            execute_accept_ms: default_execute_accept_ms(),
            cancel_grace_ms: default_cancel_grace_ms(),
            frames: FramesConfig::default(),
            max_streams_per_register: default_max_per_register(),
            max_capabilities_per_register: default_max_per_register(),
            max_inject_per_register: default_max_per_register(),
            max_provide_per_register: default_max_per_register(),
            max_fibers_per_conn: default_max_fibers_per_conn(),
            max_register_bytes: default_max_register_bytes(),
            max_nominal_hz: default_max_nominal_hz(),
            max_replay_page: default_max_replay_page(),
            max_event_bytes: default_max_event_bytes(),
            max_connections: default_max_connections(),
            uds_mode: default_uds_mode(),
            peer_uids: default_peer_uids(),
            register_timeout_ms: default_register_timeout_ms(),
            ws_allow_unauthenticated: false,
            #[cfg(feature = "test-hooks")]
            stream_setup_delay_ms: 0,
        }
    }
}

/// `OEH_RPC_UDS` 覆盖配置里 `uds` 的路径，但不覆盖"要不要开 UDS 监听"这个决定：跟
/// `oeh rpc --uds`（同名 env）配对，挪个 socket 位置只用设一个环境变量，服务端/客户端两边
/// 跟着换，不用改 profile。但如果 profile 显式写了 `uds: null`（意思是"不开 UDS，只用
/// stdio"），一个为了别的、跟这次 run 无关的 `oeh rpc` 调用而设的 `OEH_RPC_UDS` 不该把这
/// 个决定悄悄翻回"开"（Task 3 Fix round 1，L1）。
fn resolve_uds(configured: Option<PathBuf>) -> Option<PathBuf> {
    configured.map(|configured| {
        std::env::var_os("OEH_RPC_UDS")
            .map(PathBuf::from)
            .unwrap_or(configured)
    })
}

/// 这次 run 的帧通道命名空间。
///
/// 优先级：配置里显式写的 > 世界日志 run 目录的名字 > `local-<pid>`。
///
/// 最后那个退路是一个**生产**上的修正，不是测试便利：iceoryx2 的服务名是机器全局的，而
/// 退到一个常数 `"local"` 的话，同一台机器上两个没开持久化日志的内核就会共用
/// `oeh/local/lowstate`——第二个内核的 egress 成了第一个内核那条通道上的第二个发布者，
/// 两边的帧都会送到每一个订阅者那里，而任何地方都不会报出来。pid 是进程级唯一的，正好是
/// 这个名字该有的粒度。
fn resolve_run_tag(configured: Option<String>, run_dir: Option<PathBuf>) -> String {
    configured
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| run_dir.and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned())))
        .unwrap_or_else(|| format!("local-{}", std::process::id()))
}

/// `id` "rpc-server"：`inject` 的八个服务是 `KernelRouter` 方法表要用的全部家什。
/// `provide`s `"rpc"`（`Arc<ConnRegistry>`），给同一进程内其他想看连接数的插件用。
pub struct RpcServerPlugin;

impl Plugin for RpcServerPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("rpc-server")
            .inject(&[
                "kernel",
                "clock",
                "world-log",
                "frame-store",
                "state",
                "fsm",
                "brains",
                "admission",
            ])
            .provide(&["rpc"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };

        let kernel = ctx.get::<KernelHandle>("kernel")?;
        let clock = ctx.get::<ClockService>("clock")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let frames = ctx.get::<FrameStore>("frame-store")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let fsm = ctx.get::<RobotFsm>("fsm")?;
        let brains = ctx.get::<Brains>("brains")?;
        let admission = ctx.get::<Admission>("admission")?;

        let uds = resolve_uds(cfg.uds);
        let ws = match &cfg.ws {
            // 开 ws 要两个键一起写（见 `Config::ws_allow_unauthenticated`）。报错里把两个
            // 键都点名：读到这行日志的人要做的决定是"我到底要不要一条没有鉴权的传输"，
            // 而不是"这个字段该填什么"。
            Some(_) if !cfg.ws_allow_unauthenticated => {
                return Err(PluginError::Config {
                    plugin: ctx.plugin_name().to_string(),
                    reason: "`ws` is set but `ws_allow_unauthenticated` is not: the WebSocket                              listener has no peer authentication at all (no SO_PEERCRED, no                              token) — set `ws_allow_unauthenticated: true` to say so out loud,                              or drop `ws` and use the UDS listener"
                        .to_string(),
                })
            }
            Some(s) => Some(s.parse::<SocketAddr>().map_err(|e| PluginError::Config {
                plugin: ctx.plugin_name().to_string(),
                reason: format!("invalid `ws` address `{s}`: {e}"),
            })?),
            None => None,
        };

        let registry = Arc::new(ConnRegistry::with_max(cfg.max_connections));
        let run_tag = resolve_run_tag(cfg.run_tag, log.run_dir());

        // 帧通道：`stream/subscribe` 的 egress 和跨进程插件的 ingress 都挂在它上面。
        let channels = crate::frames::Frames::new_with_log(
            run_tag.clone(),
            cfg.frames.max_bytes,
            frames.clone(),
            log.clone(),
        );
        // egress 线程跟着**这个 Fiber**停，不是跟着最后一个 `Arc<Services>`：那份 `Arc` 被
        // 每条活着的连接和每个在飞的处理函数各攥一份，一条迟迟不断的连接会让发布线程比
        // 撤销了的 `rpc-server` 活得久。登记在监听器之前 = 撤销时排在监听器**之后**：先
        // 不再接受新的 `stream/subscribe`，再停发布。
        let stop_channels = channels.clone();
        ctx.on_dispose(Box::new(move || stop_channels.shutdown()));
        let services = Arc::new(Services {
            kernel,
            clock,
            log: log.clone(),
            frames,
            state,
            fsm,
            brains,
            admission,
            conns: registry.clone(),
            channels,
            approval_gate: parking_lot::Mutex::new(()),
        });
        let router_config = RouterConfig {
            heartbeat_ms: cfg.heartbeat_ms,
            execute_accept_ms: cfg.execute_accept_ms,
            cancel_grace_ms: cfg.cancel_grace_ms,
            run_tag,
            frames_max_bytes: cfg.frames.max_bytes,
            max_streams_per_register: cfg.max_streams_per_register,
            max_capabilities_per_register: cfg.max_capabilities_per_register,
            max_inject_per_register: cfg.max_inject_per_register,
            max_provide_per_register: cfg.max_provide_per_register,
            max_fibers_per_conn: cfg.max_fibers_per_conn,
            max_register_bytes: cfg.max_register_bytes,
            max_nominal_hz: cfg.max_nominal_hz,
            max_replay_page: cfg.max_replay_page,
            max_event_bytes: cfg.max_event_bytes,
            register_timeout_ms: cfg.register_timeout_ms,
            #[cfg(feature = "test-hooks")]
            stream_setup_delay_ms: cfg.stream_setup_delay_ms,
        };
        let router = KernelRouter::new(services, router_config);
        // 心跳监视任务跟着这个 Fiber 走：`rpc-server` 一被撤销，扫描就停，而不是等到
        // 最后一条连接放开它手里的 `Arc<dyn Router>`（见 `KernelRouter::monitor`）。
        ctx.on_dispose(router.monitor_disposer());
        let router: Arc<dyn Router> = router;

        let listeners = Listeners {
            uds,
            ws,
            stdio: cfg.stdio,
            uds_auth: UdsAuth {
                mode: cfg.uds_mode,
                peer_uids: cfg.peer_uids,
            },
        };
        // 监听失败（比如 ws 端口被占）是配置错误：`PluginError::Other` 会让这个 Fiber
        // `Failed`，`oeh run` 看到 `report().failed` 非空就拒绝启动——fail-closed，不是
        // "起了一半，缺一条传输"。
        let dispose = spawn_listeners(listeners, router, log, registry.clone())
            .map_err(|e| PluginError::Other(format!("rpc-server failed to bind listeners: {e}")))?;
        ctx.on_dispose(dispose);
        ctx.provide("rpc", registry)
    }
}

inventory::submit! { PluginFactory { id: "rpc-server", build: || Box::new(RpcServerPlugin) } }

#[cfg(test)]
mod tests {
    use super::{resolve_run_tag, resolve_uds, Config};

    /// Task 3 Fix round 1, L1: a disabled UDS listener (`uds: null` in the profile) must
    /// stay disabled no matter what `OEH_RPC_UDS` happens to be set to in the environment —
    /// deliberately does not touch the environment itself (mutating a process-global from a
    /// test that runs concurrently with others in the same binary would be its own hazard);
    /// the `None` input alone already exercises the branch that matters.
    #[test]
    fn oeh_rpc_uds_env_does_not_resurrect_a_disabled_uds_listener() {
        assert_eq!(resolve_uds(None), None);
    }

    /// 帧通道的命名空间：配置优先，然后是 run 目录名，最后是**进程唯一**的退路。
    /// 那个退路是一条生产不变式——两个没开持久化日志的内核不能共用同一条共享内存通道。
    #[test]
    fn run_tag_prefers_config_then_run_dir_then_a_process_unique_fallback() {
        let run_dir = Some(std::path::PathBuf::from("/var/log/oeh/run-0000000001234"));
        assert_eq!(
            resolve_run_tag(Some("bench".into()), run_dir.clone()),
            "bench"
        );
        assert_eq!(resolve_run_tag(None, run_dir), "run-0000000001234");
        // 空字符串（YAML 里写了 `run_tag: ""`）不算一个名字。
        for configured in [None, Some(String::new()), Some("  ".into())] {
            assert_eq!(
                resolve_run_tag(configured, None),
                format!("local-{}", std::process::id())
            );
        }
    }

    #[test]
    fn null_and_empty_config_share_the_defaults() {
        let manual = Config::default();
        let empty: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(manual.uds, empty.uds);
        assert_eq!(manual.ws, empty.ws);
        assert_eq!(manual.stdio, empty.stdio);
        assert_eq!(manual.heartbeat_ms, empty.heartbeat_ms);
        assert_eq!(manual.execute_accept_ms, empty.execute_accept_ms);
        assert_eq!(manual.cancel_grace_ms, empty.cancel_grace_ms);
        assert_eq!(manual.frames.max_bytes, empty.frames.max_bytes);
        assert_eq!(manual.run_tag, empty.run_tag);
        assert_eq!(empty.run_tag, None);
        assert_eq!(empty.uds, Some(std::path::PathBuf::from("./run/oeh.sock")));
        // `ws` 默认**不开**（Plan 3 遗留 I5）：要开就得在 profile 里写出来，而且要连
        // `ws_allow_unauthenticated` 一起写。
        assert_eq!(empty.ws, None);
        assert_eq!(
            manual.ws_allow_unauthenticated,
            empty.ws_allow_unauthenticated
        );
        assert!(!empty.ws_allow_unauthenticated);
        assert!(!empty.stdio);
        assert_eq!(empty.heartbeat_ms, 1000);
        assert_eq!(empty.execute_accept_ms, 2000);
        assert_eq!(empty.cancel_grace_ms, 500);
        assert_eq!(empty.frames.max_bytes, 1024 * 1024);
        assert_eq!(
            manual.max_streams_per_register,
            empty.max_streams_per_register
        );
        assert_eq!(manual.max_register_bytes, empty.max_register_bytes);
        assert_eq!(empty.max_streams_per_register, 64);
        assert_eq!(empty.max_capabilities_per_register, 64);
        assert_eq!(empty.max_inject_per_register, 64);
        assert_eq!(empty.max_provide_per_register, 64);
        assert_eq!(empty.max_fibers_per_conn, 16);
        assert_eq!(empty.max_register_bytes, 256 * 1024);
        assert_eq!(manual.max_nominal_hz, empty.max_nominal_hz);
        assert_eq!(empty.max_nominal_hz, 20_000.0);
        assert_eq!(manual.max_replay_page, empty.max_replay_page);
        assert_eq!(empty.max_replay_page, 4096);
        assert_eq!(manual.max_event_bytes, empty.max_event_bytes);
        assert_eq!(empty.max_event_bytes, 64 * 1024);
        assert_eq!(manual.max_connections, empty.max_connections);
        assert_eq!(empty.max_connections, 256);
        assert_eq!(manual.uds_mode, empty.uds_mode);
        assert_eq!(empty.uds_mode, 0o600);
        assert_eq!(manual.peer_uids, empty.peer_uids);
        // 默认白名单就是**这个进程自己的 uid**——不是"谁都行"。
        assert_eq!(
            empty.peer_uids,
            crate::transport::current_uid()
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert_eq!(manual.register_timeout_ms, empty.register_timeout_ms);
        assert_eq!(empty.register_timeout_ms, 5000);
        #[cfg(feature = "test-hooks")]
        assert_eq!(empty.stream_setup_delay_ms, 0);
    }

    /// 只写 `ws`、不写 `ws_allow_unauthenticated` 是**配置错误**，不是一条被忽略的配置。
    /// 这条测试钉的是那个 `and`：解析得下来（两个键各自都是合法的），拦它的是
    /// `apply` 里那一判。
    #[test]
    fn ws_parses_on_its_own_but_only_opens_together_with_the_acknowledgement() {
        let c: Config = serde_json::from_value(serde_json::json!({"ws": "127.0.0.1:8765"}))
            .expect("`ws` alone is still a well-formed config");
        assert_eq!(c.ws.as_deref(), Some("127.0.0.1:8765"));
        assert!(!c.ws_allow_unauthenticated);
        let both: Config = serde_json::from_value(
            serde_json::json!({"ws": "127.0.0.1:8765", "ws_allow_unauthenticated": true}),
        )
        .unwrap();
        assert!(both.ws_allow_unauthenticated);
    }

    /// 测试钩子在**产品构建**里根本不是一个配置项：没有 `test-hooks` feature 时
    /// `deny_unknown_fields` 会把它拒掉。（带着 feature 跑测试时它当然认得——那正是
    /// 它存在的场合，所以这条断言分两边写。）
    #[test]
    fn the_stream_setup_delay_hook_is_not_a_production_config_key() {
        let parsed: Result<Config, _> =
            serde_json::from_value(serde_json::json!({"stream_setup_delay_ms": 20}));
        #[cfg(feature = "test-hooks")]
        assert!(parsed.is_ok(), "{:?}", parsed.err());
        #[cfg(not(feature = "test-hooks"))]
        assert!(parsed.is_err());
    }
}
