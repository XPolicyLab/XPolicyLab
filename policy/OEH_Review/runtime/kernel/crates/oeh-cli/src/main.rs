use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use oeh_admission as _;
use oeh_brain as _;
use oeh_distill as _;
use oeh_kernel::factories::factory_ids;
use oeh_kernel::profile::load_profile;
use oeh_kernel::world_log::{read_segments, resolve_log_dir};
use oeh_kernel::{Kernel, KernelHandle, KernelSendError};
use oeh_predict as _;

#[derive(Parser)]
#[command(name = "oeh", about = "OpenEmbodied-Harness kernel")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 按 profile 启动内核，跑到 Ctrl-C 或 --for 秒数
    Run {
        #[arg(long)]
        profile: PathBuf,
        #[arg(long, default_value = "bundles")]
        bundles_dir: PathBuf,
        #[arg(long = "patch")]
        patches: Vec<PathBuf>,
        #[arg(long = "for", value_parser = parse_for)]
        for_secs: Option<f64>,
        /// 把 profile 里 `rpc-server` 行的配置强制改成 `{stdio: true, uds: null, ws: null}`
        /// （没有这一行就报错）：stdio 内核是一进程对一客户端的模式，同一台机器上跑第二个
        /// 这样的进程时，留着默认的 uds/ws 会去抢第一个进程已经绑定的 socket/端口，触发
        /// fail-closed 拒绝启动（Task 3 Fix round 1，M4）。一旦启用，`oeh run` 的 stdout
        /// 让给 JSON-RPC 用，插件表/"kernel stopped" 改打 stderr。
        #[arg(long)]
        rpc_stdio: bool,
    },
    /// 日志工具
    Log {
        #[command(subcommand)]
        cmd: LogCmd,
    },
    /// 列出编译进内核的插件 id
    Plugins,
    /// 连到运行中的内核
    Rpc {
        #[arg(long, default_value = "./run/oeh.sock", env = "OEH_RPC_UDS")]
        uds: PathBuf,
        #[command(subcommand)]
        cmd: RpcCmd,
    },
}

#[derive(Subcommand)]
enum RpcCmd {
    /// 调一个方法：oeh rpc call robot/state '{}'
    Call {
        method: String,
        #[arg(default_value = "{}")]
        params: String,
    },
    /// 实时打印世界日志（log/tail）：oeh rpc tail --prefix act
    Tail {
        #[arg(long)]
        prefix: Vec<String>,
        #[arg(long)]
        after: Option<u64>,
    },
}

/// `--for` 只接受有限的非负秒数：NaN / inf / 负数会让 `Duration::from_secs_f64` panic
/// 或者让内核立刻退出，在参数解析阶段就说清楚。
fn parse_for(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| format!("`{s}` is not a number of seconds"))?;
    if !v.is_finite() {
        return Err(format!("`{s}` must be a finite number of seconds"));
    }
    if v < 0.0 {
        return Err(format!("`{s}` must be >= 0 seconds"));
    }
    if v > 1.0e7 {
        return Err("--for must be <= 1e7 seconds".into());
    }
    Ok(v)
}

#[derive(Subcommand)]
enum LogCmd {
    /// 顺序回放一个日志目录，按行打印 JSON
    Replay {
        dir: PathBuf,
        #[arg(long)]
        prefix: Option<String>,
        /// 只打印这个 trace_id 的事件：一次 turn 的 `brain/*` 与 `act/*` 全串在它上面（§12）
        #[arg(long)]
        trace: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 日志走 stderr：`oeh log replay` 的 stdout 必须是纯 JSON
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?),
        )
        .init();
    match Cli::parse().cmd {
        Cmd::Plugins => {
            for id in factory_ids() {
                println!("{id}");
            }
        }
        Cmd::Log {
            cmd: LogCmd::Replay { dir, prefix, trace },
        } => {
            // 只扫描一次目录：resolve_log_dir 选出的 run 目录已经含 seg-*，直接喂给
            // read_segments，避免它内部再扫一遍（也避免两次扫描之间目录内容发生变化）。
            let resolved = resolve_log_dir(&dir)?;
            eprintln!("replaying {}", resolved.display());
            for ev in read_segments(&resolved)? {
                // 两个过滤器是与的关系：`--prefix act --trace t-1-2-3` 就是"这次 turn 的
                // 准入链"。都不给就全打。
                let keep = prefix
                    .as_deref()
                    .map(|p| ev.kind.prefix() == p)
                    .unwrap_or(true)
                    && trace
                        .as_deref()
                        .map(|t| ev.trace_id.as_deref() == Some(t))
                        .unwrap_or(true);
                if keep {
                    println!("{}", serde_json::to_string(&ev)?);
                }
            }
        }
        Cmd::Run {
            profile,
            bundles_dir,
            patches,
            for_secs,
            rpc_stdio,
        } => {
            // `#[tokio::main]` 的 worker 线程这时已经在跑，但此刻还没有任何一方在读这个
            // 变量：插件读它的地方（`brain-native` 的 `protected_paths` /
            // `default_protected`）都在下面的 `Kernel::boot` 里，`set_var` 的竞态只和
            // 并发的读/写有关，所以这里仍是安全的。
            //
            // 为什么内核要替调用方设这个变量：一份 profile 里的相对路径说的是"相对这份
            // profile"，而进程 cwd 是启动它的那个 shell 的事。`systemctl` 起的内核
            // （cwd `/`）和手敲起来的内核（cwd 是仓库根）必须拿到同一套 RSI 护栏。
            // 已经设了就不动：那是操作员自己说的话，命令行不该替他改口。
            if std::env::var_os("OEH_PROFILE").is_none() {
                let abs = std::fs::canonicalize(&profile).unwrap_or_else(|_| profile.clone());
                std::env::set_var("OEH_PROFILE", abs);
            }
            let mut rows = load_profile(&profile, &bundles_dir, &patches)?;
            if rpc_stdio {
                let row = rows
                    .iter_mut()
                    .find(|r| r.id == "rpc-server")
                    .ok_or_else(|| {
                        anyhow::anyhow!("--rpc-stdio requires a `rpc-server` row in the profile")
                    })?;
                // 也把 `uds`/`ws` 都清成 `null`：stdio 是一进程对一客户端的模式，第二个
                // 这样的进程如果还留着默认的 uds/ws，会去抢第一个进程已经绑定的 socket/
                // 端口——`rpc-server` 按 brief 的 fail-closed 规则拒绝启动（Task 3 Fix
                // round 1，M4）。
                match &mut row.config {
                    serde_json::Value::Null => {
                        row.config = serde_json::json!({"stdio": true, "uds": null, "ws": null});
                    }
                    serde_json::Value::Object(map) => {
                        map.insert("stdio".to_string(), serde_json::Value::Bool(true));
                        map.insert("uds".to_string(), serde_json::Value::Null);
                        map.insert("ws".to_string(), serde_json::Value::Null);
                    }
                    _ => anyhow::bail!("rpc-server row's `config` must be a JSON object"),
                }
            }
            // stdio 传输启用时，stdout 是 JSON-RPC 的线路，内核状态表/"kernel stopped"
            // 改打 stderr（FAILED/PENDING 本来就在 stderr，不用动）。
            let stdio_active = rows.iter().any(|r| {
                !r.disabled
                    && r.id == "rpc-server"
                    && r.config
                        .get("stdio")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
            });
            let kernel = Kernel::boot(rows)?;
            for (k, s) in kernel.states() {
                if stdio_active {
                    eprintln!("{k:<20} {s:?}");
                } else {
                    println!("{k:<20} {s:?}");
                }
            }
            for (k, e) in &kernel.report().failed {
                eprintln!("FAILED {k}: {e}");
            }
            for (k, m) in &kernel.report().pending {
                eprintln!("PENDING {k}: missing {m:?}");
            }
            // fail-closed：有插件挂了就别跑。Pending 是允许的——跨进程 provider 可能晚到。
            // 但仍然要走一遍 run()：已经激活成功的插件（clock/frame-store/world-log 等）需要
            // 被 dispose_all 撤销，不能因为 bail 提前退出就把它们的 Effect（比如 world-log
            // 打开的文件句柄）留在原地。
            let failed = kernel.report().failed.len();
            if failed > 0 {
                let h = kernel.handle();
                shutdown_kernel(&h).await?;
                kernel.run().await?;
                anyhow::bail!("{failed} plugin(s) failed to activate; refusing to run");
            }
            let handle = kernel.handle();
            let run = tokio::spawn(kernel.run());
            match for_secs {
                Some(s) => tokio::time::sleep(Duration::from_secs_f64(s)).await,
                None => {
                    tokio::signal::ctrl_c().await?;
                }
            }
            shutdown_kernel(&handle).await?;
            run.await??;
            if stdio_active {
                eprintln!("kernel stopped");
            } else {
                println!("kernel stopped");
            }
        }
        Cmd::Rpc { uds, cmd } => rpc_main(uds, cmd).await?,
    }
    Ok(())
}

/// 投一条 `Op::Shutdown`，Op 队列满就退避重试。
///
/// 关机命令和别的 Op 走同一条有界通道（它不能插队——前面排着的 Op 是已经被接受的
/// 工作），所以 `Busy` 是一个真实的结果：一个刚被 `--for` 叫停、队列里还堆着一秒钟
/// 帧的内核，第一次 `shutdown()` 完全可能被拒。把那当成失败退出，等于让 `oeh run`
/// 在最正常的那条路径上偶发地报错、并且跳过 `run.await`（插件的 Effect 就留在原地了）。
async fn shutdown_kernel(h: &KernelHandle) -> anyhow::Result<()> {
    for _ in 0..50 {
        match h.shutdown() {
            Ok(()) => return Ok(()),
            Err(KernelSendError::Busy) => tokio::time::sleep(Duration::from_millis(10)).await,
            Err(e) => return Err(e.into()),
        }
    }
    h.shutdown().map_err(Into::into)
}

/// `oeh rpc call|tail`：一条到运行中内核的 UDS 客户端连接。
async fn rpc_main(uds: PathBuf, cmd: RpcCmd) -> anyhow::Result<()> {
    let client = oeh_rpc::client::Client::connect_uds(&uds)
        .await
        .map_err(|e| anyhow::anyhow!("connect {}: {e}", uds.display()))?;
    match cmd {
        RpcCmd::Call { method, params } => {
            let params: serde_json::Value = serde_json::from_str(&params)?;
            match client.call(&method, params).await {
                Ok(v) => println!("{}", serde_json::to_string_pretty(&v)?),
                Err(e) => {
                    eprintln!("{}", serde_json::to_string_pretty(&e)?);
                    std::process::exit(1);
                }
            }
        }
        RpcCmd::Tail { prefix, after } => {
            // 先订阅再发 log/tail：`Client::notifications()` 之后才建立的订阅收不到
            // 订阅之前已经广播过的通知（`broadcast::Sender` 对零接收端的发送直接丢弃），
            // 反过来做会在“服务端已经开始推 event”和“我们开始收”之间留一个丢事件的窗口。
            let mut notes = client.notifications();
            if let Err(e) = client
                .call(
                    "log/tail",
                    serde_json::json!({"after": after, "prefixes": prefix}),
                )
                .await
            {
                eprintln!("{}", serde_json::to_string_pretty(&e)?);
                std::process::exit(1);
            }
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => return Ok(()),
                    // 服务端把连接关了（内核停了，或者 `shutdown` 之后被关掉）。这一条
                    // 必须自己有一条分支：`notes` 是从 `Client` 上 `subscribe()` 来的
                    // 接收端，而 `client` 活着就意味着发送端活着——`RecvError::Closed`
                    // 下面那条分支因此在实践中永远不会被选中（Task 4 fix；那条分支留着
                    // 是为了 `client` 真的被丢掉的那种未来写法）。少了这一条的话，内核
                    // 一停，这个循环就在一个永远不会再来消息的通道上静静挂死，`oeh rpc
                    // tail` 看起来像"在正常地等事件"。
                    _ = client.wait_closed() => {
                        eprintln!("connection to kernel closed");
                        std::process::exit(1);
                    }
                    note = notes.recv() => {
                        match note {
                            Ok(n) if n.method == "event" => {
                                println!("{}", serde_json::to_string(&n.params)?);
                            }
                            Ok(_) => {}
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                // 服务端断了连接（内核停了，或者连接被 `shutdown`
                                // 之后关掉）：这跟操作者主动 Ctrl-C 是两回事，不该用同一
                                // 个退出码——不然"内核没了"看起来跟"我按了 Ctrl-C"一样
                                // 干净（Task 3 Fix round 1 nit）。实际走到的是上面那条
                                // `wait_closed()` 分支；这里是同一个结论的兜底。
                                eprintln!("connection to kernel closed");
                                std::process::exit(1);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
