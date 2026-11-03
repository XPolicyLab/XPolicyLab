//! `log/tail` · `log/replay` · `log/flush`——世界日志的三个读出口：跟着**现在**（tail）、
//! 回看**过去**（replay）、把还在缓冲里的落盘（flush）。

use std::path::{Component, PathBuf};
use std::sync::Arc;

use oeh_kernel::world_log::{read_segments_after, LogReadError};
use oeh_proto::rpc::types::{Empty, LogReplay, LogReplayResult, LogTail, LogTailResult};
use oeh_proto::rpc::{method, RpcError};
use oeh_proto::WorldEvent;
use parking_lot::Mutex;
use serde_json::Value;

use crate::conn::Conn;
use crate::handlers::{
    ensure_cleanup_hook, internal, invalid_params, ok, refused, release_if_closed,
};
use crate::router::{parse_params, RouterConfig, Services};

/// 交接期（见 [`Tail::buffer`]）最多重复几轮"取走缓冲、发出去"。世界日志的事件是稀疏的，
/// 正常情况下第一轮就空了；这个上限只保证一个持续高频的写入者不会把交接拖成死循环
/// （到点就直接开闸，代价是那一瞬间可能有少数几条事件的顺序和 seq 不完全一致）。
const MAX_HANDOFF_ROUNDS: usize = 64;

/// 一次 `log/tail` 最多补发多少条历史（只留**最新**的这么多）。
///
/// 这不是省事，是保命：补发是一串同步的 `conn.notify`（`try_send`），出站队列满了
/// `try_enqueue_response` 就会 `mark_closed()` 把整条连接判死。`WorldLog` 的默认保留窗口
/// 是 4096，出站队列也是 4096——`log/tail {"after": 0}` 一次就能把队列塞满，再加上此刻
/// 恰好排着的一条 `plugin/state` 或 `brain/turn` 就溢出，于是"控制台重连、要一遍积压"
/// 这个最自然的动作把自己的连接杀掉了。留出四分之三的队列给正常流量。
///
/// 被截掉的那一段不是没了：`from` 如实报告这次订阅覆盖到哪一条，更早的历史走
/// `log/replay`（它是按页读盘的，本来就是干这个的）。
const LOG_TAIL_BACKFILL_MAX: usize = 1024;

/// 一次 `log/tail` 最多认多少个前缀。
///
/// [`Tail::matches`] 跑在世界日志的观察者链上，也就是**同步地跑在 `WorldLog::append`
/// 里**——包括内核 Op 任务和 500 Hz 的帧 ingest 路径上的那些 append（`kind_has_prefix`
/// 是手写的，就是为了在那里不分配）。一条带一百万个短前缀的 `log/tail`（远在
/// `MAX_LINE` 之内）会让此后每一条事件的 append 先做一百万次字符串比较：帧 ingest 卡住，
/// `distill-watchdog` 开始报 `health/stale`，规则链于是拒掉一切——fail-closed 的坍塌，
/// 但仍然是坍塌，而且只要一个格式完全合法的请求（Plan 3 最终评审 I1）。
///
/// 16 已经很宽：`oeh rpc tail` 的 `--prefix` 实际用法是两三个。过滤器在订阅**建立时**
/// 就定下来（存进 `Tail.prefixes`），每条事件只是走一遍这最多 16 个前缀，不分配。
const MAX_TAIL_PREFIXES: usize = 16;

/// `log/replay` 解析不出目录时的**固定**错误文本。
///
/// 刻意不带 `io::Error`：错误文本里的 "No such file or directory" / "Permission denied"
/// 本身就是一个存在性/权限探针——任何能连上这个 socket 的客户端都能拿它去问"这台机器上
/// 有没有 /root/xxx"。回放读不到就是读不到，调用方不需要知道是哪一种读不到。
const REPLAY_DIR_UNAVAILABLE: &str = "replay dir not available";

/// 一条连接的 tail 订阅。挂在世界日志的观察者上，**跑在 append 线程上**，所以只做两件
/// 非阻塞的事：要么把事件推进交接缓冲，要么 `conn.notify`（`try_send`）。
struct Tail {
    /// 空 = 全要。匹配的是 kind 的**段**：`"obs"` 命中 `obs/person`，但不命中一个将来
    /// 可能出现的 `obsolete/x`。
    prefixes: Vec<String>,
    /// `Weak`：观察者活在世界日志里，而撤销它的 `Disposer` 活在这条连接的 `ConnState`
    /// 里。两边都用强引用就是一个环——连接关了，它和它拉着的一切都回收不掉。
    conn: std::sync::Weak<Conn>,
    /// `Some` = 还在**交接期**：补发还没跑完，新到的事件先在这里排队，不直接发。
    ///
    /// 交接期存在的理由是这两件事都必须成立：观察者必须在读 `since()` **之前**挂上
    /// （否则这两步之间 append 的事件谁也不发，客户端看到的是一个无声的缺口），而补发
    /// 的事件又必须排在实时事件**前面**（否则客户端先看到 seq 120、再看到 seq 100，
    /// 一个按 seq 推进自己游标的消费者会把补发的那一段整段丢掉）。先挂后补、期间入队，
    /// 两个条件同时满足。
    buffer: Mutex<Option<Vec<WorldEvent>>>,
}

/// `kind` 是不是落在 `p` 这个前缀底下。按 `/` 分段比，不是裸字符串前缀：`"act"` 不该
/// 命中一个将来可能出现的 `"actuator/..."`，否则一次订阅的含义会随着新 kind 的加入
/// 悄悄变宽。
///
/// 手写而不是 `kind.starts_with(&format!("{p}/"))`：这个函数跑在 `append` 的观察者链上
/// （每条事件 × 每个前缀一次），不该为此分配一个 `String`。
fn kind_has_prefix(kind: &str, p: &str) -> bool {
    kind == p || (kind.len() > p.len() && kind.as_bytes()[p.len()] == b'/' && kind.starts_with(p))
}

impl Tail {
    fn matches(&self, ev: &WorldEvent) -> bool {
        self.prefixes.is_empty()
            || self
                .prefixes
                .iter()
                .any(|p| kind_has_prefix(ev.kind.as_str(), p))
    }

    /// 观察者入口。`buffer` 这把锁只罩着一次 `push`，绝不跨 `notify`：`notify` 在队列满
    /// 的时候会走背压路径（`mark_closed` → 记一条 `health/rpc_backpressure` → 又一次
    /// `append` → 又一次跑到这里），带着锁进去就是同一个线程自己和自己死锁。
    fn offer(&self, ev: &WorldEvent) {
        if !self.matches(ev) {
            return;
        }
        {
            let mut g = self.buffer.lock();
            if let Some(buf) = g.as_mut() {
                buf.push(ev.clone());
                return;
            }
        }
        self.send(ev);
    }

    fn send(&self, ev: &WorldEvent) {
        let Some(conn) = self.conn.upgrade() else {
            return;
        };
        let params = serde_json::to_value(ev).unwrap_or(Value::Null);
        let _ = conn.notify(method::EVENT, params);
    }

    /// 补发完之后开闸：把交接期攒下的事件按序发出去，然后把 `buffer` 置空——置空和
    /// "确认它已经空了"在**同一个**临界区里完成，观察者因此不会挤进一个"我刚看完缓冲、
    /// 还没开闸"的窗口里把自己那条事件排到已发出去的事件前面。
    ///
    /// `already_sent` 是补发阶段最后走过的 seq：交接期入队的事件里，seq 不超过它的那些
    /// 补发已经走过一遍了（观察者是在读 `since()` 之前挂上的），跳过去，不重复。
    fn open_gate(&self, already_sent: u64) {
        for _ in 0..MAX_HANDOFF_ROUNDS {
            let (batch, done) = {
                let mut g = self.buffer.lock();
                match g.as_mut() {
                    None => (Vec::new(), true),
                    Some(buf) if buf.is_empty() => {
                        *g = None;
                        (Vec::new(), true)
                    }
                    Some(buf) => (std::mem::take(buf), false),
                }
            };
            for ev in &batch {
                if ev.seq > already_sent {
                    self.send(ev);
                }
            }
            if done {
                return;
            }
        }
        // 轮数用完了（有人在以极高频率写日志）：直接开闸，剩下的那一批照发。
        let rest = self.buffer.lock().take().unwrap_or_default();
        for ev in &rest {
            if ev.seq > already_sent {
                self.send(ev);
            }
        }
    }
}

/// 订阅世界日志：先把 `after` 之后**已经在**内存窗口里的事件补发一遍，再持续推送新的。
///
/// - `after` 缺省 `next_seq - 1`，也就是"只要新的"。
/// - `after` 已经滑出内存保留窗口（`CursorTooOld`），或者要补的历史超过
///   [`LOG_TAIL_BACKFILL_MAX`]：从能给的最旧那条开始。
/// - `from` 是这次订阅**覆盖范围**的第一条 seq：从它起的每一条事件都经过了这次订阅的
///   判定。注意"覆盖"不等于"会收到"——真正推给你的是其中匹配 `prefixes` 的那些，所以
///   第一条到达的通知的 `seq` 可能远大于 `from`，也可能一条都不来。比 `from` 更早的
///   历史不在这次订阅里，要的话走 `log/replay`（它按页读盘，本来就是干这个的）。
///
/// 一条连接只有一个 tail：再调一次先把上一次的观察者拆掉，否则同一条事件会按订阅次数
/// 重复推送。整个"拆旧 → 挂新 → 补发 → 存新"是一段临界区（[`Conn::lock_tail`]），
/// 因为 `dispatch_line` 给每个 Request 单独 `tokio::spawn`，一条连接上两次 `log/tail`
/// 是真并发——分开做的话两边都会读到"还没有旧订阅"，各挂一个观察者，后写的那个把前一个
/// 的 `Disposer` 覆盖掉（`FnOnce`，丢掉不等于执行），于是一个谁也撤不掉的观察者永远留在
/// 世界日志里。
pub async fn tail(s: Arc<Services>, conn: Arc<Conn>, params: Value) -> Result<Value, RpcError> {
    let p: LogTail = parse_params(&params)?;
    // 见 [`MAX_TAIL_PREFIXES`]：这个上限守的是 `append` 那条路，不是这次调用。
    if p.prefixes.len() > MAX_TAIL_PREFIXES {
        return Err(invalid_params(format_args!(
            "{} prefixes, over the limit of {MAX_TAIL_PREFIXES}",
            p.prefixes.len()
        )));
    }
    let after = p
        .after
        .unwrap_or_else(|| s.log.next_seq().saturating_sub(1));
    ensure_cleanup_hook(&conn);

    // 这一段里不持有任何 `parking_lot` 锁跨 `observe`/`since`/`notify`；这把是 tokio 的
    // 异步锁，只串行化"一条连接上的 log/tail"。
    let _tail_guard = conn.lock_tail().await;

    // 旧订阅先出锁再拆（`Disposer` 会去拿世界日志的锁）。
    let old = conn.state.lock().tail.take();
    if let Some(d) = old {
        d();
    }

    let tail = Arc::new(Tail {
        prefixes: p.prefixes,
        conn: Arc::downgrade(&conn),
        buffer: Mutex::new(Some(Vec::new())),
    });
    let observer = tail.clone();
    let disposer = s
        .log
        .observe(Arc::new(move |ev: &WorldEvent| observer.offer(ev)));

    // 补发。观察者已经挂上了，所以这一步和上一步之间 append 的事件不会丢——它们要么在
    // 这份快照里，要么在交接缓冲里（`open_gate` 会按 `already_sent` 去重）。
    let (mut backlog, mut from) = match s.log.since(after) {
        Ok(v) => (v, after.saturating_add(1)),
        Err(e) => match s.log.since(e.oldest_retained.saturating_sub(1)) {
            Ok(v) => (v, e.oldest_retained),
            // 保留窗口在这两条语句之间又滚了一次（要 >4096 次 append，极窄）。与其
            // `unwrap_or_default()` 报一个没有任何事件支撑的 `from`，不如如实把覆盖
            // 范围对齐到此刻还能给的最旧那条。
            Err(e2) => (Vec::new(), e2.oldest_retained),
        },
    };
    if backlog.len() > LOG_TAIL_BACKFILL_MAX {
        backlog.drain(..backlog.len() - LOG_TAIL_BACKFILL_MAX);
        from = backlog.first().map(|e| e.seq).unwrap_or(from);
    }
    let mut already_sent = from.saturating_sub(1);
    for ev in &backlog {
        already_sent = already_sent.max(ev.seq);
        if tail.matches(ev) {
            tail.send(ev);
        }
    }
    tail.open_gate(already_sent);

    conn.state.lock().tail = Some(disposer);
    release_if_closed(&conn);
    ok(LogTailResult { from })
}

/// 把调用方给的 `dir` 解析成一个**确实在这次 run 的日志根底下**的目录。
///
/// 三道闸门，缺一不可（少任何一道，这个方法就是"读任意目录"）：
///
/// 1. 这次 run 没有持久化日志 → `REFUSED`，**在看 `dir` 之前**。不然一个什么都不落盘的
///    内核照样能被当成任意目录的读取器用。
/// 2. `dir` 只能是一个**非空**的相对路径，且每一段都是 `Component::Normal`——`/`、
///    `C:\`、`.`、`..` 一律 `INVALID_PARAMS`。实际上就是"一个 run 标签"，这也是这个
///    参数唯一的用法。（`PathBuf::join` 不做规范化，所以 `"../secret"` 不挡就是直接
///    越狱；空串则会悄悄变成"最新的那个 run"的别名——在沙箱里，但没人文档过，也没人
///    测过。）
/// 3. 拼出来之后 `canonicalize` 两边，要求结果仍在日志根底下。这一道挡的是软链接：
///    日志根里一个指向别处的 `run-x` 符号链接，前两道全过，第三道才拦得住。它的强度
///    取决于日志根有多深——根就是 `/` 的话，任何 `Normal` 名字都"在根底下"，闸门形同
///    虚设，所以那种配置直接 `REFUSED`。
///
/// 任何 `canonicalize`/IO 失败都编成 [`REPLAY_DIR_UNAVAILABLE`] 这个固定文本，理由见
/// 那个常量。
fn replay_dir(s: &Services, dir: Option<String>) -> Result<PathBuf, RpcError> {
    // (1)
    let Some(run_dir) = s.log.run_dir() else {
        return Err(refused(
            "this run has no persisted world log (world-log has no `dir`)",
        ));
    };
    // 日志根 = run 目录的父目录（`world-log` 插件按 `<dir>/run-<ms>` 建 run 目录）。
    let root = match run_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let Some(dir) = dir else {
        // 没给就是"这次 run 自己"，不必再验。
        return Ok(run_dir);
    };
    // (2) 空串也在这里挡掉：它一个 component 都没有，`all(..)` 对它是平凡成立的，
    // 拼出来又正好是日志根本身。
    let dir = PathBuf::from(dir);
    if dir.components().next().is_none()
        || !dir.components().all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(invalid_params(
            "`dir` must be a non-empty relative path under the log root (a run tag), \
             with no `..`, `.` or root components",
        ));
    }
    // (3)
    let canonical_root = root
        .canonicalize()
        .map_err(|_| refused(REPLAY_DIR_UNAVAILABLE))?;
    // 根是文件系统根的话，`starts_with` 对任何绝对路径都成立，第 3 道闸门就不存在了
    // （`dir: "etc"` 会解析成 `/etc`）。这只可能来自一个荒唐的 `world-log.dir` 配置，
    // 但闸门的强度不该悄悄取决于日志根有多深。
    if canonical_root.components().count() < 2 {
        return Err(refused(REPLAY_DIR_UNAVAILABLE));
    }
    let resolved = canonical_root
        .join(&dir)
        .canonicalize()
        .map_err(|_| refused(REPLAY_DIR_UNAVAILABLE))?;
    if !resolved.starts_with(&canonical_root) {
        return Err(refused(REPLAY_DIR_UNAVAILABLE));
    }
    Ok(resolved)
}

/// 从段文件里读一页历史。
///
/// 读盘跑在 [`tokio::task::spawn_blocking`] 上：`Router::call` 的 future 跑在同一个
/// 多线程 runtime 的 worker 上，而那些 worker 还跑着 `Kernel::run`、脑的路由任务和
/// 500 Hz 的帧 ingest——一次几百 MB 的段文件解析会把其中一个 worker 钉住好几秒。
///
/// 用 [`read_segments_after`] 而不是 `read_segments`：后者读整个 run 再在内存里筛，
/// 于是翻 N 页的总代价是 O(N × 整个 run)。前者按段文件名跳过整段、读满 `limit + 1` 条
/// 就停，一页的代价就是一页。
///
/// `done` 是"过滤之后没有更多了"，不是"读完了整个目录"：调用方据此决定还要不要接着翻页。
///
/// `limit` 被夹进 `1..=RouterConfig::max_replay_page`（默认 4096），静默——见那个字段。
pub async fn replay(
    s: Arc<Services>,
    cfg: Arc<RouterConfig>,
    params: Value,
) -> Result<Value, RpcError> {
    let p: LogReplay = parse_params(&params)?;
    let dir = replay_dir(&s, p.dir)?;
    // 静默夹进 `1..=max_replay_page`，不报错：`done: false` 就是"还有，接着翻页"，调用方
    // 本来就得照着翻。不夹的话一个 `limit: 100000000` 会把整个 run 读进一个 `Vec` 再
    // 序列化成一行（见 `RouterConfig::max_replay_page`，Plan 3 最终评审 C2）。
    let (after, limit) = (p.after, p.limit.clamp(1, cfg.max_replay_page));
    let read = tokio::task::spawn_blocking(move || read_segments_after(&dir, after, limit))
        .await
        .map_err(|e| internal(format_args!("replay task failed: {e}")))?;
    let (events, done) = read.map_err(|e| match e {
        // 沙箱之内的 IO 失败仍然分得出"没有"和"不许读"（`io::Error` 的 Display 不带
        // 路径，所以漏不出路径，但这个区分本身还是个探针）。归到同一句固定文本里。
        // Gap / UnknownRequired 说的是日志**内容**坏了，不是文件系统的事，照实报——
        // 一个损坏的段文件正是运维要看见的东西。
        LogReadError::Io(_) => refused(REPLAY_DIR_UNAVAILABLE),
        other => internal(format_args!("replay failed: {other}")),
    })?;
    ok(LogReplayResult { events, done })
}

/// 把 sink 的缓冲刷到盘上。没有 sink（内存日志）时是空操作，不是错误——"没有要刷的"
/// 和"刷成功了"对调用方是同一件事。
pub fn flush(s: &Services) -> Result<Value, RpcError> {
    s.log.flush().map_err(internal)?;
    ok(Empty {})
}

#[cfg(test)]
mod tests {
    use super::kind_has_prefix;

    #[test]
    fn prefixes_match_kind_segments_not_raw_strings() {
        assert!(kind_has_prefix("obs/person", "obs"));
        assert!(kind_has_prefix("obs/person", "obs/person"));
        assert!(kind_has_prefix("act/proposed", "act"));
        // 段边界：一个将来出现的 `actuator/*` 不该被一个 `act` 订阅捎带上。
        assert!(!kind_has_prefix("actuator/cmd", "act"));
        assert!(!kind_has_prefix("safetynet/x", "safety"));
        assert!(!kind_has_prefix("obs/person", "obs/per"));
        assert!(!kind_has_prefix("obs", "obs/person"));
    }
}
