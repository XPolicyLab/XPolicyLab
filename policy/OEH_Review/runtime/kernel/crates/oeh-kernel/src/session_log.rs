use crate::observation::Observation;
use crate::proposal::Outcome;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::time::{Duration, Instant};

/// 每个脑实例一份：模型看到的和它说过的。与世界日志通过 `source_seq_range` 关联。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    /// 一次 attach 的开场白：这份文件从这里开始属于这个 epoch 的 turn 编号。
    Attached {
        epoch: u64,
        brain_id: String,
        t_kernel: u64,
    },
    TurnStart {
        turn: u64,
        t_kernel: u64,
    },
    Observation {
        turn: u64,
        observation: Observation,
    },
    Steer {
        turn: u64,
        observation: Observation,
    },
    Interrupt {
        turn: u64,
        t_kernel: u64,
    },
    TurnEnd {
        turn: u64,
        t_kernel: u64,
        reason: String,
    },
    Proposal {
        turn: u64,
        capability: String,
        params: Value,
        trace_id: String,
    },
    /// 上面那条 `Proposal` 的结局。靠 `proposal_id`（`{trace_id}-p{n}`）配对。
    Outcome {
        turn: u64,
        proposal_id: String,
        outcome: Outcome,
    },
    Say {
        turn: u64,
        text: String,
    },
    AskHuman {
        turn: u64,
        question: String,
    },
    /// 模型这一步吐出来的自由文本（Plan 4：一个 turn 里可以有很多步）。`step` 是
    /// **turn 内**的步号，从 0 起。
    ModelOutput {
        turn: u64,
        step: u64,
        text: String,
    },
    /// 模型这一步要调的工具。`call_id` 由脑分配，和下面那条 `ToolResult` 配对——
    /// 一步里可以有多个并发的工具调用，光靠 `step` 分不开。
    ToolCall {
        turn: u64,
        step: u64,
        call_id: String,
        name: String,
        args: Value,
    },
    /// 上面那条 `ToolCall` 的结果。`ok` 是"这次调用成不成"，`result` 是喂回模型的那一份。
    ToolResult {
        turn: u64,
        step: u64,
        call_id: String,
        ok: bool,
        result: Value,
    },
}

/// 写线程的命令。日志行本身已经序列化好（序列化在调用线程做，写线程只管落盘）。
enum Cmd {
    Line(Vec<u8>),
    /// 排空到这一刻为止的全部行，然后回执。FIFO 保证"到这一刻为止"的含义。
    Flush(SyncSender<()>),
    /// 测试用：让写线程睡一会儿，模拟一块卡住的盘（见
    /// [`SessionLog::stall_writer_for_tests`]）。
    Stall(Duration),
}

/// 写线程队列的深度。满了就丢最新的那条并计数（见 [`SessionLog::record`]）。
const QUEUE: usize = 1024;
/// 每这么多行 flush 一次 `BufWriter`。
const BATCH: usize = 16;
/// 空闲这么久也 flush 一次：一个每分钟只说一句话的脑，不该让那句话在缓冲里等第 16 条。
const IDLE_FLUSH: Duration = Duration::from_millis(50);
/// 丢弃的报告节流：每份会话日志每秒最多一条 `warn!`（计数照样每条都计）。丢弃的触发
/// 条件就是"持续过载"，每丢一条打一行日志等于在最不该增加写压力的时刻按比例增加写
/// 压力。和 `health/op_backpressure`、`health/frame_egress_drop` 同一个做法。
const DROP_REPORT_PERIOD: Duration = Duration::from_secs(1);
/// [`SessionLog::flush`] 的总预算。写线程卡住时调用方必须能回来。
const FLUSH_BUDGET: Duration = Duration::from_secs(1);

/// 一个脑的会话日志。**写盘在自己的线程上**：`record` 只把一行序列化好丢进一个有界
/// 队列（`try_send`）就回来，绝不在调用线程上碰文件。
///
/// 为什么：`record` 的调用线程是 brains 的路由任务（500 Hz 的世界日志观察者下游）和
/// 任何一个 `TurnHandle::propose` 的调用者。一次 `write_all + flush` 是两个系统调用，
/// 落在一块慢盘上就是几十毫秒——那几十毫秒里整条路由链（所有脑的唤醒、steer、
/// interrupt）停着。会话日志是**记账**，它的延迟不该变成内核的延迟。
///
/// 代价是有界队列满时会丢行，而**缺口可以落在任何地方，不只是尾部**：`try_send` 丢的
/// 是"入队的那一刻队列恰好满"的那一条，写线程随后腾出一格，下一条又进得去。所以文件里
/// 完全可能有 B 没有 A。`dropped()` 数得出丢了多少（`warn!` 每秒最多一条）。
///
/// 保留下来的行**严格保序**：`record` 可以被任意多个线程并发调用（路由任务、任何一个
/// `TurnHandle::propose` 的调用者），但 std mpsc 按 `try_send` 返回 `Ok` 的先后给出全序，
/// 写线程是唯一的消费者。也就是：**有序，但可能有缺口**。
///
/// 读会话的人（Plan 4 的回放）必须按这条语义写：一条被丢的 `ToolCall` 会留下一条
/// `call_id` 配不上任何调用的孤儿 `ToolResult`，反过来也一样——按 `call_id` 配对，配不上
/// 的当作"这一步的记账掉了"，而不是当作错误。
pub struct SessionLog {
    path: PathBuf,
    /// `Drop` 里要先把它放掉（写线程据此收到 `Disconnected` 并退出），再 join。
    tx: Option<SyncSender<Cmd>>,
    dropped: AtomicU64,
    /// 上一条丢弃 `warn!` 的时刻。`Mutex` 而不是原子：这条路径每秒最多做一次真正的
    /// 工作，而"这一秒该由谁来报"不该有两个线程同时认为自己赢了（和
    /// `oeh_rpc::frames::Drops` 同一个写法）。只在丢弃分支上取，绝不跨 `warn!` 持有。
    last_warn: Mutex<Option<Instant>>,
    writer: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl SessionLog {
    /// 打开（或追加到）`{dir}/{brain_id}.session.jsonl` 并起一条写线程。
    ///
    /// 建目录和 `open` 仍然在**调用线程**上同步做：`Brains::attach` 要靠这一步的
    /// `io::Result` 当场失败（一个打不开的会话日志不该变成一个已经挂上、却什么都记不下
    /// 来的脑）。真正的写盘从此归写线程。
    pub fn open(dir: &Path, brain_id: &str) -> io::Result<Self> {
        // `create_dir_private` 而不是 `create_dir_all`：见 [`crate::fsutil`]——进程里别的
        // 线程会短暂地动 `umask`，而一个没有 `x` 位的会话目录，症状是下一行的 `open` 报
        // `EACCES`，原因在另一个调用栈里。
        crate::fsutil::create_dir_private(dir)?;
        let path = dir.join(format!("{brain_id}.session.jsonl"));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let (tx, rx) = sync_channel::<Cmd>(QUEUE);
        let writer = std::thread::Builder::new()
            .name(format!("oeh-session-{brain_id}"))
            .spawn(move || write_loop(file, rx))?;
        Ok(Self {
            path,
            tx: Some(tx),
            dropped: AtomicU64::new(0),
            last_warn: Mutex::new(None),
            writer: Mutex::new(Some(writer)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 队列满时被丢掉的条目数（累计，不清零）。
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// 记一条。**不阻塞**：序列化 + 一次 `try_send`。
    ///
    /// 队列满 → 这一条丢掉、计数、返回 `Ok`：丢一行会话日志不是调用方能处理的错误
    /// （提案已经提了、话已经说了），把它报成 `Err` 只会让每个调用点各自决定要不要因为
    /// 记账失败而改变行为。写线程没了（panic 过）才是 `Err`。
    ///
    /// `warn!` 每 [`DROP_REPORT_PERIOD`] 最多一条（载荷是累计的 `dropped`）：丢弃只发生
    /// 在持续过载时，一条一行的日志正好在最不该写的时候按比例增加写压力。
    pub fn record(&self, entry: &SessionEntry) -> io::Result<()> {
        let mut line = serde_json::to_vec(entry).map_err(io::Error::other)?;
        line.push(b'\n');
        let Some(tx) = self.tx.as_ref() else {
            return Err(io::Error::other("session log is closed"));
        };
        match tx.try_send(Cmd::Line(line)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                let n = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                let due = {
                    let mut last = self.last_warn.lock();
                    let now = Instant::now();
                    let due = last.is_none_or(|t| now.duration_since(t) >= DROP_REPORT_PERIOD);
                    if due {
                        *last = Some(now);
                    }
                    due
                };
                if due {
                    tracing::warn!(
                        path = %self.path.display(),
                        dropped = n,
                        "session log queue is full; dropping entries"
                    );
                }
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(io::Error::other("session log writer thread is gone"))
            }
        }
    }

    /// 测试用：让写线程在排到这条命令时睡 `d`，模拟一块卡住的盘。排在它后面的行会把
    /// 队列填满，于是 `record` 开始丢、`flush()`/`Drop` 开始等——那正是"detach 不能把这
    /// 段等待关进 brains 锁里"要钉住的场景。
    #[doc(hidden)]
    pub fn stall_writer_for_tests(&self, d: Duration) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.try_send(Cmd::Stall(d));
        }
    }

    /// 等写线程把此刻为止入队的行全部落盘。有界（[`FLUSH_BUDGET`]）：写线程卡在一次
    /// 写上时，调用方仍然必须回来。
    pub fn flush(&self) -> io::Result<()> {
        self.flush_within(FLUSH_BUDGET)
    }

    /// [`SessionLog::flush`]，但预算由调用方给。多份日志要在**一个**总预算里排空时用它
    /// （见 [`crate::Brains::flush_sessions`]）：n 份日志不该变成 n 倍的等待。
    ///
    /// `budget` 为 0 也会老老实实试一次（投得进去、且写线程此刻就绪时照样成功），只是
    /// 不等。
    pub(crate) fn flush_within(&self, budget: Duration) -> io::Result<()> {
        let Some(tx) = self.tx.as_ref() else {
            return Err(io::Error::other("session log is closed"));
        };
        let deadline = Instant::now() + budget;
        let (ack, ack_rx) = sync_channel::<()>(1);
        let mut cmd = Cmd::Flush(ack);
        loop {
            match tx.try_send(cmd) {
                Ok(()) => break,
                Err(TrySendError::Full(back)) => {
                    // 队列满：写线程正在排，等它腾出位置——但只等到 deadline。
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "session log flush timed out",
                        ));
                    }
                    cmd = back;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(io::Error::other("session log writer thread is gone"));
                }
            }
        }
        match ack_rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(()) => Ok(()),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "session log flush timed out",
            )),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err(io::Error::other("session log writer thread is gone"))
            }
        }
    }
}

impl Drop for SessionLog {
    /// 排空、关队列、收工。契约是**有界的尽力而为**，不是无条件的耐久性：
    ///
    /// - `flush()` 成功（常态）：`drop(log)` 返回之后，已经 `record` 过的每一条都在盘上，
    ///   写线程已经 join、文件句柄已经关闭。测试与"关机后回放会话"依赖的是**这一条**。
    /// - `flush()` 超时（写线程卡在一次系统调用里，最长 [`FLUSH_BUDGET`]）：**不 join**。
    ///   队列里剩下的行（最多 [`QUEUE`] 条）和 `BufWriter` 里没落盘的那些（最多
    ///   [`BATCH`] - 1 条）此刻**还不在文件里**，文件句柄也还被那条线程攥着（`drop` 返回
    ///   不等于 fd 已释放）。进程继续活着的话它醒来后会自己写完并收尾；**进程随即退出
    ///   （`main` 返回 / 运行时析构）的话，那一段就此丢失**，而 `dropped()` **不**计它
    ///   ——那个计数只数入队失败的条目。
    ///
    /// 为什么不 join：那是唯一一条没有时间上界的等待，而它换不来有界的耐久性——写线程
    /// 卡多久就等多久。有界是硬要求（`drop` 会在 detach、关机这些路径上跑，见
    /// `Brains` 的 detach：它必须在 brains 锁之外发生，而且不该没有上界）。
    ///
    /// **要保证关机之后会话日志是完整的，别指望 `Drop`：先调
    /// [`crate::Brains::flush_sessions`]**（`Kernel::run` 的关机路径已经这么做了）。它在
    /// 撤销之前排空，那时写线程通常早已追上。
    fn drop(&mut self) {
        let flushed = self.flush();
        if let Err(err) = &flushed {
            tracing::warn!(path = %self.path.display(), %err, "session log final flush failed");
        }
        // 先放掉发送端：写线程的 `recv_timeout` 这才会收到 `Disconnected` 并退出。
        self.tx = None;
        let handle = self.writer.lock().take();
        if let (Ok(()), Some(h)) = (flushed, handle) {
            let _ = h.join();
        }
    }
}

/// 写线程：唯一一个在 `open` 之后碰这个文件的东西。
fn write_loop(file: File, rx: Receiver<Cmd>) {
    let mut w = BufWriter::new(file);
    let mut unflushed = 0usize;
    loop {
        match rx.recv_timeout(IDLE_FLUSH) {
            Ok(Cmd::Line(line)) => {
                if let Err(err) = w.write_all(&line) {
                    tracing::warn!(%err, "session log write failed");
                }
                unflushed += 1;
                if unflushed >= BATCH {
                    flush_quietly(&mut w, &mut unflushed);
                }
            }
            Ok(Cmd::Stall(d)) => std::thread::sleep(d),
            Ok(Cmd::Flush(ack)) => {
                flush_quietly(&mut w, &mut unflushed);
                // 回执发不出去（调用方已经超时走了）不是错误。
                let _ = ack.send(());
            }
            Err(RecvTimeoutError::Timeout) => flush_quietly(&mut w, &mut unflushed),
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    flush_quietly(&mut w, &mut 0);
}

fn flush_quietly(w: &mut BufWriter<File>, unflushed: &mut usize) {
    if let Err(err) = w.flush() {
        tracing::warn!(%err, "session log flush failed");
    }
    *unflushed = 0;
}

#[derive(Debug, thiserror::Error)]
pub enum SessionReadError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("line {line}: cannot parse session entry")]
    Parse { line: usize },
}

pub fn read_session(path: &Path) -> Result<Vec<SessionEntry>, SessionReadError> {
    let mut out = Vec::new();
    for (i, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).map_err(|_| SessionReadError::Parse { line: i + 1 })?);
    }
    Ok(out)
}
