use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use oeh_proto::{EventDraft, WorldEvent};
use parking_lot::Mutex;

use crate::clock::Clock;
use crate::plugin::Disposer;

pub trait LogSink: Send {
    fn write(&mut self, ev: &WorldEvent) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
}

type Observer = Arc<dyn Fn(&WorldEvent) + Send + Sync>;

struct LogInner {
    next_seq: u64,
    recent: VecDeque<WorldEvent>,
    retain: usize,
    observers: Vec<(u64, Observer)>,
    next_obs: u64,
    sink: Option<Box<dyn LogSink>>,
}

/// `since()` 的游标已经滑出内存保留窗口。
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("cursor {after} predates the retained window; oldest retained seq is {oldest_retained}")]
pub struct CursorTooOld {
    pub after: u64,
    pub oldest_retained: u64,
}

/// append-only 世界日志。`append` 是内存提交点；持久化由 sink 完成，sink 失败不影响提交。
pub struct WorldLog {
    clock: Arc<dyn Clock>,
    inner: Arc<Mutex<LogInner>>,
    sink_errors: AtomicU64,
    run_dir: Option<PathBuf>,
}

impl WorldLog {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            inner: Arc::new(Mutex::new(LogInner {
                next_seq: 1,
                recent: VecDeque::new(),
                retain: 4096,
                observers: Vec::new(),
                next_obs: 1,
                sink: None,
            })),
            sink_errors: AtomicU64::new(0),
            run_dir: None,
        }
    }

    pub fn with_sink(self, sink: Box<dyn LogSink>) -> Self {
        self.inner.lock().sink = Some(sink);
        self
    }

    /// 这次 run 的日志目录。由 `world-log` 插件设置；没有持久化 sink 时是 `None`。
    /// 其他插件（会话日志）把自己的产物放进同一个 run 目录，两次运行就不会互相交叠。
    pub fn with_run_dir(mut self, dir: PathBuf) -> Self {
        self.run_dir = Some(dir);
        self
    }

    pub fn run_dir(&self) -> Option<PathBuf> {
        self.run_dir.clone()
    }

    pub fn with_retain(self, n: usize) -> Self {
        self.inner.lock().retain = n;
        self
    }

    pub fn append(&self, draft: EventDraft) -> WorldEvent {
        let (ev, observers) = {
            let mut g = self.inner.lock();
            // 时钟在锁内采样：在锁外取会让两个并发的 append 产生 seq N 的时间戳晚于 seq N+1。
            let t_kernel = self.clock.now_ns();
            let ev = draft.into_event(g.next_seq, t_kernel);
            g.next_seq += 1;
            if let Some(sink) = g.sink.as_mut() {
                if let Err(e) = sink.write(&ev) {
                    self.sink_errors.fetch_add(1, Ordering::SeqCst);
                    tracing::warn!(seq = ev.seq, error = %e, "world log sink write failed");
                }
            }
            while g.recent.len() >= g.retain.max(1) {
                g.recent.pop_front();
            }
            g.recent.push_back(ev.clone());
            let obs: Vec<Observer> = g.observers.iter().map(|(_, o)| o.clone()).collect();
            (ev, obs)
        };
        for o in observers {
            o(&ev);
        }
        ev
    }

    /// 观察者在锁释放后被调用；并发 append 之间的通知顺序不保证。需要顺序的消费者以 `seq`
    /// 为准，用 `since()` 拉取。
    pub fn observe(&self, f: Observer) -> Disposer {
        let mut g = self.inner.lock();
        let id = g.next_obs;
        g.next_obs += 1;
        g.observers.push((id, f));
        let inner = self.inner.clone();
        Box::new(move || inner.lock().observers.retain(|(i, _)| *i != id))
    }

    pub fn next_seq(&self) -> u64 {
        self.inner.lock().next_seq
    }

    /// 当前挂着的观察者数。`#[doc(hidden)]`——不是给正常调用方用的 API，存在的理由是
    /// 观察者泄漏（注册了却没人能撤销）是一类只能从计数上看出来的 bug：`append` 会在
    /// 每次提交时克隆一遍整张表，一条泄漏的观察者就是一笔永久的、随提交次数放大的开销，
    /// 而从外面看它什么症状都没有。
    #[doc(hidden)]
    pub fn observer_count(&self) -> usize {
        self.inner.lock().observers.len()
    }

    /// 内存里 seq > after 的事件。游标比保留窗口还旧时返回 [`CursorTooOld`]：静默截断会
    /// 让消费者以为自己没漏事件，必须由调用方决定是重放段文件还是重置游标。
    pub fn since(&self, after: u64) -> Result<Vec<WorldEvent>, CursorTooOld> {
        let g = self.inner.lock();
        let oldest = g.recent.front().map(|e| e.seq);
        if let Some(o) = oldest {
            if after.saturating_add(1) < o {
                return Err(CursorTooOld {
                    after,
                    oldest_retained: o,
                });
            }
        }
        Ok(g.recent.iter().filter(|e| e.seq > after).cloned().collect())
    }

    pub fn flush(&self) -> io::Result<()> {
        match self.inner.lock().sink.as_mut() {
            Some(s) => s.flush(),
            None => Ok(()),
        }
    }

    pub fn sink_errors(&self) -> u64 {
        self.sink_errors.load(Ordering::SeqCst)
    }
}

/// `seg-{first_seq:012}.jsonl`，超过 segment_bytes 开新段。
pub struct JsonlSegmentSink {
    dir: PathBuf,
    segment_bytes: u64,
    current: Option<BufWriter<File>>,
    written: u64,
}

impl JsonlSegmentSink {
    pub fn new(dir: &Path, segment_bytes: u64) -> io::Result<Self> {
        // 见 [`crate::fsutil`]：这个目录要么是刚被别人建出来的 run 目录，要么是这里建的，
        // 两种情况都要把权限落到 0700——段文件是审计材料。
        crate::fsutil::create_dir_private(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            segment_bytes,
            current: None,
            written: 0,
        })
    }

    /// `create_new`：段文件必须是新的。追加进已有段会让 seq 从 1 重新开始而文件里已经有
    /// 1..N，回放时必然报 Gap——宁可在写入点就响。日志目录按 run 分子目录，正常路径下不会撞。
    fn open_segment(&mut self, first_seq: u64) -> io::Result<()> {
        let path = self.dir.join(format!("seg-{first_seq:012}.jsonl"));
        let f = OpenOptions::new().create_new(true).write(true).open(path)?;
        self.current = Some(BufWriter::new(f));
        self.written = 0;
        Ok(())
    }
}

impl LogSink for JsonlSegmentSink {
    fn write(&mut self, ev: &WorldEvent) -> io::Result<()> {
        if self.current.is_none() || self.written >= self.segment_bytes {
            if let Some(mut c) = self.current.take() {
                c.flush()?;
            }
            self.open_segment(ev.seq)?;
        }
        let line = serde_json::to_string(ev).map_err(io::Error::other)?;
        let w = self.current.as_mut().expect("segment open");
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        self.written += line.len() as u64 + 1;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(c) = self.current.as_mut() {
            c.flush()?;
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LogReadError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("seq gap: expected {expected}, found {found}")]
    Gap { expected: u64, found: u64 },
    /// `file` 是出问题的那个段文件的**文件名**（不是路径），`line` 是**它自己**的行号
    /// （不是这次调用累计读了多少行）。翻页读取会整段跳过，一个累计计数在那里指不到
    /// 任何一行。
    ///
    /// 只给文件名：这条错误会原样经 `log/replay` 回到任何一个连得上边界的客户端
    /// （`oeh-rpc` 的 handler 把它当成"日志内容坏了"照实报，那是对的——一个损坏的段
    /// 文件正是运维要看见的东西），而一条绝对路径顺带泄露的是这台机器的目录布局。
    /// 段文件名（`seg-{first_seq:012}.jsonl`）加上调用方自己给的 `dir`，已经足够唯一
    /// 定位那一行。
    #[error("{file}:{line}: unknown required event; refusing to rebuild")]
    UnknownRequired { file: String, line: usize },
}

fn has_segment_file(dir: &Path) -> io::Result<bool> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if let Ok(name) = entry.file_name().into_string() {
            if name.starts_with("seg-") && name.ends_with(".jsonl") {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// 回放目录解析：目录里直接有 `seg-*.jsonl` 就用目录自身（旧布局/单个 run 目录）；否则按
/// 名字降序遍历 `run-*` 子目录，返回第一个含至少一个 `seg-*.jsonl` 的目录——world-log 插件
/// 每次 apply 一个 run（重启或重新激活都是新 run），更晚的 run 目录可能刚创建还没写入任何
/// 段文件，此时应回退到更早、确实有内容的 run。两者都没有时返回目录自身——那是一个空日志。
pub fn resolve_log_dir(dir: &Path) -> io::Result<PathBuf> {
    let mut runs: Vec<(String, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if name.starts_with("seg-") && name.ends_with(".jsonl") {
            return Ok(dir.to_path_buf());
        }
        if name.starts_with("run-") && entry.file_type()?.is_dir() {
            runs.push((name, entry.path()));
        }
    }
    runs.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, path) in runs {
        if has_segment_file(&path)? {
            return Ok(path);
        }
    }
    Ok(dir.to_path_buf())
}

/// 段文件名里编着它的第一条 seq（`seg-{first_seq:012}.jsonl`）；读不出来时返回 `None`，
/// 调用方据此保守处理（不跳过这个文件）。
fn segment_first_seq(path: &Path) -> Option<u64> {
    path.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("seg-"))
        .and_then(|n| n.strip_suffix(".jsonl"))
        .and_then(|n| n.parse::<u64>().ok())
}

/// 按段文件名排序的段文件列表（[`resolve_log_dir`] 选定的目录里的）。
fn segment_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("seg-") && n.ends_with(".jsonl"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}

/// [`parse_line`] 的结果。
enum ParsedLine {
    Event(WorldEvent),
    /// `required: false` 的未知事件：这一条跳过，但它的 `seq`（如果还读得出来）仍然要
    /// 推进连续性游标，否则紧跟它的那些正常 seq 会被误报成 Gap。
    SkippedOptional(Option<u64>),
}

/// 一行解成一条事件。`required: false` 的未知事件是"跳过"，`required: true` 的是错误
/// ——重建/回放不能悄悄丢掉一条声称自己必需的记录。
///
/// [`read_segments`]（重建）和 [`read_segments_after`]（翻页）共用这一份：两边曾经各抄
/// 一遍，字节级等价但是两处要同步维护的判定。
fn parse_line(line: &str, file: &Path, line_no: usize) -> Result<ParsedLine, LogReadError> {
    let unknown = || LogReadError::UnknownRequired {
        // 文件名，不是路径——见 [`LogReadError::UnknownRequired`] 上的说明。没有文件名
        // 的路径（不可能走到，`segment_files` 给的都是真实文件）退回整个路径的 Display。
        file: file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.display().to_string()),
        line: line_no,
    };
    match serde_json::from_str::<WorldEvent>(line) {
        Ok(ev) => Ok(ParsedLine::Event(ev)),
        Err(_) => {
            let raw: serde_json::Value = serde_json::from_str(line).map_err(|_| unknown())?;
            if raw.get("required") == Some(&serde_json::Value::Bool(false)) {
                Ok(ParsedLine::SkippedOptional(
                    raw.get("seq").and_then(|v| v.as_u64()),
                ))
            } else {
                Err(unknown())
            }
        }
    }
}

/// **翻页**读取：`seq > after` 的前 `limit` 条，外加"还有没有更多"。
///
/// 和 [`read_segments`] 的分工：那个是**重建**路径——读整个 run、校验 seq 从 1 起连续、
/// 把结果整个交出来（`oeh replay`、日志完整性检查用）。这个是**服务**路径
/// （`log/replay` 的一页）：
///
/// - 整段跳过：段文件名编着它的第一条 seq，所以"下一段的第一条 seq - 1 <= after"就意味着
///   这一段整个都在游标后面，连打开都不必；
/// - 读到 `limit + 1` 条就停，不把剩下的文件读完（返回值的 `bool` 就是这第 `limit + 1`
///   条存不存在：`done == 没有更多了`）。
///
/// 这两条合起来把一次翻页的代价从"整个 run"降到"这一页"。一个控制台按默认的
/// `limit: 1000` 翻一个 4 GB 的 run，原来是每页都重读重解析 4 GB（总量 O(n²)），现在是
/// 每页只碰它真正要的那一段。
///
/// 代价说清楚：跳过整段意味着这里**不**做 [`read_segments`] 的 seq 连续性校验——一次
/// 翻页读不到它跳过的那些段，也就无从校验。要"这份日志完整吗"这个结论，用
/// [`read_segments`]。单行的 `required` 语义两边一致：`required: false` 的未知事件跳过，
/// `required: true` 的报 [`LogReadError::UnknownRequired`]。
pub fn read_segments_after(
    dir: &Path,
    after: u64,
    limit: usize,
) -> Result<(Vec<WorldEvent>, bool), LogReadError> {
    let dir = resolve_log_dir(dir)?;
    let files = segment_files(&dir)?;
    let firsts: Vec<Option<u64>> = files.iter().map(|p| segment_first_seq(p)).collect();
    let mut out = Vec::new();
    for (i, f) in files.iter().enumerate() {
        // 下一段从 `next_first` 开始 ⇒ 这一段最后一条是 `next_first - 1`。它不比 `after`
        // 新，这一段就整个在游标后面。最后一段没有"下一段"，无从判断，照常读。
        if let Some(Some(next_first)) = firsts.get(i + 1) {
            if next_first.saturating_sub(1) <= after {
                continue;
            }
        }
        // 行号按**文件**从 1 起：整段跳过意味着一个累计计数指不到任何一行。
        for (line_no, line) in BufReader::new(File::open(f)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let ParsedLine::Event(ev) = parse_line(&line, f, line_no + 1)? else {
                continue;
            };
            if ev.seq <= after {
                continue;
            }
            if out.len() == limit {
                // 这是第 `limit + 1` 条：只用来回答"还有更多"，不交出去。
                return Ok((out, false));
            }
            out.push(ev);
        }
    }
    Ok((out, true))
}

/// 回放读取：先用 [`resolve_log_dir`] 选定 run 目录，再按段文件名排序，校验 seq 从 1 起连续
/// （第一条必须是 seq==1，缺失的首段视为 Gap），跳过 required=false 的未知事件（跳过的行也
/// 推进连续性游标，避免把它们之后正常的 seq 误报为 Gap）。空目录是合法的空日志。
pub fn read_segments(dir: &Path) -> Result<Vec<WorldEvent>, LogReadError> {
    let dir = resolve_log_dir(dir)?;
    let files = segment_files(&dir)?;
    let mut out = Vec::new();
    let mut expected: Option<u64> = Some(1);
    for f in files {
        for (line_no, line) in BufReader::new(File::open(&f)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let seq = match parse_line(&line, &f, line_no + 1)? {
                // 跳过的那一行也推进游标，不然它后面正常的 seq 会被误报成 Gap。
                ParsedLine::SkippedOptional(seq) => seq,
                ParsedLine::Event(ev) => {
                    let seq = ev.seq;
                    out.push(ev);
                    Some(seq)
                }
            };
            let Some(seq) = seq else { continue };
            if let Some(exp) = expected {
                if seq != exp {
                    // 这一条已经 push 进去了，但整个调用要以 Err 收场，所以无所谓。
                    return Err(LogReadError::Gap {
                        expected: exp,
                        found: seq,
                    });
                }
            }
            expected = Some(seq + 1);
        }
    }
    Ok(out)
}
