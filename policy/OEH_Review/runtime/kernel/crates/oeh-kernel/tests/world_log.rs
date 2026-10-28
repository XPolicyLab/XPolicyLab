use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use oeh_kernel::world_log::{
    read_segments, read_segments_after, resolve_log_dir, CursorTooOld, JsonlSegmentSink,
    LogReadError, WorldLog,
};
use oeh_kernel::ManualClock;
use oeh_proto::{EventDraft, EventKind};
use serde_json::json;

fn draft(kind: &str) -> EventDraft {
    EventDraft::new(EventKind::new(kind).unwrap(), "test", json!({"k": 1}))
}

#[test]
fn append_assigns_contiguous_seq_and_kernel_time() {
    let clock = Arc::new(ManualClock::new(50));
    let log = WorldLog::new(clock.clone());
    let a = log.append(draft("obs/person"));
    clock.advance(7);
    let b = log.append(draft("act/proposed"));
    assert_eq!((a.seq, a.t_kernel), (1, 50));
    assert_eq!((b.seq, b.t_kernel), (2, 57));
    assert_eq!(log.next_seq(), 3);
    assert_eq!(log.since(1).unwrap().len(), 1);
    assert_eq!(log.since(0).unwrap()[0].seq, 1);
}

#[test]
fn observers_see_events_until_disposed() {
    let log = WorldLog::new(Arc::new(ManualClock::new(0)));
    let last = Arc::new(AtomicU64::new(0));
    let l2 = last.clone();
    let d = log.observe(Arc::new(move |e| l2.store(e.seq, Ordering::SeqCst)));
    log.append(draft("obs/x"));
    assert_eq!(last.load(Ordering::SeqCst), 1);
    d();
    log.append(draft("obs/x"));
    assert_eq!(last.load(Ordering::SeqCst), 1);
}

#[test]
fn jsonl_segments_roll_and_replay_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let sink = JsonlSegmentSink::new(dir.path(), 200).unwrap(); // 很小的段，强制滚动
    let log = WorldLog::new(Arc::new(ManualClock::new(0))).with_sink(Box::new(sink));
    for i in 0..10 {
        log.append(draft(if i % 2 == 0 { "obs/a" } else { "health/b" }));
    }
    log.flush().unwrap();
    let files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(
        files.len() >= 2,
        "expected multiple segments, got {files:?}"
    );
    assert!(files
        .iter()
        .all(|f| f.starts_with("seg-") && f.ends_with(".jsonl")));
    let events = read_segments(dir.path()).unwrap();
    assert_eq!(events.len(), 10);
    assert_eq!(
        events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (1..=10).collect::<Vec<_>>()
    );
    assert_eq!(log.sink_errors(), 0);
}

#[test]
fn replay_rejects_gaps_and_unknown_required_events() {
    let dir = tempfile::tempdir().unwrap();
    let mk = |seq: u64, extra: &str| {
        format!(r#"{{"seq":{seq},"t_kernel":1,"kind":"obs/a","source":"t","payload":{{}}{extra}}}"#)
    };
    std::fs::write(
        dir.path().join("seg-000000000001.jsonl"),
        format!("{}\n{}\n", mk(1, ""), mk(3, "")),
    )
    .unwrap();
    assert!(matches!(
        read_segments(dir.path()),
        Err(LogReadError::Gap {
            expected: 2,
            found: 3
        })
    ));

    let dir = tempfile::tempdir().unwrap();
    // 第二行 kind 非法但 required=false → 跳过；第三行非法且 required 缺省 → 拒绝
    let bad_optional = r#"{"seq":2,"t_kernel":1,"kind":"future/thing","source":"t","payload":{},"required":false}"#;
    std::fs::write(
        dir.path().join("seg-000000000001.jsonl"),
        format!("{}\n{}\n{}\n", mk(1, ""), bad_optional, mk(3, "")),
    )
    .unwrap();
    let ev = read_segments(dir.path()).unwrap();
    assert_eq!(ev.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![1, 3]);

    let dir = tempfile::tempdir().unwrap();
    let bad_required = r#"{"seq":2,"t_kernel":1,"kind":"future/thing","source":"t","payload":{}}"#;
    std::fs::write(
        dir.path().join("seg-000000000001.jsonl"),
        format!("{}\n{}\n", mk(1, ""), bad_required),
    )
    .unwrap();
    assert!(matches!(
        read_segments(dir.path()),
        Err(LogReadError::UnknownRequired { line: 2, .. })
    ));
}

#[test]
fn replay_rejects_missing_leading_segment() {
    let dir = tempfile::tempdir().unwrap();
    let mk = |seq: u64| {
        format!(r#"{{"seq":{seq},"t_kernel":1,"kind":"obs/a","source":"t","payload":{{}}}}"#)
    };
    // 第一段从 seq=2 开始：说明 seg-…01 丢了，必须拒绝重建
    std::fs::write(
        dir.path().join("seg-000000000002.jsonl"),
        format!("{}\n{}\n", mk(2), mk(3)),
    )
    .unwrap();
    assert!(matches!(
        read_segments(dir.path()),
        Err(LogReadError::Gap {
            expected: 1,
            found: 2
        })
    ));
    // 空目录仍然是合法的空日志
    let empty = tempfile::tempdir().unwrap();
    assert!(read_segments(empty.path()).unwrap().is_empty());
}

#[test]
fn read_segments_resolves_the_newest_run_dir() {
    let base = tempfile::tempdir().unwrap();
    let mk = |seq: u64, kind: &str| {
        format!(r#"{{"seq":{seq},"t_kernel":1,"kind":"{kind}","source":"t","payload":{{}}}}"#)
    };
    // 两次 boot → 两个 run 子目录，seq 各自从 1 起
    for (run, n) in [("run-0000000000001", 1u64), ("run-0000000000002", 3)] {
        let d = base.path().join(run);
        std::fs::create_dir_all(&d).unwrap();
        let body: String = (1..=n).map(|s| mk(s, "obs/a") + "\n").collect();
        std::fs::write(d.join("seg-000000000001.jsonl"), body).unwrap();
    }
    assert_eq!(
        resolve_log_dir(base.path()).unwrap(),
        base.path().join("run-0000000000002")
    );
    let ev = read_segments(base.path()).unwrap();
    assert_eq!(ev.len(), 3, "newest run wins, no gap error");
    // 单个 run 目录本身也能直接回放
    assert_eq!(
        read_segments(&base.path().join("run-0000000000001"))
            .unwrap()
            .len(),
        1
    );
    // 目录里直接有段文件时用目录自身（兼容旧布局）
    let flat = tempfile::tempdir().unwrap();
    std::fs::write(
        flat.path().join("seg-000000000001.jsonl"),
        mk(1, "obs/a") + "\n",
    )
    .unwrap();
    std::fs::create_dir_all(flat.path().join("run-0000000000009")).unwrap();
    assert_eq!(resolve_log_dir(flat.path()).unwrap(), flat.path());
    // 既没有段也没有 run：空日志
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(resolve_log_dir(empty.path()).unwrap(), empty.path());
    assert!(read_segments(empty.path()).unwrap().is_empty());
}

/// I6: 游标落在保留窗口之外必须报错，而不是静默返回截断的结果——消费者会以为自己没漏事件。
#[test]
fn since_rejects_a_cursor_older_than_the_retained_window() {
    let log = WorldLog::new(Arc::new(ManualClock::new(0))).with_retain(2);
    for _ in 0..5 {
        log.append(draft("obs/x"));
    }
    assert_eq!(
        log.since(1),
        Err(CursorTooOld {
            after: 1,
            oldest_retained: 4
        })
    );
    assert_eq!(
        log.since(3)
            .unwrap()
            .iter()
            .map(|e| e.seq)
            .collect::<Vec<_>>(),
        vec![4, 5]
    );
    assert_eq!(
        log.since(4)
            .unwrap()
            .iter()
            .map(|e| e.seq)
            .collect::<Vec<_>>(),
        vec![5]
    );
    // 空日志 + 游标 0 是合法的空结果
    let empty = WorldLog::new(Arc::new(ManualClock::new(0)));
    assert!(empty.since(0).unwrap().is_empty());
}

#[test]
fn resolve_log_dir_skips_empty_newer_run() {
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("run-0000000000001");
    let empty = dir.path().join("run-0000000000002");
    std::fs::create_dir_all(&good).unwrap();
    std::fs::create_dir_all(&empty).unwrap();
    std::fs::write(
        good.join("seg-000000000001.jsonl"),
        r#"{"seq":1,"t_kernel":1,"kind":"obs/a","source":"t","payload":{}}"#,
    )
    .unwrap();
    assert_eq!(oeh_kernel::resolve_log_dir(dir.path()).unwrap(), good);
    assert_eq!(read_segments(dir.path()).unwrap().len(), 1);
}

#[test]
fn since_saturates_at_u64_max() {
    let log = WorldLog::new(Arc::new(ManualClock::new(0)));
    log.append(draft("obs/a"));
    assert!(log.since(u64::MAX).unwrap().is_empty());
}

/// `read_segments_after` 是 `log/replay` 的**翻页**读取：一页的代价应该是一页，不是整个
/// run。两件事一起验：分页的语义（内容、边界、`done`），以及它真的跳过了整段——把已经
/// 翻过去的那些段文件删掉，翻页照样对得上，说明那一页根本没打开过它们。
#[test]
fn read_segments_after_pages_without_reading_the_whole_run() {
    let dir = tempfile::tempdir().unwrap();
    // 很小的段，30 条事件滚出好几个段文件。
    let sink = JsonlSegmentSink::new(dir.path(), 200).unwrap();
    let log = WorldLog::new(Arc::new(ManualClock::new(0))).with_sink(Box::new(sink));
    for _ in 0..30 {
        log.append(draft("obs/a"));
    }
    log.flush().unwrap();

    let (page, done) = read_segments_after(dir.path(), 0, 10).unwrap();
    assert_eq!(
        page.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (1..=10).collect::<Vec<_>>()
    );
    assert!(!done, "还有 20 条");

    let (page, done) = read_segments_after(dir.path(), 25, 10).unwrap();
    assert_eq!(
        page.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (26..=30).collect::<Vec<_>>()
    );
    assert!(done, "最后一页");

    // `limit: 0` 是"只问还有没有"，不是"读完"。
    let (page, done) = read_segments_after(dir.path(), 29, 0).unwrap();
    assert!(page.is_empty() && !done);
    let (page, done) = read_segments_after(dir.path(), 30, 0).unwrap();
    assert!(page.is_empty() && done);

    // 整段跳过：把只含 seq <= 20 的段文件全删掉，翻 `after: 25` 那一页照样对——它本来
    // 就不该打开那些文件。（段文件名编着它的第一条 seq，所以"下一段的第一条 - 1 <= after"
    // 的那些段整个在游标后面。）
    let mut files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let firsts: Vec<u64> = files
        .iter()
        .map(|p| {
            p.file_name().unwrap().to_str().unwrap()["seg-".len()..]
                .trim_end_matches(".jsonl")
                .parse()
                .unwrap()
        })
        .collect();
    let mut removed = 0;
    for (i, f) in files.iter().enumerate() {
        if firsts.get(i + 1).is_some_and(|next| next - 1 <= 25) {
            std::fs::remove_file(f).unwrap();
            removed += 1;
        }
    }
    assert!(removed > 0, "段没滚起来，这个断言什么也没验到: {firsts:?}");
    let (page, done) = read_segments_after(dir.path(), 25, 10).unwrap();
    assert_eq!(
        page.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (26..=30).collect::<Vec<_>>()
    );
    assert!(done);
}

/// Fix round 2, R4: the pager's parse error has to say **where** — and no more than where.
///
/// `read_segments_after` skips whole segments, so a counter of "lines this call happened to
/// read" points at nothing in any file. The error carries the segment's **file name** and
/// that file's own line number instead. Not the path: this text reaches any client that can
/// reach the boundary (`log/replay` reports a corrupt segment verbatim, deliberately), and
/// an absolute path would hand it this machine's directory layout for free (Task 4
/// re-review round 2 nit, carried into Task 5).
#[test]
fn read_segments_after_locates_a_bad_line_by_file_and_line() {
    let dir = tempfile::tempdir().unwrap();
    let good = |seq: u64| {
        format!(r#"{{"seq":{seq},"t_kernel":1,"kind":"obs/a","source":"t","payload":{{}}}}"#)
    };
    let bad = r#"{"seq":12,"t_kernel":1,"kind":"future/thing","source":"t","payload":{}}"#;
    // 第一段整段在游标后面（下一段从 11 起 ⇒ 这一段最后一条是 10 <= after），翻页时
    // 根本不会打开它——所以一个累计行号会把第二段的第 2 行报成"第 12 行"。
    std::fs::write(
        dir.path().join("seg-000000000001.jsonl"),
        (1..=10).map(good).collect::<Vec<_>>().join("\n") + "\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("seg-000000000011.jsonl"),
        format!("{}\n{}\n", good(11), bad),
    )
    .unwrap();

    let err = read_segments_after(dir.path(), 10, 100).unwrap_err();
    let msg = err.to_string();
    assert!(
        matches!(err, LogReadError::UnknownRequired { line: 2, .. }),
        "per-file line number, not a running total: {msg}"
    );
    assert!(
        msg.contains("seg-000000000011.jsonl"),
        "the message must name the segment: {msg}"
    );
    assert!(
        !msg.contains(dir.path().to_str().unwrap()),
        "the message must not carry the segment's path: {msg}"
    );
}
