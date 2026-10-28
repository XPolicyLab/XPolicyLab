use std::time::Duration;

use oeh_kernel::fiber::FiberState;
use oeh_kernel::kernel::{BootError, Kernel};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::world_log::read_segments;
use oeh_kernel::WorldLog;
use oeh_proto::{EventDraft, EventKind};
use serde_json::json;

fn rows(dir: &std::path::Path) -> Vec<EntryRow> {
    vec![
        EntryRow::new("frame-counter")
            .with_config(json!({"stream_id": "lowstate", "period_ms": 100})),
        EntryRow::new("mock-stream").with_config(json!({"stream_id": "lowstate", "hz": 200.0})),
        EntryRow::new("world-log").with_config(json!({"dir": dir, "segment_mb": 1})),
        EntryRow::new("frame-store"),
        EntryRow::new("clock"),
    ]
}

#[tokio::test]
async fn mock_stream_flows_into_world_log_and_shutdown_disposes() {
    let dir = tempfile::tempdir().unwrap();
    let kernel = Kernel::boot(rows(dir.path())).unwrap();
    assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
    assert!(
        kernel
            .states()
            .iter()
            .all(|(_, s)| *s == FiberState::Active),
        "{:?}",
        kernel.states()
    );
    let handle = kernel.handle();
    let log = kernel
        .registry()
        .get::<WorldLog>(&"world-log".into())
        .unwrap();

    let run = tokio::spawn(kernel.run());
    tokio::time::sleep(Duration::from_millis(450)).await;
    handle
        .emit(EventDraft::new(
            EventKind::new("obs/test").unwrap(),
            "test",
            json!({"hello": 1}),
        ))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    handle.shutdown().unwrap();
    run.await.unwrap().unwrap();

    let events = read_segments(dir.path()).unwrap();
    let counts: Vec<_> = events
        .iter()
        .filter(|e| e.kind.as_str() == "health/frame_count")
        .collect();
    assert!(
        counts.len() >= 3,
        "expected periodic frame_count events, got {}",
        counts.len()
    );
    let last = counts.last().unwrap();
    assert!(
        last.payload["count"].as_u64().unwrap() > 40,
        "{:?}",
        last.payload
    );
    assert_eq!(last.frame_refs[0].stream_id, "lowstate");
    assert_eq!(last.collapse_key.as_deref(), Some("frame_count:lowstate"));
    assert!(events.iter().any(|e| e.kind.as_str() == "obs/test"));
    assert_eq!(
        events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (1..=events.len() as u64).collect::<Vec<_>>()
    );
    assert_eq!(log.sink_errors(), 0);
}

#[tokio::test]
async fn unknown_plugin_id_is_a_boot_error() {
    let err = Kernel::boot(vec![EntryRow::new("does-not-exist")]).unwrap_err();
    assert!(matches!(err, BootError::UnknownPlugin(ref s) if s == "does-not-exist"));
}

#[tokio::test]
async fn disabled_rows_are_skipped() {
    let mut r = vec![EntryRow::new("clock")];
    let mut off = EntryRow::new("frame-store");
    off.disabled = true;
    r.push(off);
    let k = Kernel::boot(r).unwrap();
    assert_eq!(k.states().len(), 1);
}

#[tokio::test]
async fn ingest_op_reaches_frame_store() {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("frame-store"),
        EntryRow::new("mock-stream").with_config(json!({"stream_id": "s", "hz": 1.0})),
    ])
    .unwrap();
    let fs = k
        .registry()
        .get::<oeh_kernel::FrameStore>(&"frame-store".into())
        .unwrap();
    let h = k.handle();
    let run = tokio::spawn(k.run());
    h.ingest("s", bytes::Bytes::from_static(b"external"), Some(42))
        .unwrap();
    // 有界重试而不是一个猜出来的 settle 窗口：Op 一被处理就走，最多等 2 s。固定窗口
    // 在一台忙机器上要么白等，要么不够长——两种结果都不是这条测试想说的话。
    let latest = loop_until(|| fs.latest("s"))
        .await
        .expect("the ingested frame must reach the frame store");
    assert_eq!(latest.t_source, Some(42));
    h.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 测试专用插件：在自己被 dispose 时向世界日志追加一条 plugin/detach。
/// 它排在 world-log 之前注册，所以 dispose_all（逆序）会在 world-log 自身 flush 之后才 dispose 它，
/// 那条 detach 事件只能靠 Kernel::run 在 dispose_all 之后的第二次 flush 落盘。
struct DetachLogger;
impl oeh_kernel::Plugin for DetachLogger {
    fn manifest(&self) -> oeh_proto::Manifest {
        oeh_proto::Manifest::new("detach-logger").inject(&["world-log"])
    }
    fn apply(
        &self,
        ctx: &oeh_kernel::Ctx,
        _c: &serde_json::Value,
    ) -> Result<(), oeh_kernel::PluginError> {
        let log = ctx.get::<WorldLog>("world-log")?;
        ctx.on_dispose(Box::new(move || {
            log.append(EventDraft::new(
                EventKind::new("plugin/detach").unwrap(),
                "detach-logger",
                json!({"plugin": "detach-logger"}),
            ));
        }));
        Ok(())
    }
}

#[tokio::test]
async fn shutdown_flushes_events_appended_during_dispose() {
    let dir = tempfile::tempdir().unwrap();
    let rows = vec![
        EntryRow::new("detach-logger"), // 注册在 world-log 之前 → 逆序 dispose 时晚于 world-log
        EntryRow::new("world-log").with_config(json!({"dir": dir.path(), "segment_mb": 1})),
        EntryRow::new("clock"),
    ];
    let kernel = Kernel::boot_with(rows, &|id| match id {
        "detach-logger" => Some(Box::new(DetachLogger) as Box<dyn oeh_kernel::Plugin>),
        other => oeh_kernel::find_factory(other).map(|f| (f.build)()),
    })
    .unwrap();
    assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
    let log = kernel
        .registry()
        .get::<WorldLog>(&"world-log".into())
        .unwrap();
    let handle = kernel.handle();
    let run = tokio::spawn(kernel.run());
    handle
        .emit(EventDraft::new(
            EventKind::new("obs/test").unwrap(),
            "test",
            json!({}),
        ))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    handle.shutdown().unwrap();
    run.await.unwrap().unwrap();

    let events = read_segments(dir.path()).unwrap();
    assert_eq!(
        events.len() as u64,
        log.next_seq() - 1,
        "every committed seq must be on disk"
    );
    assert_eq!(events.last().unwrap().kind.as_str(), "plugin/detach");
    assert_eq!(
        events.iter().map(|e| e.seq).collect::<Vec<_>>(),
        (1..=events.len() as u64).collect::<Vec<_>>()
    );
}

/// C1: 同一个日志目录连续跑两次。每次 boot 自己的 `run-<boot_unix_ms>/` 子目录，
/// 两次的 seq 都从 1 起，互不追加；默认回放取最新 run，不会报 seq gap。
#[tokio::test]
async fn two_runs_into_the_same_dir_replay_independently() {
    let base = tempfile::tempdir().unwrap();
    for run in 0..2u64 {
        let rows = vec![
            EntryRow::new("world-log").with_config(json!({"dir": base.path(), "segment_mb": 1})),
            EntryRow::new("clock"),
        ];
        let kernel = Kernel::boot(rows).unwrap();
        assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
        let handle = kernel.handle();
        let task = tokio::spawn(kernel.run());
        for i in 0..=run {
            handle
                .emit(EventDraft::new(
                    EventKind::new("obs/test").unwrap(),
                    "test",
                    json!({"run": run, "i": i}),
                ))
                .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        handle.shutdown().unwrap();
        task.await.unwrap().unwrap();
    }

    let mut runs: Vec<_> = std::fs::read_dir(base.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    runs.sort();
    assert_eq!(runs.len(), 2, "one run dir per boot, got {runs:?}");
    assert!(runs.iter().all(|r| r.starts_with("run-")), "{runs:?}");

    // 默认回放 = 最新 run，seq 从 1 连续（先是 boot 时两个 Fiber 各一条 `plugin/state`，
    // 再是这次 run 自己的两条 `obs/test`）
    let newest = read_segments(base.path()).unwrap();
    assert_eq!(
        newest.iter().map(|e| e.seq).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    let newest_obs: Vec<_> = newest
        .iter()
        .filter(|e| e.kind.as_str() == "obs/test")
        .collect();
    assert_eq!(newest_obs.len(), 2, "the two obs/test events from this run");
    assert!(newest_obs.iter().all(|e| e.payload["run"] == 1));
    // 旧 run 仍可单独回放
    let older = read_segments(&base.path().join(&runs[0])).unwrap();
    assert_eq!(
        older.iter().map(|e| e.seq).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let older_obs: Vec<_> = older
        .iter()
        .filter(|e| e.kind.as_str() == "obs/test")
        .collect();
    assert_eq!(older_obs.len(), 1);
    assert_eq!(older_obs[0].payload["run"], 0);
}

/// I7: Op 循环每次都从注册表解析服务。boot 时还没有 world-log 的内核，在运行中拿到
/// world-log 之后必须开始把 Emit 落进去，而不是一直往地上丢。
#[tokio::test]
async fn emit_reaches_a_world_log_that_appears_after_boot() {
    use std::sync::Arc;
    let kernel = Kernel::boot(vec![EntryRow::new("clock")]).unwrap();
    let reg = kernel.registry().clone();
    let handle = kernel.handle();
    let run = tokio::spawn(kernel.run());

    let mk = |n: u64| EventDraft::new(EventKind::new("obs/test").unwrap(), "test", json!({"n": n}));
    // 还没有 world-log：这条只能被丢弃
    handle.emit(mk(0)).unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let log = Arc::new(WorldLog::new(Arc::new(oeh_kernel::MonotonicClock::new())));
    reg.provide(
        "world-log".into(),
        oeh_kernel::FiberId(99),
        log.clone() as Arc<dyn std::any::Any + Send + Sync>,
    )
    .unwrap();
    handle.emit(mk(1)).unwrap();
    // 有界重试而不是一个固定窗口：等的是"那条 Emit 被处理并落进后来的 world-log"，
    // 20 ms 在一台忙机器上不够。
    let got = loop_until(|| {
        let evs = log.since(0).unwrap();
        (!evs.is_empty()).then_some(evs)
    })
    .await
    .unwrap_or_default();
    assert_eq!(got.len(), 1, "later-provided world-log must receive the Op");
    assert_eq!(got[0].payload["n"], 1);
    handle.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 有界重试：每 20 ms 试一次，最多 2 s，第一次拿到 `Some` 就返回。
async fn loop_until<T>(mut f: impl FnMut() -> Option<T>) -> Option<T> {
    for _ in 0..100 {
        if let Some(v) = f() {
            return Some(v);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    f()
}

/// R1：**有序关机必须把会话日志排空**，哪怕写线程正落后——而且不能指望
/// `SessionLog::drop` 去做这件事。
///
/// 这里的关键布景是"测试自己攥着一个 `Arc<Brains>` 和 attach 返回的 `Disposer`"：那正是
/// 真实形状（RPC 服务器攥着同一个 `Arc`，脑的句柄还没归还），于是 `dispose_all` 之后
/// `Inner` 仍然活着，`SessionLog` 一个都不会析构，`Drop` 里那次排空根本不会发生。
/// 关机路径必须自己排空（`Kernel::run` 里 `settle()` 之后、`dispose_all` 之前那次
/// `flush_sessions()`）。
///
/// 写线程用 stall 钩子按住 400 ms 再放开：`run()` 返回时那条记账必须已经在盘上。
/// 少了那次 `flush_sessions()`，此刻文件里什么都没有。
#[tokio::test]
async fn shutdown_flushes_session_logs_even_when_the_writer_is_slow() {
    use oeh_kernel::session_log::{read_session, SessionEntry};
    use oeh_kernel::{BrainSink, Brains, Observation, TurnHandle, WakePolicy};
    struct Nop;
    impl BrainSink for Nop {
        fn on_turn(&self, _o: Observation, _t: TurnHandle) {}
        fn steer(&self, _o: Observation) {}
        fn interrupt(&self) {}
    }

    let dir = tempfile::tempdir().unwrap();
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("brains").with_config(json!({"session_dir": dir.path()})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let brains = k.registry().get::<Brains>(&"brains".into()).unwrap();
    let handle = k.handle();
    let run = tokio::spawn(k.run());

    let _attached = brains
        .attach("fast", WakePolicy::default(), std::sync::Arc::new(Nop))
        .unwrap();
    let path = {
        let session = brains.session_log_for_tests("fast").unwrap();
        session.stall_writer_for_tests(Duration::from_millis(400));
        session
            .record(&SessionEntry::ModelOutput {
                turn: 1,
                step: 0,
                text: "last words".into(),
            })
            .unwrap();
        session.path().to_path_buf()
        // 句柄在这里还回去：brains 里那份才是最后一个强引用。
    };

    handle.shutdown().unwrap();
    run.await.unwrap().unwrap();

    let entries = read_session(&path).unwrap();
    assert!(
        entries
            .iter()
            .any(|e| matches!(e, SessionEntry::ModelOutput { text, .. } if text == "last words")),
        "an orderly shutdown must flush the session log even when the writer is behind: {entries:?}"
    );
}
