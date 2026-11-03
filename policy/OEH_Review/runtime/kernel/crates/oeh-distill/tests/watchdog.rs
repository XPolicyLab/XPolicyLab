use oeh_distill as _;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Kernel, WorldLog};
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn watchdog_reports_fresh_then_stale() {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("frame-store"),
        EntryRow::new("mock-stream").with_config(json!({"stream_id": "lowstate", "hz": 100.0})),
        EntryRow::new("distill-watchdog")
            .with_config(json!({"streams": {"lowstate": 0.1, "head_rgb": 0.5}, "period_ms": 20})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let h = k.handle();
    let run = tokio::spawn(k.run());
    tokio::time::sleep(Duration::from_millis(120)).await;
    assert_eq!(state.get("fresh.lowstate"), Some(json!(true)));
    assert_eq!(
        state.get("fresh.head_rgb"),
        Some(json!(false)),
        "never-seen stream is stale"
    );
    assert_eq!(
        state.get("last_frame_t_kernel.head_rgb"),
        Some(json!(null)),
        "never-seen stream has no frame time"
    );
    // 停掉流：直接 dispose 不方便，改为等 mock 流的帧年龄超过阈值不可行；用短阈值 + 长等待验证 stale 边沿
    h.shutdown().unwrap();
    run.await.unwrap().unwrap();
    let stale: Vec<_> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "health/stale")
        .collect();
    let lowstate_first = stale
        .iter()
        .find(|e| e.payload["stream"] == "lowstate")
        .unwrap();
    assert_eq!(
        lowstate_first.payload["stale"],
        json!(false),
        "first lowstate edge is fresh"
    );
    assert!(stale
        .iter()
        .any(|e| e.payload["stream"] == "head_rgb" && e.payload["stale"] == json!(true)));
    assert_eq!(
        lowstate_first.collapse_key.as_deref(),
        Some("stale:lowstate")
    );
    assert_eq!(lowstate_first.salience, 220);
}

#[tokio::test]
async fn watchdog_goes_stale_when_frames_stop() {
    use oeh_kernel::FrameStore;
    use oeh_proto::{StreamSpec, Tier};
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("frame-store"),
        EntryRow::new("distill-watchdog")
            .with_config(json!({"streams": {"s": 0.05}, "period_ms": 10})),
    ])
    .unwrap();
    let fs = k
        .registry()
        .get::<FrameStore>(&"frame-store".into())
        .unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    let _reg = fs
        .register(StreamSpec {
            stream_id: "s".into(),
            schema: "x".into(),
            nominal_hz: 10.0,
            tier: Tier::Small,
        })
        .unwrap();
    fs.ingest("s", bytes::Bytes::new(), None).unwrap();
    tokio::time::sleep(Duration::from_millis(25)).await;
    assert_eq!(state.get("fresh.s"), Some(json!(true)));
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert_eq!(state.get("fresh.s"), Some(json!(false)));
    let age = state.get("age_s.s").unwrap().as_f64().unwrap();
    assert!(age > 0.05, "age_s should exceed threshold, got {age}");
    // I8: `age_s` 是相对采样时刻的派生量；原始帧时间必须一起发出来。
    let last = state
        .get("last_frame_t_kernel.s")
        .unwrap()
        .as_u64()
        .expect("u64 once a frame has been seen");
    assert!(last > 0);
    let entries = state.snapshot_entries();
    assert_eq!(
        entries["age_s.s"].t_kernel, entries["last_frame_t_kernel.s"].t_kernel,
        "one atomic write per stream"
    );
}
