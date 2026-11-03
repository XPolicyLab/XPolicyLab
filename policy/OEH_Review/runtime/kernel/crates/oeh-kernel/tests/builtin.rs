use std::sync::Arc;

use oeh_kernel::factories::{factory_ids, find_factory};
use oeh_kernel::fiber::{FiberState, FiberTree};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::{ClockService, FrameStore, ServiceRegistry, WorldLog};
use serde_json::json;

#[test]
fn builtin_factories_are_registered() {
    let ids = factory_ids();
    for id in ["clock", "frame-store", "world-log"] {
        assert!(ids.contains(&id), "missing factory {id}; have {ids:?}");
        assert!(find_factory(id).is_some());
    }
    assert!(find_factory("nope").is_none());
}

#[test]
fn builtins_activate_and_provide_services() {
    let dir = tempfile::tempdir().unwrap();
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg.clone());
    // 故意把依赖 clock 的放前面
    tree.add(
        EntryRow::new("world-log").with_config(json!({"dir": dir.path(), "segment_mb": 1})),
        (find_factory("world-log").unwrap().build)(),
    );
    tree.add(
        EntryRow::new("frame-store").with_config(json!({"window_s": 0.5})),
        (find_factory("frame-store").unwrap().build)(),
    );
    tree.add(
        EntryRow::new("clock"),
        (find_factory("clock").unwrap().build)(),
    );
    let r = tree.settle();
    assert!(r.failed.is_empty(), "{r:?}");
    assert_eq!(tree.state_of("world-log"), Some(FiberState::Active));
    let clock = reg.get::<ClockService>(&"clock".into()).unwrap();
    let _ = clock.0.now_ns();
    let fs = reg.get::<FrameStore>(&"frame-store".into()).unwrap();
    assert!(fs.streams().is_empty());
    let log = reg.get::<WorldLog>(&"world-log".into()).unwrap();
    assert_eq!(log.next_seq(), 1);
    tree.dispose_all();
    assert!(reg.names().is_empty());
}

#[test]
fn world_log_without_dir_is_memory_only() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg.clone());
    tree.add(
        EntryRow::new("clock"),
        (find_factory("clock").unwrap().build)(),
    );
    tree.add(
        EntryRow::new("world-log"),
        (find_factory("world-log").unwrap().build)(),
    );
    let r = tree.settle();
    assert!(r.failed.is_empty(), "{r:?}");
}

#[test]
fn bad_config_fails_closed() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg.clone());
    tree.add(
        EntryRow::new("clock"),
        (find_factory("clock").unwrap().build)(),
    );
    tree.add(
        EntryRow::new("frame-store").with_config(json!({"window_s": "two"})),
        (find_factory("frame-store").unwrap().build)(),
    );
    let r = tree.settle();
    assert_eq!(tree.state_of("frame-store"), Some(FiberState::Failed));
    assert!(r.failed[0].1.contains("frame-store"));
    assert!(r.failed[0].1.contains("expected f64"));
    assert!(!reg.available(&"frame-store".into()));
}

/// I4: 坏 hz 必须在 apply 里 fail-closed，而不是等 `Duration::from_secs_f64` 在
/// spawn_interval 里 panic。NaN / inf 在 JSON 里表示不出来（serde_json 把它们写成 null），
/// 所以走配置的坏值只能是 null 或 <= 0；两条路径都必须 Failed 且不留下半注册的流。
#[test]
fn mock_stream_rejects_bad_hz() {
    for (hz, want) in [
        (json!(0.0), "hz must be finite, > 0 and >= 1/3600"),
        (json!(-1.0), "hz must be finite, > 0 and >= 1/3600"),
        (json!(f64::NAN), "expected f64"), // json! 把非有限浮点写成 null
    ] {
        let reg = Arc::new(ServiceRegistry::default());
        let mut tree = FiberTree::new(reg.clone());
        tree.add(
            EntryRow::new("clock"),
            (find_factory("clock").unwrap().build)(),
        );
        tree.add(
            EntryRow::new("frame-store"),
            (find_factory("frame-store").unwrap().build)(),
        );
        tree.add(
            EntryRow::new("mock-stream").with_config(json!({"stream_id": "s", "hz": hz})),
            (find_factory("mock-stream").unwrap().build)(),
        );
        let r = tree.settle();
        assert_eq!(
            tree.state_of("mock-stream"),
            Some(FiberState::Failed),
            "hz={hz} must fail closed"
        );
        let msg = &r.failed.iter().find(|(k, _)| k == "mock-stream").unwrap().1;
        assert!(msg.contains(want), "hz={hz}: {msg}");
        let fs = reg.get::<FrameStore>(&"frame-store".into()).unwrap();
        assert!(fs.streams().is_empty(), "no stream left behind");
    }
}

#[tokio::test]
async fn mock_stream_lowstate_mode_emits_typed_frames() {
    use oeh_kernel::{FrameStore, Kernel};
    use oeh_proto::payload::LowState;
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("frame-store"),
        EntryRow::new("mock-stream").with_config(json!({"stream_id": "lowstate", "hz": 200.0, "payload": "lowstate", "lowstate": {"gravity_proj_z": -0.5}})),
    ]).unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let fs = k
        .registry()
        .get::<FrameStore>(&"frame-store".into())
        .unwrap();
    // 有界重试（每 20 ms，最多 2 s）而不是"睡够 12 个 tick"：等的是第一帧真的到，
    // 不是一段猜出来的时间。
    let mut got = None;
    for _ in 0..100 {
        got = fs.latest("lowstate");
        if got.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let f = got.expect("mock-stream must produce a frame");
    let ls = LowState::from_bytes(&f.payload).unwrap();
    assert_eq!(ls.gravity_proj_z, -0.5);
    assert_eq!(fs.streams()[0].schema, LowState::SCHEMA);
}

/// 用户在 `payload: lowstate` 下显式配了别的 `schema`：行为上仍然被强制为
/// `LowState::SCHEMA`（下游按 schema 分流解码，配置出的 payload/schema 不一致会直接
/// 破坏解码）；覆盖动作本身只通过 tracing::warn! 可观测，这里不断言日志，只断言最终生效
/// 的 schema 没有被用户的配置带偏。
#[tokio::test]
async fn mock_stream_lowstate_mode_overrides_configured_schema() {
    use oeh_kernel::{FrameStore, Kernel};
    use oeh_proto::payload::LowState;
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("frame-store"),
        EntryRow::new("mock-stream").with_config(json!({
            "stream_id": "lowstate2",
            "hz": 200.0,
            "schema": "custom@x",
            "payload": "lowstate",
        })),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let fs = k
        .registry()
        .get::<FrameStore>(&"frame-store".into())
        .unwrap();
    assert_eq!(fs.streams()[0].schema, LowState::SCHEMA);
}

/// I5: world-log 没有持久化目录时，会话日志自己造一个 `run-<boot_unix_ms>` 子目录——
/// 两次运行的 turn 号都从 1 起，绝不能写进同一个文件。
#[tokio::test]
async fn sessions_land_in_a_run_dir_even_without_a_world_log_dir() {
    use oeh_kernel::{BrainSink, Brains, Observation, TurnHandle, WakePolicy};
    struct Nop;
    impl BrainSink for Nop {
        fn on_turn(&self, _o: Observation, _t: TurnHandle) {}
        fn steer(&self, _o: Observation) {}
        fn interrupt(&self) {}
    }
    let dir = tempfile::tempdir().unwrap();
    let k = oeh_kernel::Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"), // 无 dir → 无 run_dir
        EntryRow::new("state-registry"),
        EntryRow::new("brains").with_config(json!({"session_dir": dir.path()})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let brains = k.registry().get::<Brains>(&"brains".into()).unwrap();
    let _d = brains
        .attach("fast", WakePolicy::default(), Arc::new(Nop))
        .unwrap();
    let runs: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(runs.len(), 1, "{runs:?}");
    assert!(runs[0].starts_with("run-"), "{runs:?}");
    assert!(dir
        .path()
        .join(&runs[0])
        .join("fast.session.jsonl")
        .exists());
}
