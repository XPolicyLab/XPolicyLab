use oeh_admission as _;
use oeh_distill as _;
use oeh_kernel::profile::load_profile;
use oeh_kernel::session_log::{read_session, SessionEntry};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::world_log::{read_segments, resolve_log_dir};
use oeh_kernel::Kernel;
use oeh_proto::{EventDraft, EventKind, ServiceName};
use serde_json::json;
use std::time::Duration;

/// dev-mock 整条链跑起来：合成流 → 蒸馏器 → 世界日志/状态寄存器 → mock 脑被唤醒 → 提案。
///
/// **唤醒由测试自己给**（`obs/gesture`，`wake/fast.yaml` 里是 `wake`），不指望开机那条
/// `safety/fsm{STANDING}`。理由和 `oeh-admission` 的 `plan2b_e2e` 一样、也是同一个race：
/// `mock-brain` 是 bundle 的最后一行，而它之前的行（500 Hz 的 `mock-stream` +
/// `distill-fsm`）在它 attach 之前就已经在产事件了。`brain/attach` 把游标定在"此刻最后
/// 一条"，所以那条 `safety/fsm` 只要抢在前面落进日志，这个脑就永远等不到它**唯一**的
/// 唤醒证据（`distill-fsm` 只在边沿上发，不会再有第二条），于是 `brain/turn_start` 那条
/// 断言在机器忙的时候随机变红。测试自己发一条就跟那场竞赛无关了：`safety/fsm` 仍然被
/// 断言出现过，只是不再兼任唯一的闹钟。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dev_mock_profile_wakes_mock_brain_on_fsm_transition() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let tmp = tempfile::tempdir().unwrap();
    let patch = tmp.path().join("patch.yaml");
    std::fs::write(
        &patch,
        format!(
            "- {{ id: world-log, config: {{ dir: {0}/logs }} }}\n- {{ id: brains, config: {{ session_dir: {0}/sessions }} }}\n",
            tmp.path().display()
        ),
    )
    .unwrap();
    // mock-brain 的 `wake` 是相对进程 cwd 的路径；profile/bundles 也是。
    std::env::set_current_dir(root).unwrap();
    let rows = load_profile(
        std::path::Path::new("profiles/dev-mock.yaml"),
        std::path::Path::new("bundles"),
        &[patch],
    )
    .unwrap();
    let kernel = Kernel::boot(rows).unwrap();
    assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
    assert!(kernel.report().pending.is_empty(), "{:?}", kernel.report());
    let state = kernel
        .registry()
        .get::<StateRegistry>(&ServiceName::from("state"))
        .unwrap();
    let h = kernel.handle();
    let run = tokio::spawn(kernel.run());
    // 等**条件**，不是等一段猜出来的时间：`distill-fsm` 把 BOOT 推到 STANDING（500 Hz
    // 的流，第一帧 2 ms 就到）。这条唤醒要发在它后面，第一份观测里的 `fsm` 才确定是
    // STANDING 而不是 BOOT。
    let mut standing = false;
    for _ in 0..200 {
        if state.get("fsm") == Some(json!("STANDING")) {
            standing = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(standing, "distill-fsm 没能把 FSM 推到 STANDING");
    // `run()` 已经在跑 = boot 早就结束 = `mock-brain` 已经 attach 过了，所以这条事件
    // 必然落在它的游标之后。
    h.emit(EventDraft::new(
        EventKind::new("obs/gesture").unwrap(),
        "test",
        json!({"gesture": "wave"}),
    ))
    .unwrap();
    tokio::time::sleep(Duration::from_millis(400)).await;
    h.shutdown().unwrap();
    run.await.unwrap().unwrap();
    let events = read_segments(&tmp.path().join("logs")).unwrap();
    let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
    assert!(
        kinds.contains(&"safety/fsm"),
        "distill-fsm moved BOOT→STANDING: {kinds:?}"
    );
    assert!(
        kinds.contains(&"health/stale"),
        "watchdog reported: {kinds:?}"
    );
    assert!(
        kinds.contains(&"brain/turn_start"),
        "mock brain woke: {kinds:?}"
    );
    assert!(kinds.contains(&"brain/mock_turn"));
    // I5: 会话日志跟着世界日志的 run 目录走（`brains.session_dir` 只在 world-log 没有
    // 持久化目录时才用得上），两次运行不会写进同一个文件。
    let run_dir = resolve_log_dir(&tmp.path().join("logs")).unwrap();
    assert!(
        run_dir
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("run-"),
        "{run_dir:?}"
    );
    assert!(
        !tmp.path().join("sessions").exists(),
        "session_dir is unused once world-log has a run dir"
    );
    let session = read_session(&run_dir.join("sessions/fast.session.jsonl")).unwrap();
    assert!(
        matches!(&session[0], SessionEntry::Attached { brain_id, .. } if brain_id == "fast"),
        "{:?}",
        session.first()
    );
    let obs = session
        .iter()
        .find_map(|e| match e {
            SessionEntry::Observation { observation, .. } => Some(observation),
            _ => None,
        })
        .unwrap();
    assert_eq!(obs.state["fsm"], serde_json::json!("STANDING"));
    // 唤醒发在 FSM 到 STANDING 之后，所以 `distill-fsm` 写的那几个键此刻必然在寄存器里。
    assert!(
        obs.state.contains_key("gravity_proj_z"),
        "distill-fsm state is visible"
    );
    let (lo, hi) = obs.source_seq_range;
    assert!(
        events.iter().any(|e| e.seq == lo) && events.iter().any(|e| e.seq == hi),
        "observation is reconstructable from the world log"
    );
}
