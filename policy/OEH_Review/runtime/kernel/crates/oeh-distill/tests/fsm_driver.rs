use oeh_distill as _;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::{FrameStore, Kernel, WorldLog};
use oeh_proto::payload::LowState;
use oeh_proto::{EventDraft, EventKind, StreamSpec, Tier};
use serde_json::json;
use std::time::Duration;

fn boot() -> Kernel {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("frame-store"),
        EntryRow::new("robot-fsm"),
        EntryRow::new("distill-fsm").with_config(json!({"stream_id": "lowstate"})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    k
}
fn distiller_errors(log: &WorldLog) -> Vec<oeh_proto::WorldEvent> {
    log.since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "health/distiller_error")
        .collect()
}
fn stale(log: &WorldLog, stale: bool) {
    log.append(EventDraft::new(
        EventKind::new("health/stale").unwrap(),
        "test",
        json!({"stream": "lowstate", "age_s": 1.0, "stale": stale}),
    ));
}
fn ls(g: f64, active: bool) -> bytes::Bytes {
    LowState {
        gravity_proj_z: g,
        rl_policy_active: active,
        mode_machine: 1,
        joint_q: vec![],
    }
    .to_bytes()
}

#[tokio::test]
async fn boot_to_standing_then_tipping_to_estop() {
    let k = boot();
    let fs = k
        .registry()
        .get::<FrameStore>(&"frame-store".into())
        .unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    let _r = fs
        .register(StreamSpec {
            stream_id: "lowstate".into(),
            schema: LowState::SCHEMA.into(),
            nominal_hz: 500.0,
            tier: Tier::Small,
        })
        .unwrap();
    fs.ingest("lowstate", ls(-0.98, false), None).unwrap();
    assert_eq!(
        fsm.state(),
        FsmState::Boot,
        "policy inactive → stay in BOOT"
    );
    fs.ingest("lowstate", ls(-0.98, true), None).unwrap();
    assert_eq!(fsm.state(), FsmState::Standing);
    fs.ingest("lowstate", ls(-0.3, true), None).unwrap();
    assert_eq!(fsm.state(), FsmState::EmergencyStop);
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    fs.ingest("lowstate", b"not json".as_slice().into(), None)
        .unwrap();
    assert_eq!(
        fsm.state(),
        FsmState::EmergencyStop,
        "garbage frame is ignored"
    );
    // I6: 第一条解析失败必须留痕（§11：蒸馏器异常 → 记 health/distiller_error，流继续）。
    let errs = distiller_errors(&log);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(errs[0].payload["plugin"], json!("distill-fsm"));
    assert_eq!(errs[0].payload["stream"], json!("lowstate"));
    assert_eq!(errs[0].payload["count"], json!(1));
    assert_eq!(errs[0].salience, 200);
    assert_eq!(
        errs[0].collapse_key.as_deref(),
        Some("distiller_error:distill-fsm")
    );
    // 之后按每 100 条采样，不再逐条刷屏
    for _ in 0..10 {
        fs.ingest("lowstate", b"not json".as_slice().into(), None)
            .unwrap();
    }
    assert_eq!(distiller_errors(&log).len(), 1, "sampled, not per-frame");
}

/// I6: 流的 schema 与 `LowState` 对不上时必须 fail closed —— 拿别的 schema 的字节去猜
/// `gravity_proj_z` 会让安全 FSM 基于垃圾做判断。
#[tokio::test]
async fn schema_mismatch_is_reported_once_and_fails_closed() {
    let k = boot();
    let fs = k
        .registry()
        .get::<FrameStore>(&"frame-store".into())
        .unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let _r = fs
        .register(StreamSpec {
            stream_id: "lowstate".into(),
            schema: "other@json".into(),
            nominal_hz: 500.0,
            tier: Tier::Small,
        })
        .unwrap();
    for _ in 0..5 {
        // 字节本身是合法的 LowState，唯一的问题是流声明的 schema 不是它。
        fs.ingest("lowstate", ls(-0.98, true), None).unwrap();
    }
    assert_eq!(fsm.state(), FsmState::Boot, "fail closed: no transition");
    let errs = distiller_errors(&log);
    assert_eq!(errs.len(), 1, "exactly one report: {errs:?}");
    assert_eq!(errs[0].payload["reason"], json!("schema mismatch"));
    assert_eq!(errs[0].payload["expected"], json!(LowState::SCHEMA));
    assert_eq!(errs[0].payload["found"], json!("other@json"));
    assert_eq!(errs[0].salience, 200);
    assert_eq!(
        errs[0].collapse_key.as_deref(),
        Some("distiller_error:distill-fsm")
    );
}

#[tokio::test]
async fn stale_lowstate_faults() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    stale(&log, true);
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert_eq!(fsm.state(), FsmState::Fault);
}

/// I4: 之前谁也出不了 FAULT，机器人就此卡死。distill-fsm 自己发的 FAULT 由它自己收回：
/// 流重新变新鲜 → FAULT → RECOVERING → STANDING。EMERGENCY_STOP 仍然只能由操作员解除。
#[tokio::test]
async fn fault_recovers_when_stream_is_fresh_again() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    stale(&log, true);
    assert_eq!(fsm.state(), FsmState::Fault);
    stale(&log, false);
    assert_eq!(fsm.state(), FsmState::Standing, "auto-recovered");
    let chain: Vec<(String, String, String)> = log
        .since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == "safety/fsm")
        .map(|e| {
            (
                e.payload["from"].as_str().unwrap().to_string(),
                e.payload["to"].as_str().unwrap().to_string(),
                e.payload["reason"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(
        chain,
        vec![
            ("BOOT".into(), "STANDING".into(), "test".into()),
            ("STANDING".into(), "FAULT".into(), "lowstate stale".into()),
            (
                "FAULT".into(),
                "RECOVERING".into(),
                "lowstate fresh again".into()
            ),
            (
                "RECOVERING".into(),
                "STANDING".into(),
                "lowstate fresh again".into()
            ),
        ]
    );
    // fresh 事件再来一次不产生任何迁移（已经不在 FAULT 了）
    stale(&log, false);
    assert_eq!(fsm.state(), FsmState::Standing);
    // EMERGENCY_STOP 不自动恢复：那是 Plan 2b 的 admission/操作员的事
    fsm.transition(FsmState::EmergencyStop, "tipping", "test")
        .unwrap();
    stale(&log, false);
    assert_eq!(
        fsm.state(),
        FsmState::EmergencyStop,
        "estop is operator-only"
    );
}

/// 自动恢复只适用于**自己**驱动出来的 FAULT：别的插件/操作员把 FSM 按进 FAULT 时，
/// 一条 `health/stale{stale:false}` 不能把它捞出来——那条边不归 distill-fsm 管。
#[tokio::test]
async fn only_self_raised_fault_is_auto_recovered() {
    let k = boot();
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let fsm = k.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    fsm.transition(FsmState::Standing, "t", "t").unwrap();
    fsm.transition(FsmState::Fault, "someone else", "test")
        .unwrap();
    log.append(EventDraft::new(
        EventKind::new("health/stale").unwrap(),
        "test",
        json!({"stream": "lowstate", "age_s": 0.0, "stale": false}),
    ));
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert_eq!(
        fsm.state(),
        FsmState::Fault,
        "foreign Fault is not auto-recovered"
    );
}
