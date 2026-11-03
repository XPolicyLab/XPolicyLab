use oeh_distill as _;
use oeh_distill::sim_ready::{ID, KIND};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::{Kernel, WorldLog};
use oeh_proto::{EventDraft, EventKind};
use serde_json::{json, Value};

fn boot() -> Kernel {
    let mut row = EntryRow::new(ID);
    row.config = json!({"benchmark": "robotwin"});
    let kernel = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("robot-fsm"),
        row,
    ])
    .unwrap();
    assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
    kernel
}

fn emit(log: &WorldLog, payload: Value) {
    log.append(EventDraft::new(
        EventKind::new(KIND).unwrap(),
        "test",
        payload,
    ));
}

#[tokio::test]
async fn requires_reset_policy_connection_and_matching_benchmark() {
    let kernel = boot();
    let log = kernel
        .registry()
        .get::<WorldLog>(&"world-log".into())
        .unwrap();
    let fsm = kernel.registry().get::<RobotFsm>(&"fsm".into()).unwrap();
    let valid = json!({
        "benchmark": "robotwin", "episode_id": "one", "observation_sha256": "a".repeat(64),
        "native_reset_observed": true, "policy_connected": true,
    });
    for replacement in [
        json!({"benchmark": "libero"}),
        json!({"native_reset_observed": false}),
        json!({"policy_connected": false}),
        json!({"observation_sha256": "bad"}),
    ] {
        let mut invalid = valid.clone();
        invalid
            .as_object_mut()
            .unwrap()
            .extend(replacement.as_object().unwrap().clone());
        emit(&log, invalid);
        assert_eq!(fsm.state(), FsmState::Boot);
    }
    emit(&log, valid);
    assert_eq!(fsm.state(), FsmState::Standing);
}
