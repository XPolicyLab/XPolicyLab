use oeh_kernel::state::StateRegistry;
use oeh_kernel::ManualClock;
use serde_json::json;
use std::sync::Arc;

#[test]
fn set_get_snapshot_and_age() {
    let clock = Arc::new(ManualClock::new(100));
    let s = StateRegistry::new(clock.clone());
    s.set("clear_path", json!(true), "distill-ground");
    clock.advance(50);
    s.set("fsm", json!("STANDING"), "robot-fsm");
    assert_eq!(s.get("clear_path"), Some(json!(true)));
    let snap = s.snapshot();
    assert_eq!(
        snap.keys().cloned().collect::<Vec<_>>(),
        vec!["clear_path", "fsm"]
    );
    let e = s.entry("clear_path").unwrap();
    assert_eq!((e.t_kernel, e.source.as_str()), (100, "distill-ground"));
    assert_eq!(s.age_ns("clear_path", 175), Some(75));
    assert_eq!(s.age_ns("nope", 175), None);
}

#[test]
fn builtin_provides_state_service() {
    use oeh_kernel::fiber::FiberTree;
    use oeh_kernel::profile::EntryRow;
    use oeh_kernel::{find_factory, ServiceRegistry};
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg.clone());
    tree.add(
        EntryRow::new("state-registry"),
        (find_factory("state-registry").unwrap().build)(),
    );
    tree.add(
        EntryRow::new("clock"),
        (find_factory("clock").unwrap().build)(),
    );
    let r = tree.settle();
    assert!(r.failed.is_empty(), "{r:?}");
    assert!(reg.get::<StateRegistry>(&"state".into()).is_ok());
}

/// I8: 一组相关寄存器必须一次锁内写完——观测读的是一次快照，跨键撕裂会让脑看到
/// 自相矛盾的状态。
#[test]
fn set_many_is_atomic_across_keys_and_shares_one_timestamp() {
    let clock = Arc::new(ManualClock::new(10));
    let s = StateRegistry::new(clock.clone());
    s.set("old", json!(1), "x");
    clock.advance(5);
    s.set_many(
        &[
            ("person_near", json!(true)),
            ("nearest_person_m", json!(0.4)),
        ],
        "distill-ground",
    );
    assert_eq!(s.get("person_near"), Some(json!(true)));
    assert_eq!(s.get("nearest_person_m"), Some(json!(0.4)));
    let entries = s.snapshot_entries();
    assert_eq!(
        entries.keys().cloned().collect::<Vec<_>>(),
        vec!["nearest_person_m", "old", "person_near"]
    );
    assert_eq!(entries["person_near"].t_kernel, 15);
    assert_eq!(
        entries["nearest_person_m"].t_kernel, entries["person_near"].t_kernel,
        "one timestamp for the whole group"
    );
    assert_eq!(entries["person_near"].source, "distill-ground");
    assert_eq!(
        entries["old"].t_kernel, 10,
        "untouched keys keep their own time"
    );
}
