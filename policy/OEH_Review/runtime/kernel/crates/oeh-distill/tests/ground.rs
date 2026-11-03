use oeh_distill as _;
use oeh_kernel::profile::EntryRow;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Kernel, WorldLog};
use oeh_proto::payload::DetectionsScan;
use oeh_proto::{EventDraft, EventKind};
use serde_json::json;

fn scan(person: Option<f64>, obstacle: Option<f64>, clear: bool) -> EventDraft {
    let p = DetectionsScan {
        nearest_person_m: person,
        nearest_obstacle_m: obstacle,
        clear_path_raw: clear,
        source: "test".into(),
    };
    EventDraft::new(
        EventKind::new("obs/detections").unwrap(),
        "test",
        serde_json::to_value(p).unwrap(),
    )
}

#[tokio::test]
async fn ground_applies_hysteresis_and_emits_edges_only() {
    let k = Kernel::boot(vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("state-registry"),
        EntryRow::new("distill-ground")
            .with_config(json!({"obstacle_m": 0.6, "person_m": 0.8, "hysteresis_m": 0.1})),
    ])
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    let log = k.registry().get::<WorldLog>(&"world-log".into()).unwrap();
    let state = k.registry().get::<StateRegistry>(&"state".into()).unwrap();
    log.append(scan(Some(2.0), Some(2.0), true)); // far: clear_path true edge
    log.append(scan(Some(0.75), Some(2.0), true)); // person enters (<= 0.8)
    log.append(scan(Some(0.85), Some(2.0), true)); // still near (<= 0.9)
    log.append(scan(Some(0.95), Some(0.5), true)); // person leaves (> 0.9); obstacle enters → clear_path false
    log.append(scan(Some(0.95), Some(0.5), true)); // no change → no events
    let kinds: Vec<String> = log
        .since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.prefix() == "obs" && e.kind.as_str() != "obs/detections")
        .map(|e| format!("{}:{}", e.kind, e.payload))
        .collect();
    assert_eq!(
        kinds,
        vec![
            r#"obs/clear_path:{"clear":true}"#.to_string(),
            r#"obs/person:{"near":true,"nearest_person_m":0.75}"#.to_string(),
            r#"obs/person:{"near":false,"nearest_person_m":0.95}"#.to_string(),
            r#"obs/obstacle:{"near":true,"nearest_obstacle_m":0.5}"#.to_string(),
            r#"obs/clear_path:{"clear":false}"#.to_string(),
        ]
    );
    assert_eq!(state.get("clear_path"), Some(json!(false)));
    assert_eq!(state.get("person_near"), Some(json!(false)));
    let obs_ev = log
        .since(0)
        .unwrap()
        .into_iter()
        .find(|e| e.kind.as_str() == "obs/obstacle")
        .unwrap();
    assert_eq!(
        (obs_ev.salience, obs_ev.collapse_key.as_deref()),
        (180, Some("obstacle"))
    );
}
