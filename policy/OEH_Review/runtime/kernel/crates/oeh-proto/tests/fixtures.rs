use oeh_proto::{Manifest, StreamSpec, WorldEvent};

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)).unwrap()
}

#[test]
fn world_event_fixture_round_trips() {
    let ev: WorldEvent = serde_json::from_str(&fixture("world_event.json")).unwrap();
    assert_eq!(ev.kind.as_str(), "obs/person");
    let again: WorldEvent = serde_json::from_str(&serde_json::to_string(&ev).unwrap()).unwrap();
    assert_eq!(ev, again);
}

#[test]
fn stream_spec_and_manifest_fixtures_parse() {
    let s: StreamSpec = serde_json::from_str(&fixture("stream_spec.json")).unwrap();
    assert_eq!(s.nominal_hz, 500.0);
    let m: Manifest = serde_json::from_str(&fixture("manifest.json")).unwrap();
    assert_eq!(m.streams.len(), 2);
    assert_eq!(m.provide[0].0, "adapter.unitree");
}
