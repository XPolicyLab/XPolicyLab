use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 世界日志里 `<prefix>/<name>` 的合法前缀。
///
/// `capability` 记的是**能力面本身**的变化（`capability/overridden` /
/// `capability/restored`：一个适配器顶替了内置 `say`，或者它走了内核把嗓子接回去）。
/// 它不属于 `act`——`act/*` 是一次提案的流水账，而这两条说的是"下一次提案会由谁执行"，
/// 早于任何一次提案发生；也不属于 `plugin`——那是 Fiber 的状态机，一个 Fiber 可以带零个
/// 能力上下线。
pub const EVENT_PREFIXES: &[&str] = &[
    "obs",
    "act",
    "safety",
    "health",
    "brain",
    "plugin",
    "patch",
    "capability",
];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProtoError {
    #[error(
        "invalid event kind `{0}`: expected <prefix>/<name> with prefix in {EVENT_PREFIXES:?}"
    )]
    InvalidEventKind(String),
}

/// `<prefix>/<name>`，prefix 受 EVENT_PREFIXES 约束。
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct EventKind(String);

impl EventKind {
    pub fn new(s: &str) -> Result<Self, ProtoError> {
        match s.split_once('/') {
            Some((p, n)) if EVENT_PREFIXES.contains(&p) && !n.is_empty() => Ok(Self(s.to_string())),
            _ => Err(ProtoError::InvalidEventKind(s.to_string())),
        }
    }
    pub fn prefix(&self) -> &str {
        self.0.split_once('/').map(|(p, _)| p).unwrap_or("")
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for EventKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        EventKind::new(&s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 指向帧存储里的一帧；原始帧不进日志。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRef {
    pub stream_id: String,
    pub frame_seq: u64,
    pub t_kernel: u64,
}

fn default_true() -> bool {
    true
}

/// 世界日志里的一条事件（seq 与 t_kernel 由内核分配）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldEvent {
    pub seq: u64,
    pub t_kernel: u64,
    pub kind: EventKind,
    pub source: String,
    #[serde(default)]
    pub frame_refs: Vec<FrameRef>,
    #[serde(default)]
    pub salience: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collapse_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    pub payload: Value,
    #[serde(default = "default_true")]
    pub required: bool,
}

/// 尚未分配 seq / t_kernel 的事件。
#[derive(Clone, Debug, PartialEq)]
pub struct EventDraft {
    pub kind: EventKind,
    pub source: String,
    pub frame_refs: Vec<FrameRef>,
    pub salience: u8,
    pub collapse_key: Option<String>,
    pub trace_id: Option<String>,
    pub payload: Value,
    pub required: bool,
}

impl EventDraft {
    pub fn new(kind: EventKind, source: impl Into<String>, payload: Value) -> Self {
        Self {
            kind,
            source: source.into(),
            frame_refs: Vec::new(),
            salience: 0,
            collapse_key: None,
            trace_id: None,
            payload,
            required: true,
        }
    }
    pub fn with_frame_refs(mut self, refs: Vec<FrameRef>) -> Self {
        self.frame_refs = refs;
        self
    }
    pub fn with_salience(mut self, s: u8) -> Self {
        self.salience = s;
        self
    }
    pub fn with_collapse_key(mut self, k: impl Into<String>) -> Self {
        self.collapse_key = Some(k.into());
        self
    }
    pub fn with_trace_id(mut self, t: impl Into<String>) -> Self {
        self.trace_id = Some(t.into());
        self
    }
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
    pub fn into_event(self, seq: u64, t_kernel: u64) -> WorldEvent {
        WorldEvent {
            seq,
            t_kernel,
            kind: self.kind,
            source: self.source,
            frame_refs: self.frame_refs,
            salience: self.salience,
            collapse_key: self.collapse_key,
            trace_id: self.trace_id,
            payload: self.payload,
            required: self.required,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_kind_requires_known_prefix() {
        assert!(EventKind::new("obs/person").is_ok());
        assert!(EventKind::new("act/proposed").is_ok());
        assert_eq!(EventKind::new("obs/person").unwrap().prefix(), "obs");
        assert!(matches!(
            EventKind::new("nope/x"),
            Err(ProtoError::InvalidEventKind(_))
        ));
        assert!(EventKind::new("obs/").is_err());
        assert!(EventKind::new("obs").is_err());
    }

    #[test]
    fn world_event_round_trips_and_defaults_required_true() {
        let json = r#"{"seq":7,"t_kernel":1000,"kind":"health/stale","source":"distill-watchdog","payload":{"stream":"lowstate"}}"#;
        let ev: WorldEvent = serde_json::from_str(json).unwrap();
        assert!(ev.required);
        assert!(ev.frame_refs.is_empty());
        assert_eq!(ev.salience, 0);
        let back = serde_json::to_string(&ev).unwrap();
        let ev2: WorldEvent = serde_json::from_str(&back).unwrap();
        assert_eq!(ev, ev2);
    }

    #[test]
    fn draft_builder_sets_fields() {
        let d = EventDraft::new(
            EventKind::new("obs/person").unwrap(),
            "yolo",
            serde_json::json!({"d": 0.9}),
        )
        .with_salience(200)
        .with_collapse_key("person:0")
        .with_frame_refs(vec![FrameRef {
            stream_id: "head_rgb".into(),
            frame_seq: 3,
            t_kernel: 99,
        }])
        .optional();
        assert_eq!(d.salience, 200);
        assert_eq!(d.collapse_key.as_deref(), Some("person:0"));
        assert_eq!(d.frame_refs.len(), 1);
        assert!(!d.required);
    }
}
