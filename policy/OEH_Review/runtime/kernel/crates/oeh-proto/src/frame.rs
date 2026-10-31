use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::event::FrameRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Small,
    Large,
}

/// 适配器声明的一条流。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamSpec {
    pub stream_id: String,
    pub schema: String,
    pub nominal_hz: f64,
    pub tier: Tier,
}

/// 帧存储里的一帧。不序列化到 JSON（跨进程走 flatbuffers，Plan 3）。
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub stream_id: String,
    pub frame_seq: u64,
    pub t_kernel: u64,
    pub t_source: Option<u64>,
    pub payload: Bytes,
}

impl Frame {
    pub fn frame_ref(&self) -> FrameRef {
        FrameRef {
            stream_id: self.stream_id.clone(),
            frame_seq: self.frame_seq,
            t_kernel: self.t_kernel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_spec_round_trip() {
        let s = StreamSpec {
            stream_id: "lowstate".into(),
            schema: "oeh.LowState".into(),
            nominal_hz: 500.0,
            tier: Tier::Small,
        };
        let j = serde_json::to_string(&s).unwrap();
        assert!(j.contains("\"tier\":\"small\""));
        assert_eq!(serde_json::from_str::<StreamSpec>(&j).unwrap(), s);
    }

    #[test]
    fn frame_ref_copies_identity() {
        let f = Frame {
            stream_id: "a".into(),
            frame_seq: 5,
            t_kernel: 42,
            t_source: None,
            payload: Bytes::from_static(b"xy"),
        };
        assert_eq!(
            f.frame_ref(),
            FrameRef {
                stream_id: "a".into(),
                frame_seq: 5,
                t_kernel: 42
            }
        );
    }
}
