use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// 机器人低层状态摘要（适配器从 rt/lowstate 提炼；不含原始关节力矩）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowState {
    pub gravity_proj_z: f64,
    pub rl_policy_active: bool,
    pub mode_machine: u8,
    #[serde(default)]
    pub joint_q: Vec<f64>,
}

impl LowState {
    pub const SCHEMA: &'static str = "oeh.LowState@json";
    pub fn to_bytes(&self) -> Bytes {
        Bytes::from(serde_json::to_vec(self).expect("LowState is always serializable"))
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(b)
    }
}

/// `obs/detections` 事件的 payload：视觉 sidecar 的原始扫描结果，尚未做迟滞。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionsScan {
    #[serde(default)]
    pub nearest_person_m: Option<f64>,
    #[serde(default)]
    pub nearest_obstacle_m: Option<f64>,
    pub clear_path_raw: bool,
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lowstate_bytes_round_trip() {
        let s = LowState {
            gravity_proj_z: -0.97,
            rl_policy_active: true,
            mode_machine: 1,
            joint_q: vec![0.1, -0.2],
        };
        let b = s.to_bytes();
        assert_eq!(LowState::from_bytes(&b).unwrap(), s);
        assert_eq!(LowState::SCHEMA, "oeh.LowState@json");
    }
    #[test]
    fn detections_scan_defaults() {
        let v: DetectionsScan =
            serde_json::from_str(r#"{"clear_path_raw":true,"source":"yolo"}"#).unwrap();
        assert!(v.nearest_person_m.is_none() && v.nearest_obstacle_m.is_none());
    }
}
