//! Inline images are accepted only from the exact observation that woke this turn.
//! No path or external URL loading, and no image bytes in the text prompt.
use std::collections::HashSet;

use oeh_kernel::observation::Observation;
use serde_json::Value;

use crate::model::{ImageInput, Message, ModelError};

pub const CAMERA_EVENT: &str = "obs/robot_rgb";

/// The evaluation lane never renders the kernel's unrestricted state snapshot.
/// Reconstruct the camera event rather than deleting known oracle field names.
pub fn isolate_policy_observation(obs: &Observation) -> Result<Observation, ModelError> {
    if obs.wake_reason.kind != CAMERA_EVENT {
        return Err(ModelError::Protocol(
            "isolated policy requires a camera wake".into(),
        ));
    }
    let source = obs
        .events
        .iter()
        .find(|e| e.kind == CAMERA_EVENT && e.last_seq == obs.wake_reason.seq)
        .ok_or_else(|| ModelError::Protocol("missing exact policy observation".into()))?;
    let p = &source.last;
    let mut payload = serde_json::Map::new();
    // `subgoal` is the brain's own earlier VLA prompt (single-arm profile), never evaluator data.
    for key in [
        "observation_id",
        "task_instruction",
        "subgoal",
        "phase",
        "target",
    ] {
        if let Some(value) = p.get(key) {
            if !value.is_string() {
                return Err(ModelError::Protocol(format!("invalid policy {key}")));
            }
            payload.insert(key.into(), value.clone());
        }
    }
    for key in ["observation_seq", "epoch"] {
        if let Some(value) = p.get(key) {
            if value.as_u64().is_none() {
                return Err(ModelError::Protocol(format!("invalid policy {key}")));
            }
            payload.insert(key.into(), value.clone());
        }
    }
    let mut robot = serde_json::Map::new();
    // Dual-arm (RoboDojo, RoboTwin) and single-arm (LIBERO achieved EEF10 plus the
    // commanded native gripper sign) embodiment profiles. Numeric arrays only.
    for key in [
        "left_ee_pose",
        "right_ee_pose",
        "left_ee_joint_state",
        "right_ee_joint_state",
        "left_gripper_command",
        "right_gripper_command",
        "achieved_eef10",
        "gripper_command",
    ] {
        if let Some(value) = p.get("robot_state").and_then(|s| s.get(key)) {
            let values = value
                .as_array()
                .ok_or_else(|| ModelError::Protocol("invalid robot state".into()))?;
            if !values
                .iter()
                .all(|v| v.as_f64().is_some_and(f64::is_finite))
            {
                return Err(ModelError::Protocol("non-numeric robot state".into()));
            }
            robot.insert(key.into(), value.clone());
        }
    }
    payload.insert("robot_state".into(), Value::Object(robot));
    if let Some(steps) = p
        .get("previous_receipt")
        .and_then(|r| r.get("executed_steps"))
    {
        if steps.as_u64().is_none() {
            return Err(ModelError::Protocol("invalid receipt step count".into()));
        }
        payload.insert(
            "previous_receipt".into(),
            serde_json::json!({"executed_steps": steps}),
        );
    }
    if let Some(escalation) = p.get("escalation") {
        payload.insert("escalation".into(), escalation_note(escalation)?);
    }
    payload.insert(
        "images".into(),
        p.get("images").cloned().unwrap_or(Value::Null),
    );
    let mut result = obs.clone();
    result.state.clear();
    result.events = vec![source.clone()];
    result.events[0].first = Value::Object(payload.clone());
    result.events[0].last = Value::Object(payload);
    result.events[0].first_frame_refs.clear();
    result.events[0].last_frame_refs.clear();
    result.source_seq_range = (obs.wake_reason.seq, obs.wake_reason.seq);
    result.truncated = 0;
    // Validate attachments before accepting the reconstructed observation.
    user_message(&result, String::new())?;
    Ok(result)
}

/// Why a hybrid episode handed this one decision to the slow brain. Only a closed set of
/// reasons and the simulated time pass; anything else in the note is a protocol error.
pub const ESCALATION_REASONS: [&str; 4] = [
    "gate_rejection",
    "stall_no_motion",
    "stall_gripper_oscillation",
    "periodic_review",
];

fn escalation_note(value: &Value) -> Result<Value, ModelError> {
    let invalid = || ModelError::Protocol("invalid escalation note".into());
    let note = value.as_object().ok_or_else(invalid)?;
    if note.len() != 2 {
        return Err(invalid());
    }
    let reason = note
        .get("reason")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let time = note
        .get("sim_time_s")
        .and_then(Value::as_f64)
        .ok_or_else(invalid)?;
    if !ESCALATION_REASONS.contains(&reason) || !time.is_finite() || time < 0.0 {
        return Err(invalid());
    }
    Ok(serde_json::json!({"reason": reason, "sim_time_s": time}))
}

pub fn user_message(obs: &Observation, content: String) -> Result<Message, ModelError> {
    if obs.wake_reason.kind != CAMERA_EVENT {
        return Ok(Message::User { content });
    }
    let event = obs
        .events
        .iter()
        .find(|e| e.kind == CAMERA_EVENT && e.last_seq == obs.wake_reason.seq)
        .ok_or_else(|| ModelError::Protocol("missing exact camera wake observation".into()))?;
    let observation_id = event
        .last
        .get("observation_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    if observation_id.is_empty() || observation_id.len() > 128 {
        return Err(ModelError::Protocol(
            "missing camera observation identity".into(),
        ));
    }
    let images: Vec<ImageInput> =
        serde_json::from_value(event.last.get("images").cloned().unwrap_or(Value::Null))
            .map_err(|_| ModelError::Protocol("invalid camera image descriptors".into()))?;
    if images.is_empty() || images.len() > 3 {
        return Err(ModelError::Protocol(
            "expected one to three camera images".into(),
        ));
    }
    let mut names = HashSet::new();
    for image in &images {
        image.validate()?;
        if !names.insert(&image.camera) {
            return Err(ModelError::Protocol("duplicate camera name".into()));
        }
    }
    Ok(Message::UserMultimodal { content, images })
}

pub fn text_payload(payload: &Value) -> Value {
    let mut value = payload.clone();
    if let Some(object) = value.as_object_mut() {
        if let Some(images) = object.remove("images") {
            let summary: Vec<Value> = images
                .as_array()
                .into_iter()
                .flatten()
                .map(|image| {
                    serde_json::json!({"camera": image.get("camera"),
                    "inline_bytes": image.get("data_url").and_then(Value::as_str).map(str::len)})
                })
                .collect();
            object.insert("image_attachments".into(), Value::Array(summary));
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use oeh_kernel::observation::{CollapsedEvent, WakeReason};
    use oeh_kernel::wake::WakeAction;
    use serde_json::json;

    fn observation() -> Observation {
        let payload = json!({"observation_id":"ep-step-1", "images":[
            {"camera":"head","data_url":"data:image/png;base64,aGVsbG8="}]});
        Observation {
            t_kernel: 1,
            wake_reason: WakeReason {
                kind: CAMERA_EVENT.into(),
                seq: 7,
                action: WakeAction::Wake,
            },
            state: Default::default(),
            source_seq_range: (7, 7),
            truncated: 0,
            events: vec![CollapsedEvent {
                kind: CAMERA_EVENT.into(),
                collapse_key: "".into(),
                first_seq: 7,
                last_seq: 7,
                count: 1,
                salience: 1,
                first: payload.clone(),
                last: payload,
                first_frame_refs: vec![],
                last_frame_refs: vec![],
            }],
        }
    }

    #[test]
    fn exact_wake_image_is_attached_and_not_rendered_as_text() {
        let obs = observation();
        let text = crate::native::prompt::render_observation(&obs);
        assert!(!text.contains("aGVsbG8="));
        assert!(text.contains("ep-step-1"));
        let Message::UserMultimodal { images, .. } = user_message(&obs, text).unwrap() else {
            panic!()
        };
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].camera, "head");
    }

    #[test]
    fn stale_collapsed_camera_event_is_not_substituted() {
        let mut obs = observation();
        obs.events[0].last_seq = 8;
        assert!(user_message(&obs, String::new()).is_err());
    }

    #[test]
    fn external_paths_duplicates_and_unbounded_images_are_rejected() {
        for url in [
            "https://example.com/x.png",
            "file:///secret.png",
            "data:image/png;base64,????",
        ] {
            let mut obs = observation();
            obs.events[0].last["images"][0]["data_url"] = json!(url);
            assert!(user_message(&obs, String::new()).is_err());
        }
        let mut obs = observation();
        let image = obs.events[0].last["images"][0].clone();
        obs.events[0].last["images"] = json!([image.clone(), image]);
        assert!(user_message(&obs, String::new()).is_err());
    }

    #[test]
    fn isolated_policy_drops_state_events_and_nested_judge_fields() {
        let mut obs = observation();
        obs.state.insert("native_success".into(), json!(true));
        obs.events[0].last["native_success"] = json!(true);
        obs.events[0].last["robot_state"] = json!({"left_ee_joint_state":[0.2], "score":1});
        obs.events[0].last["previous_receipt"] = json!({"executed_steps":2,"native_success":true});
        let mut oracle = obs.events[0].clone();
        oracle.kind = "judge/outcome".into();
        oracle.last = json!({"score":1});
        obs.events.push(oracle);
        let projected = isolate_policy_observation(&obs).unwrap();
        let text = crate::native::prompt::render_observation(&projected);
        assert!(!text.contains("native_success"));
        assert!(!text.contains("score"));
        assert!(text.contains("executed_steps"));
        assert!(projected.state.is_empty());
        assert_eq!(projected.events.len(), 1);
        obs.wake_reason.kind = "brain/inject".into();
        assert!(isolate_policy_observation(&obs).is_err());
    }

    #[test]
    fn single_arm_state_and_subgoal_pass_but_stay_numeric_and_textual() {
        let mut obs = observation();
        obs.events[0].last["subgoal"] = json!("lift the bowl");
        obs.events[0].last["robot_state"] = json!({"achieved_eef10":[0.1,0.2,0.3,1,0,0,0,1,0,-1],
            "gripper_command":[-1.0], "native_success":[1.0]});
        let text =
            crate::native::prompt::render_observation(&isolate_policy_observation(&obs).unwrap());
        assert!(text.contains("achieved_eef10") && text.contains("gripper_command"));
        assert!(text.contains("lift the bowl"));
        assert!(!text.contains("native_success"));
        obs.events[0].last["robot_state"] = json!({"achieved_eef10":["success"]});
        assert!(isolate_policy_observation(&obs).is_err());
        obs.events[0].last["robot_state"] = json!({});
        obs.events[0].last["subgoal"] = json!({"reward": 1});
        assert!(isolate_policy_observation(&obs).is_err());
    }

    #[test]
    fn only_a_closed_escalation_note_reaches_the_slow_brain() {
        let mut obs = observation();
        obs.events[0].last["escalation"] = json!({"reason":"stall_no_motion","sim_time_s":12.5});
        let text =
            crate::native::prompt::render_observation(&isolate_policy_observation(&obs).unwrap());
        assert!(text.contains("stall_no_motion") && text.contains("12.5"));
        for bad in [
            json!({"reason":"native_success","sim_time_s":1.0}),
            json!({"reason":"periodic_review","sim_time_s":1.0,"score":1}),
            json!({"reason":"periodic_review","sim_time_s":-1.0}),
            json!("periodic_review"),
        ] {
            obs.events[0].last["escalation"] = bad;
            assert!(isolate_policy_observation(&obs).is_err());
        }
    }
}
