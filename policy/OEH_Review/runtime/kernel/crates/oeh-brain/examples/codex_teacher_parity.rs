//! 线上一致性测试的 Rust 一侧：照一份场景 JSON 驱动 [`CodexClient`]，把每一步的结果打到 stdout。
//!
//! 由 `python/oeh-policy/tests/test_codex_teacher_parity.py` 构建并调用；Python 一侧用同一份场景
//! 驱动 `codex_teacher.CodexClient`，两边对着同一个 `fake_codex_app_server.py`。
//!
//! 场景：`{"cfg", "workdir", "prompt", "accounts", "argv_prefix", "output_schema", "episode",
//! "launch": {"attempts", "timeout_s", "backoff_s"}?, "ops": [...]}`；op 是
//! `{"op": "preflight"}`、`{"op": "packet", "text", "images", "intro"}`、`{"op": "reask", "text"}`、
//! `{"op": "create"}`（成功时把输出记进对话，同 TeacherBrain）。

use std::time::Duration;

use oeh_brain::teacher::{
    CodexClient, CodexConversation, LaunchPolicy, TeacherConfig, TeacherError,
};
use serde_json::{json, Value};

fn raised(error: TeacherError) -> Value {
    match error {
        TeacherError::Fatal { message, errors } => {
            json!({"class": "TeacherFatal", "message": message, "errors": errors})
        }
        TeacherError::Unavailable { message, errors } => {
            json!({"class": "TeacherUnavailable", "message": message, "errors": errors})
        }
        TeacherError::Unexpected(message) => json!({"class": "Unexpected", "message": message}),
    }
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: codex_teacher_parity <scenario.json>");
    let scenario: Value = serde_json::from_slice(&std::fs::read(&path).expect("read scenario"))
        .expect("scenario JSON");
    let cfg = TeacherConfig::from_json(&scenario["cfg"]).expect("teacher config");
    let mut client = CodexClient::new(
        cfg.clone(),
        scenario["workdir"].as_str().expect("workdir"),
        scenario["prompt"].as_str().expect("prompt"),
        strings(&scenario["accounts"]),
        scenario.get("argv_prefix").map(strings),
    )
    .expect("client");
    if let Some(launch) = scenario.get("launch") {
        let secs = |v: &Value| Duration::from_secs_f64(v.as_f64().expect("seconds"));
        client = client.with_launch_policy(LaunchPolicy {
            attempts: launch["attempts"].as_u64().expect("attempts") as u32,
            timeout: secs(&launch["timeout_s"]),
            backoff: launch["backoff_s"]
                .as_array()
                .expect("backoff_s")
                .iter()
                .map(secs)
                .collect(),
        });
    }
    let mut conversation = CodexConversation::new(
        &cfg,
        scenario["episode"].as_str().expect("episode"),
        scenario["output_schema"].clone(),
    );
    let mut results = Vec::new();
    for op in scenario["ops"].as_array().expect("ops") {
        let kind = op["op"].as_str().expect("op");
        let result = match kind {
            "preflight" => match client.preflight() {
                Ok(checked) => json!({"op": kind, "ok": checked}),
                Err(e) => json!({"op": kind, "raised": raised(e)}),
            },
            "packet" => {
                let images = op["images"].as_array().cloned().unwrap_or_default();
                conversation.add_packet(
                    op["text"].as_str().expect("text"),
                    &images,
                    op["intro"].as_bool().unwrap_or(false),
                );
                json!({"op": kind})
            }
            "reask" => {
                conversation.add_error(op["text"].as_str().expect("text"));
                json!({"op": kind})
            }
            "create" => match client.create(&conversation.body()) {
                Ok((data, errors)) => {
                    let output = data["output"].as_array().cloned().unwrap_or_default();
                    let call = conversation.add_output(&output);
                    json!({"op": kind, "ok": data, "errors": errors, "call": call})
                }
                Err(e) => json!({"op": kind, "raised": raised(e)}),
            },
            other => panic!("unknown op {other}"),
        };
        results.push(result);
    }
    let summary = json!({
        "launches": client.launches, "account": client.account, "items": conversation.items,
        "server_left": client.has_server(),
    });
    client.close();
    println!("{}", json!({"results": results, "summary": summary}));
}
