//! `oeh_brain::teacher::CodexClient` 对着 Python 测试用的那个替身 app-server
//! （`python/oeh-policy/tests/fake_codex_app_server.py`）跑：控制流上的几条，和
//! `test_codex_teacher.py` 里同名的几条对应。线上逐字段一致另由
//! `test_codex_teacher_parity.py` 证明。没有 `python3` 时跳过（只有第一条不需要它）。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use oeh_brain::teacher::{
    CodexClient, CodexConversation, LaunchPolicy, TeacherConfig, TeacherError,
};
use serde_json::{json, Value};

fn fake() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/oeh-policy/tests/fake_codex_app_server.py")
}

fn python() -> Option<String> {
    let ok = Command::new("python3")
        .arg("-c")
        .arg("pass")
        .status()
        .is_ok_and(|s| s.success());
    ok.then(|| "python3".to_owned())
}

fn cfg(timeout_s: f64) -> TeacherConfig {
    TeacherConfig::from_json(&json!({
        "model": "gpt-6-luna", "reasoning_effort": "xhigh", "request_timeout_s": timeout_s,
        "max_attempts": 2, "retry_delays_s": [0],
    }))
    .unwrap()
}

/// 一个替身：状态目录与剧本走 argv 前缀里的 `env`，不碰本测试进程的环境（测试并行跑）。
fn client(
    state: &Path,
    plan: Value,
    extra: &[(&str, &str)],
    accounts: &[&str],
    timeout_s: f64,
) -> CodexClient {
    let mut prefix = vec![
        "env".to_owned(),
        format!("FAKE_CODEX_STATE={}", state.display()),
        format!("FAKE_CODEX_PLAN={plan}"),
    ];
    prefix.extend(extra.iter().map(|(k, v)| format!("{k}={v}")));
    prefix.push(python().unwrap());
    prefix.push(fake().display().to_string());
    CodexClient::new(
        cfg(timeout_s),
        state,
        "the frozen prompt",
        accounts.iter().map(|a| (*a).to_owned()).collect(),
        Some(prefix),
    )
    .unwrap()
}

fn body() -> Value {
    let mut conversation = CodexConversation::new(&cfg(1.0), "ep", json!({"type": "object"}));
    let packet = json!({"request_id": "r1", "decision": 0, "student_proposal": {"rows": 4},
        "current_eef": {"left": {"position": [0.0, 0.0, 0.5], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                        "right": {"position": [0.0, 0.0, 0.5], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}});
    conversation.add_packet(
        &packet.to_string(),
        &[json!({"type": "input_image", "image_url": "data:image/png;base64,AAAA"})],
        true,
    );
    conversation.body()
}

fn methods(state: &Path) -> Vec<String> {
    std::fs::read_to_string(state.join("requests.jsonl"))
        .unwrap_or_default()
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["method"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect()
}

#[test]
fn an_app_server_that_exits_is_retried_then_given_up_without_waiting_out_the_handshake() {
    let dir = tempfile::tempdir().unwrap();
    let mut teacher = CodexClient::new(
        cfg(10.0),
        dir.path(),
        "p",
        vec!["/acct/a".into()],
        Some(vec!["sh".into(), "-c".into(), "exit 3".into()]),
    )
    .unwrap()
    .with_launch_policy(LaunchPolicy {
        backoff: vec![Duration::ZERO; 2],
        ..LaunchPolicy::default()
    });
    let started = Instant::now();
    let error = teacher.preflight().unwrap_err();
    teacher.close();
    let TeacherError::Unavailable { message, .. } = error else {
        panic!("{error:?}")
    };
    assert!(
        message.contains("did not start in 3 attempt(s)") && message.contains("exited"),
        "{message}"
    );
    assert!(started.elapsed() < Duration::from_secs(15));
    assert_eq!(teacher.launches, 3);
}

#[test]
fn a_turn_returns_the_decision_with_its_usage() {
    let Some(_) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut teacher = client(
        dir.path(),
        json!([{"mode": "student", "steps": 3}]),
        &[],
        &["/acct/a"],
        10.0,
    );
    let checked = teacher.preflight().unwrap();
    let (data, errors) = teacher.create(&body()).unwrap();
    teacher.close();
    assert_eq!(checked["rate_limits"]["plan"], "pro");
    assert!(errors.is_empty());
    let arguments: Value =
        serde_json::from_str(data["output"][0]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments["request_id"], "r1");
    assert_eq!(data["output"][0]["name"], "robodojo_execute");
    assert_eq!(data["usage"]["input_tokens"], 1000);
    assert_eq!(data["usage"]["input_tokens_details"]["cached_tokens"], 600);
    assert!(!teacher.has_server());
}

#[test]
fn an_exhausted_account_hands_the_same_thread_to_the_next() {
    let Some(_) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let plan = json!([{"error": "usageLimitExceeded"}, {"mode": "student", "steps": 2}]);
    let mut teacher = client(dir.path(), plan, &[], &["/acct/a", "/acct/b"], 10.0);
    let (data, errors) = teacher.create(&body()).unwrap();
    teacher.close();
    assert_eq!(data["codex"]["account"], 1);
    assert_eq!(data["codex"]["switches"][0]["code"], "usageLimitExceeded");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0]["code"], "usageLimitExceeded");
    assert!(methods(dir.path()).contains(&"thread/resume".to_owned()));
}

#[test]
fn a_refused_login_is_fatal_with_its_status() {
    let Some(_) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut teacher = client(
        dir.path(),
        json!([{"error": "unauthorized"}]),
        &[],
        &["/acct/a"],
        10.0,
    );
    let error = teacher.create(&body()).unwrap_err();
    teacher.close();
    let TeacherError::Fatal { errors, .. } = error else {
        panic!("{error:?}")
    };
    assert_eq!(errors.last().unwrap()["status"], 401);
    assert_eq!(errors.last().unwrap()["fatal"], true);
}

#[test]
fn the_preflight_refuses_a_model_codex_does_not_serve() {
    let Some(_) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut teacher = client(
        dir.path(),
        json!([{}]),
        &[("FAKE_CODEX_MODELS", "gpt-5.5")],
        &["/acct/a"],
        10.0,
    );
    let error = teacher.preflight().unwrap_err();
    teacher.close();
    let TeacherError::Fatal { errors, .. } = error else {
        panic!("{error:?}")
    };
    assert_eq!(errors[0]["code"], "model_not_found");
}

#[test]
fn a_slow_turn_is_interrupted_and_retried_then_given_up() {
    let Some(_) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut teacher = client(dir.path(), json!([{"sleep": 30}]), &[], &["/acct/a"], 0.5);
    let error = teacher.create(&body()).unwrap_err();
    teacher.close();
    assert!(
        matches!(error, TeacherError::Unavailable { .. }),
        "{error:?}"
    );
    assert_eq!(
        methods(dir.path())
            .iter()
            .filter(|m| *m == "turn/interrupt")
            .count(),
        2
    );
}

#[test]
fn closing_from_another_thread_ends_a_turn_in_flight() {
    let Some(_) = python() else { return };
    let dir = tempfile::tempdir().unwrap();
    let mut teacher = client(dir.path(), json!([{"sleep": 30}]), &[], &["/acct/a"], 60.0);
    let handle = teacher.close_handle();
    let closer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(1500));
        handle.close();
    });
    let started = Instant::now();
    let error = teacher.create(&body()).unwrap_err();
    closer.join().unwrap();
    teacher.close();
    assert!(
        matches!(error, TeacherError::Unavailable { .. }),
        "{error:?}"
    );
    assert!(started.elapsed() < Duration::from_secs(20));
}
