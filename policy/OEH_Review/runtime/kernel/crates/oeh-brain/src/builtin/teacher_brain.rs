//! `brain-teacher` 插件：复审老师的脑内通道（`review.teacher.channel: brain`），见
//! [`crate::teacher::lane`]。由 `oeh_policy.runtime` 在 brain 通道下写进 bundle，替代 Python 的
//! `TeacherBrain`；配置全部来自已冻结、已校验的策略配置，这里只再挡一遍形状。
//!
//! ```yaml
//! - id: brain-teacher
//!   config:
//!     brain_id: teacher
//!     wake: {default: ignore, rules: {obs/teacher_packet: wake, safety/estop: interrupt}, budget: {...}}
//!     output: /abs/attempt/review          # runtime.output；调用记录写在 <output>/teacher/
//!     episode_id: <hex>                    # prompt_cache_key = oeh-review-<episode_id>
//!     teacher: {api: codex_app_server, model: …, reasoning_effort: …, …}   # 冻结的 review.teacher
//!     accounts: [/abs/login-a, …]          # Python 读好的 codex_accounts_file
//!     prompt_file: /abs/attempt/review/teacher/prompt.md
//!     output_schema: {…}                   # review.response_schema()
//!     capability: oeh_manipulate
//!     turn_seconds: 1800
//!     method: teacher_codex
//!     launch: {attempts: 3, timeout_s: 120, backoff_s: [10, 30]}   # codex_teacher.LAUNCH_*
//!     adopt: {argv: [python, -c, <relay>, /tmp/oeh-relay-x/codex.sock], account: 0,   # 可选：预启动的
//!             rate_limits: {…}, next_id: 4, preflight: {…}}                         # app-server（TIME-1）
//! ```
//!
//! `apply` 里就把 app-server 的预检放进阻塞线程（[`TeacherLane::new`]）：内核一起来就开始握手，
//! Python 的 `Runtime.start` 随后来取结果。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{BrainSink, Brains, Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::teacher::codex_app_server::{Adopted, CodexClient, LaunchPolicy, TeacherConfig};
use crate::teacher::lane::{LaneConfig, TeacherLane};

const ID: &str = "brain-teacher";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Launch {
    attempts: u32,
    timeout_s: f64,
    backoff_s: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    brain_id: String,
    /// 唤醒策略，内联（和 Python `TeacherBrain` 的那份 dict 一样）。
    wake: Value,
    output: PathBuf,
    episode_id: String,
    teacher: Value,
    accounts: Vec<String>,
    prompt_file: PathBuf,
    output_schema: Value,
    capability: String,
    turn_seconds: Value,
    method: String,
    #[serde(default)]
    launch: Option<Launch>,
    /// 策略进程的 cwd：Python 通道的 app-server 在那里起。
    #[serde(default)]
    process_cwd: Option<PathBuf>,
    /// 内核自己往环境里加、策略进程里没有的变量（`oeh run` 设的 `OEH_PROFILE`）。
    #[serde(default)]
    env_remove: Vec<String>,
    /// 策略进程预启动的 app-server，本通道接着用（TIME-1；codex_teacher.py `CodexClient.hand_over`）。
    #[serde(default)]
    adopt: Option<Adopt>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Adopt {
    argv: Vec<String>,
    account: usize,
    rate_limits: Option<Value>,
    next_id: u64,
    preflight: Value,
}

pub struct TeacherBrainPlugin;

impl Plugin for TeacherBrainPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["brains", "world-log"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let brains = ctx.get::<Brains>("brains")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let bad = |reason: String| PluginError::Config {
            plugin: ID.into(),
            reason,
        };
        // 目前只有 Codex app-server 这一种传输搬了过来（Python 的 validate_review 也这么挡）。
        let api = cfg.teacher.get("api").and_then(Value::as_str);
        if api != Some("codex_app_server") {
            return Err(bad(format!(
                "the brain channel runs the codex_app_server teacher only, not {}",
                api.unwrap_or("<none>")
            )));
        }
        if cfg.accounts.is_empty() {
            return Err(bad("accounts must name at least one login directory".into()));
        }
        if let Some(adopt) = &cfg.adopt {
            if adopt.argv.is_empty() || adopt.account >= cfg.accounts.len() {
                return Err(bad(
                    "adopt needs a relay argv and one of the accounts".into()
                ));
            }
        }
        if !cfg.turn_seconds.is_number() {
            return Err(bad("turn_seconds must be a number".into()));
        }
        let policy = WakePolicy::from_yaml(&cfg.wake.to_string())
            .map_err(|e| bad(format!("wake policy: {e}")))?;
        let prompt = std::fs::read_to_string(&cfg.prompt_file)
            .map_err(|e| bad(format!("prompt {}: {e}", cfg.prompt_file.display())))?;
        let teacher = TeacherConfig::from_json(&cfg.teacher).map_err(bad)?;
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| bad("brain-teacher needs to be applied on a tokio runtime".into()))?;
        let mut client = CodexClient::new(
            teacher.clone(),
            cfg.output.join("teacher"),
            prompt,
            cfg.accounts,
            None,
        )
        .map_err(|e| bad(format!("codex client: {e}")))?
        .with_process(cfg.process_cwd, cfg.env_remove)
        // Python 通道的 runtime 把它设成 <output>/codex-app-server（Runtime.codex_logs）。
        .with_logs_dir(Some(cfg.output.join("codex-app-server")));
        if let Some(adopt) = cfg.adopt {
            client = client.with_adopted(Adopted {
                argv: adopt.argv,
                account: adopt.account,
                rate_limits: adopt.rate_limits,
                next_id: adopt.next_id,
                preflight: adopt.preflight,
            });
        }
        if let Some(launch) = cfg.launch {
            let secs = |s: f64| Duration::from_secs_f64(s.max(0.0));
            client = client.with_launch_policy(LaunchPolicy {
                attempts: launch.attempts,
                timeout: secs(launch.timeout_s),
                backoff: launch.backoff_s.into_iter().map(secs).collect(),
            });
        }
        let lane = TeacherLane::new(
            LaneConfig {
                output: cfg.output,
                teacher: cfg.teacher,
                capability: cfg.capability,
                turn_seconds: cfg.turn_seconds,
                method: cfg.method,
            },
            &teacher,
            &cfg.episode_id,
            cfg.output_schema,
            client,
            log.clone(),
            rt,
        );
        let observer = log.observe(Arc::new({
            let lane = lane.clone();
            move |ev| lane.on_world_event(ev)
        }));
        let sink: Arc<dyn BrainSink> = Arc::new(lane.clone());
        let detach = match brains.attach(&cfg.brain_id, policy, sink) {
            Ok(d) => d,
            Err(e) => {
                observer();
                lane.shutdown();
                return Err(bad(e.to_string()));
            }
        };
        ctx.on_dispose(Box::new(move || {
            detach();
            observer();
            lane.shutdown();
        }));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(TeacherBrainPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_typo_in_the_config_is_rejected() {
        let err = serde_json::from_value::<Config>(json!({
            "brain_id": "teacher", "wake": {}, "output": "/o", "episode_id": "e", "teacher": {},
            "accounts": [], "prompt_file": "/p", "output_schema": {}, "capability": "c",
            "turn_seconds": 1, "method": "m", "lanuch": {}
        }))
        .err()
        .expect("unknown field");
        assert!(err.to_string().contains("lanuch"), "{err}");
    }
}
