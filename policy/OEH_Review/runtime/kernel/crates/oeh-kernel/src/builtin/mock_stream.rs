use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use oeh_proto::payload::LowState;
use oeh_proto::{Manifest, StreamSpec, Tier};
use serde::Deserialize;
use serde_json::Value;

use crate::factories::PluginFactory;
use crate::frame_store::FrameStore;
use crate::plugin::{Ctx, Plugin, PluginError};

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum PayloadMode {
    Counter,
    Lowstate,
}
fn default_payload_mode() -> PayloadMode {
    PayloadMode::Counter
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LowStateCfg {
    #[serde(default = "d_grav")]
    gravity_proj_z: f64,
    #[serde(default = "d_true")]
    rl_policy_active: bool,
    #[serde(default = "d_mode")]
    mode_machine: u8,
}
fn d_grav() -> f64 {
    -0.98
}
fn d_true() -> bool {
    true
}
fn d_mode() -> u8 {
    1
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    stream_id: String,
    #[serde(default = "default_schema")]
    schema: String,
    hz: f64,
    #[serde(default = "default_payload")]
    payload_bytes: usize,
    #[serde(default = "default_payload_mode")]
    payload: PayloadMode,
    #[serde(default)]
    lowstate: Option<LowStateCfg>,
}
fn default_schema() -> String {
    "oeh.Mock".into()
}
fn default_payload() -> usize {
    16
}

/// 合成流：按 hz 周期推帧。`payload: "counter"`（默认）时 payload 前 8 字节是小端计数器；
/// `payload: "lowstate"` 时每帧是一份 `LowState::to_bytes()`，schema 强制为
/// `LowState::SCHEMA`。第一帧在一个周期之后出现（`Ctx::spawn_interval` 的首个 tick 不再
/// 立即触发），不会和 boot 完就投递的 Op 抢跑。
pub struct MockStreamPlugin;

impl Plugin for MockStreamPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("mock-stream").inject(&["frame-store"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        // 非有限 hz，或者 hz 小到 1.0 / hz 溢出/超过一小时一帧，都会让
        // `Duration::from_secs_f64(1.0 / hz)` 在 spawn_interval 里 panic 或排出没人关心的
        // 超长周期；在这里 fail-closed。
        let period = 1.0 / cfg.hz;
        if !(cfg.hz.is_finite() && cfg.hz > 0.0 && period.is_finite() && period <= 3600.0) {
            return Err(PluginError::Config {
                plugin: "mock-stream".into(),
                reason: "hz must be finite, > 0 and >= 1/3600".into(),
            });
        }
        let fs = ctx.get::<FrameStore>("frame-store")?;
        let payload_mode = cfg.payload;
        // lowstate 模式下 schema 强制为 `LowState::SCHEMA`：下游按 schema 分流解码，不能让
        // 用户配置出 schema 和实际 payload 对不上的流。用户显式配了别的 schema 时这是一次
        // 静默覆盖，值得一条 warn 让排障的人不用去猜。
        let schema = if payload_mode == PayloadMode::Lowstate {
            if cfg.schema != default_schema() {
                tracing::warn!(
                    configured = %cfg.schema,
                    forced = LowState::SCHEMA,
                    "mock-stream: payload=lowstate forces schema (ignoring configured schema)"
                );
            }
            LowState::SCHEMA.to_string()
        } else {
            cfg.schema
        };
        // 注册是可逆 Effect：插件退化时这条流也要消失
        let unregister = fs
            .register(StreamSpec {
                stream_id: cfg.stream_id.clone(),
                schema,
                nominal_hz: cfg.hz,
                tier: Tier::Small,
            })
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(unregister);
        let counter = Arc::new(AtomicU64::new(0));
        let n = cfg.payload_bytes.max(8);
        let id = cfg.stream_id;
        let lowstate_cfg = cfg.lowstate.unwrap_or(LowStateCfg {
            gravity_proj_z: d_grav(),
            rl_policy_active: d_true(),
            mode_machine: d_mode(),
        });
        ctx.spawn_interval(Duration::from_secs_f64(1.0 / cfg.hz), move || {
            let payload = if payload_mode == PayloadMode::Lowstate {
                LowState {
                    gravity_proj_z: lowstate_cfg.gravity_proj_z,
                    rl_policy_active: lowstate_cfg.rl_policy_active,
                    mode_machine: lowstate_cfg.mode_machine,
                    joint_q: vec![],
                }
                .to_bytes()
            } else {
                let c = counter.fetch_add(1, Ordering::SeqCst);
                let mut buf = vec![0u8; n];
                buf[..8].copy_from_slice(&c.to_le_bytes());
                Bytes::from(buf)
            };
            let _ = fs.ingest(&id, payload, None);
        });
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: "mock-stream", build: || Box::new(MockStreamPlugin) } }
