//! Simulation episode readiness. This does not assert a physical robot state.
//!
//! A benchmark client emits this only after a native reset produced a first
//! observation and its policy connection reset succeeded. ARX ledger events and
//! hardware state are deliberately outside this plugin's contract.

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventKind, Manifest, WorldEvent};
use serde::Deserialize;
use serde_json::{json, Value};

pub const ID: &str = "distill-sim-ready";
pub const KIND: &str = "obs/sim_policy_ready";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    benchmark: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    benchmark: String,
    episode_id: String,
    observation_sha256: String,
    native_reset_observed: bool,
    policy_connected: bool,
}

pub struct SimReadyPlugin;

impl Plugin for SimReadyPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["world-log", "state", "fsm"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        if cfg.benchmark.is_empty() || cfg.benchmark == "robodojo" {
            return Err(PluginError::Config {
                plugin: ID.into(),
                reason: "simulation benchmark identity required".into(),
            });
        }
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let fsm = ctx.get::<RobotFsm>("fsm")?;
        let kind = EventKind::new(KIND).expect("static");
        let benchmark = cfg.benchmark;
        let disposer = log.observe(Arc::new(move |event: &WorldEvent| {
            if event.kind != kind || fsm.state() != FsmState::Boot {
                return;
            }
            let Ok(ready) = serde_json::from_value::<Ready>(event.payload.clone()) else {
                return;
            };
            if ready.benchmark != benchmark
                || ready.episode_id.is_empty()
                || ready.observation_sha256.len() != 64
                || !ready
                    .observation_sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit())
                || !ready.native_reset_observed
                || !ready.policy_connected
            {
                return;
            }
            state.set_many(
                &[
                    ("sim.benchmark", json!(ready.benchmark)),
                    ("sim.episode_id", json!(ready.episode_id)),
                    ("sim.observation_sha256", json!(ready.observation_sha256)),
                ],
                ID,
            );
            let _ = fsm.transition(FsmState::Standing, "simulation reset and policy ready", ID);
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(SimReadyPlugin) } }
