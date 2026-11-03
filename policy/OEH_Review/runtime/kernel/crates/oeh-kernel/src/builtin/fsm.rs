use std::sync::Arc;

use oeh_proto::Manifest;
use serde_json::Value;

use crate::factories::PluginFactory;
use crate::fsm::RobotFsm;
use crate::plugin::{Ctx, Plugin, PluginError};
use crate::state::StateRegistry;
use crate::world_log::WorldLog;

pub struct RobotFsmPlugin;
impl Plugin for RobotFsmPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("robot-fsm")
            .inject(&["world-log", "state"])
            .provide(&["fsm"])
    }
    fn apply(&self, ctx: &Ctx, _config: &Value) -> Result<(), PluginError> {
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        ctx.provide("fsm", Arc::new(RobotFsm::new(log, state)))
    }
}
inventory::submit! { PluginFactory { id: "robot-fsm", build: || Box::new(RobotFsmPlugin) } }
