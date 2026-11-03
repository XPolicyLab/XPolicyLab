use crate::clock::ClockService;
use crate::factories::PluginFactory;
use crate::plugin::{Ctx, Plugin, PluginError};
use crate::state::StateRegistry;
use oeh_proto::Manifest;
use serde_json::Value;
use std::sync::Arc;

pub struct StateRegistryPlugin;
impl Plugin for StateRegistryPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("state-registry")
            .inject(&["clock"])
            .provide(&["state"])
    }
    fn apply(&self, ctx: &Ctx, _config: &Value) -> Result<(), PluginError> {
        let clock = ctx.get::<ClockService>("clock")?;
        ctx.provide("state", Arc::new(StateRegistry::new(clock.0.clone())))
    }
}
inventory::submit! { PluginFactory { id: "state-registry", build: || Box::new(StateRegistryPlugin) } }
