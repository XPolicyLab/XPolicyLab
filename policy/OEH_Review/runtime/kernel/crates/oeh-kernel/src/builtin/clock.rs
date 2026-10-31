use std::sync::Arc;

use oeh_proto::Manifest;
use serde_json::Value;

use crate::clock::{ClockService, MonotonicClock};
use crate::factories::PluginFactory;
use crate::plugin::{Ctx, Plugin, PluginError};

pub struct ClockPlugin;

impl Plugin for ClockPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("clock").provide(&["clock"])
    }
    fn apply(&self, ctx: &Ctx, _config: &Value) -> Result<(), PluginError> {
        ctx.provide(
            "clock",
            Arc::new(ClockService(Arc::new(MonotonicClock::new()))),
        )
    }
}

inventory::submit! { PluginFactory { id: "clock", build: || Box::new(ClockPlugin) } }
