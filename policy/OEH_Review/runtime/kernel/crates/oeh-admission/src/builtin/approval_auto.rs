//! `approval-auto` 插件：把无人值守的审批装到准入服务上。
//!
//! 默认 `allow = false`——没人看着的时候，`Ask` 的结论是拒绝，而不是放行。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::approval::AutoApproval;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    allow: bool,
}

pub struct ApprovalAutoPlugin;

impl Plugin for ApprovalAutoPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("approval-auto").inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.set_approval(Arc::new(AutoApproval { allow: cfg.allow })));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: "approval-auto", build: || Box::new(ApprovalAutoPlugin) } }

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn default_denies() {
        assert!(!Config::default().allow);
        let empty: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(!empty.allow);
        let on: Config = serde_json::from_value(serde_json::json!({"allow": true})).unwrap();
        assert!(on.allow);
    }
}
