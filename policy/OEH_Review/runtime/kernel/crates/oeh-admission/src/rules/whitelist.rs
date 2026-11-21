//! `rule-whitelist`：只有列在 `allowed` 里的能力名能过。
//!
//! 能力面是动态的（执行器插件随时能注册新能力），白名单是运维对"这台机器今天允许做
//! 哪些事"的静态声明。没配 = 一条都不放行：一个空白名单是"还没配置"，不是"随便做"。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-whitelist";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    allowed: Vec<String>,
}

/// 允许的能力名清单。
#[derive(Debug, Clone, Default)]
pub struct WhitelistRule {
    pub allowed: Vec<String>,
}

impl From<Config> for WhitelistRule {
    fn from(c: Config) -> Self {
        Self { allowed: c.allowed }
    }
}

impl Rule for WhitelistRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        let cap = ctx.spec.name.as_str();
        if self.allowed.iter().any(|a| a == cap) {
            incoming
        } else {
            deny(incoming, format!("capability `{cap}` is not whitelisted"))
        }
    }
}

pub struct WhitelistRulePlugin;

impl Plugin for WhitelistRulePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.register_rule(Arc::new(WhitelistRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(WhitelistRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_an_empty_list_and_rejects_typos() {
        let c: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(WhitelistRule::from(c).allowed.is_empty());
        assert!(serde_json::from_value::<Config>(serde_json::json!({"allow": ["say"]})).is_err());
    }
}
