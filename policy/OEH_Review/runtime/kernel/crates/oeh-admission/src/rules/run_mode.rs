//! `rule-run-mode`：一个总闸，决定这次运行里 motion 类能力的默认待遇。
//!
//! `observe` 看不做（全拒）、`confirm` 每一步都要人点头（全问）、`active` 交给链上
//! 其余规则判断。非 motion 的能力（说话、查询、`stop`）不受这个闸门影响——观察模式下
//! 机器人还得能说话。
//!
//! 默认是 `observe`：漏配一个 `mode` 的结果是机器人不动，而不是它按最宽松的模式跑。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::capability::CapabilityClass;
use crate::rules::{deny, tighten};
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-run-mode";

/// 运行模式。序列化名是小写的 `observe` / `confirm` / `active`（profile 里就这么写）。
#[derive(Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RunMode {
    #[default]
    Observe,
    Confirm,
    Active,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    mode: RunMode,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RunModeRule {
    pub mode: RunMode,
}

impl From<Config> for RunModeRule {
    fn from(c: Config) -> Self {
        Self { mode: c.mode }
    }
}

impl Rule for RunModeRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class != CapabilityClass::Motion {
            return incoming;
        }
        match self.mode {
            RunMode::Observe => deny(incoming, "run_mode observe"),
            RunMode::Confirm => tighten(
                incoming,
                Verdict::Ask {
                    reason: "run_mode confirm".into(),
                },
            ),
            RunMode::Active => incoming,
        }
    }
}

pub struct RunModeRulePlugin;

impl Plugin for RunModeRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(RunModeRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(RunModeRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mode_parses_lowercase_and_defaults_to_observe() {
        let c: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!(c.mode, RunMode::Observe);
        for (s, m) in [
            ("observe", RunMode::Observe),
            ("confirm", RunMode::Confirm),
            ("active", RunMode::Active),
        ] {
            let c: Config = serde_json::from_value(json!({ "mode": s })).unwrap();
            assert_eq!(c.mode, m);
        }
        assert!(serde_json::from_value::<Config>(json!({"mode": "ACTIVE"})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"modes": "active"})).is_err());
    }
}
