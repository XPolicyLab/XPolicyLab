//! `rule-pose`：站不直就不许动。
//!
//! `gravity_proj_z` 是重力方向在机体 z 轴上的投影（`distill-fsm` 从 lowstate 里蒸出来
//! 的）：直立时约 -1，越接近 0 越是倒下去。`min_gravity_z`（默认 -0.85）是允许的上界，
//! 超过它就是在倾倒。
//!
//! 读不到这个寄存器 → Deny：没有人在看姿态，比"看着是歪的"更危险。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::capability::CapabilityClass;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-pose";

/// `distill-fsm` 写的寄存器名。
const KEY: &str = "gravity_proj_z";

fn d_min_gravity_z() -> f64 {
    -0.85
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_min_gravity_z")]
    min_gravity_z: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            min_gravity_z: d_min_gravity_z(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PoseRule {
    pub min_gravity_z: f64,
}

impl Default for PoseRule {
    fn default() -> Self {
        Self {
            min_gravity_z: d_min_gravity_z(),
        }
    }
}

impl From<Config> for PoseRule {
    fn from(c: Config) -> Self {
        Self {
            min_gravity_z: c.min_gravity_z,
        }
    }
}

impl Rule for PoseRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class != CapabilityClass::Motion {
            return incoming;
        }
        let Some(g) = ctx.f64(KEY) else {
            return deny(incoming, format!("{KEY} is unknown"));
        };
        // 非有限值（配置或蒸馏坏了）按"不知道"处理：`NaN > thr` 是假，不能因此放行。
        if !g.is_finite() {
            return deny(incoming, format!("{KEY} is not a finite number"));
        }
        if g > self.min_gravity_z {
            return deny(
                incoming,
                format!("{KEY} {g} > {} (tipping)", self.min_gravity_z),
            );
        }
        incoming
    }
}

pub struct PoseRulePlugin;

impl Plugin for PoseRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(PoseRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(PoseRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn threshold_defaults_to_the_standing_one() {
        let c: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!(c.min_gravity_z, -0.85);
        assert_eq!(PoseRule::default().min_gravity_z, -0.85);
        let c: Config = serde_json::from_value(json!({"min_gravity_z": -0.9})).unwrap();
        assert_eq!(c.min_gravity_z, -0.9);
        assert!(serde_json::from_value::<Config>(json!({"min_gravity": -0.9})).is_err());
    }
}
