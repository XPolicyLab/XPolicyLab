//! `rule-step-reserve`：给回臂留够原生步数。
//!
//! 设计文档 §1.4 的最后一句："`finish` 前必须为 `return_to_start` 预留 ≥ 120 步。"
//! 原生预算是一局比赛唯一不可再生的资源：花光了就只剩一个没法收尾的姿势，而收尾本身
//! 也要走路。
//!
//! `native_steps_remaining` 这个寄存器上 **读不出来 = 不知道 = Deny**（fail-closed）：
//! 键不存在、`null`、不是整数、负数，一律当"不知道还剩多少"处理，而不是
//! `rule-scene` 的 `Dist::Far` 那种"看不见就是无穷远"。
//!
//! 门是 `class == Motion`，所以 `arx_x5_finish`（`control`）本来就不在管辖范围里——
//! **不要**把它写进 `exempt`：写进去会让人以为这条规则对它发过言。

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

pub const NAME: &str = "rule-step-reserve";

const REMAINING: &str = "native_steps_remaining";

fn d_reserve_steps() -> u64 {
    120
}
fn d_exempt() -> Vec<String> {
    ["arx_x5_return_to_start", "arx_x5_wait", "arx_x5_reset"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_reserve_steps")]
    reserve_steps: u64,
    #[serde(default = "d_exempt")]
    exempt: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            reserve_steps: d_reserve_steps(),
            exempt: d_exempt(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepReserveRule {
    pub reserve_steps: u64,
    pub exempt: Vec<String>,
}

impl Default for StepReserveRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for StepReserveRule {
    fn from(c: Config) -> Self {
        Self {
            reserve_steps: c.reserve_steps,
            exempt: c.exempt,
        }
    }
}

impl Rule for StepReserveRule {
    fn name(&self) -> &str {
        NAME
    }

    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class != CapabilityClass::Motion {
            return incoming;
        }
        if self.exempt.contains(&ctx.spec.name) {
            return incoming;
        }
        // `as_i64` 对浮点数返回 None：一个"剩 119.5 步"的读数同样是读不出来。
        let Some(remaining) = ctx.value(REMAINING).and_then(|v| v.as_i64()) else {
            return deny(incoming, format!("{NAME}: {REMAINING} is unknown"));
        };
        if remaining < 0 {
            return deny(
                incoming,
                format!("{NAME}: {REMAINING} is negative ({remaining})"),
            );
        }
        if (remaining as u64) < self.reserve_steps {
            return deny(
                incoming,
                format!(
                    "{NAME}: only {remaining} native steps left, {} reserved for returning home",
                    self.reserve_steps
                ),
            );
        }
        incoming
    }
}

pub struct StepReserveRulePlugin;

impl Plugin for StepReserveRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(StepReserveRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(StepReserveRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_reserve_a_hundred_and_twenty_steps_for_the_three_escape_hatches() {
        let r = StepReserveRule::default();
        assert_eq!(r.reserve_steps, 120);
        assert_eq!(
            r.exempt,
            vec!["arx_x5_return_to_start", "arx_x5_wait", "arx_x5_reset"]
        );
        assert!(
            !r.exempt.iter().any(|c| c == "arx_x5_finish"),
            "finish 是 control，靠 class 门排除，不靠名单"
        );
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"reserve_steps": 200})).unwrap();
        assert_eq!(c.reserve_steps, 200);
        assert_eq!(c.exempt, d_exempt());
        let c: Config = serde_json::from_value(json!({"exempt": ["arx_x5_wait"]})).unwrap();
        assert_eq!(c.exempt, vec!["arx_x5_wait"]);
        assert_eq!(c.reserve_steps, 120);
    }

    #[test]
    fn a_typo_is_a_boot_error_not_a_silently_ignored_key() {
        assert!(serde_json::from_value::<Config>(json!({"reserve": 120})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"exempts": []})).is_err());
    }
}
