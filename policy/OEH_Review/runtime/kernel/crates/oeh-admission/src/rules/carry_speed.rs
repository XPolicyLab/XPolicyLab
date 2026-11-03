//! `rule-carry-speed`：持物时把解析运动的 `speed` 夹到 carry mode 的上限。
//!
//! 来历是 organize_table 的第二条经验（设计文档 §1.2）：论文 IK 服务器把每次更新限在
//! 2 cm / 0.1 rad，25 Hz 下一个全速步的加速度足以把夹住的闹钟甩出手指（3/3 次侧向搬运
//! 丢失）。记忆里"speed 0.25–0.4"是给脑的建议；这里的 0.4 是硬边界。
//!
//! 这条规则像 [`rule-clamp`](crate::rules::clamp) 一样**只改参数、不改判定**——除了
//! 一种情况：账本读不出来。`holding.left` / `holding.right` 这两个寄存器上
//! **`null` 的含义是"不知道"**（fail-closed，Deny），而不是 `rule-scene` 的
//! `Dist::Far` 那种"视野里没有 ⇒ 无穷远 ⇒ 放行"——不知道手里有没有东西，就不知道这
//! 一步该不该慢下来，那就不许动。
//!
//! 两条边界照抄 `rule-clamp`：`speed` 没写就**不补**（默认值归执行器管），`caps` 之外
//! 的能力一个字节都不动。

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

pub const NAME: &str = "rule-carry-speed";

fn d_caps() -> Vec<String> {
    ["arx_x5_move_eef", "arx_x5_move_relative"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}
fn d_max_carry_speed() -> f64 {
    0.4
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_caps")]
    caps: Vec<String>,
    #[serde(default = "d_max_carry_speed")]
    max_carry_speed: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            caps: d_caps(),
            max_carry_speed: d_max_carry_speed(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CarrySpeedRule {
    pub caps: Vec<String>,
    pub max_carry_speed: f64,
}

impl Default for CarrySpeedRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for CarrySpeedRule {
    fn from(c: Config) -> Self {
        Self {
            caps: c.caps,
            max_carry_speed: c.max_carry_speed,
        }
    }
}

/// `Ok(true)` = 至少一只手夹着东西，`Ok(false)` = 两只手都空着，`Err` = 不知道。
///
/// `ctx.value` 对不存在的键返回 `None`；`Value::Null` 落到 `_` 分支，因此同样是"不知道"。
fn any_holding(ctx: &RuleCtx) -> Result<bool, String> {
    let mut any = false;
    for arm in ["left", "right"] {
        let key = format!("holding.{arm}");
        match ctx.value(&key) {
            Some(Value::Bool(b)) => any |= *b,
            _ => return Err(format!("{key} is unknown")),
        }
    }
    Ok(any)
}

impl Rule for CarrySpeedRule {
    fn name(&self) -> &str {
        NAME
    }

    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class != CapabilityClass::Motion || !self.caps.contains(&ctx.spec.name) {
            return incoming;
        }
        let holding = match any_holding(ctx) {
            Ok(h) => h,
            Err(why) => return deny(incoming, format!("{NAME}: {why}")),
        };
        if !holding {
            return incoming;
        }
        // 持物：只夹已经在那儿的 `speed`，不凭空造一个。
        let Some(obj) = params.as_object_mut() else {
            return incoming;
        };
        let Some(slot) = obj.get_mut("speed") else {
            return incoming;
        };
        let Some(v) = slot.as_f64().filter(|v| v.is_finite()) else {
            return incoming;
        };
        if v > self.max_carry_speed {
            if let Some(n) = serde_json::Number::from_f64(self.max_carry_speed) {
                *slot = Value::Number(n);
            }
        }
        incoming
    }
}

pub struct CarrySpeedRulePlugin;

impl Plugin for CarrySpeedRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(CarrySpeedRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(CarrySpeedRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_are_the_two_analytic_move_capabilities_at_carry_speed() {
        let r = CarrySpeedRule::default();
        assert_eq!(r.caps, vec!["arx_x5_move_eef", "arx_x5_move_relative"]);
        assert!((r.max_carry_speed - 0.4).abs() < 1e-12);
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"max_carry_speed": 0.25})).unwrap();
        assert!((c.max_carry_speed - 0.25).abs() < 1e-12);
        assert_eq!(c.caps, d_caps(), "没写的字段仍是默认值");
        let c: Config = serde_json::from_value(json!({"caps": ["arx_x5_move_eef"]})).unwrap();
        assert_eq!(c.caps, vec!["arx_x5_move_eef"]);
        assert!((c.max_carry_speed - 0.4).abs() < 1e-12);
    }

    #[test]
    fn a_typo_is_a_boot_error_not_a_silently_ignored_key() {
        assert!(serde_json::from_value::<Config>(json!({"max_speed": 0.4})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"cap": ["x"]})).is_err());
    }
}
