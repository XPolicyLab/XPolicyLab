//! `rule-clamp`：按能力名把数值参数夹进包络。
//!
//! 这是链上唯一一条改 params 的规则，也是唯一一条**不改判定**的规则：它不回答"能不能
//! 做"，只回答"最多做到多少"。`evaluate_chain` 会把改动前后的值记成 `act/clamped`，
//! 准入随后拿夹过的那份 params 重新过一遍能力自己的 schema。
//!
//! 两条边界：包络里有、但 params 里没有的参数**不补**（脑没提的参数不该被规则凭空造
//! 出来，能力的默认值归执行器管）；包络里没有的能力一个字节都不动。

use std::collections::BTreeMap;
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-clamp";

/// 能力名 → 参数名 → `[min, max]`。
pub type Envelopes = BTreeMap<String, BTreeMap<String, [f64; 2]>>;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    envelopes: Envelopes,
}

#[derive(Debug, Clone, Default)]
pub struct ClampRule {
    pub envelopes: Envelopes,
}

impl From<Config> for ClampRule {
    fn from(c: Config) -> Self {
        Self {
            envelopes: c.envelopes,
        }
    }
}

/// 把 `v` 夹进 `[lo, hi]`，夹不动就返回 `None`（调用方据此决定改不改 params）。
///
/// 配置写反（`lo > hi`）时按正过来的区间夹，不 panic——`f64::clamp` 在 `lo > hi` 或
/// 边界是 `NaN` 时会 panic，一个配置错误不该把规则炸掉（炸掉会被链记成 Deny，但那是
/// 兜底，不是这条规则该有的行为）。
fn clamped(v: f64, bounds: [f64; 2]) -> Option<f64> {
    let [a, b] = bounds;
    if a.is_nan() || b.is_nan() {
        return None;
    }
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let c = v.max(lo).min(hi);
    if c == v || !c.is_finite() {
        None
    } else {
        Some(c)
    }
}

impl Rule for ClampRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        let Some(envelope) = self.envelopes.get(&ctx.spec.name) else {
            return incoming;
        };
        // params 不是 object（脑发了个数组/字符串）：词汇校验那一关的事，这里不插手。
        let Some(obj) = params.as_object_mut() else {
            return incoming;
        };
        for (param, bounds) in envelope {
            // 缺的参数不补：只夹已经在那儿的。
            let Some(slot) = obj.get_mut(param) else {
                continue;
            };
            let Some(v) = slot.as_f64() else { continue };
            if let Some(c) = clamped(v, *bounds) {
                // `from_f64` 只在非有限时返回 None，`clamped` 已经排除了那种情况。
                if let Some(n) = serde_json::Number::from_f64(c) {
                    *slot = Value::Number(n);
                }
            }
        }
        incoming
    }
}

pub struct ClampRulePlugin;

impl Plugin for ClampRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(ClampRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(ClampRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn clamping_is_total_on_odd_bounds() {
        assert_eq!(clamped(0.9, [-0.2, 0.3]), Some(0.3));
        assert_eq!(clamped(-0.9, [-0.2, 0.3]), Some(-0.2));
        assert_eq!(clamped(0.1, [-0.2, 0.3]), None, "区间内不动");
        assert_eq!(clamped(0.9, [0.3, -0.2]), Some(0.3), "写反了也照夹");
        assert_eq!(clamped(0.9, [f64::NAN, 0.3]), None, "坏边界不动");
        assert_eq!(clamped(f64::NAN, [-0.2, 0.3]), Some(-0.2));
    }

    #[test]
    fn envelopes_parse_from_profile_shape() {
        let c: Config = serde_json::from_value(
            json!({"envelopes": {"walk": {"vx": [-0.2, 0.3], "duration_s": [0, 2.0]}}}),
        )
        .unwrap();
        assert_eq!(c.envelopes["walk"]["duration_s"], [0.0, 2.0]);
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert!(empty.envelopes.is_empty());
        assert!(serde_json::from_value::<Config>(json!({"envelope": {}})).is_err());
    }
}
