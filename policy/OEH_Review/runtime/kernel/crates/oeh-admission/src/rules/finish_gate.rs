//! `rule-finish-gate`：宣告成功之前，两只夹爪得松开、两条臂得回家。
//!
//! 设计文档 §3.3 的最后一行：`finish(success)` 而任一夹爪命令 < 0.8 或臂没回位 → Ask
//! （提示脑先回臂开夹爪）。
//!
//! **永远不 Deny。** 一条被拒的 `finish` 会让这一局**根本拿不到原生分数**——而"让一局
//! 死掉的 episode 仍然被结算"正是整条路线存在的理由（`arx_x5_finish` 被声明成 `control`
//! 也是为了这个：`rule-fsm` 只拦 `Motion`，所以 FSM 不在 STANDING 的时候它照样跑得动）。
//! 所以这里最紧只到 `Ask`：探索 profile 的 `approval-auto{allow:true}` 会放它过去，
//! 评测 profile 的 `approval-auto{allow:false}` 把它变成一次拒绝——那是 profile 的
//! 决定，不是这条规则替它做的。
//!
//! **没有 class 门**（finish 是 `control`），只认名字。
//!
//! `gripper_cmd.left` / `.right` 和 `arms_home` 这几个寄存器上 **`null` 的含义是
//! "不知道"**（fail-closed，走 Ask），不是 `rule-scene` 的 `Dist::Far` 那种"看不见就
//! 是无穷远、于是放行"。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::Admission;
use crate::rules::tighten;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-finish-gate";

const ARMS_HOME: &str = "arms_home";
const ARMS: [&str; 2] = ["left", "right"];

fn d_cap() -> String {
    "arx_x5_finish".into()
}
fn d_min_gripper_cmd() -> f64 {
    0.8
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_cap")]
    cap: String,
    #[serde(default = "d_min_gripper_cmd")]
    min_gripper_cmd: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cap: d_cap(),
            min_gripper_cmd: d_min_gripper_cmd(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FinishGateRule {
    pub cap: String,
    pub min_gripper_cmd: f64,
}

impl Default for FinishGateRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for FinishGateRule {
    fn from(c: Config) -> Self {
        Self {
            cap: c.cap,
            min_gripper_cmd: c.min_gripper_cmd,
        }
    }
}

impl Rule for FinishGateRule {
    fn name(&self) -> &str {
        NAME
    }

    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        self.evaluate_detailed(params, ctx, incoming).0
    }

    fn evaluate_detailed(
        &self,
        params: &mut Value,
        ctx: &RuleCtx,
        incoming: Verdict,
    ) -> (Verdict, Option<Value>) {
        if ctx.spec.name != self.cap {
            return (incoming, None);
        }
        // 只有"宣告成功"要过这道门。失败/放弃照样要能收尾。
        if params.get("status").and_then(|v| v.as_str()) != Some("success") {
            return (incoming, None);
        }

        let mut unmet: Vec<String> = Vec::new();
        let mut seen = serde_json::Map::new();
        for arm in ARMS {
            let key = format!("gripper_cmd.{arm}");
            let read = ctx.value(&key).cloned().unwrap_or(Value::Null);
            let ok = read
                .as_f64()
                .filter(|v| v.is_finite())
                .is_some_and(|v| v >= self.min_gripper_cmd);
            seen.insert(key.clone(), read);
            if !ok {
                unmet.push(key);
            }
        }
        let home = ctx.value(ARMS_HOME).cloned().unwrap_or(Value::Null);
        if home != Value::Bool(true) {
            unmet.push(ARMS_HOME.to_string());
        }
        seen.insert(ARMS_HOME.to_string(), home);

        if unmet.is_empty() {
            return (incoming, Some(json!({"finish_gate": seen})));
        }
        let detail = json!({"finish_gate": seen, "unmet": unmet.clone()});
        (
            tighten(
                incoming,
                Verdict::Ask {
                    reason: format!(
                        "{NAME}: open both grippers and return to start before declaring success \
                         (unmet: {})",
                        unmet.join(", ")
                    ),
                },
            ),
            Some(detail),
        )
    }
}

pub struct FinishGateRulePlugin;

impl Plugin for FinishGateRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(FinishGateRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(FinishGateRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_gate_arx_x5_finish_at_eight_tenths_open() {
        let r = FinishGateRule::default();
        assert_eq!(r.cap, "arx_x5_finish");
        assert!((r.min_gripper_cmd - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"min_gripper_cmd": 0.95})).unwrap();
        assert!((c.min_gripper_cmd - 0.95).abs() < 1e-12);
        assert_eq!(c.cap, "arx_x5_finish");
        let c: Config = serde_json::from_value(json!({"cap": "other_finish"})).unwrap();
        assert_eq!(c.cap, "other_finish");
        assert!((c.min_gripper_cmd - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_typo_is_a_boot_error_not_a_silently_ignored_key() {
        assert!(serde_json::from_value::<Config>(json!({"capability": "x"})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"min_gripper": 0.8})).is_err());
    }
}
