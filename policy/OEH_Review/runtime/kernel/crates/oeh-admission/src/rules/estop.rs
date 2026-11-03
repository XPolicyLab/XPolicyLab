//! `rule-estop`：急停按下时，任何有风险的能力都不准入。
//!
//! "按下"这个事实由准入服务在取快照时算好，交给 `ctx.estop_engaged`：它是
//! `AdmissionConfig::estop_flag` 这个文件存不存在、**或者** `distill-estop` 写下的
//! `estop` 寄存器为真（两条路各有各的默认路径，只改一个的 profile 不该把这条链解掉）。
//! 规则自己不碰文件系统，也不读寄存器——判定必须基于同一个瞬间的世界。
//!
//! `risk_level == none` 的能力（`say` / `ask_human` / `stop`）照常放行：急停期间还得能
//! 说话、还得能再按一次停。解除急停的 `recover` 是 high 风险，因此也被这条规则挡住——
//! 它的出路是操作员/审批把标志文件撤掉，而不是规则链放水。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::capability::RiskLevel;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-estop";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {}

/// 无配置：标志文件路径属于准入服务（`AdmissionConfig::estop_flag`）。
#[derive(Debug, Clone, Copy, Default)]
pub struct EstopRule;

impl Rule for EstopRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.estop_engaged && ctx.spec.risk_level != RiskLevel::None {
            return deny(incoming, "estop engaged");
        }
        incoming
    }
}

pub struct EstopRulePlugin;

impl Plugin for EstopRulePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let _cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.register_rule(Arc::new(EstopRule)));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(EstopRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_takes_nothing() {
        assert!(serde_json::from_value::<Config>(serde_json::json!({})).is_ok());
        assert!(
            serde_json::from_value::<Config>(serde_json::json!({"flag_path": "/tmp/x"})).is_err(),
            "标志文件路径归 admission 管，配在规则上是笔误"
        );
    }
}
