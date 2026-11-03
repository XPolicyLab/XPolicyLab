//! `rule-fsm`：按安全状态机的当前态挡下不该发生的动作。
//!
//! 两条：motion 类能力只能在 `STANDING | ENGAGED | ACTING` 里跑（和
//! [`oeh_kernel::fsm::RobotFsm::allows_motion`] 同一份判据）；`recover` 只有在
//! `EMERGENCY_STOP | FAULT` 里才有意义——它不是一个"顺手把状态机推一推"的能力。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::fsm::FsmState;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::capability::CapabilityClass;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-fsm";

/// 内置的"把机器人从急停/故障带回来"的能力名（见 `crate::builtin::admission`）。
const RECOVER: &str = "recover";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {}

/// 无配置：判据全在 FSM 本身的语义里。
#[derive(Debug, Clone, Copy, Default)]
pub struct FsmRule;

/// `RuleCtx` 只带着 FSM 的**快照**（规则读到的世界必须是同一个瞬间的），所以这里重新
/// 实现 `allows_motion` 而不是去问 `RobotFsm`——两处判据必须一致。
fn allows_motion(s: FsmState) -> bool {
    matches!(s, FsmState::Standing | FsmState::Engaged | FsmState::Acting)
}

impl Rule for FsmRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class == CapabilityClass::Motion && !allows_motion(ctx.fsm) {
            return deny(incoming, format!("fsm {}", ctx.fsm));
        }
        if ctx.spec.name == RECOVER && !matches!(ctx.fsm, FsmState::EmergencyStop | FsmState::Fault)
        {
            return deny(
                incoming,
                format!("recover needs EMERGENCY_STOP or FAULT, fsm is {}", ctx.fsm),
            );
        }
        incoming
    }
}

pub struct FsmRulePlugin;

impl Plugin for FsmRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(FsmRule)));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(FsmRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_states_match_the_fsm_and_config_takes_nothing() {
        for s in [FsmState::Standing, FsmState::Engaged, FsmState::Acting] {
            assert!(allows_motion(s), "{s}");
        }
        for s in [
            FsmState::Boot,
            FsmState::EmergencyStop,
            FsmState::Fault,
            FsmState::Recovering,
        ] {
            assert!(!allows_motion(s), "{s}");
        }
        assert!(serde_json::from_value::<Config>(serde_json::json!({})).is_ok());
        assert!(serde_json::from_value::<Config>(serde_json::json!({"mode": "active"})).is_err());
    }
}
