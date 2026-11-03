//! `rule-mode`：探索与评测是两个 profile，这条规则是 profile 对自己的断言。
//!
//! 设计文档 §1.5："放宽的步预算只用于 bootstrapping……exploration 允许 `reset` 与多次
//! attempt，evaluation 不允许。两种模式是两个 profile，不是两个 flag。"
//!
//! 这条规则不读任何状态寄存器，所以"`null` 是不是未知"的问题在它这里不出现——它读的
//! 只有自己的配置。`arx_x5_reset` 因此在评测 profile 里被拒两次：一次是
//! `rule-whitelist`（操作者的静态声明：这一局能做什么），一次是这里（profile 对自己
//! 是哪一种运行的断言）。重复是故意的，两边必须一致。
//!
//! **没有 class 门**：`reset` 在本路线上被声明成 `motion`（见 S1 合约 §1 的两处偏离），
//! 但这条规则管的是"这一局允许哪些能力"，与它是不是 motion 无关。
//!
//! 默认 `evaluation`：漏配一个 `mode` 的结果是探索专用能力被拒，而不是评测局里悄悄多
//! 出一个 `reset`。配置里写了别的字符串是启动错误；万一真有一个别的值到了运行期
//! （直接构造出来的规则），按 evaluation 处理——紧的那一档。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-mode";

pub const EXPLORATION: &str = "exploration";
pub const EVALUATION: &str = "evaluation";

fn d_mode() -> String {
    EVALUATION.into()
}
fn d_exploration_only() -> Vec<String> {
    vec!["arx_x5_reset".to_string()]
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_mode")]
    mode: String,
    #[serde(default = "d_exploration_only")]
    exploration_only: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: d_mode(),
            exploration_only: d_exploration_only(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArxModeRule {
    /// `"exploration"` 或 `"evaluation"`。插件在 `apply` 里校验。
    pub mode: String,
    pub exploration_only: Vec<String>,
}

impl Default for ArxModeRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for ArxModeRule {
    fn from(c: Config) -> Self {
        Self {
            mode: c.mode,
            exploration_only: c.exploration_only,
        }
    }
}

impl Rule for ArxModeRule {
    fn name(&self) -> &str {
        NAME
    }

    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if self.mode == EXPLORATION {
            return incoming;
        }
        if self.exploration_only.contains(&ctx.spec.name) {
            return deny(
                incoming,
                format!(
                    "{NAME}: `{}` is exploration-only and this run is `{}`",
                    ctx.spec.name, self.mode
                ),
            );
        }
        incoming
    }
}

pub struct ArxModeRulePlugin;

impl Plugin for ArxModeRulePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        if cfg.mode != EXPLORATION && cfg.mode != EVALUATION {
            return Err(PluginError::Config {
                plugin: NAME.into(),
                reason: format!(
                    "mode must be `{EXPLORATION}` or `{EVALUATION}`, got `{}`",
                    cfg.mode
                ),
            });
        }
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.register_rule(Arc::new(ArxModeRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(ArxModeRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_default_is_evaluation_with_reset_on_the_exploration_only_list() {
        let r = ArxModeRule::default();
        assert_eq!(r.mode, EVALUATION);
        assert_eq!(r.exploration_only, vec!["arx_x5_reset"]);
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"mode": "exploration"})).unwrap();
        assert_eq!(c.mode, "exploration");
        assert_eq!(c.exploration_only, d_exploration_only());
        let c: Config = serde_json::from_value(json!({"exploration_only": ["a", "b"]})).unwrap();
        assert_eq!(c.exploration_only, vec!["a", "b"]);
        assert_eq!(c.mode, EVALUATION);
    }

    #[test]
    fn a_typo_is_a_boot_error_not_a_silently_ignored_key() {
        assert!(serde_json::from_value::<Config>(json!({"modes": "exploration"})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"exploration": ["x"]})).is_err());
    }
}
