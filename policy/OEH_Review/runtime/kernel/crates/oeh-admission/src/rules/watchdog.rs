//! `rule-watchdog`：感知断流的时候不许动。
//!
//! 读的是 `distill-watchdog` 写下的 `last_frame_t_kernel.<stream>` 寄存器（那条流最后
//! 一帧的内核时间戳），和判定时刻 `ctx.now_ns` 相减就是**帧**的年龄。刻意不用寄存器
//! 自己的写入时间（`RuleCtx::age_ns`）：蒸馏器每个周期都会把寄存器重写一遍，它一直很
//! "新"，新的是那次写，不是帧。
//!
//! 缺键、`null`（这条流从没来过帧）、值不是整数 → 一律当"不知道"，Deny。

use std::collections::BTreeMap;
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

pub const NAME: &str = "rule-watchdog";

fn d_streams() -> BTreeMap<String, f64> {
    [("lowstate".to_string(), 0.5), ("head_rgb".to_string(), 2.0)]
        .into_iter()
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_streams")]
    streams: BTreeMap<String, f64>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            streams: d_streams(),
        }
    }
}

/// 流名 → 允许的最大帧年龄（秒）。
#[derive(Debug, Clone)]
pub struct WatchdogRule {
    pub streams: BTreeMap<String, f64>,
}

impl Default for WatchdogRule {
    fn default() -> Self {
        Self {
            streams: d_streams(),
        }
    }
}

impl From<Config> for WatchdogRule {
    fn from(c: Config) -> Self {
        Self { streams: c.streams }
    }
}

impl Rule for WatchdogRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class != CapabilityClass::Motion {
            return incoming;
        }
        for (stream, max_age_s) in &self.streams {
            let key = format!("last_frame_t_kernel.{stream}");
            let Some(t_kernel) = ctx.value(&key).and_then(|v| v.as_u64()) else {
                return deny(incoming, format!("no frame time for stream `{stream}`"));
            };
            let age_s = ctx.now_ns.saturating_sub(t_kernel) as f64 / 1e9;
            // NaN 阈值（配置写坏了）也算超时：比较为假的那一半不能变成放行。
            if max_age_s.is_nan() || age_s > *max_age_s {
                return deny(
                    incoming,
                    format!("stream `{stream}` is stale: {age_s:.3}s > {max_age_s}s"),
                );
            }
        }
        incoming
    }
}

pub struct WatchdogRulePlugin;

impl Plugin for WatchdogRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(WatchdogRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(WatchdogRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_cover_lowstate_and_head_rgb() {
        let c: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!(c.streams, d_streams());
        let c: Config = serde_json::from_value(json!({"streams": {"lowstate": 0.2}})).unwrap();
        assert_eq!(c.streams.len(), 1);
        assert!(serde_json::from_value::<Config>(json!({"stream": {}})).is_err());
    }
}
