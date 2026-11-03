//! `rule-keepout`：已经放好的东西是禁区，目标位姿和到它的直线都不许进去。
//!
//! 这条规则的来历是 g7 那一局（设计文档 §1.4）：四件东西全都放好了，回臂的时候左臂在
//! 显示器旁停滞、蹭掉了 figurine，整局输掉。掉的不是目标点——目标点在别处——而是**路上**
//! 蹭到的。所以判据是"点到线段的距离"，不是"终点到障碍的距离"。
//!
//! 受保护的点来自 `distill-placement` 写进状态寄存器的 `placement_verified.<object>`。
//! 这些寄存器上 **读不出来 = 不知道 = Deny**（fail-closed）：`null`、类型不对、`xyz`
//! 不是三个有限数，一律当"有一个我看不懂的受保护物体"处理——不是 `rule-scene` 的
//! `Dist::Far` 那种"看不见就是无穷远"。一个读不出位置的受保护物体，恰恰是最不该开过去的。
//!
//! 探索用 `verdict: ask`（脑可以解释一句再过去），评测用 `verdict: deny`。
//!
//! 便宜是硬要求：这条规则在执行期间每 `recheck_ms`（默认 200 ms）重跑一次，而只有 Deny
//! 会打断正在跑的原语。

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

pub const NAME: &str = "rule-keepout";

/// `distill-placement` 写的寄存器前缀。
const PREFIX: &str = "placement_verified.";
const ARMS: [&str; 2] = ["left", "right"];
const DELTA_KEYS: [&str; 3] = ["dx", "dy", "dz"];

fn d_caps() -> Vec<String> {
    ["arx_x5_move_eef", "arx_x5_move_relative"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}
fn d_radius_m() -> f64 {
    0.10
}
fn d_verdict() -> String {
    "ask".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_caps")]
    caps: Vec<String>,
    #[serde(default = "d_radius_m")]
    radius_m: f64,
    #[serde(default = "d_verdict")]
    verdict: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            caps: d_caps(),
            radius_m: d_radius_m(),
            verdict: d_verdict(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeepoutRule {
    pub caps: Vec<String>,
    pub radius_m: f64,
    /// `"ask"` 或 `"deny"`。插件在 `apply` 里校验；别的字符串在这里按 `"ask"` 处理
    /// （更紧的那一档留给配置对的人，不是留给配置错的人）。
    pub verdict: String,
}

impl Default for KeepoutRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for KeepoutRule {
    fn from(c: Config) -> Self {
        Self {
            caps: c.caps,
            radius_m: c.radius_m,
            verdict: c.verdict,
        }
    }
}

fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|d| d.is_finite())
}

fn vec3(v: &Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    if a.len() != 3 {
        return None;
    }
    let mut out = [0.0f64; 3];
    for (slot, item) in out.iter_mut().zip(a) {
        *slot = finite(item)?;
    }
    Some(out)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm3(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// 点 `p` 到线段 `a→b` 的距离。退化成一个点（`a == b`）时就是点到点的距离。
fn point_segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ap = sub(p, a);
    let denom = dot(ab, ab);
    let t = if denom <= 0.0 {
        0.0
    } else {
        (dot(ap, ab) / denom).clamp(0.0, 1.0)
    };
    norm3([ap[0] - ab[0] * t, ap[1] - ab[1] * t, ap[2] - ab[2] * t])
}

/// 这一步走过的那条直线：`(当前 EEF, 目标点)`。
type Segment = ([f64; 3], [f64; 3]);

/// 一个受保护的点：位置 + 它自己的半径（条目里写了就用条目的，覆盖配置里的默认值）。
struct Protected {
    key: String,
    xyz: [f64; 3],
    radius_m: f64,
}

impl KeepoutRule {
    fn protected(&self, ctx: &RuleCtx) -> Result<Vec<Protected>, String> {
        let mut out = Vec::new();
        for (key, entry) in ctx.state.iter() {
            if !key.starts_with(PREFIX) {
                continue;
            }
            let Some(obj) = entry.value.as_object() else {
                return Err(format!("{key} is unreadable"));
            };
            let Some(xyz) = obj.get("xyz").and_then(vec3) else {
                return Err(format!("{key}.xyz is unreadable"));
            };
            let radius_m = obj
                .get("radius_m")
                .and_then(finite)
                .unwrap_or(self.radius_m);
            out.push(Protected {
                key: key.clone(),
                xyz,
                radius_m,
            });
        }
        Ok(out)
    }

    /// 这条臂这一步的线段 `(当前 EEF, 目标点)`；这条臂没给目标就是 `Ok(None)`。
    fn segment(&self, arm: &str, params: &Value, ctx: &RuleCtx) -> Result<Option<Segment>, String> {
        let Some(spec) = params.get(arm) else {
            return Ok(None);
        };
        if spec.is_null() {
            return Err(format!("{arm} is null (unknown target)"));
        }
        let obj = spec
            .as_object()
            .ok_or_else(|| format!("{arm} is not an object"))?;

        let mut delta = [0.0f64; 3];
        let mut has_target = false;
        let mut absolute: Option<[f64; 3]> = None;
        if let Some(v) = obj.get("position") {
            absolute = Some(vec3(v).ok_or_else(|| format!("{arm}.position is unreadable"))?);
            has_target = true;
        } else {
            for (i, k) in DELTA_KEYS.iter().enumerate() {
                if let Some(v) = obj.get(*k) {
                    delta[i] = finite(v).ok_or_else(|| format!("{arm}.{k} is unreadable"))?;
                    has_target = true;
                }
            }
        }
        if !has_target {
            return Ok(None);
        }

        let key = format!("eef_position.{arm}");
        let here = ctx
            .value(&key)
            .and_then(vec3)
            .ok_or_else(|| format!("{key} is unknown"))?;
        let target = match absolute {
            Some(p) => p,
            None => [here[0] + delta[0], here[1] + delta[1], here[2] + delta[2]],
        };
        Ok(Some((here, target)))
    }

    fn breach(&self, incoming: Verdict, reason: String) -> Verdict {
        if self.verdict == "deny" {
            deny(incoming, reason)
        } else {
            tighten(incoming, Verdict::Ask { reason })
        }
    }
}

impl Rule for KeepoutRule {
    fn name(&self) -> &str {
        NAME
    }

    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.class != CapabilityClass::Motion || !self.caps.contains(&ctx.spec.name) {
            return incoming;
        }
        let protected = match self.protected(ctx) {
            Ok(p) => p,
            Err(why) => return deny(incoming, format!("{NAME}: {why}")),
        };
        if protected.is_empty() {
            return incoming;
        }
        for arm in ARMS {
            let seg = match self.segment(arm, params, ctx) {
                Ok(Some(s)) => s,
                Ok(None) => continue,
                Err(why) => return deny(incoming, format!("{NAME}: {why}")),
            };
            for p in &protected {
                let d = point_segment_distance(p.xyz, seg.0, seg.1);
                if d < p.radius_m {
                    return self.breach(
                        incoming,
                        format!(
                            "{NAME}: {arm} path passes {d:.3} m from {} (keep-out {} m)",
                            p.key, p.radius_m
                        ),
                    );
                }
            }
        }
        incoming
    }
}

pub struct KeepoutRulePlugin;

impl Plugin for KeepoutRulePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        if cfg.verdict != "ask" && cfg.verdict != "deny" {
            return Err(PluginError::Config {
                plugin: NAME.into(),
                reason: format!("verdict must be `ask` or `deny`, got `{}`", cfg.verdict),
            });
        }
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.register_rule(Arc::new(KeepoutRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(KeepoutRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_are_ten_centimetres_and_ask() {
        let r = KeepoutRule::default();
        assert_eq!(r.caps, vec!["arx_x5_move_eef", "arx_x5_move_relative"]);
        assert!((r.radius_m - 0.10).abs() < 1e-12);
        assert_eq!(r.verdict, "ask");
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"verdict": "deny"})).unwrap();
        assert_eq!(c.verdict, "deny");
        assert!((c.radius_m - 0.10).abs() < 1e-12);
        assert_eq!(c.caps, d_caps());
    }

    #[test]
    fn a_typo_is_a_boot_error_not_a_silently_ignored_key() {
        assert!(serde_json::from_value::<Config>(json!({"radius": 0.1})).is_err());
        assert!(serde_json::from_value::<Config>(json!({"verdicts": "ask"})).is_err());
    }

    #[test]
    fn the_distance_is_to_the_segment_not_to_its_endpoints() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        // 线段正中上方 5 cm：两个端点都在 0.5 m 开外，线段却只有 5 cm。
        let p = [0.5, 0.05, 0.0];
        assert!((point_segment_distance(p, a, b) - 0.05).abs() < 1e-12);
        assert!((norm3(sub(p, a)) - 0.5024937810560445).abs() < 1e-9);
        // 线段之外：投影被夹到端点上。
        assert!((point_segment_distance([2.0, 0.0, 0.0], a, b) - 1.0).abs() < 1e-12);
        // 退化线段。
        assert!((point_segment_distance([0.0, 0.3, 0.0], a, a) - 0.3).abs() < 1e-12);
    }
}
