//! `rule-scene`：拿 `distill-ground` 蒸出来的场景状态挡下"会撞到东西/撞到人"的动作。
//!
//! 两组能力，两套判据：
//!
//! - `motion_caps`（走、靠近、转身）：要求 `clear_path == true`，且最近障碍、最近的人
//!   都在各自阈值之外。
//! - `gesture_caps`（挥手、摆姿势）：只看人——胳膊挥出去的半径比走一步小得多，但离人
//!   太近一样会打到。
//!
//! 距离的三种读法必须分清楚：`null` 是"视野里没有"，也就是**无穷远**（放行）；键不存在
//! 是"没有蒸馏器在看"，也就是**不知道**（Deny）；值不是数字同样是不知道。`clear_path`
//! 缺失单独拎出来说，因为它是这条规则的入口条件——没有它就没有"地面是空的"这个事实。

use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::admission::Admission;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-scene";

const CLEAR_PATH: &str = "clear_path";
const OBSTACLE: &str = "nearest_obstacle_m";
const PERSON: &str = "nearest_person_m";

fn d_obstacle_m() -> f64 {
    0.6
}
fn d_person_m() -> f64 {
    0.8
}
fn d_gesture_person_m() -> f64 {
    0.5
}
fn d_motion_caps() -> Vec<String> {
    ["walk", "approach", "turn"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}
fn d_gesture_caps() -> Vec<String> {
    ["gesture", "static_pose"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_obstacle_m")]
    obstacle_m: f64,
    #[serde(default = "d_person_m")]
    person_m: f64,
    #[serde(default = "d_gesture_person_m")]
    gesture_person_m: f64,
    #[serde(default = "d_motion_caps")]
    motion_caps: Vec<String>,
    #[serde(default = "d_gesture_caps")]
    gesture_caps: Vec<String>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            obstacle_m: d_obstacle_m(),
            person_m: d_person_m(),
            gesture_person_m: d_gesture_person_m(),
            motion_caps: d_motion_caps(),
            gesture_caps: d_gesture_caps(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SceneRule {
    pub obstacle_m: f64,
    pub person_m: f64,
    pub gesture_person_m: f64,
    pub motion_caps: Vec<String>,
    pub gesture_caps: Vec<String>,
}

impl Default for SceneRule {
    fn default() -> Self {
        Config::default().into()
    }
}

impl From<Config> for SceneRule {
    fn from(c: Config) -> Self {
        Self {
            obstacle_m: c.obstacle_m,
            person_m: c.person_m,
            gesture_person_m: c.gesture_person_m,
            motion_caps: c.motion_caps,
            gesture_caps: c.gesture_caps,
        }
    }
}

/// 一次距离读数。
enum Dist {
    /// 视野里没有这类东西（`null`）：无穷远。
    Far,
    Meters(f64),
    /// 键不存在或者值不是数字：不知道。
    Unknown,
}

fn distance(ctx: &RuleCtx, key: &str) -> Dist {
    match ctx.value(key) {
        None => Dist::Unknown,
        Some(v) if v.is_null() => Dist::Far,
        Some(v) => match v.as_f64() {
            Some(d) => Dist::Meters(d),
            None => Dist::Unknown,
        },
    }
}

/// `Ok(())` = 这个方向上是安全的。`NaN` 落到 `d > threshold` 的假分支，也就是不安全。
fn beyond(ctx: &RuleCtx, key: &str, threshold: f64, what: &str) -> Result<(), String> {
    match distance(ctx, key) {
        Dist::Far => Ok(()),
        Dist::Unknown => Err(format!("{key} is unknown")),
        Dist::Meters(d) if d > threshold => Ok(()),
        Dist::Meters(d) => Err(format!("{what} at {d}m is within {threshold}m")),
    }
}

impl Rule for SceneRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, _params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        let cap = &ctx.spec.name;
        if self.motion_caps.iter().any(|c| c == cap) {
            match ctx.value(CLEAR_PATH).map(|v| v.as_bool()) {
                Some(Some(true)) => {}
                Some(Some(false)) => return deny(incoming, "path is not clear"),
                Some(None) => return deny(incoming, format!("{CLEAR_PATH} is not a boolean")),
                None => return deny(incoming, format!("{CLEAR_PATH} is unknown")),
            }
            if let Err(e) = beyond(ctx, OBSTACLE, self.obstacle_m, "obstacle") {
                return deny(incoming, e);
            }
            if let Err(e) = beyond(ctx, PERSON, self.person_m, "person") {
                return deny(incoming, e);
            }
            return incoming;
        }
        if self.gesture_caps.iter().any(|c| c == cap) {
            if let Err(e) = beyond(ctx, PERSON, self.gesture_person_m, "person") {
                return deny(incoming, e);
            }
        }
        incoming
    }
}

pub struct SceneRulePlugin;

impl Plugin for SceneRulePlugin {
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
        ctx.on_dispose(adm.register_rule(Arc::new(SceneRule::from(cfg))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(SceneRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_the_ground_distiller() {
        let r = SceneRule::default();
        assert_eq!(
            (r.obstacle_m, r.person_m, r.gesture_person_m),
            (0.6, 0.8, 0.5)
        );
        assert_eq!(r.motion_caps, vec!["walk", "approach", "turn"]);
        assert_eq!(r.gesture_caps, vec!["gesture", "static_pose"]);
        let c: Config = serde_json::from_value(json!({"motion_caps": ["walk"]})).unwrap();
        assert_eq!(c.motion_caps, vec!["walk"]);
        assert_eq!(c.obstacle_m, 0.6, "其余字段仍是默认值");
        assert!(serde_json::from_value::<Config>(json!({"obstacle": 1.0})).is_err());
    }
}
