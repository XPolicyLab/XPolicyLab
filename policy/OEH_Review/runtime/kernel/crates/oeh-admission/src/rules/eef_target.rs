//! Optional fixed-base arm gate. Refuses relative targets; never clamps them.
//! Only the configured adapter's kernel-authenticated event source may update pose.
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{Manifest, WorldEvent};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::Admission;
use crate::rules::deny;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-eef-target";
const POSE: &str = "eef.pose";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    adapter_source: String,
    max_age_s: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pose {
    position: [f64; 3],
    quaternion_wxyz: [f64; 4],
    gripper_opening: f64,
    step: u64,
    remaining_steps: u64,
}

fn unit(q: &[f64; 4]) -> bool {
    q.iter().all(|x| x.is_finite())
        && (q.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs() <= 1e-4
}

fn valid_pose(value: &Value) -> bool {
    serde_json::from_value::<Pose>(value.clone()).is_ok_and(|p| {
        p.position.iter().all(|x| x.is_finite())
            && unit(&p.quaternion_wxyz)
            && p.gripper_opening.is_finite()
            && (0.0..=1.0).contains(&p.gripper_opening)
    })
}

struct EefTargetRule {
    max_age_s: f64,
}

impl Rule for EefTargetRule {
    fn name(&self) -> &str {
        NAME
    }
    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        if ctx.spec.name != "eef_target" {
            return incoming;
        }
        let pose = ctx.value(POSE).filter(|p| valid_pose(p));
        let Some(pose) = pose else {
            return deny(incoming, "EEF pose unavailable or invalid");
        };
        if ctx.age_ns(POSE).unwrap_or(u64::MAX) as f64 / 1e9 > self.max_age_s {
            return deny(incoming, "EEF pose stale");
        }
        let pose: Pose = serde_json::from_value(pose.clone()).expect("validated above");
        let p = params
            .get("position")
            .and_then(|v| serde_json::from_value::<[f64; 3]>(v.clone()).ok());
        let q = params
            .get("quaternion_wxyz")
            .and_then(|v| serde_json::from_value::<[f64; 4]>(v.clone()).ok());
        let (Some(p), Some(q)) = (p, q) else {
            return deny(incoming, "finite EEF position/quaternion required");
        };
        if !p.iter().all(|v| v.is_finite()) || !unit(&q) {
            return deny(incoming, "finite EEF position and unit quaternion required");
        }
        if params
            .get("gripper_close")
            .and_then(Value::as_bool)
            .is_none()
            || !params
                .get("steps")
                .and_then(Value::as_u64)
                .is_some_and(|n| (1..=5).contains(&n))
        {
            return deny(incoming, "boolean gripper and 1..5 integral steps required");
        }
        let distance = p
            .iter()
            .zip(pose.position)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if distance > 0.05 + 1e-9 {
            return deny(incoming, "EEF target exceeds 5 cm");
        }
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt()
            * pose
                .quaternion_wxyz
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
        let dot = q
            .iter()
            .zip(pose.quaternion_wxyz)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs()
            / norm;
        if 2.0 * dot.clamp(0.0, 1.0).acos() > 0.35 + 1e-9 {
            return deny(incoming, "EEF target exceeds 0.35 rad");
        }
        incoming
    }
}

pub struct EefTargetPlugin;
impl Plugin for EefTargetPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission", "world-log", "state", "fsm"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        // Invalid configuration must not become an unbounded freshness gate.
        if !cfg.max_age_s.is_finite()
            || cfg.max_age_s <= 0.0
            || !cfg.adapter_source.starts_with("rpc:")
        {
            return Err(PluginError::Other(
                "positive finite max_age_s and rpc: adapter_source required".into(),
            ));
        }
        let state = ctx.get::<StateRegistry>("state")?;
        let fsm = ctx.get::<RobotFsm>("fsm")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        ctx.on_dispose(log.observe(Arc::new(move |ev: &WorldEvent| {
            if ev.source != cfg.adapter_source {
                return;
            }
            if ev.kind.as_str() == "obs/eef" {
                let valid = valid_pose(&ev.payload);
                state.set(
                    POSE,
                    if valid {
                        ev.payload.clone()
                    } else {
                        Value::Null
                    },
                    NAME,
                );
                if valid {
                    let pose: Pose =
                        serde_json::from_value(ev.payload.clone()).expect("validated above");
                    state.set_many(
                        &[
                            ("env.step", json!(pose.step)),
                            ("env.remaining_steps", json!(pose.remaining_steps)),
                        ],
                        NAME,
                    );
                    if fsm.state() == FsmState::Boot {
                        let _ = fsm.transition(
                            FsmState::Standing,
                            "fixed-base adapter supplied a valid initial EEF pose",
                            NAME,
                        );
                    }
                }
            } else if ev.kind.as_str() == "obs/task" {
                if let Some(text) = ev.payload.get("instruction").and_then(Value::as_str) {
                    state.set("task.instruction", json!(text), NAME);
                }
            }
        })));
        let adm = ctx.get::<Admission>("admission")?;
        ctx.on_dispose(adm.register_rule(Arc::new(EefTargetRule {
            max_age_s: cfg.max_age_s,
        })));
        Ok(())
    }
}
inventory::submit! { PluginFactory { id: NAME, build: || Box::new(EefTargetPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilitySpec;
    use oeh_kernel::state::StateEntry;
    use std::collections::BTreeMap;

    #[test]
    fn refuses_bounds_and_stale_pose_without_changing_params() {
        let spec: CapabilitySpec = serde_json::from_value(json!({"name":"eef_target","class":"motion","risk_level":"medium","params_schema":{},"lease_ttl_s":2.0})).unwrap();
        let mut state = BTreeMap::from([(
            POSE.into(),
            StateEntry {
                value: json!({"position":[0,0,0],"quaternion_wxyz":[1,0,0,0],"gripper_opening":1,"step":0,"remaining_steps":220}),
                t_kernel: 0,
                source: NAME.into(),
            },
        )]);
        let rule = EefTargetRule { max_age_s: 1.0 };
        let good = json!({"position":[0.05,0,0],"quaternion_wxyz":[1,0,0,0],"gripper_close":false,"steps":5});
        for (target, age, allowed) in [
            (good.clone(), 0, true),
            (
                json!({"position":[0.04,0.04,0],"quaternion_wxyz":[1,0,0,0],"gripper_close":false,"steps":5}),
                0,
                false,
            ),
            (
                json!({"position":[0,0,0],"quaternion_wxyz":[(0.2_f64).cos(),(0.2_f64).sin(),0,0],"gripper_close":false,"steps":1}),
                0,
                false,
            ),
            (
                json!({"position":[0,0,0],"quaternion_wxyz":[1,0,0,0],"gripper_close":false,"steps":6}),
                0,
                false,
            ),
            (good.clone(), 2_000_000_000, false),
        ] {
            let ctx = RuleCtx {
                now_ns: age,
                fsm: FsmState::Standing,
                state: &state,
                spec: &spec,
                estop_engaged: false,
            };
            let mut params = target.clone();
            assert_eq!(
                rule.evaluate(&mut params, &ctx, Verdict::Allow) == Verdict::Allow,
                allowed
            );
            assert_eq!(params, target);
            assert_eq!(
                rule.evaluate(
                    &mut params,
                    &ctx,
                    Verdict::Deny {
                        reason: "upstream".into()
                    }
                ),
                Verdict::Deny {
                    reason: "upstream".into()
                }
            );
        }
        state.clear();
        let ctx = RuleCtx {
            now_ns: 0,
            fsm: FsmState::Standing,
            state: &state,
            spec: &spec,
            estop_engaged: false,
        };
        assert!(matches!(
            rule.evaluate(&mut good.clone(), &ctx, Verdict::Allow),
            Verdict::Deny { .. }
        ));
    }
}
