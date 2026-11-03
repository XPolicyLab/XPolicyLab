//! `rule-dryrun`：在真的动起来之前，先在世界模型里跑一遍这次动作。
//!
//! 其余八条规则看的都是"现在"：现在路空不空、现在站着没有、现在流还新不新鲜。这条规则
//! 看的是"接下来"——把提案的速度指令交给 `predictor` 外推一遍，如果预测里的任何一个时刻
//! 会闯进安全包络（障碍 `obstacle_m` 0.5 m / 人 `person_m` 1.0 m 之内），就把判定收紧到
//! `Ask`：让人看一眼那条轨迹，而不是等 `rule-scene` 在 5 Hz 的复检里发现"已经太近了"。
//!
//! 三条纪律：
//!
//! - **只收紧**。这条规则永远只可能把 `Allow` 变成 `Ask`（走 `tighten`）。一次预测不是
//!   一个事实，不该拿它去 `Deny`——世界模型可以是错的，而它错的时候代价应该是多问一次人。
//! - **只管 motion**。`say` / `ask_human` / `stop` 不改机器人的位姿，对它们做运动学外推
//!   只会得到一条原地不动的轨迹和一次白问。
//! - **预测失败 = `Ask`**。预测器没给出结果（配错的 horizon、模型自己报错）意味着"这次
//!   我不知道接下来会怎样"，fail-closed 的方向是问人，不是放行。
//!
//! 读不到的距离（`None`：没有蒸馏器在看，或者视野里没有这类东西）不触发 `Ask`：那件事
//! 是 `rule-scene` 的，它对"不知道"的反应是 `Deny`，比这条严厉得多。这条规则只对
//! **预测出来的、确凿的**闯入发言。
//!
//! 留痕走 `RuleTrace::detail`（`Rule::evaluate_detailed`）：`dryrun: {min_obstacle_m,
//! min_person_m}` 只进审计（`act/admitted.verdict_trace` / `act/refused.trace`），**不进
//! params**。交给执行器的那一份参数必须正好是夹取链产出的那一份——把算给人看的中间量
//! 写进去，执行器会收到一个它没要过的键，链会多出一条 `act/clamped`，而夹过的 params
//! 还要重新过一遍能力自己的 schema（`additionalProperties: false` 的 motion 能力会因此
//! 次次被拒）。
//!
//! **fail-closed 靠 profile 里的 `required_rules`。** 这条规则 inject 了 `predictor`：
//! 没有世界模型时它停在 `Pending`，链上根本没有它——`Kernel::boot` 不会因此失败。真正
//! 让"少了预测器 = 不放行"成立的是 `admission.required_rules` 里的那一项：一份挂了
//! `rule-dryrun` 的 profile **必须**同时把 `rule-dryrun` 写进 `admission` 的
//! `required_rules`，否则掉了 `predict-kinematic` 那一行只会让这条安全规则悄悄消失。
//! 写了的话，`evaluate_chain` 的起始判定就是 `Ask{"missing required rules: rule-dryrun"}`
//! ——每一次提案都要人确认（没装审批服务就是直接拒绝）。见 `rules.rs` 里的
//! `a_profile_that_requires_dryrun_without_a_predictor_refuses_every_motion`。

use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_predict::predictor::{
    state_ref_from, ActionStep, PredictedState, Predictor, PredictorService,
};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::Admission;
use crate::capability::CapabilityClass;
use crate::rules::tighten;
use crate::verdict::{Rule, RuleCtx, Verdict};

pub const NAME: &str = "rule-dryrun";

/// 预测失败时的理由。一个字符串，不分类：调用方（人）的反应都是"看一眼"。
const PREDICTOR_ERROR: &str = "dryrun: predictor error";

/// 状态或预测里出现了非有限的距离时的理由。和预测失败分开：一个报错的模型和一个给出
/// `NaN` 的模型是两种坏法，日志里要分得出来。
const INVALID_STATE: &str = "dryrun: invalid state";

/// `duration_s` 不是一个能外推的时长时的理由。
///
/// `ActionStep::from_params` 只在这个键**缺席**时才补 `DEFAULT_DURATION_S`，所以一个写
/// 明的 `0`（`rule-clamp` 的出厂包络 `duration_s: [0, 2.0]` 允许它）会一路走成
/// `horizon = 0` → `steps = 0` → 空轨迹 → 没有闯入 → 放行，而执行器那边照样按它自己配
/// 的时长跑起来。也就是说：**一个 `duration_s: 0` 的提案是这条安全规则的一个旁路**。
/// 负数和 `NaN` 同理（`try_from_secs_f64` 会把它们变成 `PREDICTOR_ERROR`，那是在说
/// "模型坏了"，而事实是参数坏了）。这一类一律 `Ask` ——fail-closed 的方向是问人。
const INVALID_DURATION: &str = "dryrun: invalid duration";

fn d_obstacle_m() -> f64 {
    0.5
}
fn d_person_m() -> f64 {
    1.0
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_obstacle_m")]
    obstacle_m: f64,
    #[serde(default = "d_person_m")]
    person_m: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            obstacle_m: d_obstacle_m(),
            person_m: d_person_m(),
        }
    }
}

/// 预演规则。前两个字段就是它的安全包络，测试里直接构造。
#[derive(Clone)]
pub struct DryRunRule {
    /// 预测轨迹上障碍的最小容许距离。
    pub obstacle_m: f64,
    /// 预测轨迹上人的最小容许距离。
    pub person_m: f64,
    /// 世界模型。规则插件在 `apply` 时从 `predictor` 服务取来存进实例——链评估是同步的，
    /// 没有机会在那时候再去查注册表。
    pub predictor: Arc<dyn Predictor>,
}

impl DryRunRule {
    pub fn new(predictor: Arc<dyn Predictor>) -> Self {
        Self {
            obstacle_m: d_obstacle_m(),
            person_m: d_person_m(),
            predictor,
        }
    }

    fn from_config(c: Config, predictor: Arc<dyn Predictor>) -> Self {
        Self {
            obstacle_m: c.obstacle_m,
            person_m: c.person_m,
            predictor,
        }
    }
}

/// 轨迹上第一个闯进 `limit` 的状态（按时间先后）。
fn first_breach(
    states: &[PredictedState],
    limit: f64,
    pick: fn(&PredictedState) -> Option<f64>,
) -> Option<(f64, f64)> {
    states
        .iter()
        .find_map(|s| pick(s).filter(|d| *d < limit).map(|d| (d, s.t_offset_s)))
}

/// 轨迹上这一类距离的最小值（都不知道 → `None`）。
///
/// `NaN` **会**成为最小值：`m < NaN` 是假，于是落到 `_` 分支把累加器换成 `NaN`。这是
/// 故意的——一个 `NaN` 距离是坏数据，不该被当成"没看见"而从留痕里消失（它在
/// `evaluate_detailed` 里已经先被挡成 `Ask` 了，这里只是不给它第二条活路）。
fn min_distance(
    states: &[PredictedState],
    pick: fn(&PredictedState) -> Option<f64>,
) -> Option<f64> {
    states
        .iter()
        .filter_map(pick)
        .fold(None, |acc: Option<f64>, d| match acc {
            Some(m) if m < d => Some(m),
            _ => Some(d),
        })
}

/// 两个距离要么是"不知道"（`None`），要么是有限的数。
///
/// 为什么在规则边界上挡：`NaN < limit` 是**假**，所以一个 `NaN` 距离会被 `first_breach`
/// 读成"没有闯入"——最危险的那个方向（看起来很安全）。`predict-kinematic` 自己就拒非
/// 有限输入，但 `Predictor` 是一个可替换的槽位，一条安全规则不该指望每个未来的模型都
/// 自觉。蒸馏状态那一侧今天到不了这里（JSON 表示不了 `NaN`，`as_f64` 给的是 `None`），
/// 但 `StateRef` 是一个 `f64` 结构、不是只能从 JSON 来，所以两边一起挡。
fn distances_finite(obstacle: Option<f64>, person: Option<f64>) -> bool {
    [obstacle, person].into_iter().flatten().all(f64::is_finite)
}

/// 机器人**现在**的速度：蒸馏状态里有就用，没有就是 0。
///
/// 不是这次提案要的速度——那一份只进 `ActionStep`。`StateRef.v*` 的语义是"现在在以多快
/// 动"，两者混起来，一个从 `state.v*` 起算积分、再叠加 `actions[0]` 的预测器就会把指令
/// 速度算两遍。
fn current_velocity(ctx: &RuleCtx) -> (f64, f64, f64) {
    (
        ctx.f64("vx").unwrap_or(0.0),
        ctx.f64("vy").unwrap_or(0.0),
        ctx.f64("wz").unwrap_or(0.0),
    )
}

/// `Ask{reason}`，走 `tighten` 之前的那一半。
fn ask(reason: impl Into<String>) -> Verdict {
    Verdict::Ask {
        reason: reason.into(),
    }
}

fn obstacle_of(s: &PredictedState) -> Option<f64> {
    s.nearest_obstacle_m
}
fn person_of(s: &PredictedState) -> Option<f64> {
    s.nearest_person_m
}

impl Rule for DryRunRule {
    fn name(&self) -> &str {
        NAME
    }

    /// 判定本体在 [`Rule::evaluate_detailed`] 里；这一个入口只是把留痕丢掉。
    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict {
        self.evaluate_detailed(params, ctx, incoming).0
    }

    fn evaluate_detailed(
        &self,
        params: &mut Value,
        ctx: &RuleCtx,
        incoming: Verdict,
    ) -> (Verdict, Option<Value>) {
        if ctx.spec.class != CapabilityClass::Motion {
            return (incoming, None);
        }
        // 提案要的速度只进动作时间线；`StateRef.v*` 是机器人现在的速度。
        let action = ActionStep::from_params(params);
        // 时长先判：一条推不出来的轨迹不是"安全"，见 [`INVALID_DURATION`]。
        if !action.duration_s.is_finite() || action.duration_s <= 0.0 {
            return (tighten(incoming, ask(INVALID_DURATION)), None);
        }
        let state = state_ref_from(ctx.state, current_velocity(ctx), ctx.now_ns);
        if !distances_finite(state.nearest_obstacle_m, state.nearest_person_m) {
            return (tighten(incoming, ask(INVALID_STATE)), None);
        }
        // `try_from_secs_f64` 而不是 `from_secs_f64`：后者对负数和 `NaN` 是 panic，而
        // `duration_s` 是脑给的数。一条 panic 的规则会被链记成 Deny——那是兜底，不是
        // 设计（见 `rules` 模块的纪律）。
        let Ok(horizon) = Duration::try_from_secs_f64(action.duration_s) else {
            return (tighten(incoming, ask(PREDICTOR_ERROR)), None);
        };
        let prediction =
            match self
                .predictor
                .predict(&state, std::slice::from_ref(&action), horizon)
            {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(error = %e, "dryrun: prediction failed");
                    return (tighten(incoming, ask(PREDICTOR_ERROR)), None);
                }
            };
        if prediction
            .states
            .iter()
            .any(|s| !distances_finite(s.nearest_obstacle_m, s.nearest_person_m))
        {
            return (tighten(incoming, ask(INVALID_STATE)), None);
        }

        // 留痕先算：不管最后是放行还是问人，审计都该看得见这条轨迹最近处是多少。
        let detail = Some(json!({
            "dryrun": {
                "min_obstacle_m": min_distance(&prediction.states, obstacle_of),
                "min_person_m": min_distance(&prediction.states, person_of),
            }
        }));

        // 报**第一次**闯进包络的那一刻，不是全程最小值：最小值常常是被夹在 0 的那个
        // 末端状态，而"什么时候开始危险"才是人要看的东西。全程最小值在留痕里。
        if let Some((d, t)) = first_breach(&prediction.states, self.obstacle_m, obstacle_of) {
            return (
                tighten(
                    incoming,
                    ask(format!("dryrun: predicted obstacle {d:.1} m at t+{t:.1} s")),
                ),
                detail,
            );
        }
        if let Some((d, t)) = first_breach(&prediction.states, self.person_m, person_of) {
            return (
                tighten(
                    incoming,
                    ask(format!("dryrun: predicted person {d:.1} m at t+{t:.1} s")),
                ),
                detail,
            );
        }
        (incoming, detail)
    }
}

pub struct DryRunRulePlugin;

impl Plugin for DryRunRulePlugin {
    /// `predictor` 在 `inject` 里：没有世界模型就没有这条规则，它停在 `Pending`。
    /// **这本身不是 fail-closed**——`Kernel::boot` 照样成功，链上只是少了一条规则。
    /// fail-closed 由 profile 给：`admission.required_rules` 里写上 `rule-dryrun`，
    /// 整条链的起点就变成 `Ask{"missing required rules: rule-dryrun"}`（见
    /// `evaluate_chain`），于是"配了这条规则却没配预测器"的后果是每次提案都要人确认，
    /// 而不是这条规则悄悄不见了。见模块头。
    fn manifest(&self) -> Manifest {
        Manifest::new(NAME).inject(&["admission", "predictor"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let adm = ctx.get::<Admission>("admission")?;
        let predictor = ctx.get::<PredictorService>("predictor")?.0.clone();
        ctx.on_dispose(adm.register_rule(Arc::new(DryRunRule::from_config(cfg, predictor))));
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: NAME, build: || Box::new(DryRunRulePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{CapabilitySpec, RiskLevel};

    fn at(t: f64, obstacle: Option<f64>, person: Option<f64>) -> PredictedState {
        PredictedState {
            t_offset_s: t,
            x: 0.0,
            y: 0.0,
            yaw: 0.0,
            nearest_obstacle_m: obstacle,
            nearest_person_m: person,
        }
    }

    #[test]
    fn defaults_are_the_safety_envelope() {
        let c = Config::default();
        assert_eq!((c.obstacle_m, c.person_m), (0.5, 1.0));
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!((empty.obstacle_m, empty.person_m), (0.5, 1.0));
        assert!(serde_json::from_value::<Config>(json!({"obstacle": 1.0})).is_err());
    }

    /// 非有限的距离在规则边界上就被挡住：`NaN < limit` 是假，放它进去等于"看起来很安全"。
    #[test]
    fn non_finite_distances_are_rejected_at_the_boundary() {
        assert!(distances_finite(Some(1.0), None));
        assert!(distances_finite(None, None), "不知道不是坏数据");
        assert!(!distances_finite(Some(f64::NAN), None));
        assert!(!distances_finite(Some(1.0), Some(f64::INFINITY)));
        // 坏数据不会从留痕里消失：NaN 会成为最小值。
        let m = min_distance(
            &[at(0.1, Some(1.0), None), at(0.2, Some(f64::NAN), None)],
            obstacle_of,
        );
        assert!(m.expect("min").is_nan(), "{m:?}");
    }

    /// M6：`duration_s: 0` 从前是一条旁路（空轨迹 = 没有闯入 = 放行），而执行器照样按
    /// 自己配的时长跑起来。非正、非有限一律问人。
    #[test]
    fn a_duration_that_cannot_be_extrapolated_asks() {
        let rule = DryRunRule::new(Arc::new(NoTrajectory));
        let spec = CapabilitySpec {
            name: "walk".into(),
            class: CapabilityClass::Motion,
            risk_level: RiskLevel::Medium,
            params_schema: json!({"type": "object"}),
            lease_ttl_s: 1.0,
            overridable: false,
        };
        let state = std::collections::BTreeMap::new();
        let ctx = RuleCtx {
            now_ns: 0,
            fsm: oeh_kernel::fsm::FsmState::Standing,
            state: &state,
            spec: &spec,
            estop_engaged: false,
        };
        let verdict = |d: Value| {
            let mut params = json!({"vx": 0.5, "duration_s": d});
            rule.evaluate(&mut params, &ctx, Verdict::Allow)
        };
        for bad in [json!(0.0), json!(-1.0), json!(0)] {
            assert_eq!(
                verdict(bad.clone()),
                ask(INVALID_DURATION),
                "duration_s: {bad}"
            );
        }
        // 缺席仍然是 `DEFAULT_DURATION_S`（这条规则的老行为），正数照常往下走。
        let mut params = json!({"vx": 0.5});
        assert_eq!(
            rule.evaluate(&mut params, &ctx, Verdict::Allow),
            Verdict::Allow
        );
        let mut params = json!({"vx": 0.5, "duration_s": 1.5});
        assert_eq!(
            rule.evaluate(&mut params, &ctx, Verdict::Allow),
            Verdict::Allow
        );
    }

    /// 一个什么都不预测的世界模型：这个用例只看 `duration_s` 那道闸。
    struct NoTrajectory;
    impl Predictor for NoTrajectory {
        fn name(&self) -> &str {
            "none"
        }
        fn predict(
            &self,
            _state: &oeh_predict::predictor::StateRef,
            _actions: &[ActionStep],
            _horizon: Duration,
        ) -> Result<oeh_predict::predictor::Prediction, String> {
            Ok(oeh_predict::predictor::Prediction {
                states: Vec::new(),
                uncertainty: Vec::new(),
            })
        }
    }

    #[test]
    fn first_breach_is_the_earliest_one_and_min_ignores_unknowns() {
        let traj = [
            at(0.1, Some(1.2), None),
            at(0.2, Some(0.4), Some(2.0)),
            at(0.3, Some(0.1), Some(0.5)),
        ];
        assert_eq!(first_breach(&traj, 0.5, obstacle_of), Some((0.4, 0.2)));
        assert_eq!(first_breach(&traj, 0.05, obstacle_of), None);
        assert_eq!(min_distance(&traj, obstacle_of), Some(0.1));
        assert_eq!(min_distance(&traj, person_of), Some(0.5));
        assert_eq!(min_distance(&[at(0.1, None, None)], obstacle_of), None);
    }
}
