//! `predict-kinematic`：不需要学习、不需要地图的保底世界模型。
//!
//! 它做的事只有一件：把当前速度指令按 `dt` 常速外推到 horizon，每一步把两类距离减去
//! 这一步走过的路程。**假设障碍就在正前方**——这是这个模型里唯一的"世界知识"，也是
//! 它唯一敢下的结论：往前走一米，离障碍就近一米。真实的障碍在侧前方时它会高估风险，
//! 那正是保底模型该犯的错（`rule-dryrun` 因此最多多问一次人，不会少问一次）。
//!
//! 不确定度是线性长大的（`uncertainty_per_step × 步数`）：常速外推越往后越不可信，而
//! 这个模型没有任何东西可以用来给出一个更聪明的数。读它的人只该把它当"别太信第 20 步"
//! 的提醒，不是一个标定过的方差。

use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::predictor::{
    ActionStep, PredictedState, Prediction, Predictor, PredictorService, StateRef,
};

pub const ID: &str = "predict-kinematic";

/// 一次预测最多推多少步。
///
/// horizon 来自提案的 `duration_s`（脑给的数）和配置里的 `dt_s`：一个
/// `duration_s: 1e9` 会让下面那个循环去要 1e10 个 `PredictedState`，分配失败走的是
/// `handle_alloc_error`（abort，不是 panic，规则链的 `catch_unwind` 接不住）。超过就
/// 报错——`rule-dryrun` 把预测错误收紧成 `Ask`，一个荒唐的时长于是变成"让人看一眼"，
/// 而不是内核当场消失。
pub const MAX_STEPS: usize = 10_000;

/// 默认积分步长，秒。
///
/// 公开的原因：一个按周期调用预测器的消费者（`distill-surprise`）必须保证自己的周期
/// **不短于**预测器的步长，否则一次预测一个状态都推不出来。`Predictor` 合约里没有"你
/// 的步长是多少"这个问题（换成学出来的模型时它可能根本没有固定步长），所以消费者只能
/// 拿这个默认值当下限来校验。
pub const DEFAULT_DT_S: f64 = 0.1;

fn d_dt_s() -> f64 {
    DEFAULT_DT_S
}
fn d_uncertainty_per_step() -> f64 {
    0.05
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_dt_s")]
    dt_s: f64,
    #[serde(default = "d_uncertainty_per_step")]
    uncertainty_per_step: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            dt_s: d_dt_s(),
            uncertainty_per_step: d_uncertainty_per_step(),
        }
    }
}

/// 常速外推预测器。字段就是它的配置，测试里直接构造。
#[derive(Clone, Copy, Debug)]
pub struct KinematicPredictor {
    /// 积分步长，秒。
    pub dt_s: f64,
    /// 每一步累加的不确定度。
    pub uncertainty_per_step: f64,
}

impl Default for KinematicPredictor {
    fn default() -> Self {
        Self {
            dt_s: d_dt_s(),
            uncertainty_per_step: d_uncertainty_per_step(),
        }
    }
}

impl From<Config> for KinematicPredictor {
    fn from(c: Config) -> Self {
        Self {
            dt_s: c.dt_s,
            uncertainty_per_step: c.uncertainty_per_step,
        }
    }
}

/// `t` 时刻生效的速度：按先后拼起来的动作时间线，走完了就是 0（动作做完机器人会停）。
fn velocity_at(actions: &[ActionStep], t: f64) -> (f64, f64, f64) {
    let mut start = 0.0;
    for a in actions {
        if a.duration_s <= 0.0 {
            continue;
        }
        let end = start + a.duration_s;
        if t < end {
            return (a.vx, a.vy, a.wz);
        }
        start = end;
    }
    (0.0, 0.0, 0.0)
}

/// 走过 `travelled` 米之后的剩余距离。不知道的距离照样不知道（不编一个数出来），
/// 知道的不会小于 0——"负的距离"不是一个物理量，读它的人会当成"更远"。
fn shrink(d: Option<f64>, travelled: f64) -> Option<f64> {
    d.map(|d| (d - travelled).max(0.0))
}

impl Predictor for KinematicPredictor {
    fn name(&self) -> &str {
        ID
    }

    fn predict(
        &self,
        state: &StateRef,
        actions: &[ActionStep],
        horizon: Duration,
    ) -> Result<Prediction, String> {
        let dt = self.dt_s;
        if !(dt.is_finite() && dt > 0.0) {
            return Err(format!("dt_s must be finite and > 0, got {dt}"));
        }
        // 非有限的输入不做预测：`NaN` 推出来的"距离"会一路变成一次假的安全判断，而
        // `NaN < threshold` 恰好是假——那正是最危险的方向（看起来很安全）。
        for a in actions {
            if ![a.vx, a.vy, a.wz, a.duration_s]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(format!("action must be finite, got {a:?}"));
            }
        }
        for d in [state.nearest_obstacle_m, state.nearest_person_m]
            .into_iter()
            .flatten()
        {
            if !d.is_finite() {
                return Err(format!("distance must be finite, got {d}"));
            }
        }

        let horizon_s = horizon.as_secs_f64();
        // 步数**向上**取整：比一个 `dt` 还短的 horizon（`walk{duration_s: 0.09}`）也要
        // 推出一步来。取整到 0 的话预测是空的，而 `rule-dryrun` 会把"模型什么也没说"
        // 读成"模型说没事"——一次没被预演过的运动就这么放过去了。
        //
        // 1e-9 的余量是给二进制浮点的：2.0 / 0.1 算出来是 19.999999999999996（向上取整
        // 会凭空多推一步），而 2.0000000000000004 / 0.1 是 20.000000000000004。减掉再
        // ceil，两边都落在 20 上。horizon 为 0 时仍然是 0 步（不动就是不动）。
        let steps = (horizon_s / dt - 1e-9).ceil().max(0.0);
        if steps > MAX_STEPS as f64 {
            return Err(format!(
                "horizon {horizon_s} s at dt {dt} s needs {steps} steps, more than {MAX_STEPS}"
            ));
        }
        let steps = steps as usize;

        let mut states = Vec::with_capacity(steps);
        let mut uncertainty = Vec::with_capacity(steps);
        let (mut x, mut y, mut yaw, mut travelled) = (0.0, 0.0, 0.0, 0.0f64);
        for i in 1..=steps {
            // 这一步开始时生效的速度。`t_offset` 直接由 `i * dt` 算，不累加——累加的
            // 误差会一路长到 horizon 的尾巴上。
            let (vx, vy, wz) = velocity_at(actions, (i - 1) as f64 * dt);
            // 沿运动方向走了多远。转身（`wz`）不算：它不把机器人往任何东西上送。
            travelled += (vx * vx + vy * vy).sqrt() * dt;
            yaw += wz * dt;
            x += (vx * yaw.cos() - vy * yaw.sin()) * dt;
            y += (vx * yaw.sin() + vy * yaw.cos()) * dt;
            states.push(PredictedState {
                t_offset_s: i as f64 * dt,
                x,
                y,
                yaw,
                nearest_obstacle_m: shrink(state.nearest_obstacle_m, travelled),
                nearest_person_m: shrink(state.nearest_person_m, travelled),
            });
            uncertainty.push((self.uncertainty_per_step * i as f64) as f32);
        }
        Ok(Prediction {
            states,
            uncertainty,
        })
    }
}

/// 提供 `predictor` 服务。
pub struct KinematicPlugin;

impl Plugin for KinematicPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).provide(&["predictor"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let p = KinematicPredictor::from(cfg);
        // 配错了的 `dt_s` 是启动错误，不是每次预测都报一遍的运行期错误。
        if !(p.dt_s.is_finite() && p.dt_s > 0.0) {
            return Err(PluginError::Config {
                plugin: ID.into(),
                reason: format!("dt_s must be finite and > 0, got {}", p.dt_s),
            });
        }
        if !p.uncertainty_per_step.is_finite() || p.uncertainty_per_step < 0.0 {
            return Err(PluginError::Config {
                plugin: ID.into(),
                reason: format!(
                    "uncertainty_per_step must be finite and >= 0, got {}",
                    p.uncertainty_per_step
                ),
            });
        }
        let svc: Arc<dyn Predictor> = Arc::new(p);
        ctx.provide("predictor", Arc::new(PredictorService(svc)))
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(KinematicPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn null_and_empty_config_share_the_defaults() {
        let manual = Config::default();
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!((manual.dt_s, empty.dt_s), (0.1, 0.1));
        assert_eq!(
            (manual.uncertainty_per_step, empty.uncertainty_per_step),
            (0.05, 0.05)
        );
        assert!(serde_json::from_value::<Config>(json!({"dt": 0.1})).is_err());
    }

    #[test]
    fn velocity_follows_the_action_timeline_and_stops_after_it() {
        let steps = [
            ActionStep {
                vx: 0.5,
                vy: 0.0,
                wz: 0.0,
                duration_s: 1.0,
            },
            ActionStep {
                vx: 0.0,
                vy: 0.0,
                wz: 0.3,
                duration_s: 1.0,
            },
        ];
        assert_eq!(velocity_at(&steps, 0.0), (0.5, 0.0, 0.0));
        assert_eq!(velocity_at(&steps, 0.999), (0.5, 0.0, 0.0));
        assert_eq!(velocity_at(&steps, 1.0), (0.0, 0.0, 0.3));
        assert_eq!(velocity_at(&steps, 2.0), (0.0, 0.0, 0.0), "动作做完就停");
        assert_eq!(velocity_at(&[], 0.0), (0.0, 0.0, 0.0));
    }

    #[test]
    fn shrink_never_goes_below_zero_and_keeps_none() {
        assert_eq!(shrink(Some(2.0), 0.5), Some(1.5));
        assert_eq!(shrink(Some(2.0), 5.0), Some(0.0), "不会是负的");
        assert_eq!(shrink(None, 5.0), None, "不知道就还是不知道");
    }
}
