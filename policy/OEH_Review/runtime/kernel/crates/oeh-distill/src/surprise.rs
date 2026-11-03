//! `distill-surprise`：把"世界和世界模型不一致"蒸馏成一条事件。
//!
//! 每个周期做两件事：拿**上一个周期对"现在"的预测**和现在真实的状态比一比，然后为
//! 下一个周期再预测一次。差得离谱就发一条 `obs/surprise`——脑的唤醒策略可以据此醒过来
//! 看一眼，而不是靠脑自己每隔一会儿去轮询世界。
//!
//! 为什么比的是这两维：位置无法直接观测（没有里程计、没有定位），能比的只有蒸馏状态里
//! 那几个由传感器直接支撑的量。`nearest_obstacle_m` 是"世界模型算得准不准"最直接的反映
//! （常速外推说还有 1.9 m，实际只剩 0.5 m，那就是有东西突然出现或者机器人没按指令动）；
//! `gravity_proj_z` 是"机器人自己还是不是它以为的样子"（预测认为姿态不变，实际倾了）。
//!
//! 两维共用一个 `err_threshold_m`：`gravity_proj_z` 是无量纲的重力投影，0.3 的差意味着
//! 机体倾了一大截——和 0.3 m 的距离误差一样值得被看见，一个阈值不必拆成两个。
//!
//! 只在两边都有数的时候才比距离：`None`（视野里没有 / 没有蒸馏器在看）不是"惊讶"，
//! 把它当成 0 m 会让每一次检测丢帧都炸出一条事件。
//!
//! 周期由注入的时钟驱动、由 `ctx.spawn_interval` 敲点。核心 [`Surprise`] 不自己起任务：
//! 测试拿 `ManualClock` 直接 `tick()`，不睡任何真实时间。

use std::sync::{Arc, Weak};
use std::time::Duration;

use oeh_kernel::clock::{Clock, ClockService};
use oeh_kernel::factories::PluginFactory;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Disposer, FrameStore, Plugin, PluginError, WorldLog};
use oeh_predict::kinematic::DEFAULT_DT_S;
use oeh_predict::predictor::{
    state_ref_from, ActionStep, PredictedState, Predictor, PredictorService, StateRef,
};
use oeh_proto::{EventDraft, EventKind, FrameRef, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

pub const ID: &str = "distill-surprise";

/// 帧引用取这条流的最新一帧：`nearest_*` 与 `gravity_proj_z` 都是从它蒸出来的，
/// 一条惊讶事件该指回它是在看哪一帧时发现的。
const LOWSTATE: &str = "lowstate";

const SAL_SURPRISE: u8 = 180;
const COLLAPSE_KEY: &str = "surprise";

/// 一次运动起跑的事件。
const STARTED: &str = "act/started";

/// 一次执行**结束**的事件。收到属于记着的那条指令的任何一条，就把指令清掉：一次被抢占
/// 或被撤销的 `walk{duration_s: 2.0}` 在 0.2 s 就停了，继续按 2 s 外推它等于在预测一段
/// 根本没发生的位移，于是每一拍都"惊讶"一次——而抢占、急停、FSM 撤销都是这套系统里
/// 正常的控制路径。
///
/// `act/refused` **不在**这张表里：流水线里唯一一条可能在 `act/started` 之后出现的
/// `act/refused` 是"重复的 proposal_id"，而那一条恰恰说明记着的那次运动**还在跑**
/// （正因为还在跑，重名的那次才被拒）——拿它清指令会把一次正在进行的运动当成已经停了。
const TERMINAL: [&str; 4] = [
    "act/result",
    "act/cancelled",
    "act/preempted",
    "act/lease_expired",
];

/// 一次预测往前看几个周期。
///
/// 一个周期就够对账了（下一拍比的就是 +1 个周期那一步），多推几步是给**迟到的那一拍**
/// 用的：`ctx.spawn_interval` 是 `MissedTickBehavior::Delay`，机器一忙，100 ms 的周期
/// 可能 300 ms 才醒。那时候手上要是只有一个 +0.1 s 的预测，拿它去比 0.3 s 之后的世界，
/// 报出来的"误差"完全是调度延迟造出来的。多推几步，就能挑出真正对得上流逝时间的那一步。
const HORIZON_PERIODS: u32 = 4;

fn d_period_ms() -> u64 {
    100
}
fn d_err_threshold_m() -> f64 {
    0.3
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_period_ms")]
    period_ms: u64,
    #[serde(default = "d_err_threshold_m")]
    err_threshold_m: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            period_ms: d_period_ms(),
            err_threshold_m: d_err_threshold_m(),
        }
    }
}

/// 周期至少要有预测器的一个步长。
///
/// 短了的话一次预测一个状态都推不出来，这个蒸馏器于是**永远不发**惊讶——一个只在 debug
/// 日志里露面的哑火。`Predictor` 合约里没有"你的步长是多少"这个问题（换成学出来的模型
/// 时它可能根本没有固定步长），所以这里拿 `predict-kinematic` 的默认步长
/// [`DEFAULT_DT_S`] 当下限。代价说清楚：把 `predict-kinematic` 的 `dt_s` 调小的 profile
/// 也得把 `period_ms` 留在 100 ms 以上，否则这道保守的拦截会挡住一份其实可行的配置——
/// 宁可启动时报错，也不要一个安静地什么都不做的蒸馏器。
fn check_period(period_ms: u64) -> Result<(), String> {
    let min_ms = (DEFAULT_DT_S * 1000.0).round() as u64;
    if period_ms < min_ms {
        return Err(format!(
            "period_ms {period_ms} is shorter than the predictor step {min_ms} ms; \
             a prediction would have no state to check against"
        ));
    }
    Ok(())
}

/// 最近一条 `act/started` 记下来的速度指令。
#[derive(Clone, Debug)]
struct Cmd {
    /// 哪一次提案下的指令。终局事件靠它认出"停的是不是我记着的这一条"。
    proposal_id: String,
    vx: f64,
    vy: f64,
    wz: f64,
    duration_s: f64,
    started_ns: u64,
}

impl Cmd {
    /// `now` 这一刻还剩多少时长；已经做完了 → `None`（机器人停着）。
    fn remaining_s(&self, now: u64) -> Option<f64> {
        let elapsed = now.saturating_sub(self.started_ns) as f64 / 1e9;
        let left = self.duration_s - elapsed;
        (left > 0.0).then_some(left)
    }
}

/// 上一个周期留下的预期。`gravity_proj_z` 不在 [`PredictedState`] 里——常速外推不预测
/// 姿态，所以预期就是上一周期观测到的那个值（"姿态不变"）。
///
/// 存的是整条轨迹加上"这份预测是什么时候做的"，而不是单独一个状态：对账要挑的是
/// **和实际流逝时间对得上**的那一步（见 [`HORIZON_PERIODS`]）。
struct Expect {
    made_ns: u64,
    states: Vec<PredictedState>,
    gravity_proj_z: f64,
}

impl Expect {
    /// `now` 这一刻该拿哪一步来对账：`t_offset_s` 最接近实际流逝时间的那一个。
    ///
    /// 流逝时间超出这条轨迹（最后一步再加半步）→ `None`：这一拍来得太晚，手上的预测
    /// 根本没覆盖到它，硬比就是在报告一个由调度延迟造出来的误差。
    fn at(&self, now: u64) -> Option<&PredictedState> {
        let elapsed = now.saturating_sub(self.made_ns) as f64 / 1e9;
        // 第一步的 offset 就是一个步长。
        let step = self.states.first()?.t_offset_s;
        let last = self.states.last()?.t_offset_s;
        if elapsed > last + step / 2.0 {
            return None;
        }
        self.states.iter().min_by(|a, b| {
            (a.t_offset_s - elapsed)
                .abs()
                .total_cmp(&(b.t_offset_s - elapsed).abs())
        })
    }
}

/// 惊讶检测的核心。插件把它挂在周期任务上；测试直接 [`Surprise::tick`]。
pub struct Surprise {
    clock: Arc<dyn Clock>,
    frames: Arc<FrameStore>,
    state: Arc<StateRegistry>,
    log: Arc<WorldLog>,
    predictor: Arc<dyn Predictor>,
    /// 每次预测往前看多远：[`HORIZON_PERIODS`] 个周期。
    horizon: Duration,
    err_threshold_m: f64,
    last_cmd: Mutex<Option<Cmd>>,
    expect: Mutex<Option<Expect>>,
}

impl Surprise {
    pub fn new(
        clock: Arc<dyn Clock>,
        frames: Arc<FrameStore>,
        state: Arc<StateRegistry>,
        log: Arc<WorldLog>,
        predictor: Arc<dyn Predictor>,
        period_ms: u64,
        err_threshold_m: f64,
    ) -> Arc<Self> {
        Arc::new(Self {
            clock,
            frames,
            state,
            log,
            predictor,
            horizon: Duration::from_millis(period_ms.max(1) * u64::from(HORIZON_PERIODS)),
            err_threshold_m,
            last_cmd: Mutex::new(None),
            expect: Mutex::new(None),
        })
    }

    /// 订阅世界日志，记下最近一条 `act/started` 的速度指令。
    ///
    /// 观察者持 `Weak`：日志攥着观察者，而 [`Surprise`] 攥着日志——强引用就是一个谁也
    /// 放不掉谁的环。
    pub fn attach(self: &Arc<Self>) -> Disposer {
        let weak = Arc::downgrade(self);
        self.log.observe(Arc::new(move |ev: &WorldEvent| {
            let kind = ev.kind.as_str();
            let started = kind == STARTED;
            if !started && !TERMINAL.contains(&kind) {
                return;
            }
            let Some(me) = Weak::upgrade(&weak) else {
                return;
            };
            if started {
                me.note_started(ev);
            } else {
                me.note_terminal(ev);
            }
        }))
    }

    /// 一条 `act/started`：params 里带速度分量的才是运动指令（`say` / `ask_human` 不改
    /// 机器人的速度，记下来只会让预测凭空多出一段位移）。
    fn note_started(&self, ev: &WorldEvent) {
        let params = &ev.payload["params"];
        if !ActionStep::params_carry_velocity(params) {
            return;
        }
        let a = ActionStep::from_params(params);
        *self.last_cmd.lock() = Some(Cmd {
            proposal_id: proposal_id(ev).unwrap_or_default().to_string(),
            vx: a.vx,
            vy: a.vy,
            wz: a.wz,
            duration_s: a.duration_s,
            started_ns: ev.t_kernel,
        });
    }

    /// 一条终局事件：属于记着的那次提案，就把指令清掉（机器人不动了）。别人的终局
    /// 事件不碰——同一时刻只有一个 motion 在跑，但 speech/query 的终局事件一样会经过
    /// 这里。
    fn note_terminal(&self, ev: &WorldEvent) {
        let Some(pid) = proposal_id(ev) else {
            return;
        };
        let mut g = self.last_cmd.lock();
        if g.as_ref().is_some_and(|c| c.proposal_id == pid) {
            *g = None;
        }
    }

    /// `now` 这一刻还在跑的动作（做完了、被清掉了、或者从来没有过 → `None`）。
    fn active_action(&self, now: u64) -> Option<ActionStep> {
        let g = self.last_cmd.lock();
        let cmd = g.as_ref()?;
        let left = cmd.remaining_s(now)?;
        Some(ActionStep {
            vx: cmd.vx,
            vy: cmd.vy,
            wz: cmd.wz,
            duration_s: left,
        })
    }

    /// 一个周期：先和上一周期的预期对账，再为下一个周期预测。
    pub fn tick(&self) {
        let now = self.clock.now_ns();
        let action = self.active_action(now);
        let cmd = action
            .as_ref()
            .map(|a| (a.vx, a.vy, a.wz))
            .unwrap_or((0.0, 0.0, 0.0));
        let actual = state_ref_from(&self.state.snapshot_entries(), cmd, now);

        // 1. 和上一个周期对"现在"的预期对账。`take()` 之后锁就放掉了：`compare` 会
        //    `append`，而 `WorldLog::append` 会同步跑完所有观察者（包括本插件自己那
        //    个），绝不能带着这把私有锁进去。
        let expect = self.expect.lock().take();
        if let Some(e) = expect {
            // 挑和实际流逝时间对得上的那一步；来得太晚就这一拍不对账。
            match e.at(now) {
                Some(s) => self.compare(s, e.gravity_proj_z, &actual),
                None => tracing::debug!(
                    "tick arrived past the prediction window; nothing to check against"
                ),
            }
        }

        // 2. 为后面几拍预测：正常节拍下用的是 +1 个周期那一步，多出来的几步只在这一拍
        //    迟到时才派上用场。
        let actions: Vec<ActionStep> = action.into_iter().collect();
        match self.predictor.predict(&actual, &actions, self.horizon) {
            Ok(p) => match p.states.is_empty() {
                false => {
                    *self.expect.lock() = Some(Expect {
                        made_ns: now,
                        states: p.states,
                        gravity_proj_z: actual.gravity_proj_z,
                    })
                }
                // 一个状态都没推出来 = 周期比预测器的步长还短。启动时 `check_period`
                // 已经挡过一次（拿的是 `predict-kinematic` 的默认步长），这里是换了
                // 别的预测器时的兜底。
                true => tracing::debug!(
                    horizon_ms = self.horizon.as_millis(),
                    "predictor returned no state; period shorter than its step?"
                ),
            },
            Err(e) => {
                tracing::debug!(error = %e, "prediction failed; this period has nothing to check against")
            }
        }
    }

    /// 挑出来的那一步预期 vs 现在的实际。超阈值就记一条 `obs/surprise`。
    fn compare(&self, expect: &PredictedState, expect_gravity: f64, actual: &StateRef) {
        // 距离只在两边都有数的时候才比：`None`（没有蒸馏器在看 / 视野里没有这类东西）
        // 不是"惊讶"，把它当 0 m 会让每一次检测丢帧都炸出一条事件。
        let d_obstacle = match (expect.nearest_obstacle_m, actual.nearest_obstacle_m) {
            (Some(p), Some(a)) => p - a,
            _ => 0.0,
        };
        let d_gravity = expect_gravity - actual.gravity_proj_z;
        let err_vec = [d_obstacle, d_gravity];
        if !err_vec.iter().any(|e| e.abs() > self.err_threshold_m) {
            return;
        }
        let refs: Vec<FrameRef> = match self.frames.latest(LOWSTATE) {
            Some(f) => vec![f.frame_ref()],
            None => Vec::new(),
        };
        self.log.append(
            EventDraft::new(
                EventKind::new("obs/surprise").expect("static"),
                ID,
                json!({
                    "err_vec": err_vec,
                    // `gravity_proj_z` 不在 `PredictedState` 里（常速外推不预测姿态），
                    // 但它是 err_vec 的第二维，读事件的人得看得见它的预期值。
                    "predicted": {
                        "t_offset_s": expect.t_offset_s,
                        "nearest_obstacle_m": expect.nearest_obstacle_m,
                        "nearest_person_m": expect.nearest_person_m,
                        "gravity_proj_z": expect_gravity,
                    },
                    "actual": serde_json::to_value(actual).unwrap_or(Value::Null),
                }),
            )
            .with_salience(SAL_SURPRISE)
            .with_collapse_key(COLLAPSE_KEY)
            .with_frame_refs(refs),
        );
    }
}

/// 一条 `act/*` 事件里的 `proposal_id`（准入写的每条 `act/*` 都带它）。
fn proposal_id(ev: &WorldEvent) -> Option<&str> {
    ev.payload.get("proposal_id").and_then(Value::as_str)
}

/// 提供惊讶检测。
pub struct SurprisePlugin;

impl Plugin for SurprisePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["clock", "frame-store", "state", "world-log", "predictor"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        check_period(cfg.period_ms).map_err(|reason| PluginError::Config {
            plugin: ID.into(),
            reason,
        })?;
        let clock = ctx.get::<ClockService>("clock")?.0.clone();
        let frames = ctx.get::<FrameStore>("frame-store")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let predictor = ctx.get::<PredictorService>("predictor")?.0.clone();
        let s = Surprise::new(
            clock,
            frames,
            state,
            log,
            predictor,
            cfg.period_ms,
            cfg.err_threshold_m,
        );
        ctx.on_dispose(s.attach());
        ctx.spawn_interval(Duration::from_millis(cfg.period_ms.max(1)), move || {
            s.tick()
        });
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(SurprisePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn null_and_empty_config_share_the_defaults() {
        let manual = Config::default();
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!((manual.period_ms, empty.period_ms), (100, 100));
        assert_eq!((manual.err_threshold_m, empty.err_threshold_m), (0.3, 0.3));
        assert!(serde_json::from_value::<Config>(json!({"period": 10})).is_err());
    }

    /// 周期短于预测器的步长是启动错误，不是一个安静地什么都不做的蒸馏器。
    #[test]
    fn a_period_shorter_than_the_predictor_step_is_a_config_error() {
        assert!(check_period(100).is_ok(), "默认周期正好等于默认步长");
        assert!(check_period(250).is_ok());
        let err = check_period(50).expect_err("50 ms < 100 ms");
        assert!(err.contains("period_ms 50"), "{err}");
        assert!(err.contains("100 ms"), "{err}");
    }

    /// 迟到的一拍挑的是和流逝时间对得上的那一步；超出窗口就不对账。
    #[test]
    fn expect_picks_the_step_that_matches_the_elapsed_time() {
        let at = |t: f64, d: f64| PredictedState {
            t_offset_s: t,
            x: 0.0,
            y: 0.0,
            yaw: 0.0,
            nearest_obstacle_m: Some(d),
            nearest_person_m: None,
        };
        let e = Expect {
            made_ns: 1_000_000_000,
            states: vec![at(0.1, 1.5), at(0.2, 1.0), at(0.3, 0.5), at(0.4, 0.0)],
            gravity_proj_z: -0.98,
        };
        assert_eq!(e.at(1_100_000_000).map(|s| s.t_offset_s), Some(0.1), "准时");
        assert_eq!(
            e.at(1_300_000_000).map(|s| s.t_offset_s),
            Some(0.3),
            "迟到两拍"
        );
        assert_eq!(e.at(1_260_000_000).map(|s| s.t_offset_s), Some(0.3), "就近");
        assert_eq!(
            e.at(1_400_000_000).map(|s| s.t_offset_s),
            Some(0.4),
            "窗口边界"
        );
        assert!(e.at(1_600_000_000).is_none(), "太晚了，这一拍不对账");
    }

    #[test]
    fn a_command_expires_when_its_duration_runs_out() {
        let c = Cmd {
            proposal_id: "p1".into(),
            vx: 1.0,
            vy: 0.0,
            wz: 0.0,
            duration_s: 0.5,
            started_ns: 1_000_000_000,
        };
        assert_eq!(c.remaining_s(1_000_000_000), Some(0.5));
        assert!(c.remaining_s(1_400_000_000).expect("还在跑") > 0.0);
        assert_eq!(c.remaining_s(1_500_000_000), None, "做完了");
        assert_eq!(c.remaining_s(9_000_000_000), None);
    }
}
