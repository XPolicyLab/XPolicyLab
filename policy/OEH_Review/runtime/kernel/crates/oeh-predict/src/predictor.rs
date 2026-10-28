//! `Predictor` 合约与它的输入/输出类型。
//!
//! 合约刻意做得很窄：给一份"现在"（[`StateRef`]）和一串要做的动作（[`ActionStep`]），
//! 在一个时间窗内给出若干个未来状态（[`PredictedState`]）和每一步的不确定度。窄有窄的
//! 理由——规则链要在一次判定里同步调用它，惊讶检测要在每个周期调用它，两边都不能等一个
//! 模型加载、更不能等网络。
//!
//! [`StateRef`] 不是机器人的全部状态，只是**世界模型认识的那几个量**：姿态投影、两类
//! 距离、当前速度指令。距离是 `Option<f64>`：`None` = 不知道（没有蒸馏器在看，或者视野
//! 里没有这类东西）。预测器不替 `None` 编一个数出来——一个编出来的距离会变成一次假的
//! 安全判断，而"不知道"这件事本身有别的规则（`rule-scene`）负责。
//!
//! 错误用 `String`：预测失败不是一个要分类处理的东西，调用方（`rule-dryrun`）的反应
//! 永远是同一个——收紧到 `Ask`，让人来看。

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::state::StateEntry;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 蒸馏状态里这三个键是预测的输入，和 `distill-ground` / `distill-fsm` 写下的名字一致。
pub const GRAVITY_PROJ_Z: &str = "gravity_proj_z";
pub const NEAREST_OBSTACLE_M: &str = "nearest_obstacle_m";
pub const NEAREST_PERSON_M: &str = "nearest_person_m";

/// 一次动作没有写 `duration_s` 时按多久算。
///
/// 不是 0：一次没写时长的 `walk` 不是"原地不动"，而是"没说什么时候停"——按 0 预测
/// 等于预测它什么也不会发生，`rule-dryrun` 就永远拦不下它。1 s 是一个短到不会危言耸听、
/// 长到足以看见一次碰撞的窗口。
pub const DEFAULT_DURATION_S: f64 = 1.0;

/// 预测的起点：一次判定或一个蒸馏周期看到的世界，压成世界模型认识的那几个量。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateRef {
    pub t_kernel: u64,
    pub gravity_proj_z: f64,
    pub nearest_obstacle_m: Option<f64>,
    pub nearest_person_m: Option<f64>,
    pub vx: f64,
    pub vy: f64,
    pub wz: f64,
}

/// 一段恒定速度的动作。一串 `ActionStep` 按先后拼成一条动作时间线。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionStep {
    pub vx: f64,
    pub vy: f64,
    pub wz: f64,
    pub duration_s: f64,
}

impl ActionStep {
    /// 从一次提案/一条 `act/started` 的 params 里读出速度指令。
    ///
    /// 缺省：速度 0（没写就是不往那个方向动），时长 [`DEFAULT_DURATION_S`]。值不是数字
    /// 一律当成没写——params 已经过过一遍能力自己的 schema，这里再挑剔也只是多一种
    /// 拒绝的方式。
    pub fn from_params(params: &Value) -> Self {
        let f = |k: &str| params.get(k).and_then(Value::as_f64);
        Self {
            vx: f("vx").unwrap_or(0.0),
            vy: f("vy").unwrap_or(0.0),
            wz: f("wz").unwrap_or(0.0),
            duration_s: f("duration_s").unwrap_or(DEFAULT_DURATION_S),
        }
    }

    /// params 里有没有任何一个速度分量——"这是不是一次运动指令"的判据。
    pub fn params_carry_velocity(params: &Value) -> bool {
        ["vx", "vy", "wz"]
            .iter()
            .any(|k| params.get(k).and_then(Value::as_f64).is_some())
    }
}

/// 预测出来的一个未来状态。位姿是相对起点的机体坐标（起点 `x=y=yaw=0`）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictedState {
    pub t_offset_s: f64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub nearest_obstacle_m: Option<f64>,
    pub nearest_person_m: Option<f64>,
}

/// 一次预测的全部输出。`uncertainty[i]` 是 `states[i]` 的不确定度，两者等长。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub states: Vec<PredictedState>,
    pub uncertainty: Vec<f32>,
}

/// 世界模型的合约。实现必须是 `Send + Sync`：规则链在任意线程上同步调用它。
pub trait Predictor: Send + Sync {
    fn name(&self) -> &str;
    /// 从 `state` 出发、按 `actions` 这条动作时间线，推到 `horizon` 为止。
    fn predict(
        &self,
        state: &StateRef,
        actions: &[ActionStep],
        horizon: Duration,
    ) -> Result<Prediction, String>;
}

/// 注册表里以具体类型提供（`Arc<dyn Predictor>` 是胖指针，不能直接 downcast）——
/// 和 `oeh_kernel::ClockService` 同一个做法。服务名 `"predictor"`。
pub struct PredictorService(pub Arc<dyn Predictor>);

/// 从蒸馏状态的一份快照构造 [`StateRef`]。
///
/// `last_cmd` 是当前生效的速度指令（最近一条 `act/started` 的 params，或者没有动作在跑
/// 时的 0）——蒸馏状态里没有"机器人现在被命令以多快走"这件事，它只在准入那一侧。
///
/// 读不出来的键：距离是 `None`（不知道），`gravity_proj_z` 是 0.0。姿态用 0.0 兜底而不是
/// -1.0（直立）有意为之：用它的地方（`distill-surprise` 的第二维）比的是"预测值与实际值
/// 之差"，两边走的是同一个读法，缺失因此不会凭空造出一次惊讶；而拿"直立"当兜底就会。
pub fn state_ref_from(
    state: &BTreeMap<String, StateEntry>,
    last_cmd: (f64, f64, f64),
    t_kernel: u64,
) -> StateRef {
    // `as_f64` 对 `null`、字符串、对象都给 `None`——"没这个键"和"值不是个数"在这里是
    // 同一件事：都不知道。
    let number = |k: &str| state.get(k).and_then(|e| e.value.as_f64());
    let (vx, vy, wz) = last_cmd;
    StateRef {
        t_kernel,
        gravity_proj_z: number(GRAVITY_PROJ_Z).unwrap_or(0.0),
        nearest_obstacle_m: number(NEAREST_OBSTACLE_M),
        nearest_person_m: number(NEAREST_PERSON_M),
        vx,
        vy,
        wz,
    }
}
