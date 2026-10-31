//! `predict-kinematic` 的数值用例：常速外推走过的路程、距离怎么缩、不确定度怎么长，
//! 以及 `StateRef` 怎么从一份蒸馏状态快照读出来。
//!
//! 浮点断言一律带显式容差（1e-6）：这些数是一串 `dt` 累加出来的，逐位相等不是这里要
//! 验的东西。

use std::collections::BTreeMap;
use std::time::Duration;

use oeh_kernel::state::StateEntry;
use oeh_predict::kinematic::{KinematicPredictor, MAX_STEPS};
use oeh_predict::predictor::{state_ref_from, ActionStep, Predictor, StateRef};
use serde_json::{json, Value};

const TOL: f64 = 1e-6;

#[track_caller]
fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= TOL,
        "expected {want} ± {TOL}, got {got}"
    );
}

fn entry(v: Value) -> StateEntry {
    StateEntry {
        value: v,
        t_kernel: 7,
        source: "test".into(),
    }
}

fn snapshot(pairs: &[(&str, Value)]) -> BTreeMap<String, StateEntry> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), entry(v.clone())))
        .collect()
}

fn walking(obstacle: Option<f64>, person: Option<f64>) -> StateRef {
    StateRef {
        t_kernel: 0,
        gravity_proj_z: -0.98,
        nearest_obstacle_m: obstacle,
        nearest_person_m: person,
        vx: 0.5,
        vy: 0.0,
        wz: 0.0,
    }
}

fn step(vx: f64, duration_s: f64) -> ActionStep {
    ActionStep {
        vx,
        vy: 0.0,
        wz: 0.0,
        duration_s,
    }
}

/// 规格里的那个例子：`vx = 0.5` 走 2 s，初始障碍 2.0 m。dt 0.1 → 20 个状态，最后一步
/// 走过 1.0 m，于是障碍剩 1.0 m；不确定度每步长 0.05，严格递增。
#[test]
fn two_seconds_at_half_a_metre_per_second_eats_one_metre_of_clearance() {
    let p = KinematicPredictor::default();
    let out = p
        .predict(
            &walking(Some(2.0), Some(3.0)),
            &[step(0.5, 2.0)],
            Duration::from_secs_f64(2.0),
        )
        .expect("predict");

    assert_eq!(out.states.len(), 20, "2 s / 0.1 s");
    assert_eq!(out.uncertainty.len(), out.states.len());

    let last = out.states.last().expect("20 个状态");
    close(last.t_offset_s, 2.0);
    close(last.x, 1.0);
    close(last.y, 0.0);
    close(last.yaw, 0.0);
    close(last.nearest_obstacle_m.expect("知道障碍在哪"), 1.0);
    close(last.nearest_person_m.expect("知道人在哪"), 2.0);

    // 第一步：0.1 s 走 0.05 m。
    let first = &out.states[0];
    close(first.t_offset_s, 0.1);
    close(first.nearest_obstacle_m.expect("obstacle"), 1.95);

    // 不确定度 = 0.05 × 步数，严格递增。
    close(f64::from(out.uncertainty[0]), 0.05);
    close(f64::from(out.uncertainty[19]), 1.0);
    assert!(
        out.uncertainty.windows(2).all(|w| w[1] > w[0]),
        "严格递增: {:?}",
        out.uncertainty
    );
}

/// 不知道的距离照样不知道；知道的不会被推成负数；动作做完之后机器人就停了，不再往前吃。
#[test]
fn unknown_distances_stay_unknown_and_known_ones_never_go_below_zero() {
    let p = KinematicPredictor::default();
    let out = p
        .predict(
            &walking(Some(0.2), None),
            &[step(0.5, 1.0)],
            Duration::from_secs_f64(2.0),
        )
        .expect("predict");
    assert_eq!(out.states.len(), 20);
    assert!(
        out.states.iter().all(|s| s.nearest_person_m.is_none()),
        "None 不该被编成一个数"
    );
    assert!(
        out.states
            .iter()
            .all(|s| s.nearest_obstacle_m.expect("obstacle") >= 0.0),
        "距离不会是负的"
    );
    close(out.states[19].nearest_obstacle_m.expect("obstacle"), 0.0);
    // 动作只有 1 s：1 s 之后停在 0.5 m 处，不再往前。
    let last = out.states.last().expect("states");
    close(last.x, 0.5);
    close(out.states[9].x, 0.5);
}

/// 转身不吃余量：`wz` 只改朝向，没有沿运动方向的位移。
#[test]
fn turning_in_place_does_not_eat_clearance() {
    let p = KinematicPredictor::default();
    let turn = ActionStep {
        vx: 0.0,
        vy: 0.0,
        wz: 1.0,
        duration_s: 1.0,
    };
    let out = p
        .predict(
            &walking(Some(1.0), None),
            &[turn],
            Duration::from_secs_f64(1.0),
        )
        .expect("predict");
    close(out.states[9].nearest_obstacle_m.expect("obstacle"), 1.0);
    close(out.states[9].yaw, 1.0);
    close(out.states[9].x, 0.0);
}

/// 比一个 `dt` 还短的 horizon 也要推出一步来：空预测会被 `rule-dryrun` 读成"模型说没事"。
#[test]
fn a_horizon_shorter_than_one_step_still_predicts_one_step() {
    let p = KinematicPredictor::default();
    let out = p
        .predict(
            &walking(Some(1.0), None),
            &[step(0.5, 1.0)],
            Duration::from_secs_f64(0.09),
        )
        .expect("predict");
    assert_eq!(out.states.len(), 1, "向上取整，不是 0 步");
    close(out.states[0].t_offset_s, 0.1);
    close(out.states[0].nearest_obstacle_m.expect("obstacle"), 0.95);
    assert_eq!(out.uncertainty.len(), 1);
    // horizon 为 0 仍然是 0 步：不动就是不动。
    let none = p
        .predict(
            &walking(Some(1.0), None),
            &[step(0.5, 1.0)],
            Duration::from_secs_f64(0.0),
        )
        .expect("predict");
    assert!(none.states.is_empty(), "{:?}", none.states);
}

/// 荒唐的 horizon 报错而不是去要几十 GB 内存（分配失败是 abort，`catch_unwind` 接不住）。
#[test]
fn an_absurd_horizon_is_an_error_not_an_allocation() {
    let p = KinematicPredictor::default();
    let err = p
        .predict(
            &walking(Some(1.0), None),
            &[step(0.5, 1.0)],
            Duration::from_secs_f64(MAX_STEPS as f64),
        )
        .expect_err("horizon 超过上限");
    assert!(err.contains("horizon"), "{err}");
    // 上限之内照常跑。
    assert!(p
        .predict(
            &walking(Some(1.0), None),
            &[step(0.5, 1.0)],
            Duration::from_secs_f64(1.0)
        )
        .is_ok());
}

/// 非有限的输入不做预测：`NaN` 推出来的"距离"会变成一次假的安全判断。
#[test]
fn non_finite_inputs_are_refused() {
    let p = KinematicPredictor::default();
    let err = p
        .predict(
            &walking(Some(1.0), None),
            &[step(f64::NAN, 1.0)],
            Duration::from_secs_f64(1.0),
        )
        .expect_err("NaN 速度");
    assert!(err.contains("finite"), "{err}");

    let mut bad = walking(Some(1.0), None);
    bad.nearest_obstacle_m = Some(f64::NAN);
    assert!(p
        .predict(&bad, &[step(0.1, 1.0)], Duration::from_secs_f64(1.0))
        .is_err());
}

/// `StateRef` 从蒸馏状态的快照读出来：距离读不到就是 `None`（不编数），速度来自最近的
/// 动作指令。
#[test]
fn state_ref_reads_the_distilled_snapshot() {
    let s = snapshot(&[
        ("gravity_proj_z", json!(-0.97)),
        ("nearest_obstacle_m", json!(1.5)),
        ("nearest_person_m", json!(null)),
        ("clear_path", json!(true)),
    ]);
    let r = state_ref_from(&s, (0.3, 0.0, -0.2), 42);
    assert_eq!(r.t_kernel, 42);
    close(r.gravity_proj_z, -0.97);
    close(r.nearest_obstacle_m.expect("obstacle"), 1.5);
    assert_eq!(r.nearest_person_m, None, "null = 视野里没有人");
    close(r.vx, 0.3);
    close(r.vy, 0.0);
    close(r.wz, -0.2);

    // 什么都没蒸出来：距离不知道，姿态按 0.0 兜底（不是"直立"）。
    let empty = state_ref_from(&BTreeMap::new(), (0.0, 0.0, 0.0), 1);
    assert_eq!(empty.nearest_obstacle_m, None);
    assert_eq!(empty.nearest_person_m, None);
    close(empty.gravity_proj_z, 0.0);

    // 值的类型不对，一样是"不知道"。
    let junk = snapshot(&[("nearest_obstacle_m", json!("close"))]);
    assert_eq!(
        state_ref_from(&junk, (0.0, 0.0, 0.0), 1).nearest_obstacle_m,
        None
    );
}

/// `ActionStep::from_params`：params 缺省 → 速度 0、时长 1 s。
#[test]
fn action_step_reads_proposal_params() {
    let a = ActionStep::from_params(&json!({"vx": 0.5, "duration_s": 2.0}));
    close(a.vx, 0.5);
    close(a.vy, 0.0);
    close(a.wz, 0.0);
    close(a.duration_s, 2.0);
    let d = ActionStep::from_params(&json!({"vx": 0.5}));
    close(d.duration_s, 1.0);
    assert!(ActionStep::params_carry_velocity(&json!({"wz": 0.1})));
    assert!(!ActionStep::params_carry_velocity(&json!({"text": "hi"})));
}
