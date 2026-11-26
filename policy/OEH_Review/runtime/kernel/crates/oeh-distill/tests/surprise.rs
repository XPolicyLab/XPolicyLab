//! `distill-surprise` 的周期用例。
//!
//! 时间由 `ManualClock` 给，节拍由测试自己敲（直接调 `tick()`）：一条关于"上一个周期
//! 预测了什么"的断言不该依赖调度器什么时候醒。判别信号是世界日志里的事件，不是睡够了。

use std::sync::Arc;

use oeh_distill::surprise::Surprise;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{FrameStore, ManualClock, WorldLog};
use oeh_predict::kinematic::KinematicPredictor;
use oeh_predict::predictor::Predictor;
use oeh_proto::{EventDraft, EventKind, StreamSpec, Tier, WorldEvent};
use serde_json::json;

const PERIOD_MS: u64 = 100;
const PERIOD_NS: u64 = PERIOD_MS * 1_000_000;
const TOL: f64 = 1e-6;

#[track_caller]
fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= TOL,
        "expected {want} ± {TOL}, got {got}"
    );
}

/// 一套直接构造的惊讶检测器：手动时钟、真的帧存储/状态寄存器/世界日志，保底预测器。
struct Rig {
    clock: Arc<ManualClock>,
    log: Arc<WorldLog>,
    state: Arc<StateRegistry>,
    surprise: Arc<Surprise>,
    _observer: oeh_kernel::Disposer,
    _stream: oeh_kernel::Disposer,
}

impl Rig {
    fn new() -> Rig {
        let clock = Arc::new(ManualClock::new(0));
        let log = Arc::new(WorldLog::new(clock.clone()));
        let frames = Arc::new(FrameStore::new(clock.clone(), 2.0));
        let state = Arc::new(StateRegistry::new(clock.clone()));
        let predictor: Arc<dyn Predictor> = Arc::new(KinematicPredictor::default());
        let stream = frames
            .register(StreamSpec {
                stream_id: "lowstate".into(),
                schema: "lowstate".into(),
                nominal_hz: 10.0,
                tier: Tier::Small,
            })
            .expect("register lowstate");
        frames
            .ingest("lowstate", bytes::Bytes::new(), None)
            .expect("ingest");
        let surprise = Surprise::new(
            clock.clone(),
            frames,
            state.clone(),
            log.clone(),
            predictor,
            PERIOD_MS,
            0.3,
        );
        let observer = surprise.attach();
        Rig {
            clock,
            log,
            state,
            surprise,
            _observer: observer,
            _stream: stream,
        }
    }

    /// 一条 `act/started`：准入起跑一次运动时写的就是这个形状。
    fn started(&self, params: serde_json::Value) {
        self.started_as("p1", params);
    }

    fn started_as(&self, proposal_id: &str, params: serde_json::Value) {
        self.act("act/started", proposal_id, json!({"params": params}));
    }

    /// 一条终局事件（`act/result` / `act/cancelled` / `act/preempted` / …）。
    fn ended(&self, kind: &str, proposal_id: &str) {
        self.act(kind, proposal_id, json!({"ok": true}));
    }

    fn act(&self, kind: &str, proposal_id: &str, extra: serde_json::Value) {
        let mut payload = json!({
            "brain_id": "fast", "proposal_id": proposal_id, "capability": "walk"
        });
        if let (Some(o), Some(e)) = (payload.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        self.log.append(EventDraft::new(
            EventKind::new(kind).expect("static"),
            "admission",
            payload,
        ));
    }

    fn observe(&self, pairs: &[(&str, serde_json::Value)]) {
        self.state.set_many(pairs, "test");
    }

    fn tick(&self) {
        self.surprise.tick();
    }

    fn advance(&self) {
        self.clock.advance(PERIOD_NS);
    }

    fn surprises(&self) -> Vec<WorldEvent> {
        self.log
            .since(0)
            .expect("log")
            .into_iter()
            .filter(|e| e.kind.as_str() == "obs/surprise")
            .collect()
    }
}

/// 规格里的那个场景：以 1 m/s 走着，上一个周期预测 0.1 s 后障碍剩 1.9 m，实际只剩
/// 0.5 m → 一条 `obs/surprise`，`err_vec[0] ≈ 1.4`。再一个周期世界没有再变 → 不再发。
#[test]
fn a_prediction_that_misses_by_more_than_the_threshold_becomes_one_event() {
    let rig = Rig::new();
    rig.started(json!({"vx": 1.0, "duration_s": 2.0}));
    rig.observe(&[
        ("nearest_obstacle_m", json!(2.0)),
        ("gravity_proj_z", json!(-0.98)),
    ]);

    // 第一个周期：还没有"上一个周期的预测"，不该发任何东西。
    rig.tick();
    assert!(rig.surprises().is_empty(), "第一个周期没有可对账的预测");

    // 第二个周期：世界变了（障碍突然只剩 0.5 m），预测说还有 1.9 m。
    rig.advance();
    rig.observe(&[("nearest_obstacle_m", json!(0.5))]);
    rig.tick();

    let evs = rig.surprises();
    assert_eq!(evs.len(), 1, "一条，不是每个周期一条");
    let e = &evs[0];
    assert_eq!(e.source, "distill-surprise");
    assert_eq!(e.salience, 180);
    assert_eq!(e.collapse_key.as_deref(), Some("surprise"));
    assert_eq!(
        e.frame_refs.first().map(|r| r.stream_id.as_str()),
        Some("lowstate"),
        "惊讶指回它是在看哪一帧时发现的: {:?}",
        e.frame_refs
    );
    let err = e.payload["err_vec"].as_array().expect("err_vec");
    assert_eq!(err.len(), 2, "两维：距离与姿态");
    close(err[0].as_f64().expect("err_vec[0]"), 1.4);
    close(err[1].as_f64().expect("err_vec[1]"), 0.0);
    close(
        e.payload["predicted"]["nearest_obstacle_m"]
            .as_f64()
            .expect("predicted"),
        1.9,
    );
    close(
        e.payload["actual"]["nearest_obstacle_m"]
            .as_f64()
            .expect("actual"),
        0.5,
    );

    // 第三个周期：世界不再变（0.5 m 原地），预测的 0.4 m 和实际的 0.5 m 只差 0.1 m。
    rig.advance();
    rig.tick();
    assert_eq!(rig.surprises().len(), 1, "阈值之内不再发");
}

/// 第二维：常速外推认为姿态不变，机器人却倒了下去。
#[test]
fn a_tipping_robot_is_a_surprise_too() {
    let rig = Rig::new();
    rig.observe(&[
        ("nearest_obstacle_m", json!(2.0)),
        ("gravity_proj_z", json!(-0.98)),
    ]);
    rig.tick();
    rig.advance();
    rig.observe(&[("gravity_proj_z", json!(-0.5))]);
    rig.tick();

    let evs = rig.surprises();
    assert_eq!(evs.len(), 1);
    let err = evs[0].payload["err_vec"].as_array().expect("err_vec");
    close(err[0].as_f64().expect("err_vec[0]"), 0.0);
    close(err[1].as_f64().expect("err_vec[1]"), -0.48);
    close(
        evs[0].payload["actual"]["gravity_proj_z"]
            .as_f64()
            .expect("actual"),
        -0.5,
    );
}

/// 没有动作在跑时预测是静止的（不凭空吃余量），世界也没变 → 一条都不发。
#[test]
fn a_quiet_world_produces_nothing() {
    let rig = Rig::new();
    rig.observe(&[
        ("nearest_obstacle_m", json!(2.0)),
        ("gravity_proj_z", json!(-0.98)),
    ]);
    for _ in 0..4 {
        rig.tick();
        rig.advance();
    }
    assert!(rig.surprises().is_empty(), "{:?}", rig.surprises());
}

/// 一次被提前结束的运动不再被外推。
///
/// `walk{vx:5.0, duration_s:2.0}` 在起跑的同一刻就被撤销（抢占、急停、FSM 撤销都是这套
/// 系统里正常的控制路径）。不清指令的话，这个蒸馏器会按标称的 2 s 继续预测：每一拍预测
/// 障碍近 0.5 m，而真实世界一动不动——于是每一拍都"惊讶"一次，全是假的。
#[test]
fn a_motion_that_ends_early_stops_being_extrapolated() {
    let rig = Rig::new();
    rig.observe(&[
        ("nearest_obstacle_m", json!(2.0)),
        ("gravity_proj_z", json!(-0.98)),
    ]);
    rig.started_as("p1", json!({"vx": 5.0, "duration_s": 2.0}));
    rig.ended("act/result", "p1");
    for _ in 0..4 {
        rig.tick();
        rig.advance();
    }
    assert!(
        rig.surprises().is_empty(),
        "结束了的运动不该再被预测: {:?}",
        rig.surprises()
    );
}

/// 别人的终局事件不碰我记着的这条指令。
#[test]
fn a_terminal_event_for_another_proposal_is_ignored() {
    let rig = Rig::new();
    rig.observe(&[
        ("nearest_obstacle_m", json!(2.0)),
        ("gravity_proj_z", json!(-0.98)),
    ]);
    rig.started_as("p1", json!({"vx": 5.0, "duration_s": 2.0}));
    rig.ended("act/result", "p-other");
    rig.tick();
    rig.advance();
    rig.tick();
    // 指令还在 → 预测的是"障碍近了 0.5 m"，实际没动 → 0.5 > 0.3，一条惊讶。
    assert_eq!(rig.surprises().len(), 1, "记着的运动还在跑");
}

/// 迟到的一拍对的是**和流逝时间对得上**的那一步，不是永远的"+1 个周期"。
///
/// `spawn_interval` 是 `MissedTickBehavior::Delay`：机器一忙，100 ms 的周期可能 300 ms
/// 才醒。这一拍要是拿 +0.1 s 的那一步去比 0.3 s 之后的世界，报出来的误差全是调度延迟
/// 造出来的（这里就是 1.0 m，远超 0.3 m 的阈值）。
#[test]
fn a_late_tick_compares_against_the_step_that_matches_the_elapsed_time() {
    let rig = Rig::new();
    rig.observe(&[
        ("nearest_obstacle_m", json!(2.0)),
        ("gravity_proj_z", json!(-0.98)),
    ]);
    rig.started_as("p1", json!({"vx": 5.0, "duration_s": 2.0}));
    rig.tick(); // 预测：+0.1 s → 1.5 m，+0.2 s → 1.0 m，+0.3 s → 0.5 m，+0.4 s → 0.0 m
    for _ in 0..3 {
        rig.advance();
    }
    // 三个周期之后才醒，而世界确实按预测走了 1.5 m。
    rig.observe(&[("nearest_obstacle_m", json!(0.5))]);
    rig.tick();
    assert!(
        rig.surprises().is_empty(),
        "世界和预测是一致的，误差是调度带来的: {:?}",
        rig.surprises()
    );

    // 再晚到预测窗口之外（+0.4 s 是最后一步）：这一拍干脆不对账。
    for _ in 0..6 {
        rig.advance();
    }
    rig.observe(&[("nearest_obstacle_m", json!(2.0))]);
    rig.tick();
    assert!(
        rig.surprises().is_empty(),
        "超出窗口就不对账: {:?}",
        rig.surprises()
    );
}
