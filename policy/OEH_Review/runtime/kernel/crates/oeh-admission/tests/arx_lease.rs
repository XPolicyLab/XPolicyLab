//! 设计文档 §4 的 S1 验收行 5：`finish(success)` 的租约在 8 分钟的 hold 里不过期。
//!
//! **这个文件验的是 `Admission` 的租约机制本身，不是出厂的那份 bundle。** 两件事情让它
//! 没法做成一次 profile 启动：
//!
//! 1. `crates/oeh-kernel/src/builtin/clock.rs` 把 `MonotonicClock` 写死在插件里，而且
//!    这个插件不收任何配置——没有任何 profile 行、patch 键或环境变量能换成 `ManualClock`。
//!    `crates/oeh-admission/tests/admission.rs` 的
//!    `advancing_the_injected_clock_past_the_lease_expires_it_at_once` 早就记过这件事，
//!    它同样是手工搭零件。
//! 2. 真实时间的版本要等满 8 分钟墙钟，而 `scripts/ci.sh` 给整段 `cargo test --workspace`
//!    的硬上限是 `timeout 300`。
//!
//! bundle 层面的那条保证由另一处给出：`python/oeh-vla/tests/test_arx_s1_integration.py`
//! 里 `ArxMotionPlugin` 的后台心跳任务（`_hold_lease`，0.25 s 一次）在一次真实的
//! `arx_x5_finish` hold 里持续发 `capability/heartbeat`——那条是"插件真的会心跳"，这条
//! 是"心跳真的能把租约续住"。两条合起来才是验收行 5。
//!
//! 时钟从 1 µs 起而不是 0：`sweep_leases` 拿 `last_heartbeat_ns == 0` 当"一次心跳都没
//! 有过"的哨兵，起点为 0 的话两种情形分不开。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oeh_admission as _;
use oeh_admission::admission::{Admission, AdmissionConfig};
use oeh_admission::capability::{
    CancelReason, ExecCtx, ExecOutcome, ExecutionCall, Executor, HeartbeatHandle,
};
use oeh_admission::verdict::{Rule, RuleCtx, Verdict};
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::proposal::{BoxFuture, Outcome, Proposal, ProposalId, Proposer};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{ManualClock, WorldLog};
use serde_json::{json, Value};

/// `arx_x5_finish` 的租约：合约 §1 的 `lease_ttl_s`。
const TTL_S: f64 = 5.0;
/// 插件 `_hold_lease` 的心跳周期（`ArxMotionPlugin.heartbeat_interval_s`）。
const BEAT_S: f64 = 0.25;
/// 模拟的 hold 时长。RPent 的证据把一次 ~1000 步的 hold 放在 ~7.5 min 墙钟上，
/// `ArxPrimitives.finish_wait_s` 的默认值 900 s 就是按这个定的。
const HOLD_S: f64 = 480.0;
const GRACE_MS: u64 = 500;

/// 一条永远放行的规则：这个文件验的是租约，不是判定。
struct AllowAll;
impl Rule for AllowAll {
    fn name(&self) -> &str {
        "allow-all"
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, incoming: Verdict) -> Verdict {
        incoming
    }
}

/// 一直跑到被撤销为止的执行器，并把心跳句柄交给测试。
///
/// 心跳由**测试**按模拟时间盖戳，而不是由执行器 future 自己 `sleep`：这里的时间是
/// `ManualClock`，`tokio::time::sleep` 跟它毫无关系，让 future 自己 sleep 只会把这个
/// 用例变成一次真实的 8 分钟等待。`heartbeat_handle()` 正是为"盖戳的人不是 future
/// 本身"这种情形准备的（跨进程执行器的心跳来自连接的读循环）。
#[derive(Default)]
struct HeldByHeartbeat {
    handle: Mutex<Option<HeartbeatHandle>>,
    saw_cancel: Arc<AtomicBool>,
    started: Arc<AtomicU64>,
}

impl HeldByHeartbeat {
    /// 盖一次心跳戳，返回是否盖成了（执行器还没起跑时是 false）。
    fn beat(&self) -> bool {
        match self.handle.lock().unwrap().as_ref() {
            Some(h) => {
                h.beat();
                true
            }
            None => false,
        }
    }
}

impl Executor for HeldByHeartbeat {
    fn execute(&self, _c: ExecutionCall, mut ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        *self.handle.lock().unwrap() = Some(ctx.heartbeat_handle());
        self.started.fetch_add(1, Ordering::SeqCst);
        let saw = self.saw_cancel.clone();
        Box::pin(async move {
            ctx.cancelled().await;
            saw.store(true, Ordering::SeqCst);
            ExecOutcome {
                ok: true,
                result: json!({"_finish": true}),
                error: None,
            }
        })
    }
}

/// 手工搭起来的准入：`ManualClock` + 一条 AllowAll + 一个 `arx_x5_finish` 能力。
struct Parts {
    clock: Arc<ManualClock>,
    adm: Arc<Admission>,
    log: Arc<WorldLog>,
    exec: Arc<HeldByHeartbeat>,
    _tmp: tempfile::TempDir,
    _rule: oeh_kernel::Disposer,
    _cap: oeh_kernel::Disposer,
}

fn parts() -> Parts {
    let tmp = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1_000));
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let fsm = Arc::new(RobotFsm::new(log.clone(), state.clone()));
    fsm.transition(FsmState::Standing, "test", "test").unwrap();
    let adm = Admission::new(
        clock.clone(),
        log.clone(),
        state,
        fsm,
        AdmissionConfig {
            // 定时扫描形同虚设（一分钟一次）：这个用例自己按快门。
            lease_check_ms: 60_000,
            lease_grace_ms: GRACE_MS,
            recheck_ms: 0,
            // 默认是 `/tmp/oeh_estop`——开发机上真有那个文件的话，这个提案会被
            // `rule-estop` 之外的急停事实拦下，而这个用例只该看见自己造的世界。
            estop_flag: tmp.path().join("estop"),
            ..Default::default()
        },
    )
    .unwrap();
    let rule = adm.register_rule(Arc::new(AllowAll));
    let exec = Arc::new(HeldByHeartbeat::default());
    let cap = adm
        .capabilities()
        .register(
            // 合约 §1：`arx_x5_finish` 是 control / low，`lease_ttl_s` 5.0。
            serde_json::from_value(json!({
                "name": "arx_x5_finish", "class": "control", "risk_level": "low",
                "params_schema": {"type": "object", "additionalProperties": false,
                    "required": ["status", "summary"], "properties": {
                        "status": {"type": "string", "enum": ["success", "failure"]},
                        "summary": {"type": "string", "minLength": 1, "maxLength": 4096}}},
                "lease_ttl_s": TTL_S
            }))
            .unwrap(),
            exec.clone() as Arc<dyn Executor>,
        )
        .unwrap();
    Parts {
        clock,
        adm,
        log,
        exec,
        _tmp: tmp,
        _rule: rule,
        _cap: cap,
    }
}

fn finish_proposal(id: &str) -> Proposal {
    Proposal {
        proposal_id: ProposalId(id.into()),
        brain_id: "slow".into(),
        turn: 1,
        trace_id: format!("t-{id}"),
        capability: "arx_x5_finish".into(),
        params: json!({"status": "success", "summary": "holding to native termination"}),
    }
}

fn count_kind(log: &WorldLog, kind: &str) -> usize {
    log.since(0)
        .unwrap()
        .iter()
        .filter(|e| e.kind.as_str() == kind)
        .count()
}

/// 等条件成立（5 ms × 200 = 1 s 上限）。这里等的全是"某个任务已经跑到某一步"，
/// 和 `ManualClock` 上的模拟时间无关。
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    for _ in 0..200 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

// ================================================================

/// 心跳不断，模拟时间走过 8 分钟：一条 `act/lease_expired` 都没有，执行依然登记着。
///
/// 这是 `finish(success)` 那段后台 hold 的租约保证：TTL 是 5 s，hold 是 480 s，也就是
/// 96 个 TTL——没有续租的话它早该在第一个 TTL 就被摘掉了。
#[tokio::test]
async fn a_heartbeating_finish_hold_keeps_its_lease_alive_across_eight_simulated_minutes() {
    let p = parts();
    let adm = p.adm.clone();
    let running = tokio::spawn(async move { adm.propose(finish_proposal("hold")).await });

    // 同步点：执行真的起跑了（`act/started` 已经落日志），租约从这一刻算起。
    wait_until("finish 起跑", || {
        count_kind(&p.log, "act/started") == 1 && p.exec.started.load(Ordering::SeqCst) == 1
    })
    .await;

    // 0.25 s 一拍，走满 480 s，每一拍都按一次租约扫描的快门。生产路径上那个快门由
    // `lease_check_ms` 的定时任务调同一个函数。
    let beats = (HOLD_S / BEAT_S) as u64;
    for _ in 0..beats {
        p.clock.advance((BEAT_S * 1e9) as u64);
        assert!(p.exec.beat(), "执行器起跑之后心跳句柄必须在");
        p.adm.sweep_leases_now().await;
    }

    assert_eq!(
        count_kind(&p.log, "act/lease_expired"),
        0,
        "续着租的 hold 不该过期；日志: {:?}",
        p.log
            .since(0)
            .unwrap()
            .iter()
            .map(|e| e.kind.as_str().to_string())
            .collect::<Vec<_>>()
    );
    assert!(
        !p.exec.saw_cancel.load(Ordering::SeqCst),
        "执行器不该看见取消"
    );
    assert!(!running.is_finished(), "hold 还在跑");

    // 收尾：撤销掉，别把这个任务留在运行时里。
    assert!(p
        .adm
        .cancel_exec(&ProposalId("hold".into()), CancelReason::Stop));
    let out = running.await.unwrap();
    assert!(matches!(out, Outcome::Cancelled { .. }), "{out:?}");
}

/// 对照组：同一个 hold，停止心跳，三个 TTL 之后租约过期。
///
/// 没有这一半，上一条用例在"扫描根本没生效"的世界里也会通过。
#[tokio::test]
async fn a_finish_hold_that_stops_heartbeating_does_lose_its_lease() {
    let p = parts();
    let adm = p.adm.clone();
    let running = tokio::spawn(async move { adm.propose(finish_proposal("silent")).await });
    wait_until("finish 起跑", || {
        count_kind(&p.log, "act/started") == 1 && p.exec.started.load(Ordering::SeqCst) == 1
    })
    .await;

    // 先盖一戳，让"有过心跳"这件事成立（`last_heartbeat_ns == 0` 是另一条路）。
    assert!(p.exec.beat());
    // 然后闭嘴：时间一步跨过三个 TTL 加宽限期。
    p.clock
        .advance((3.0 * TTL_S * 1e9) as u64 + GRACE_MS * 1_000_000);
    p.adm.sweep_leases_now().await;

    assert_eq!(count_kind(&p.log, "act/lease_expired"), 1);
    let out = running.await.unwrap();
    assert_eq!(
        out,
        Outcome::Cancelled {
            reason: "lease expired".into()
        }
    );
    assert!(p.exec.saw_cancel.load(Ordering::SeqCst));
}
