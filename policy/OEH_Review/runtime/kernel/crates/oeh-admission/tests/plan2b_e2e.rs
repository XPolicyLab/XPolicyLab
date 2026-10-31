//! Plan 2b 端到端：mock 脑在 `dev-mock` 里提案 `walk`，准入链把它拒下来或者放行。
//!
//! 两个场景跑的是同一份真实 profile（`profiles/dev-mock.yaml` → `bundles/dev-mock.yaml`：
//! 合成 lowstate 流 → 蒸馏器 → 准入链八条规则 → mock 执行器 → 按 `wake/fast.yaml` 被唤醒
//! 的 mock 脑），差别只在"世界给了什么":
//!
//! - 场景 1：没有任何 `obs/detections` → `distill-ground` 没写 `clear_path` → `rule-scene`
//!   不知道地面是不是空的 → Deny → `act/refused{stage:"rules"}`，脑拿到 `refused`。
//! - 场景 2：先喂一条"路是空的"的检测，再用一条 `obs/gesture` 把脑叫醒 → 八条规则全过
//!   → `act/clamped`（`vx` 0.5 被 `rule-clamp` 夹成 0.3）→ `act/admitted` → `act/started`
//!   → `act/result{ok:true}`，FSM 走完 `ENGAGED → ACTING → STANDING`，脑拿到 `executed`。
//!
//! **时序全部靠有界轮询**（[`until`]：20 ms 一次，最多 2 s），等的是条件，不是一段猜出来
//! 的时间。以前这个文件用固定 sleep 卡时序，于是场景 1 在机器忙的时候会红：`mock-brain`
//! 那一行 attach 时游标从"此刻最后一条"起算，而 500 Hz 的 lowstate 流在开机后 ~2 ms 就能
//! 把 `distill-fsm` 推出一条 `safety/fsm{STANDING}`——它要是抢在 `brain/attach` 前面落进
//! 日志，那个脑就永远等不到自己唯一的唤醒证据，150 ms 之后测试拿到一个空的事件列表。
//! 现在唤醒由测试自己发一条 `obs/gesture` 给出（`wake/fast.yaml` 里它是 `wake`），开机
//! 那条 `safety/fsm` 抢没抢到都不影响结论。
//!
//! 剩下唯一一段真实时间是 `wake/fast.yaml` 的 500 ms 最小唤醒间隔——那是**被测系统**的
//! 行为，不是测试的等待策略，所以场景 2 等的是"预算恢复了"这个条件（拿内核时钟和日志里
//! 最后一次唤醒算），不是 `sleep(600)`。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use oeh_admission as _;
use oeh_distill as _;
use oeh_kernel::profile::{load_profile, EntryRow};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::world_log::{read_segments, resolve_log_dir};
use oeh_kernel::{ClockService, Kernel, KernelHandle, WorldLog};
use oeh_proto::{EventDraft, EventKind, ServiceName, WorldEvent};
use serde_json::{json, Value};

// ---------------------------------------------------------------- 脚手架

/// dev-mock 的行，但世界日志、会话目录和急停标志都挪进 `tmp`。
///
/// `world-log.dir` / `brains.session_dir` 走 patch 文件（和 `plan2a_e2e.rs` 一样）。
/// 两个急停标志路径（`distill-estop.flag_path` 与 `admission.estop_flag`）默认都指向
/// `/tmp/oeh_estop`：开发机上真要有这么一个文件，场景 2 会被 `rule-estop` 拦下、脑还会
/// 被一条 `safety/estop` 打断——测试只该看见自己造的世界。两条**必须一起**改成同一个
/// 路径：只改一个的话 `admission` 会在 boot 当场 fail-closed（`estop flag mismatch`，
/// 见 `oeh_admission::builtin::admission`）。`admission` 那一行只改这一个键（而不是整行
/// patch 掉）：`required_rules` 等其余配置必须原样来自 bundle，否则这个测试就验不到
/// dev-mock 真正的规则集了。
///
/// 仓库里的几条路径（profile、bundles、`mock-brain` 的 `wake`）一律按**绝对路径**传。
/// 这里曾经是一次 `set_current_dir(root)`：进程的 cwd 是全局的，而同一个测试二进制里的
/// 测试是并发跑的——一个测试换掉 cwd，另一个测试正在解析的相对路径就跟着换了意思。
fn dev_mock_rows(tmp: &Path) -> Vec<EntryRow> {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let patch = tmp.join("patch.yaml");
    std::fs::write(
        &patch,
        format!(
            "- {{ id: world-log, config: {{ dir: {0}/logs }} }}\n\
             - {{ id: brains, config: {{ session_dir: {0}/sessions }} }}\n\
             - {{ id: distill-estop, config: {{ flag_path: {0}/estop }} }}\n",
            tmp.display()
        ),
    )
    .unwrap();
    let mut rows = load_profile(
        &root.join("profiles/dev-mock.yaml"),
        &root.join("bundles"),
        &[PathBuf::from(&patch)],
    )
    .unwrap();
    for r in &mut rows {
        match r.id.as_str() {
            // `distill-estop` 那一条由 patch 文件改；这一条必须和它指着同一个文件。
            "admission" => r.config["estop_flag"] = json!(tmp.join("estop")),
            // 唤醒策略由 `mock-brain` 自己按路径读，bundle 里写的是仓库相对路径。
            "mock-brain" => r.config["wake"] = json!(root.join("wake/fast.yaml")),
            _ => {}
        }
    }
    rows
}

/// 跑起来的 dev-mock：句柄、`run()` 的 JoinHandle，以及三个用来等条件的服务句柄。
struct Live {
    h: KernelHandle,
    run: tokio::task::JoinHandle<Result<(), oeh_kernel::BootError>>,
    log: Arc<WorldLog>,
    state: Arc<StateRegistry>,
    clock: Arc<ClockService>,
}

/// 启动 dev-mock。boot 报告必须是干净的。
fn boot(tmp: &Path) -> Live {
    let kernel = Kernel::boot(dev_mock_rows(tmp)).unwrap();
    assert!(kernel.report().failed.is_empty(), "{:?}", kernel.report());
    assert!(kernel.report().pending.is_empty(), "{:?}", kernel.report());
    let reg = kernel.registry().clone();
    let log = reg
        .get::<WorldLog>(&ServiceName::from("world-log"))
        .unwrap();
    let state = reg
        .get::<StateRegistry>(&ServiceName::from("state"))
        .unwrap();
    let clock = reg
        .get::<ClockService>(&ServiceName::from("clock"))
        .unwrap();
    let h = kernel.handle();
    Live {
        h,
        run: tokio::spawn(kernel.run()),
        log,
        state,
        clock,
    }
}

/// 有界轮询：20 ms 一次，最多 2 s，等 `cond` 变成真。超时就带着日志里的 kind 列表炸掉。
///
/// 判别信号永远是**条件成立**，不是"睡够了"。20 ms 只是查看的节拍。
async fn until(live: &Live, what: &str, cond: impl Fn(&Live) -> bool) {
    for _ in 0..100 {
        if cond(live) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let evs = live.log.since(0).unwrap_or_default();
    panic!("等 {what} 超过 2 s；世界日志: {:?}", kinds(&evs));
}

/// 世界日志里有没有满足 `pred` 的事件。
fn seen(live: &Live, pred: impl Fn(&WorldEvent) -> bool) -> bool {
    live.log.since(0).unwrap_or_default().iter().any(pred)
}

/// 结局已经回到脑手里了：世界日志里有一条 `brain/mock_outcome`，而且它和 `act/<kind>`
/// 是同一个 trace。
///
/// 为什么这要单独等一下。`brain/mock_outcome` 不是准入写的，是 `mock-brain` 在
/// `turn.propose(...)` 落定之后、在一个**独立的 tokio 任务**里写的
/// （`crates/oeh-kernel/src/builtin/mock_brain.rs:72-88`）。`act/refused` / `act/result`
/// 一落地这个测试就 `shutdown()`，而那一下会把还没跑到 `log.append` 的那个任务连同它
/// 要写的事件一起收走——于是日志里有 `brain/turn_end`（收尾是 brains 自己做的）却没有
/// `brain/mock_outcome`，下面那条 `unwrap` 就地炸掉。机器闲的时候那个任务总是先跑完，
/// 所以这条竞态只在整个 workspace 一起跑、调度不那么客气的时候才现身。
fn outcome_back_at_the_brain(live: &Live, act_kind: &str) -> bool {
    let evs = live.log.since(0).unwrap_or_default();
    let Some(trace) = evs
        .iter()
        .find(|e| e.kind.as_str() == act_kind)
        .and_then(|e| e.trace_id.clone())
    else {
        return false;
    };
    evs.iter().any(|e| {
        e.kind.as_str() == "brain/mock_outcome" && e.trace_id.as_deref() == Some(trace.as_str())
    })
}

/// 准入链要读的两份事实都到位了：`rule-watchdog` 要 lowstate 的帧时间（`distill-watchdog`
/// 的第一个 tick 写的），`rule-fsm` 要 motion 允许的状态（`distill-fsm` 从第一帧推出来的
/// STANDING）。两件事都有对应的世界事件，不用猜时间。
fn data_plane_ready(live: &Live) -> bool {
    seen(live, |e| {
        e.kind.as_str() == "health/stale"
            && e.payload["stream"] == "lowstate"
            && e.payload["stale"] == false
    }) && seen(live, |e| {
        e.kind.as_str() == "safety/fsm" && e.payload["to"] == "STANDING"
    })
}

/// `wake/fast.yaml` 的 `min_interval_ms: 500` 已经满足：日志里最后一次**真的开出 turn**
/// 的唤醒距今 ≥ 500 ms（没有过就无条件满足）。用内核自己的时钟算，不是 `sleep(600)`。
///
/// 这只是预算的一半——另一半 `max_consecutive_wakes: 3` 的计数器只在 `finish_turn` 里、
/// 且间隔已经过去时才归零，测试这边重建那套记账既啰嗦又容易建错。所以不去建：
/// [`wake_until_turn`] 直接看结果（那条 gesture 的 `brain/wake` 是不是 suppressed），
/// 被抑制就再发一条。
fn wake_budget_ready(live: &Live) -> bool {
    let last = live
        .log
        .since(0)
        .unwrap_or_default()
        .iter()
        .filter(|e| e.kind.as_str() == "brain/wake" && e.payload["suppressed"] == false)
        .map(|e| e.t_kernel)
        .max();
    match last {
        None => true,
        Some(t) => live.clock.0.now_ns().saturating_sub(t) >= 500_000_000,
    }
}

/// 发 `obs/gesture` 把脑叫醒，直到它**真的**开出一个 turn。
///
/// 每条 gesture 的判定都会在日志里留下一条 `brain/wake{kind:"obs/gesture", suppressed:…}`
/// （`Plan::StartTurn` 与 `Plan::Suppressed` 都记），所以"这次叫醒成没成"是可以直接读出来
/// 的，不用猜。被抑制就等预算恢复再来一次；连着四次都被抑制说明预算模型出了别的问题，
/// 那时候炸掉比静静地超时有用得多。
async fn wake_until_turn(live: &Live) {
    for _ in 0..4 {
        until(live, "唤醒预算恢复", wake_budget_ready).await;
        let from = live.log.next_seq();
        emit(&live.h, "obs/gesture", json!({"gesture": "wave"}));
        until(live, "这条 gesture 的 brain/wake 判定", |l| {
            gesture_wake(l, from).is_some()
        })
        .await;
        if gesture_wake(live, from) == Some(false) {
            return;
        }
    }
    panic!("连发四条 obs/gesture 都被唤醒预算抑制了");
}

/// seq ≥ `from` 的第一条属于 `obs/gesture` 的 `brain/wake` 的 `suppressed` 值。
fn gesture_wake(live: &Live, from: u64) -> Option<bool> {
    live.log
        .since(0)
        .unwrap_or_default()
        .iter()
        .find(|e| {
            e.seq >= from && e.kind.as_str() == "brain/wake" && e.payload["kind"] == "obs/gesture"
        })
        .map(|e| e.payload["suppressed"] == true)
}

fn emit(h: &KernelHandle, kind: &str, payload: Value) {
    h.emit(EventDraft::new(
        EventKind::new(kind).unwrap(),
        "test",
        payload,
    ))
    .unwrap();
}

fn events(tmp: &Path) -> Vec<WorldEvent> {
    let run_dir = resolve_log_dir(&tmp.join("logs")).unwrap();
    read_segments(&run_dir).unwrap()
}

fn first<'a>(evs: &'a [WorldEvent], kind: &str) -> &'a WorldEvent {
    evs.iter()
        .find(|e| e.kind.as_str() == kind)
        .unwrap_or_else(|| panic!("no {kind} in {:?}", kinds(evs)))
}

/// 事件的 kind，按顺序。对 `Vec<WorldEvent>` 和 `Vec<&WorldEvent>` 都能用。
fn kinds<E: std::borrow::Borrow<WorldEvent>>(evs: &[E]) -> Vec<&str> {
    evs.iter().map(|e| e.borrow().kind.as_str()).collect()
}

/// 某个 trace 下的事件，按 seq。
fn trace_of<'a>(evs: &'a [WorldEvent], trace: &str) -> Vec<&'a WorldEvent> {
    evs.iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace))
        .collect()
}

/// 每一次 `act/proposed` 的 trace_id，按顺序。
fn proposal_traces(evs: &[WorldEvent]) -> Vec<String> {
    evs.iter()
        .filter(|e| e.kind.as_str() == "act/proposed")
        .map(|e| e.trace_id.clone().expect("act/* 必须带 trace_id"))
        .collect()
}

/// 规则 trace（`act/refused.trace` / `act/admitted.verdict_trace`）里某条规则那一步。
fn step<'a>(trace: &'a Value, rule: &str) -> &'a Value {
    trace
        .as_array()
        .unwrap_or_else(|| panic!("规则 trace 不是数组: {trace}"))
        .iter()
        .find(|t| t["rule"] == rule)
        .unwrap_or_else(|| panic!("规则 trace 里没有 {rule}: {trace}"))
}

// ---------------------------------------------------------------- 场景 1

/// 开机 → 数据面就位 → 一条 `obs/gesture` 把脑叫醒 → 提案 `walk`。此刻还没有任何检测，
/// `distill-ground` 一个寄存器都没写过，`rule-scene` 读不到 `clear_path` → Deny。脑在
/// `brain/mock_outcome` 里拿到 `refused`。
///
/// 唤醒**由测试给**：开机时 `distill-fsm` 那条 `safety/fsm{STANDING}` 也会唤醒这个脑，
/// 但它和 `brain/attach` 谁先谁后取决于机器忙不忙（attach 的游标从"此刻最后一条"起算，
/// 抢在前面的那条脑就看不见了）。自己发一条就跟那场竞赛无关了；万一 `safety/fsm` 确实
/// 抢先唤醒过一次，这条 gesture 要么被唤醒预算抑制、要么再开一个同样被拒的 turn——
/// 断言看的是**第一对** `act/proposed`/`act/refused`，两种情况下都成立。
///
/// `rule-scene` 那一步的 outgoing 是 deny：要么它自己给的（没有 `clear_path`），要么它
/// 原样传下上游的 Deny（规则只能收紧）。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_detections_the_scene_rule_refuses_the_walk() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(tmp.path());
    until(&live, "brain/attach", |l| {
        seen(l, |e| e.kind.as_str() == "brain/attach")
    })
    .await;
    until(&live, "数据面就位", data_plane_ready).await;
    emit(&live.h, "obs/gesture", json!({"gesture": "wave"}));
    until(&live, "act/refused", |l| {
        seen(l, |e| e.kind.as_str() == "act/refused")
    })
    .await;
    // 下面要读这个 trace 的 `brain/mock_outcome`，它比 `act/refused` 晚一步且在另一个
    // 任务里；不等它就关内核会把它截掉（见 [`outcome_back_at_the_brain`]）。
    until(&live, "brain/mock_outcome", |l| {
        outcome_back_at_the_brain(l, "act/refused")
    })
    .await;
    live.h.shutdown().unwrap();
    live.run.await.unwrap().unwrap();

    let evs = events(tmp.path());
    let proposed = first(&evs, "act/proposed");
    let trace = proposed.trace_id.clone().expect("act/* 必须带 trace_id");
    assert_eq!(proposed.payload["brain_id"], "fast");
    assert_eq!(proposed.payload["capability"], "walk");
    assert_eq!(
        proposed.payload["params"],
        json!({"vx": 0.5, "duration_s": 0.3})
    );

    let refused = first(&evs, "act/refused");
    assert_eq!(
        refused.trace_id.as_deref(),
        Some(trace.as_str()),
        "提案和拒绝是同一个 trace"
    );
    assert_eq!(refused.payload["stage"], "rules");
    let scene = step(&refused.payload["trace"], "rule-scene");
    assert_eq!(
        scene["outgoing"]["verdict"], "deny",
        "没有检测 → rule-scene 出来的判定是 deny: {scene}"
    );

    // 结局回到了脑手里。
    let outcome = first(&evs, "brain/mock_outcome");
    assert_eq!(outcome.payload["brain_id"], "fast");
    assert_eq!(outcome.payload["outcome"]["outcome"], "refused");
    // 被拒之后 turn 照常收尾，没有卡住的 turn。
    assert!(kinds(&evs).contains(&"brain/turn_end"), "{:?}", kinds(&evs));
    assert!(
        !kinds(&evs).contains(&"act/started"),
        "被拒的提案不该起跑: {:?}",
        kinds(&evs)
    );
}

// ---------------------------------------------------------------- 场景 2

/// 同一份 profile，但先告诉世界"路是空的"：`obs/detections` → `distill-ground` 写下
/// `clear_path` / 两个距离 → 唤醒预算恢复之后用一条 `obs/gesture` 把脑叫醒 → 八条规则
/// 全过 → `rule-clamp` 把 `vx` 从 0.5 夹到 0.3 → 执行 → FSM 走一圈回 STANDING。
///
/// 断言挂在**那次被执行的**提案的 trace 上（`act/result` 带的那个），不是"第二个提案"：
/// 开机那条 `safety/fsm` 有没有抢在 attach 前面决定了前面有没有一个被拒的 turn，而那件事
/// 和这个用例要验的东西无关。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clear_scene_admits_the_walk_with_clamped_params() {
    let tmp = tempfile::tempdir().unwrap();
    let live = boot(tmp.path());
    until(&live, "brain/attach", |l| {
        seen(l, |e| e.kind.as_str() == "brain/attach")
    })
    .await;
    until(&live, "数据面就位", data_plane_ready).await;
    emit(
        &live.h,
        "obs/detections",
        json!({
            "clear_path_raw": true,
            "nearest_person_m": 2.0,
            "nearest_obstacle_m": 2.0,
            "source": "test"
        }),
    );
    // `rule-scene` 读的是 `distill-ground` 写下的寄存器，不是那条事件本身——等的就是它。
    until(&live, "clear_path 寄存器", |l| {
        l.state.get("clear_path") == Some(json!(true))
    })
    .await;
    // 叫醒：唤醒预算（最小间隔 + 连续唤醒上限）由 `wake_until_turn` 按结果处理——被抑制
    // 就再来一条，而不是在这里重建内核的那套记账。
    wake_until_turn(&live).await;
    // walk 自带 duration_s 0.3；等的是它真的跑完，不是 700 ms。
    until(&live, "act/result", |l| {
        seen(l, |e| e.kind.as_str() == "act/result")
    })
    .await;
    // 同上：~第 500 行读的 `brain/mock_outcome` 是脑那一侧写的，和 `act/result` 不是
    // 同一个任务（见 [`outcome_back_at_the_brain`]）。
    until(&live, "brain/mock_outcome", |l| {
        outcome_back_at_the_brain(l, "act/result")
    })
    .await;
    live.h.shutdown().unwrap();
    live.run.await.unwrap().unwrap();

    let evs = events(tmp.path());
    let trace = first(&evs, "act/result")
        .trace_id
        .clone()
        .expect("act/* 必须带 trace_id");
    assert!(
        proposal_traces(&evs).contains(&trace),
        "被执行的那个 trace 也该有 act/proposed: {:?}",
        kinds(&evs)
    );
    let mine = trace_of(&evs, &trace);

    // 这个 turn 的准入链，按序（同一个 trace 下还有 brain/* 的 bookkeeping，这里只看
    // `act/*`）。`act/clamped` 在 `act/admitted` 之前：夹取发生在规则链跑完的那一刻，
    // 准入随后才拿夹过的 params 去判定（见 `admission::pipeline`）。
    let acts: Vec<&WorldEvent> = mine
        .iter()
        .copied()
        .filter(|e| e.kind.prefix() == "act")
        .collect();
    assert_eq!(
        kinds(&acts),
        vec![
            "act/proposed",
            "act/clamped",
            "act/admitted",
            "act/started",
            "act/result"
        ],
        "{:?}",
        mine.iter().map(|e| &e.payload).collect::<Vec<_>>()
    );

    let ev = |k: &str| *mine.iter().find(|e| e.kind.as_str() == k).unwrap();
    let clamped = ev("act/clamped");
    assert_eq!(clamped.payload["rule"], "rule-clamp");
    assert_eq!(
        clamped.payload["before"],
        json!({"vx": 0.5, "duration_s": 0.3})
    );
    assert_eq!(
        clamped.payload["after"],
        json!({"vx": 0.3, "duration_s": 0.3})
    );

    // 八条规则全在链上（少一条的话初始判定就是 Ask，会多出 act/ask）。
    let admitted = ev("act/admitted");
    let rules: Vec<&str> = admitted.payload["verdict_trace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["rule"].as_str().unwrap())
        .collect();
    assert_eq!(
        rules,
        vec![
            "rule-whitelist",
            "rule-fsm",
            "rule-run-mode",
            "rule-watchdog",
            "rule-pose",
            "rule-clamp",
            "rule-scene",
            "rule-estop"
        ]
    );
    assert_eq!(
        step(&admitted.payload["verdict_trace"], "rule-scene")["outgoing"]["verdict"],
        "allow"
    );

    // 执行器拿到的是夹过的 params。
    let started = ev("act/started");
    assert_eq!(
        started.payload["params"],
        json!({"vx": 0.3, "duration_s": 0.3})
    );
    let result = ev("act/result");
    assert_eq!(result.payload["ok"], true);

    // `frame_refs`：执行这一段的两条事件都指回它们当时看到的那一帧 lowstate。判定那
    // 一段（proposed/clamped/admitted）读的是蒸馏状态的快照，不指任何一帧，所以不带。
    let refs_of = |e: &WorldEvent| e.frame_refs.first().map(|r| r.stream_id.clone());
    assert_eq!(
        refs_of(started).as_deref(),
        Some("lowstate"),
        "act/started 该带 lowstate 的最新帧: {:?}",
        started.frame_refs
    );
    assert_eq!(
        refs_of(result).as_deref(),
        Some("lowstate"),
        "act/result 该带 lowstate 的最新帧: {:?}",
        result.frame_refs
    );
    assert!(
        result.frame_refs[0].frame_seq >= started.frame_refs[0].frame_seq,
        "收尾看到的帧不会比起跑时更旧: {:?} → {:?}",
        started.frame_refs,
        result.frame_refs
    );
    assert!(
        first(&evs, "act/admitted").frame_refs.is_empty(),
        "判定事件不带帧引用"
    );

    // FSM：准入把它推到 ACTING，执行完（`act/result` 之后）交还 STANDING。
    let fsm_after: Vec<(u64, &str)> = evs
        .iter()
        .filter(|e| e.kind.as_str() == "safety/fsm" && e.seq > admitted.seq)
        .map(|e| (e.seq, e.payload["to"].as_str().unwrap()))
        .take(3)
        .collect();
    assert_eq!(
        fsm_after.iter().map(|(_, to)| *to).collect::<Vec<_>>(),
        vec!["ENGAGED", "ACTING", "STANDING"],
        "整段 safety/fsm: {:?}",
        evs.iter()
            .filter(|e| e.kind.as_str() == "safety/fsm")
            .map(|e| e.payload["to"].clone())
            .collect::<Vec<_>>()
    );
    assert!(
        fsm_after.last().unwrap().0 > result.seq,
        "回到 STANDING 发生在 act/result 之后"
    );

    // 脑拿到的结局是 executed。`brain/mock_outcome` 和这一串 `act/*` 共用同一个
    // trace_id（§12），所以它就在 `mine` 里，不用绕道 turn 编号去找。
    let outcome = mine
        .iter()
        .find(|e| e.kind.as_str() == "brain/mock_outcome")
        .unwrap_or_else(|| panic!("trace {trace} 没有 brain/mock_outcome: {:?}", kinds(&mine)));
    assert_eq!(outcome.payload["outcome"]["outcome"], "executed");
    let turn = mine
        .iter()
        .find(|e| e.kind.as_str() == "brain/turn_start")
        .expect("这个 trace 的 turn_start")
        .payload["turn"]
        .clone();
    assert_eq!(outcome.payload["turn"], turn);
}
