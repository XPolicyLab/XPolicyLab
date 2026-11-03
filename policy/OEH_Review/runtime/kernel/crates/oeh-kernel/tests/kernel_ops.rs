use std::sync::Arc;

use oeh_kernel::fiber::FiberState;
use oeh_kernel::op::KernelHandle;
use oeh_kernel::plugin::{Ctx, Plugin, PluginError};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{Kernel, KernelConfig, KernelSendError};
use oeh_proto::{EventDraft, EventKind, Manifest};
use serde_json::Value;

struct Needs(&'static str);
impl Plugin for Needs {
    fn manifest(&self) -> Manifest {
        Manifest::new("needs").inject(&[self.0])
    }
    fn apply(&self, _ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        Ok(())
    }
}
struct Gives(&'static str);
impl Plugin for Gives {
    fn manifest(&self) -> Manifest {
        Manifest::new("gives").provide(&[self.0])
    }
    fn apply(&self, ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        ctx.provide(self.0, Arc::new(1u8))
    }
}

fn resolver(id: &str) -> Option<Box<dyn Plugin>> {
    match id {
        "clock" => oeh_kernel::find_factory("clock").map(|f| (f.build)()),
        "world-log" => oeh_kernel::find_factory("world-log").map(|f| (f.build)()),
        "needs-x" => Some(Box::new(Needs("x"))),
        _ => None,
    }
}

#[tokio::test]
async fn add_fiber_settles_dependents_and_emits_plugin_state() {
    let rows = vec![
        EntryRow::new("clock"),
        EntryRow::new("world-log"),
        EntryRow::new("needs-x"),
    ];
    let kernel = Kernel::boot_with(rows, &resolver).unwrap();
    assert_eq!(kernel.states()[2].1, FiberState::Pending);
    let log = kernel
        .registry()
        .get::<WorldLog>(&"world-log".into())
        .unwrap();
    let kh = kernel
        .registry()
        .get::<KernelHandle>(&"kernel".into())
        .unwrap();
    let run = tokio::spawn(kernel.run());

    let added = kh
        .add_fiber(EntryRow::new("gives-x"), Box::new(Gives("x")))
        .unwrap()
        .await
        .unwrap();
    assert_eq!(added.state, FiberState::Active);
    // 依赖 x 的 needs-x 在同一次 settle 里被激活
    let evs = log.since(0).unwrap();
    let states: Vec<(String, String)> = evs
        .iter()
        .filter(|e| e.kind.as_str() == "plugin/state")
        .map(|e| {
            (
                e.payload["plugin"].as_str().unwrap().to_string(),
                e.payload["state"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert!(
        states.contains(&("needs-x".into(), "pending".into())),
        "{states:?}"
    );
    assert!(
        states.contains(&("needs-x".into(), "active".into())),
        "{states:?}"
    );
    assert!(
        states.contains(&("gives-x".into(), "active".into())),
        "{states:?}"
    );

    kh.dispose_fiber(added.id).unwrap().await.unwrap();
    let evs = log.since(0).unwrap();
    let last_needs = evs
        .iter()
        .rfind(|e| e.kind.as_str() == "plugin/state" && e.payload["plugin"] == "needs-x")
        .unwrap();
    assert_eq!(last_needs.payload["state"], "pending");
    let last_gives = evs
        .iter()
        .rfind(|e| e.kind.as_str() == "plugin/state" && e.payload["plugin"] == "gives-x")
        .unwrap();
    assert_eq!(last_gives.payload["state"], "disposed");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

#[tokio::test]
async fn kernel_service_is_available_to_plugins_before_rows_load() {
    let kernel = Kernel::boot_with(vec![EntryRow::new("clock")], &resolver).unwrap();
    assert!(kernel.registry().available(&"kernel".into()));
    let kh = kernel
        .registry()
        .get::<KernelHandle>(&"kernel".into())
        .unwrap();
    let run = tokio::spawn(kernel.run());
    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// 撤掉的 Fiber 不会无限堆在树上——但最近的那些还在。
///
/// `dispose` 是终态、行留着，这是对的（刚掉的那个还要能被问到）。问题是有了跨进程插件
/// 之后，"加一行"成了客户端能随意重复的动作：注册→断连→再来，每轮留一行，而
/// `settle()` 和 `states_detailed()` 每次都要走一遍整个 `Vec`——第 N 轮的代价是 Θ(N)，
/// 全落在同时还要排帧 ingest 的那条内核任务上。
///
/// 这条测试同时钉住两头：上限（`MAX_DISPOSED_ROWS`）和"刚撤掉的那个必须还在"。
#[tokio::test]
async fn disposed_rows_are_capped_but_the_latest_survives() {
    let kernel = Kernel::boot_with(vec![EntryRow::new("clock")], &resolver).unwrap();
    let kh = kernel
        .registry()
        .get::<KernelHandle>(&"kernel".into())
        .unwrap();
    let run = tokio::spawn(kernel.run());

    let churn = oeh_kernel::fiber::FiberTree::MAX_DISPOSED_ROWS + 8;
    let mut last_id = 0;
    for i in 0..churn {
        let added = kh
            .add_fiber(
                EntryRow::new("gives-x").with_name(format!("g{i}")),
                Box::new(Gives("x")),
            )
            .unwrap()
            .await
            .unwrap();
        assert_eq!(added.state, FiberState::Active, "cycle {i}");
        last_id = added.id.0;
        kh.dispose_fiber(added.id).unwrap().await.unwrap();
    }

    // 剪枝之后树还是好的，而且 id 从不回头：一个被剪掉的 id 要是被重新发出去，一条
    // 迟到的 `plugin/state` 就会被认成新 Fiber 的状态。`FiberAdded` 的回执本身也是
    // 从 `states_detailed()` 里找出来的，所以这一下同时证明那条查找没被剪坏。
    let probe = kh
        .add_fiber(
            EntryRow::new("gives-x").with_name("probe"),
            Box::new(Gives("x")),
        )
        .unwrap()
        .await
        .unwrap();
    assert_eq!(probe.state, FiberState::Active);
    assert_eq!(probe.id.0, last_id + 1, "ids must stay monotonic");

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}

/// Op 通道有界：内核来不及处理时，生产者拿到的是 `Busy`，而不是一条无上限增长的队列。
/// 丢掉的那些由内核每 `backpressure_report_ms` 汇总成一条 `health/op_backpressure`。
///
/// 内核**还没 `run`**：队列只进不出，`op_queue` 之后的每一条都必须被拒。这是这条路径
/// 唯一不靠时序就能造出来的过载形状。
#[tokio::test]
async fn a_full_op_queue_reports_busy_and_logs_backpressure() {
    let kernel = Kernel::boot_with_config(
        vec![EntryRow::new("clock"), EntryRow::new("world-log")],
        &resolver,
        KernelConfig {
            op_queue: 4096,
            // 50 ms 而不是默认的 1 s：汇总周期是参数，测试不必真的等一秒。
            backpressure_report_ms: 50,
        },
    )
    .unwrap();
    let kh = kernel
        .registry()
        .get::<KernelHandle>(&"kernel".into())
        .unwrap();
    let log = kernel
        .registry()
        .get::<WorldLog>(&"world-log".into())
        .unwrap();
    let mut busy = 0u64;
    for _ in 0..5000 {
        if let Err(KernelSendError::Busy) = kh.emit(EventDraft::new(
            EventKind::new("obs/test").unwrap(),
            "t",
            serde_json::json!({}),
        )) {
            busy += 1;
        }
    }
    assert!(busy >= 900, "{busy}");
    assert_eq!(kh.dropped(), busy);

    let run = tokio::spawn(kernel.run());
    // 有界重试，不是固定的 settle 窗口：条件成立就走，最多等 2 s。游标跟着走——排空
    // 这 4096 条 `obs/test` 会把世界日志的保留窗口整个翻过一遍，`since(0)` 从中途起
    // 就是 `CursorTooOld` 了。
    let mut reported = None;
    let mut cursor = 0u64;
    for _ in 0..100 {
        let batch = match log.since(cursor) {
            Ok(b) => b,
            Err(too_old) => log
                .since(too_old.oldest_retained.saturating_sub(1))
                .unwrap_or_default(),
        };
        if let Some(last) = batch.last() {
            cursor = last.seq;
        }
        reported = batch
            .into_iter()
            .find(|e| e.kind.as_str() == "health/op_backpressure");
        if reported.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let ev = reported.expect("the kernel must report the dropped Ops on the world log");
    assert!(
        ev.payload["dropped"].as_u64().unwrap() >= 900,
        "{}",
        ev.payload
    );
    // 汇总之后归零：同一批丢弃不会每个周期再记一遍。
    assert_eq!(kh.dropped(), 0);

    kh.shutdown().unwrap();
    run.await.unwrap().unwrap();
}
