//! 帧通道的两个方向，各自对着一个**另外的** iceoryx2 Node 验（同一个进程里开两个 Node
//! 是允许的，共享内存那一层根本不在乎它们是不是同一个进程——这正是"跨进程"能在一个测试
//! 二进制里被如实模拟的原因）。
//!
//! - ingress：别人往 `oeh/<run_tag>/<stream_id>` 发样本 → 内核的 `FrameStore` 里长出帧；
//! - egress：`Frames::subscribe` 之后，内核里 `ingest` 进去的帧能被外面的订阅者读到。
//!
//! run 标签里带进程号（`t{pid}` / `e{pid}`）：这些服务名是**机器全局**的共享内存对象，
//! 两个并排跑的测试二进制（或者上一次跑剩下的东西）不该和这一次撞到同一条通道上。

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use oeh_kernel::clock::MonotonicClock;
use oeh_kernel::frame_store::FrameStore;
use oeh_kernel::WorldLog;
use oeh_proto::{StreamSpec, Tier};
use oeh_rpc::frames::{
    channel_name, schema_id, Frames, HISTORY, MAX_PUBLISHERS, MAX_SUBSCRIBERS, SUB_BUFFER,
};

fn spec(stream_id: &str, schema: &str, hz: f64) -> StreamSpec {
    StreamSpec {
        stream_id: stream_id.into(),
        schema: schema.into(),
        nominal_hz: hz,
        tier: Tier::Small,
    }
}

/// 扮演"远端适配器"的那一侧：自己的 Node、自己的 publisher，按内核那边约定好的通道名发。
fn publish_from_another_node(
    channel: String,
    payloads: Vec<Vec<u8>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        use iceoryx2::prelude::*;
        let node = NodeBuilder::new().create::<ipc::Service>().unwrap();
        let service = node
            .service_builder(&ServiceName::new(&channel).unwrap())
            .publish_subscribe::<[u8]>()
            .subscriber_max_buffer_size(SUB_BUFFER)
            .history_size(HISTORY)
            .max_publishers(MAX_PUBLISHERS)
            .max_subscribers(MAX_SUBSCRIBERS)
            .open_or_create()
            .unwrap();
        let ev = node
            .service_builder(&ServiceName::new(&format!("{channel}/ev")).unwrap())
            .event()
            .open_or_create()
            .unwrap();
        let publisher = service
            .publisher_builder()
            .initial_max_slice_len(1 << 16)
            .create()
            .unwrap();
        let notifier = ev.notifier_builder().create().unwrap();
        std::thread::sleep(Duration::from_millis(200)); // 让订阅者先起来
        for p in payloads {
            publisher
                .loan_slice_uninit(p.len())
                .unwrap()
                .write_from_slice(&p)
                .send()
                .unwrap();
            notifier.notify().unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(200));
    })
}

#[tokio::test]
async fn remote_frames_are_ingested_into_the_frame_store() {
    let store = Arc::new(FrameStore::new(Arc::new(MonotonicClock::new()), 2.0));
    let tag = format!("t{}", std::process::id());
    let _reg = store
        .register(StreamSpec {
            stream_id: "arm_state".into(),
            schema: "oeh.ArmState".into(),
            nominal_hz: 50.0,
            tier: Tier::Small,
        })
        .unwrap();
    let frames = Frames::new(tag.clone(), 1 << 20, store.clone());
    let dispose = frames.start_ingress("arm_state").unwrap();
    let h = publish_from_another_node(
        channel_name(&tag, "arm_state"),
        (0..20u8).map(|i| vec![i; 8]).collect(),
    );
    h.join().unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let latest = store.latest("arm_state").unwrap();
    assert_eq!(latest.payload, Bytes::from(vec![19u8; 8]));
    assert!(store.stats("arm_state").unwrap().frames_total >= 20);
    dispose();

    // `dispose()` 回来的时候那条线程必须**真的**没了（它 join 过），而不只是"被要求停下"。
    // 判据不是 join 本身（那只说明它返回了），是通道上后来的帧一个都不该再进来：插件撤了
    // 之后还在 ingest 的 ingress，会让一条已经从 `stream/list` 上消失的流继续长帧。
    let after_dispose = store.stats("arm_state").unwrap().frames_total;
    publish_from_another_node(
        channel_name(&tag, "arm_state"),
        (0..5u8).map(|i| vec![100 + i; 8]).collect(),
    )
    .join()
    .unwrap();
    assert_eq!(
        store.stats("arm_state").unwrap().frames_total,
        after_dispose,
        "the ingress thread is still running after its disposer returned"
    );
}

#[tokio::test]
async fn local_stream_is_published_on_subscribe_and_readable_by_another_node() {
    let store = Arc::new(FrameStore::new(Arc::new(MonotonicClock::new()), 2.0));
    let tag = format!("e{}", std::process::id());
    let _reg = store
        .register(StreamSpec {
            stream_id: "lowstate".into(),
            schema: "oeh.LowState".into(),
            nominal_hz: 500.0,
            tier: Tier::Small,
        })
        .unwrap();
    let frames = Frames::new(tag.clone(), 1 << 20, store.clone());
    let ch = frames.subscribe("lowstate", false).unwrap();
    assert_eq!(ch.channel, channel_name(&tag, "lowstate"));
    // 字面量，不是 `schema_id("oeh.LowState")`：拿函数去和它自己比，等于什么都没断言。
    assert_eq!(ch.schema_id, "oeh.LowState@json");
    assert_eq!(schema_id("oeh.LowState"), ch.schema_id);
    let got = Arc::new(parking_lot::Mutex::new(Vec::<Vec<u8>>::new()));
    let g2 = got.clone();
    let channel = ch.channel.clone();
    let reader = std::thread::spawn(move || {
        use iceoryx2::prelude::*;
        let node = NodeBuilder::new().create::<ipc::Service>().unwrap();
        let service = node
            .service_builder(&ServiceName::new(&channel).unwrap())
            .publish_subscribe::<[u8]>()
            .subscriber_max_buffer_size(SUB_BUFFER)
            .history_size(HISTORY)
            .max_publishers(MAX_PUBLISHERS)
            .max_subscribers(MAX_SUBSCRIBERS)
            .open_or_create()
            .unwrap();
        let sub = service.subscriber_builder().create().unwrap();
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(3) {
            while let Ok(Some(s)) = sub.receive() {
                g2.lock().push(s.payload().to_vec());
            }
            if g2.lock().len() >= 10 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    for i in 0..10u8 {
        store
            .ingest("lowstate", Bytes::from(vec![i; 4]), None)
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    reader.join().unwrap();
    let got = got.lock();
    assert!(got.len() >= 10, "{}", got.len());
    assert!(got.contains(&vec![9u8; 4]));
}

/// 一条流上 ingress 和 egress 互斥，而且这个判定发生在**锁里**，不是调用方传进来的那个
/// 提示上。
///
/// 两者共存就是一个自激回路：内核 egress 发出去的帧被自己的 ingress 收回来，`ingest` 再
/// 喂给 egress……只受 CPU 限制。窗口本来在"流已经注册、标记还没打上"之间，所以
/// `RemotePlugin::apply` 先打标再 register，而 `subscribe` 里那个 `remote_published` 参数
/// 被当成**过期的**信息对待——这条测试就是拿一个过期的 `false` 去撞它。
#[tokio::test]
async fn a_remote_stream_never_gets_an_egress_until_its_plugin_is_gone() {
    let store = Arc::new(FrameStore::new(Arc::new(MonotonicClock::new()), 2.0));
    let tag = format!("x{}", std::process::id());
    let frames = Frames::new(tag.clone(), 1 << 20, store.clone());

    // `RemotePlugin::apply` 的次序：`begin_remote` 在 register 之前，`commit` 在它之后。
    let mark = frames.begin_remote("arm_state");
    let unregister = store
        .register(spec("arm_state", "oeh.ArmState", 50.0))
        .unwrap();
    mark.commit();
    assert!(frames.is_remote("arm_state"));

    let ch = frames.subscribe("arm_state", false).unwrap();
    assert_eq!(ch.channel, channel_name(&tag, "arm_state"));
    assert!(
        !frames.has_egress("arm_state"),
        "a remotely published stream must never get an egress publisher"
    );
    // 没有回路：没人往通道上发，帧数就不动。
    let before = store.stats("arm_state").unwrap().frames_total;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(store.stats("arm_state").unwrap().frames_total, before);

    // 插件撤了（Effects 逆序：停 ingress → 注销流 → 交还标记），这条流回到"本地"。
    unregister();
    drop(mark);
    assert!(!frames.is_remote("arm_state"));

    // 同名的流由一个本地生产者重新注册：这一次 `stream/subscribe` 该起 egress 了。
    let _reg = store
        .register(spec("arm_state", "oeh.ArmState", 50.0))
        .unwrap();
    let again = frames
        .subscribe("arm_state", frames.is_remote("arm_state"))
        .unwrap();
    assert_eq!(again.channel, ch.channel);
    assert!(frames.has_egress("arm_state"));
}

/// `frames.max_bytes` 是**一帧**的上限：超过它的帧在进队列之前就被丢掉、计数，并且变成
/// 世界日志里的一条 `health/frame_egress_drop`。
///
/// 为什么用超大帧而不是灌满队列来验这条路径：它是确定性的（回调同步跑在 `ingest` 里），
/// 而"把 64 深的队列灌满"取决于发布线程当时排到没有。
#[tokio::test]
async fn oversize_frames_are_dropped_counted_and_reported_to_the_world_log() {
    let clock = Arc::new(MonotonicClock::new());
    let store = Arc::new(FrameStore::new(clock.clone(), 2.0));
    let log = Arc::new(WorldLog::new(clock));
    let tag = format!("d{}", std::process::id());
    let _reg = store.register(spec("wide", "oeh.Wide", 10.0)).unwrap();
    let frames = Frames::new_with_log(tag, 64, store.clone(), log.clone());
    frames.subscribe("wide", false).unwrap();
    assert!(frames.has_egress("wide"));

    store
        .ingest("wide", Bytes::from(vec![1u8; 16]), None)
        .unwrap();
    assert_eq!(frames.egress_dropped("wide"), 0);
    store
        .ingest("wide", Bytes::from(vec![2u8; 4096]), None)
        .unwrap();
    assert_eq!(frames.egress_dropped("wide"), 1);
    // 帧本身照常进 `FrameStore`（通道的上限不是内核的上限）。
    assert_eq!(store.stats("wide").unwrap().frames_total, 2);

    let events = log.since(0).unwrap();
    let ev = events
        .iter()
        .find(|e| e.kind.as_str() == "health/frame_egress_drop")
        .unwrap_or_else(|| panic!("no health/frame_egress_drop event: {events:?}"));
    assert_eq!(ev.payload["stream_id"], "wide");
    assert_eq!(ev.payload["dropped"], 1);
}

/// 标记的两个阶段：`begin_remote` 只挡住**新建**，一根汗毛都不碰已经存在的 egress；
/// 只有注册成功之后的 `commit` 才去扫。
///
/// 这是 R1：一次注册失败（跨进程插件声明的流名撞上了一个本地生产者）一上来就拆掉别人的
/// egress 的话，那个消费者手里的通道名从此永远安静，而且没有任何人会告诉它。
#[tokio::test]
async fn a_pending_remote_mark_never_touches_an_existing_egress() {
    let store = Arc::new(FrameStore::new(Arc::new(MonotonicClock::new()), 2.0));
    let tag = format!("p{}", std::process::id());
    let frames = Frames::new(tag, 1 << 20, store.clone());
    let _reg = store
        .register(spec("lowstate", "oeh.LowState", 50.0))
        .unwrap();
    frames.subscribe("lowstate", false).unwrap();
    assert!(frames.has_egress("lowstate"));

    {
        // 一个跨进程插件声明了同一个流名。注册（下一步）注定失败，但在那之前：
        let _pending = frames.begin_remote("lowstate");
        // 新的 egress 建不起来了（自激回路的那道闸门照样关着）……
        assert!(frames.is_remote("lowstate"));
        // ……而已经在工作的那一条原封不动。
        assert!(
            frames.has_egress("lowstate"),
            "begin_remote must not tear down an egress that belongs to somebody else"
        );
    }
    // 注册失败 → 凭据随栈消失 → 这条流从来没有变成过"远端的"。
    assert!(!frames.is_remote("lowstate"));
    assert!(frames.has_egress("lowstate"));

    // 对照：注册**成功**的那条路上，`commit` 才扫掉残留的 egress。
    let mark = frames.begin_remote("lowstate");
    assert!(frames.has_egress("lowstate"));
    mark.commit();
    assert!(!frames.has_egress("lowstate"));
}
