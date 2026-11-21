use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use oeh_kernel::frame_store::{FrameError, FrameStore, MAX_RING_FRAMES};
use oeh_kernel::ManualClock;
use oeh_proto::{StreamSpec, Tier};

fn spec(id: &str, hz: f64) -> StreamSpec {
    StreamSpec {
        stream_id: id.into(),
        schema: "oeh.Mock".into(),
        nominal_hz: hz,
        tier: Tier::Small,
    }
}

#[test]
fn ingest_stamps_kernel_time_and_monotonic_seq() {
    let clock = Arc::new(ManualClock::new(100));
    let fs = FrameStore::new(clock.clone(), 1.0);
    let _reg = fs.register(spec("lowstate", 10.0)).unwrap();
    let f1 = fs
        .ingest("lowstate", Bytes::from_static(b"a"), Some(5))
        .unwrap();
    clock.advance(10);
    let f2 = fs
        .ingest("lowstate", Bytes::from_static(b"b"), None)
        .unwrap();
    assert_eq!((f1.frame_seq, f1.t_kernel, f1.t_source), (1, 100, Some(5)));
    assert_eq!((f2.frame_seq, f2.t_kernel), (2, 110));
    assert_eq!(
        fs.latest("lowstate").unwrap().payload,
        Bytes::from_static(b"b")
    );
    assert_eq!(
        fs.get(&f1.frame_ref()).unwrap().payload,
        Bytes::from_static(b"a")
    );
}

/// Plan 3 最终评审 C1：环容量是**声明**出来的（`nominal_hz` 来自 manifest 或插件配置），
/// 所以它必须有上限。`nominal_hz: 1e12 × window_s: 2` 会算出 2e12 帧，
/// `VecDeque::with_capacity` 于是去要 176 TB——分配失败走 `handle_alloc_error`，那是
/// abort，`FiberTree::activate` 的 `catch_unwind` 接不住，内核进程当场没（实测 SIGABRT）。
///
/// 边界协议那一侧另有一道 `max_nominal_hz` 护栏；这一道是给**所有**调用方的，包括进程内
/// 插件和将来任何一个直接用这个 `pub` API 的人。
#[test]
fn an_absurd_nominal_hz_cannot_ask_for_an_absurd_ring() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 2.0);
    let _reg = fs.register(spec("firehose", 1e12)).unwrap();
    assert_eq!(fs.stats("firehose").unwrap().capacity, MAX_RING_FRAMES);
    // 非有限值落在**下界**上，不是上界：`NaN as usize` 是 0，`clamp` 把它抬到 8。
    let _nan = fs.register(spec("nan", f64::NAN)).unwrap();
    assert_eq!(fs.stats("nan").unwrap().capacity, 8);
    // 正常的声明一点没变：500 Hz × 2 s。
    let _normal = fs.register(spec("lowstate", 500.0)).unwrap();
    assert_eq!(fs.stats("lowstate").unwrap().capacity, 1000);
}

#[test]
fn ring_evicts_oldest_but_latest_survives() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let _reg = fs.register(spec("s", 1.0)).unwrap(); // cap = clamp(8, MAX_RING_FRAMES) = 8
    let mut first = None;
    for i in 0..20u8 {
        let f = fs.ingest("s", Bytes::from(vec![i]), None).unwrap();
        if i == 0 {
            first = Some(f);
        }
    }
    let st = fs.stats("s").unwrap();
    assert_eq!((st.frames_total, st.buffered, st.capacity), (20, 8, 8));
    assert!(fs.get(&first.unwrap().frame_ref()).is_none(), "evicted");
    assert_eq!(fs.latest("s").unwrap().frame_seq, 20);
}

#[test]
fn unknown_and_duplicate_streams_are_errors() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    assert!(matches!(
        fs.ingest("x", Bytes::new(), None),
        Err(FrameError::UnknownStream(_))
    ));
    let _reg = fs.register(spec("x", 1.0)).unwrap();
    assert!(matches!(
        fs.register(spec("x", 1.0)),
        Err(FrameError::AlreadyRegistered(_))
    ));
}

#[test]
fn subscribers_run_synchronously_until_disposed() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let _reg = fs.register(spec("s", 1.0)).unwrap();
    let seen = Arc::new(AtomicU64::new(0));
    let s2 = seen.clone();
    let disposer = fs
        .subscribe(
            "s",
            Arc::new(move |f| s2.store(f.frame_seq, Ordering::SeqCst)),
        )
        .unwrap();
    fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    disposer();
    fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 1, "no delivery after dispose");
}

#[test]
fn subscriber_may_reenter_store() {
    // 订阅者回调里再读 latest 不能死锁
    let fs = Arc::new(FrameStore::new(Arc::new(ManualClock::new(0)), 1.0));
    let _reg = fs.register(spec("s", 1.0)).unwrap();
    let fs2 = fs.clone();
    let ok = Arc::new(AtomicU64::new(0));
    let ok2 = ok.clone();
    let _d = fs
        .subscribe(
            "s",
            Arc::new(move |_| {
                if fs2.latest("s").is_some() {
                    ok2.fetch_add(1, Ordering::SeqCst);
                }
            }),
        )
        .unwrap();
    fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(ok.load(Ordering::SeqCst), 1);
}

#[test]
fn subscribe_before_register_receives_frames_after_register() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let seen = Arc::new(AtomicU64::new(0));
    let s2 = seen.clone();
    let disposer = fs
        .subscribe(
            "later",
            Arc::new(move |f| s2.store(f.frame_seq, Ordering::SeqCst)),
        )
        .unwrap();
    let _reg = fs.register(spec("later", 1.0)).unwrap();
    fs.ingest("later", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    disposer();
    fs.ingest("later", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 1, "no delivery after dispose");
}

#[test]
fn disposing_pending_subscription_prevents_later_delivery() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let seen = Arc::new(AtomicU64::new(0));
    let s2 = seen.clone();
    let disposer = fs
        .subscribe(
            "later",
            Arc::new(move |_| {
                s2.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .unwrap();
    disposer();
    let _reg = fs.register(spec("later", 1.0)).unwrap();
    fs.ingest("later", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 0);
}

/// I5: 注册是可逆 Effect。撤销后流消失、订阅者回到 pending，重新注册自动重连。
#[test]
fn registration_is_reversible_and_reattaches_subscribers() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let seen = Arc::new(AtomicU64::new(0));
    let s2 = seen.clone();
    let _sub = fs
        .subscribe(
            "s",
            Arc::new(move |_| {
                s2.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .unwrap();

    let unregister = fs.register(spec("s", 1.0)).unwrap();
    assert_eq!(fs.streams().len(), 1);
    // 注册期间重复注册仍然是错误
    assert!(matches!(
        fs.register(spec("s", 1.0)),
        Err(FrameError::AlreadyRegistered(_))
    ));
    fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 1);

    unregister();
    assert!(fs.streams().is_empty(), "streams() drops the stream");
    assert!(matches!(
        fs.ingest("s", Bytes::new(), None),
        Err(FrameError::UnknownStream(_))
    ));

    // 重新注册成功，之前挂着的订阅者重新收到帧
    let _again = fs.register(spec("s", 1.0)).unwrap();
    assert_eq!(fs.streams().len(), 1);
    fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 2);
}

/// `schema_of` 是 `streams()` 的单流版本：蒸馏器每帧都要确认 schema 没被换过，
/// 不能为此克隆整张表。撤销注册后必须立刻变 `None`，重新注册后跟着新的 spec 走。
#[test]
fn schema_of_tracks_the_current_registration() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    assert_eq!(fs.schema_of("s"), None, "unregistered stream has no schema");
    let d = fs.register(spec("s", 1.0)).unwrap();
    assert_eq!(fs.schema_of("s"), Some("oeh.Mock".to_string()));
    assert_eq!(fs.schema_of("other"), None);
    d();
    assert_eq!(fs.schema_of("s"), None, "unregistered again");
    let _d2 = fs
        .register(StreamSpec {
            schema: "oeh.Other".into(),
            ..spec("s", 1.0)
        })
        .unwrap();
    assert_eq!(
        fs.schema_of("s"),
        Some("oeh.Other".to_string()),
        "follows the new registration"
    );
}

/// 撤销后留下的过期 disposer 不能拆掉后来的那次注册。
#[test]
fn stale_unregister_does_not_remove_a_newer_registration() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let first = fs.register(spec("s", 1.0)).unwrap();
    first();
    let _second = fs.register(spec("s", 1.0)).unwrap();
    let stale = fs.register(spec("s", 1.0));
    assert!(matches!(stale, Err(FrameError::AlreadyRegistered(_))));
    fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(fs.streams().len(), 1);
}

#[test]
fn frame_seq_continues_across_reregistration() {
    let fs = FrameStore::new(Arc::new(ManualClock::new(0)), 1.0);
    let d = fs.register(spec("s", 1.0)).unwrap();
    fs.ingest("s", Bytes::new(), None).unwrap();
    fs.ingest("s", Bytes::new(), None).unwrap();
    d();
    let _d2 = fs.register(spec("s", 1.0)).unwrap();
    let f = fs.ingest("s", Bytes::new(), None).unwrap();
    assert_eq!(
        f.frame_seq, 3,
        "frame_seq must not restart, or old FrameRefs alias new frames"
    );
}
