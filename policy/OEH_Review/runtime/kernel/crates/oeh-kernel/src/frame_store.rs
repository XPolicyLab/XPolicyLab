use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use bytes::Bytes;
use oeh_proto::{Frame, FrameRef, StreamSpec};
use parking_lot::Mutex;

use crate::clock::Clock;
use crate::plugin::Disposer;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FrameError {
    #[error("unknown stream `{0}`")]
    UnknownStream(String),
    #[error("stream `{0}` already registered")]
    AlreadyRegistered(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamStats {
    pub frames_total: u64,
    pub buffered: usize,
    pub capacity: usize,
}

type Subscriber = Arc<dyn Fn(&Frame) + Send + Sync>;

struct StreamBuf {
    /// 每次注册一个新 id：撤销只拆自己那次注册，过期的 disposer 碰不到后来的那次。
    reg_id: u64,
    spec: StreamSpec,
    ring: VecDeque<Frame>,
    capacity: usize,
    next_seq: u64,
    latest: Option<Frame>,
    subscribers: Vec<(u64, Subscriber)>,
}

/// 订阅可以先于 `register` 到来：未知流的订阅先挂在 `pending`，
/// `register` 时整批搬进新建的 `StreamBuf`。订阅 id 全局唯一（`next_sub`），
/// 这样一个 disposer 不用关心它当时落在哪张表里。
struct Inner {
    streams: HashMap<String, StreamBuf>,
    pending: HashMap<String, Vec<(u64, Subscriber)>>,
    next_sub: u64,
    next_reg: u64,
    /// 撤销注册时保存该流最后发出的 `frame_seq`，重新注册时从这里续上——否则重启的
    /// mock-stream/相机驱动会让 `frame_seq` 归零，旧的 `FrameRef` 就会和新帧撞号。
    last_seq: HashMap<String, u64>,
}

/// 一条流的环形缓冲最多留多少帧。
///
/// 容量本来完全由 `nominal_hz × window_s` 算出来，而 `nominal_hz` 是**声明**出来的
/// （`plugin/register` 的 manifest、`mock-stream` 的配置），不是量出来的：一个
/// `nominal_hz: 1e12` 会让下面那行算出 2e12，`VecDeque::with_capacity` 于是去要 176 TB。
/// 分配失败走的是 `handle_alloc_error`——那是 abort，不是 panic，`FiberTree::activate`
/// 的 `catch_unwind` 接不住，整个内核进程当场没（Plan 3 最终评审 C1，实测 SIGABRT）。
///
/// 边界协议那一侧已经有一道 `max_nominal_hz` 护栏（`rpc-server` 的配置项），这一道是
/// 给**所有**调用方的：这个 API 是 `pub` 的，一个进程内插件写错一个数字不该有本事
/// 杀掉内核。65536 帧在一条 500 Hz 的流上是 131 秒的窗口，比任何真实的 `window_s`
/// 都长得多，所以对正常配置它完全不可见；真被夹到的那些，代价只是"回看窗口比声明的
/// 短"，不是丢数据（最新值寄存器不受影响）。
pub const MAX_RING_FRAMES: usize = 1 << 16;

/// 每条流一个环形缓冲 + 最新值寄存器。原始帧不进日志。
pub struct FrameStore {
    clock: Arc<dyn Clock>,
    window_s: f64,
    inner: Arc<Mutex<Inner>>,
}

impl FrameStore {
    pub fn new(clock: Arc<dyn Clock>, window_s: f64) -> Self {
        Self {
            clock,
            window_s,
            inner: Arc::new(Mutex::new(Inner {
                streams: HashMap::new(),
                pending: HashMap::new(),
                next_sub: 1,
                next_reg: 1,
                last_seq: HashMap::new(),
            })),
        }
    }

    fn shared(&self) -> Arc<Mutex<Inner>> {
        self.inner.clone()
    }

    /// 注册一条流，返回撤销它的 disposer（注册是可逆 Effect）。撤销时 StreamBuf 连同环形
    /// 缓冲一起丢弃，订阅者搬回 `pending`——重新注册会把它们接回去，而不是让它们静默失联。
    pub fn register(&self, spec: StreamSpec) -> Result<Disposer, FrameError> {
        let stream_id = spec.stream_id.clone();
        let reg_id = {
            let mut g = self.inner.lock();
            if g.streams.contains_key(&spec.stream_id) {
                return Err(FrameError::AlreadyRegistered(spec.stream_id));
            }
            // `clamp` 的上界见 [`MAX_RING_FRAMES`]：下界 8 是"再慢的流也要能回看几帧"，
            // 上界是"再离谱的声明也不能把内核变成一次巨型分配"。`NaN as usize` 是 0，
            // 所以非有限的 `nominal_hz` 落在下界上，也不会变成一个巨大的容量。
            let capacity =
                ((spec.nominal_hz * self.window_s).ceil() as usize).clamp(8, MAX_RING_FRAMES);
            let subscribers = g.pending.remove(&spec.stream_id).unwrap_or_default();
            let reg_id = g.next_reg;
            g.next_reg += 1;
            let next_seq = g.last_seq.get(&spec.stream_id).copied().unwrap_or(0) + 1;
            g.streams.insert(
                spec.stream_id.clone(),
                StreamBuf {
                    reg_id,
                    spec,
                    ring: VecDeque::with_capacity(capacity),
                    capacity,
                    next_seq,
                    latest: None,
                    subscribers,
                },
            );
            reg_id
        };
        let handle = self.shared();
        Ok(Box::new(move || {
            let mut g = handle.lock();
            if g.streams.get(&stream_id).map(|s| s.reg_id) != Some(reg_id) {
                return;
            }
            let buf = g.streams.remove(&stream_id).expect("checked above");
            g.last_seq.insert(stream_id.clone(), buf.next_seq - 1);
            if !buf.subscribers.is_empty() {
                g.pending
                    .entry(stream_id)
                    .or_default()
                    .extend(buf.subscribers);
            }
        }))
    }

    pub fn ingest(
        &self,
        stream_id: &str,
        payload: Bytes,
        t_source: Option<u64>,
    ) -> Result<Frame, FrameError> {
        let (frame, subs) = {
            let mut g = self.inner.lock();
            // 时钟在锁内采样：frame_seq 和 t_kernel 必须同序，否则并发 ingest 会让
            // seq N 的时间戳晚于 seq N+1。
            let t_kernel = self.clock.now_ns();
            let s = g
                .streams
                .get_mut(stream_id)
                .ok_or_else(|| FrameError::UnknownStream(stream_id.to_string()))?;
            let frame = Frame {
                stream_id: stream_id.to_string(),
                frame_seq: s.next_seq,
                t_kernel,
                t_source,
                payload,
            };
            s.next_seq += 1;
            if s.ring.len() == s.capacity {
                s.ring.pop_front();
            }
            s.ring.push_back(frame.clone());
            s.latest = Some(frame.clone());
            let subs: Vec<Subscriber> = s.subscribers.iter().map(|(_, f)| f.clone()).collect();
            (frame, subs)
        };
        for f in &subs {
            f(&frame);
        }
        Ok(frame)
    }

    pub fn latest(&self, stream_id: &str) -> Option<Frame> {
        self.inner
            .lock()
            .streams
            .get(stream_id)
            .and_then(|s| s.latest.clone())
    }

    pub fn get(&self, r: &FrameRef) -> Option<Frame> {
        self.inner
            .lock()
            .streams
            .get(&r.stream_id)
            .and_then(|s| s.ring.iter().find(|f| f.frame_seq == r.frame_seq).cloned())
    }

    /// 订阅者契约：回调在 ingest 线程（通常是 tokio worker）上**同步**执行，持锁之外但在
    /// 产帧路径上——必须是有界、非阻塞的，且不能 panic（panic 会穿过 `ingest` 的调用者）。
    /// 重活交给有界 channel 的消费端，别在回调里做。
    pub fn subscribe(&self, stream_id: &str, f: Subscriber) -> Result<Disposer, FrameError> {
        let mut g = self.inner.lock();
        let id = g.next_sub;
        g.next_sub += 1;
        match g.streams.get_mut(stream_id) {
            Some(s) => s.subscribers.push((id, f)),
            None => g
                .pending
                .entry(stream_id.to_string())
                .or_default()
                .push((id, f)),
        }
        drop(g);
        Ok(self.make_unsubscribe(stream_id.to_string(), id))
    }

    fn make_unsubscribe(&self, stream_id: String, id: u64) -> Disposer {
        let handle = self.shared();
        Box::new(move || {
            let mut g = handle.lock();
            if let Some(s) = g.streams.get_mut(&stream_id) {
                s.subscribers.retain(|(i, _)| *i != id);
            }
            if let Some(p) = g.pending.get_mut(&stream_id) {
                p.retain(|(i, _)| *i != id);
            }
        })
    }

    /// 一条流当前声明的 schema。`streams()` 会克隆全部 spec 并排序——蒸馏器要在每帧
    /// （500 Hz）上确认 schema 没被换过，用这个只查一次表、只克隆一个字符串。
    pub fn schema_of(&self, stream_id: &str) -> Option<String> {
        self.inner
            .lock()
            .streams
            .get(stream_id)
            .map(|s| s.spec.schema.clone())
    }

    pub fn streams(&self) -> Vec<StreamSpec> {
        let g = self.inner.lock();
        let mut v: Vec<_> = g.streams.values().map(|s| s.spec.clone()).collect();
        v.sort_by(|a, b| a.stream_id.cmp(&b.stream_id));
        v
    }

    pub fn stats(&self, stream_id: &str) -> Option<StreamStats> {
        self.inner
            .lock()
            .streams
            .get(stream_id)
            .map(|s| StreamStats {
                frames_total: s.next_seq - 1,
                buffered: s.ring.len(),
                capacity: s.capacity,
            })
    }
}
