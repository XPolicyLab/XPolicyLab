use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 内核单调时钟：唯一时间权威，单位纳秒。
pub trait Clock: Send + Sync + 'static {
    /// 纳秒，内核启动以来的单调时间。
    fn now_ns(&self) -> u64;
}

pub struct MonotonicClock {
    start: Instant,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}
impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}
impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }
}

/// 测试用可控时钟。
pub struct ManualClock {
    now: AtomicU64,
}
impl ManualClock {
    pub fn new(start_ns: u64) -> Self {
        Self {
            now: AtomicU64::new(start_ns),
        }
    }
    pub fn advance(&self, ns: u64) {
        self.now.fetch_add(ns, Ordering::SeqCst);
    }
    pub fn set(&self, ns: u64) {
        self.now.store(ns, Ordering::SeqCst);
    }
}
impl Clock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
}

/// 注册表里以具体类型提供（`Arc<dyn Clock>` 是胖指针，不能直接 downcast）。
pub struct ClockService(pub Arc<dyn Clock>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_never_goes_backwards() {
        let c = MonotonicClock::new();
        let a = c.now_ns();
        let b = c.now_ns();
        assert!(b >= a);
    }

    #[test]
    fn manual_clock_is_controllable() {
        let c = ManualClock::new(10);
        assert_eq!(c.now_ns(), 10);
        c.advance(5);
        assert_eq!(c.now_ns(), 15);
        c.set(3);
        assert_eq!(c.now_ns(), 3);
    }

    #[test]
    fn clock_service_wraps_dyn_clock() {
        let svc = ClockService(std::sync::Arc::new(ManualClock::new(7)));
        assert_eq!(svc.0.now_ns(), 7);
    }
}
