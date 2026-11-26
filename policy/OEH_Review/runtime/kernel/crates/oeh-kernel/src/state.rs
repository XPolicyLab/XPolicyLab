use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

use crate::clock::Clock;

#[derive(Clone, Debug, PartialEq)]
pub struct StateEntry {
    pub value: Value,
    pub t_kernel: u64,
    pub source: String,
}

/// 蒸馏状态的最新值寄存器。蒸馏器写，观测构造与规则读。
pub struct StateRegistry {
    clock: Arc<dyn Clock>,
    inner: Mutex<BTreeMap<String, StateEntry>>,
}

impl StateRegistry {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            inner: Mutex::new(BTreeMap::new()),
        }
    }
    pub fn set(&self, key: &str, value: Value, source: &str) {
        let t_kernel = self.clock.now_ns();
        self.inner.lock().insert(
            key.to_string(),
            StateEntry {
                value,
                t_kernel,
                source: source.to_string(),
            },
        );
    }
    /// 一次加锁写多个键。观测构造读的是 `snapshot()` 的一次快照：分成多次 `set` 写的
    /// 一组相关寄存器会被读到中间态（例如 `person_near=true` 但 `nearest_person_m`
    /// 还是上一帧的值），脑看到的就是一份自相矛盾的状态。
    pub fn set_many(&self, entries: &[(&str, Value)], source: &str) {
        let t_kernel = self.clock.now_ns();
        let mut g = self.inner.lock();
        for (k, v) in entries {
            g.insert(
                (*k).to_string(),
                StateEntry {
                    value: v.clone(),
                    t_kernel,
                    source: source.to_string(),
                },
            );
        }
    }
    pub fn get(&self, key: &str) -> Option<Value> {
        self.inner.lock().get(key).map(|e| e.value.clone())
    }
    pub fn entry(&self, key: &str) -> Option<StateEntry> {
        self.inner.lock().get(key).cloned()
    }
    pub fn snapshot(&self) -> BTreeMap<String, Value> {
        self.inner
            .lock()
            .iter()
            .map(|(k, e)| (k.clone(), e.value.clone()))
            .collect()
    }
    /// 带 `t_kernel` / `source` 的完整快照：一次加锁，跨键一致。需要判断“这个值有多旧”
    /// 而不只是“值是多少”的消费者用它，而不是逐键 `entry()`。
    pub fn snapshot_entries(&self) -> BTreeMap<String, StateEntry> {
        self.inner.lock().clone()
    }
    pub fn age_ns(&self, key: &str, now_ns: u64) -> Option<u64> {
        self.inner
            .lock()
            .get(key)
            .map(|e| now_ns.saturating_sub(e.t_kernel))
    }
}
