//! `distill-budget`：原生步预算的水位线。
//!
//! 输入是 `obs/arx_x5_ledger` 的 `native_steps_remaining`。每个水位线在**一局里只发一次**
//! （`BTreeSet` 锁存，`episode_id` 变化时清空并重新武装）：`wake/arx.yaml` 把
//! `health/native_budget` 映射成 `steer`，每条事件都发会把脑的预算打光。
//!
//! 默认水位 `[300, 150, 120]` 来自设计 §3.4；120 与 `rule-step-reserve.reserve_steps` 对齐——
//! 越过它之后除 `return_to_start` / `wait` / `reset` 外的运动都会被拒，脑必须在那之前收手。
//!
//! 剩余步数一次跨过多个水位时，按**从大到小**逐个补发：脑看到的是完整的下降序列，而不是
//! 只有最后一条。

use std::collections::BTreeSet;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::arx_ledger::{distiller_error, sample, LEDGER_KIND};

pub const ID: &str = "distill-budget";

fn d_marks() -> Vec<u64> {
    vec![300, 150, 120]
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_marks")]
    marks: Vec<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self { marks: d_marks() }
    }
}

#[derive(Deserialize, Debug, Clone)]
struct BudgetPayload {
    episode_id: String,
    #[serde(default)]
    native_steps_remaining: Option<i64>,
    #[serde(default)]
    native_step_limit: Option<i64>,
}

#[derive(Default)]
struct Latched {
    episode: Option<String>,
    fired: BTreeSet<u64>,
}

pub struct ArxBudgetPlugin;

impl Plugin for ArxBudgetPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["world-log", "state"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        // 从大到小，一次下降补齐全部越过的水位。
        let mut marks = cfg.marks.clone();
        marks.sort_unstable();
        marks.dedup();
        marks.reverse();
        let log = ctx.get::<WorldLog>("world-log")?;
        let kind = EventKind::new(LEDGER_KIND).expect("static");
        let latched = Arc::new(Mutex::new(Latched::default()));
        let bad = Arc::new(AtomicU64::new(0));
        let log2 = log.clone();
        let disposer = log.observe(Arc::new(move |ev: &WorldEvent| {
            if ev.kind != kind {
                return;
            }
            let p: BudgetPayload = match serde_json::from_value(ev.payload.clone()) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(n) = sample(&bad) {
                        log2.append(distiller_error(ID, LEDGER_KIND, n, e.to_string()));
                    }
                    return;
                }
            };
            let Some(remaining) = p.native_steps_remaining else {
                return;
            };
            // 判定与锁存在锁内一次完成，`append` 在锁外。
            let fire: Vec<u64> = {
                let mut l = latched.lock();
                if l.episode.as_deref() != Some(p.episode_id.as_str()) {
                    l.episode = Some(p.episode_id.clone());
                    l.fired.clear();
                }
                marks
                    .iter()
                    .copied()
                    .filter(|m| remaining <= *m as i64 && l.fired.insert(*m))
                    .collect()
            };
            for mark in fire {
                log2.append(
                    EventDraft::new(
                        EventKind::new("health/native_budget").expect("static"),
                        ID,
                        json!({
                            "mark": mark,
                            "native_steps_remaining": remaining,
                            "native_step_limit": p.native_step_limit,
                        }),
                    )
                    .with_salience(200)
                    .with_collapse_key("native_budget"),
                );
            }
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(ArxBudgetPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_design_marks() {
        assert_eq!(Config::default().marks, vec![300, 150, 120]);
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty.marks, vec![300, 150, 120]);
    }

    #[test]
    fn partial_config_replaces_only_marks() {
        let c: Config = serde_json::from_value(json!({"marks": [500]})).unwrap();
        assert_eq!(c.marks, vec![500]);
    }

    #[test]
    fn typoed_key_is_an_error() {
        assert!(serde_json::from_value::<Config>(json!({"mark": [500]})).is_err());
    }
}
