use oeh_kernel::factories::PluginFactory;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{ClockService, Ctx, FrameStore, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    streams: BTreeMap<String, f64>,
    #[serde(default = "d_period")]
    period_ms: u64,
}
fn d_period() -> u64 {
    50
}

pub struct WatchdogPlugin;
impl Plugin for WatchdogPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("distill-watchdog").inject(&["clock", "frame-store", "world-log", "state"])
    }
    fn apply(&self, ctx: &Ctx, config: &serde_json::Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let clock = ctx.get::<ClockService>("clock")?.0.clone();
        let fs = ctx.get::<FrameStore>("frame-store")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let last: Arc<Mutex<HashMap<String, bool>>> = Arc::new(Mutex::new(HashMap::new())); // stream → stale?
        let kind = EventKind::new("health/stale").expect("static");
        ctx.spawn_interval(Duration::from_millis(cfg.period_ms), move || {
            let now = clock.now_ns();
            for (stream, max_age) in &cfg.streams {
                let last_frame = fs.latest(stream).map(|f| f.t_kernel);
                let age_s = last_frame.map(|t| now.saturating_sub(t) as f64 / 1e9);
                let stale = age_s.map(|a| a > *max_age).unwrap_or(true);
                // `age_s` 是相对采样时刻的派生量，读的人无从判断它有多旧；把原始的
                // `t_kernel` 一起发出去（没有帧则为 null）。三个键一次锁内写完。
                let (k_age, k_fresh, k_last) = (
                    format!("age_s.{stream}"),
                    format!("fresh.{stream}"),
                    format!("last_frame_t_kernel.{stream}"),
                );
                state.set_many(
                    &[
                        (k_age.as_str(), json!(age_s)),
                        (k_fresh.as_str(), json!(!stale)),
                        (k_last.as_str(), json!(last_frame)),
                    ],
                    "distill-watchdog",
                );
                // 决定要不要发事件 + 记录新状态都在 `last` 锁内完成；锁在这个块结束时释放，
                // `append` 在锁外调用——不能带着这把私有锁跨进 `append`。
                let fire = {
                    let mut l = last.lock();
                    let prev = l.get(stream).copied().unwrap_or(true);
                    let first = !l.contains_key(stream);
                    l.insert(stream.clone(), stale);
                    stale != prev || (first && stale)
                };
                if fire {
                    log.append(
                        EventDraft::new(
                            kind.clone(),
                            "distill-watchdog",
                            json!({"stream": stream, "age_s": age_s, "stale": stale}),
                        )
                        .with_salience(220)
                        .with_collapse_key(format!("stale:{stream}")),
                    );
                }
            }
        });
        Ok(())
    }
}
inventory::submit! { PluginFactory { id: "distill-watchdog", build: || Box::new(WatchdogPlugin) } }
