use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oeh_proto::{EventDraft, EventKind, FrameRef, Manifest};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::factories::PluginFactory;
use crate::frame_store::FrameStore;
use crate::plugin::{Ctx, Plugin, PluginError};
use crate::world_log::WorldLog;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    stream_id: String,
    #[serde(default = "default_period")]
    period_ms: u64,
}
fn default_period() -> u64 {
    1000
}

/// 示例蒸馏器：数帧，周期性发 health/frame_count。
pub struct FrameCounterPlugin;

impl Plugin for FrameCounterPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("frame-counter").inject(&["frame-store", "world-log"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let fs = ctx.get::<FrameStore>("frame-store")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let count = Arc::new(AtomicU64::new(0));
        let last: Arc<Mutex<Option<FrameRef>>> = Arc::new(Mutex::new(None));
        let (c2, l2) = (count.clone(), last.clone());
        let disposer = fs
            .subscribe(
                &cfg.stream_id,
                Arc::new(move |f| {
                    c2.fetch_add(1, Ordering::SeqCst);
                    *l2.lock() = Some(f.frame_ref());
                }),
            )
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(disposer);
        let stream_id = cfg.stream_id;
        let kind = EventKind::new("health/frame_count").expect("static kind");
        let prev = AtomicU64::new(0);
        ctx.spawn_interval(Duration::from_millis(cfg.period_ms), move || {
            let total = count.load(Ordering::SeqCst);
            let since = total - prev.swap(total, Ordering::SeqCst);
            let refs: Vec<FrameRef> = last.lock().clone().into_iter().collect();
            log.append(
                EventDraft::new(
                    kind.clone(),
                    "frame-counter",
                    json!({"stream_id": stream_id, "count": total, "since_last": since}),
                )
                .with_frame_refs(refs)
                .with_collapse_key(format!("frame_count:{stream_id}"))
                .with_salience(10),
            );
        });
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: "frame-counter", build: || Box::new(FrameCounterPlugin) } }
