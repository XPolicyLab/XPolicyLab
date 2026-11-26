//! `distill-native`：原生终止的一次性播报。
//!
//! **`distill-native` does NOT transition the FSM.** Design §3.4 says "interrupt", which is a
//! brain wake action, not an FSM edge. Driving `* → FAULT` from a world-log observer would
//! synchronously reach admission's `safety/fsm` observer and cancel the very execution that is
//! still inside the emitting capability body — the brain would get `Outcome::Cancelled` instead
//! of the terminal primitive's result, for the last primitive of essentially every episode
//! (`crates/oeh-admission/src/execution.rs:313`, and `execution.rs:478-479` drops the late
//! `capability/result`). Post-terminal motion is instead refused by `rule-step-reserve`
//! (`native_steps_remaining` goes to 0) and, for the exempt capabilities, by the Python body's
//! `ArxEpisodeTerminal`.
//!
//! 也就是说：这里只发一条 `safety/native_terminal`（`wake/arx.yaml` 把它映射成 `interrupt`），
//! 别的什么都不做。每局锁存一次——原生终止之后账本事件还会继续来（`finish` 仍要把这一局
//! 结算掉），每条都发会把脑反复打断。

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::arx_ledger::{distiller_error, sample, LEDGER_KIND};

pub const ID: &str = "distill-native";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {}

#[derive(Deserialize, Debug, Clone)]
struct NativePayload {
    episode_id: String,
    #[serde(default)]
    native_step: Option<i64>,
    #[serde(default)]
    native_terminal: Option<bool>,
    #[serde(default)]
    native_success: Option<bool>,
}

pub struct ArxNativePlugin;

impl Plugin for ArxNativePlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["world-log", "state"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let _cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let log = ctx.get::<WorldLog>("world-log")?;
        let kind = EventKind::new(LEDGER_KIND).expect("static");
        // 已经播报过终止的那一局。
        let fired: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let bad = Arc::new(AtomicU64::new(0));
        let log2 = log.clone();
        let disposer = log.observe(Arc::new(move |ev: &WorldEvent| {
            if ev.kind != kind {
                return;
            }
            let p: NativePayload = match serde_json::from_value(ev.payload.clone()) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(n) = sample(&bad) {
                        log2.append(distiller_error(ID, LEDGER_KIND, n, e.to_string()));
                    }
                    return;
                }
            };
            if p.native_terminal != Some(true) {
                return;
            }
            // 判定与锁存在锁内一次完成，`append` 在锁外。
            let fire = {
                let mut g = fired.lock();
                if g.as_deref() == Some(p.episode_id.as_str()) {
                    false
                } else {
                    *g = Some(p.episode_id.clone());
                    true
                }
            };
            if !fire {
                return;
            }
            let success = p.native_success.unwrap_or(false);
            log2.append(
                EventDraft::new(
                    EventKind::new("safety/native_terminal").expect("static"),
                    ID,
                    json!({
                        "episode_id": p.episode_id,
                        "native_step": p.native_step,
                        "native_success": p.native_success,
                        "reason": if success { "native success" } else { "terminated or truncated" },
                    }),
                )
                .with_salience(240)
                .with_collapse_key("native_terminal"),
            );
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(ArxNativePlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_takes_nothing_and_rejects_typos() {
        let _: Config = serde_json::from_value(json!({})).unwrap();
        let _ = Config::default();
        assert!(serde_json::from_value::<Config>(json!({"latch": true})).is_err());
    }
}
