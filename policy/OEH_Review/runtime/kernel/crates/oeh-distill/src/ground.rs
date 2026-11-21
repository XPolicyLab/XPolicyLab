use oeh_kernel::factories::PluginFactory;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::payload::DetectionsScan;
use oeh_proto::{EventDraft, EventKind, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_obstacle")]
    obstacle_m: f64,
    #[serde(default = "d_person")]
    person_m: f64,
    #[serde(default = "d_hyst")]
    hysteresis_m: f64,
}
fn d_obstacle() -> f64 {
    0.6
}
fn d_person() -> f64 {
    0.8
}
fn d_hyst() -> f64 {
    0.1
}
impl Default for Config {
    fn default() -> Self {
        Self {
            obstacle_m: d_obstacle(),
            person_m: d_person(),
            hysteresis_m: d_hyst(),
        }
    }
}

#[derive(Default)]
struct Latched {
    person_near: bool,
    obstacle_near: bool,
    clear_path: Option<bool>,
}

/// 迟滞判定：进入 <= thr，退出 > thr + hyst。None = 无穷远。
fn near(prev: bool, d: Option<f64>, thr: f64, hyst: f64) -> bool {
    match d {
        None => false,
        Some(d) => {
            if prev {
                d <= thr + hyst
            } else {
                d <= thr
            }
        }
    }
}

pub struct GroundPlugin;
impl Plugin for GroundPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("distill-ground").inject(&["world-log", "state"])
    }
    fn apply(&self, ctx: &Ctx, config: &serde_json::Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let latched = Arc::new(Mutex::new(Latched::default()));
        let (log2, state2) = (log.clone(), state.clone());
        let kind_detections = EventKind::new("obs/detections").expect("static");
        let disposer = log.observe(Arc::new(move |ev: &WorldEvent| {
            if ev.kind != kind_detections {
                return;
            }
            let Ok(scan) = serde_json::from_value::<DetectionsScan>(ev.payload.clone()) else {
                return;
            };
            // 迟滞判定 + 边沿判断 + 落盘全部在 `latched` 锁内一次性完成，把最终的 p/o/clear
            // 和"要不要发事件"一起抄出来；锁在这个块结束时释放，随后才 `append`——
            // `WorldLog::append` 会同步回调所有观察者（包括我们自己，对自己产出的事件会在
            // 最上面的 kind 判断处早退），绝不能带着这把私有锁跨进 `append`。
            let (p, o, clear, fire_person, fire_obstacle, fire_clear) = {
                let mut l = latched.lock();
                let p = near(
                    l.person_near,
                    scan.nearest_person_m,
                    cfg.person_m,
                    cfg.hysteresis_m,
                );
                let o = near(
                    l.obstacle_near,
                    scan.nearest_obstacle_m,
                    cfg.obstacle_m,
                    cfg.hysteresis_m,
                );
                let clear = scan.clear_path_raw && !o;
                let fire_person = p != l.person_near;
                let fire_obstacle = o != l.obstacle_near;
                let fire_clear = l.clear_path != Some(clear);
                l.person_near = p;
                l.obstacle_near = o;
                l.clear_path = Some(clear);
                (p, o, clear, fire_person, fire_obstacle, fire_clear)
            };
            // 五个寄存器一次锁内写完：分开写会让观测读到“person_near 已经翻了但
            // nearest_person_m 还是上一帧”的中间态。
            state2.set_many(
                &[
                    ("nearest_person_m", json!(scan.nearest_person_m)),
                    ("nearest_obstacle_m", json!(scan.nearest_obstacle_m)),
                    ("person_near", json!(p)),
                    ("obstacle_near", json!(o)),
                    ("clear_path", json!(clear)),
                ],
                "distill-ground",
            );
            let refs = ev.frame_refs.clone();
            if fire_person {
                log2.append(
                    EventDraft::new(
                        EventKind::new("obs/person").expect("static"),
                        "distill-ground",
                        json!({"near": p, "nearest_person_m": scan.nearest_person_m}),
                    )
                    .with_salience(200)
                    .with_collapse_key("person")
                    .with_frame_refs(refs.clone()),
                );
            }
            if fire_obstacle {
                log2.append(
                    EventDraft::new(
                        EventKind::new("obs/obstacle").expect("static"),
                        "distill-ground",
                        json!({"near": o, "nearest_obstacle_m": scan.nearest_obstacle_m}),
                    )
                    .with_salience(180)
                    .with_collapse_key("obstacle")
                    .with_frame_refs(refs.clone()),
                );
            }
            if fire_clear {
                log2.append(
                    EventDraft::new(
                        EventKind::new("obs/clear_path").expect("static"),
                        "distill-ground",
                        json!({"clear": clear}),
                    )
                    .with_salience(150)
                    .with_collapse_key("clear_path")
                    .with_frame_refs(refs),
                );
            }
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}
inventory::submit! { PluginFactory { id: "distill-ground", build: || Box::new(GroundPlugin) } }
