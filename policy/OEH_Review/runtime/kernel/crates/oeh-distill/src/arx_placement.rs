//! `distill-placement`：把一次 depth 采样判成"放好了"或"没放好"，并把放好的物体变成禁区。
//!
//! 输入是 `obs/arx_x5_placement_sample`（CONTRACT §4.2），由查询能力
//! `arx_x5_query_world_map` 在带 `label` 时发出。阈值**不写死在代码里**：每个任务一份
//! JSON（`task/robodojo/<task>.json`），由 `spec_path` 指定，在 `apply` 里读；读不了或
//! 形状不对直接 [`PluginError`]——一份坏掉的阈值表必须在 boot 当场响，而不是在放置那一刻
//! 悄悄放行。
//!
//! **spec 形状（与 P5 的约定）**：物体名 → `{min_z_m, max_z_m, min_inset_m, max_motion_m}`，
//! 外加可选的 `"_default"`。任何**别的** `_` 前缀键（例如 `"_source"` 记出处）都是元数据，
//! 被忽略；不带 `_` 前缀而又不是阈值对象的值则是 [`PluginError`] 并指名那个键。
//!
//! 三个判定，按顺序取第一个失败的原因：
//!
//! 1. **高度**：`top_surface_median[2]` 落在 `[min_z_m, max_z_m]` 内。
//! 2. **内缩**：`top_surface_median` 到采样区域（`xyz_min`/`xyz_max`）四条水平边界的最小
//!    距离 ≥ `min_inset_m`。这是 g7 教训里"离前沿 ≥ 5 cm"的可计算版本——载荷里没有桌沿的
//!    绝对坐标，能算的只有它相对于被查询区域的位置，所以脑必须把 bbox 框在放置面上。
//! 3. **静止**：同一 `(episode_id, label)` 的**上一次**采样到这一次的位移 ≤ `max_motion_m`。
//!    一局里的第一条采样没有可比对象，跳过这一项（不是判失败）。
//!
//! 通过 → `obs/placement_verified` + 写 `placement_verified.<object>`（`rule-keepout` 读它）。
//! 失败 → `obs/placement_failed`，**不删任何寄存器**：一个之前验证过的放置仍然受保护，
//! 一次读坏的采样不该把禁区解掉。两条事件共用 `collapse_key`，脑看到的永远是最新结论。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::arx_ledger::{distiller_error, sample};

pub const ID: &str = "distill-placement";

/// 输入事件的 kind。
pub const SAMPLE_KIND: &str = "obs/arx_x5_placement_sample";

/// spec 里的回退条目名。别的 `_` 前缀键都是元数据。
const DEFAULT_KEY: &str = "_default";

fn d_radius() -> f64 {
    0.10
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Config {
    spec_path: PathBuf,
    #[serde(default = "d_radius")]
    keepout_radius_m: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq)]
#[serde(deny_unknown_fields)]
struct Thresholds {
    min_z_m: f64,
    max_z_m: f64,
    min_inset_m: f64,
    max_motion_m: f64,
}

#[derive(Deserialize, Debug, Clone)]
struct PlacementSample {
    label: String,
    #[serde(default)]
    episode_id: Option<String>,
    #[serde(default)]
    native_step: Option<i64>,
    xyz_min: [f64; 3],
    xyz_max: [f64; 3],
    top_surface_median: [f64; 3],
}

/// 读并校验阈值表。`_` 前缀键（`_default` 除外）是元数据；别的键必须是阈值对象。
fn load_spec(path: &PathBuf) -> Result<BTreeMap<String, Thresholds>, PluginError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        PluginError::Other(format!(
            "{ID}: cannot read placement spec `{}`: {e}",
            path.display()
        ))
    })?;
    let raw: BTreeMap<String, Value> = serde_json::from_str(&text).map_err(|e| {
        PluginError::Other(format!(
            "{ID}: placement spec `{}` is not a JSON object: {e}",
            path.display()
        ))
    })?;
    let mut out = BTreeMap::new();
    for (key, value) in raw {
        if key.starts_with('_') && key != DEFAULT_KEY {
            continue; // 出处之类的元数据，忽略。
        }
        let t: Thresholds = serde_json::from_value(value).map_err(|e| {
            PluginError::Other(format!(
                "{ID}: placement spec `{}` entry `{key}` is not a threshold object: {e}",
                path.display()
            ))
        })?;
        out.insert(key, t);
    }
    Ok(out)
}

/// `(episode_id, label)` → 上一次采样的 `top_surface_median`，静止判定用。
type PrevSamples = BTreeMap<(String, String), [f64; 3]>;

fn all_finite(a: &[f64; 3]) -> bool {
    a.iter().all(|x| x.is_finite())
}

/// 水平方向到采样区域边界的最小距离。见模块文档第 2 条。
fn inset(point: &[f64; 3], lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
    let mut m = f64::INFINITY;
    for axis in 0..2 {
        m = m.min(point[axis] - lo[axis]).min(hi[axis] - point[axis]);
    }
    m
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub struct ArxPlacementPlugin;

impl Plugin for ArxPlacementPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["world-log", "state"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        let spec = Arc::new(load_spec(&cfg.spec_path)?);
        let radius = cfg.keepout_radius_m;
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let kind = EventKind::new(SAMPLE_KIND).expect("static");
        // (episode_id, label) → 上一次的 top_surface_median。
        let prev: Arc<Mutex<PrevSamples>> = Arc::new(Mutex::new(PrevSamples::new()));
        let bad = Arc::new(AtomicU64::new(0));
        let log2 = log.clone();
        let disposer = log.observe(Arc::new(move |ev: &WorldEvent| {
            // 第一句按 kind 过滤并早退：我们自己发的两条也会走一遍所有观察者。
            if ev.kind != kind {
                return;
            }
            let p: PlacementSample = match serde_json::from_value(ev.payload.clone()) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(n) = sample(&bad) {
                        log2.append(distiller_error(ID, SAMPLE_KIND, n, e.to_string()));
                    }
                    return;
                }
            };
            if !all_finite(&p.top_surface_median)
                || !all_finite(&p.xyz_min)
                || !all_finite(&p.xyz_max)
            {
                if let Some(n) = sample(&bad) {
                    log2.append(distiller_error(
                        ID,
                        SAMPLE_KIND,
                        n,
                        "non-finite sample geometry".to_string(),
                    ));
                }
                return;
            }
            let Some(t) = spec
                .get(&p.label)
                .or_else(|| spec.get(DEFAULT_KEY))
                .copied()
            else {
                return; // 这个任务不关心这个物体。
            };
            let episode = p.episode_id.clone().unwrap_or_default();
            // 上一次采样的读取与更新在锁内一次完成；`append` 在锁外。
            let previous = {
                let mut g = prev.lock();
                g.insert((episode.clone(), p.label.clone()), p.top_surface_median)
            };

            let z = p.top_surface_median[2];
            let ins = inset(&p.top_surface_median, &p.xyz_min, &p.xyz_max);
            let motion = previous.map(|q| distance(&p.top_surface_median, &q));
            let reason = if z < t.min_z_m {
                Some("height_below_min")
            } else if z > t.max_z_m {
                Some("height_above_max")
            } else if ins < t.min_inset_m {
                Some("inset_below_min")
            } else if motion.map(|m| m > t.max_motion_m).unwrap_or(false) {
                Some("moved")
            } else {
                None
            };

            let xyz = json!(p.top_surface_median);
            match reason {
                None => {
                    let key = format!("placement_verified.{}", p.label);
                    state.set_many(
                        &[(
                            key.as_str(),
                            json!({
                                "xyz": xyz.clone(),
                                "radius_m": radius,
                                "native_step": p.native_step,
                                "episode_id": p.episode_id,
                            }),
                        )],
                        ID,
                    );
                    log2.append(
                        EventDraft::new(
                            EventKind::new("obs/placement_verified").expect("static"),
                            ID,
                            json!({
                                "object": p.label,
                                "xyz": xyz.clone(),
                                "radius_m": radius,
                                "native_step": p.native_step,
                                "episode_id": p.episode_id,
                            }),
                        )
                        .with_salience(150)
                        .with_collapse_key(format!("placement:{}", p.label)),
                    );
                }
                Some(reason) => {
                    log2.append(
                        EventDraft::new(
                            EventKind::new("obs/placement_failed").expect("static"),
                            ID,
                            json!({
                                "object": p.label,
                                "reason": reason,
                                "xyz": xyz,
                                "thresholds": t,
                                "native_step": p.native_step,
                            }),
                        )
                        .with_salience(200)
                        .with_collapse_key(format!("placement:{}", p.label)),
                    );
                }
            }
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(ArxPlacementPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_path_is_required_and_radius_defaults() {
        let c: Config = serde_json::from_value(json!({"spec_path": "/tmp/x.json"})).unwrap();
        assert_eq!(c.spec_path, PathBuf::from("/tmp/x.json"));
        assert!((c.keepout_radius_m - 0.10).abs() < 1e-12);
        assert!(serde_json::from_value::<Config>(json!({})).is_err());
    }

    #[test]
    fn partial_config_keeps_the_other_defaults() {
        let c: Config =
            serde_json::from_value(json!({"spec_path": "/tmp/x.json", "keepout_radius_m": 0.25}))
                .unwrap();
        assert!((c.keepout_radius_m - 0.25).abs() < 1e-12);
    }

    #[test]
    fn typoed_key_is_an_error() {
        assert!(serde_json::from_value::<Config>(
            json!({"spec_path": "/tmp/x.json", "keepout_radiusm": 0.25})
        )
        .is_err());
        // 阈值对象自己也是 deny_unknown_fields：写错一个阈值名不能被当成"用默认值"。
        assert!(serde_json::from_value::<Thresholds>(json!({
            "min_z_m": 0.0, "max_z_m": 1.0, "min_inset_m": 0.05, "max_motion_mm": 0.01
        }))
        .is_err());
    }

    #[test]
    fn inset_is_the_nearest_horizontal_boundary() {
        let lo = [0.0, 0.0, 0.0];
        let hi = [1.0, 1.0, 1.0];
        assert!((inset(&[0.5, 0.5, 0.9], &lo, &hi) - 0.5).abs() < 1e-12);
        assert!((inset(&[0.02, 0.5, 0.9], &lo, &hi) - 0.02).abs() < 1e-12);
        // z 不参与：内缩是水平量。
        assert!((inset(&[0.5, 0.5, 0.0], &lo, &hi) - 0.5).abs() < 1e-12);
    }
}
