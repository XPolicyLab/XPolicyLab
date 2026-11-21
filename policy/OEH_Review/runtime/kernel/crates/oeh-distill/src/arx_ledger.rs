//! `distill-arx-ledger`：把 `obs/arx_x5_ledger` 事件蒸馏成寄存器，并驱动 `BOOT → STANDING`。
//!
//! 这条蒸馏器是 ARX-X5 操作路线的**事实来源**：运动插件在每个原语结束时发一条账本事件
//! （CONTRACT §4.1），这里把它摊平成 CONTRACT §4.4 的那组寄存器，准入规则读的就是它们。
//!
//! 三件事值得单独说：
//!
//! 1. **`null` 在这里的意思是"未知"，不是"什么都没有"。** `gripper_cmd.*` / `holding.*` /
//!    `arms_home` 三组键即使在载荷里缺席也会被写成 [`serde_json::Value::Null`]——规则那一侧
//!    （`rule-carry-speed` / `rule-hold-no-rotation` / `rule-finish-gate`）把 `Null` 当作
//!    fail-closed 的未知处理。**绝不能**省略这些键，也绝不能拿一个数去顶替：省略会让规则
//!    读到上一局留下来的旧值，那比"未知"危险得多。
//!    而 `gripper_measured.*` / `native_*` / `eef_*` 的未知编码是**键缺席**（CONTRACT §4.4），
//!    所以载荷里没有它们时这里不写。
//!
//! 2. **排序守卫按 `(episode_id, registration, seq)`。** 同一 `episode_id` **且**同一
//!    `registration` 下 `seq <= last_seq` 的事件被丢弃；`episode_id` 或 `registration` 任一
//!    变化都把水位重置。插件中途重连时 `seq` 会回到 0 而 `registration` 加一，于是这条
//!    seq-0 事件被**接受**——只按 `seq` 判重的守卫会让寄存器在重连后永远冻结。
//!
//! 3. **`BOOT → STANDING` 归本插件。** `crates/oeh-kernel/src/fsm.rs:44` 里这条边合法，而
//!    唯一的另一个驱动者 `distill-fsm` 被刻意排除在 `bundles/robodojo-arx-x5.yaml` 之外
//!    （它读的是 G1 的 `lowstate` 流，这条路线上根本没有），所以不存在两个写者。本包里
//!    **没有别的蒸馏器驱动任何 FSM 边**。
//!
//! 一次事件的全部寄存器写走**一次** [`StateRegistry::set_many`]：分次写会让规则读到撕裂的
//! 账本（`holding.left` 已经翻了而 `gripper_cmd.left` 还是上一条事件的值）。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

pub const ID: &str = "distill-arx-ledger";

/// 输入事件的 kind。
pub const LEDGER_KIND: &str = "obs/arx_x5_ledger";

/// 解析失败每这么多条采样一条 `health/distiller_error`；第 1 条总是发。
/// 逐条发会在一个坏掉的适配器下淹掉世界日志。
pub(crate) const ERROR_SAMPLE: u64 = 100;

#[derive(Deserialize, Debug, Clone, Copy, Default)]
pub(crate) struct NumPair {
    #[serde(default)]
    pub(crate) left: Option<f64>,
    #[serde(default)]
    pub(crate) right: Option<f64>,
}

#[derive(Deserialize, Debug, Clone, Copy, Default)]
pub(crate) struct BoolPair {
    #[serde(default)]
    pub(crate) left: Option<bool>,
    #[serde(default)]
    pub(crate) right: Option<bool>,
}

#[derive(Deserialize, Debug, Clone, Default)]
struct EefPose {
    #[serde(default)]
    position: Option<Vec<f64>>,
    #[serde(default)]
    quaternion_wxyz: Option<Vec<f64>>,
}

#[derive(Deserialize, Debug, Clone, Default)]
struct EefPair {
    #[serde(default)]
    left: Option<EefPose>,
    #[serde(default)]
    right: Option<EefPose>,
}

/// CONTRACT §4.1 的载荷。只有身份三件套是必填的——别的字段缺席按 §4.4 的未知编码处理，
/// 而不是把整条事件当成坏事件丢掉。
#[derive(Deserialize, Debug, Clone)]
struct LedgerPayload {
    episode_id: String,
    seq: i64,
    registration: i64,
    #[serde(default)]
    gripper_cmd: NumPair,
    #[serde(default)]
    gripper_measured: NumPair,
    #[serde(default)]
    holding: BoolPair,
    #[serde(default)]
    native_step: Option<i64>,
    #[serde(default)]
    native_step_limit: Option<i64>,
    #[serde(default)]
    native_steps_remaining: Option<i64>,
    #[serde(default)]
    native_terminal: Option<bool>,
    #[serde(default)]
    native_success: Option<bool>,
    #[serde(default)]
    arms_home: Option<bool>,
    #[serde(default)]
    eef: EefPair,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {}

/// 采样后的解析失败事件。四条蒸馏器共用同一个形状，`plugin` 字段区分来源。
pub(crate) fn distiller_error(
    id: &'static str,
    kind: &str,
    count: u64,
    reason: String,
) -> EventDraft {
    EventDraft::new(
        EventKind::new("health/distiller_error").expect("static"),
        id,
        json!({"plugin": id, "kind": kind, "count": count, "reason": reason}),
    )
    .with_salience(200)
    .with_collapse_key(format!("distiller_error:{id}"))
}

/// 采样计数器：第 1 条与每 [`ERROR_SAMPLE`] 条返回 `Some(n)`。
pub(crate) fn sample(counter: &AtomicU64) -> Option<u64> {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n.is_multiple_of(ERROR_SAMPLE) {
        Some(n)
    } else {
        None
    }
}

/// 定长、全有限的数组才写进寄存器：长度不对或含非有限数时按"未知"处理（键缺席），
/// 而不是把一段垃圾几何交给 `rule-keepout` 去算距离。
fn finite_array(v: &Option<Vec<f64>>, n: usize) -> Option<Value> {
    let a = v.as_ref()?;
    if a.len() != n || !a.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some(Value::from(a.clone()))
}

/// 有限数才写：`f64::NAN` 在 JSON 里序列化成 `null`，那会被规则读成"未知"，正是我们要的。
fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite())
}

pub struct ArxLedgerPlugin;

impl Plugin for ArxLedgerPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["world-log", "state", "fsm"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let _cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let fsm = ctx.get::<RobotFsm>("fsm")?;
        let kind = EventKind::new(LEDGER_KIND).expect("static");
        // (episode_id, registration, last_seq)
        let guard: Arc<Mutex<Option<(String, i64, i64)>>> = Arc::new(Mutex::new(None));
        let bad = Arc::new(AtomicU64::new(0));
        let log2 = log.clone();
        let disposer = log.observe(Arc::new(move |ev: &WorldEvent| {
            // 第一句就按 kind 过滤并早退：`append` 会同步回调所有观察者，包括我们自己
            // 追加的 `safety/fsm` / `health/distiller_error`。
            if ev.kind != kind {
                return;
            }
            let p: LedgerPayload = match serde_json::from_value(ev.payload.clone()) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(n) = sample(&bad) {
                        log2.append(distiller_error(ID, LEDGER_KIND, n, e.to_string()));
                    }
                    return;
                }
            };
            // 判定在锁内一次完成，写寄存器与 `append`（`transition` 内部会 append）在锁外。
            let accept = {
                let mut g = guard.lock();
                match &*g {
                    Some((ep, reg, last))
                        if *ep == p.episode_id && *reg == p.registration && p.seq <= *last =>
                    {
                        false
                    }
                    _ => {
                        *g = Some((p.episode_id.clone(), p.registration, p.seq));
                        true
                    }
                }
            };
            if !accept {
                return;
            }

            let mut kv: Vec<(String, Value)> = Vec::with_capacity(24);
            // null = 未知的三组：键**一定**写。
            kv.push(("gripper_cmd.left".into(), json!(finite(p.gripper_cmd.left))));
            kv.push((
                "gripper_cmd.right".into(),
                json!(finite(p.gripper_cmd.right)),
            ));
            kv.push(("holding.left".into(), json!(p.holding.left)));
            kv.push(("holding.right".into(), json!(p.holding.right)));
            kv.push(("arms_home".into(), json!(p.arms_home)));
            // 键缺席 = 未知的那些：载荷里没有就不写。
            if let Some(v) = finite(p.gripper_measured.left) {
                kv.push(("gripper_measured.left".into(), json!(v)));
            }
            if let Some(v) = finite(p.gripper_measured.right) {
                kv.push(("gripper_measured.right".into(), json!(v)));
            }
            if let Some(v) = p.native_step {
                kv.push(("native_step".into(), json!(v)));
            }
            if let Some(v) = p.native_step_limit {
                kv.push(("native_step_limit".into(), json!(v)));
            }
            if let Some(v) = p.native_steps_remaining {
                kv.push(("native_steps_remaining".into(), json!(v)));
            }
            if let Some(v) = p.native_terminal {
                kv.push(("native_terminal".into(), json!(v)));
            }
            if let Some(v) = p.native_success {
                kv.push(("native_success".into(), json!(v)));
            }
            for (arm, pose) in [("left", &p.eef.left), ("right", &p.eef.right)] {
                let Some(pose) = pose else { continue };
                if let Some(v) = finite_array(&pose.position, 3) {
                    kv.push((format!("eef_position.{arm}"), v));
                }
                if let Some(v) = finite_array(&pose.quaternion_wxyz, 4) {
                    kv.push((format!("eef_quaternion.{arm}"), v));
                }
            }
            kv.push(("arx_episode_id".into(), json!(p.episode_id)));
            kv.push(("arx_ledger_seq".into(), json!(p.seq)));
            kv.push(("arx_ledger_registration".into(), json!(p.registration)));

            let borrowed: Vec<(&str, Value)> =
                kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            // 一次 `set_many`：规则读到的永远是同一条事件的一致快照。
            state.set_many(&borrowed, ID);

            // 寄存器先写、FSM 后翻：脑被 `safety/fsm` 叫醒时账本必须已经在位。
            if fsm.state() == FsmState::Boot {
                if let Err(e) = fsm.transition(FsmState::Standing, "arx_x5 ledger established", ID)
                {
                    tracing::debug!(%e, "fsm");
                }
            }
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(ArxLedgerPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_takes_nothing_and_rejects_typos() {
        let _: Config = serde_json::from_value(json!({})).unwrap();
        assert!(serde_json::from_value::<Config>(json!({"mode": "x"})).is_err());
        // 默认值就是空配置本身：这条蒸馏器没有可调项。
        let _ = Config::default();
    }

    #[test]
    fn finite_array_rejects_wrong_length_and_non_finite() {
        assert!(finite_array(&Some(vec![0.0, 1.0]), 3).is_none());
        assert!(finite_array(&Some(vec![0.0, 1.0, f64::NAN]), 3).is_none());
        assert_eq!(
            finite_array(&Some(vec![0.0, 1.0, 2.0]), 3),
            Some(json!([0.0, 1.0, 2.0]))
        );
    }

    #[test]
    fn error_sampling_fires_on_first_and_every_hundredth() {
        let c = AtomicU64::new(0);
        assert_eq!(sample(&c), Some(1));
        for _ in 2..ERROR_SAMPLE {
            assert_eq!(sample(&c), None);
        }
        assert_eq!(sample(&c), Some(ERROR_SAMPLE));
    }
}
