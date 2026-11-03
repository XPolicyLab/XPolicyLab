//! `distill-hold`：从账本事件里蒸馏"抓空"与"打滑"两个事实。
//!
//! 输入是 `obs/arx_x5_ledger`（CONTRACT §4.1）里同一条事件上的 `gripper_cmd`（**命令**）
//! 与 `gripper_measured`（**测量**）。这两个量的来源不同是整条路线的根因（设计 §1.1）：
//! 命令来自 `executed_action[6]/[13]`，测量来自 `measured_gripper_openings`。拿
//! `states[6]/[13]` 当测量会让"夹住物体"的判定对每一次真实抓取都为假。
//!
//! 两个判定，**"抓空"压过"打滑"**：一次塌到 `empty_meas_below` 以下是更强、更终局的事实，
//! 同一个物理事件报两条会把脑叫醒两次。
//!
//! 只发**边沿**：
//!
//! - `grasp_empty`：进入"命令闭合但测量贴到 0"这个状态的那一条事件发一次；状态持续不再发。
//! - `hold_slip`：测量相对**上一条**事件下降超过 `slip_delta` 时发；它天然就是边沿。
//!
//! 某一臂的命令或测量在这条事件里未知时，这条臂被跳过且**不更新锁存**——一条读不到的
//! 观测不该把"上一次测量"抹掉。`episode_id` 变化时两臂的锁存全部清空。
//!
//! 寄存器 `grasp_empty.<arm>` / `hold_slip.<arm>` 写的是**这条事件的判定**（不是"曾经发生过"），
//! 所以边沿被抑制的那条事件上 `grasp_empty` 仍然是 `true`——事实还在，只是不再叫醒脑。
//!
//! # "持物中"用账本里的 `holding`，这条蒸馏器不自己重算
//!
//! `hold_slip` 的那句"持物中"必须和系统其它地方的"持物中"是同一件事。全系统的判定只有
//! 一个出处：`oeh_vla.arx_ledger.holding(cmd, measured, closed_below=0.5,
//! contact_above=0.05)`，它算出来的结果由 Python 侧写进事件的 `holding` 字段，再由
//! `distill-arx-ledger` 写进 `holding.<arm>` 寄存器，`rule-carry-speed` 与
//! `rule-hold-no-rotation` 读的都是它。
//!
//! 这里曾经**另外推了一遍**：`prev_holding = cmd < empty_cmd_below && meas >= empty_meas_below`，
//! 也就是把"持物"的命令阈值当成了 `grasp_empty` 的 0.3。两个阈值不是一回事——0.3 是"这条
//! 命令闭到几乎贴死"，0.5 才是"这是一条闭合命令"。后果是一次 `set_gripper(opening=0.40)`
//! 的轻夹（宽物体的正常握法，`holding` 为真、两条准入规则都按持物处理）在这条蒸馏器眼里
//! 不算持物，于是 0.71 → 0.55 的滑脱一条事件都不发，脑永远不会被叫醒。
//!
//! 所以改成直接读 `holding`，重复的谓词整个删掉：三处对"持物"的理解从此不可能再漂开。
//! `holding.<arm>` 为 `null`（未知）时**保持上一次的锁存**，与"命令或测量未知就跳过这条臂"
//! 是同一条纪律——读不到的一条观测不该把已知的状态抹掉。

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use oeh_kernel::factories::PluginFactory;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::arx_ledger::{distiller_error, sample, BoolPair, NumPair, LEDGER_KIND};

pub const ID: &str = "distill-hold";

fn d_empty_cmd() -> f64 {
    0.3
}
fn d_empty_meas() -> f64 {
    0.05
}
fn d_slip() -> f64 {
    0.1
}

#[derive(Deserialize, Clone, Copy)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_empty_cmd")]
    empty_cmd_below: f64,
    #[serde(default = "d_empty_meas")]
    empty_meas_below: f64,
    #[serde(default = "d_slip")]
    slip_delta: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            empty_cmd_below: d_empty_cmd(),
            empty_meas_below: d_empty_meas(),
            slip_delta: d_slip(),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
struct HoldPayload {
    episode_id: String,
    #[serde(default)]
    gripper_cmd: NumPair,
    #[serde(default)]
    gripper_measured: NumPair,
    /// 权威的"是否持物"，由 Python 侧的 `arx_ledger.holding` 算好写进来。
    #[serde(default)]
    holding: BoolPair,
    #[serde(default)]
    native_step: Option<i64>,
}

#[derive(Default, Clone, Copy)]
struct ArmLatch {
    prev_measured: Option<f64>,
    prev_holding: bool,
    prev_empty: bool,
}

#[derive(Default)]
struct Latched {
    episode: Option<String>,
    left: ArmLatch,
    right: ArmLatch,
}

impl Latched {
    fn arm(&mut self, arm: &str) -> &mut ArmLatch {
        if arm == "left" {
            &mut self.left
        } else {
            &mut self.right
        }
    }
}

/// 一条臂在这条事件上的结论。
struct ArmVerdict {
    arm: &'static str,
    cmd: f64,
    measured: f64,
    previous: Option<f64>,
    empty: bool,
    slip: bool,
    emit_empty: bool,
    emit_slip: bool,
}

pub struct ArxHoldPlugin;

impl Plugin for ArxHoldPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["world-log", "state"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let kind = EventKind::new(LEDGER_KIND).expect("static");
        let latched = Arc::new(Mutex::new(Latched::default()));
        let bad = Arc::new(AtomicU64::new(0));
        let log2 = log.clone();
        let disposer = log.observe(Arc::new(move |ev: &WorldEvent| {
            // 第一句按 kind 过滤并早退：`append` 同步回调所有观察者，包括我们自己发的两条。
            if ev.kind != kind {
                return;
            }
            let p: HoldPayload = match serde_json::from_value(ev.payload.clone()) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(n) = sample(&bad) {
                        log2.append(distiller_error(ID, LEDGER_KIND, n, e.to_string()));
                    }
                    return;
                }
            };
            // 判定 + 锁存更新都在锁内一次做完；`append` 在锁外——不能带着这把私有锁跨进去。
            let verdicts: Vec<ArmVerdict> = {
                let mut l = latched.lock();
                if l.episode.as_deref() != Some(p.episode_id.as_str()) {
                    *l = Latched {
                        episode: Some(p.episode_id.clone()),
                        ..Default::default()
                    };
                }
                let mut out = Vec::with_capacity(2);
                for (arm, cmd, meas, held) in [
                    (
                        "left",
                        p.gripper_cmd.left,
                        p.gripper_measured.left,
                        p.holding.left,
                    ),
                    (
                        "right",
                        p.gripper_cmd.right,
                        p.gripper_measured.right,
                        p.holding.right,
                    ),
                ] {
                    // 任一通道未知 → 跳过这条臂，锁存保持不动。
                    let (Some(cmd), Some(meas)) = (cmd, meas) else {
                        continue;
                    };
                    if !cmd.is_finite() || !meas.is_finite() {
                        continue;
                    }
                    let a = l.arm(arm);
                    let empty = cmd < cfg.empty_cmd_below && meas < cfg.empty_meas_below;
                    // "持物中"是**上一条事件**上的事实：这一条测到的下降是相对上一条量出来的。
                    let slip = !empty
                        && a.prev_holding
                        && a.prev_measured
                            .map(|prev| prev - meas > cfg.slip_delta)
                            .unwrap_or(false);
                    let v = ArmVerdict {
                        arm,
                        cmd,
                        measured: meas,
                        previous: a.prev_measured,
                        empty,
                        slip,
                        emit_empty: empty && !a.prev_empty,
                        emit_slip: slip,
                    };
                    a.prev_measured = Some(meas);
                    // 账本里的权威判定；`null`（未知）保持上一次的锁存，不重算、不清空。
                    a.prev_holding = held.unwrap_or(a.prev_holding);
                    a.prev_empty = empty;
                    out.push(v);
                }
                out
            };
            if verdicts.is_empty() {
                return;
            }
            let mut kv: Vec<(String, Value)> = Vec::with_capacity(4);
            for v in &verdicts {
                kv.push((format!("grasp_empty.{}", v.arm), json!(v.empty)));
                kv.push((format!("hold_slip.{}", v.arm), json!(v.slip)));
            }
            let borrowed: Vec<(&str, Value)> =
                kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            state.set_many(&borrowed, ID);
            for v in &verdicts {
                if v.emit_empty {
                    log2.append(
                        EventDraft::new(
                            EventKind::new("obs/grasp_empty").expect("static"),
                            ID,
                            json!({
                                "arm": v.arm,
                                "gripper_cmd": v.cmd,
                                "gripper_measured": v.measured,
                                "native_step": p.native_step,
                            }),
                        )
                        .with_salience(210)
                        .with_collapse_key(format!("grasp_empty:{}", v.arm)),
                    );
                } else if v.emit_slip {
                    log2.append(
                        EventDraft::new(
                            EventKind::new("obs/hold_slip").expect("static"),
                            ID,
                            json!({
                                "arm": v.arm,
                                "gripper_cmd": v.cmd,
                                "previous": v.previous,
                                "gripper_measured": v.measured,
                                "drop": v.previous.map(|prev| prev - v.measured),
                                "native_step": p.native_step,
                            }),
                        )
                        .with_salience(190)
                        .with_collapse_key(format!("hold_slip:{}", v.arm)),
                    );
                }
            }
        }));
        ctx.on_dispose(disposer);
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(ArxHoldPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_design_table() {
        let c = Config::default();
        assert!((c.empty_cmd_below - 0.3).abs() < 1e-12);
        assert!((c.empty_meas_below - 0.05).abs() < 1e-12);
        assert!((c.slip_delta - 0.1).abs() < 1e-12);
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert!((empty.empty_cmd_below - 0.3).abs() < 1e-12);
    }

    #[test]
    fn partial_config_keeps_the_other_defaults() {
        let c: Config = serde_json::from_value(json!({"slip_delta": 0.2})).unwrap();
        assert!((c.slip_delta - 0.2).abs() < 1e-12);
        assert!((c.empty_cmd_below - 0.3).abs() < 1e-12);
        assert!((c.empty_meas_below - 0.05).abs() < 1e-12);
    }

    #[test]
    fn typoed_key_is_an_error() {
        assert!(serde_json::from_value::<Config>(json!({"slip_deltaa": 0.2})).is_err());
    }
}
