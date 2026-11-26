use oeh_kernel::factories::PluginFactory;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Ctx, FrameStore, Plugin, PluginError, WorldLog};
use oeh_proto::payload::LowState;
use oeh_proto::{EventDraft, EventKind, Frame, Manifest, WorldEvent};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "d_stream")]
    stream_id: String,
    #[serde(default = "d_stand")]
    stand_gravity_z: f64,
    #[serde(default = "d_tip")]
    tipping_gravity_z: f64,
}
fn d_stream() -> String {
    "lowstate".into()
}
fn d_stand() -> f64 {
    -0.85
}
fn d_tip() -> f64 {
    -0.6
}
impl Default for Config {
    fn default() -> Self {
        Self {
            stream_id: d_stream(),
            stand_gravity_z: d_stand(),
            tipping_gravity_z: d_tip(),
        }
    }
}

/// 解析失败每这么多条采样一次事件；第 1 条总是发。逐条发会把世界日志淹掉（流是 500 Hz）。
const ERROR_SAMPLE: u64 = 100;

/// 流 schema 校验的结论。对不上就忽略该流：拿别的 schema 的字节去猜 `gravity_proj_z`
/// 会让安全 FSM 基于垃圾做判断，宁可不动。结论跟着 schema 走，不是一锤定音——流可以被
/// 撤销再以别的 schema 注册（订阅者会被接回新的 `StreamBuf`）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum SchemaCheck {
    Unchecked,
    Ok,
    Mismatch,
}

enum Verdict {
    Proceed,
    Ignore,
    Reject(Option<String>),
}

fn distiller_error(payload: serde_json::Value) -> EventDraft {
    EventDraft::new(
        EventKind::new("health/distiller_error").expect("static"),
        "distill-fsm",
        payload,
    )
    .with_salience(200)
    .with_collapse_key("distiller_error:distill-fsm")
}

/// FSM 归属契约（谁能发哪条边）：
///
/// - **distill-fsm（本插件）**：`BOOT → STANDING`（策略激活且直立）、`* → EMERGENCY_STOP`
///   （倾倒）、`* → FAULT`（`health/stale{stale:true}`）、`FAULT → RECOVERING → STANDING`
///   （同一条流重新变新鲜）。自己发的 FAULT 由自己收回。
/// - **admission（Plan 2b）**：`STANDING → ENGAGED → ACTING → STANDING`，以及
///   `EMERGENCY_STOP → RECOVERING`——急停只能由操作员/审批解除，蒸馏器永远不碰这条边。
///
/// 迁移都走 [`RobotFsm::transition`]，它在世界日志观察者里被调用：因此本插件的观察者
/// 只对 `health/stale` 反应，绝不对自己产出的 `safety/fsm` 反应（否则会重入那把锁）。
pub struct FsmDriverPlugin;
impl Plugin for FsmDriverPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("distill-fsm").inject(&["frame-store", "fsm", "world-log", "state"])
    }
    fn apply(&self, ctx: &Ctx, config: &serde_json::Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        let fs = ctx.get::<FrameStore>("frame-store")?;
        let fsm = ctx.get::<RobotFsm>("fsm")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let bad = Arc::new(AtomicU64::new(0));
        // 结论 + 得出它时那条流声明的 schema。
        let schema = Arc::new(Mutex::new((SchemaCheck::Unchecked, None::<String>)));
        let (fsm2, state2, cfg2, bad2) = (fsm.clone(), state.clone(), cfg.clone(), bad.clone());
        let (log2, fs2) = (log.clone(), fs.clone());
        let sub = fs
            .subscribe(
                &cfg.stream_id,
                Arc::new(move |f: &Frame| {
                    // 每帧读一次流声明的 schema：变了就重查，否则换过 schema 的流会一直
                    // 吃着上一次的结论（Ok 的话就是拿别的编码当 LowState 解）。
                    let found = fs2.schema_of(&cfg2.stream_id);
                    // 判定在 `schema` 锁内一次完成，`append` 在锁外——`WorldLog::append`
                    // 会同步回调全部观察者，不能带着这把私有锁跨进去。
                    let verdict = {
                        let mut g = schema.lock();
                        let (check, seen) = &mut *g;
                        if *check == SchemaCheck::Unchecked || *seen != found {
                            *seen = found.clone();
                            if found.as_deref() == Some(LowState::SCHEMA) {
                                *check = SchemaCheck::Ok;
                                Verdict::Proceed
                            } else {
                                *check = SchemaCheck::Mismatch;
                                Verdict::Reject(found)
                            }
                        } else if *check == SchemaCheck::Ok {
                            Verdict::Proceed
                        } else {
                            Verdict::Ignore
                        }
                    };
                    match verdict {
                        Verdict::Ignore => return,
                        Verdict::Reject(found) => {
                            log2.append(distiller_error(json!({
                                    "plugin": "distill-fsm",
                                    "stream": cfg2.stream_id,
                                    "count": 1,
                                    "reason": "schema mismatch",
                                    "expected": LowState::SCHEMA,
                                    "found": found,
                            })));
                            return;
                        }
                        Verdict::Proceed => {}
                    }
                    let Ok(ls) = LowState::from_bytes(&f.payload) else {
                        let n = bad2.fetch_add(1, Ordering::Relaxed) + 1;
                        state2.set(
                            &format!("parse_errors.{}", cfg2.stream_id),
                            json!(n),
                            "distill-fsm",
                        );
                        if n == 1 || n % ERROR_SAMPLE == 0 {
                            log2.append(distiller_error(json!({
                                "plugin": "distill-fsm",
                                "stream": cfg2.stream_id,
                                "count": n,
                            })));
                        }
                        return;
                    };
                    state2.set("gravity_proj_z", json!(ls.gravity_proj_z), "distill-fsm");
                    state2.set(
                        "rl_policy_active",
                        json!(ls.rl_policy_active),
                        "distill-fsm",
                    );
                    let s = fsm2.state();
                    if s == FsmState::Boot
                        && ls.rl_policy_active
                        && ls.gravity_proj_z <= cfg2.stand_gravity_z
                    {
                        if let Err(e) = fsm2.transition(
                            FsmState::Standing,
                            "policy active and upright",
                            "distill-fsm",
                        ) {
                            tracing::debug!(%e, "fsm");
                        }
                    } else if fsm2.allows_motion() && ls.gravity_proj_z > cfg2.tipping_gravity_z {
                        if let Err(e) =
                            fsm2.transition(FsmState::EmergencyStop, "tipping", "distill-fsm")
                        {
                            tracing::debug!(%e, "fsm");
                        }
                    }
                }),
            )
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(sub);
        let stale_kind = EventKind::new("health/stale").expect("static");
        let stream = cfg.stream_id.clone();
        // 「这条 FAULT 是我发的」：只有自己驱动出来的 FAULT 才由自己收回。别人（操作员、
        // admission、别的插件）把 FSM 按进 FAULT 时，一条 fresh 事件不能替他们撤销。
        let raised_fault = Arc::new(AtomicBool::new(false));
        let obs = log.observe(Arc::new(move |ev: &WorldEvent| {
            if ev.kind != stale_kind || ev.payload["stream"] != stream.as_str() {
                return;
            }
            let is_stale = ev.payload["stale"] == json!(true);
            let s = fsm.state();
            if is_stale && s != FsmState::Fault {
                match fsm.transition(FsmState::Fault, "lowstate stale", "distill-fsm") {
                    Ok(_) => raised_fault.store(true, Ordering::SeqCst),
                    Err(e) => tracing::debug!(%e, "fsm"),
                }
            } else if !is_stale && s != FsmState::Fault {
                // 已经有人把它带离 FAULT 了：我们那条 FAULT 不再是当前状态，撤销权作废。
                raised_fault.store(false, Ordering::SeqCst);
            } else if !is_stale && raised_fault.load(Ordering::SeqCst) {
                // 自己发的 FAULT 自己收回。EMERGENCY_STOP 不走这里：它只能由操作员/审批解除。
                for to in [FsmState::Recovering, FsmState::Standing] {
                    if let Err(e) = fsm.transition(to, "lowstate fresh again", "distill-fsm") {
                        tracing::debug!(%e, "fsm");
                        break;
                    }
                }
                // 真的离开 FAULT 了才交还撤销权：中途失败的话下一条 fresh 事件还能再试。
                raised_fault.store(fsm.state() == FsmState::Fault, Ordering::SeqCst);
            }
        }));
        ctx.on_dispose(obs);
        Ok(())
    }
}
inventory::submit! { PluginFactory { id: "distill-fsm", build: || Box::new(FsmDriverPlugin) } }
