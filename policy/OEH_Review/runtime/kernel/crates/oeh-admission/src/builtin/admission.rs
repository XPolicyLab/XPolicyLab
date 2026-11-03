//! `admission` 插件：把准入服务挂上（`brains` 的 `Proposer`、`admission` 服务），并注册
//! 五个不依赖任何硬件的内置能力。
//!
//! - `say{text}` / `ask_human{question}`（speech, 无风险）：执行器什么也不做，"说过了"
//!   这件事由流水线的 `act/result` 记下来。其中 `say` 是五个里唯一**可被顶替**的
//!   （`overridable`）：一台真机器人自己有嗓子，内置的这个只是没有机器人时的兜底。
//! - `ask_slow_brain{q}`（query, 无风险）：执行器把问题记成一条 `brain/ask`。脑之间只经
//!   世界日志说话，所以"提问"也是一次**提案**——白名单、规则链、审批一个都不少。
//! - `stop`（control, 无风险）：撤销正在跑的 motion。
//! - `recover`（control, 高风险）：`EMERGENCY_STOP|FAULT → RECOVERING → STANDING`。这是
//!   急停唯一的出路，蒸馏器永远不碰这条边（见 `oeh_distill::fsm_driver` 的归属契约）。

use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Weak};

use oeh_kernel::clock::ClockService;
use oeh_kernel::factories::PluginFactory;
use oeh_kernel::fsm::{FsmState, RobotFsm};
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::state::StateRegistry;
use oeh_kernel::{Brains, Ctx, FrameStore, Plugin, PluginError, WorldLog};
use oeh_proto::{EventDraft, EventKind, Manifest};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::admission::{Admission, AdmissionConfig, SOURCE};
use crate::capability::{
    CancelReason, CapabilityClass, CapabilitySpec, ExecCtx, ExecOutcome, ExecutionCall, Executor,
    RiskLevel,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    required_rules: Vec<String>,
    #[serde(default = "d_ask_timeout_ms")]
    ask_timeout_ms: u64,
    #[serde(default = "d_lease_check_ms")]
    lease_check_ms: u64,
    #[serde(default = "d_lease_grace_ms")]
    lease_grace_ms: u64,
    #[serde(default = "d_recheck_ms")]
    recheck_ms: u64,
    #[serde(default = "d_estop_flag")]
    estop_flag: PathBuf,
}
fn d_ask_timeout_ms() -> u64 {
    20_000
}
fn d_lease_check_ms() -> u64 {
    100
}
fn d_lease_grace_ms() -> u64 {
    500
}
fn d_recheck_ms() -> u64 {
    200
}
fn d_estop_flag() -> PathBuf {
    PathBuf::from("/tmp/oeh_estop")
}
impl Default for Config {
    fn default() -> Self {
        Self {
            required_rules: Vec::new(),
            ask_timeout_ms: d_ask_timeout_ms(),
            lease_check_ms: d_lease_check_ms(),
            lease_grace_ms: d_lease_grace_ms(),
            recheck_ms: d_recheck_ms(),
            estop_flag: d_estop_flag(),
        }
    }
}
impl From<Config> for AdmissionConfig {
    fn from(c: Config) -> Self {
        Self {
            required_rules: c.required_rules,
            ask_timeout_ms: c.ask_timeout_ms,
            lease_check_ms: c.lease_check_ms,
            lease_grace_ms: c.lease_grace_ms,
            recheck_ms: c.recheck_ms,
            estop_flag: c.estop_flag,
        }
    }
}

/// `distill-estop` 那一行的 id。急停标志文件有**两个**读者，它是另一个。
const DISTILL_ESTOP: &str = "distill-estop";

/// 把一个可能是相对的路径规整成可比较的形式：按进程 cwd 绝对化，去掉 `.`，就地折叠 `..`。
///
/// **不碰文件系统。** `std::fs::canonicalize` 要求路径已经存在，而急停标志文件恰恰在
/// "没按下"的时候不存在——那是常态，不是例外。代价是符号链接不展开：`./run/estop` 和
/// `/var/oeh/estop` 即便是同一个文件也会被判成不一致。这个方向是对的（宁可让一份可疑的
/// profile 起不来，也不要让两个读者各盯各的），而且两行本来就该照抄同一个字符串。
fn normalised(p: &Path) -> PathBuf {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            // 根上的 `..` 还是根（posix 的规矩），别把根 pop 掉。
            Component::ParentDir if out.parent().is_some() => {
                out.pop();
            }
            Component::ParentDir => {}
            other => out.push(other),
        }
    }
    out
}

/// 急停的两条路必须指着同一个文件，对不上就在 boot 当场 fail-closed。
///
/// 急停的权威在内核外面（`estop-listener` 写一个标志文件）。内核这边有两个读者：
/// `distill-estop` 把它蒸馏成 `estop` 寄存器 + `safety/estop` 事件（脑靠它被打断），
/// 准入自己在取快照时也直接看一眼那个文件（`Admission::estop_engaged` 把两者取或，
/// `rule-estop` 读的是那个合并结果）。两边各有各的默认路径（都是 `/tmp/oeh_estop`），
/// 于是一份只改了其中一个的 profile 会让另一个继续盯着默认路径——那条路在急停那一刻会
/// 说"没按下"，而写这份 profile 的人完全有理由以为自己已经改好了。
///
/// 所以它是**配置错误**，不是运行时降级：按或取合并只在两条路都指对时才是纵深防御，
/// 一条指错的时候它只是在替一个错误打掩护。
///
/// `distill-estop` 不在表里 → 没什么可比的（只有准入这一个读者）。它在表里但没配
/// `flag_path` → 它用的是自己的默认值，和 [`d_estop_flag`] 是同一个路径（两个默认值
/// 是一份约定，`oeh_distill::estop` 那边的文档也这么写着）。配了但不是字符串 → 当场说
/// 清楚是这个字段的类型不对，而不是退回默认值再报一个"路径不一致"、指着一条没人写过的
/// 路径（`distill-estop` 自己起来的时候也会因为同一个字段失败，但那条消息落在另一行上）。
fn check_estop_agreement(ctx: &Ctx, mine: &Path) -> Result<(), PluginError> {
    let Some(sibling) = ctx.sibling_config(DISTILL_ESTOP) else {
        return Ok(());
    };
    let theirs = match sibling.get("flag_path") {
        None | Some(Value::Null) => d_estop_flag(),
        Some(Value::String(s)) => PathBuf::from(s),
        Some(other) => {
            return Err(PluginError::Config {
                plugin: "admission".into(),
                reason: format!("distill-estop.flag_path is not a string: {other}"),
            })
        }
    };
    if normalised(mine) == normalised(&theirs) {
        return Ok(());
    }
    Err(PluginError::Config {
        plugin: "admission".into(),
        reason: format!(
            "estop flag mismatch: admission.estop_flag={} distill-estop.flag_path={}",
            mine.display(),
            theirs.display()
        ),
    })
}

fn spec(name: &str, class: CapabilityClass, risk: RiskLevel, schema: Value) -> CapabilitySpec {
    CapabilitySpec {
        name: name.to_string(),
        class,
        risk_level: risk,
        params_schema: schema,
        lease_ttl_s: 5.0,
        overridable: false,
    }
}

/// 标成"适配器可以顶替"。**只有 `say` 用它**，理由见下面那一行的注释。
fn overridable(mut s: CapabilitySpec) -> CapabilitySpec {
    s.overridable = true;
    s
}

fn one_string(field: &str) -> Value {
    json!({
        "type": "object",
        "properties": {field: {"type": "string"}},
        "required": [field],
        "additionalProperties": false
    })
}

/// `say` / `ask_human`：没有副作用的能力，把自己的参数原样回给 `act/result`。
struct Echo {
    field: &'static str,
}
impl Executor for Echo {
    fn execute(&self, call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let field = self.field;
        let value = call.params.get(field).cloned().unwrap_or(Value::Null);
        Box::pin(async move {
            ExecOutcome {
                ok: true,
                result: json!({ field: value }),
                error: None,
            }
        })
    }
}

/// 一条 `brain/ask` 的 salience。和 `act/admitted` / `act/refused` 同级（150）：一个脑向
/// 另一个脑提问是要被看见的事，但它不是安全事件。
const SAL_BRAIN_ASK: u8 = 150;

/// `brain/ask.q` 的上限（1 KiB），和 `brain/answer.text` 同一个数
/// （`oeh_brain::native::MAX_ANSWER_TEXT`）。
///
/// 提问和回答是一对，日志里没有理由一边有界一边没有：`q` 是**模型给的一个工具参数**，
/// 8 MB 只隔着一次 `ask_slow_brain` 调用，而世界日志的 payload 本身没有上限。问题的
/// 全文另有去处——提案的 `act/proposed.params`，靠同一个 `trace_id` 对得上。
const MAX_ASK_Q: usize = 1024;

/// 截到 `limit` 字节，且不切坏 UTF-8。返回 `(内容, 是否截断)`。
fn cap(text: &str, limit: usize) -> (&str, bool) {
    if text.len() <= limit {
        return (text, false);
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], true)
}

/// `ask_slow_brain{q}`：把问题记进世界日志，仅此而已。
///
/// **谁来回答不由这里决定**——由世界日志和各个脑自己的 wake 策略决定（`wake/slow.yaml`
/// 对 `brain/ask` 是 `wake`）。执行器不认识"慢脑"这个概念，也没有任何脑对脑的通道：多挂
/// 一个脑、换一个脑的名字，这一行代码都不用动。
///
/// payload 里 `brain_id` 与 `from` 是同一个值（提问的那个脑）：`brain_id` 是全局约束——
/// 每条 `brain/*` 都要有，内核的防自激守卫读的就是它（提问者因此不会被自己的提问叫醒）；
/// `from` 是这条事件自己的语义字段，回答靠它知道该回给谁。
///
/// `ask_id` 就是提案 id：提问和回答之间没有第二套编号，`act/*` 那一串本来就用它。
struct AskSlowBrain {
    log: Arc<WorldLog>,
}
impl Executor for AskSlowBrain {
    fn execute(&self, call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let log = self.log.clone();
        Box::pin(async move {
            let p = call.proposal;
            let ask_id = p.proposal_id.0;
            // schema 已经在准入的词汇这一步校过 `q` 是字符串；这里仍然不 `unwrap`——
            // 执行器 panic 掉的代价是整条流水线，而兜底只要一个空串。
            let q = call.params.get("q").and_then(Value::as_str).unwrap_or("");
            // 截断了就明说（没截断时这个字段根本不出现）：一个读日志的人不能把半个
            // 问题当成全文，回答的那个脑也不能。见 [`MAX_ASK_Q`]。
            let (q, cut) = cap(q, MAX_ASK_Q);
            let mut payload = json!({
                "brain_id": p.brain_id,
                "from": p.brain_id,
                "q": q,
                "ask_id": ask_id,
            });
            if cut {
                payload["truncated"] = Value::Bool(true);
            }
            log.append(
                EventDraft::new(
                    EventKind::new("brain/ask").expect("static kind"),
                    SOURCE,
                    payload,
                )
                .with_salience(SAL_BRAIN_ASK)
                .with_trace_id(p.trace_id.as_str()),
            );
            ExecOutcome {
                ok: true,
                result: json!({ "ask_id": ask_id }),
                error: None,
            }
        })
    }
}

/// `stop`：撤销正在跑的 motion。持 `Weak`——能力注册表在 `Admission` 里面，拿强引用会
/// 让服务永远活着。
struct Stop {
    adm: Weak<Admission>,
}
impl Executor for Stop {
    fn execute(&self, _call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let adm = self.adm.upgrade();
        Box::pin(async move {
            match adm {
                Some(a) => {
                    let cancelled = a.cancel_active(CancelReason::Stop);
                    ExecOutcome {
                        ok: true,
                        result: json!({ "cancelled": cancelled }),
                        error: None,
                    }
                }
                None => ExecOutcome {
                    ok: false,
                    result: Value::Null,
                    error: Some("admission service is gone".into()),
                },
            }
        })
    }
}

/// `recover`：把 FSM 从急停/故障带回 STANDING。不在这两态时失败——`recover` 不是一个
/// "顺便把状态机推一推"的能力。
struct Recover {
    fsm: Arc<RobotFsm>,
}
impl Executor for Recover {
    fn execute(&self, _call: ExecutionCall, _ctx: ExecCtx) -> BoxFuture<ExecOutcome> {
        let fsm = self.fsm.clone();
        Box::pin(async move {
            let from = fsm.state();
            if !matches!(from, FsmState::EmergencyStop | FsmState::Fault) {
                return ExecOutcome {
                    ok: false,
                    result: Value::Null,
                    error: Some(format!(
                        "recover needs EMERGENCY_STOP or FAULT, fsm is {from}"
                    )),
                };
            }
            for to in [FsmState::Recovering, FsmState::Standing] {
                if let Err(e) = fsm.transition(to, "operator recover", SOURCE) {
                    return ExecOutcome {
                        ok: false,
                        result: Value::Null,
                        error: Some(e.to_string()),
                    };
                }
            }
            ExecOutcome {
                ok: true,
                result: json!({"from": from.to_string(), "to": FsmState::Standing.to_string()}),
                error: None,
            }
        })
    }
}

/// 提供 `admission` 服务：脑的提案经它变成执行。
pub struct AdmissionPlugin;

impl Plugin for AdmissionPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new("admission")
            .inject(&["clock", "world-log", "state", "fsm", "brains"])
            .provide(&["admission"])
    }
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = if config.is_null() {
            Config::default()
        } else {
            ctx.parse_config(config)?
        };
        // 先检查再挂任何东西：配置错误不该留下半个装好的服务。
        check_estop_agreement(ctx, &cfg.estop_flag)?;
        let clock = ctx.get::<ClockService>("clock")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let state = ctx.get::<StateRegistry>("state")?;
        let fsm = ctx.get::<RobotFsm>("fsm")?;
        let brains = ctx.get::<Brains>("brains")?;
        // 帧存储是**可选**的：不在 `inject` 里（挂不挂它都不该影响准入判不判得了），
        // 用 `try_get` 取。取到了，`act/started` / `act/result` 就带上 lowstate 的最新
        // 帧引用；没取到就不带。代价说清楚：`try_get` 看的是**此刻**注册表里有什么，
        // 所以 profile 里 `frame-store` 必须排在 `admission` 前面——排在后面的话这一行
        // 照常起来，只是从此不写 `frame_refs`（声明式的 `inject` 才和行的先后无关）。
        let frames = ctx.try_get::<FrameStore>("frame-store");
        let adm = match frames {
            Some(f) => Admission::with_frames(
                clock.0.clone(),
                log.clone(),
                state,
                fsm.clone(),
                cfg.into(),
                f,
            ),
            None => Admission::new(clock.0.clone(), log.clone(), state, fsm.clone(), cfg.into()),
        }
        .map_err(|e| PluginError::Other(e.to_string()))?;

        let builtins: Vec<(CapabilitySpec, Arc<dyn Executor>)> = vec![
            (
                // 五个内置里唯一可以被顶替的一个：**说话是身体的一部分**。一台带扬声器
                // 的机器人自己有嗓子，而这里这个 `say` 只是把话记进世界日志——同一个
                // 名字，只有一个真的发出声音。所以它是**兜底**：没有适配器的时候它保证
                // `say` 在能力面上（脑的工具表不因为少一台机器人而变形），适配器一上线
                // 就让位，走了再接回来。
                //
                // 别的四个反过来：`stop` 是撤销当前执行的那一下、`recover` 是急停唯一的
                // 出路、`ask_*` 是脑与人/脑与脑的通道——语义全在内核这一侧。让适配器
                // 顶替 `stop`，就是把"叫停"交给了要被叫停的那一方。
                overridable(spec(
                    "say",
                    CapabilityClass::Speech,
                    RiskLevel::None,
                    one_string("text"),
                )),
                Arc::new(Echo { field: "text" }),
            ),
            (
                spec(
                    "ask_human",
                    CapabilityClass::Speech,
                    RiskLevel::None,
                    one_string("question"),
                ),
                Arc::new(Echo { field: "question" }),
            ),
            (
                spec(
                    "ask_slow_brain",
                    CapabilityClass::Query,
                    RiskLevel::None,
                    one_string("q"),
                ),
                Arc::new(AskSlowBrain { log }),
            ),
            (
                spec(
                    "stop",
                    CapabilityClass::Control,
                    RiskLevel::None,
                    json!({"type": "object", "additionalProperties": false}),
                ),
                Arc::new(Stop {
                    adm: Arc::downgrade(&adm),
                }),
            ),
            (
                spec(
                    "recover",
                    CapabilityClass::Control,
                    RiskLevel::High,
                    json!({"type": "object", "additionalProperties": false}),
                ),
                Arc::new(Recover { fsm }),
            ),
        ];
        for (s, e) in builtins {
            let d = adm
                .capabilities()
                .register(s, e)
                .map_err(|e| PluginError::Other(e.to_string()))?;
            ctx.on_dispose(d);
        }
        ctx.on_dispose(brains.set_proposer(adm.clone()));
        ctx.provide("admission", adm)
    }
}

inventory::submit! { PluginFactory { id: "admission", build: || Box::new(AdmissionPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_and_empty_config_share_the_defaults() {
        let manual = Config::default();
        let empty: Config = serde_json::from_value(json!({})).unwrap();
        assert_eq!(manual.ask_timeout_ms, empty.ask_timeout_ms);
        assert_eq!(manual.lease_check_ms, empty.lease_check_ms);
        assert_eq!(manual.estop_flag, empty.estop_flag);
        assert_eq!(empty.ask_timeout_ms, 20_000);
        assert_eq!(empty.lease_check_ms, 100);
        assert_eq!(manual.recheck_ms, empty.recheck_ms);
        assert_eq!(empty.recheck_ms, 200);
        assert_eq!(manual.lease_grace_ms, empty.lease_grace_ms);
        assert_eq!(empty.lease_grace_ms, 500);
        assert_eq!(empty.estop_flag, PathBuf::from("/tmp/oeh_estop"));
        assert!(empty.required_rules.is_empty());
    }

    /// M2：`q` 和 `brain/answer.text` 同界，截断了就带记号，且永远不切坏一个字符。
    #[test]
    fn an_over_long_question_is_capped_the_same_way_an_answer_is() {
        let (short, cut) = cap("你好", MAX_ASK_Q);
        assert_eq!((short, cut), ("你好", false));
        let long = "\u{4e00}".repeat(MAX_ASK_Q);
        let (cut_text, cut) = cap(&long, MAX_ASK_Q);
        assert!(cut);
        assert!(cut_text.len() <= MAX_ASK_Q);
        // 切在字符边界上：三字节字符，1024 不是 3 的倍数。
        assert_eq!(cut_text.chars().count(), MAX_ASK_Q / 3);
    }

    #[test]
    fn unknown_config_key_is_rejected() {
        assert!(serde_json::from_value::<Config>(json!({"ask_timeout": 1})).is_err());
    }
}
