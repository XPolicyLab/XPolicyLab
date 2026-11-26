//! 规则链的判定结果（`Verdict`）、规则合约（`Rule`）与链评估（`evaluate_chain`）。
//!
//! 这个模块是纯逻辑：不碰锁、不碰时钟、不做 I/O。`RuleCtx` 把一次判定需要的只读快照
//! （当前时间、FSM 态、蒸馏状态、能力描述、estop 是否按下）打包传进来，规则读它、
//! 改 `params`、返回新的 `Verdict`。
//!
//! 单调守卫：一条规则只能把判定收紧（Allow → Ask → Deny），不能放松。谁要是返回了
//! 一个 rank 更低的判定，链评估把它当成篡改（tamper）处理，强制改成 Deny，并在
//! trace 里如实记下来——这样审计的时候能看出"哪条规则想放水"，而不是被它悄悄得逞。

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use oeh_kernel::fsm::FsmState;
use oeh_kernel::state::StateEntry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::capability::CapabilitySpec;

/// 一次判定的结果。三档从松到紧：`Allow`（放行）、`Ask`（需要人工确认，带原因）、
/// `Deny`（拒绝，带原因）。序列化成 `{"verdict": "allow"}` / `{"verdict": "ask", "reason": ..}`
/// 这样的带标签形式，方便直接落盘或者传给审批 UI。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "lowercase")]
pub enum Verdict {
    Allow,
    Ask { reason: String },
    Deny { reason: String },
}

impl Verdict {
    /// 紧度排序：`Allow` = 0（最松），`Ask` = 1，`Deny` = 2（最紧）。
    pub fn rank(&self) -> u8 {
        match self {
            Verdict::Allow => 0,
            Verdict::Ask { .. } => 1,
            Verdict::Deny { .. } => 2,
        }
    }

    /// `self` 是否至少和 `than` 一样紧（rank 不更低）。链评估用它做单调守卫：一条规则
    /// 的输出必须对输入满足这个关系，否则视为篡改。
    pub fn is_tighter_or_equal(&self, than: &Verdict) -> bool {
        self.rank() >= than.rank()
    }
}

/// 一次规则求值需要的只读上下文：当前内核时间、FSM 态、蒸馏状态快照、被提案的能力
/// 描述，以及 estop 是否按下。规则只能读，不能改——要改的是传给 `evaluate` 的
/// `params`。
pub struct RuleCtx<'a> {
    pub now_ns: u64,
    pub fsm: FsmState,
    pub state: &'a BTreeMap<String, StateEntry>,
    pub spec: &'a CapabilitySpec,
    pub estop_engaged: bool,
}

impl RuleCtx<'_> {
    /// 蒸馏状态里某个键的当前值（如果存在）。
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.state.get(key).map(|e| &e.value)
    }

    /// 便捷读取：把值当 `f64` 解释（`serde_json::Value::as_f64`，接受整数和浮点数）。
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.value(key).and_then(|v| v.as_f64())
    }

    /// 便捷读取：把值当 `bool` 解释。
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.value(key).and_then(|v| v.as_bool())
    }

    /// 某个键的值有多“旧”：`now_ns` 减去它最后一次写入时的内核时间戳
    /// （`StateEntry::t_kernel`），用饱和减法防止时钟倒退造成下溢。
    pub fn age_ns(&self, key: &str) -> Option<u64> {
        self.state
            .get(key)
            .map(|e| self.now_ns.saturating_sub(e.t_kernel))
    }
}

/// 一条规则：给定（可变的）params、只读上下文和链里上一条规则留下来的判定，产出这条
/// 规则自己的判定。规则可以就地修改 `params`（比如把超速的 `vx` 夹到上限），链评估
/// 会对比修改前后的值并记进 trace。
///
/// 规则实现必须是 `Send + Sync`：它们会被塞进 `Arc<dyn Rule>`，可能被多个线程持有。
pub trait Rule: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, params: &mut Value, ctx: &RuleCtx, incoming: Verdict) -> Verdict;

    /// 判定 + 这条规则想留在审计 trace 里的东西（[`RuleTrace::detail`]）。
    ///
    /// 默认实现就是 `evaluate`，不留任何 detail——链评估只调这一个入口，所以想留痕的
    /// 规则覆写它、其余规则一个字都不用改。
    ///
    /// 为什么需要这么一个入口：规则唯一能改的东西是 `params`，而 `params` 是**真的会
    /// 交给执行器去跑的那一份**——把"这条轨迹最近处 0.3 m"这种算给人看的中间量写进去，
    /// 就等于在一次执行的输入里塞进一个执行器没要过的键（还会多出一条 `act/clamped`，
    /// 并且要重新过一遍能力自己的 schema）。留痕和改参数是两件事，走两条路。
    fn evaluate_detailed(
        &self,
        params: &mut Value,
        ctx: &RuleCtx,
        incoming: Verdict,
    ) -> (Verdict, Option<Value>) {
        (self.evaluate(params, ctx, incoming), None)
    }
}

/// 一条规则在链里跑过之后留下的审计记录：它收到的判定、它（实际生效的）判定、是否
/// 篡改/panic、它对 params 做了什么改动（如果有），以及它自己想留下的说明（如果有）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleTrace {
    pub rule: String,
    pub incoming: Verdict,
    pub outgoing: Verdict,
    #[serde(default)]
    pub tamper: bool,
    #[serde(default)]
    pub panicked: bool,
    /// `Some({"before": .., "after": ..})`，仅在这条规则实际改了 params 且输入判定
    /// 不是 `Deny` 的时候才有值。
    #[serde(default)]
    pub clamped: Option<Value>,
    /// 规则自己留下的说明（[`Rule::evaluate_detailed`]）：算了什么、看到了什么。只进
    /// 审计（`act/admitted.verdict_trace` / `act/refused.trace`），**不进 params**，
    /// 因此不影响交给执行器的那一份、也不会多出一条 `act/clamped`。panic 掉的规则不
    /// 留 detail（那不是它"完成"的输出）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

/// 整条链跑完之后的结果：最终判定、（可能被规则改过的）params、每条规则的 trace，
/// 以及"链里是否发生过篡改"的汇总标记。
pub struct ChainResult {
    pub verdict: Verdict,
    pub params: Value,
    pub trace: Vec<RuleTrace>,
    pub tampered: bool,
}

/// 跑一整条规则链。
///
/// 初始判定：`rules` 非空且 `required_rules`（按名字，即 tuple 里的 `String`）全部在
/// `rules` 里出现，才是 `Allow`；否则是 `Ask{reason}`，把缺失的规则名字列出来（`rules`
/// 为空时直接说"没有注册规则"）。这个初始判定只影响喂给第一条规则的 `incoming`——
/// **不会**跳过规则循环本身：即便因为缺了某个必需规则而以 `Ask` 起步，所有已注册的
/// 规则依然按顺序全部跑一遍。这是有意为之的安全属性：如果因为"缺一个规则"就跳过
/// 剩下已经注册好的规则（比如一个真正做急停检查的 `rule-estop`），那些规则就永远
/// 不会运行，而这个起步的 `Ask` 完全可能被后续别的审批流程直接放行——相当于"少配一
/// 条规则"反而绕过了"已经配好的"规则。所以：已注册的规则永远全部跑一遍；起步的
/// `Ask` 只是像任何其他 `incoming` 一样，可以被后面的规则收紧（比如某条规则直接
/// `Deny`），也一样受单调守卫约束（谁想把这个 `Ask` 放松成 `Allow`，一样按 tamper
/// 处理）。
///
/// 之后按顺序跑每条规则：
/// - 规则在 `catch_unwind` 里跑，规则跑之前先克隆一份 params；
/// - panic 了：`candidate_params` 上任何改动一律丢弃、还原成规则跑之前的样子——
///   `catch_unwind` 只挡住栈展开，挡不住 panic 之前已经写进 `candidate_params`
///   （它是一个独立的 `Value`，不是被跨越 unwind 边界共享的状态，但规则确实可能在
///   panic 之前已经改过它）的半成品改动，那不是这条规则"完成"的输出，绝不能当成
///   `clamped` 记下来；这条规则的 outgoing 记成 `Deny{"rule <name> panicked"}`，
///   `panicked = true`；
/// - 没 panic：规则跑完之后按 `!=` 比较 params 有没有变化；变化了、且输入判定不是
///   `Deny`，记 `clamped = Some({before, after})`；变化了、但输入判定是 `Deny`，把
///   params 改动撤销（还原成规则跑之前的样子）；再看这条规则返回的判定：如果比输入
///   判定更松（rank 更低），视为篡改——`tamper = true`，实际生效的 outgoing 改成
///   `Deny{"rule <name> attempted to loosen the verdict"}`，并把 `ChainResult.tampered`
///   置为 `true`；否则原样采用规则返回的判定；
/// - 规则通过 [`Rule::evaluate_detailed`] 留下的 `detail` 原样记进 trace（panic 的那次
///   没有 detail）。它只进审计，不碰 `params`。
pub fn evaluate_chain(
    rules: &[(String, Arc<dyn Rule>)],
    required_rules: &[String],
    params: Value,
    ctx: &RuleCtx,
) -> ChainResult {
    let mut verdict = initial_verdict(rules, required_rules);
    let mut params = params;
    let mut trace = Vec::with_capacity(rules.len());
    let mut tampered = false;

    for (name, rule) in rules {
        let incoming = verdict.clone();
        let before = params.clone();
        let incoming_was_deny = matches!(incoming, Verdict::Deny { .. });

        let mut candidate_params = before.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            rule.evaluate_detailed(&mut candidate_params, ctx, incoming.clone())
        }));

        let mut tamper = false;
        let mut panicked = false;
        let mut detail = None;
        let outgoing = match result {
            Ok((v, d)) => {
                detail = d;
                if v.is_tighter_or_equal(&incoming) {
                    v
                } else {
                    tamper = true;
                    tampered = true;
                    Verdict::Deny {
                        reason: format!("rule {name} attempted to loosen the verdict"),
                    }
                }
            }
            Err(_) => {
                panicked = true;
                Verdict::Deny {
                    reason: format!("rule {name} panicked"),
                }
            }
        };

        let mut clamped = None;
        if panicked {
            // 规则 panic 了：`catch_unwind` 只挡住了栈展开，挡不住它 panic 之前已经
            // 写进 `candidate_params` 堆内存里的半成品改动——那些改动不是这条规则
            // "完成"的结果，不能当成它的输出，一律撤销，也绝不记进 `clamped`。
            candidate_params = before.clone();
        } else if candidate_params != before {
            if incoming_was_deny {
                // 输入已经是 Deny：规则不该再改 params，改动一律撤销。
                candidate_params = before.clone();
            } else {
                clamped = Some(json!({"before": before, "after": candidate_params}));
            }
        }
        params = candidate_params;

        trace.push(RuleTrace {
            rule: name.clone(),
            incoming,
            outgoing: outgoing.clone(),
            tamper,
            panicked,
            clamped,
            detail,
        });
        verdict = outgoing;
    }

    ChainResult {
        verdict,
        params,
        trace,
        tampered,
    }
}

fn initial_verdict(rules: &[(String, Arc<dyn Rule>)], required_rules: &[String]) -> Verdict {
    if rules.is_empty() {
        return Verdict::Ask {
            reason: "no rules registered".into(),
        };
    }
    let missing: Vec<&str> = required_rules
        .iter()
        .filter(|req| !rules.iter().any(|(name, _)| name == *req))
        .map(|s| s.as_str())
        .collect();
    if missing.is_empty() {
        Verdict::Allow
    } else {
        Verdict::Ask {
            reason: format!("missing required rules: {}", missing.join(", ")),
        }
    }
}
