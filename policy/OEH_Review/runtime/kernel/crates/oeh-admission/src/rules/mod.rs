//! 十六条内置安全规则，每条一个文件、每条一个插件（`rule-<name>`）。
//!
//! 每个文件里有三样东西：规则本体（`pub struct`，字段就是它的配置，测试里直接构造）、
//! 私有的 `Config`（`deny_unknown_fields`，配错一个键是启动错误，不是静默忽略），以及
//! 一个 inject `admission` 的插件——`apply` 里把规则注册进准入服务，并把 disposer 交给
//! fiber，插件被撤销时规则跟着从链上摘掉。（`rule-dryrun` 还 inject 一个 `predictor`：
//! 没有世界模型就没有这条规则，它停在 Pending 而不是悄悄放行。）
//!
//! 两条贯穿所有规则的纪律：
//!
//! - **只能收紧**。规则拿到上游判定 `incoming`，要么原样传回，要么给出一个更紧的。
//!   放松是篡改（`evaluate_chain` 会把它按 Deny 处理并记 `safety/tamper_attempt`），
//!   所以这里所有"我想 Deny/Ask"的地方都走 [`deny`] / [`tighten`]，由它们和 `incoming`
//!   取更紧的那个。
//! - **fail-closed**。读不到的状态（键不存在、值的类型不对、`NaN`）一律按最坏情况处理：
//!   不知道地面是不是空的、不知道机器人是不是站着、不知道流是不是还活着，就不让动。
//!   规则绝不 panic——一条 panic 的规则会被链记成 Deny，但那是兜底，不是设计。

pub mod arx_mode;
pub mod carry_speed;
pub mod clamp;
pub mod dryrun;
pub mod eef_target;
pub mod estop;
pub mod finish_gate;
pub mod fsm;
pub mod hold_no_rotation;
pub mod keepout;
pub mod pose;
pub mod run_mode;
pub mod scene;
pub mod step_reserve;
pub mod watchdog;
pub mod whitelist;

use crate::verdict::Verdict;

/// 取更紧的那个判定：`own` 严格更紧才采用它，否则原样传回 `incoming`。
///
/// 相等紧度时留 `incoming`（比如两条规则都 Deny）：先发现问题的那条规则的理由是根因，
/// 后面的规则不该把它盖掉。
pub(crate) fn tighten(incoming: Verdict, own: Verdict) -> Verdict {
    if own.rank() > incoming.rank() {
        own
    } else {
        incoming
    }
}

/// `tighten(incoming, Deny{reason})` 的简写。
pub(crate) fn deny(incoming: Verdict, reason: impl Into<String>) -> Verdict {
    tighten(
        incoming,
        Verdict::Deny {
            reason: reason.into(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(r: &str) -> Verdict {
        Verdict::Ask { reason: r.into() }
    }

    #[test]
    fn tighten_only_ever_tightens() {
        assert_eq!(tighten(Verdict::Allow, ask("a")), ask("a"));
        assert_eq!(tighten(ask("a"), Verdict::Allow), ask("a"));
        assert_eq!(tighten(ask("a"), ask("b")), ask("a"), "同紧度留上游的理由");
        assert_eq!(deny(ask("a"), "b"), Verdict::Deny { reason: "b".into() });
        assert_eq!(
            deny(Verdict::Deny { reason: "a".into() }, "b"),
            Verdict::Deny { reason: "a".into() },
            "上游的 Deny 理由是根因"
        );
    }
}
