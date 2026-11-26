//! 一次执行的生老病死：登记、租约、抢占、撤销、收尾。
//!
//! [`crate::admission`] 里的流水线负责"准不准"，这里负责"准了之后发生什么"——谁在跑、
//! 心跳还新不新鲜、被谁撤销了、结束时 FSM 该不该交还。所有对外可见的事件都由
//! [`Inner::act`] 写出，和流水线用同一套 payload 约定。
//!
//! 锁的纪律和 [`crate::admission`] 一样：`Inner::s` 只在短临界区里持有。认领撤销权
//! （改 `cancel_reason`、让出 `active_motion`）在锁里一次完成，真正的 `ExecControl::cancel`
//! 和 `append` 都在锁外——`append` 会同步跑世界日志的全部观察者。

use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::fsm::FsmState;
use oeh_kernel::proposal::{Outcome, Proposal, ProposalId};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::admission::{
    deny_detail, AdmState, Inner, SAL_CANCELLED, SAL_LEASE, SAL_PREEMPTED, SAL_RESULT, SAL_STARTED,
    SOURCE,
};
use crate::capability::{
    CancelReason, CapabilityClass, CapabilitySpec, ExecControl, ExecCtx, ExecOutcome,
    ExecutionCall, Executor,
};
use crate::verdict::{RuleTrace, Verdict};

/// 接班者把 motion 的位子接过来时，位子上原来是什么。
pub(crate) enum Claimed {
    /// 一次还活着的执行：调用方负责 `cancel()` 并记 `act/preempted`。
    Preempted(Proposal, Arc<ExecControl>),
    /// 位子原来的主人**租约已经过期**：它的终局事件（`act/lease_expired`）在过期那一刻
    /// 就记过了，绝不能再补第二条；但 ACTING 这块地确实换了主人，交接照立。
    TookOverExpired(ProposalId),
    /// 没有 motion 在跑。
    Nothing,
}

impl Claimed {
    /// 被接手的那次执行的 id（如果有）。交接告吹时要靠它把旗子放下来。
    fn victim(&self) -> Option<ProposalId> {
        match self {
            Claimed::Preempted(v, _) => Some(v.proposal_id.clone()),
            Claimed::TookOverExpired(pid) => Some(pid.clone()),
            Claimed::Nothing => None,
        }
    }
}

/// 一次进行中的执行。
pub(crate) struct ExecEntry {
    /// 被准入的提案；`params` 是规则链夹过之后、真正交给执行器的那一份。撤销事件要靠它
    /// 填 `brain_id`/`proposal_id`/`capability`/`trace_id`，所以整份留着。
    proposal: Proposal,
    class: CapabilityClass,
    /// 0 = 不设租约。
    lease_ttl_ns: u64,
    /// 起跑时刻：执行器还没来得及发第一次心跳时，租约从这里起算。
    started_ns: u64,
    /// 登记这次执行时世界日志的下一个 seq：比它旧的 `safety/fsm` 事件是这次执行**开始
    /// 之前**的历史，不能拿来撤销它。观察者→通道→任务这条链是异步的，一条"刚开机时
    /// 站起来了"的旧事件完全可能在 motion 起跑之后才被消费到。
    started_seq: u64,
    /// 这条登记的世代号，全局单调。同一个 `ProposalId` 被重新登记时世代必然不同，于是
    /// 一个活过了自己那次执行的后台任务（复检）认不出新的那一个——否则它会拿着**旧**
    /// 能力的 spec 和旧的 params 去评估新的执行，一次 Deny 就撤掉一个与它无关的运动。
    epoch: u64,
    ctl: Arc<ExecControl>,
    /// 跑这次执行的执行器。租约过期时要回头叫它的 `stop()`——叫的必须是**失联的这一个**，
    /// 而不是"当前正在跑的那个 motion"。
    exec: Arc<dyn Executor>,
    /// 租约在这一刻过期了（内核时间）。过期之后这次执行仍然占着 motion 的位子：失联的
    /// 执行器可能还在让机器人动，位子悄悄空出来就意味着下一个 motion 可以在它之上叠着
    /// 跑。`lease_grace_ms` 过完还没等回它，`sweep_leases` 强行摘掉这条登记。
    expired_ns: Option<u64>,
    /// 谁撤销了这次执行、为什么。置上之后收尾不再记 `act/result`——终局事件
    /// （`act/preempted` / `act/lease_expired` / `act/cancelled`）在撤销那一刻就记过了。
    cancel_reason: Option<String>,
    /// 这次执行是被**抢占**掉的：ACTING 这块地已经交给接班的 motion 了，收尾时不许再
    /// 把 FSM 交还成 STANDING。
    ///
    /// 交接不是一个瞬间：接班者先认领（让出 `active_motion`）、再驱 FSM、最后才登记
    /// 自己。被抢占者的收尾要是正好落在中间那一段，它会看到"没有别的 motion 在跑"，
    /// 于是多发一次 `ACTING → STANDING`——接班者要么得把 FSM 重新推一遍（凭空多出
    /// 一组迁移事件），要么干脆被登记进一个 STANDING 的机器人里，然后被起跑后的复查
    /// 以 `reason:"fsm"` 自己取消掉。所以这面旗子在认领的同一把锁里就立起来。
    handed_over: bool,
    /// 这次执行里已经报过 `safety/tamper_attempt` 的规则名。复检每一拍都重跑整条链，
    /// 同一条"想放水"的规则会一拍报一次——salience 255 是操作员的告警通道，同一条
    /// 消息刷屏等于把它淹掉。一次执行里一条规则只响一次。
    tamper_seen: BTreeSet<String>,
}

impl AdmState {
    /// 认领一次执行的撤销权（幂等）：已经被别人认领过就返回 `None`，这样同一次执行
    /// 不会记出两条终局事件。返回值给锁外用：真正的 `cancel()` 与 append 都在锁外。
    /// `epoch` 是 `Some` 时还要求登记的世代号对得上：陈旧的后台任务只认自己那一次执行。
    fn claim(
        &mut self,
        pid: &ProposalId,
        epoch: Option<u64>,
        reason: CancelReason,
    ) -> Option<(Proposal, Arc<ExecControl>)> {
        let e = self.execs.get_mut(pid)?;
        if epoch.is_some_and(|want| want != e.epoch) {
            return None;
        }
        if e.cancel_reason.is_some() {
            return None;
        }
        e.cancel_reason = Some(reason.as_str().to_string());
        // 抢占是交接，不是收摊：ACTING 归接班者，被抢占者的收尾不再碰 FSM。其余撤销
        // （stop / 租约 / fsm）之后没人接班，照常交还。
        e.handed_over = reason == CancelReason::Preempted;
        let claimed = (e.proposal.clone(), e.ctl.clone());
        // 租约过期是唯一不让位的撤销：执行器失联了，它可能还在动。位子继续由它占着，
        // 新的 motion 只能走抢占那条明确的路（那条路会撤销它并接管 ACTING），而不是
        // 从一个"空出来的"位子上悄悄起跑。
        if reason != CancelReason::LeaseExpired && self.active_motion.as_ref() == Some(pid) {
            // 撤销的那一刻就让位：后来的 motion 不必等这个执行器慢慢收尾才能起跑。
            self.active_motion = None;
        }
        Some(claimed)
    }

    /// 交接告吹（接班者最终没能起跑）：把旗子放下，让被抢占者的收尾照常交还 FSM。
    /// 返回被抢占者是否还在——已经收过尾的话，得由调用方自己把 ACTING 还回去。
    fn release_hand_over(&mut self, pid: &ProposalId) -> bool {
        match self.execs.get_mut(pid) {
            Some(e) => {
                e.handed_over = false;
                true
            }
            None => false,
        }
    }

    /// 接班者把 motion 的位子接过来：立交接旗、让出 `active_motion`，两件事在同一个
    /// 临界区里完成。
    ///
    /// 唯一可能"占着位子却已经带着 `cancel_reason`"的是租约过期的那一个（见 `claim`
    /// 里的让位规则）。它的终局事件早记过了，所以这里只接手 ACTING，不补第二条终局，
    /// 也不再 `cancel()` 一遍——过期那一刻就撤过了。
    fn claim_for_handover(&mut self) -> Claimed {
        let Some(pid) = self.active_motion.clone() else {
            return Claimed::Nothing;
        };
        let Some(e) = self.execs.get_mut(&pid) else {
            // 位子挂着一个不存在的登记：不该发生，稳妥起见把它放掉。
            self.active_motion = None;
            return Claimed::Nothing;
        };
        // ACTING 归接班者：被接手的那个收尾时不许再把 FSM 还成 STANDING。这面旗子和
        // "让出位子"必须在同一个临界区里立起来（见 `ExecEntry::handed_over`）。
        e.handed_over = true;
        let already_terminal = e.cancel_reason.is_some();
        let victim = (e.proposal.clone(), e.ctl.clone());
        if !already_terminal {
            e.cancel_reason = Some(CancelReason::Preempted.as_str().to_string());
        }
        self.active_motion = None;
        if already_terminal {
            Claimed::TookOverExpired(pid)
        } else {
            Claimed::Preempted(victim.0, victim.1)
        }
    }

    /// 读当前 motion 并认领它的撤销权——两步在同一个临界区里完成，否则"读到的那个"
    /// 可能在认领之前就换成了另一次执行。
    fn claim_active(&mut self, reason: CancelReason) -> Option<(Proposal, Arc<ExecControl>)> {
        let pid = self.active_motion.clone()?;
        self.claim(&pid, None, reason)
    }

    /// 某次执行的登记世代号。登记不在了 → `None`。
    pub(crate) fn exec_epoch(&self, pid: &ProposalId) -> Option<u64> {
        self.execs.get(pid).map(|e| e.epoch)
    }

    /// 下一个登记世代号。
    fn next_epoch(&mut self) -> u64 {
        self.next_exec_epoch += 1;
        self.next_exec_epoch
    }

    /// 认领一次租约过期，并记下过期时刻（宽限期从这里起算）。额外带回这次执行的执行器：
    /// 叫停要打给它自己。
    #[allow(clippy::type_complexity)]
    fn claim_expired(
        &mut self,
        pid: &ProposalId,
        now: u64,
    ) -> Option<(Proposal, Arc<ExecControl>, Arc<dyn Executor>)> {
        let (proposal, ctl) = self.claim(pid, None, CancelReason::LeaseExpired)?;
        let e = self.execs.get_mut(pid)?;
        e.expired_ns = Some(now);
        Some((proposal, ctl, e.exec.clone()))
    }

    /// 宽限期过完还没收尾的失联执行：强行摘掉登记，返回 `(pid, 要不要交还 ACTING)`。
    fn force_remove_stale(&mut self, now: u64, grace_ns: u64) -> Vec<(ProposalId, bool)> {
        let stale: Vec<ProposalId> = self
            .execs
            .iter()
            .filter(|(_, e)| {
                e.expired_ns
                    .is_some_and(|t| now.saturating_sub(t) >= grace_ns)
            })
            .map(|(pid, _)| pid.clone())
            .collect();
        stale
            .into_iter()
            .filter_map(|pid| {
                let e = self.execs.remove(&pid)?;
                if self.active_motion.as_ref() == Some(&pid) {
                    self.active_motion = None;
                }
                Some((pid, e.class == CapabilityClass::Motion && !e.handed_over))
            })
            .collect()
    }

    /// 同上，但只认领"在 `seq` 这条事件之前就已经登记"的 motion：晚于它起跑的执行不该
    /// 被一条更旧的事件撤销。
    fn claim_active_since(
        &mut self,
        seq: u64,
        reason: CancelReason,
    ) -> Option<(Proposal, Arc<ExecControl>)> {
        let pid = self.active_motion.clone()?;
        if self.execs.get(&pid)?.started_seq > seq {
            return None;
        }
        self.claim(&pid, None, reason)
    }

    /// 服务被撤销时把还在跑的执行全部认领下来：返回它们的句柄，由调用方在锁外
    /// `cancel()`。没有了租约监视任务，就没人再看着这些执行器了。
    pub(crate) fn cancel_all(&mut self, reason: &str) -> Vec<Arc<ExecControl>> {
        self.active_motion = None;
        self.execs
            .values_mut()
            .filter(|e| e.cancel_reason.is_none())
            .map(|e| {
                e.cancel_reason = Some(reason.to_string());
                e.ctl.clone()
            })
            .collect()
    }
}

impl Inner {
    /// 被准入的 motion 起跑前把 FSM 推到 `ACTING`（`STANDING → ENGAGED → ACTING` 两步）。
    /// 推不上去就是推不上去：调用方据此撤销这次执行，绝不在 FSM 不允许的状态下动。
    fn drive_to_acting(&self, capability: &str) -> Result<(), String> {
        let reason = format!("admitted {capability}");
        let steps: &[FsmState] = match self.fsm.state() {
            // 抢占：已经在 ACTING 里，保持不动。
            FsmState::Acting => return Ok(()),
            FsmState::Standing => &[FsmState::Engaged, FsmState::Acting],
            FsmState::Engaged => &[FsmState::Acting],
            other => return Err(format!("fsm state {other} does not allow motion")),
        };
        for to in steps {
            self.fsm
                .transition(*to, &reason, SOURCE)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// 认领之后的收尾（撤销 + 记 `act/cancelled{reason}`）。认领在锁里，这一步在锁外。
    fn cancelled(
        &self,
        claimed: Option<(Proposal, Arc<ExecControl>)>,
        reason: CancelReason,
        detail: Option<String>,
    ) -> bool {
        let Some((proposal, ctl)) = claimed else {
            return false;
        };
        ctl.cancel();
        let mut extra = json!({"reason": reason.as_str()});
        if let Some(d) = detail {
            extra["detail"] = json!(d);
        }
        self.act(&proposal, "act/cancelled", SAL_CANCELLED, extra);
        true
    }

    /// 撤销一次执行并记 `act/cancelled{reason}`（`detail` 是"为什么"的补充，比如复检里
    /// 那条给出 Deny 的规则）。已经被撤销过 / 已经结束了 → `false`。
    pub(crate) fn cancel_exec(
        &self,
        pid: &ProposalId,
        epoch: Option<u64>,
        reason: CancelReason,
        detail: Option<String>,
    ) -> bool {
        let claimed = self.s.lock().claim(pid, epoch, reason);
        self.cancelled(claimed, reason, detail)
    }

    pub(crate) fn cancel_active_motion(&self, reason: CancelReason) -> bool {
        let claimed = self.s.lock().claim_active(reason);
        self.cancelled(claimed, reason, None)
    }

    /// `safety/fsm` 任务用：只撤销那些在这条事件之前就起跑了的 motion。
    pub(crate) fn cancel_active_motion_since(&self, seq: u64, reason: CancelReason) -> bool {
        let claimed = self.s.lock().claim_active_since(seq, reason);
        self.cancelled(claimed, reason, None)
    }

    /// 执行期间的规则复检（规格 §5.3：`walk` 这类持续能力由 `rule-scene` 以 5 Hz 复检，
    /// 失败即 cancel）。准入只看起跑那一瞬间的世界——一个人走到正在行走的机器人前面，
    /// 在这之前不会产生任何撤销。复检跑的是**登记时那份夹过的 params**：规则只判"现在
    /// 还准不准这么走"，不再改参数（执行器已经拿着那一份在跑了）。
    ///
    /// 返回"要不要继续复检"。
    pub(crate) fn recheck_exec(
        &self,
        pid: &ProposalId,
        epoch: u64,
        spec: &CapabilitySpec,
        params: &Value,
    ) -> bool {
        // 登记没了（收过尾了）、世代号对不上（同一个 pid 已经被另一次执行占了）、或者
        // 已经被别人撤销了：这次执行的故事结束了，这个任务该退场。
        let proposal = {
            let g = self.s.lock();
            match g.execs.get(pid) {
                Some(e) if e.epoch == epoch && e.cancel_reason.is_none() => e.proposal.clone(),
                _ => return false,
            }
        };
        let re = self.evaluate_chain_now(spec, params.clone());
        // 只报这次执行里还没报过的那些规则（见 `ExecEntry::tamper_seen`）。顺带再确认
        // 一次登记还是自己那一条：评估期间它完全可能已经收尾了。
        let fresh: Vec<RuleTrace> = {
            let mut g = self.s.lock();
            match g.execs.get_mut(pid) {
                Some(e) if e.epoch == epoch => re
                    .trace
                    .iter()
                    .filter(|t| t.tamper && e.tamper_seen.insert(t.rule.clone()))
                    .cloned()
                    .collect(),
                _ => return false,
            }
        };
        self.record_tamper(&proposal, &fresh);
        // 规则 panic 不去重：它强制 Deny，这次执行紧接着就被撤销，最多一条。
        self.record_rule_errors(&proposal, &re.trace);
        // `Ask` 不撤：准入已经过了（必要时还有人点过头），复检只负责在"现在明确不行"
        // 的时候叫停。
        if let Verdict::Deny { reason } = &re.verdict {
            self.cancel_exec(
                pid,
                Some(epoch),
                CancelReason::Rules,
                Some(deny_detail(&re.trace, reason)),
            );
            return false;
        }
        true
    }

    /// 心跳老过租约的执行一律撤销：执行器可以慢，但不能失联——失联的执行器意味着没人
    /// 在看着机器人动。
    pub(crate) async fn sweep_leases(&self) {
        // 顺手收尸：跑完的流水线任务会一直留在 `JoinSet` 里直到被 join。这把锁只在这个
        // 块里拿着，绝不跨 await。
        self.reap_pipelines();
        let now = self.clock.now_ns();
        let expired: Vec<(ProposalId, u64, f64)> = {
            let g = self.s.lock();
            g.execs
                .iter()
                .filter_map(|(pid, e)| {
                    if e.cancel_reason.is_some() || e.lease_ttl_ns == 0 {
                        return None;
                    }
                    let hb = e.ctl.last_heartbeat_ns();
                    let base = if hb == 0 { e.started_ns } else { hb };
                    let age = now.saturating_sub(base);
                    (age > e.lease_ttl_ns).then(|| (pid.clone(), age, e.lease_ttl_ns as f64 / 1e9))
                })
                .collect()
        };
        for (pid, age, ttl_s) in expired {
            let claimed = self.s.lock().claim_expired(&pid, now);
            let Some((proposal, ctl, exec)) = claimed else {
                continue;
            };
            ctl.cancel();
            self.act(
                &proposal,
                "act/lease_expired",
                SAL_LEASE,
                json!({
                    "reason": CancelReason::LeaseExpired.as_str(),
                    "age_ms": age / 1_000_000,
                    "lease_ttl_s": ttl_s,
                }),
            );
            // 规格 §5.3："心跳超过 lease TTL 记 `act/lease_expired`，内核向适配器发
            // stop"。撤销只是"别再往下做"，而失联的执行器按定义就是不再看取消通道的
            // 那种——所以再明确地叫一次停，打给**它自己**。
            spawn_stop(exec, &pid);
        }

        // 宽限期到点：失联的执行器再也不会自己收尾了。强行摘掉登记，把位子和 ACTING
        // 还回去——不然一次失联就永远占着 motion 的唯一名额。它之后要是又活过来调
        // `finish`，会发现自己的登记已经不在，于是什么都不做。
        let forced = {
            let mut g = self.s.lock();
            g.force_remove_stale(now, self.lease_grace_ns())
        };
        for (pid, _) in &forced {
            tracing::warn!(proposal = %pid.0, "lease grace expired; dropping the registration");
        }
        if forced.iter().any(|(_, releases)| *releases) {
            let _gate = self.admit.lock().await;
            self.release_fsm_if_idle("lease grace expired");
        }
    }

    /// 没有 motion 在跑了就把 ACTING 交还成 STANDING。别人（蒸馏器、操作员）已经把 FSM
    /// 带去别处时不插手：那不是我们的边。
    ///
    /// **调用者必须持有 `admit` gate**（或者像 `Drop` 那样明知故犯）：判断和迁移之间要是
    /// 挤进一次起跑，那个 motion 会看到"FSM 还是 ACTING"从而不驱动、登记自己，然后被这
    /// 里的迁移落回 STANDING——它就在一台自称 STANDING 的机器人上跑完了整趟。
    pub(crate) fn release_fsm_if_idle(&self, reason: &str) {
        // 先出锁再迁移：`transition` 里有 append，绝不能带着 `s` 锁进去。
        let idle = self.s.lock().active_motion.is_none();
        if idle && self.fsm.state() == FsmState::Acting {
            if let Err(e) = self.fsm.transition(FsmState::Standing, reason, SOURCE) {
                tracing::debug!(%e, "fsm");
            }
        }
    }

    /// 一次执行收尾：摘掉登记、定结局、必要时把 FSM 交还给 `STANDING`。
    ///
    /// 是 `async` 的：交还 FSM 那一步要在 `admit` gate 里做，和"抢占 → 驱 FSM → 登记 →
    /// 起跑"互斥（见 `release_fsm_if_idle`）。不会死锁——`start` 从不 await `finish`，
    /// 收尾跑在自己那个 spawn 出来的任务上。
    async fn finish(&self, p: &Proposal, outcome: ExecOutcome) -> Outcome {
        let (cancel_reason, releases_fsm) = {
            let mut g = self.s.lock();
            let entry = g.execs.remove(&p.proposal_id);
            if g.active_motion.as_ref() == Some(&p.proposal_id) {
                g.active_motion = None;
            }
            match entry {
                // 只有 motion 占着 ACTING，也只有没被交接出去的那个才交还它。
                Some(e) => (
                    e.cancel_reason,
                    e.class == CapabilityClass::Motion && !e.handed_over,
                ),
                // 登记已经不在了：只有租约宽限期到点的强制摘除会走到这一步。那一刻终局
                // 事件（`act/lease_expired`）记过了，位子和 ACTING 也都还回去了——失联的
                // 执行器现在才回来，什么都不该再动，更不能补一条 `act/result`。
                None => {
                    return Outcome::Cancelled {
                        reason: CancelReason::LeaseExpired.as_str().into(),
                    }
                }
            }
        };
        let out = match cancel_reason {
            // 终局事件在撤销那一刻已经记过（`act/preempted` / `act/lease_expired` /
            // `act/cancelled`），这里不再补一条 `act/result`：一次提案只有一个终局。
            Some(r) if r == CancelReason::Preempted.as_str() => Outcome::Preempted,
            Some(reason) => Outcome::Cancelled { reason },
            None => {
                let extra = if outcome.ok {
                    json!({"ok": true, "result": outcome.result})
                } else {
                    json!({"ok": false, "error": outcome.error.clone().unwrap_or_default()})
                };
                self.act_framed(p, "act/result", SAL_RESULT, extra);
                if outcome.ok {
                    Outcome::Executed {
                        result: outcome.result,
                    }
                } else {
                    Outcome::Failed {
                        error: outcome.error.unwrap_or_default(),
                    }
                }
            }
        };
        // motion 以任何理由结束（被抢占的除外，那块地已经交给接班者了）→ 交还 ACTING。
        if releases_fsm {
            let _gate = self.admit.lock().await;
            self.release_fsm_if_idle("motion finished");
        }
        out
    }
}

/// 给一次正在跑的 motion 起一个周期复检任务。持 `Weak`：服务没了它自己结束；登记不在
/// 了（收过尾 / 被撤销）也结束。
fn spawn_recheck(
    inner: &Arc<Inner>,
    pid: ProposalId,
    epoch: u64,
    spec: CapabilitySpec,
    params: Value,
    mut cancel: watch::Receiver<bool>,
) {
    let period = Duration::from_millis(inner.recheck_ms().clamp(1, 60_000));
    let weak = Arc::downgrade(inner);
    tokio::spawn(async move {
        let mut iv = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = iv.tick() => {}
                // 这次执行被撤销了，或者它的登记连同 `ExecControl` 一起没了（发送端一掉
                // `changed()` 就报错）：立刻退场，不必等到下一拍。多活一拍，就够一次
                // `ProposalId` 复用钻进来。
                _ = cancel.changed() => break,
            }
            let Some(i) = weak.upgrade() else { break };
            if !i.recheck_exec(&pid, epoch, &spec, &params) {
                break;
            }
        }
    });
}

/// 叫一次失联的执行器停下来。`stop()` 和 `execute()` 一样是第三方代码，**调用本身**就
/// 可能 panic，所以同样包在 `catch_unwind` 里；返回的 future spawn 出去就不管——一个
/// 已经失联的执行器不值得内核再等它。
fn spawn_stop(exec: Arc<dyn Executor>, pid: &ProposalId) {
    match catch_unwind(AssertUnwindSafe(|| exec.stop(pid))) {
        Ok(fut) => {
            tokio::spawn(fut);
        }
        Err(_) => tracing::warn!(proposal = %pid.0, "executor panicked in stop()"),
    }
}

/// 执行器没能给出结果时的结局。
fn failed(error: &str) -> ExecOutcome {
    ExecOutcome {
        ok: false,
        result: Value::Null,
        error: Some(error.to_string()),
    }
}

/// 抢占、FSM 驱动、登记、执行、收尾。
pub(crate) async fn start(
    inner: &Arc<Inner>,
    p: Proposal,
    spec: CapabilitySpec,
    params: Value,
    exec: Arc<dyn Executor>,
) -> Outcome {
    let (class, lease_ttl_s) = (spec.class, spec.lease_ttl_s);
    let (ectx, ctl) = ExecCtx::new_pair(
        p.trace_id.clone(),
        p.proposal_id.clone(),
        inner.clock.clone(),
    );
    let ctl = Arc::new(ctl);
    // 规则链夹过之后的那一份提案：登记、撤销事件、以及交给执行器的 `ExecutionCall`
    // 全用它。执行器手里绝不能出现未经夹取的 params——`call.proposal.params` 要是还是
    // 原样，随手读它的执行器就把整条规则链绕过去了。
    let clamped = Proposal {
        params: params.clone(),
        ..p.clone()
    };
    let lease_ttl_ns = if lease_ttl_s.is_finite() && lease_ttl_s > 0.0 {
        (lease_ttl_s * 1e9) as u64
    } else {
        0
    };

    let motion = class == CapabilityClass::Motion;
    let epoch;
    // 接手的是一个租约已经过期的 motion 时，它的 id。
    let mut took_over_expired: Option<String> = None;
    {
        // motion 的"抢占 → 驱 FSM → 登记 → 起跑"整段在同一把 gate 下：并发被准入的两个
        // motion 否则会各自看到"没有 motion 在跑"。`act/started` 也在里面——出了 gate 再
        // 记，抢占它的那条 `act/preempted{by: 自己}` 就可能排在自己的 `act/started`
        // 前面，日志读起来像是"还没起跑就被抢了"。非 motion 不参与这场排队。
        let _gate = if motion {
            Some(inner.admit.lock().await)
        } else {
            None
        };
        // 重复的 `proposal_id`：登记表按它索引，重名会把前一次执行悄悄顶掉——先收尾的
        // 那个把另一个的登记一起摘走，剩下那次执行的租约、撤销权和 FSM 就再也没人管了。
        // 挡在抢占**之前**：拒一个重名的提案，不该顺带把正在跑的 motion 抢掉。
        if inner.s.lock().execs.contains_key(&p.proposal_id) {
            return inner.refuse(&p, "vocabulary", "duplicate proposal_id".into(), None);
        }
        if motion {
            let claimed = inner.s.lock().claim_for_handover();
            let handed_over_from = claimed.victim();
            match claimed {
                Claimed::Preempted(victim, vctl) => {
                    vctl.cancel();
                    inner.act(
                        &victim,
                        "act/preempted",
                        SAL_PREEMPTED,
                        json!({"by": p.proposal_id.0}),
                    );
                }
                // 租约过期的那一个已经有终局事件了，这里不补第二条——但接班者的
                // `act/started` 要说清楚 ACTING 是从谁手里接过来的，否则日志读起来就是
                // 一条 `act/lease_expired` 之后凭空出现在 ACTING 机器人上的起跑。
                Claimed::TookOverExpired(pid) => took_over_expired = Some(pid.0),
                Claimed::Nothing => {}
            }
            if let Err(detail) = inner.drive_to_acting(&p.capability) {
                // FSM 不让动就不动。已经记过 `act/admitted`，所以这里用撤销而不是拒绝：
                // 准入是过了，是世界在最后一刻变了。
                inner.act(
                    &p,
                    "act/cancelled",
                    SAL_CANCELLED,
                    json!({"reason": CancelReason::Fsm.as_str(), "detail": detail}),
                );
                // 交接告吹：没人接班，ACTING 还是得有人还。被抢占者还在就交还给它的
                // 收尾，已经收过尾了就现在还。
                if let Some(vid) = handed_over_from {
                    if !inner.s.lock().release_hand_over(&vid) {
                        inner.release_fsm_if_idle("motion finished");
                    }
                }
                return Outcome::Cancelled {
                    reason: CancelReason::Fsm.as_str().into(),
                };
            }
        }
        // 起跑时刻与起跑 seq 在 gate 里、紧挨着 `execs.insert` 取。在 gate 外取的话，排队
        // 等 gate 的那段时间会白白吃掉租约；而一条落在"取值"与"登记"之间的 `safety/fsm`
        // 会显得比这次执行更**旧**，于是把一次刚起跑的运动白撤一遍。
        let started_ns = inner.clock.now_ns();
        let started_seq = inner.log.next_seq();
        epoch = {
            let mut g = inner.s.lock();
            let epoch = g.next_epoch();
            g.execs.insert(
                p.proposal_id.clone(),
                ExecEntry {
                    proposal: clamped.clone(),
                    class,
                    lease_ttl_ns,
                    started_ns,
                    started_seq,
                    epoch,
                    ctl: ctl.clone(),
                    exec: exec.clone(),
                    expired_ns: None,
                    cancel_reason: None,
                    handed_over: false,
                    tamper_seen: BTreeSet::new(),
                },
            );
            if motion {
                g.active_motion = Some(p.proposal_id.clone());
            }
            epoch
        };

        let mut started = json!({"params": params, "lease_ttl_s": lease_ttl_s});
        if let Some(victim) = &took_over_expired {
            started["took_over_expired"] = json!(victim);
        }
        inner.act_framed(&p, "act/started", SAL_STARTED, started);
        // 驱到 ACTING 和登记之间可能挤进一条把 FSM 带走的事件——那一刻我们还没登记，
        // 观察者任务找不到 motion 可撤。补一次判定，撤销落在执行器起跑之前（`ExecCtx`
        // 一开始就是取消态，执行器第一次看就知道该回去）。
        if motion && inner.fsm.state() != FsmState::Acting {
            inner.cancel_exec(&p.proposal_id, None, CancelReason::Fsm, None);
        }
    }

    // 起跑了就开始复检：世界会变，准入只看了起跑那一瞬间。只有 motion 参与——speech /
    // query / control 都是一下子就完的，没有"执行期间"可言。
    if motion && inner.recheck_ms() > 0 {
        spawn_recheck(
            inner,
            p.proposal_id.clone(),
            epoch,
            spec,
            params.clone(),
            ctl.subscribe(),
        );
    }

    let call = ExecutionCall {
        capability: p.capability.clone(),
        params,
        proposal: clamped,
    };
    // `execute()` 本身也是第三方代码，它可能在**返回 future 之前**就 panic——那时还没有
    // 任务可言，`tokio::spawn` 挡不住这一下，unwind 会一路穿出 `start` 和 `pipeline`，
    // 收尾永远不跑：登记摘不掉、FSM 永远停在 ACTING 而并没有 motion 在跑。所以调用本身
    // 也包在 `catch_unwind` 里，炸了就当成一次普通的失败结局走正常收尾。
    let started = catch_unwind(AssertUnwindSafe(|| exec.execute(call, ectx)));
    // 两级任务：内层只跑执行器（可能 panic，也可能在提案方把 future 丢掉之后还在跑），
    // 外层负责收尾。收尾绝不能跟着执行器一起消失——否则登记摘不掉、FSM 永远停在 ACTING。
    let exec_task = started.map(tokio::spawn).ok();
    let inner2 = inner.clone();
    let done = tokio::spawn(async move {
        let outcome = match exec_task {
            Some(t) => match t.await {
                Ok(o) => o,
                Err(e) => failed(if e.is_panic() {
                    "executor panicked"
                } else {
                    "executor task aborted"
                }),
            },
            None => failed("executor panicked"),
        };
        inner2.finish(&p, outcome).await
    });
    match done.await {
        Ok(out) => out,
        Err(_) => Outcome::Failed {
            error: "admission execution task failed".into(),
        },
    }
}
