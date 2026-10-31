use std::sync::Arc;
use std::time::Duration;

use oeh_admission::admission::{Admission, AdmissionConfig};
use oeh_admission::approval::{
    decide_with_timeout, Approval, AskRequest, AutoApproval, ChannelApproval, Decision,
};
use oeh_admission::capability::{CapabilityClass, CapabilitySpec, RiskLevel};
use oeh_kernel::clock::{Clock, MonotonicClock};
use oeh_kernel::fsm::RobotFsm;
use oeh_kernel::proposal::{Proposal, ProposalId};
use oeh_kernel::state::StateRegistry;
use oeh_kernel::WorldLog;
use serde_json::json;

fn req() -> AskRequest {
    AskRequest {
        proposal: Proposal {
            proposal_id: ProposalId("t-p1".into()),
            brain_id: "fast".into(),
            turn: 1,
            trace_id: "t".into(),
            capability: "walk".into(),
            params: json!({}),
        },
        spec: CapabilitySpec {
            name: "walk".into(),
            class: CapabilityClass::Motion,
            risk_level: RiskLevel::Medium,
            params_schema: json!({}),
            lease_ttl_s: 5.0,
            overridable: false,
        },
        reason: "confirm".into(),
        trace: vec![],
    }
}

#[tokio::test]
async fn auto_channel_none_and_timeout() {
    assert!(matches!(
        decide_with_timeout(
            Some(Arc::new(AutoApproval { allow: true })),
            req(),
            Duration::from_millis(50)
        )
        .await,
        Decision::Allow { .. }
    ));
    assert!(matches!(
        decide_with_timeout(
            Some(Arc::new(AutoApproval { allow: false })),
            req(),
            Duration::from_millis(50)
        )
        .await,
        Decision::Deny { .. }
    ));
    assert!(matches!(
        decide_with_timeout(None, req(), Duration::from_millis(50)).await,
        Decision::Deny { reason, .. } if reason.contains("no approval")
    ));
    let (chan, mut rx) = ChannelApproval::new();
    let c2: Arc<dyn Approval> = chan.clone();
    let answer = tokio::spawn(async move {
        let (r, tx) = rx.recv().await.unwrap();
        assert_eq!(r.proposal.capability, "walk");
        tx.send(Decision::Allow {
            by: "operator".into(),
            reason: "ok".into(),
        })
        .unwrap();
    });
    assert!(matches!(
        decide_with_timeout(Some(c2), req(), Duration::from_millis(500)).await,
        Decision::Allow { by, .. } if by == "operator"
    ));
    answer.await.unwrap();
    let (chan, _rx) = ChannelApproval::new(); // 没人回答 → 超时 Deny
    assert!(matches!(
        decide_with_timeout(Some(chan), req(), Duration::from_millis(30)).await,
        Decision::Deny { reason, .. } if reason.contains("timeout")
    ));
}

/// `Admission::approval()`：装着谁就还谁，撤了就还 `None`。
///
/// 这是给**不走准入链**的调用方开的口子（`brain-codex` 要把 codex 的
/// `requestApproval` 转给同一个人）。`set_approval` 的 `Disposer` 语义在这里也要成立：
/// 后设者覆盖前一个，撤销只拆自己那一次。
#[tokio::test]
async fn the_installed_approver_can_be_read_back() {
    let clock: Arc<dyn Clock> = Arc::new(MonotonicClock::new());
    let log = Arc::new(WorldLog::new(clock.clone()));
    let state = Arc::new(StateRegistry::new(clock.clone()));
    let fsm = Arc::new(RobotFsm::new(log.clone(), state.clone()));
    let adm = Admission::new(clock, log, state, fsm, AdmissionConfig::default()).unwrap();

    assert!(adm.approval().is_none(), "没装就是没装");

    let d = adm.set_approval(Arc::new(AutoApproval { allow: false }));
    let got = adm.approval().expect("刚装上");
    assert_eq!(got.name(), "approval-auto");
    // 拿到的是个能用的服务，不只是个名字。
    assert!(matches!(got.decide(req()).await, Decision::Deny { .. }));

    // 后设者覆盖前一个。
    let (chan, _rx) = ChannelApproval::new();
    let d2 = adm.set_approval(chan);
    assert_eq!(adm.approval().expect("换人了").name(), "approval-channel");

    // 撤销只拆自己那一次：`d` 装的那个早就被覆盖了，它的 Disposer 不该误伤后来者。
    d();
    assert_eq!(adm.approval().expect("还在").name(), "approval-channel");
    d2();
    assert!(adm.approval().is_none());
}
