use oeh_proto::rpc::{method, types, Message};
use serde::de::DeserializeOwned;
use serde_json::Value;

fn fixture(m: &str) -> Value {
    let p = format!(
        "{}/fixtures/rpc/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        m.replace('/', ".")
    );
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}")))
        .unwrap()
}

fn round<T: DeserializeOwned + serde::Serialize + PartialEq + std::fmt::Debug>(v: &Value) {
    let t: T = serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("{v}: {e}"));
    let back: T = serde_json::from_value(serde_json::to_value(&t).unwrap()).unwrap();
    assert_eq!(t, back);
}

#[test]
fn every_method_has_a_fixture_that_parses_as_a_message() {
    for m in method::ALL {
        let f = fixture(m);
        if let Some(req) = f.get("request") {
            assert!(
                matches!(
                    serde_json::from_value::<Message>(req.clone()).unwrap(),
                    Message::Request(_)
                ),
                "{m}"
            );
            assert!(
                matches!(
                    serde_json::from_value::<Message>(f["response"].clone()).unwrap(),
                    Message::Response(_)
                ),
                "{m}"
            );
            assert_eq!(req["method"], *m);
        } else {
            let n = &f["notification"];
            assert!(
                matches!(
                    serde_json::from_value::<Message>(n.clone()).unwrap(),
                    Message::Notification(_)
                ),
                "{m}"
            );
            assert_eq!(n["method"], *m);
        }
    }
}

#[test]
fn params_and_results_round_trip_through_their_types() {
    use types::*;
    let p = |m: &str| fixture(m)["request"]["params"].clone();
    let r = |m: &str| fixture(m)["response"]["result"].clone();
    let n = |m: &str| fixture(m)["notification"]["params"].clone();
    round::<Empty>(&p(method::ROBOT_CAPABILITIES));
    round::<RobotCapabilitiesResult>(&r(method::ROBOT_CAPABILITIES));
    round::<Empty>(&p(method::ROBOT_STATE));
    round::<RobotStateResult>(&r(method::ROBOT_STATE));
    round::<Empty>(&p(method::ROBOT_HEALTH));
    round::<RobotHealthResult>(&r(method::ROBOT_HEALTH));
    round::<BrainAttach>(&p(method::BRAIN_ATTACH));
    round::<BrainAttachResult>(&r(method::BRAIN_ATTACH));
    round::<BrainRef>(&p(method::BRAIN_DETACH));
    round::<Empty>(&r(method::BRAIN_DETACH));
    round::<BrainRef>(&p(method::BRAIN_CURSOR));
    round::<BrainCursorResult>(&r(method::BRAIN_CURSOR));
    round::<WakePolicyUpdate>(&p(method::BRAIN_WAKE_POLICY_UPDATE));
    round::<Empty>(&r(method::BRAIN_WAKE_POLICY_UPDATE));
    round::<BrainInject>(&p(method::BRAIN_INJECT));
    round::<BrainInjectResult>(&r(method::BRAIN_INJECT));
    round::<TurnSteer>(&p(method::TURN_STEER));
    round::<Empty>(&r(method::TURN_STEER));
    round::<BrainRef>(&p(method::TURN_INTERRUPT));
    round::<Empty>(&r(method::TURN_INTERRUPT));
    round::<TurnPropose>(&p(method::TURN_PROPOSE));
    round::<TurnProposeResult>(&r(method::TURN_PROPOSE));
    round::<TurnSay>(&p(method::TURN_SAY));
    round::<TurnProposeResult>(&r(method::TURN_SAY));
    round::<TurnAskHuman>(&p(method::TURN_ASK_HUMAN));
    round::<TurnProposeResult>(&r(method::TURN_ASK_HUMAN));
    round::<TurnDone>(&p(method::TURN_DONE));
    round::<Empty>(&r(method::TURN_DONE));
    round::<Empty>(&p(method::STREAM_LIST));
    round::<StreamListResult>(&r(method::STREAM_LIST));
    round::<StreamSubscribe>(&p(method::STREAM_SUBSCRIBE));
    round::<StreamChannel>(&r(method::STREAM_SUBSCRIBE));
    round::<PluginRegister>(&p(method::PLUGIN_REGISTER));
    round::<PluginRegisterResult>(&r(method::PLUGIN_REGISTER));
    round::<FiberRef>(&p(method::PLUGIN_HEARTBEAT));
    round::<Empty>(&r(method::PLUGIN_HEARTBEAT));
    round::<FiberRef>(&p(method::PLUGIN_UNREGISTER));
    round::<Empty>(&r(method::PLUGIN_UNREGISTER));
    round::<EventEmit>(&p(method::EVENT_EMIT));
    round::<EventEmitResult>(&r(method::EVENT_EMIT));
    round::<LogTail>(&p(method::LOG_TAIL));
    round::<LogTailResult>(&r(method::LOG_TAIL));
    round::<LogReplay>(&p(method::LOG_REPLAY));
    round::<LogReplayResult>(&r(method::LOG_REPLAY));
    round::<Empty>(&p(method::LOG_FLUSH));
    round::<Empty>(&r(method::LOG_FLUSH));
    round::<ApprovalAttach>(&p(method::APPROVAL_ATTACH));
    round::<Empty>(&r(method::APPROVAL_ATTACH));
    round::<Empty>(&p(method::APPROVAL_DETACH));
    round::<Empty>(&r(method::APPROVAL_DETACH));
    round::<ApprovalDetached>(&n(method::APPROVAL_DETACHED));
    round::<ApprovalRequest>(&p(method::APPROVAL_REQUEST));
    round::<ApprovalDecision>(&r(method::APPROVAL_REQUEST));
    round::<CapabilityExecute>(&p(method::CAPABILITY_EXECUTE));
    round::<CapabilityAccepted>(&r(method::CAPABILITY_EXECUTE));
    round::<ExecRef>(&n(method::CAPABILITY_HEARTBEAT));
    round::<CapabilityResult>(&n(method::CAPABILITY_RESULT));
    round::<CapabilityCancel>(&n(method::CAPABILITY_CANCEL));
    round::<BrainTurn>(&n(method::BRAIN_TURN));
    round::<BrainSteerNotice>(&n(method::BRAIN_STEER));
    round::<BrainRef>(&n(method::BRAIN_INTERRUPT));
    round::<oeh_proto::WorldEvent>(&n(method::EVENT));
    round::<FiberStateInfo>(&n(method::PLUGIN_STATE));
    round::<ShutdownNotice>(&n(method::SHUTDOWN));
}

/// `capability/execute` 的不变量：admission 保证顶层 `params == proposal.params`
/// （两者都是规则链跑完之后的那一份）；这份 fixture 必须体现这一点，否则一个照抄
/// fixture 写测试的 `Executor` 会验证一种不可能发生的状态。
#[test]
fn capability_execute_fixture_keeps_the_params_invariant() {
    let req = fixture(method::CAPABILITY_EXECUTE)["request"]["params"].clone();
    let params = req["params"].clone();
    let proposal_params = req["proposal"]["params"].clone();
    assert_eq!(
        params, proposal_params,
        "capability/execute: params must equal proposal.params"
    );
}

#[test]
fn message_untagged_order_distinguishes_the_three_shapes() {
    let req: Message =
        serde_json::from_str(r#"{"jsonrpc":"2.0","id":"k1","method":"x","params":{}}"#).unwrap();
    assert!(matches!(req, Message::Request(_)));
    let resp: Message = serde_json::from_str(r#"{"jsonrpc":"2.0","id":7,"result":{}}"#).unwrap();
    assert!(matches!(resp, Message::Response(_)));
    let err: Message = serde_json::from_str(
        r#"{"jsonrpc":"2.0","id":7,"error":{"code":-32601,"message":"nope"}}"#,
    )
    .unwrap();
    assert!(matches!(err, Message::Response(_)));
    let n: Message =
        serde_json::from_str(r#"{"jsonrpc":"2.0","method":"event","params":{}}"#).unwrap();
    assert!(matches!(n, Message::Notification(_)));
}
