use std::sync::Arc;

use oeh_admission::capability::{
    validate_params, CapabilityClass, CapabilityError, CapabilityRegistry, CapabilitySpec, ExecCtx,
    ExecOutcome, ExecutionCall, Executor, RiskLevel,
};
use oeh_kernel::proposal::{BoxFuture, Proposal, ProposalId};
use oeh_kernel::ManualClock;
use serde_json::json;

struct Noop;
impl Executor for Noop {
    fn execute(&self, _c: ExecutionCall, _x: ExecCtx) -> BoxFuture<ExecOutcome> {
        Box::pin(async {
            ExecOutcome {
                ok: true,
                result: json!({}),
                error: None,
            }
        })
    }
}

fn spec(name: &str) -> CapabilitySpec {
    CapabilitySpec {
        name: name.into(),
        class: CapabilityClass::Motion,
        risk_level: RiskLevel::Medium,
        params_schema: json!({"type": "object", "properties": {"vx": {"type": "number", "minimum": -0.5, "maximum": 0.5}, "duration_s": {"type": "number", "minimum": 0}}, "required": ["vx"], "additionalProperties": false}),
        lease_ttl_s: 5.0,
        overridable: false,
    }
}

#[test]
fn registry_register_get_list_and_duplicate() {
    let r = CapabilityRegistry::new();
    let d = r.register(spec("walk"), Arc::new(Noop)).unwrap();
    assert!(matches!(
        r.register(spec("walk"), Arc::new(Noop)),
        Err(CapabilityError::AlreadyRegistered(_))
    ));
    assert_eq!(r.list().len(), 1);
    assert!(r.get("walk").is_some());
    d();
    assert!(r.get("walk").is_none());
    let _d2 = r.register(spec("walk"), Arc::new(Noop)).unwrap();
}

#[test]
fn params_validation_subset() {
    let s = spec("walk").params_schema;
    assert!(validate_params(&s, &json!({"vx": 0.2})).is_ok());
    assert!(validate_params(&s, &json!({"vx": 0.9}))
        .unwrap_err()
        .contains("maximum"));
    assert!(validate_params(&s, &json!({}))
        .unwrap_err()
        .contains("required"));
    assert!(validate_params(&s, &json!({"vx": "fast"}))
        .unwrap_err()
        .contains("type"));
    assert!(validate_params(&s, &json!({"vx": 0.1, "wz": 1}))
        .unwrap_err()
        .contains("additional"));
    assert!(
        validate_params(&json!({}), &json!({"anything": 1})).is_ok(),
        "empty schema accepts anything"
    );
    assert!(validate_params(
        &json!({"type":"object","properties":{"mode":{"type":"string","enum":["a","b"]}}}),
        &json!({"mode":"c"})
    )
    .unwrap_err()
    .contains("enum"));

    // `integer` must accept whole-number floats (`1.0`), not just serde_json's i64/u64
    // storage variants — the check is on the mathematical value, not the representation.
    let int_schema = json!({"type":"object","properties":{"n":{"type":"integer"}}});
    assert!(validate_params(&int_schema, &json!({"n": 1})).is_ok());
    assert!(validate_params(&int_schema, &json!({"n": 1.0})).is_ok());
    assert!(validate_params(&int_schema, &json!({"n": 1.5}))
        .unwrap_err()
        .contains("type"));
}

#[tokio::test]
async fn exec_ctx_cancel_and_heartbeat() {
    let clock = Arc::new(ManualClock::new(10));
    let (mut ctx, ctl) =
        ExecCtx::new_pair("t-1".into(), ProposalId("t-1-p1".into()), clock.clone());
    assert!(!ctx.is_cancelled());
    ctx.heartbeat();
    assert_eq!(ctl.last_heartbeat_ns(), 10);
    ctl.cancel();
    assert!(ctx.is_cancelled());
    ctx.cancelled().await; // 已取消时立即返回
    let _ = Proposal {
        proposal_id: ProposalId("x".into()),
        brain_id: "b".into(),
        turn: 1,
        trace_id: "t".into(),
        capability: "walk".into(),
        params: json!({}),
    };
}
