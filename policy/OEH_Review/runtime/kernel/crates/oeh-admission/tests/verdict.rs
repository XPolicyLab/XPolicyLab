use std::collections::BTreeMap;
use std::sync::Arc;

use oeh_admission::capability::{CapabilityClass, CapabilitySpec, RiskLevel};
use oeh_admission::verdict::{evaluate_chain, Rule, RuleCtx, Verdict};
use oeh_kernel::fsm::FsmState;
use serde_json::{json, Value};

// `std::panic::set_hook`/`take_hook` are process-global, not per-thread or per-test, so
// only one test in this file that triggers a rule panic should be running at a time —
// `clamp_is_traced_and_panic_is_deny` and `panicking_rule_cannot_leak_partial_params`
// both do this; if `cargo test` ever ran them concurrently, their hook install/restore
// could race and let a panic message leak to stdout (this doesn't affect correctness of
// the assertions, only how clean the test output is).

struct Fixed(&'static str, Verdict);
impl Rule for Fixed {
    fn name(&self) -> &str {
        self.0
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
        self.1.clone()
    }
}
struct Pass(&'static str);
impl Rule for Pass {
    fn name(&self) -> &str {
        self.0
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
        i
    }
}
struct Clamp;
impl Rule for Clamp {
    fn name(&self) -> &str {
        "clamp"
    }
    fn evaluate(&self, p: &mut Value, _c: &RuleCtx, i: Verdict) -> Verdict {
        if let Some(v) = p.get_mut("vx") {
            if v.as_f64().unwrap_or(0.0) > 0.3 {
                *v = json!(0.3);
            }
        }
        i
    }
}
struct Panicky;
impl Rule for Panicky {
    fn name(&self) -> &str {
        "panicky"
    }
    fn evaluate(&self, _p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
        panic!("boom")
    }
}
struct PanicAfterMutate;
impl Rule for PanicAfterMutate {
    fn name(&self) -> &str {
        "panic_after_mutate"
    }
    fn evaluate(&self, p: &mut Value, _c: &RuleCtx, _i: Verdict) -> Verdict {
        p["vx"] = json!(999.0);
        panic!("boom after mutate")
    }
}

fn spec() -> CapabilitySpec {
    CapabilitySpec {
        name: "walk".into(),
        class: CapabilityClass::Motion,
        risk_level: RiskLevel::Medium,
        params_schema: json!({}),
        lease_ttl_s: 5.0,
        overridable: false,
    }
}
fn ctx<'a>(
    state: &'a BTreeMap<String, oeh_kernel::state::StateEntry>,
    spec: &'a CapabilitySpec,
) -> RuleCtx<'a> {
    RuleCtx {
        now_ns: 0,
        fsm: FsmState::Standing,
        state,
        spec,
        estop_engaged: false,
    }
}

#[test]
fn empty_or_missing_required_rules_start_at_ask() {
    let st = BTreeMap::new();
    let sp = spec();
    let c = ctx(&st, &sp);
    let r = evaluate_chain(&[], &[], json!({}), &c);
    assert!(matches!(r.verdict, Verdict::Ask { .. }));

    // A missing required rule only sets the *initial* verdict to `Ask` — the rule
    // loop still runs every registered rule from there. A passthrough rule that just
    // echoes its `incoming` verdict neither tightens nor loosens it, so the chain
    // ends at the same `Ask{"missing required rules: b"}`, the rule still shows up
    // in the trace, and nothing gets flagged as tampering.
    let rules: Vec<(String, Arc<dyn Rule>)> = vec![("a".into(), Arc::new(Pass("a")))];
    let r = evaluate_chain(&rules, &["a".into(), "b".into()], json!({}), &c);
    assert!(matches!(r.verdict, Verdict::Ask { ref reason } if reason.contains("b")));
    assert_eq!(r.trace.len(), 1);
    assert!(!r.tampered);

    let r = evaluate_chain(&rules, &["a".into()], json!({}), &c);
    assert_eq!(r.verdict, Verdict::Allow);

    // A present rule can still tighten the missing-required-rule `Ask` all the way
    // to `Deny` — the precondition failure never skips already-registered rules.
    let rules: Vec<(String, Arc<dyn Rule>)> = vec![
        ("a".into(), Arc::new(Pass("a"))),
        (
            "deny".into(),
            Arc::new(Fixed(
                "deny",
                Verdict::Deny {
                    reason: "no".into(),
                },
            )),
        ),
    ];
    let r = evaluate_chain(&rules, &["a".into(), "b".into()], json!({}), &c);
    assert!(matches!(r.verdict, Verdict::Deny { .. }));
}

#[test]
fn verdict_only_tightens_and_tamper_is_recorded() {
    let st = BTreeMap::new();
    let sp = spec();
    let c = ctx(&st, &sp);
    let rules: Vec<(String, Arc<dyn Rule>)> = vec![
        (
            "ask".into(),
            Arc::new(Fixed(
                "ask",
                Verdict::Ask {
                    reason: "confirm".into(),
                },
            )),
        ),
        ("loosen".into(), Arc::new(Fixed("loosen", Verdict::Allow))),
        (
            "deny".into(),
            Arc::new(Fixed(
                "deny",
                Verdict::Deny {
                    reason: "no".into(),
                },
            )),
        ),
    ];
    let r = evaluate_chain(&rules, &[], json!({}), &c);
    assert!(matches!(r.verdict, Verdict::Deny { .. }));
    assert!(r.tampered);
    assert_eq!(r.trace.len(), 3);
    assert!(r.trace[1].tamper && matches!(r.trace[1].outgoing, Verdict::Deny { .. }));
    assert_eq!(
        r.trace[0].outgoing,
        Verdict::Ask {
            reason: "confirm".into()
        }
    );
}

#[test]
fn clamp_is_traced_and_panic_is_deny() {
    let st = BTreeMap::new();
    let sp = spec();
    let c = ctx(&st, &sp);
    let rules: Vec<(String, Arc<dyn Rule>)> = vec![("clamp".into(), Arc::new(Clamp))];
    let r = evaluate_chain(&rules, &[], json!({"vx": 0.9}), &c);
    assert_eq!(r.params["vx"], json!(0.3));
    assert_eq!(
        r.trace[0].clamped,
        Some(json!({"before": {"vx": 0.9}, "after": {"vx": 0.3}}))
    );
    assert_eq!(r.verdict, Verdict::Allow);

    // Panicking rules print a panic message; install a silent hook so test output stays
    // clean, then restore the default hook afterwards.
    std::panic::set_hook(Box::new(|_| {}));
    let rules: Vec<(String, Arc<dyn Rule>)> = vec![("panicky".into(), Arc::new(Panicky))];
    let r = evaluate_chain(&rules, &[], json!({}), &c);
    let _ = std::panic::take_hook();
    assert!(r.trace[0].panicked && matches!(r.verdict, Verdict::Deny { .. }));
}

#[test]
fn panicking_rule_cannot_leak_partial_params() {
    let st = BTreeMap::new();
    let sp = spec();
    let c = ctx(&st, &sp);

    // A rule that mutates `params` before panicking must not leave that mutation
    // behind: `catch_unwind` only stops the stack unwind, it doesn't roll back the
    // heap mutation the rule already made to `candidate_params` before panicking, so
    // the chain evaluator has to discard it explicitly rather than treat it as a
    // (false) `clamped` change.
    std::panic::set_hook(Box::new(|_| {}));
    let rules: Vec<(String, Arc<dyn Rule>)> =
        vec![("panic_after_mutate".into(), Arc::new(PanicAfterMutate))];
    let r = evaluate_chain(&rules, &[], json!({"vx": 0.1}), &c);
    let _ = std::panic::take_hook();

    assert_eq!(r.params["vx"], json!(0.1));
    assert!(r.trace[0].panicked);
    assert!(r.trace[0].clamped.is_none());
    assert!(matches!(r.verdict, Verdict::Deny { .. }));
}
