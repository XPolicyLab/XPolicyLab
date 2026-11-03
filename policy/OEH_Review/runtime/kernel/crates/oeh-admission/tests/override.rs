//! 内置 `say` 可以被适配器顶替，别的内置名字不行。
//!
//! 为什么单单是 `say`：**说话是身体的一部分**。一台带扬声器的机器人自己有嗓子，而内核
//! 的内置 `say` 只是把话记进世界日志——两者同名，但只有一个真的发出声音。所以内置的
//! `say` 是一个**兜底**：没有适配器的时候它保证 `say` 这个名字在能力面上（脑的工具表
//! 不会因为少了一台机器人而变形），适配器一上线就该让位。
//!
//! `stop` / `recover` / `ask_*` 相反：`stop` 是内核撤销当前执行的那一下，`recover` 是急停
//! 唯一的出路，`ask_*` 是脑与人、脑与脑之间的通道——它们的语义全在内核这一侧，一个
//! 适配器顶替掉 `stop` 就等于把"叫停"交给了要被叫停的那一方。这些名字重名照旧是
//! `AlreadyRegistered`，而且整次 `plugin/register` 失败。

use std::sync::Arc;

use oeh_admission::capability::{
    CapabilityClass, CapabilityError, CapabilityRegistry, CapabilitySpec, ExecCtx, ExecOutcome,
    ExecutionCall, Executor, RiskLevel, KERNEL_PROVIDER,
};
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::{Disposer, ManualClock, WorldLog};
use serde_json::{json, Value};

struct Nop;
impl Executor for Nop {
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

fn text_schema() -> Value {
    json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
}

/// 内核的 `say`：speech、可被顶替。
fn builtin_say() -> CapabilitySpec {
    CapabilitySpec {
        name: "say".into(),
        class: CapabilityClass::Speech,
        risk_level: RiskLevel::None,
        params_schema: text_schema(),
        lease_ttl_s: 5.0,
        overridable: true,
    }
}

/// 内核独占的名字。`ask_human` 也是 speech——它在这里是为了证明"类别对"并不够，
/// 被顶替的那一项自己得说 `overridable`。
fn kernel_only(name: &str, class: CapabilityClass) -> CapabilitySpec {
    CapabilitySpec {
        name: name.into(),
        class,
        risk_level: RiskLevel::None,
        params_schema: json!({"type": "object"}),
        lease_ttl_s: 5.0,
        overridable: false,
    }
}

/// 适配器声明的那一份：线格式上 `overridable` 永远是 false（客户端说了不算，见
/// `oeh_rpc::handlers::plugin::spec_of`），`lease_ttl_s` 故意和内置的不同，这样
/// "生效的是哪一份 spec"是可断言的。
fn adapter_say(class: CapabilityClass) -> CapabilitySpec {
    CapabilitySpec {
        name: "say".into(),
        class,
        risk_level: RiskLevel::None,
        params_schema: text_schema(),
        lease_ttl_s: 7.5,
        overridable: false,
    }
}

fn registry() -> (CapabilityRegistry, Arc<WorldLog>) {
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(1))));
    (CapabilityRegistry::with_log(log.clone()), log)
}

/// 世界日志里这一类事件的 `(payload, salience)`，按顺序。
fn events(log: &WorldLog, kind: &str) -> Vec<(Value, u8)> {
    log.since(0)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind.as_str() == kind)
        .map(|e| (e.payload, e.salience))
        .collect()
}

/// 一次**被拒**的注册的错误。不能用 `unwrap_err`：`Disposer` 不是 `Debug`，而这里要断言
/// 的恰恰是"没有拿到 Disposer"。
fn refused(r: Result<Disposer, CapabilityError>) -> CapabilityError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("the registration was accepted; it had to be refused"),
    }
}

fn provider_of(r: &CapabilityRegistry, name: &str) -> Option<String> {
    r.list_entries()
        .into_iter()
        .find(|e| e.spec.name == name)
        .map(|e| e.provider)
}

#[test]
fn an_adapters_speech_capability_takes_over_the_built_in_say() {
    let (r, log) = registry();
    let builtin: Arc<dyn Executor> = Arc::new(Nop);
    let _b = r.register(builtin_say(), builtin.clone()).unwrap();
    assert_eq!(provider_of(&r, "say").as_deref(), Some(KERNEL_PROVIDER));

    let robot: Arc<dyn Executor> = Arc::new(Nop);
    let _d = r
        .register_from(
            "py-voice",
            adapter_say(CapabilityClass::Speech),
            robot.clone(),
        )
        .unwrap();

    let (spec, exec) = r.get("say").unwrap();
    assert!(Arc::ptr_eq(&exec, &robot), "lookup must reach the adapter");
    // 生效的是适配器那一份**完整的** spec，不只是执行器：租约、schema 都是它的。
    assert_eq!(spec.lease_ttl_s, 7.5);
    assert!(!spec.overridable, "an override is not itself overridable");
    assert_eq!(provider_of(&r, "say").as_deref(), Some("py-voice"));
    // 顶替不是"多一项"：能力面上 `say` 仍然只有一个。
    assert_eq!(r.list().iter().filter(|s| s.name == "say").count(), 1);

    assert_eq!(
        events(&log, "capability/overridden"),
        vec![(json!({"name": "say", "by": "py-voice"}), 100)]
    );
}

#[test]
fn dropping_the_overriding_registration_puts_the_built_in_back() {
    let (r, log) = registry();
    let builtin: Arc<dyn Executor> = Arc::new(Nop);
    let _b = r.register(builtin_say(), builtin.clone()).unwrap();
    let d = r
        .register_from(
            "py-voice",
            adapter_say(CapabilityClass::Speech),
            Arc::new(Nop),
        )
        .unwrap();

    // 适配器断连 / 注销 / Fiber 被 dispose 走的都是这一条：注册返回的那个 Disposer。
    d();

    let (spec, exec) = r.get("say").unwrap();
    assert!(Arc::ptr_eq(&exec, &builtin), "the built-in must be back");
    assert_eq!(spec.lease_ttl_s, 5.0);
    assert!(spec.overridable, "and it is overridable again");
    assert_eq!(provider_of(&r, "say").as_deref(), Some(KERNEL_PROVIDER));
    assert_eq!(
        events(&log, "capability/restored"),
        vec![(json!({"name": "say", "from": "py-voice"}), 100)]
    );
}

#[test]
fn only_a_speech_capability_may_take_say_over() {
    let (r, log) = registry();
    let builtin: Arc<dyn Executor> = Arc::new(Nop);
    let _b = r.register(builtin_say(), builtin.clone()).unwrap();

    // 一个叫 `say` 的 motion：名字对、可顶替标记也对，但类别不对——脑的工具表里
    // `say` 是"说一句话"，用一个运动能力顶替它就是换掉了语义。
    let e = refused(r.register_from(
        "py-voice",
        adapter_say(CapabilityClass::Motion),
        Arc::new(Nop),
    ));
    assert!(matches!(e, CapabilityError::AlreadyRegistered(n) if n == "say"));

    assert!(Arc::ptr_eq(&r.get("say").unwrap().1, &builtin));
    assert_eq!(provider_of(&r, "say").as_deref(), Some(KERNEL_PROVIDER));
    assert!(events(&log, "capability/overridden").is_empty());
}

#[test]
fn the_names_the_kernel_keeps_are_still_a_duplicate() {
    let (r, log) = registry();
    let _s = r
        .register(kernel_only("stop", CapabilityClass::Control), Arc::new(Nop))
        .unwrap();
    let _a = r
        .register(
            kernel_only("ask_human", CapabilityClass::Speech),
            Arc::new(Nop),
        )
        .unwrap();

    for name in ["stop", "ask_human"] {
        let mut spec = adapter_say(CapabilityClass::Speech);
        spec.name = name.into();
        let e = refused(r.register_from("py-voice", spec, Arc::new(Nop)));
        assert!(
            matches!(&e, CapabilityError::AlreadyRegistered(n) if n == name),
            "{name}: {e:?}"
        );
        assert_eq!(provider_of(&r, name).as_deref(), Some(KERNEL_PROVIDER));
    }
    assert!(events(&log, "capability/overridden").is_empty());
}

#[test]
fn a_second_adapter_cannot_take_say_from_the_first() {
    let (r, _log) = registry();
    let _b = r.register(builtin_say(), Arc::new(Nop)).unwrap();
    let first: Arc<dyn Executor> = Arc::new(Nop);
    let _d = r
        .register_from(
            "py-voice",
            adapter_say(CapabilityClass::Speech),
            first.clone(),
        )
        .unwrap();

    // 顶替过的那一项不再是"可顶替"的：两台机器人抢同一张嘴没有一个正确答案，而先到的
    // 那台正在用它。第二次注册整个失败（和今天的重名一模一样），不是悄悄接管。
    let e = refused(r.register_from(
        "py-voice-2",
        adapter_say(CapabilityClass::Speech),
        Arc::new(Nop),
    ));
    assert!(matches!(e, CapabilityError::AlreadyRegistered(n) if n == "say"));
    assert!(Arc::ptr_eq(&r.get("say").unwrap().1, &first));
}

#[test]
fn a_built_in_disposed_while_overridden_does_not_come_back() {
    let (r, log) = registry();
    let b = r.register(builtin_say(), Arc::new(Nop)).unwrap();
    let d = r
        .register_from(
            "py-voice",
            adapter_say(CapabilityClass::Speech),
            Arc::new(Nop),
        )
        .unwrap();

    // 内核先撤（admission 那一行被 dispose），适配器后走。兜底的那一份已经没有主人了，
    // 不能在适配器断连时被请回来——那会让一个已经撤销的执行器重新出现在能力面上。
    b();
    assert!(r.get("say").is_some(), "the adapter still holds `say`");
    d();
    assert!(r.get("say").is_none(), "nothing to fall back to");
    assert!(events(&log, "capability/restored").is_empty());
}
