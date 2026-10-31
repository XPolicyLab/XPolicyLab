use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use oeh_kernel::fiber::{FiberState, FiberTree};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::{Ctx, Plugin, PluginError, ServiceRegistry};
use oeh_proto::Manifest;
use serde_json::Value;

/// 提供一个服务；可被外部开关“撤销”来模拟 Provider 消失。
struct Provider {
    name: &'static str,
    disposed: Arc<AtomicUsize>,
}
impl Plugin for Provider {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name).provide(&[self.name])
    }
    fn apply(&self, ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        ctx.provide(self.name, Arc::new(String::from(self.name)))?;
        let d = self.disposed.clone();
        ctx.on_dispose(Box::new(move || {
            d.fetch_add(1, Ordering::SeqCst);
        }));
        Ok(())
    }
}

struct Consumer {
    name: &'static str,
    needs: &'static [&'static str],
    applied: Arc<AtomicUsize>,
    disposed: Arc<AtomicUsize>,
}
impl Plugin for Consumer {
    fn manifest(&self) -> Manifest {
        Manifest::new(self.name).inject(self.needs)
    }
    fn apply(&self, ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        for n in self.needs {
            ctx.get::<String>(n)?;
        }
        self.applied.fetch_add(1, Ordering::SeqCst);
        let d = self.disposed.clone();
        ctx.on_dispose(Box::new(move || {
            d.fetch_add(1, Ordering::SeqCst);
        }));
        Ok(())
    }
}

struct Broken {
    provides_but_forgets: bool,
    effect_ran: Arc<AtomicBool>,
}
impl Plugin for Broken {
    fn manifest(&self) -> Manifest {
        if self.provides_but_forgets {
            Manifest::new("broken").provide(&["ghost"])
        } else {
            Manifest::new("broken")
        }
    }
    fn apply(&self, ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        let e = self.effect_ran.clone();
        ctx.on_dispose(Box::new(move || e.store(true, Ordering::SeqCst)));
        if self.provides_but_forgets {
            Ok(())
        } else {
            Err(PluginError::Other("boom".into()))
        }
    }
}

fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
}

#[test]
fn activation_is_order_independent() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg);
    let (applied, disposed) = counters();
    // consumer 行在 provider 之前：行顺序不携带加载语义
    tree.add(
        EntryRow::new("c"),
        Box::new(Consumer {
            name: "c",
            needs: &["a"],
            applied: applied.clone(),
            disposed,
        }),
    );
    tree.add(
        EntryRow::new("a"),
        Box::new(Provider {
            name: "a",
            disposed: Arc::new(AtomicUsize::new(0)),
        }),
    );
    let r = tree.settle();
    assert_eq!(tree.state_of("a"), Some(FiberState::Active));
    assert_eq!(tree.state_of("c"), Some(FiberState::Active));
    assert_eq!(applied.load(Ordering::SeqCst), 1);
    assert_eq!(r.activated, vec!["a".to_string(), "c".to_string()]);
    assert!(r.pending.is_empty());
}

#[test]
fn missing_provider_leaves_consumer_pending_with_reason() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg);
    let (applied, disposed) = counters();
    tree.add(
        EntryRow::new("c"),
        Box::new(Consumer {
            name: "c",
            needs: &["nothing"],
            applied,
            disposed,
        }),
    );
    let r = tree.settle();
    assert_eq!(tree.state_of("c"), Some(FiberState::Pending));
    assert_eq!(
        r.pending,
        vec![(
            "c".to_string(),
            vec![oeh_proto::ServiceName::from("nothing")]
        )]
    );
}

#[test]
fn provider_loss_cascades_to_pending_and_reverses_effects() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg.clone());
    let a_disposed = Arc::new(AtomicUsize::new(0));
    let (b_applied, b_disposed) = counters();
    let (c_applied, c_disposed) = counters();
    let a = tree.add(
        EntryRow::new("a"),
        Box::new(Provider {
            name: "a",
            disposed: a_disposed.clone(),
        }),
    );
    tree.add(
        EntryRow::new("b"),
        Box::new(Provider {
            name: "b",
            disposed: Arc::new(AtomicUsize::new(0)),
        }),
    );
    // b2 consumes a and provides "b2"; c consumes b2 → 两级级联
    tree.add(
        EntryRow::new("b2"),
        Box::new(Bridge {
            applied: b_applied.clone(),
            disposed: b_disposed.clone(),
        }),
    );
    tree.add(
        EntryRow::new("c"),
        Box::new(Consumer {
            name: "c",
            needs: &["b2"],
            applied: c_applied.clone(),
            disposed: c_disposed.clone(),
        }),
    );
    tree.settle();
    assert_eq!(tree.state_of("c"), Some(FiberState::Active));

    tree.dispose(a);
    let r = tree.settle();
    assert_eq!(tree.state_of("a"), Some(FiberState::Disposed));
    assert_eq!(tree.state_of("b2"), Some(FiberState::Pending));
    assert_eq!(tree.state_of("c"), Some(FiberState::Pending));
    assert_eq!(tree.state_of("b"), Some(FiberState::Active));
    assert_eq!(a_disposed.load(Ordering::SeqCst), 1);
    assert_eq!(b_disposed.load(Ordering::SeqCst), 1);
    assert_eq!(c_disposed.load(Ordering::SeqCst), 1);
    assert!(!reg.available(&oeh_proto::ServiceName::from("b2")));
    assert_eq!(r.deactivated, vec!["b2".to_string(), "c".to_string()]);

    // provider 回来 → 依赖者自动重新激活
    tree.add(
        EntryRow::new("a"),
        Box::new(Provider {
            name: "a",
            disposed: Arc::new(AtomicUsize::new(0)),
        }),
    );
    tree.settle();
    assert_eq!(tree.state_of("b2"), Some(FiberState::Active));
    assert_eq!(tree.state_of("c"), Some(FiberState::Active));
    assert_eq!(c_applied.load(Ordering::SeqCst), 2);
}

/// inject a, provide b2
struct Bridge {
    applied: Arc<AtomicUsize>,
    disposed: Arc<AtomicUsize>,
}
impl Plugin for Bridge {
    fn manifest(&self) -> Manifest {
        Manifest::new("b2").inject(&["a"]).provide(&["b2"])
    }
    fn apply(&self, ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        ctx.get::<String>("a")?;
        ctx.provide("b2", Arc::new(String::from("b2")))?;
        self.applied.fetch_add(1, Ordering::SeqCst);
        let d = self.disposed.clone();
        ctx.on_dispose(Box::new(move || {
            d.fetch_add(1, Ordering::SeqCst);
        }));
        Ok(())
    }
}

#[test]
fn apply_error_marks_failed_and_reverses_partial_effects() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg);
    let ran = Arc::new(AtomicBool::new(false));
    tree.add(
        EntryRow::new("broken"),
        Box::new(Broken {
            provides_but_forgets: false,
            effect_ran: ran.clone(),
        }),
    );
    let r = tree.settle();
    assert_eq!(tree.state_of("broken"), Some(FiberState::Failed));
    assert!(
        ran.load(Ordering::SeqCst),
        "partial effects must be reversed"
    );
    assert_eq!(r.failed.len(), 1);
    assert!(tree.last_error("broken").unwrap().contains("boom"));
}

#[test]
fn declared_provide_not_provided_is_failed() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg);
    tree.add(
        EntryRow::new("broken"),
        Box::new(Broken {
            provides_but_forgets: true,
            effect_ran: Arc::new(AtomicBool::new(false)),
        }),
    );
    let r = tree.settle();
    assert_eq!(tree.state_of("broken"), Some(FiberState::Failed));
    assert!(r.failed[0].1.contains("ghost"));
}

#[test]
fn dispose_all_reverses_everything() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg.clone());
    let ad = Arc::new(AtomicUsize::new(0));
    let (ca, cd) = counters();
    tree.add(
        EntryRow::new("a"),
        Box::new(Provider {
            name: "a",
            disposed: ad.clone(),
        }),
    );
    tree.add(
        EntryRow::new("c"),
        Box::new(Consumer {
            name: "c",
            needs: &["a"],
            applied: ca,
            disposed: cd.clone(),
        }),
    );
    tree.settle();
    tree.dispose_all();
    assert_eq!(ad.load(Ordering::SeqCst), 1);
    assert_eq!(cd.load(Ordering::SeqCst), 1);
    assert!(reg.names().is_empty());
    assert!(tree
        .states()
        .iter()
        .all(|(_, s)| *s == FiberState::Disposed));
}

/// I4: apply 里 panic 不能穿过 settle()。必须走和 Err 一样的 fail-closed 路径。
struct Panicker {
    effect_ran: Arc<AtomicBool>,
}
impl Plugin for Panicker {
    fn manifest(&self) -> Manifest {
        Manifest::new("panicker")
    }
    fn apply(&self, ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        let e = self.effect_ran.clone();
        ctx.on_dispose(Box::new(move || e.store(true, Ordering::SeqCst)));
        panic!("kaboom");
    }
}

#[test]
fn apply_panic_is_contained_and_reverses_effects() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg);
    let ran = Arc::new(AtomicBool::new(false));
    tree.add(
        EntryRow::new("panicker"),
        Box::new(Panicker {
            effect_ran: ran.clone(),
        }),
    );
    // panic hook 静音：这条 panic 是被测行为，不该污染测试输出
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = tree.settle();
    std::panic::set_hook(prev);

    assert_eq!(tree.state_of("panicker"), Some(FiberState::Failed));
    assert!(
        ran.load(Ordering::SeqCst),
        "effects registered before the panic must be reversed"
    );
    assert_eq!(r.failed.len(), 1);
    let err = tree.last_error("panicker").unwrap();
    assert!(err.contains("panicked"), "{err}");
    assert!(err.contains("kaboom"), "{err}");
}

/// 一个什么都不做、永远激活成功的插件（剪枝只关心行本身，不关心行里装的是什么）。
struct Trivial;
impl Plugin for Trivial {
    fn manifest(&self) -> Manifest {
        Manifest::new("trivial")
    }
    fn apply(&self, _ctx: &Ctx, _c: &Value) -> Result<(), PluginError> {
        Ok(())
    }
}

/// `prune_disposed` 的两头：上限，以及"最近撤掉的那些还在"。
///
/// `dispose` 留着行是对的——刚掉的那个还要能被问到——但跨进程插件（`plugin/register`）
/// 把"加一行"变成了客户端能随意重复的动作，于是"留着"必须有个头。剪掉的那些代价说清楚：
/// `state_of`/`last_error` 在那些 key 上从 `Some(Disposed)` 变成 `None`。
#[test]
fn prune_disposed_caps_the_tombstones_and_keeps_the_newest() {
    let reg = Arc::new(ServiceRegistry::default());
    let mut tree = FiberTree::new(reg);
    let cap = FiberTree::MAX_DISPOSED_ROWS;
    let churn = cap + 8;

    let live = tree.add(EntryRow::new("p").with_name("live"), Box::new(Trivial));
    let mut ids = Vec::new();
    for i in 0..churn {
        let id = tree.add(
            EntryRow::new("p").with_name(format!("g{i}")),
            Box::new(Trivial),
        );
        tree.dispose(id);
        ids.push(id);
    }
    assert_eq!(
        tree.ids().len(),
        churn + 1,
        "nothing is dropped before prune"
    );

    tree.prune_disposed();

    assert_eq!(
        tree.ids().len(),
        cap + 1,
        "the live row plus exactly the cap"
    );
    assert!(tree.ids().contains(&live), "a live row is never pruned");
    assert!(
        tree.ids().contains(ids.last().unwrap()),
        "the most recently disposed row must survive"
    );
    assert!(
        !tree.ids().contains(&ids[0]),
        "the oldest disposed row must be gone"
    );
    // 剪掉的 key 从树上看就是"没有过"。
    assert!(tree.state_of("g0").is_none());
    assert!(tree.last_error("g0").is_none());
    assert_eq!(
        tree.state_of(&format!("g{}", churn - 1)),
        Some(FiberState::Disposed)
    );
    // id 不回头：剪枝不碰 `next_id`。
    let next = tree.add(EntryRow::new("p").with_name("after"), Box::new(Trivial));
    assert!(next.0 > ids.last().unwrap().0);
}
