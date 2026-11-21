use std::any::Any;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use oeh_proto::{Manifest, ServiceName};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::services::{FiberId, ServiceRegistry};

pub type Disposer = Box<dyn FnOnce() + Send>;

/// 插件表里每一行的配置，按行的 `id` 索引（见 [`Ctx::sibling_config`]）。
pub type SiblingConfigs = BTreeMap<String, Value>;

/// 插件挂载期间登记的副作用，卸载时逆序撤销。
#[derive(Default)]
pub struct Effects {
    disposers: Vec<Disposer>,
}

impl Effects {
    pub fn push(&mut self, d: Disposer) {
        self.disposers.push(d);
    }
    pub fn len(&self) -> usize {
        self.disposers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.disposers.is_empty()
    }
    pub fn dispose(self) {
        for d in self.disposers.into_iter().rev() {
            d();
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("service `{0}` is not available")]
    MissingService(String),
    #[error("service `{0}` has a different type than requested")]
    ServiceType(String),
    #[error("service `{0}` is already provided by another fiber")]
    AlreadyProvided(String),
    #[error("invalid config for plugin `{plugin}`: {reason}")]
    Config { plugin: String, reason: String },
    #[error("{0}")]
    Other(String),
}

pub trait Plugin: Send + Sync + 'static {
    fn manifest(&self) -> Manifest;
    /// 在此期间通过 ctx 登记的所有副作用都属于当前 Fiber。
    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError>;
}

/// `alive` 和 `effects` 必须在同一把锁下：分开读写会让 `take_effects_and_kill` 置位和 drain
/// 之间挤进来的 `on_dispose` 把 disposer 推进一个再也不会被 drain 的 ctx（服务注册泄漏）。
struct CtxEffects {
    alive: bool,
    effects: Effects,
}

struct CtxInner {
    fiber: FiberId,
    plugin_name: String,
    registry: Arc<ServiceRegistry>,
    state: Mutex<CtxEffects>,
}

/// 插件看到的运行时句柄。可 Clone 进回调。
#[derive(Clone)]
pub struct Ctx {
    inner: Arc<CtxInner>,
    /// 同一张插件表里**别的行**的配置，`apply` 那一刻的只读快照。见
    /// [`Ctx::sibling_config`]。
    siblings: Arc<SiblingConfigs>,
}

impl Ctx {
    pub fn new(
        fiber: FiberId,
        plugin_name: impl Into<String>,
        registry: Arc<ServiceRegistry>,
    ) -> Self {
        Self {
            inner: Arc::new(CtxInner {
                fiber,
                plugin_name: plugin_name.into(),
                registry,
                state: Mutex::new(CtxEffects {
                    alive: true,
                    effects: Effects::default(),
                }),
            }),
            siblings: Arc::new(SiblingConfigs::new()),
        }
    }

    /// 内核用：把这一轮的插件表快照挂上去（`FiberTree::activate`）。
    pub(crate) fn with_siblings(mut self, siblings: Arc<SiblingConfigs>) -> Self {
        self.siblings = siblings;
        self
    }

    /// 插件表里另一行的配置（按行的 `id`），`apply` 那一刻的只读快照。
    ///
    /// **只用来做一致性检查，不用来读别人的设置。** 存在的理由是急停这类"同一个事实由
    /// 两行各配一遍"的地方：`admission.estop_flag` 和 `distill-estop.flag_path` 指的是
    /// 同一个标志文件，一份只改了其中一个的 profile 会让另一个继续盯着默认路径——急停
    /// 那一刻它说"没按下"，而写 profile 的人完全有理由以为自己已经改好了。这种不一致
    /// 必须在 boot 当场响，而不是等到急停那天。
    ///
    /// 拿别人的配置去**驱动自己的行为**是另一回事，不要这么用：那会让两行之间多出一条
    /// 不在 `Manifest` 里的隐式依赖，而 `inject`/`provide` 才是这个工作区表达依赖的地方
    /// （它有 settle 的顺序保证，这里没有——快照里那一行此刻可能还没 apply，甚至根本
    /// 起不来）。
    ///
    /// 快照是**整张表**（包括调用方自己那一行）在 `apply` 那一刻的样子；同一个 `id` 出现
    /// 多行时给的是**第一行**，`disabled: true` 的行根本不在表里。
    ///
    /// **只在 boot / settle 的那一刻成立。** 一致性检查跑在 `apply` 里，而 `apply` 不会
    /// 因为别人后来被加进来（`Op::AddFiber`）而重跑——运行时新插进来的一行不受这里任何
    /// 检查的约束。
    pub fn sibling_config(&self, plugin_id: &str) -> Option<Value> {
        self.siblings.get(plugin_id).cloned()
    }

    pub fn fiber_id(&self) -> FiberId {
        self.inner.fiber
    }
    pub fn plugin_name(&self) -> &str {
        &self.inner.plugin_name
    }
    pub fn registry(&self) -> &Arc<ServiceRegistry> {
        &self.inner.registry
    }

    pub fn provide<T: Any + Send + Sync>(
        &self,
        name: &str,
        value: Arc<T>,
    ) -> Result<(), PluginError> {
        let sname = ServiceName::from(name);
        self.inner
            .registry
            .provide(sname.clone(), self.inner.fiber, value)?;
        let reg = self.inner.registry.clone();
        let fid = self.inner.fiber;
        self.on_dispose(Box::new(move || {
            reg.revoke(&sname, fid);
        }));
        Ok(())
    }

    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Result<Arc<T>, PluginError> {
        self.inner.registry.get::<T>(&ServiceName::from(name))
    }

    pub fn try_get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        self.get::<T>(name).ok()
    }

    /// 登记一个撤销动作。Fiber 已退化时立即执行。检查与入队在同一把锁下完成，
    /// 所以不会出现“检查时还活着、入队时已经被 drain”的丢失窗口。
    pub fn on_dispose(&self, d: Disposer) {
        {
            let mut g = self.inner.state.lock();
            if g.alive {
                g.effects.push(d);
                return;
            }
        }
        d();
    }

    pub fn parse_config<C: DeserializeOwned>(&self, v: &Value) -> Result<C, PluginError> {
        serde_json::from_value(v.clone()).map_err(|e| PluginError::Config {
            plugin: self.inner.plugin_name.clone(),
            reason: e.to_string(),
        })
    }

    /// 周期任务；需要在 tokio runtime 内调用。第一次 tick 在一个周期之后触发（不是立即），
    /// 这样刚 boot 完就投递的 Op 不会和插件的首帧竞争。
    pub fn spawn_interval(&self, period: Duration, f: impl Fn() + Send + 'static) {
        let h = tokio::spawn(async move {
            let mut iv = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
            iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                iv.tick().await;
                f();
            }
        });
        self.on_dispose(Box::new(move || h.abort()));
    }

    pub fn spawn(&self, fut: impl Future<Output = ()> + Send + 'static) {
        let h = tokio::spawn(fut);
        self.on_dispose(Box::new(move || h.abort()));
    }

    /// 内核用：取走全部 Effects 并标记死亡（同一把锁里一次完成）。
    pub fn take_effects_and_kill(&self) -> Effects {
        let mut g = self.inner.state.lock();
        g.alive = false;
        std::mem::take(&mut g.effects)
    }

    pub fn effect_count(&self) -> usize {
        self.inner.state.lock().effects.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn effects_dispose_in_reverse_order() {
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut e = Effects::default();
        for i in 0..3 {
            let o = order.clone();
            e.push(Box::new(move || o.lock().push(i)));
        }
        assert_eq!(e.len(), 3);
        e.dispose();
        assert_eq!(*order.lock(), vec![2, 1, 0]);
    }

    #[test]
    fn ctx_provide_registers_revoke_effect() {
        let reg = Arc::new(ServiceRegistry::default());
        let ctx = Ctx::new(FiberId(1), "p", reg.clone());
        ctx.provide("svc", Arc::new(5u32)).unwrap();
        assert_eq!(*ctx.get::<u32>("svc").unwrap(), 5);
        assert!(ctx.try_get::<u32>("nope").is_none());
        ctx.take_effects_and_kill().dispose();
        assert!(!reg.available(&ServiceName::from("svc")));
    }

    #[test]
    fn dead_ctx_runs_disposer_immediately() {
        let reg = Arc::new(ServiceRegistry::default());
        let ctx = Ctx::new(FiberId(1), "p", reg);
        ctx.take_effects_and_kill().dispose();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        ctx.on_dispose(Box::new(move || {
            h.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    /// I3: kill 与 drain 必须是同一个原子步骤——kill 之前登记的 disposer 一条都不能丢，
    /// kill 之后登记的立刻执行（而不是推进一个没人会 drain 的死 ctx）。
    #[test]
    fn kill_boundary_loses_no_disposer() {
        let reg = Arc::new(ServiceRegistry::default());
        let ctx = Ctx::new(FiberId(1), "p", reg);
        let before = Arc::new(AtomicUsize::new(0));
        let b2 = before.clone();
        ctx.on_dispose(Box::new(move || {
            b2.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(ctx.effect_count(), 1);

        let effects = ctx.take_effects_and_kill();
        assert_eq!(effects.len(), 1, "pre-kill disposer must be handed back");
        assert_eq!(before.load(Ordering::SeqCst), 0, "not run yet");

        // kill 之后登记 → 立即执行，不会留在已经被 drain 的 ctx 里
        let after = Arc::new(AtomicUsize::new(0));
        let a2 = after.clone();
        ctx.on_dispose(Box::new(move || {
            a2.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(after.load(Ordering::SeqCst), 1);
        assert_eq!(ctx.effect_count(), 0);

        effects.dispose();
        assert_eq!(before.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parse_config_reports_plugin_name() {
        #[derive(Debug, serde::Deserialize)]
        struct C {
            #[allow(dead_code)]
            hz: f64,
        }
        let reg = Arc::new(ServiceRegistry::default());
        let ctx = Ctx::new(FiberId(1), "mock-stream", reg);
        let err = ctx
            .parse_config::<C>(&serde_json::json!({"hz": "fast"}))
            .unwrap_err();
        assert!(matches!(err, PluginError::Config { ref plugin, .. } if plugin == "mock-stream"));
    }
}
