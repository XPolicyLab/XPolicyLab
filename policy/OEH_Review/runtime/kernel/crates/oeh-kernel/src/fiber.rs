use std::sync::Arc;

use oeh_proto::{Manifest, ServiceName};
use serde::{Deserialize, Serialize};

use crate::plugin::{Ctx, Plugin, PluginError, SiblingConfigs};
use crate::profile::EntryRow;
use crate::services::{FiberId, ServiceRegistry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FiberState {
    Pending,
    /// 只在 `settle()` 内部的 `activate()` 期间短暂存在；`settle()` 返回前必然变成
    /// `Active` 或 `Failed`。`Kernel::publish_states`（`plugin/state` 世界日志事件）只在
    /// 两次 `settle()` 之间采样，永远不会广播 `"state":"loading"`——消费者不应该等它出现。
    Loading,
    Active,
    Failed,
    Disposed,
}

pub struct Fiber {
    pub id: FiberId,
    pub row: EntryRow,
    pub manifest: Manifest,
    plugin: Box<dyn Plugin>,
    pub state: FiberState,
    ctx: Option<Ctx>,
    pub last_error: Option<String>,
}

#[derive(Debug, Default, PartialEq)]
pub struct SettleReport {
    pub activated: Vec<String>,
    pub deactivated: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub pending: Vec<(String, Vec<ServiceName>)>,
}

/// 插件实例树。`settle()` 把树推到不动点：
/// 所有 inject 满足的 Pending 变 Active，所有 inject 丢失的 Active 退回 Pending。
pub struct FiberTree {
    fibers: Vec<Fiber>,
    registry: Arc<ServiceRegistry>,
    next_id: u32,
}

impl FiberTree {
    pub fn new(registry: Arc<ServiceRegistry>) -> Self {
        Self {
            fibers: Vec::new(),
            registry,
            next_id: 1,
        }
    }

    pub fn registry(&self) -> &Arc<ServiceRegistry> {
        &self.registry
    }

    pub fn add(&mut self, row: EntryRow, plugin: Box<dyn Plugin>) -> FiberId {
        // `FiberId(u32::MAX)` 是 `kernel` 服务的哨兵 provider id（kernel.rs）；一个真实
        // Fiber 绝不能撞上它。2^32-2 次 `add` 才会碰到，实际不可达，但既然这个哨兵已经
        // 是承重结构，留一个断言记下这个耦合。
        debug_assert!(
            self.next_id < u32::MAX,
            "FiberId(u32::MAX) is reserved for the kernel service sentinel"
        );
        let id = FiberId(self.next_id);
        self.next_id += 1;
        let manifest = plugin.manifest();
        self.fibers.push(Fiber {
            id,
            row,
            manifest,
            plugin,
            state: FiberState::Pending,
            ctx: None,
            last_error: None,
        });
        id
    }

    fn missing_injects(&self, f: &Fiber) -> Vec<ServiceName> {
        f.manifest
            .inject
            .iter()
            .filter(|s| !self.registry.available(s))
            .cloned()
            .collect()
    }

    fn deactivate(&mut self, idx: usize) {
        let f = &mut self.fibers[idx];
        if let Some(ctx) = f.ctx.take() {
            ctx.take_effects_and_kill().dispose();
        }
        f.state = FiberState::Pending;
    }

    /// 这一刻表里每一行的配置，按 `id`。给 [`Ctx::sibling_config`] 用——只做跨行的一致性
    /// 检查（急停标志路径那种"同一个事实配两遍"的地方）。
    ///
    /// 每次 `activate` 现取一份：行可以在两次 activate 之间被加进来（`Op::AddFiber`），
    /// 缓存一份就会让后来的行看见一张过期的表。表只有几十行、配置是几十字节的 YAML，
    /// 而 activate 只发生在 boot 与 settle 的不动点迭代里。
    fn sibling_configs(&self) -> Arc<SiblingConfigs> {
        let mut m = SiblingConfigs::new();
        for f in &self.fibers {
            m.entry(f.row.id.clone())
                .or_insert_with(|| f.row.config.clone());
        }
        Arc::new(m)
    }

    fn activate(&mut self, idx: usize) -> Result<(), String> {
        let fid = self.fibers[idx].id;
        let name = self.fibers[idx].row.key().to_string();
        let siblings = self.sibling_configs();
        self.fibers[idx].state = FiberState::Loading;
        let ctx = Ctx::new(fid, name, self.registry.clone()).with_siblings(siblings);
        let f = &self.fibers[idx];
        // catch_unwind：panic 的 apply 不能穿过 settle()——那会把已登记的 Effects 连同
        // ctx 一起漏掉。panic 走和 Err 完全一样的 fail-closed 路径。
        let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f.plugin.apply(&ctx, &f.row.config)
        }));
        let result = match applied {
            Ok(r) => r.map_err(|e| e.to_string()),
            Err(payload) => Err(format!("plugin panicked: {}", panic_message(&*payload))),
        }
        .and_then(|_| {
            // fail-closed：声明 provide 的服务必须真的由本 Fiber 提供
            for p in &f.manifest.provide {
                if self.registry.provider_of(p) != Some(fid) {
                    return Err(format!(
                        "declared provide `{p}` was not provided during apply"
                    ));
                }
            }
            Ok(())
        });
        let f = &mut self.fibers[idx];
        match result {
            Ok(()) => {
                f.ctx = Some(ctx);
                f.state = FiberState::Active;
                f.last_error = None;
                Ok(())
            }
            Err(e) => {
                ctx.take_effects_and_kill().dispose();
                f.state = FiberState::Failed;
                f.last_error = Some(e.clone());
                Err(e)
            }
        }
    }

    pub fn settle(&mut self) -> SettleReport {
        let mut report = SettleReport::default();
        loop {
            let mut changed = false;
            for i in 0..self.fibers.len() {
                if self.fibers[i].state == FiberState::Active
                    && !self.missing_injects(&self.fibers[i]).is_empty()
                {
                    self.deactivate(i);
                    report
                        .deactivated
                        .push(self.fibers[i].row.key().to_string());
                    changed = true;
                }
            }
            for i in 0..self.fibers.len() {
                if self.fibers[i].state == FiberState::Pending
                    && self.missing_injects(&self.fibers[i]).is_empty()
                {
                    let key = self.fibers[i].row.key().to_string();
                    match self.activate(i) {
                        Ok(()) => report.activated.push(key),
                        Err(e) => report.failed.push((key, e)),
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for f in &self.fibers {
            if f.state == FiberState::Pending {
                report
                    .pending
                    .push((f.row.key().to_string(), self.missing_injects(f)));
            }
        }
        report
    }

    pub fn dispose(&mut self, id: FiberId) {
        if let Some(i) = self.fibers.iter().position(|f| f.id == id) {
            if self.fibers[i].state == FiberState::Active {
                self.deactivate(i);
            }
            self.fibers[i].state = FiberState::Disposed;
        }
    }

    /// 一棵树上最多留多少条 `Disposed` 的行。
    ///
    /// `dispose` 只把状态改成终态，行本身留着——这是**故意**的：刚被撤掉的那个 Fiber 还
    /// 要出现在 `plugin/state` / `robot/health` 里，一个"消失了"的插件和一个"从没存在过"
    /// 的插件对控制台不是一回事。但留着的代价随时间线性增长，而跨进程插件把"加一行"
    /// 变成了一个**客户端**能随意重复的动作（`plugin/register` → 断连 → 再来）：每一轮
    /// 留下一行（连同客户端给的 config 与 manifest），而 `settle()`/`states_detailed()`
    /// 每次都要走一遍整个 `Vec`——于是第 N 轮的代价是 Θ(N)，一个反复重连的适配器就是
    /// Θ(N²)，全落在同时还要排 500 Hz 帧 ingest 的那条内核任务上。
    ///
    /// 32 是"最近撤掉的那些还看得见"和"不无限增长"之间的线。
    pub const MAX_DISPOSED_ROWS: usize = 32;

    /// 只保留最近 [`FiberTree::MAX_DISPOSED_ROWS`] 条 `Disposed` 的行，其余丢弃。
    ///
    /// 调用时机（`Kernel::run` 的 `Op::DisposeFiber` 分支）必须在 `settle()` **和**
    /// `publish_states()` 之后：正在退休的那一行的 `plugin/state{disposed}` 得先发出去，
    /// 行才能丢——否则它的终局事件就此消失，客户端等的正是那一条。
    ///
    /// 代价说清楚：被剪掉的 key 上，[`FiberTree::state_of`] 和 [`FiberTree::last_error`]
    /// 从 `Some(Disposed)` 变成 `None`（"从没有过"和"很久以前撤掉了"从树上看不出区别
    /// 了）。边界那一侧不受影响——`robot/health` 读的是路由自己维护的 `FiberStates`
    /// 快照，不是这棵树。
    ///
    /// 保持 `Vec` 的相对顺序（`dispose_all` 逆序遍历它），不动 `next_id`：id 必须始终
    /// 单调，一个被剪掉的 id 绝不能被重新发出去。
    pub fn prune_disposed(&mut self) {
        let mut disposed: Vec<FiberId> = self
            .fibers
            .iter()
            .filter(|f| f.state == FiberState::Disposed)
            .map(|f| f.id)
            .collect();
        if disposed.len() <= Self::MAX_DISPOSED_ROWS {
            return;
        }
        // id 单调递增，但排一次不依赖"`Vec` 一定按 id 排好"这个（今天成立的）巧合。
        disposed.sort_unstable();
        let keep_from = disposed[disposed.len() - Self::MAX_DISPOSED_ROWS];
        self.fibers
            .retain(|f| f.state != FiberState::Disposed || f.id >= keep_from);
    }

    /// 树上现有的全部 Fiber id（含 `Disposed` 的）。`Kernel` 用它把自己那张
    /// "上次广播过的状态" 表裁到同一个集合上。
    pub fn ids(&self) -> Vec<FiberId> {
        self.fibers.iter().map(|f| f.id).collect()
    }

    pub fn dispose_all(&mut self) {
        for i in (0..self.fibers.len()).rev() {
            if self.fibers[i].state == FiberState::Active {
                self.deactivate(i);
            }
            self.fibers[i].state = FiberState::Disposed;
        }
    }

    /// 同名多行时返回最后一个未 Disposed 的（重新 add 的 provider 会覆盖旧的 Disposed 行）。
    pub fn state_of(&self, key: &str) -> Option<FiberState> {
        let live = self
            .fibers
            .iter()
            .rev()
            .find(|f| f.row.key() == key && f.state != FiberState::Disposed);
        live.or_else(|| self.fibers.iter().rev().find(|f| f.row.key() == key))
            .map(|f| f.state)
    }

    pub fn last_error(&self, key: &str) -> Option<String> {
        self.fibers
            .iter()
            .rev()
            .find(|f| f.row.key() == key)
            .and_then(|f| f.last_error.clone())
    }

    pub fn states(&self) -> Vec<(String, FiberState)> {
        self.fibers
            .iter()
            .map(|f| (f.row.key().to_string(), f.state))
            .collect()
    }

    /// 同 `states()`，但带上 id 与出错原因——`kernel` 服务用它来广播 `plugin/state`。
    pub fn states_detailed(&self) -> Vec<FiberInfo> {
        self.fibers
            .iter()
            .map(|f| FiberInfo {
                id: f.id,
                key: f.row.key().to_string(),
                state: f.state,
                error: f.last_error.clone(),
            })
            .collect()
    }
}

/// `FiberTree::states_detailed` 的一行。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiberInfo {
    pub id: FiberId,
    pub key: String,
    pub state: FiberState,
    pub error: Option<String>,
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

pub fn describe_error(e: &PluginError) -> String {
    e.to_string()
}
