//! `brain-native` 插件：把一个 [`NativeBrain`] 挂到内核的 `brains` 服务上。
//!
//! 配置：
//!
//! ```yaml
//! - id: brain-native
//!   config:
//!     brain_id: fast
//!     wake: wake/fast.yaml            # 相对进程 cwd，和 mock-brain 一样
//!     persona: "你是一台四足机器人的快脑。"
//!     model: { scripted: tests/scripts/fast.yaml }   # 或 { provider: openai, model: gpt-5 }
//!     providers: providers/default.yaml              # 相对进程 cwd；只有 provider 模型用得上
//!     fallback: [{ provider: anthropic, model: claude-sonnet-4 }]   # 可选：降级链
//!     retry: { max_attempts: 3, base_ms: 250, cap_ms: 4000 }        # 可选
//!     max_steps: 8
//!     max_tokens: 2048
//!     temperature: 0.2                # 可省；省了（或写 null）就不发这个字段
//!     recall_root: ./memory           # 可选：默认 = RSI store 的根
//!     step_timeout_ms: 60000
//!     oeh_home: ./run/oeh-home        # 可选：默认 $OEH_HOME，再默认 ./run/oeh-home
//!     robot_id: g1                    # RSI store 在 <oeh_home>/robots/<robot_id>
//!     protected_paths: [./bundles]    # 可选：相对 profile 文件所在目录；默认按
//!                                     #       $OEH_PROFILE + 进程 cwd 推
//!     review: { every_turns: 10, on_session_end: true, max_steps: 4 }
//! ```
//!
//! **RSI 知识存储无条件打开**（`<oeh_home>/robots/<robot_id>`），哪怕两个 review 触发
//! 器都关着：`# Skills` 段（system 前缀）和 `recall_root` 的默认值都从它来，而"这台机器
//! 人的 skill 和记忆在哪"不该随一个触发器的开关变化。
//!
//! **受保护路径为空 = 激活失败**（fail-closed）。空集合的 `covers` 只会老老实实返回
//! false，也就是护栏一声不响地失效；一个能改自己 skill 的脑配上一组空护栏，比一个起不
//! 来的脑危险得多。
//!
//! **模型在 `apply` 里就造好。** 不认识的 provider、不认识的 `cache`、没开 `tls` 却配了
//! `https://`——都是这一行插件激活失败（错误点名是哪一个），而不是 attach 成功之后每次
//! 唤醒记一条 request-error：一行配错的声明，操作员该看到病因不是症状。
//!
//! **缺 API key 不是配置错**：attach 照常成功，第一次真要发请求时传输还 `MissingKey`，
//! 瀑布不重试、直接换下一个 runtime；整条链都没 key 才记 `brain/request-error` 并
//! `done("error: missing api key …")`（计划的全局约束）。
//!
//! **能力面按 turn 对账。** 工具表是 `Admission::capabilities().list()` 的一份投影，每个
//! turn 开始时若 `CapabilityRegistry::generation()` 变了就重新投影，一个 turn 之内不再
//! 变（模型不能在半个 turn 里看到工具消失）。进程外适配器晚于脑注册的能力，下一个 turn
//! 就能用；system 前缀只在工具表真变了时才变。行序仍是一条纪律：执行器那几行排在
//! `brain-native` 前面，第一个 turn 就有工具；脑这一行永远放在 bundle 的最后。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use oeh_admission::admission::Admission;
use oeh_kernel::factories::PluginFactory;
use oeh_kernel::wake::WakePolicy;
use oeh_kernel::world_log::WorldLog;
use oeh_kernel::{Brains, Ctx, Plugin, PluginError};
use oeh_proto::Manifest;
use serde::Deserialize;
use serde_json::Value;

use crate::builtin::scripted_model::ScriptedModel;
use crate::model::Model;
use crate::native::{NativeBrain, NativeConfig, ReviewConfig};
use crate::providers::{build_model_with_log, FallbackChain, Providers};
use crate::retry::RetryPolicy;
use crate::rsi::{KnowledgeStore, ProtectedPaths};
use crate::tools::ToolRegistry;

const ID: &str = "brain-native";

/// 模型声明：`{ scripted: <路径> }` 或 `{ provider: <名字>, model: <id> }`。
///
/// 写成一个两者皆可选的结构而不是 untagged 枚举：untagged 解不出来时 serde 只会说
/// "data did not match any variant"，而这里最常见的错法（`provider` 写了、`model` 忘了）
/// 值得一句点名的错误。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelCfg {
    #[serde(default)]
    scripted: Option<PathBuf>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

/// fallback 链里的一站。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRef {
    provider: String,
    model: String,
}

/// 退避参数。整块可省；省了就是规格的默认值。
#[derive(Deserialize)]
#[serde(deny_unknown_fields, default)]
struct RetryCfg {
    max_attempts: u32,
    base_ms: u64,
    cap_ms: u64,
}

impl Default for RetryCfg {
    fn default() -> Self {
        let p = RetryPolicy::default();
        RetryCfg {
            max_attempts: p.max_attempts,
            base_ms: p.base_ms,
            cap_ms: p.cap_ms,
        }
    }
}

impl From<RetryCfg> for RetryPolicy {
    fn from(c: RetryCfg) -> RetryPolicy {
        RetryPolicy {
            max_attempts: c.max_attempts,
            base_ms: c.base_ms,
            cap_ms: c.cap_ms,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    brain_id: String,
    #[serde(default)]
    isolated_camera_turns: bool,
    #[serde(default)]
    record_policy_requests: bool,
    #[serde(default)]
    require_complete_observation: bool,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
    /// YAML 唤醒策略文件，相对进程 cwd（和 `mock-brain` 同一条规则）。
    wake: PathBuf,
    #[serde(default)]
    persona: String,
    model: ModelCfg,
    /// 厂商声明，相对进程 cwd。只有 provider 模型或 `fallback` 非空时才会去读它——
    /// 一个纯剧本的脑不该因为仓库里没有这个文件就起不来。
    #[serde(default = "d_providers")]
    providers: PathBuf,
    /// 主 runtime 失败之后依次尝试的降级链。
    #[serde(default)]
    fallback: Vec<ModelRef>,
    #[serde(default)]
    retry: RetryCfg,
    #[serde(default = "d_max_steps")]
    max_steps: u32,
    #[serde(default = "d_max_tokens")]
    max_tokens: u32,
    /// 省掉或写 `null` = 不发 `temperature`，用厂商自己的默认值。**没有默认数**：
    /// OpenAI 的推理模型对任何 `temperature` 都回 400，而 realtime 只接受 0.6–1.2——
    /// 一个跨 wire 通用的默认值并不存在。
    #[serde(default)]
    temperature: Option<f32>,
    /// 省略 = 用 RSI store 的根（`<oeh_home>/robots/<robot_id>`）。那正是这个脑的
    /// skill 和记忆所在，`recall_*` 本来就是用来翻它们的。
    #[serde(default)]
    recall_root: Option<PathBuf>,
    #[serde(default = "d_step_timeout_ms")]
    step_timeout_ms: u64,
    /// RSI 的家。省略时看 `$OEH_HOME`，再省略就是 `./run/oeh-home`（相对进程 cwd，和
    /// `wake` / `providers` 同一条规则）。
    #[serde(default)]
    oeh_home: Option<PathBuf>,
    /// 这台机器人的 id：store 在 `<oeh_home>/robots/<robot_id>`。
    #[serde(default = "d_robot_id")]
    robot_id: String,
    /// RSI 永远写不进去的路径前缀。
    ///
    /// 相对路径相对 **profile 文件所在目录**解析（见 [`profile_base_dir`]），不是进程
    /// cwd：这张表是 profile 说的话，不该随启动它的那个 shell 站在哪里变。
    ///
    /// **省略**（不是空表）= 按 `$OEH_PROFILE` + 进程 cwd 推（见
    /// [`ProtectedPaths::from_profile`]）。写成 `[]` 是另一件事：那是明确地说"什么都不
    /// 保护"，而那正是这个字段存在要挡的东西，所以它和"推出来还是空"一样直接拒绝激活。
    /// 用 `Option` 而不是 `Vec::is_empty()` 判断，是因为后者把"没写"和"写了空表"读成同
    /// 一件事——一个想关掉护栏的人会得到一套护栏，一个忘了配的人也是，两种都没有诊断。
    #[serde(default)]
    protected_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    review: ReviewCfg,
}

/// review 的触发与预算。整块可省。
#[derive(Deserialize)]
#[serde(deny_unknown_fields, default)]
struct ReviewCfg {
    every_turns: u32,
    on_session_end: bool,
    max_steps: u32,
    /// 整段 review 的墙上时限。**0 是配置错误**，不是"不设限"——见
    /// [`ReviewConfig::timeout_ms`]。
    timeout_ms: u64,
}

impl Default for ReviewCfg {
    fn default() -> Self {
        let d = ReviewConfig::default();
        ReviewCfg {
            every_turns: d.every_turns,
            on_session_end: d.on_session_end,
            max_steps: d.max_steps,
            timeout_ms: d.timeout_ms,
        }
    }
}

impl From<ReviewCfg> for ReviewConfig {
    fn from(c: ReviewCfg) -> ReviewConfig {
        ReviewConfig {
            every_turns: c.every_turns,
            on_session_end: c.on_session_end,
            max_steps: c.max_steps,
            timeout_ms: c.timeout_ms,
        }
    }
}

fn d_robot_id() -> String {
    "g1".into()
}

/// RSI store 的根：`<oeh_home>/robots/<robot_id>`。
fn store_root(home: Option<&PathBuf>, robot_id: &str) -> PathBuf {
    let home = home.cloned().unwrap_or_else(|| {
        std::env::var_os("OEH_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("./run/oeh-home"))
    });
    home.join("robots").join(robot_id)
}

/// `robot_id` 会被拼进路径，所以它得是一段普通的文件名。
///
/// 这里不是在防模型（模型碰不到配置），是在防一次打错的 profile：一个
/// `robot_id: ../../etc` 会让 store 的根落在仓库外面，而 store 自己的路径检查全部以
/// "根是对的"为前提。
fn valid_robot_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
        && id != "."
        && id != ".."
}

/// 默认的受保护集合：`$OEH_PROFILE`（有就用）+ 进程 cwd 下的仓库布局。
///
/// `$OEH_PROFILE` 没设时拿 `<cwd>/profiles` 顶上那个位置——`from_profile` 本来就会
/// 无条件加 `profiles/`，所以这只是一次无害的重复，而不是"少保护了一个目录"。
/// 空护栏集合的拒绝理由。显式 `protected_paths: []` 和"推出来是空的"共用这一条：对操
/// 作员来说它们是同一个事实（这个 store 没有任何写不进去的地方），而两条措辞不同的错
/// 误只会让人以为自己遇到的是另一个问题。
const EMPTY_PROTECTED: &str =
    "refusing to open the RSI knowledge store with an empty protected-path set: \
     set `protected_paths` to a non-empty list, or omit it and run from the repo root \
     so the default (profile / bundles / patch / wake / profiles) can be derived";

/// `review.timeout_ms: 0` 的拒绝理由。
const EMPTY_REVIEW_TIMEOUT: &str =
    "review.timeout_ms must be > 0: the maintenance phase is exempt from the kernel's \
     turn_timeout_ms sweep, so an unbounded review can wedge the brain in Maintenance \
     with no turn_end and no way back except a directed interrupt";

/// 护栏集合：显式配置优先，省略则推。**两种来源里的空都是错误**——这个函数是那条判断
/// 唯一的所在地，所以 `protected_paths: []` 和"推出来是空的"走的是同一行代码，不可能
/// 有一种被漏掉。
///
/// `derived` 是个闭包而不是直接调 [`default_protected`]：后者读进程 cwd 和
/// `$OEH_PROFILE`，在今天的仓库里永远推得出非空的一组，于是"推出来是空的"那条分支就
/// 没有任何办法被测到——而它恰恰是 fail-closed 的那一半。
fn resolve_protected(
    configured: Option<&Vec<PathBuf>>,
    base: &Path,
    derived: impl FnOnce() -> ProtectedPaths,
) -> Result<ProtectedPaths, String> {
    let p = match configured {
        Some(v) => ProtectedPaths::new(v.iter().map(|p| base.join(p))),
        None => derived(),
    };
    if p.is_empty() {
        return Err(EMPTY_PROTECTED.into());
    }
    Ok(p)
}

/// 配置里的**相对** `protected_paths` 相对谁解析：这份 profile 文件所在的目录
/// （`$OEH_PROFILE` 的父目录），没有就退回进程 cwd。
///
/// 为什么不是 cwd：这张表是"这台机器人**哪些文件不许自己改**"，而那是 profile 说了算
/// 的事，不是启动它的那个 shell 当时站在哪里说了算的事。按 cwd 解析的话，
/// `systemctl` 起的内核（cwd `/`）和手敲起来的内核（cwd 是仓库根）拿到的是两套护栏，
/// 而前者那一套一个目录都不存在——护栏一声不响地失效，症状要等到一次 RSI 写入之后才
/// 出现。bundle 里那句"一台机器人的 RSI 护栏不该随一次 `cd` 变化"从这里开始才是真的。
///
/// `base.join(p)` 对**绝对**路径是原样返回（`Path::join` 的语义），所以写绝对路径的
/// 配置（测试、以及任何想把话说死的部署）一个字都不用改。
fn profile_base_dir() -> PathBuf {
    profile_base_dir_from(std::env::var_os("OEH_PROFILE").map(PathBuf::from))
}

/// [`profile_base_dir`] 的纯函数内核（环境变量读一次，判断在这里测）。
fn profile_base_dir_from(profile: Option<PathBuf>) -> PathBuf {
    let cwd = || std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match profile {
        // `$OEH_PROFILE` 指的是 profile **文件**（`oeh run --profile` 就是这么设的），
        // 所以基准是它的父目录。父目录取不出来（一个光秃秃的文件名）时退回 cwd：那两
        // 者本来就是同一个目录。
        Some(p) => match p.parent() {
            Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
            _ => cwd(),
        },
        None => cwd(),
    }
}

fn default_protected() -> ProtectedPaths {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let profile = std::env::var_os("OEH_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| cwd.join("profiles"));
    ProtectedPaths::from_profile(&profile, &cwd)
}
fn d_providers() -> PathBuf {
    PathBuf::from("providers/default.yaml")
}
fn d_max_steps() -> u32 {
    8
}
fn d_max_tokens() -> u32 {
    2048
}
fn d_step_timeout_ms() -> u64 {
    60_000
}

pub struct NativeBrainPlugin;

impl Plugin for NativeBrainPlugin {
    fn manifest(&self) -> Manifest {
        Manifest::new(ID).inject(&["brains", "admission", "world-log"])
    }

    fn apply(&self, ctx: &Ctx, config: &Value) -> Result<(), PluginError> {
        let cfg: Config = ctx.parse_config(config)?;
        if cfg.isolated_camera_turns
            && (cfg.max_steps != 1
                || cfg.review.every_turns != 0
                || cfg.review.on_session_end
                || cfg.allowed_tools.as_ref().is_none_or(|v| v.is_empty()))
        {
            return Err(PluginError::Config {
                plugin: ID.into(),
                reason: "isolated camera turns require one step, disabled review and an explicit tool allowlist".into(),
            });
        }
        let brains = ctx.get::<Brains>("brains")?;
        let adm = ctx.get::<Admission>("admission")?;
        let log = ctx.get::<WorldLog>("world-log")?;
        let bad = |reason: String| PluginError::Config {
            plugin: ID.into(),
            reason,
        };

        let text = std::fs::read_to_string(&cfg.wake)
            .map_err(|e| bad(format!("wake policy {}: {e}", cfg.wake.display())))?;
        let policy = WakePolicy::from_yaml(&text)
            .map_err(|e| bad(format!("wake policy {}: {e}", cfg.wake.display())))?;

        // 厂商声明**按需**读：一个纯剧本的脑（dev-native profile、全部 e2e 测试）不该
        // 因为进程 cwd 下没有 providers/default.yaml 就起不来。
        let needs_providers = cfg.model.provider.is_some() || !cfg.fallback.is_empty();
        let declared = if needs_providers {
            Some(Providers::from_path(&cfg.providers).map_err(bad)?)
        } else {
            None
        };
        // 脚本模型额外留一份具体类型的句柄：它要挂进服务注册表供测试断言（见下）。
        let (model, scripted): (Arc<dyn Model>, Option<Arc<ScriptedModel>>) =
            // `declared` 与 `provider` 是同一个条件算出来的（上面那两行），所以 provider
            // 分支里它一定是 `Some`。把它写进**模式**而不是在分支里 `expect`：那个不变式
            // 隔着十几行，让编译器守着比让读代码的人守着便宜。
            match (
                &cfg.model.scripted,
                &cfg.model.provider,
                &cfg.model.model,
                &declared,
            ) {
                (Some(path), None, None, _) => {
                    let m = Arc::new(ScriptedModel::from_path(path).map_err(bad)?);
                    (m.clone(), Some(m))
                }
                (None, Some(provider), Some(model), Some(p)) => {
                    // 超时就是 step 的超时：对一个 turn 来说"服务器不回话"和"流读不完"
                    // 是同一件事。
                    let m = build_model_with_log(
                        p,
                        provider,
                        model,
                        cfg.step_timeout_ms,
                        Some(log.clone()),
                    )
                    .map_err(bad)?;
                    (m, None)
                }
                _ => {
                    return Err(bad(
                        "model needs exactly one of `{scripted: <path>}` or `{provider: <name>, model: <id>}`"
                            .into(),
                    ))
                }
            };

        // 降级链也在这里解析：第二顺位配错了要现在就知道，而不是等到主 provider 真的
        // 挂掉、最需要它的那一刻。
        let fallbacks = match &declared {
            Some(p) if !cfg.fallback.is_empty() => {
                let pairs: Vec<(String, String)> = cfg
                    .fallback
                    .iter()
                    .map(|r| (r.provider.clone(), r.model.clone()))
                    .collect();
                FallbackChain::resolve(p, &pairs, cfg.step_timeout_ms)
                    .map_err(bad)?
                    .models(Some(log.clone()))
                    .map_err(bad)?
            }
            _ => Vec::new(),
        };

        // RSI 知识存储。护栏先算再开店：一组空护栏比开不起来的店危险得多（见模块头）。
        if !valid_robot_id(&cfg.robot_id) {
            return Err(bad(format!(
                "robot_id {:?} must be one plain path segment of [A-Za-z0-9._-]",
                cfg.robot_id
            )));
        }
        // `review.timeout_ms: 0` 先挡下来。维护相位不在内核 `turn_timeout_ms` 的扫描范
        // 围里，所以"不设限"在这里的真实含义是"一次卡住的模型调用可以永久冻住这个脑"。
        if cfg.review.timeout_ms == 0 {
            return Err(bad(EMPTY_REVIEW_TIMEOUT.into()));
        }
        let protected = resolve_protected(
            cfg.protected_paths.as_ref(),
            &profile_base_dir(),
            default_protected,
        )
        .map_err(bad)?;
        let root = store_root(cfg.oeh_home.as_ref(), &cfg.robot_id);
        let store = Arc::new(
            KnowledgeStore::open(&root, protected, Some(log.clone()))
                .map_err(|e| bad(format!("rsi knowledge store {}: {e}", root.display())))?,
        );

        // **底表**：能力面那一半不在这里投。`NativeBrain` 自己攥着注册表，在每个 turn
        // 开头按 `CapabilityRegistry::generation` 对一次账（见 `with_knowledge`），所以
        // 一个晚到的适配器注册的 `walk` 会出现在下一个 turn 的工具表里，而不是永远看
        // 不见。这也把 bundle 的行序从"脑必须排最后"降级成一条建议：排在前面只影响
        // **第一个** turn 之前的那一份投影。
        let mut tools = ToolRegistry::new();
        // `recall_*` 默认指向 store 的根：那就是这个脑的 skill 和记忆所在。review 那张
        // 表**不含** `recall_*`（见 [`crate::native::review`]），两边互不影响。
        let recall_root = cfg
            .recall_root
            .clone()
            .unwrap_or_else(|| store.root().to_path_buf());
        tools.add_recall(recall_root.clone());
        // 脑间提问：无条件注册，不看能力面里有没有 `ask_slow_brain`。它是每个脑都该有的
        // 一件事（"问问别人"），而能力面是一份快照——按快照决定给不给这个工具，等于让
        // bundle 的行序悄悄决定一个脑会不会提问。能力没挂上时提案被准入挡下，模型收到
        // 一条 `ok:false`，这比一张时有时无的工具表好排查。
        tools.add_ask_slow_brain();

        let sink = NativeBrain::with_knowledge(
            NativeConfig {
                brain_id: cfg.brain_id.clone(),
                isolated_camera_turns: cfg.isolated_camera_turns,
                record_policy_requests: cfg.record_policy_requests,
                require_complete_observation: cfg.require_complete_observation,
                allowed_tools: cfg.allowed_tools,
                persona: cfg.persona,
                max_steps: cfg.max_steps,
                max_tokens: cfg.max_tokens,
                temperature: cfg.temperature,
                recall_root: Some(recall_root),
                review: cfg.review.into(),
                step_timeout_ms: cfg.step_timeout_ms,
            },
            model,
            fallbacks,
            cfg.retry.into(),
            tools,
            log,
            Some(store),
            Some(adm.capabilities().clone()),
        );
        // `Weak`，不是 `Arc`：这个闭包活在 fiber 的 Effects 里，攥一个强引用就意味着
        // detach（`Attached` 被摘掉）之后 sink 还活着，`Drop` 那一下也就不会在 detach
        // 的那一刻发生。
        let weak = Arc::downgrade(&sink);
        let disposer = brains
            .attach(&cfg.brain_id, policy, sink)
            .map_err(|e| PluginError::Other(e.to_string()))?;
        ctx.on_dispose(disposer);
        // 注册在 attach 的 disposer **之后** = 撤销时**先**跑（Effects 逆序 dispose）：
        // 先把还在跑的 turn 任务掐掉，再 detach。反过来的话 `brain/detach` 与 abort 之间
        // 会有一段窗口，那个已经没人认领的 turn 还能往世界日志里写它的 `brain/*`。
        // `NativeBrain` 的 `Drop` 是同一件事的兜底（直接 detach、profile 重载都走它）。
        ctx.on_dispose(Box::new(move || {
            if let Some(s) = weak.upgrade() {
                s.abort_current();
            }
        }));

        // 脚本模型按 `brain-native.<brain_id>.model` 挂进服务注册表：e2e 测试要能断言
        // "这一步的 user message 里有什么"，而 prompt 全文按纪律不进世界日志。名字带
        // brain_id，同一个进程里两个脚本脑不会撞车；撤销随 fiber 走，没有全局状态。
        if let Some(m) = scripted {
            ctx.provide(&format!("{ID}.{}.model", cfg.brain_id), m)?;
        }
        Ok(())
    }
}

inventory::submit! { PluginFactory { id: ID, build: || Box::new(NativeBrainPlugin) } }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value) -> Result<Config, serde_json::Error> {
        serde_json::from_value(v)
    }

    #[test]
    fn the_documented_defaults_are_the_brief_s_numbers() {
        let c = parse(json!({
            "brain_id": "fast",
            "wake": "wake/fast.yaml",
            "model": {"scripted": "s.yaml"}
        }))
        .unwrap();
        assert_eq!(c.max_steps, 8);
        assert_eq!(c.step_timeout_ms, 60_000);
        assert_eq!(c.max_tokens, 2048);
        assert_eq!(c.persona, "");
        assert!(!c.record_policy_requests);
        assert!(!c.require_complete_observation);
        // I2：`temperature` **没有**默认数——省掉就是不发这个字段。
        assert!(c.temperature.is_none());
        assert!(c.recall_root.is_none());
        assert_eq!(c.providers, PathBuf::from("providers/default.yaml"));
        assert!(c.fallback.is_empty());
        assert_eq!(RetryPolicy::from(c.retry), RetryPolicy::default());
        // RSI：默认 10 个 turn 一次，会话末补一次，一次最多 4 步。
        assert_eq!(ReviewConfig::from(c.review), ReviewConfig::default());
        assert_eq!(c.robot_id, "g1");
        assert!(c.oeh_home.is_none());
        assert!(c.protected_paths.is_none());
    }

    /// `oeh_home` 省略时看 `$OEH_HOME`，再省略就是 `./run/oeh-home`。**不读环境变量**
    /// 的那半边在这里直接验（显式配置永远优先）。
    #[test]
    fn the_store_root_is_home_robots_id() {
        assert_eq!(
            store_root(Some(&PathBuf::from("/srv/oeh")), "g1"),
            PathBuf::from("/srv/oeh/robots/g1")
        );
    }

    /// `robot_id` 被拼进路径，所以一次打错的 profile 不能把 store 的根挪到仓库外面。
    #[test]
    fn a_robot_id_that_is_not_one_plain_segment_is_rejected() {
        assert!(valid_robot_id("g1"));
        assert!(valid_robot_id("g1-lab_2"));
        for bad in ["", ".", "..", "a/b", "../etc", "a b"] {
            assert!(!valid_robot_id(bad), "{bad:?} 本该被拒");
        }
    }

    /// 默认的受保护集合永远非空（`bundles` / `patch` / `wake` / `profiles` 是无条件
    /// 加的），所以显式配置那一半才是 fail-closed 平时真正会挡到的东西。
    #[test]
    fn the_default_protected_set_is_never_empty() {
        assert!(!default_protected().is_empty());
        assert!(resolve_protected(None, Path::new("."), default_protected).is_ok());
    }

    /// **两种**空护栏都拒绝激活，而且拒绝理由逐字相同：显式写 `protected_paths: []`
    /// （"我不要护栏"），以及推导出来是空的（跑在一个什么都推不出来的 cwd 里）。
    ///
    /// 显式的那一条是真正的回归点：从前 `Vec::is_empty()` 把"没写"和"写了空表"读成同
    /// 一件事，于是一个明确要求关掉护栏的人会**悄悄**拿到一整套默认护栏——既没有得到
    /// 他要的东西，也没有收到任何诊断。
    #[test]
    fn an_empty_protected_path_set_is_refused_whichever_way_it_got_empty() {
        let explicit =
            resolve_protected(Some(&Vec::new()), Path::new("."), default_protected).unwrap_err();
        let derived = resolve_protected(None, Path::new("."), ProtectedPaths::default).unwrap_err();
        assert_eq!(explicit, derived);
        assert_eq!(explicit, EMPTY_PROTECTED);
        assert!(explicit.contains("empty protected-path set"), "{explicit}");
        // 而一份非空的显式配置照常通过（`[]` 被拒不是"显式配置一律被拒"）。
        assert!(resolve_protected(
            Some(&vec![PathBuf::from("/srv/guarded")]),
            Path::new("."),
            ProtectedPaths::default
        )
        .is_ok());
    }

    /// M4：一份从**别处**加载的 profile，它的相对 `protected_paths` 落在 profile 自己
    /// 的目录下，而不是进程 cwd 下。
    ///
    /// 这正是回归点：从前两者是同一件事（`ProtectedPaths::new` → `canonicalize` 按 cwd
    /// 解析），于是 bundle 里那句"一台机器人的 RSI 护栏不该随一次 `cd` 变化"是一句假话
    /// ——`systemctl` 起的内核（cwd `/`）会拿到一套一个目录都不存在的护栏。
    ///
    /// 测试进程的 cwd 是 crate 目录（不是这个 tempdir），所以"解析到了 profile 那边"
    /// 和"解析到了 cwd 这边"两种结果在这里是明确分得开的。
    #[test]
    fn a_relative_protected_path_follows_the_profile_not_the_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let deploy = tmp.path().canonicalize().unwrap();
        let profile = deploy.join("profiles/g1-sim.yaml");
        std::fs::create_dir_all(deploy.join("profiles")).unwrap();
        std::fs::create_dir_all(deploy.join("bundles")).unwrap();
        std::fs::write(&profile, "bundles: []\n").unwrap();

        // `$OEH_PROFILE` 指向那份 profile 文件 → 基准是它的目录。
        let base = profile_base_dir_from(Some(profile.clone()));
        assert_eq!(base, deploy.join("profiles"));

        let p = resolve_protected(
            Some(&vec![PathBuf::from("../bundles"), PathBuf::from(".")]),
            &base,
            ProtectedPaths::default,
        )
        .unwrap();
        assert!(p.covers(&deploy.join("bundles/unitree-sim.yaml")));
        assert!(p.covers(&profile));
        // 而 cwd 底下那个同名目录**没有**被保护：这一条就是"没跟着 cwd 走"的判据。
        let cwd = std::env::current_dir().unwrap();
        assert_ne!(cwd, deploy.join("profiles"));
        assert!(!p.covers(&cwd.join("bundles/unitree-sim.yaml")));

        // 绝对路径不受影响（`Path::join` 对绝对路径原样返回）——测试与任何想把话说死的
        // 部署一个字都不用改。
        let abs = resolve_protected(
            Some(&vec![deploy.join("bundles")]),
            Path::new("/nowhere"),
            ProtectedPaths::default,
        )
        .unwrap();
        assert!(abs.covers(&deploy.join("bundles/base.yaml")));

        // 没有 `$OEH_PROFILE` 就是老行为：进程 cwd。
        assert_eq!(profile_base_dir_from(None), cwd);
    }

    /// `review.timeout_ms` 有默认数，而 `0` 是配置错误——维护相位不在内核
    /// `turn_timeout_ms` 的扫描范围里，所以"不设限"= 一次卡住的模型调用可以永久冻住
    /// 这个脑。
    #[test]
    fn a_review_timeout_of_zero_is_a_config_error() {
        assert_eq!(ReviewConfig::default().timeout_ms, 30_000);
        let c = parse(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"scripted": "s.yaml"},
            "review": {"timeout_ms": 0}
        }))
        .unwrap();
        // 解析得出来（0 是个合法的 u64），激活时才拒——拒绝理由说清楚为什么没有
        // "不设限"这个选项。
        assert_eq!(c.review.timeout_ms, 0);
        assert!(EMPTY_REVIEW_TIMEOUT.contains("turn_timeout_ms"));
        let c = parse(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"scripted": "s.yaml"},
            "review": {"timeout_ms": 1500}
        }))
        .unwrap();
        assert_eq!(ReviewConfig::from(c.review).timeout_ms, 1500);
    }

    /// review 块里没写的字段还是默认值。
    #[test]
    fn a_partial_review_block_keeps_the_other_defaults() {
        let c = parse(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"scripted": "s.yaml"},
            "review": {"every_turns": 0}
        }))
        .unwrap();
        let r = ReviewConfig::from(c.review);
        assert_eq!(r.every_turns, 0);
        assert!(r.on_session_end);
        assert_eq!(r.max_steps, 4);
        assert_eq!(r.timeout_ms, 30_000);
    }

    #[test]
    fn a_typo_in_the_review_block_is_rejected() {
        let err = parse_err(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"scripted": "s.yaml"},
            "review": {"every_turn": 2}
        }));
        assert!(err.contains("every_turn"), "{err}");
    }

    /// 写了数就是那个数，显式的 `null` 和省略同义（两种写法都是"用厂商的默认值"）。
    #[test]
    fn temperature_parses_as_a_number_or_as_an_explicit_null() {
        let with = |t: Value| {
            parse(json!({
                "brain_id": "fast",
                "wake": "w.yaml",
                "model": {"scripted": "s.yaml"},
                "temperature": t,
            }))
            .unwrap()
            .temperature
        };
        assert_eq!(with(json!(0.2)), Some(0.2));
        assert_eq!(with(Value::Null), None);
    }

    #[test]
    fn a_fallback_chain_and_a_partial_retry_block_parse() {
        let c = parse(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"provider": "openai", "model": "gpt-5"},
            "fallback": [{"provider": "anthropic", "model": "claude"}],
            "retry": {"base_ms": 10}
        }))
        .unwrap();
        assert_eq!(c.fallback.len(), 1);
        assert_eq!(c.fallback[0].provider, "anthropic");
        // 块里没写的字段还是默认值。
        let p = RetryPolicy::from(c.retry);
        assert_eq!((p.max_attempts, p.base_ms, p.cap_ms), (3, 10, 4000));
    }

    #[test]
    fn a_typo_in_the_fallback_chain_is_rejected() {
        let err = parse_err(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"scripted": "s.yaml"},
            "fallback": [{"provider": "anthropic", "modle": "claude"}]
        }));
        assert!(err.contains("modle"), "{err}");
    }

    /// `Config` 不是 `Debug`（里面没什么值得 Debug 的），所以用 `let Err(..) else`
    /// 而不是 `unwrap_err`。
    fn parse_err(v: Value) -> String {
        match parse(v) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("这份配置本该被拒"),
        }
    }

    #[test]
    fn a_typo_in_the_config_is_rejected_rather_than_ignored() {
        let err = parse_err(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"scripted": "s.yaml"},
            "max_step": 2
        }));
        assert!(err.contains("max_step"), "{err}");
    }

    #[test]
    fn a_model_block_with_a_typo_is_rejected_too() {
        let err = parse_err(json!({
            "brain_id": "fast",
            "wake": "w.yaml",
            "model": {"script": "s.yaml"}
        }));
        assert!(err.contains("script"), "{err}");
    }
}
