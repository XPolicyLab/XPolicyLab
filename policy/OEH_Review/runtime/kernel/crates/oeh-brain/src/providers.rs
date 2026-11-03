//! 厂商是**声明**，不是代码（规格 §7.2）。
//!
//! 一份 `providers.yaml` 说清楚每家的 wire（`api_mode`）、端点、从哪个环境变量拿 key、
//! 要不要打 prompt 缓存点；[`build_model`] 按这份声明挑传输、造 [`Model`]。换一家厂商
//! 是改一行 YAML，不是加一个 `match` 分支——turn 循环、工具表、重试瀑布都不认识
//! "openai" 这个词。
//!
//! ```yaml
//! providers:
//!   openai:    { api_mode: responses,          base_url: https://api.openai.com/v1, env: OPENAI_API_KEY }
//!   anthropic: { api_mode: anthropic_messages, base_url: https://api.anthropic.com,  env: ANTHROPIC_API_KEY, cache: system_and_3 }
//! ```
//!
//! **配错了的东西在 boot 的那一刻就失败**：不认识的 provider、不认识的 `cache` 模式、
//! 没开 `tls` feature 却配了 `https://`（这台开发机上 rustls 编不过，见 `Cargo.toml`）。
//! 这些都是"这行配置本身不成立"。
//!
//! **缺 API key 不在此列**（计划的全局约束）：`env` 指的环境变量没设时 [`Runtime`] 照样
//! 造得出来，`api_key` 是 `None`，attach 成功；真到了要发请求的那一刻传输还
//! [`ModelError::MissingKey`](crate::model::ModelError::MissingKey)（点名是哪个变量），
//! 瀑布不重试它、直接换下一个 runtime。key 是**运行期**的东西——换一个 key 不该要重启
//! 一次 profile，而一条链上第二顺位有 key 时第一顺位没 key 不该让整个脑起不来。

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use oeh_kernel::world_log::WorldLog;
use serde::Deserialize;

use crate::model::Model;
use crate::transport::{
    AnthropicMessagesModel, ChatCompletionsModel, ClaudeCodeExecModel, CodexExecModel,
    RealtimeModel, ResponsesModel,
};

/// 一家厂商的声明。
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDecl {
    pub api_mode: ApiMode,
    /// 省略时只有 `realtime` 有默认值（OpenAI 的 realtime 端点），别的 wire 必须写。
    #[serde(default)]
    pub base_url: Option<String>,
    /// API key 从这个环境变量读。
    pub env: String,
    /// prompt 缓存策略。目前只有 `system_and_3`（Anthropic）：system 前缀 + 最后三条
    /// user 打 `cache_control`。
    #[serde(default)]
    pub cache: Option<String>,
    /// 额外的请求头（`BTreeMap` 而不是 `HashMap`：请求体要字节级稳定，头也一样按名字排序）。
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// 只给订阅 wire（`codex_exec` / `claude_code_exec`）：CLI 可执行文件（默认 `codex` / `claude`）。
    #[serde(default)]
    pub program: Option<String>,
    /// 只给订阅 wire：推理强度（Codex 的 `model_reasoning_effort`、Claude Code 的 `--effort`），冻结在
    /// 声明里，不随 turn 变。
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// 六种 wire。`codex_exec` 与 `claude_code_exec` 不是 HTTP：每次调用起一个 CLI 子进程，用账号
/// 订阅而不是 API key——`env` 指向的变量存的是 `CODEX_HOME` 登录目录 / `claude setup-token`
/// 给的 token。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiMode {
    ChatCompletions,
    AnthropicMessages,
    Responses,
    Realtime,
    CodexExec,
    ClaudeCodeExec,
}

impl ApiMode {
    /// 进 `brain/request.api_mode` 的名字，和 YAML 里写的是同一个词。
    pub fn as_str(self) -> &'static str {
        match self {
            ApiMode::ChatCompletions => "chat_completions",
            ApiMode::AnthropicMessages => "anthropic_messages",
            ApiMode::Responses => "responses",
            ApiMode::Realtime => "realtime",
            ApiMode::CodexExec => "codex_exec",
            ApiMode::ClaudeCodeExec => "claude_code_exec",
        }
    }

    /// 起 CLI 子进程、用账号订阅的 wire（没有端点，`program` / `reasoning_effort` 只对它们有意义）。
    pub fn is_subscription(self) -> bool {
        matches!(self, ApiMode::CodexExec | ApiMode::ClaudeCodeExec)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Providers {
    pub providers: BTreeMap<String, ProviderDecl>,
}

/// 唯一认识的缓存策略。
const CACHE_SYSTEM_AND_3: &str = "system_and_3";

impl Providers {
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        serde_yaml_ng::from_str(text).map_err(|e| e.to_string())
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("providers {}: {e}", path.display()))?;
        Self::from_yaml(&text).map_err(|e| format!("providers {}: {e}", path.display()))
    }

    /// 把一条声明 + 一个 model id 解析成一个可用的 [`Runtime`]，顺带把"这个脑根本跑不
    /// 起来"的三种情况在这里就拦下：不认识的 provider、没有 TLS 却要 `https://`、
    /// 环境变量里没有 key。
    pub fn runtime(&self, provider: &str, model: &str, timeout_ms: u64) -> Result<Runtime, String> {
        let decl = self.providers.get(provider).ok_or_else(|| {
            let known: Vec<&str> = self.providers.keys().map(String::as_str).collect();
            format!("unknown provider `{provider}` (declared: {known:?})")
        })?;
        let base_url = match (&decl.base_url, decl.api_mode) {
            (Some(u), _) => u.trim_end_matches('/').to_string(),
            (None, ApiMode::Realtime) => DEFAULT_REALTIME_URL.to_string(),
            (None, m) if m.is_subscription() => String::new(),
            (None, _) => return Err(format!("provider `{provider}` needs a base_url")),
        };
        if needs_tls(&base_url) && cfg!(not(feature = "tls")) {
            return Err(format!("tls feature not enabled: {base_url}"));
        }
        if !decl.api_mode.is_subscription()
            && (decl.program.is_some() || decl.reasoning_effort.is_some())
        {
            return Err(format!(
                "provider `{provider}`: program/reasoning_effort only apply to codex_exec and claude_code_exec"
            ));
        }
        if let Some(c) = &decl.cache {
            if c != CACHE_SYSTEM_AND_3 {
                return Err(format!(
                    "provider `{provider}`: unknown cache mode `{c}` (only `{CACHE_SYSTEM_AND_3}`)"
                ));
            }
        }
        // 空字符串当没有：`OPENAI_API_KEY=` 是"忘了填"，不是一个 key。
        let api_key = std::env::var(&decl.env).ok().filter(|k| !k.is_empty());
        Ok(Runtime {
            provider: provider.to_string(),
            model: model.to_string(),
            api_mode: decl.api_mode,
            base_url,
            env: decl.env.clone(),
            api_key,
            cache: decl.cache.clone(),
            headers: decl.headers.clone(),
            timeout_ms,
            program: decl.program.clone(),
            reasoning_effort: decl.reasoning_effort.clone(),
        })
    }
}

/// `openai-realtime` 这类只声明了 `api_mode: realtime` 的 provider 的端点。
const DEFAULT_REALTIME_URL: &str = "wss://api.openai.com/v1/realtime";

fn needs_tls(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("wss://")
}

/// 一个**可以真的发请求**的运行时：声明里的东西加上解析好的 model id、key 与超时。
///
/// `api_key` 是 `Option`：环境变量没设时这里就是 `None`，attach 照常成功，第一次真要
/// 发请求时传输还 [`ModelError::MissingKey`](crate::model::ModelError::MissingKey)（带着
/// `env` 的名字），瀑布不重试、直接换下一个 runtime。
#[derive(Clone, Debug)]
pub struct Runtime {
    pub provider: String,
    pub model: String,
    pub api_mode: ApiMode,
    pub base_url: String,
    /// key 的来源，只用来把 `MissingKey` 说清楚。
    pub env: String,
    pub api_key: Option<String>,
    pub cache: Option<String>,
    pub headers: BTreeMap<String, String>,
    /// 一次请求（含把流读完）的上限。0 = 不设限。
    pub timeout_ms: u64,
    /// 订阅 wire 的可执行文件；别的 wire 是 `None`。
    pub program: Option<String>,
    /// 订阅 wire 的推理强度；别的 wire 是 `None`。
    pub reasoning_effort: Option<String>,
}

impl Runtime {
    /// `<provider>/<model>`：[`Model::name`] 与 `brain/request` 里的身份。
    pub fn name(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }

    /// 这个 runtime 要不要打 prompt 缓存点。
    pub fn caches(&self) -> bool {
        self.cache.as_deref() == Some(CACHE_SYSTEM_AND_3)
    }
}

/// 按 `api_mode` 挑传输。
pub fn build_model(
    p: &Providers,
    provider: &str,
    model: &str,
    timeout_ms: u64,
) -> Result<Arc<dyn Model>, String> {
    build_model_with_log(p, provider, model, timeout_ms, None)
}

/// [`build_model`]，外加 `realtime` 写 `obs/speech` 用的世界日志。
///
/// 分成两个函数而不是给 `build_model` 加一个参数：只有 realtime 用得上日志（语音转写是
/// 一条观测，别的脑与回放也该看见，规格 §7.2），而三种 HTTP 传输没有任何理由拿着一个
/// 能往世界日志里写东西的句柄。
pub fn build_model_with_log(
    p: &Providers,
    provider: &str,
    model: &str,
    timeout_ms: u64,
    log: Option<Arc<WorldLog>>,
) -> Result<Arc<dyn Model>, String> {
    transport_for(p.runtime(provider, model, timeout_ms)?, log)
}

fn transport_for(rt: Runtime, log: Option<Arc<WorldLog>>) -> Result<Arc<dyn Model>, String> {
    Ok(match rt.api_mode {
        ApiMode::ChatCompletions => Arc::new(ChatCompletionsModel::new(rt)?),
        ApiMode::AnthropicMessages => Arc::new(AnthropicMessagesModel::new(rt)?),
        ApiMode::Responses => Arc::new(ResponsesModel::new(rt)?),
        ApiMode::CodexExec => Arc::new(CodexExecModel::new(rt)?),
        ApiMode::ClaudeCodeExec => Arc::new(ClaudeCodeExecModel::new(rt)?),
        ApiMode::Realtime => {
            let m = RealtimeModel::new(rt)?;
            Arc::new(match log {
                Some(l) => m.with_world_log(l),
                None => m,
            })
        }
    })
}

/// fallback 链：`runtimes[0]` 是主 runtime，失败之后依次往后试（[`crate::retry`]）。
///
/// 链在**这里**就解析完（所有 provider 都认识、都有 key、端点都能连），而不是等到某个
/// turn 真的降级时才发现第二顺位配错了——那正是最需要它能用的时刻。
pub struct FallbackChain {
    pub runtimes: Vec<Runtime>,
}

impl FallbackChain {
    /// `pairs` 是 `(provider, model)`，主 runtime 在前。
    pub fn resolve(
        p: &Providers,
        pairs: &[(String, String)],
        timeout_ms: u64,
    ) -> Result<Self, String> {
        let mut runtimes = Vec::with_capacity(pairs.len());
        for (provider, model) in pairs {
            runtimes.push(p.runtime(provider, model, timeout_ms)?);
        }
        Ok(FallbackChain { runtimes })
    }

    /// 每个 runtime 造一个传输，顺序不变。
    pub fn models(&self, log: Option<Arc<WorldLog>>) -> Result<Vec<Arc<dyn Model>>, String> {
        self.runtimes
            .iter()
            .map(|rt| transport_for(rt.clone(), log.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 仓库里那份声明必须解得开——它是 `brain-native` 的默认配置。
    const DEFAULT_YAML: &str = include_str!("../../../providers/default.yaml");

    fn yaml(extra: &str) -> Providers {
        Providers::from_yaml(&format!(
            "providers:\n  local: {{ api_mode: chat_completions, base_url: http://127.0.0.1:1, env: OEH_TEST_PROVIDER_KEY{extra} }}\n"
        ))
        .unwrap()
    }

    /// 每个 `api_mode` 真的接到了那个传输上。
    ///
    /// 这段 `match` 是全仓唯一把 wire 的名字和实现连起来的地方，而错接两条分支编译得过、
    /// 别的测试也全绿：一个声明 `responses` 的 provider 会安安静静地说 chat_completions。
    #[test]
    fn each_api_mode_builds_its_own_transport() {
        let rt = |api_mode, base_url: &str| Runtime {
            provider: "p".into(),
            model: "m".into(),
            api_mode,
            base_url: base_url.into(),
            env: "K".into(),
            api_key: Some("k".into()),
            cache: None,
            headers: BTreeMap::new(),
            timeout_ms: 0,
            program: None,
            reasoning_effort: None,
        };
        for (mode, url) in [
            (ApiMode::ChatCompletions, "http://127.0.0.1:1"),
            (ApiMode::AnthropicMessages, "http://127.0.0.1:1"),
            (ApiMode::Responses, "http://127.0.0.1:1"),
            (ApiMode::Realtime, "ws://127.0.0.1:1"),
            (ApiMode::CodexExec, ""),
            (ApiMode::ClaudeCodeExec, ""),
        ] {
            let m = transport_for(rt(mode, url), None).unwrap();
            assert_eq!(m.api_mode(), mode.as_str());
            assert_eq!(m.name(), "p/m");
        }
    }

    #[test]
    fn the_shipped_declaration_parses_into_its_wires() {
        let p = Providers::from_yaml(DEFAULT_YAML).unwrap();
        assert_eq!(p.providers["openai"].api_mode, ApiMode::Responses);
        assert_eq!(
            p.providers["anthropic"].api_mode,
            ApiMode::AnthropicMessages
        );
        assert_eq!(
            p.providers["anthropic"].cache.as_deref(),
            Some("system_and_3")
        );
        assert_eq!(p.providers["deepseek"].api_mode, ApiMode::ChatCompletions);
        assert_eq!(p.providers["openai-realtime"].api_mode, ApiMode::Realtime);
        assert!(p.providers["openai-realtime"].base_url.is_none());
        let codex = &p.providers["codex"];
        assert_eq!(codex.api_mode, ApiMode::CodexExec);
        assert_eq!(codex.env, "OEH_CODEX_HOME");
        assert_eq!(codex.reasoning_effort.as_deref(), Some("high"));
        let rt = p.runtime("codex", "gpt-6-sol", 1000).unwrap();
        assert_eq!(rt.base_url, "");
        let claude = &p.providers["claude-code"];
        assert_eq!(claude.api_mode, ApiMode::ClaudeCodeExec);
        assert_eq!(claude.env, "CLAUDE_CODE_OAUTH_TOKEN");
        assert_eq!(claude.program.as_deref(), Some("claude"));
        let rt = p.runtime("claude-code", "claude-opus-5-5", 1000).unwrap();
        assert_eq!(rt.base_url, "");
    }

    #[test]
    fn codex_only_options_are_rejected_on_http_wires() {
        let err = Providers::from_yaml(
            "providers:\n  local: { api_mode: responses, base_url: http://x, env: K, program: codex }\n",
        )
        .unwrap()
        .runtime("local", "m", 1000)
        .unwrap_err();
        assert!(
            err.contains("only apply to codex_exec and claude_code_exec"),
            "{err}"
        );
    }

    #[test]
    fn an_https_endpoint_without_the_tls_feature_is_a_named_error() {
        let p = Providers::from_yaml(DEFAULT_YAML).unwrap();
        if cfg!(feature = "tls") {
            // 开了 tls 的构建上这条路径不该报错（缺 key 也不报）。
            assert!(p.runtime("anthropic", "claude", 1000).is_ok());
        } else {
            let err = p.runtime("anthropic", "claude", 1000).unwrap_err();
            assert_eq!(err, "tls feature not enabled: https://api.anthropic.com");
        }
    }

    /// 缺 key **不**是 boot 时的错：runtime 造得出来，`api_key` 是 `None`，名字留在
    /// `env` 上等传输去报 `MissingKey`。
    #[test]
    fn a_missing_key_still_builds_a_runtime_that_knows_which_variable_it_wanted() {
        let rt = yaml("").runtime("local", "m", 1000).unwrap();
        assert!(rt.api_key.is_none());
        assert_eq!(rt.env, "OEH_TEST_PROVIDER_KEY");
        // 造得出模型（attach 会成功），只是它发不出请求。
        let p = yaml("");
        assert!(build_model(&p, "local", "m", 1000).is_ok());
    }

    #[test]
    fn an_unknown_provider_lists_the_declared_ones() {
        let err = yaml("").runtime("nope", "m", 1000).unwrap_err();
        assert!(err.contains("unknown provider `nope`"), "{err}");
        assert!(err.contains("local"), "{err}");
    }

    #[test]
    fn an_unknown_cache_mode_is_rejected() {
        let err = yaml(", cache: everything")
            .runtime("local", "m", 1000)
            .unwrap_err();
        assert!(err.contains("unknown cache mode"), "{err}");
    }

    #[test]
    fn a_typo_in_a_declaration_is_rejected_rather_than_ignored() {
        let err = Providers::from_yaml(
            "providers:\n  local: { api_mode: responses, base_ur: http://x, env: K }\n",
        )
        .unwrap_err();
        assert!(err.contains("base_ur"), "{err}");
    }
}
