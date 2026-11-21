//! 受保护路径：RSI 永远写不进去的那几个目录。
//!
//! 规格 §10.1 列的是 profile、bundle、patch、wake 策略、规则配置、system prompt 模板。
//! 这几样东西的共同点是：**它们决定了下一次启动时这个机器人是什么**。skill 改错了，下
//! 一次 review 可以回滚；profile 或 bundle 改错了，回滚的那套机制本身可能就起不来了。
//! 所以它们不在"可改"的那一级里——写进去不是失败，是 `safety/tamper_attempt`。
//!
//! 判据是**解析之后**的绝对路径前缀，不是字符串前缀，也不是 rel 的形状：
//! [`KnowledgeStore::write`](crate::rsi::KnowledgeStore::write) 先把 rel 解析成绝对路径
//! 再问 [`ProtectedPaths::covers`]。这样一来，哪怕 store 的根被（配置错误、或者有意）
//! 架在 `bundles/` 里面，一条形状完全合法的 `skills/nav/SKILL.md` 也照样会被挡下来。

use std::path::{Component, Path, PathBuf};

/// 一组绝对路径前缀。空集合表示"什么都不保护"（[`Default`]），只在测试和没有仓库布局的
/// 场景下用——生产里由 profile 构造。
#[derive(Clone, Debug, Default)]
pub struct ProtectedPaths(Vec<PathBuf>);

impl ProtectedPaths {
    /// 收下一组路径前缀。每一条在这里就尽可能解析成真实路径，这样 `covers` 的比较两边
    /// 都是"解析过的"，符号链接绕不过去。
    pub fn new(paths: impl IntoIterator<Item = PathBuf>) -> Self {
        Self(paths.into_iter().map(|p| resolve_best_effort(&p)).collect())
    }

    /// 从一份 profile 推出这一台机器人的受保护集合。
    ///
    /// `bundles` / `patch` / `wake` / `profiles` 是**无条件**加的（哪怕此刻还不存在）：
    /// "目录还没建出来"不该变成一个可以往里写的窗口。`rules` / `prompts` 只在存在时加，
    /// 它们的名字没有像前四个那样被布局固定下来，凭空保护一个同名目录会误伤。
    pub fn from_profile(profile_path: &Path, repo_root: &Path) -> Self {
        let mut v = vec![profile_path.to_path_buf()];
        v.extend(
            ["bundles", "patch", "wake", "profiles"]
                .iter()
                .map(|d| repo_root.join(d)),
        );
        v.extend(
            ["rules", "prompts"]
                .iter()
                .map(|d| repo_root.join(d))
                .filter(|p| p.is_dir()),
        );
        Self::new(v)
    }

    /// `p` 是不是落在某个受保护前缀里（前缀本身也算）。
    ///
    /// 比较是**按路径段**的（`Path::starts_with` 的语义），所以 `…/bundles-old` 不会被
    /// `…/bundles` 覆盖；如果按字符串前缀比，那就会误伤。
    pub fn covers(&self, p: &Path) -> bool {
        if self.0.is_empty() {
            return false;
        }
        let real = resolve_best_effort(p);
        self.0.iter().any(|root| real.starts_with(root))
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// 尽力而为的 canonicalize：路径不存在时，把**存在的那一截**解析掉，剩下的段按字面接回去。
///
/// `std::fs::canonicalize` 要求整条路径都存在，而受保护集合里天然有还不存在的目录
/// （`patch/` 在第一次 patch 之前是没有的），待检查的路径更是绝大多数都还不存在
/// （正要新建的 `SKILL.md`）。直接放弃解析会让一条符号链接就绕开整个保护；只解析存在
/// 的那一截，既能拆掉链接，又不会因为"还没建出来"就漏判。
pub(crate) fn resolve_best_effort(p: &Path) -> PathBuf {
    if let Ok(c) = p.canonicalize() {
        return c;
    }
    // 从下往上找第一个存在的祖先。
    let mut rest: Vec<&std::ffi::OsStr> = Vec::new();
    let mut cur = p;
    while let (Some(parent), Some(name)) = (cur.parent(), cur.file_name()) {
        rest.push(name);
        if let Ok(base) = parent.canonicalize() {
            return push_lexical(base, rest.iter().rev().copied());
        }
        cur = parent;
    }
    // 一个祖先都解析不了（相对路径、或者整条都不存在）：退回字面量的词法归一。
    push_lexical(PathBuf::new(), p.components().map(Component::as_os_str))
}

/// 把若干段接到 `base` 后面，顺手把 `.` 和 `..` 按词法消掉。
///
/// 这些段是**没能解析**的部分，也就是磁盘上还不存在的部分——对不存在的路径来说，词法
/// 归一和物理归一是一回事，而留着 `..` 会让 `starts_with` 的前缀判定失真。
fn push_lexical<'a>(mut base: PathBuf, segs: impl Iterator<Item = &'a std::ffi::OsStr>) -> PathBuf {
    for seg in segs {
        if seg == "." {
            continue;
        }
        if seg == ".." {
            base.pop();
            continue;
        }
        base.push(seg);
    }
    base
}
