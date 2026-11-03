//! [`KnowledgeStore`]：RSI 第一级唯一的写入口。
//!
//! 一次 `write` 的提交顺序是这个文件里最要紧的东西，它决定了"崩在任意一步"之后磁盘
//! 上剩下什么：
//!
//! ```text
//! ① 校验路径（句法 → 物理 → 受保护）      失败：一个 syscall 都没发出去
//! ② 读当前内容 + 读 .meta/ 里的 created_by
//! ③ 所有权 → read-before-write            失败：磁盘没被碰过
//! ④ 旧内容写进 .blobs/<prev_sha>，fsync    崩：多一份用不上的备份（内容寻址，无害）
//! ⑤ 新建时写 .meta/<rel>.json，fsync       崩：多一条无主文件的元数据（见下）
//! ⑥ 写 <rel>.tmp-<n> → fsync → rename      崩：要么旧内容，要么新内容，没有中间态
//! ⑦ 追加账本一行，fsync                    崩：文件已经变了但账本上没有（见下）
//! ```
//!
//! **④⑤ 在 rename 之前、⑦ 在 rename 之后**，两条都是有意的：
//!
//! - 备份先于覆盖，否则"崩在备份之前"会让旧内容彻底消失，回滚无从谈起；
//! - `created_by` 先于文件本身：反过来的话，崩在中间会留下一个**没有主人**的文件，而
//!   无主文件是所有权护栏的一个洞（见 [`KnowledgeStore::write`] 里怎么处理无主文件）。
//!   反方向的代价只是一条指向不存在文件的元数据，而那是无害的：文件从来没有提交过，
//!   谈不上所有权，下一次新建会把它直接覆盖掉；
//! - 账本晚于 rename：账本上不能出现磁盘上没发生过的事（那会让回滚把内容改成一个从未
//!   存在过的版本）。代价是 rename 与追加之间有一个窗口，崩在里面会留下一次**没入账的
//!   变更**。这是被接受的窗口：[`KnowledgeStore::ledger`] 的读者据此必须容忍"文件的当前
//!   哈希不等于账本最后一行的 `new_sha256`"，而不是把它当成损坏。
//!
//! 失败一律清理临时文件（和这次新建出来的空目录）：一个 `SKILL.md.tmp-7` 躺在
//! `skills/nav/` 里，对下一次 `list` 是噪音，对人是"这次写到底成没成"的疑问。
//!
//! **已知边界**：路径检查和 rename 之间有一个 TOCTOU 窗口——检查用的是词法路径
//! `root.join(rel)`，如果在这两步之间有人往 `skills/nav` 上挂一条符号链接，rename 会跟着
//! 它走。威胁模型里够不到：这套 API 自己永远不建符号链接，脑也只能通过它碰磁盘，所以要
//! 利用这个窗口，得先有一个能往 store 根里写东西的进程外写手——那个前提下它直接改
//! `SKILL.md` 就行了，不必绕道。真要关掉它得走 `openat` + `O_NOFOLLOW` 的 dirfd，那是一
//! 笔与当前威胁模型不相称的复杂度，所以记在这里而不是做掉。

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use oeh_kernel::world_log::WorldLog;
use oeh_proto::{EventDraft, EventKind};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::ledger::{self, LedgerEntry, Op, LEDGER_FILE};
use super::protected::ProtectedPaths;

/// 这个模块写进世界日志的事件的 `source`。
const SOURCE: &str = "rsi";
/// `safety/tamper_attempt` 的显著度。比准入链里的规则篡改（255）低一档：那一条说的是
/// "安全链自己被人动了"，这一条说的是"脑想写它不该写的文件"——都要醒目，但前者更急。
const SAL_TAMPER: u8 = 200;
/// 内容寻址的备份目录。前面那个点不是装饰：句法白名单禁止任何以 `.` 开头的段，所以
/// 模型够不到这里，也够不到 `.meta/`。账本（`skill_ledger.jsonl`）没有前导点，挡住它的
/// 是**形状**白名单（三种形状都不是根下的裸文件）——两处的理由不同，改任何一处之前都得
/// 想清楚另一处还成不成立。
const BLOBS_DIR: &str = ".blobs";
const META_DIR: &str = ".meta";

/// 一份知识文档。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Doc {
    pub rel: String,
    pub text: String,
    /// 内容的 SHA-256（小写十六进制）。下一次写必须把它原样带回来——这就是
    /// read-before-write。
    pub sha256: String,
    /// 谁建的。`None` = 磁盘上没有这条记录（人手放进来的文件，或者一次崩在中途的新建）。
    pub created_by: Option<String>,
}

/// 三类可写文档。`list` 用它分类，`write` 不用——写的白名单直接由 rel 的形状决定。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// `skills/**/SKILL.md`
    Skill,
    /// `memory/MEMORY.md`
    Memory,
    /// `memory/topics/<name>.md`
    Topic,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RsiError {
    #[error("no such document: {0}")]
    NotFound(String),
    /// 受保护路径（profile / bundle / patch / wake / 规则 / prompt 模板）。伴随一条
    /// `safety/tamper_attempt`。
    #[error("refusing to write a protected path: {0}")]
    Protected(String),
    /// 带回来的哈希不是磁盘上的那一个：你读到的那一版已经不是当前版本了。
    /// `expected` / `actual` 为空串表示"那一版不存在 / 文件不存在"。
    #[error("stale read: expected sha256 {expected:?}, on disk {actual:?}")]
    StaleRead { expected: String, actual: String },
    #[error("{actor} cannot write a document created by {owner}")]
    NotOwner { owner: String, actor: String },
    #[error("io error: {0}")]
    Io(String),
    /// 路径没通过句法或物理检查。不落盘。
    #[error("bad path: {0}")]
    BadPath(String),
}

impl RsiError {
    fn io(ctx: &str, e: impl std::fmt::Display) -> Self {
        RsiError::Io(format!("{ctx}: {e}"))
    }
}

/// 故障注入点。**只在 `test-hooks` feature 下存在**。
///
/// "rename 之前失败不留半截文件"这条性质，除了在那一步真的失败一次，没有别的办法证明。
/// 但一个 `pub fn __inject_fault` 哪怕标了 `#[doc(hidden)]`，也仍然是编译进产品的公开
/// API——工作区里任何一个 crate 都调得到，调了就能让之后每一次 `write` / `rollback` 都
/// 失败。所以它被关在 `test-hooks` 后面：这个 feature 只由 `oeh-brain` 自己的
/// dev-dependency 自依赖打开（见 `Cargo.toml`），产品构建里这个类型和那个方法**不存在**。
/// 集成测试照常在 `cargo test --workspace` 里跑得到，因为 dev-dependency 对测试目标生效。
#[cfg(feature = "test-hooks")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FaultPoint {
    #[default]
    None,
    /// 临时文件写完、fsync 完，正要 rename 时失败。
    BeforeRename,
}

/// store 上挂着的故障开关。没有 `test-hooks` 时它是个零字节的空壳，
/// [`Faults::fail_before_rename`] 恒为 `false`。
#[derive(Default)]
struct Faults {
    #[cfg(feature = "test-hooks")]
    before_rename: std::sync::atomic::AtomicBool,
}

impl Faults {
    #[cfg(feature = "test-hooks")]
    fn fail_before_rename(&self) -> bool {
        self.before_rename.load(Ordering::SeqCst)
    }

    #[cfg(not(feature = "test-hooks"))]
    fn fail_before_rename(&self) -> bool {
        false
    }
}

/// `$OEH_HOME/robots/<id>` 下的知识存储。
pub struct KnowledgeStore {
    root: PathBuf,
    protected: ProtectedPaths,
    log: Option<Arc<WorldLog>>,
    /// 把"读当前内容 → 判定 → 写"串成一个临界区。read-before-write 挡的是**跨进程、
    /// 跨会话**的陈旧覆盖，挡不住同一个进程里两个并发写手同时读到同一个哈希、又都认为
    /// 自己是对的。这把锁让那种情况退化成"后一个拿到 StaleRead"。
    write_lock: Mutex<()>,
    faults: Faults,
}

impl KnowledgeStore {
    /// 建目录树（`skills/`、`memory/topics/`、`.blobs/`、`.meta/`）和一本空账本。
    ///
    /// 根目录在这里就 canonicalize：后面每一次路径检查都拿它当参照，参照本身必须是解析
    /// 过的，否则 `/tmp` 这种本身就是符号链接的前缀会让每一次比较都失败。
    pub fn open(
        root: impl Into<PathBuf>,
        protected: ProtectedPaths,
        log: Option<Arc<WorldLog>>,
    ) -> Result<Self, RsiError> {
        let root = root.into();
        for d in ["skills", "memory/topics", BLOBS_DIR, META_DIR] {
            fs::create_dir_all(root.join(d)).map_err(|e| {
                RsiError::io(&format!("cannot create {}", root.join(d).display()), e)
            })?;
        }
        let ledger_path = root.join(LEDGER_FILE);
        if !ledger_path.exists() {
            fs::File::create(&ledger_path).map_err(|e| {
                RsiError::io(&format!("cannot create {}", ledger_path.display()), e)
            })?;
        }
        let root = root
            .canonicalize()
            .map_err(|e| RsiError::io(&format!("cannot resolve {}", root.display()), e))?;
        if protected.is_empty() {
            // 空集合 = 什么都不保护。这是合法的（测试、没有仓库布局的场景），但如果是
            // 接线时把 `repo_root` 传错了，症状会是"护栏一声不响地失效"——`covers` 只会
            // 老老实实返回 false。所以开店时说一声。
            tracing::warn!(
                root = %root.display(),
                "opening an RSI knowledge store with an empty protected-path set: \
                 profile / bundle / patch / wake / rules / prompts are NOT protected"
            );
        }
        let store = Self {
            root,
            protected,
            log,
            write_lock: Mutex::new(()),
            faults: Faults::default(),
        };
        store.seed_existing_meta();
        Ok(store)
    }

    /// 给"已经躺在磁盘上、却没有 `.meta` 记录"的文档补一条 `created_by: "seed"`。
    ///
    /// 这些文件的来源是 profile 预置或者人手放进来的。不补的话它们的主人是"不明"，而
    /// 不明按别人处理（fail-closed），于是**谁都改不了**——预置一份 `SKILL.md` 等于把它
    /// 冻住。补成一个有名有姓的 `"seed"` 之后，语义就清楚了：它属于预置方，review fork
    /// 改不了它（`NotOwner{owner: "seed"}`，而不是一个莫名其妙的 `"unknown"`），而人可以
    /// 在仓库里继续维护它。
    ///
    /// 补种失败只记 `warn!`：补不上的后果是退回"不明"，方向仍然是 fail-closed，不该让
    /// 一个只读的知识目录连 store 都开不起来。
    fn seed_existing_meta(&self) {
        for kind in [Kind::Skill, Kind::Memory, Kind::Topic] {
            let Ok(rels) = self.list(kind) else { continue };
            for rel in rels {
                if !matches!(self.owner(&rel), Ok(None)) {
                    continue;
                }
                if let Err(e) = self.put_meta(&rel, SEED_OWNER) {
                    tracing::warn!(rel = %rel, error = %e, "cannot seed rsi meta");
                }
            }
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 读一份文档。对所有 actor 开放——所有权管的是"谁能改"，不是"谁能看"：review fork
    /// 必须读得到别人写的 skill，否则它连"这条知识已经有人记过了"都不知道。
    pub fn read(&self, rel: &str) -> Result<Doc, RsiError> {
        let abs = self.resolve(rel)?;
        let text = match fs::read_to_string(&abs) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(RsiError::NotFound(rel.to_string()))
            }
            Err(e) => return Err(RsiError::io(&format!("cannot read {rel}"), e)),
        };
        Ok(Doc {
            rel: rel.to_string(),
            sha256: sha256_hex(text.as_bytes()),
            text,
            created_by: self.owner(rel)?,
        })
    }

    /// 列出某一类文档的 rel 路径（字典序）。
    ///
    /// 只返回**形状合法**的路径：`list` 的结果必须每一条都能原样喂给 `read` / `write`，
    /// 否则调用方得自己再过一遍白名单。人手放进 `skills/` 的 `notes.md`、还没清掉的临时
    /// 文件、符号链接，都在这里被滤掉。
    pub fn list(&self, kind: Kind) -> Result<Vec<String>, RsiError> {
        let mut out = Vec::new();
        match kind {
            Kind::Skill => self.walk(&self.root.join("skills"), &mut out)?,
            // 只有一个候选，没必要为它走一遍整棵 `memory/`（那会把 topics 全捡起来再滤掉）。
            Kind::Memory => {
                let p = self.root.join("memory/MEMORY.md");
                if fs::symlink_metadata(&p)
                    .map(|m| m.is_file())
                    .unwrap_or(false)
                {
                    out.push("memory/MEMORY.md".to_string());
                }
            }
            Kind::Topic => self.walk(&self.root.join("memory/topics"), &mut out)?,
        }
        out.retain(|rel| check_shape(rel).is_ok() && self.kind_of(rel) == Some(kind));
        out.sort();
        Ok(out)
    }

    /// 写一份文档。
    ///
    /// - 新建：`expect_sha256` 必须是 `None`（你不可能读到过一个还不存在的版本）；
    /// - 覆盖：`expect_sha256` 必须等于磁盘上当前内容的哈希，否则 [`RsiError::StaleRead`]；
    /// - 覆盖：`created_by` 必须等于 `actor`，否则 [`RsiError::NotOwner`]。
    ///
    /// **所有权检查排在 read-before-write 前面**：越权是比陈旧更严重的一类错误，把它排
    /// 后面的话，一个越权的写手会先收到 `StaleRead`，于是它去读一遍、带上正确的哈希再
    /// 来——白白多绕一圈才撞到真正的那堵墙，日志上还留下一条误导人的 `StaleRead`。
    ///
    /// 文件存在却没有 `.meta/`（人手放进来的、或者崩在新建中途的）一律按**别人的**处理：
    /// 规格 §10.1 说 fork 只能改自己创建的，无主不等于无人认领。
    pub fn write(
        &self,
        rel: &str,
        text: &str,
        actor: &str,
        expect_sha256: Option<&str>,
    ) -> Result<Doc, RsiError> {
        let _g = self.write_lock.lock();
        let abs = self.guard(rel, actor, "write")?;
        let prev = self.current(&abs, rel)?;
        if let Some(p) = &prev {
            self.check_owner(rel, actor)?;
            let want = expect_sha256.unwrap_or("");
            if want != p.sha256 {
                return Err(RsiError::StaleRead {
                    expected: want.to_string(),
                    actual: p.sha256.clone(),
                });
            }
        } else if let Some(want) = expect_sha256 {
            return Err(RsiError::StaleRead {
                expected: want.to_string(),
                actual: String::new(),
            });
        }
        let op = if prev.is_some() {
            Op::Update
        } else {
            Op::Create
        };
        self.commit(Commit {
            rel,
            abs: &abs,
            text,
            actor,
            op,
            prev,
        })
    }

    /// 回到 `.blobs/<to_sha256>` 那一版。
    ///
    /// 走的是和 `write` 完全相同的提交路径：当前内容照样先进 `.blobs/`（回滚本身可以再
    /// 被回滚），账本照样多一行（`op = "rollback"`）。不同的只有两点——内容来自备份而
    /// 不是调用方，以及不需要 `expect_sha256`（"回到 X"本来就不依赖"现在是什么"）。
    pub fn rollback(&self, rel: &str, to_sha256: &str, actor: &str) -> Result<Doc, RsiError> {
        let _g = self.write_lock.lock();
        let abs = self.guard(rel, actor, "rollback")?;
        let Some(prev) = self.current(&abs, rel)? else {
            return Err(RsiError::NotFound(rel.to_string()));
        };
        self.check_owner(rel, actor)?;
        let blob = self.root.join(BLOBS_DIR).join(to_sha256);
        // 备份的文件名就是内容的哈希，所以"文件名合法"必须先验一遍——否则一个带 `/`
        // 或 `..` 的 sha 参数会把这里变成第二个路径逃逸入口。
        if !is_sha256(to_sha256) || !blob.is_file() {
            return Err(RsiError::NotFound(format!("{BLOBS_DIR}/{to_sha256}")));
        }
        let text = fs::read_to_string(&blob)
            .map_err(|e| RsiError::io(&format!("cannot read {}", blob.display()), e))?;
        self.commit(Commit {
            rel,
            abs: &abs,
            text: &text,
            actor,
            op: Op::Rollback,
            prev: Some(prev),
        })
    }

    /// 整本账本，按**文件顺序**（也就是提交顺序；`seq` 单调但可能有缺口）。见
    /// [`ledger::read_all`]：读不动的行会被跳过，而且最后一次
    /// 变更可能根本没入账（rename 与追加之间的窗口）。
    pub fn ledger(&self) -> Result<Vec<LedgerEntry>, RsiError> {
        ledger::read_all(&self.root.join(LEDGER_FILE))
            .map_err(|e| RsiError::io("cannot read the skill ledger", e))
    }

    /// 打开一个故障注入点。见 [`FaultPoint`]——只在 `test-hooks` feature 下存在。
    #[cfg(feature = "test-hooks")]
    #[doc(hidden)]
    pub fn __inject_fault(&self, f: FaultPoint) {
        self.faults
            .before_rename
            .store(f == FaultPoint::BeforeRename, Ordering::SeqCst);
    }

    // -----------------------------------------------------------------------
    // 内部
    // -----------------------------------------------------------------------

    /// 写路径上的守卫：路径检查 + 受保护路径，任何一条不过都记一条
    /// `safety/tamper_attempt`。
    ///
    /// 为什么 `BadPath` 也记：一条 `../bundles/x.yaml` 和一条落在 `bundles/` 里的合法
    /// rel，意图是同一个——拿知识的写入口去够不该够的东西。只给后者报警，等于让最直白
    /// 的那种尝试成为唯一悄无声息的一种。`read` 不记：读对所有 actor 开放，没有"越权读"
    /// 这回事，把读也记成篡改只会把操作员的告警通道淹掉。
    fn guard(&self, rel: &str, actor: &str, op: &str) -> Result<PathBuf, RsiError> {
        let abs = match self.resolve(rel) {
            Ok(p) => p,
            Err(e) => {
                self.tamper(actor, rel, op);
                return Err(e);
            }
        };
        if self.protected.covers(&abs) {
            self.tamper(actor, &abs.display().to_string(), op);
            return Err(RsiError::Protected(format!(
                "{} is under a protected path",
                abs.display()
            )));
        }
        Ok(abs)
    }

    fn tamper(&self, actor: &str, path: &str, op: &str) {
        let (Some(log), Ok(kind)) = (self.log.as_ref(), EventKind::new("safety/tamper_attempt"))
        else {
            return;
        };
        log.append(
            EventDraft::new(
                kind,
                SOURCE,
                json!({ "actor": actor, "path": path, "op": op }),
            )
            .with_salience(SAL_TAMPER),
        );
    }

    /// rel → 根目录内的绝对路径。先句法后物理，两层都判 [`RsiError::BadPath`]。
    fn resolve(&self, rel: &str) -> Result<PathBuf, RsiError> {
        check_shape(rel)?;
        let abs = self.root.join(rel);
        // 物理层①：从根往下，已经存在的每一段都不能是符号链接。canonicalize 只会告诉你
        // "解析完落在哪"，而一条指向根**内部**的链接同样是个问题：它让同一份内容有两个
        // rel，账本和所有权就对不上号了。
        let mut cur = self.root.clone();
        let segs: Vec<&str> = rel.split('/').collect();
        for (i, seg) in segs.iter().enumerate() {
            cur.push(seg);
            match fs::symlink_metadata(&cur) {
                Ok(m) if m.file_type().is_symlink() => {
                    return Err(RsiError::BadPath(format!(
                        "{rel:?}: {} is a symlink",
                        cur.display()
                    )))
                }
                // 中间的每一段都必须是目录。`skills/nav/SKILL.md/SKILL.md`（`SKILL.md`
                // 已经是个文件）形状上完全合法，放过去只会在最后 rename 时撞出一个
                // ENOTDIR 的 `Io`——那对调用方毫无信息量，在这里说清楚才对。
                Ok(m) if i + 1 < segs.len() && !m.is_dir() => {
                    return Err(RsiError::BadPath(format!(
                        "{rel:?}: {} is not a directory",
                        cur.display()
                    )))
                }
                _ => {}
            }
        }
        // 物理层②：父目录解析完必须还在根里（挂载点、硬链接目录之类的兜底）。
        if let Some(parent) = abs.parent() {
            if let Ok(real) = parent.canonicalize() {
                if !real.starts_with(&self.root) {
                    return Err(RsiError::BadPath(format!(
                        "{rel:?}: resolves to {} outside the store root",
                        real.display()
                    )));
                }
            }
        }
        Ok(abs)
    }

    /// 磁盘上当前那一版（不存在 → `None`）。
    fn current(&self, abs: &Path, rel: &str) -> Result<Option<Doc>, RsiError> {
        match fs::read_to_string(abs) {
            Ok(text) => Ok(Some(Doc {
                rel: rel.to_string(),
                sha256: sha256_hex(text.as_bytes()),
                text,
                created_by: self.owner(rel)?,
            })),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(RsiError::io(&format!("cannot read {rel}"), e)),
        }
    }

    /// `.meta/<sha256(rel)>.json`——**扁平**，而且名字来自 rel 的哈希而不是 rel 本身。
    ///
    /// 早先的版本把 rel 原样镜像进 `.meta/`（`.meta/<rel>.json`），那让元数据和文档共用
    /// 一个命名空间：`skills/nav/SKILL.md.json/SKILL.md` 是一条**合法**的 rel，建出来之后
    /// `.meta/skills/nav/SKILL.md.json` 就成了一个目录，于是 `skills/nav/SKILL.md` 的元数据
    /// 永远写不进去（rename 到一个目录上 = EISDIR），这个 rel 被永久占住；反过来先建
    /// `skills/nav/SKILL.md` 也一样堵死另一条。扁平 + 哈希把两个命名空间彻底分开：哈希
    /// 是单射的（同一个 rel 恒定映到同一个名字，不同 rel 几乎不可能撞），名字里没有分隔符，
    /// 所以 `.meta/` 下永远不会长出子目录，失败路径也就不会留下空目录。rel 原样存在 JSON
    /// 里，`.meta/` 依然是人能读懂的（`grep -l '"rel":"skills/nav/SKILL.md"' .meta/*`）。
    fn meta_path(&self, rel: &str) -> PathBuf {
        self.root
            .join(META_DIR)
            .join(format!("{}.json", sha256_hex(rel.as_bytes())))
    }

    /// 写一条 `.meta` 记录。只在新建（或 [`KnowledgeStore::open`] 的补种）时调用——
    /// `created_by` 一旦写下就不再改。
    fn put_meta(&self, rel: &str, created_by: &str) -> Result<(), RsiError> {
        let meta = self.meta_path(rel);
        let body = serde_json::to_vec(&Meta {
            rel: rel.to_string(),
            created_by: created_by.to_string(),
            created_at_ms: now_ms(),
        })
        .map_err(|e| RsiError::io("cannot encode rsi meta", e))?;
        atomic_write(&meta, &body, false)
            .map_err(|e| RsiError::io(&format!("cannot write {}", meta.display()), e))
    }

    fn owner(&self, rel: &str) -> Result<Option<String>, RsiError> {
        let path = self.meta_path(rel);
        match fs::read_to_string(&path) {
            Ok(s) => match serde_json::from_str::<Meta>(&s) {
                Ok(m) => Ok(Some(m.created_by)),
                // 元数据读不动 = 主人不明 = 按别人的处理（见 `check_owner`）。
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "unreadable rsi meta");
                    Ok(None)
                }
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(RsiError::io(&format!("cannot read {}", path.display()), e)),
        }
    }

    /// 所有权检查。**只对 [`Kind::Skill`] 生效**。
    ///
    /// `memory/MEMORY.md` 和 `memory/topics/*.md` 按设计是共享的：记忆是这台机器人的，
    /// 不是某一个 fork 的——快脑记下的一条偏好，慢脑的 review 必须能接着改。skill 不同，
    /// 它是行为本身，§10.1 要求"fork 只能改自己创建的"。两边都照样入账（账本上永远看得出
    /// 是谁改的），区别只在拒不拒。
    fn check_owner(&self, rel: &str, actor: &str) -> Result<(), RsiError> {
        if self.kind_of(rel) != Some(Kind::Skill) {
            return Ok(());
        }
        let owner = self
            .owner(rel)?
            .unwrap_or_else(|| UNKNOWN_OWNER.to_string());
        if owner == actor {
            return Ok(());
        }
        Err(RsiError::NotOwner {
            owner,
            actor: actor.to_string(),
        })
    }

    /// 提交一次变更。调用方已经做完全部判定，这里只负责"按顺序落盘"（见模块头）。
    ///
    /// 目录是这里唯一会**回滚**的东西：一次失败的新建不该在 `skills/` 下留一个空的技能
    /// 目录（模块头承诺"失败不留垃圾"，而 `list` 把它滤掉只是看不见，不是没有）。只删这
    /// 次自己建出来的那几层，而且用 `remove_dir`——它删不动非空目录，所以并发写手刚放
    /// 进去的东西不会被误删。
    fn commit(&self, c: Commit<'_>) -> Result<Doc, RsiError> {
        let created: Vec<PathBuf> = match c.abs.parent() {
            Some(parent) => {
                // `ancestors()` 从深到浅，`take_while(!exists)` 正好是"这次要新建的那几层"。
                let created: Vec<PathBuf> = parent
                    .ancestors()
                    .take_while(|p| !p.exists())
                    .map(Path::to_path_buf)
                    .collect();
                fs::create_dir_all(parent)
                    .map_err(|e| RsiError::io(&format!("cannot create {}", parent.display()), e))?;
                created
            }
            None => Vec::new(),
        };
        match self.commit_files(&c) {
            Ok(doc) => Ok(doc),
            Err(e) => {
                for d in &created {
                    let _ = fs::remove_dir(d);
                }
                Err(e)
            }
        }
    }

    /// [`Self::commit`] 的第 ④–⑦ 步（父目录已经建好）。
    fn commit_files(&self, c: &Commit<'_>) -> Result<Doc, RsiError> {
        let new_sha = sha256_hex(c.text.as_bytes());
        // ④ 旧内容先进备份：这一步在覆盖之前，否则"崩在备份和覆盖之间"就没有旧内容了。
        if let Some(p) = &c.prev {
            let blob = self.root.join(BLOBS_DIR).join(&p.sha256);
            if !blob.exists() {
                atomic_write(&blob, p.text.as_bytes(), false)
                    .map_err(|e| RsiError::io(&format!("cannot back up {}", c.rel), e))?;
            }
        }
        // ⑤ 新建时落 created_by。
        if c.prev.is_none() {
            self.put_meta(c.rel, c.actor)?;
        }
        // ⑥ 原子写。
        atomic_write(c.abs, c.text.as_bytes(), self.faults.fail_before_rename())
            .map_err(|e| RsiError::io(&format!("cannot write {}", c.rel), e))?;
        // ⑦ 入账。seq 取**最大值**加一，不是最后一行加一：`read_all` 会跳过读不动的行，
        // 而那正是被接受的崩溃窗口最可能留下的东西——按最后一行走的话，一条坏掉的尾行就
        // 会让 seq 倒退并复用一个已经用过的号。
        let path = self.root.join(LEDGER_FILE);
        let seq = ledger::read_all(&path)
            .map_err(|e| RsiError::io("cannot read the skill ledger", e))?
            .iter()
            .map(|e| e.seq)
            .max()
            .unwrap_or(0)
            + 1;
        let entry = LedgerEntry {
            seq,
            ts_unix_ms: now_ms(),
            actor: c.actor.to_string(),
            op: c.op,
            rel: c.rel.to_string(),
            prev_sha256: c.prev.as_ref().map(|p| p.sha256.clone()),
            new_sha256: new_sha.clone(),
            bytes: c.text.len() as u64,
        };
        if let Err(e) = ledger::append(&path, &entry) {
            // 文件已经换过去了，账本却没记上。这不是崩溃窗口（那种情况没人还活着报错），
            // 而是一次普通的 I/O 失败（ENOSPC 之类），调用方会收到 `Err` 并合理地以为
            // "没写成"——所以把对账需要的三样东西（rel、新哈希、actor）留在一条 error 里，
            // 否则没人能把磁盘上的现状和账本对上。
            tracing::error!(
                rel = %c.rel, new_sha256 = %new_sha, actor = %c.actor, error = %e,
                "rsi document was replaced but its ledger line could not be appended"
            );
            return Err(RsiError::io("cannot append to the skill ledger", e));
        }
        Ok(Doc {
            rel: c.rel.to_string(),
            text: c.text.to_string(),
            sha256: new_sha,
            created_by: Some(
                c.prev
                    .as_ref()
                    .and_then(|p| p.created_by.clone())
                    .unwrap_or_else(|| c.actor.to_string()),
            ),
        })
    }

    /// 递归收集一棵子树下的文件（**不跟符号链接**），结果是 rel 字符串。
    fn walk(&self, dir: &Path, out: &mut Vec<String>) -> Result<(), RsiError> {
        let rd = match fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(RsiError::io(&format!("cannot list {}", dir.display()), e)),
        };
        for entry in rd {
            let entry =
                entry.map_err(|e| RsiError::io(&format!("cannot list {}", dir.display()), e))?;
            let path = entry.path();
            let ft = entry
                .file_type()
                .map_err(|e| RsiError::io(&format!("cannot stat {}", path.display()), e))?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                self.walk(&path, out)?;
            } else if let Ok(rel) = path.strip_prefix(&self.root) {
                out.push(
                    rel.components()
                        .map(|c| c.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>()
                        .join("/"),
                );
            }
        }
        Ok(())
    }

    fn kind_of(&self, rel: &str) -> Option<Kind> {
        if rel == "memory/MEMORY.md" {
            Some(Kind::Memory)
        } else if rel.starts_with("memory/topics/") {
            Some(Kind::Topic)
        } else if rel.starts_with("skills/") {
            Some(Kind::Skill)
        } else {
            None
        }
    }
}

/// 一次提交的全部输入。合成一个结构体只是为了不让 `commit` 变成七个位置参数——那种签名
/// 调错一个顺序（`rel` 和 `actor` 都是 `&str`）编译器是不会响的。
struct Commit<'a> {
    rel: &'a str,
    abs: &'a Path,
    text: &'a str,
    actor: &'a str,
    op: Op,
    /// 磁盘上被覆盖掉的那一版；新建时是 `None`。
    prev: Option<Doc>,
}

/// `.meta/<sha256(rel)>.json` 的内容。`rel` 存在里面：文件名是哈希，没有它这个目录就只是
/// 一堆无法辨认的十六进制。
#[derive(Serialize, Deserialize)]
struct Meta {
    #[serde(default)]
    rel: String,
    created_by: String,
    created_at_ms: u64,
}

/// 文件存在、`.meta/` 不在时记在 [`RsiError::NotOwner`] 里的主人名。开店时的补种
/// （[`KnowledgeStore::seed_existing_meta`]）之后，这个名字只可能出现在"store 开着的时候
/// 有人从外面塞了个文件进来"这一种情况里。
const UNKNOWN_OWNER: &str = "unknown";

/// 开店时给已有文件补种的主人名。
const SEED_OWNER: &str = "seed";

/// 内容的 SHA-256，小写十六进制。
///
/// 选 SHA-256 而不是任何更快的哈希：账本和 `.blobs/` 的文件名要能被 `sha256sum` 这种
/// 完全在本项目之外的工具核对——审计材料的价值就在于不需要相信产生它的那个程序。
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// rel 的句法白名单：`skills/<seg>/…/SKILL.md`、`memory/MEMORY.md`、
/// `memory/topics/<name>.md`。
///
/// 这一层在任何一次 `stat` 之前跑完，**不碰磁盘**。它同时是路径逃逸的第一道闸：没有
/// `..`、没有绝对路径、没有空段、没有以 `.` 开头的段（于是 `.blobs/`、`.meta/` 和账本
/// 自己都在模型够不到的地方），段里只允许 `[A-Za-z0-9._-]`（于是没有空格、换行、NUL、
/// 反斜杠，也就没有"文件名里藏一条 shell 命令"那一类下游问题）。
fn check_shape(rel: &str) -> Result<(), RsiError> {
    let bad = |why: &str| Err(RsiError::BadPath(format!("{rel:?}: {why}")));
    if rel.is_empty() {
        return bad("empty");
    }
    if rel.starts_with('/') {
        return bad("absolute paths are not allowed");
    }
    let segs: Vec<&str> = rel.split('/').collect();
    for s in &segs {
        if s.is_empty() {
            return bad("empty path segment");
        }
        if s.starts_with('.') {
            // `..`、`.`，以及 `.blobs` / `.meta` 这类内部目录，一并挡在这里。
            return bad("path segments may not start with a dot");
        }
        if !s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
        {
            return bad("path segments may only contain [A-Za-z0-9._-]");
        }
    }
    match segs.as_slice() {
        // `skills/<至少一段>/SKILL.md`：技能住在自己的目录里，`skills/SKILL.md` 不是技能。
        ["skills", rest @ ..] if rest.len() >= 2 && rest[rest.len() - 1] == "SKILL.md" => Ok(()),
        ["memory", "MEMORY.md"] => Ok(()),
        ["memory", "topics", name] if name.ends_with(".md") => Ok(()),
        _ => bad("not one of skills/**/SKILL.md, memory/MEMORY.md, memory/topics/*.md"),
    }
}

/// 临时文件后缀里的进程内序号，保证同一个进程里两次写不会撞名。
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// 临时文件后缀：`<进程内序号>-<随机数>`。
///
/// 序号管进程内，随机数管进程间——同一个 store 根完全可能被两个进程同时打开（一个机器人
/// 跑着，一个人在命令行里回滚），光靠 pid 去重的话，pid 回绕加上一个没清掉的残骸就会让
/// `create_new` 撞上一个**不是自己建的**文件。随机数来自 `RandomState`（每次 `new` 都重新
/// 播种），不值得为它引一个 `rand` 依赖。
fn tmp_suffix() -> String {
    use std::hash::{BuildHasher, Hasher};
    let r = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    format!("{}-{r:016x}", TMP_SEQ.fetch_add(1, Ordering::SeqCst))
}

/// 原子写：同目录临时文件 → fsync → rename → fsync 目录。
///
/// 三个细节都不是装饰：
/// - 临时文件必须和目标**同目录**，否则 rename 可能跨文件系统而退化成"复制+删除"，
///   原子性就没了；
/// - 文件 fsync 在 rename 之前：先 rename 后 fsync 的话，崩溃可能让新名字指向一份还没
///   落盘的内容（长度对、内容是零）；
/// - 目录 fsync 在 rename 之后：目录项本身也是数据，不同步的话崩溃后可能看见旧名字。
///   这一步失败只当作"没那么持久"，不回滚——文件已经在那儿了。
fn atomic_write(path: &Path, bytes: &[u8], fail_before_rename: bool) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other(format!("{} has no parent", path.display())))?;
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other(format!("{} has no file name", path.display())))?
        .to_string_lossy()
        .into_owned();
    let tmp = dir.join(format!("{name}.tmp-{}", tmp_suffix()));
    // `create_new`：临时文件必须是新的。这一步**在清理作用域之外**——撞上一个已经存在的
    // 同名文件意味着那个文件不是我们建的，而删掉一个别人正在写的临时文件，比留下它糟糕
    // 得多。只有自己建出来的临时文件才由自己负责删。
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)?;
    let write = || -> io::Result<()> {
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        if fail_before_rename {
            return Err(io::Error::other("injected fault before rename"));
        }
        fs::rename(&tmp, path)
    };
    match write() {
        Ok(()) => {
            if let Ok(d) = fs::File::open(dir) {
                let _ = d.sync_all();
            }
            Ok(())
        }
        Err(e) => {
            // 自己建的那个收干净：半截的临时文件既不该被下一次 `create_new` 撞上，也不该
            // 留在 `skills/` 里让人以为它是一份文档。
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}
