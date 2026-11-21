//! `skill_ledger.jsonl`：每一次知识变更一行。
//!
//! 为什么是 jsonl 而不是一个数据库：这份账本是**审计材料**，和世界日志的段文件一样，
//! 必须在没有本项目任何工具的情况下也能读（`tail -f`、`jq`、`git diff`）。一行一条、
//! 只追加、写完 fsync——一条写坏的尾行最多影响它自己。
//!
//! 每行的形状是合约（控制台、回滚脚本、以后的 RSI 二级都按它读）：
//!
//! ```jsonl
//! {"seq":1,"ts_unix_ms":1758412800123,"actor":"review:fast","op":"create","rel":"skills/nav/SKILL.md","prev_sha256":null,"new_sha256":"02890d…","bytes":23}
//! ```
//!
//! `prev_sha256` 是**改之前**那一版的内容哈希，它同时是 `.blobs/` 里那份备份的文件名：
//! 账本上的一行加上 `.blobs/`，就足以把任何一次变更倒回去，不需要额外的索引。

use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// 账本在 store 根下的文件名。
pub const LEDGER_FILE: &str = "skill_ledger.jsonl";

/// 一次变更的种类。线格式是小写字符串（`"create"` / `"update"` / `"rollback"`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Create,
    Update,
    /// 从 `.blobs/` 取回旧内容。它走的是和 `update` 完全相同的写路径（同样备份、同样
    /// 记账），只是在账本上分得出来——"这次变更是一次撤销"对读账本的人是关键信息。
    Rollback,
}

impl Op {
    pub fn as_str(&self) -> &'static str {
        match self {
            Op::Create => "create",
            Op::Update => "update",
            Op::Rollback => "rollback",
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 账本里的一行。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// 从 1 开始，每行加一。下一个 seq 是从文件末尾读出来的，不是内存里攒的——
    /// 账本的真相在磁盘上，进程重启、甚至换一个 store 实例都不该让 seq 倒回去。
    pub seq: u64,
    /// 墙上时钟的毫秒数。这里**故意**用墙钟而不是内核的单调时钟：这条记录是给人看的
    /// （"上周三那次 review 改了什么"），而单调时钟的零点是进程启动时刻，跨 run 没有意义。
    pub ts_unix_ms: u64,
    pub actor: String,
    pub op: Op,
    pub rel: String,
    /// 改之前那一版的 sha256，同时是 `.blobs/` 里备份的文件名；新建时是 `null`。
    #[serde(default)]
    pub prev_sha256: Option<String>,
    pub new_sha256: String,
    pub bytes: u64,
}

/// 追加一行并 fsync。
///
/// fsync 不是可选的：账本晚于文件本身落盘（见 [`crate::rsi::store`] 的提交顺序），
/// 如果连这一步都留在页缓存里，一次掉电就可能同时丢掉"文件变了"的唯一记录。
pub(crate) fn append(path: &Path, entry: &LedgerEntry) -> io::Result<()> {
    let mut line = serde_json::to_string(entry).map_err(io::Error::other)?;
    line.push('\n');
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(line.as_bytes())?;
    f.sync_all()
}

/// 读出整本账本。
///
/// 读得很宽容：空行跳过，**读不动的行也跳过**（只记一条 `warn!`）。原因是提交顺序留了
/// 一个已知的窗口——文件已经 rename、账本还没追加完——崩在这个窗口里会留下一条半截的
/// 尾行。宁可让读者看见"少一条记录"，也不要让一条坏行把整本账本连同回滚能力一起锁死。
pub(crate) fn read_all(path: &Path) -> io::Result<Vec<LedgerEntry>> {
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<LedgerEntry>(&line) {
            Ok(e) => out.push(e),
            Err(e) => tracing::warn!(
                path = %path.display(),
                line = i + 1,
                error = %e,
                "skipping an unreadable skill ledger line"
            ),
        }
    }
    Ok(out)
}
