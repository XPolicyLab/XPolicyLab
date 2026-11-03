//! 这个进程创建目录的唯一入口，以及它为什么必须存在。
//!
//! `mkdir` 的权限是 `mode & !umask`，而 `umask` 是**进程全局**的。内核里跑着第三方库的
//! 线程，而"临时改一下 umask 再还原"是 posix 代码里很常见的一种写法——iceoryx2 的事件
//! Listener 在 `bind()` 一条 unix datagram socket 时就是这么干的（`umask(0o7177)` → bind
//! → 还原）。那个窗口里别的线程 `create_dir_all` 出来的目录少掉 `x` 位，于是紧接着在里面
//! 开文件就是 `EACCES`：症状落在受害者身上（会话日志开不出来、run 目录写不进去），原因在
//! 别人的调用栈里，中间没有任何线索。这不是假想——Task 6 的第一版就是这么把 CI 弄红的。
//!
//! 那个具体的 Listener 已经不在这个进程里了（`oeh-rpc` 的 ingress 改成了轮询），所以这里
//! 防的是**下一个**：一个进程全局的开关，任何一个依赖都可能去拨。
//!
//! 所以：**自己建出来的每一层**目录，建完把权限再设一遍。0700 也正是这些目录该有的权限
//! （会话日志里有模型看到的一切，世界日志是审计材料）。
//!
//! 但收紧权限是**纵深防御，不是前置条件**：一个已经存在、进程写得进去却 chmod 不动的目录
//! （容器里常见的 root 建好、`0777`、挂给 uid 1000 的日志卷）绝不能因此让 `world-log` 或
//! `brain/attach` 起不来。所以只有 `create_dir_all` 失败才是硬错误，`chmod` 失败是一条带
//! 路径和当前权限位的 `warn!`。
//!
//! 非 unix 上是一次普通的 `create_dir_all`。

use std::io;
use std::path::{Path, PathBuf};

/// 建一个只有属主能进的目录（幂等，父目录一并建出来）。
///
/// 硬错误只有一个：目录建不出来。权限设不上（目录是别人的）只记一条 `warn!`——见模块头。
pub(crate) fn create_dir_private(dir: &Path) -> io::Result<()> {
    // 先记下哪些层是**这次**建出来的：`create_dir_all` 不会告诉你，而只收紧叶子的话，
    // `/var/log/oeh/runs` 这种一次建两层的情况里，中间那层留着 `0o777 & !umask`——那正好是
    // 这个模块声称挡住的东西。
    let created: Vec<PathBuf> = dir
        .ancestors()
        .take_while(|p| !p.exists())
        .map(|p| p.to_path_buf())
        .collect();
    std::fs::create_dir_all(dir)?;
    // 从上往下收紧（`ancestors` 是从下往上给的）。
    for p in created.iter().rev() {
        harden_or_warn(p);
    }
    // 目录本来就在：还是试一次（能收紧就收紧），但失败只是一条日志。
    if created.is_empty() {
        harden_or_warn(dir);
    }
    Ok(())
}

/// 收紧一个目录的权限；不成功便成仁——记下来，然后继续。
fn harden_or_warn(dir: &Path) {
    if let Err(e) = harden(dir) {
        tracing::warn!(
            dir = %dir.display(),
            mode = mode_of(dir),
            error = %e,
            "could not tighten directory permissions to 0700; leaving it as it is"
        );
    }
}

/// 一个目录当前的权限位，写进日志用（读不到就是 `None`）。
#[cfg(unix)]
fn mode_of(dir: &Path) -> Option<String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(dir)
        .ok()
        .map(|m| format!("{:o}", m.permissions().mode() & 0o7777))
}

#[cfg(not(unix))]
fn mode_of(_dir: &Path) -> Option<String> {
    None
}

/// 把一个**已经存在**的目录的权限压回 0700。
#[cfg(unix)]
pub(crate) fn harden(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
pub(crate) fn harden(_dir: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// 建出来的目录是 0700，而且里面开得出文件。
    ///
    /// 这条测试**不**去动进程的 `umask` 来模拟那个窗口：同一个二进制里的测试是并发跑的，
    /// 而一个改 umask 的测试正是这个模块在防的那种东西（它会让别的测试建出进不去的目录）。
    /// 权限本身就是判据——`harden` 一旦被拿掉，`create_dir_all` 留下的就是 `0o777 & !umask`，
    /// 在任何正常的 umask 下都不是 0700。
    #[test]
    fn a_created_directory_is_private_to_its_owner() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("a/b");
        create_dir_private(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "{mode:o}");
        std::fs::File::create(dir.join("probe")).unwrap();
        // 幂等：第二次调用不该失败，也不该把权限改成别的。
        create_dir_private(&dir).unwrap();
        assert_eq!(
            std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        // 中间那层也是我们建的，也得收紧——只管叶子的话，`a` 会留着 `0o777 & !umask`。
        assert_eq!(
            std::fs::metadata(tmp.path().join("a"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        // 本来就存在的那一层（tempdir 自己）不归我们管，权限保持原样。
        create_dir_private(tmp.path()).unwrap();
    }
}
