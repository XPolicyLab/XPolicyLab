//! RSI 第一级存储（`oeh_brain::rsi`）的护栏测试。
//!
//! 规格 §10.1 把这些护栏的位置说死了：**在代码里，不在 prompt 里**。所以这份测试的每一条
//! 都不是在测"模型会不会听话"，而是在测"模型不听话时，存储层会不会替我们说不"：
//!
//! - (a) 新建要 `expect = None`，返回的哈希就是内容的 sha256，账本记一条 `create`；
//! - (b) read-before-write：哈希不对就不写，旧内容先进 `.blobs/` 才允许覆盖；
//! - (c) `created_by` 所有权：不是你建的文件你改不了；
//! - (d) `rollback` 走同一条写路径，账本上留得下痕迹；
//! - (e) `..`、绝对路径、白名单外的形状、符号链接 → `BadPath`，且一个字节都不落盘；
//! - (f) 受保护路径（profile / bundle / patch / wake / 规则 / prompt 模板）→ `Protected`
//!   且世界日志上有 `safety/tamper_attempt`；
//! - (g) rename 之前失败不留半截文件、不留半条账本；
//! - (h) `list` 递归。
//!
//! 修复轮次 1 又加了三条：`.meta` sidecar 不能和文档共用命名空间（一个 actor 因此永久
//! 占住另一个 actor 的 rel）、开店时给预置文件补种 `created_by: "seed"`、以及"所有权只管
//! skill，记忆是共享的"。

use std::fs;
use std::path::Path;
use std::sync::Arc;

use oeh_brain::rsi::{sha256_hex, FaultPoint, Kind, KnowledgeStore, Op, ProtectedPaths, RsiError};
use oeh_kernel::{ManualClock, WorldLog};

const REL: &str = "skills/nav/SKILL.md";
const V1: &str = "# nav\n\nGo to the door.\n";
/// 写死的期望值：证明 `sha256` 字段真的是内容的 SHA-256（`printf ... | sha256sum`），
/// 而不是某个自洽但不可跨工具核对的自定义摘要——账本和 `.blobs/` 的可审计性全靠这一点。
const V1_SHA: &str = "02890d174ca5a2328b243b9a3b69785b6587f8d97c9d418017058118f861abe3";
const V2: &str = "# nav\n\nGo to the door, then stop.\n";
const V2_SHA: &str = "6f7c20a21ffc30a9e205ff34041627d2a8e030ad9e0cd87718cab8ccb4d77aad";

fn open_at(root: &Path) -> KnowledgeStore {
    KnowledgeStore::open(root, ProtectedPaths::default(), None).unwrap()
}

/// 一个装着 v1 的 `skills/nav/SKILL.md`，owner 是 `review-fast`。
fn with_v1(root: &Path) -> KnowledgeStore {
    let s = open_at(root);
    s.write(REL, V1, "review-fast", None).unwrap();
    s
}

// ---------------------------------------------------------------------------
// (a) 新建
// ---------------------------------------------------------------------------

#[test]
fn creating_a_skill_needs_no_expected_hash_and_writes_one_create_line() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = open_at(&root);

    let doc = s.write(REL, V1, "review-fast", None).unwrap();
    assert_eq!(doc.rel, REL);
    assert_eq!(doc.text, V1);
    assert_eq!(doc.sha256, V1_SHA);
    assert_eq!(doc.sha256, sha256_hex(V1.as_bytes()));
    assert_eq!(doc.created_by.as_deref(), Some("review-fast"));

    // 落盘的是 Markdown 原文——`skills/**/SKILL.md` 是 grep 得到的东西，不是数据库。
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V1);
    let back = s.read(REL).unwrap();
    assert_eq!(back.text, V1);
    assert_eq!(back.sha256, V1_SHA);
    assert_eq!(back.created_by.as_deref(), Some("review-fast"));

    let led = s.ledger().unwrap();
    assert_eq!(led.len(), 1, "{led:?}");
    let e = &led[0];
    assert_eq!(e.seq, 1);
    assert_eq!(e.actor, "review-fast");
    assert_eq!(e.op, Op::Create);
    assert_eq!(e.rel, REL);
    assert_eq!(e.prev_sha256, None);
    assert_eq!(e.new_sha256, V1_SHA);
    assert_eq!(e.bytes, V1.len() as u64);
    // 墙上时钟的毫秒数：只断言"确实是个 2023 年之后的时间戳"，不断言具体值。
    assert!(e.ts_unix_ms > 1_700_000_000_000, "{}", e.ts_unix_ms);

    // 账本的线格式也是合约（控制台和审计脚本按行读它），逐键钉死。
    let raw = fs::read_to_string(root.join("skill_ledger.jsonl")).unwrap();
    assert_eq!(raw.lines().count(), 1, "{raw:?}");
    let v: serde_json::Value = serde_json::from_str(raw.lines().next().unwrap()).unwrap();
    assert_eq!(v["seq"], 1);
    assert_eq!(v["actor"], "review-fast");
    assert_eq!(v["op"], "create");
    assert_eq!(v["rel"], REL);
    assert_eq!(v["new_sha256"], V1_SHA);
    assert_eq!(v["bytes"], V1.len());
    assert!(v["ts_unix_ms"].is_u64());
}

// ---------------------------------------------------------------------------
// (b) read-before-write
// ---------------------------------------------------------------------------

#[test]
fn writing_over_a_file_requires_the_hash_you_last_read() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = with_v1(&root);

    // 不带哈希的覆盖 = 没读就写，拒绝。
    assert_eq!(
        s.write(REL, V2, "review-fast", None).unwrap_err(),
        RsiError::StaleRead {
            expected: String::new(),
            actual: V1_SHA.to_string()
        }
    );
    // 陈旧的哈希（别人在你读完之后改过了）：一样拒绝。
    let stale = sha256_hex(b"an older version");
    assert_eq!(
        s.write(REL, V2, "review-fast", Some(&stale)).unwrap_err(),
        RsiError::StaleRead {
            expected: stale.clone(),
            actual: V1_SHA.to_string()
        }
    );
    // 两次失败都没动磁盘。
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V1);
    assert_eq!(s.ledger().unwrap().len(), 1);

    // 带着当前哈希：放行。
    let doc = s.write(REL, V2, "review-fast", Some(V1_SHA)).unwrap();
    assert_eq!(doc.sha256, V2_SHA);
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V2);

    // 旧内容以内容寻址的方式留在 `.blobs/` 里——回滚要靠它。
    let blob = root.join(".blobs").join(V1_SHA);
    assert_eq!(fs::read_to_string(&blob).unwrap(), V1);

    let led = s.ledger().unwrap();
    assert_eq!(led.len(), 2, "{led:?}");
    assert_eq!(led[1].seq, 2);
    assert_eq!(led[1].op, Op::Update);
    assert_eq!(led[1].prev_sha256.as_deref(), Some(V1_SHA));
    assert_eq!(led[1].new_sha256, V2_SHA);
    assert_eq!(led[1].bytes, V2.len() as u64);
}

// ---------------------------------------------------------------------------
// (c) created_by 所有权
// ---------------------------------------------------------------------------

#[test]
fn a_fork_cannot_write_a_file_another_fork_created() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = with_v1(&root);

    // 哈希是对的、路径是对的——只是这不是它建的文件。
    assert_eq!(
        s.write(REL, V2, "review-slow", Some(V1_SHA)).unwrap_err(),
        RsiError::NotOwner {
            owner: "review-fast".to_string(),
            actor: "review-slow".to_string()
        }
    );
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V1);
    assert_eq!(s.ledger().unwrap().len(), 1);

    // 自己建的另一个文件当然写得了：拦的是越权，不是这个 actor。
    s.write("skills/greeting/SKILL.md", "# hi\n", "review-slow", None)
        .unwrap();
    assert_eq!(s.ledger().unwrap().len(), 2);

    // 所有权记在一个**扁平**的 sidecar 里，名字是 rel 的 sha256、内容里带着 rel。
    // 它和文档不共用命名空间——见 metadata_sidecars_cannot_squat_another_documents_rel。
    let meta = root
        .join(".meta")
        .join(format!("{}.json", sha256_hex(REL.as_bytes())));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&meta).unwrap()).unwrap();
    assert_eq!(v["rel"], REL);
    assert_eq!(v["created_by"], "review-fast");
    assert!(v["created_at_ms"].is_u64());
    assert_eq!(fs::read_dir(root.join(".meta")).unwrap().count(), 2);
}

// ---------------------------------------------------------------------------
// (c2) `.meta` 的命名空间（修复轮次 1 / Important #2）
// ---------------------------------------------------------------------------

/// 两个 rel 互为对方元数据的名字：`skills/nav/SKILL.md.json/SKILL.md` 是一条合法的 rel，
/// 而 `.meta/<rel>.json` 那一版的布局会让它把 `.meta/skills/nav/SKILL.md.json` 变成一个
/// **目录**，于是 `skills/nav/SKILL.md` 的元数据永远写不进去（rename 到目录上 = EISDIR），
/// 这条 rel 被一个 fork 永久占住；反过来先建也一样堵死另一条，而且还会留下一个没人清理的
/// 空目录。两个顺序都必须能建出来。
#[test]
fn metadata_sidecars_cannot_squat_another_documents_rel() {
    const SQUAT: &str = "skills/nav/SKILL.md.json/SKILL.md";
    for order in [[SQUAT, REL], [REL, SQUAT]] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("robots/g1");
        let s = open_at(&root);
        for rel in order {
            s.write(rel, V1, "review-fast", None)
                .unwrap_or_else(|e| panic!("{rel} (order {order:?}): {e}"));
        }
        assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V1);
        assert_eq!(fs::read_to_string(root.join(SQUAT)).unwrap(), V1);
        assert_eq!(s.ledger().unwrap().len(), 2);

        // `.meta/` 永远是扁平的：两个 `<64 位十六进制>.json`，一个子目录都没有。
        let mut metas: Vec<String> = fs::read_dir(root.join(".meta"))
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                assert!(e.file_type().unwrap().is_file(), "{:?}", e.path());
                e.file_name().to_string_lossy().into_owned()
            })
            .collect();
        metas.sort();
        assert_eq!(metas.len(), 2, "{metas:?}");
        for name in &metas {
            let hex = name.strip_suffix(".json").unwrap();
            assert!(
                hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()),
                "{name}"
            );
        }
        assert!(metas.contains(&format!("{}.json", sha256_hex(REL.as_bytes()))));
        assert!(metas.contains(&format!("{}.json", sha256_hex(SQUAT.as_bytes()))));

        // 文档**自己**的命名空间里的那条冲突仍然被挡下，而且是 BadPath 而不是一个
        // 含糊的 Io：`skills/nav/SKILL.md` 已经是文件，就不能再当目录用。
        let nested = "skills/nav/SKILL.md/SKILL.md";
        assert!(matches!(s.read(nested), Err(RsiError::BadPath(_))));
        assert!(
            matches!(
                s.write(nested, V2, "review-fast", None),
                Err(RsiError::BadPath(_))
            ),
            "{:?}",
            s.write(nested, V2, "review-fast", None)
        );
        assert_eq!(s.ledger().unwrap().len(), 2);
        assert!(tmp_files(&root).is_empty(), "{:?}", tmp_files(&root));
    }
}

// ---------------------------------------------------------------------------
// (c3) 预置文件的主人是 "seed"；记忆是共享的（修复轮次 1 / 控制器预裁定）
// ---------------------------------------------------------------------------

#[test]
fn preexisting_documents_are_seeded_with_a_named_owner() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    // profile 预置的样子：文件先躺在那儿，store 还没开过。
    fs::create_dir_all(root.join("skills/nav")).unwrap();
    fs::write(root.join(REL), V1).unwrap();
    fs::create_dir_all(root.join("memory")).unwrap();
    fs::write(root.join("memory/MEMORY.md"), "# memory\n").unwrap();

    let s = open_at(&root);
    assert_eq!(s.read(REL).unwrap().created_by.as_deref(), Some("seed"));
    assert_eq!(
        s.read("memory/MEMORY.md").unwrap().created_by.as_deref(),
        Some("seed")
    );
    // 预置的 skill 有名有姓地属于 seed，fork 改不了——而不是撞上一个莫名其妙的 unknown。
    assert_eq!(
        s.write(REL, V2, "review-fast", Some(V1_SHA)).unwrap_err(),
        RsiError::NotOwner {
            owner: "seed".to_string(),
            actor: "review-fast".to_string()
        }
    );
    // 补种不入账：它记录的是既成事实，不是一次变更。
    assert!(s.ledger().unwrap().is_empty());
    // 幂等：再开一次不会把 created_by 冲掉。
    let s2 = open_at(&root);
    assert_eq!(s2.read(REL).unwrap().created_by.as_deref(), Some("seed"));
}

#[test]
fn memory_is_shared_by_design_while_skills_stay_owned() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = open_at(&root);

    let m1 = s
        .write("memory/MEMORY.md", "# m\n", "review-fast", None)
        .unwrap();
    // 另一个 actor 照样写得了：记忆是这台机器人的，不是某一个 fork 的私产。
    let m2 = s
        .write(
            "memory/MEMORY.md",
            "# m2\n",
            "review-slow",
            Some(&m1.sha256),
        )
        .unwrap();
    // 但 created_by 还是第一个建它的人（所有权记录不因共享而改写）。
    assert_eq!(m2.created_by.as_deref(), Some("review-fast"));
    // topics 同理，回滚也一样开放。
    let t = s
        .write("memory/topics/people.md", "# p\n", "review-fast", None)
        .unwrap();
    s.write(
        "memory/topics/people.md",
        "# p2\n",
        "review-slow",
        Some(&t.sha256),
    )
    .unwrap();
    let back = s
        .rollback("memory/MEMORY.md", &m1.sha256, "review-slow")
        .unwrap();
    assert_eq!(back.text, "# m\n");

    // 账本上每一行记的都是真正动手的那个 actor——共享不等于查不出是谁改的。
    let led = s.ledger().unwrap();
    assert_eq!(
        led.iter().map(|e| e.actor.as_str()).collect::<Vec<_>>(),
        [
            "review-fast",
            "review-slow",
            "review-fast",
            "review-slow",
            "review-slow"
        ]
    );
    assert_eq!(led[4].op, Op::Rollback);

    // skill 仍然是有主的。
    s.write(REL, V1, "review-fast", None).unwrap();
    assert!(matches!(
        s.write(REL, V2, "review-slow", Some(V1_SHA)),
        Err(RsiError::NotOwner { .. })
    ));
    assert!(matches!(
        s.rollback(REL, V1_SHA, "review-slow"),
        Err(RsiError::NotOwner { .. })
    ));
}

// ---------------------------------------------------------------------------
// (d) rollback
// ---------------------------------------------------------------------------

#[test]
fn rollback_restores_a_previous_version_through_the_same_write_path() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = with_v1(&root);
    s.write(REL, V2, "review-fast", Some(V1_SHA)).unwrap();

    // 不是 owner 就回滚不了（回滚也是写）。
    assert_eq!(
        s.rollback(REL, V1_SHA, "review-slow").unwrap_err(),
        RsiError::NotOwner {
            owner: "review-fast".to_string(),
            actor: "review-slow".to_string()
        }
    );
    // 没有备份就回滚不了——`rollback` 只能回到账本上真实存在过的版本。
    let ghost = sha256_hex(b"never written");
    assert!(matches!(
        s.rollback(REL, &ghost, "review-fast").unwrap_err(),
        RsiError::NotFound(_)
    ));
    assert_eq!(s.ledger().unwrap().len(), 2);

    let doc = s.rollback(REL, V1_SHA, "review-fast").unwrap();
    assert_eq!(doc.text, V1);
    assert_eq!(doc.sha256, V1_SHA);
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V1);

    let led = s.ledger().unwrap();
    assert_eq!(led.len(), 3, "{led:?}");
    assert_eq!(led[2].seq, 3);
    assert_eq!(led[2].op, Op::Rollback);
    assert_eq!(led[2].actor, "review-fast");
    assert_eq!(led[2].rel, REL);
    assert_eq!(led[2].prev_sha256.as_deref(), Some(V2_SHA));
    assert_eq!(led[2].new_sha256, V1_SHA);
    // 被回滚掉的那一版也进了备份：回滚本身可以再被回滚。
    assert_eq!(
        fs::read_to_string(root.join(".blobs").join(V2_SHA)).unwrap(),
        V2
    );
}

// ---------------------------------------------------------------------------
// (e) 路径
// ---------------------------------------------------------------------------

#[test]
fn escaping_paths_are_refused_before_anything_touches_the_disk() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let outside = tmp.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("secret.md"), "secret\n").unwrap();
    let s = open_at(&root);

    let bad = [
        // 句法层：爬出根目录。
        "../profiles/x.yaml",
        "skills/../../x",
        "memory/topics/../../../x.md",
        "/etc/passwd",
        // 句法层：形状不在三类白名单里（skills/**/SKILL.md、memory/MEMORY.md、
        // memory/topics/*.md）。技能目录里放的别的文件不归 RSI 写。
        "skills/nav/notes.md",
        "skills/SKILL.md",
        "memory/topics/deep/x.md",
        "memory/OTHER.md",
        "notes.md",
        "",
        // 句法层：隐藏名——`.blobs` / `.meta` / 账本自己不能被当成文档改。
        "skills/.blobs/SKILL.md",
        "memory/topics/.hidden.md",
        ".meta/x.json",
        ".blobs/deadbeef",
        "skill_ledger.jsonl",
        // 句法层：字符集之外的字节（NUL 是最要命的那个——下游的 C 字符串会在这里截断）。
        "skills/na\0v/SKILL.md",
        "skills/nav dir/SKILL.md",
    ];
    for rel in bad {
        assert!(
            matches!(s.read(rel), Err(RsiError::BadPath(_))),
            "read {rel:?} should be BadPath, got {:?}",
            s.read(rel)
        );
        assert!(
            matches!(
                s.write(rel, "x\n", "review-fast", None),
                Err(RsiError::BadPath(_))
            ),
            "write {rel:?} should be BadPath"
        );
    }

    // 物理层：根目录里被人塞了一个指向外面的符号链接。句法上它完全合法，
    // 只有解析真实路径才看得出来。
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(
            outside.join("secret.md"),
            root.join("memory/topics/evil.md"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&outside, root.join("skills/evil")).unwrap();
        for rel in ["memory/topics/evil.md", "skills/evil/SKILL.md"] {
            assert!(
                matches!(s.read(rel), Err(RsiError::BadPath(_))),
                "read {rel:?} should be BadPath"
            );
            assert!(
                matches!(
                    s.write(rel, "pwned\n", "review-fast", None),
                    Err(RsiError::BadPath(_))
                ),
                "write {rel:?} should be BadPath"
            );
        }
    }

    // 一个字节都没落盘：外面的文件没变，账本是空的，根目录里没有临时文件残骸。
    assert_eq!(
        fs::read_to_string(outside.join("secret.md")).unwrap(),
        "secret\n"
    );
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 1);
    assert!(s.ledger().unwrap().is_empty());
    assert!(!tmp.path().join("profiles").exists());
    assert!(!root.join("x").exists());
    assert!(tmp_files(&root).is_empty(), "{:?}", tmp_files(&root));
}

// ---------------------------------------------------------------------------
// (f) 受保护路径
// ---------------------------------------------------------------------------

#[test]
fn writing_into_a_protected_directory_is_refused_and_logged_as_tampering() {
    let tmp = tempfile::tempdir().unwrap();
    // `tmp.path()` 要先 canonicalize 再往下接：`KnowledgeStore::open` 把根解析过一次
    // （store.rs 的"参照本身必须是解析过的"），所以 `safety/tamper_attempt.path` 报的是
    // **解析后**的绝对路径。macOS 上 `tempfile::tempdir()` 给的是 `/var/folders/...`，
    // 而它解析出来是 `/private/var/folders/...`——不从一个解析过的前缀出发的话，下面
    // 那条 `path` 断言会只因为这个前缀就不成立。Linux 上两者本来就相等，这一行是恒等
    // 变换；断言本身一个字都没放松。
    let repo = tmp.path().canonicalize().unwrap().join("repo");
    for d in ["profiles", "bundles", "wake", "prompts"] {
        fs::create_dir_all(repo.join(d)).unwrap();
    }
    let profile = repo.join("profiles/g1-sim.yaml");
    fs::write(&profile, "entries: []\n").unwrap();
    let protected = ProtectedPaths::from_profile(&profile, &repo);
    assert!(protected.covers(&profile));
    assert!(protected.covers(&repo.join("bundles/base.yaml")));
    assert!(protected.covers(&repo.join("wake/fast.yaml")));
    assert!(protected.covers(&repo.join("patch/whatever.jsonl")));
    assert!(protected.covers(&repo.join("prompts/system.md")));
    // `rules/` 这次没建：名字没被布局固定下来的那两个只在存在时才保护，凭空保护会误伤。
    assert!(!protected.covers(&repo.join("rules/estop.yaml")));
    assert!(!protected.covers(&repo.join("robots/g1/skills/nav/SKILL.md")));

    // store 的根被（有意地）架在受保护目录里：rel 完全合法，解析出来的绝对路径却踩在
    // bundle 上。这正是"护栏不能只看 rel 的形状"的那个case。
    let root = repo.join("bundles/robots/g1");
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let s = KnowledgeStore::open(&root, protected, Some(log.clone())).unwrap();

    let err = s.write(REL, V1, "review-fast", None).unwrap_err();
    assert!(matches!(err, RsiError::Protected(_)), "{err:?}");
    assert!(!root.join(REL).exists());
    assert!(s.ledger().unwrap().is_empty());
    assert!(tmp_files(&root).is_empty());

    let evs = log.since(0).unwrap();
    assert_eq!(evs.len(), 1, "{evs:?}");
    let ev = &evs[0];
    assert_eq!(ev.kind.as_str(), "safety/tamper_attempt");
    assert_eq!(ev.source, "rsi");
    assert_eq!(ev.salience, 200);
    assert_eq!(ev.payload["actor"], "review-fast");
    assert_eq!(ev.payload["op"], "write");
    assert_eq!(
        ev.payload["path"].as_str().unwrap(),
        root.join(REL).display().to_string()
    );

    // 回滚同样被挡，同样留证据。
    let err = s.rollback(REL, V1_SHA, "review-fast").unwrap_err();
    assert!(matches!(err, RsiError::Protected(_)), "{err:?}");
    let evs = log.since(0).unwrap();
    assert_eq!(evs.len(), 2, "{evs:?}");
    assert_eq!(evs[1].payload["op"], "rollback");

    // 越界的 rel 也是一次篡改尝试（review fork 写 `../bundles/x.yaml` 就走这条）。
    let err = s
        .write("../bundles/x.yaml", "x\n", "review-fast", None)
        .unwrap_err();
    assert!(matches!(err, RsiError::BadPath(_)), "{err:?}");
    let evs = log.since(0).unwrap();
    assert_eq!(evs.len(), 3, "{evs:?}");
    assert_eq!(evs[2].kind.as_str(), "safety/tamper_attempt");
    assert_eq!(evs[2].payload["path"], "../bundles/x.yaml");

    // 只读不是篡改：`read` 不记事件（它本来就对所有 actor 开放）。
    let _ = s.read(REL);
    let _ = s.read("../bundles/x.yaml");
    assert_eq!(log.since(0).unwrap().len(), 3);
}

// ---------------------------------------------------------------------------
// (g) 写到一半失败
// ---------------------------------------------------------------------------

#[test]
fn a_failure_before_the_rename_leaves_no_half_written_file() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = with_v1(&root);

    s.__inject_fault(FaultPoint::BeforeRename);
    let err = s.write(REL, V2, "review-fast", Some(V1_SHA)).unwrap_err();
    assert!(matches!(err, RsiError::Io(_)), "{err:?}");

    // 旧内容原封不动，没有半截文件，也没有临时文件残骸，账本没长。
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V1);
    assert_eq!(
        fs::read_dir(root.join("skills/nav"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        vec!["SKILL.md".to_string()]
    );
    assert!(tmp_files(&root).is_empty(), "{:?}", tmp_files(&root));
    assert_eq!(s.ledger().unwrap().len(), 1);

    // 注入点关掉之后照常能写：失败是失败，不是把 store 弄坏了。
    s.__inject_fault(FaultPoint::None);
    s.write(REL, V2, "review-fast", Some(V1_SHA)).unwrap();
    assert_eq!(fs::read_to_string(root.join(REL)).unwrap(), V2);
    assert_eq!(s.ledger().unwrap().len(), 2);
}

// ---------------------------------------------------------------------------
// (h) list
// ---------------------------------------------------------------------------

#[test]
fn list_walks_the_skill_tree_recursively() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("robots/g1");
    let s = open_at(&root);
    assert!(s.list(Kind::Skill).unwrap().is_empty());
    assert!(s.list(Kind::Memory).unwrap().is_empty());
    assert!(s.list(Kind::Topic).unwrap().is_empty());

    for rel in [
        REL,
        "skills/nav/door/SKILL.md",
        "skills/greeting/SKILL.md",
        "memory/MEMORY.md",
        "memory/topics/people.md",
        "memory/topics/places.md",
    ] {
        s.write(rel, "# x\n", "review-fast", None).unwrap();
    }

    assert_eq!(
        s.list(Kind::Skill).unwrap(),
        vec![
            "skills/greeting/SKILL.md".to_string(),
            "skills/nav/SKILL.md".to_string(),
            "skills/nav/door/SKILL.md".to_string(),
        ]
    );
    assert_eq!(
        s.list(Kind::Memory).unwrap(),
        vec!["memory/MEMORY.md".to_string()]
    );
    assert_eq!(
        s.list(Kind::Topic).unwrap(),
        vec![
            "memory/topics/people.md".to_string(),
            "memory/topics/places.md".to_string()
        ]
    );
}

#[test]
fn reading_a_missing_document_is_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    let s = open_at(&tmp.path().join("robots/g1"));
    assert!(matches!(s.read(REL), Err(RsiError::NotFound(_))));
    // 不存在的文件带 expect 写 = 你以为自己在覆盖某一版，其实那一版不存在。
    assert_eq!(
        s.write(REL, V1, "review-fast", Some(V1_SHA)).unwrap_err(),
        RsiError::StaleRead {
            expected: V1_SHA.to_string(),
            actual: String::new()
        }
    );
    assert!(s.ledger().unwrap().is_empty());
}

/// 根目录树里所有 `*.tmp-*` 残骸（原子写失败后不该剩下任何一个）。
fn tmp_files(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() {
                stack.push(e.path());
            } else if name.contains(".tmp-") {
                out.push(name);
            }
        }
    }
    out
}
