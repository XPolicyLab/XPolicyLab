//! 工具注册表：能力投影、recall 沙箱、批执行的并行/屏障顺序。

use std::sync::Arc;

use oeh_admission::capability::{CapabilityClass, CapabilitySpec, RiskLevel};
use oeh_brain::model::ToolCall;
use oeh_brain::rsi::{KnowledgeStore, ProtectedPaths};
use oeh_brain::tools::ToolRegistry;
use serde_json::{json, Value};

fn spec(name: &str, class: CapabilityClass) -> CapabilitySpec {
    CapabilitySpec {
        name: name.into(),
        class,
        risk_level: RiskLevel::Low,
        params_schema: json!({"type":"object"}),
        lease_ttl_s: 1.0,
        overridable: false,
    }
}

#[test]
fn capabilities_project_to_sorted_robot_tools_with_motion_exclusive() {
    let mut r = ToolRegistry::new();
    let _ = r.project_capabilities(&[
        spec("walk", CapabilityClass::Motion),
        spec("say", CapabilityClass::Speech),
    ]);
    r.add_ask_slow_brain();
    let names: Vec<String> = r.specs().iter().map(|s| s.name.clone()).collect();
    assert_eq!(names, vec!["ask_slow_brain", "robot_say", "robot_walk"]);
    assert!(!r.get("robot_walk").unwrap().0.concurrency_safe);
    assert!(r.get("robot_say").unwrap().0.concurrency_safe);
}

#[test]
fn recall_tools_are_sandboxed_to_the_root() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.md"), "hello world\nsecond line\n").unwrap();
    let mut r = ToolRegistry::new();
    r.add_recall(dir.path().to_path_buf());
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let turn = oeh_brain::tools::testing::dummy_turn();
        let (_, h) = r.get("recall_read").unwrap();
        let out = h.call(json!({"path": "notes.md"}), Some(&turn)).await;
        assert!(out.ok && out.result["content"].as_str().unwrap().contains("hello"));
        let out = h
            .call(json!({"path": "../../etc/passwd"}), Some(&turn))
            .await;
        assert!(!out.ok);
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "second"}), Some(&turn)).await;
        assert_eq!(out.result["matches"][0]["line"], 2);
    });
}

/// 沙箱挡的不只是 `..`：一个**指向外面的符号链接**是同一条检查挡下的（`canonicalize`
/// 先把它解开），而 `walk_dir` 每个条目都单独 canonicalize，所以遍历也看不见它。
///
/// 这两件事是 `Sandbox::resolve` / `walk_dir` 被写出来的全部理由，从前只有 `..` 有用例。
#[test]
fn a_symlink_that_points_outside_the_root_is_not_readable_or_walkable() {
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.md"), "SECRET\n").unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.md"), "mine\n").unwrap();
    // 沙箱里的一条链接指到沙箱外的那个文件，和一条指到沙箱外那个目录的链接。
    std::os::unix::fs::symlink(outside.path().join("secret.md"), dir.path().join("link.md"))
        .unwrap();
    std::os::unix::fs::symlink(outside.path(), dir.path().join("out")).unwrap();

    let mut r = ToolRegistry::new();
    r.add_recall(dir.path().to_path_buf());
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let turn = oeh_brain::tools::testing::dummy_turn();
        // 直接读那条链接：解开之后不在根里，拒。
        let (_, h) = r.get("recall_read").unwrap();
        let out = h.call(json!({"path": "link.md"}), Some(&turn)).await;
        assert!(!out.ok, "{out:?}");
        assert!(out.result["error"].as_str().unwrap().contains("escapes"));
        // 遍历也看不见它：grep 只在沙箱自己的文件里找。
        let (_, g) = r.get("recall_grep").unwrap();
        let out = g.call(json!({"pattern": "SECRET"}), Some(&turn)).await;
        assert_eq!(out.result["matches"], json!([]), "{out:?}");
        // glob 同理：沙箱里只有 notes.md 这一个**真**文件。
        let (_, gl) = r.get("recall_glob").unwrap();
        let out = gl.call(json!({"pattern": "**/*.md"}), Some(&turn)).await;
        assert_eq!(out.result["paths"], json!(["notes.md"]), "{out:?}");
    });
}

/// `recall_glob` 的沙箱级用例（C1 就住在这条路径上）：模式照常匹配，**过长的模式被拒**，
/// 而一个病态的 `**` 模式立刻返回而不是把一个不可 abort 的阻塞线程占住。
#[test]
fn recall_glob_matches_inside_the_sandbox_and_refuses_an_over_long_pattern() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("a/b/c")).unwrap();
    std::fs::write(dir.path().join("notes.md"), "x").unwrap();
    std::fs::write(dir.path().join("a/b/c/deep.md"), "x").unwrap();
    std::fs::write(dir.path().join("a/b/c/other.txt"), "x").unwrap();
    let mut r = ToolRegistry::new();
    r.add_recall(dir.path().to_path_buf());
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let turn = oeh_brain::tools::testing::dummy_turn();
        let (_, g) = r.get("recall_glob").unwrap();
        let glob = |p: &str| {
            let (g, turn) = (g.clone(), turn.clone());
            let p = p.to_string();
            async move { g.call(json!({ "pattern": p }), Some(&turn)).await }
        };
        let out = glob("**/*.md").await;
        assert!(out.ok);
        assert_eq!(out.result["paths"], json!(["a/b/c/deep.md", "notes.md"]));
        assert_eq!(out.result["truncated"], false);
        assert_eq!(glob("*.md").await.result["paths"], json!(["notes.md"]));
        assert_eq!(
            glob("a/b/*/*.txt").await.result["paths"],
            json!(["a/b/c/other.txt"])
        );
        // 过长的模式：和"缺 `pattern`"同一种拒法，一条 `ok:false` 回给模型。
        let out = glob(&"x".repeat(257)).await;
        assert!(!out.ok, "{out:?}");
        assert!(out.result["error"].as_str().unwrap().contains("too long"));
        // C1：18 个 `**` 打到真的沙箱上。旧匹配器在这里要几秒（还乘以走到的文件数），
        // 而它跑在一个 abort 不了的 `spawn_blocking` 线程上。
        let probe = std::time::Instant::now();
        let out = glob(&format!("{}/deep.md", vec!["**"; 18].join("/"))).await;
        assert!(out.ok, "{out:?}");
        assert_eq!(out.result["paths"], json!(["a/b/c/deep.md"]));
        let dt = probe.elapsed();
        assert!(dt < std::time::Duration::from_secs(1), "{dt:?}");
    });
}

#[tokio::test]
async fn run_batch_parallelises_safe_calls_and_barriers_exclusive_ones() {
    // 两个 speech（安全）+ 一个 motion（exclusive）+ 一个 speech：
    // 顺序必须是 [say1, say2 并行] → walk → say3。
    let calls = vec![
        ToolCall {
            id: "1".into(),
            name: "robot_say".into(),
            args: json!({"text": "a"}),
        },
        ToolCall {
            id: "2".into(),
            name: "robot_say".into(),
            args: json!({"text": "b"}),
        },
        ToolCall {
            id: "3".into(),
            name: "robot_walk".into(),
            args: json!({"vx": 0.1}),
        },
        ToolCall {
            id: "4".into(),
            name: "robot_say".into(),
            args: json!({"text": "c"}),
        },
    ];
    let (r, log) =
        oeh_brain::tools::testing::recording_registry(&["robot_say:safe", "robot_walk:exclusive"]);
    let turn = oeh_brain::tools::testing::dummy_turn();
    let outs = r.run_batch(calls, Some(&turn)).await;
    assert_eq!(
        outs.iter().map(|(c, _)| c.id.as_str()).collect::<Vec<_>>(),
        ["1", "2", "3", "4"]
    );
    let order = log.lock().clone(); // Vec<(id, "start"|"end")>
    let pos = |id: &str, ev: &str| {
        order
            .iter()
            .position(|(i, e)| i == id && e == ev)
            .unwrap_or_else(|| panic!("missing {id}/{ev} in {order:?}"))
    };
    assert!(pos("3", "start") > pos("1", "end") && pos("3", "start") > pos("2", "end"));
    assert!(pos("4", "start") > pos("3", "end"));
    // 1 和 2 有重叠
    assert!(pos("2", "start") < pos("1", "end") || pos("1", "start") < pos("2", "end"));
}

// ---------------------------------------------------------------------------
// 知识工具（RSI review fork 的那三个）
// ---------------------------------------------------------------------------

/// 一张只有 `skill_manage` / `memory_manage` / `read` 的表，actor 是 `review:fast`。
fn knowledge_registry(root: &std::path::Path, protected: ProtectedPaths) -> ToolRegistry {
    let store = Arc::new(KnowledgeStore::open(root, protected, None).unwrap());
    let mut r = ToolRegistry::new();
    r.add_knowledge(store, "review:fast".into());
    r
}

async fn call(r: &ToolRegistry, name: &str, args: Value) -> oeh_brain::tools::ToolOutcome {
    let (_, h) = r.get(name).unwrap_or_else(|| panic!("{name} 该在这张表里"));
    // `None`：review fork 没有 turn，这三个工具也从不需要一个。
    h.call(args, None).await
}

const SKILL: &str = "skills/greeting/SKILL.md";

/// 这张表**只有**三个工具：review fork 够不到世界（规格 §10.1 的"提的动作一律
/// auto-deny"在这里是"根本没有那些工具"）。两个写工具是屏障，`read` 可并行。
#[test]
fn the_knowledge_registry_has_exactly_three_tools_and_the_writes_are_barriers() {
    let dir = tempfile::tempdir().unwrap();
    let r = knowledge_registry(dir.path(), ProtectedPaths::default());
    let names: Vec<String> = r.specs().iter().map(|s| s.name.clone()).collect();
    assert_eq!(names, vec!["memory_manage", "read", "skill_manage"]);
    assert!(!r.get("skill_manage").unwrap().0.concurrency_safe);
    assert!(!r.get("memory_manage").unwrap().0.concurrency_safe);
    assert!(r.get("read").unwrap().0.concurrency_safe);
}

/// 写 → 读 → 带哈希改，以及 `append` 的读+拼+写。
#[test]
fn a_write_then_read_round_trips_the_hash_and_append_concatenates() {
    let dir = tempfile::tempdir().unwrap();
    let r = knowledge_registry(dir.path(), ProtectedPaths::default());
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let out = call(
            &r,
            "skill_manage",
            json!({"op": "create", "path": SKILL, "content": "# greeting\n"}),
        )
        .await;
        assert!(out.ok, "{:?}", out.result);
        assert_eq!(out.result["rel"], SKILL);
        let sha = out.result["sha256"].as_str().unwrap().to_string();

        let out = call(&r, "read", json!({"path": SKILL})).await;
        assert!(out.ok);
        assert_eq!(out.result["text"], "# greeting\n");
        assert_eq!(out.result["sha256"], sha.as_str());

        // `append` 自己去读当前哈希，所以不必带 `expect_sha256`；文件不存在时它就是新建。
        let out = call(
            &r,
            "memory_manage",
            json!({"op": "append", "path": "memory/MEMORY.md", "content": "- one\n"}),
        )
        .await;
        assert!(out.ok, "{:?}", out.result);
        let out = call(
            &r,
            "memory_manage",
            json!({"op": "append", "path": "memory/MEMORY.md", "content": "- two\n"}),
        )
        .await;
        assert!(out.ok, "{:?}", out.result);
        let out = call(&r, "read", json!({"path": "memory/MEMORY.md"})).await;
        assert_eq!(out.result["text"], "- one\n- two\n");

        // 带一个过时的哈希去 append → 陈旧读，不是"悄悄接在一份没读过的内容后面"。
        let out = call(
            &r,
            "memory_manage",
            json!({"op": "append", "path": "memory/MEMORY.md", "content": "- three\n",
                   "expect_sha256": sha}),
        )
        .await;
        assert!(!out.ok);
        assert!(
            out.result["error"].as_str().unwrap().contains("stale read"),
            "{:?}",
            out.result
        );
    });
}

/// 所有权：别人建的 skill 改不了，而错误原样变成一条工具错误（模型看得见是谁的）。
/// 记忆是共享的，同一个 actor 改得了别人建的那一份。
#[test]
fn someone_elses_skill_is_a_tool_error_while_memory_is_shared() {
    let dir = tempfile::tempdir().unwrap();
    // 先让**另一个** actor 建好两份文件，再用 review 的那张表去改。
    {
        let s = KnowledgeStore::open(dir.path(), ProtectedPaths::default(), None).unwrap();
        s.write(SKILL, "# theirs\n", "review:slow", None).unwrap();
        s.write("memory/MEMORY.md", "- theirs\n", "review:slow", None)
            .unwrap();
    }
    let r = knowledge_registry(dir.path(), ProtectedPaths::default());
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let sha = oeh_brain::rsi::sha256_hex(b"# theirs\n");
        let out = call(
            &r,
            "skill_manage",
            json!({"op": "update", "path": SKILL, "content": "# mine\n", "expect_sha256": sha}),
        )
        .await;
        assert!(!out.ok, "别人建的 skill 不该改得动");
        let err = out.result["error"].as_str().unwrap();
        assert!(
            err.contains("review:slow") && err.contains("review:fast"),
            "{err}"
        );

        let sha = oeh_brain::rsi::sha256_hex(b"- theirs\n");
        let out = call(
            &r,
            "memory_manage",
            json!({"op": "update", "path": "memory/MEMORY.md", "content": "- mine\n",
                   "expect_sha256": sha}),
        )
        .await;
        assert!(
            out.ok,
            "记忆是这台机器人的，不是某个 fork 的: {:?}",
            out.result
        );
    });
}

/// 受保护前缀、白名单外的形状、不认识的 `op`：三种都是**工具错误**，不是 panic，也不是
/// 一次悄悄成功的写。
#[test]
fn protected_paths_bad_shapes_and_unknown_ops_all_come_back_as_errors() {
    let dir = tempfile::tempdir().unwrap();
    // 整个 store 根都在保护里：一条形状完全合法的 rel 也得被挡下。
    let r = knowledge_registry(dir.path(), ProtectedPaths::new([dir.path().to_path_buf()]));
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let out = call(
            &r,
            "skill_manage",
            json!({"op": "create", "path": SKILL, "content": "x"}),
        )
        .await;
        assert!(!out.ok);
        assert!(
            out.result["error"].as_str().unwrap().contains("protected"),
            "{:?}",
            out.result
        );
        assert!(!dir.path().join(SKILL).exists(), "一个字节都不该落盘");

        // `op` 是两张表各自的白名单：`skill_manage` 没有 `append`，`memory_manage`
        // 没有 `create`。
        for (tool, args) in [
            (
                "skill_manage",
                json!({"op": "append", "path": SKILL, "content": "x"}),
            ),
            (
                "memory_manage",
                json!({"op": "create", "path": "memory/MEMORY.md", "content": "x"}),
            ),
        ] {
            let out = call(&r, tool, args).await;
            assert!(!out.ok, "{tool} 不该接受这个 op");
            assert!(
                out.result["error"]
                    .as_str()
                    .unwrap()
                    .contains("`op` must be one of"),
                "{:?}",
                out.result
            );
        }

        // 参数少了就说清楚，而不是拿一个空串去写。
        let out = call(&r, "skill_manage", json!({"op": "create"})).await;
        assert!(!out.ok);
        let out = call(&r, "read", json!({})).await;
        assert!(!out.ok);
        // 读一个不存在的文件是普通的 NotFound。
        let out = call(&r, "read", json!({"path": SKILL})).await;
        assert!(!out.ok);
        assert!(out.result["error"]
            .as_str()
            .unwrap()
            .contains("no such document"));
    });
}

/// 需要 turn 的工具在 turn 之外被调用：一条说得清是"没有 turn"的 `ok:false`，不是一次
/// 递给空处的提案。
///
/// review fork 走的正是这条路（`run_batch(.., None)`）——它那张表里根本没有提案类工具，
/// 而这是第三重保证：万一有一天有人往里加了一个，它也提不出案来。
#[tokio::test]
async fn a_tool_that_needs_a_turn_refuses_when_there_is_none() {
    let mut r = ToolRegistry::new();
    let _ = r.project_capabilities(&[spec("say", CapabilityClass::Speech)]);
    r.add_ask_slow_brain();
    for name in ["robot_say", "ask_slow_brain"] {
        let (_, h) = r.get(name).unwrap();
        let out = h.call(json!({"text": "hi", "q": "hi"}), None).await;
        assert!(!out.ok, "{name} 在没有 turn 时该失败");
        let err = out.result["error"].as_str().unwrap_or_default();
        assert!(err.contains("no live turn"), "{name}: {:?}", out.result);
    }
}
