//! RSI review fork 的端到端：真实内核 + `brain-native{scripted}` + 真实
//! [`KnowledgeStore`]。
//!
//! review 是这个脑唯一能**改自己**的入口，所以这份测试问的全是"它改不动什么"：
//!
//! - (a) `every_turns: 2` → 第二个 turn 的 Maintenance 相位里跑一次 review，写下一个
//!   skill；账本一行、`created_by == "review:fast"`；**下一个 turn 的 system 前缀因此
//!   不同**（`# Skills` 段变了 = prompt 缓存自然失效），而那个 turn 的工具表里没有
//!   `skill_manage`（review 的工具只在 review 里存在）；
//! - (b) review 想写 `../bundles/x.yaml` → 工具错误 + `safety/tamper_attempt`，一个
//!   字节都不落盘；
//! - (c) read-before-write：不带哈希的 `update` 是 `StaleRead`，先 `read` 再带上哈希
//!   才写得进去；
//! - (d) `on_session_end` 在 detach 时补一次 review；`every_turns: 0` +
//!   `on_session_end: false` 则一条 `brain/review` 都不该有；
//! - (e) review 的模型流断掉 → `brain/review{error}`，而触发它的那个 turn 照常 done。
//!
//! **没有一处用 sleep 当判别信号**：等事件用日志观察者，等"第 n 个 turn 收尾了"数的是
//! `brain/turn_end` 的条数。唯一的墙钟是 `wait_for` 的 10 s 兜底和 (d) 的否定断言窗口
//! ——前者只让坏掉的测试失败，后者的判别信号仍然是"事件出现即失败"。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use oeh_admission as _;
use oeh_brain::builtin::scripted_model::ScriptedModel;
use oeh_brain::model::ModelRequest;
use oeh_brain::rsi::{sha256_hex, KnowledgeStore, Op, ProtectedPaths};
use oeh_kernel::profile::EntryRow;
use oeh_kernel::{BrainStatus, Brains, FiberId, Kernel, KernelHandle, WorldLog};
use oeh_proto::{EventDraft, EventKind, ServiceName, WorldEvent};
use serde_json::{json, Value};

// ---------------------------------------------------------------- 脚手架

/// 只让 `obs/gesture` 唤醒。`safety/*: ignore` 很要紧：(b) 里的
/// `safety/tamper_attempt` 不该顺手再开一个 turn。
const WAKE_YAML: &str = "\
default: accumulate
rules:
  obs/gesture: wake
  brain/*: ignore
  act/*: ignore
  health/*: ignore
  safety/*: ignore
  plugin/*: ignore
budget: { max_consecutive_wakes: 32, min_interval_ms: 0 }
";

const PERSONA: &str = "you are a fast brain";
const SKILL: &str = "skills/greeting/SKILL.md";

struct Fixture {
    tmp: tempfile::TempDir,
    script: PathBuf,
    wake: PathBuf,
}

impl Fixture {
    fn new(script_yaml: &str) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let script = tmp.path().join("script.yaml");
        std::fs::write(&script, script_yaml).unwrap();
        let wake = tmp.path().join("wake.yaml");
        std::fs::write(&wake, WAKE_YAML).unwrap();
        Fixture { tmp, script, wake }
    }

    fn home(&self) -> PathBuf {
        self.tmp.path().join("oeh-home")
    }

    fn store_root(&self) -> PathBuf {
        self.home().join("robots").join("g1")
    }

    /// 测试这一侧的只读句柄。开在同一个根上：`open` 对已经有 `.meta` 的文件不做任何事。
    fn store(&self) -> KnowledgeStore {
        KnowledgeStore::open(self.store_root(), ProtectedPaths::default(), None).unwrap()
    }

    fn rows(&self, review: Value, max_steps: u32) -> Vec<EntryRow> {
        self.rows_with(review, max_steps, 5000)
    }

    /// `step_timeout_ms` 可调的那一版。(f) 要的正是 **0**——"单步不设限"，也就是
    /// Important #1 说的那个配置：内核的 turn 超时扫不到 Maintenance，所以在
    /// `review.timeout_ms` 之前，那一段根本没有任何界。
    fn rows_with(&self, review: Value, max_steps: u32, step_timeout_ms: u64) -> Vec<EntryRow> {
        self.rows_full(review, max_steps, step_timeout_ms, 0)
    }

    /// 内核 `turn_timeout_ms` 也可调的那一版。(i) 要的正是**非 0**：别的用例全配 0，
    /// 而那恰恰关掉了唯一能看见"review 有没有真的进 Maintenance"的那道机制。
    fn rows_full(
        &self,
        review: Value,
        max_steps: u32,
        step_timeout_ms: u64,
        turn_timeout_ms: u64,
    ) -> Vec<EntryRow> {
        let tmp = self.tmp.path();
        vec![
            EntryRow::new("clock"),
            EntryRow::new("world-log"),
            EntryRow::new("state-registry"),
            EntryRow::new("robot-fsm"),
            EntryRow::new("brains").with_config(json!({
                "session_dir": tmp.join("sessions"),
                // 多数用例配 0：review 跑在 Maintenance 里（不受 turn 超时约束），
                // 别让内核的超时扫描给它们添第二个变量。(i) 专门配非 0 去验那条性质。
                "turn_timeout_ms": turn_timeout_ms,
            })),
            EntryRow::new("admission").with_config(json!({
                "required_rules": [],
                "estop_flag": tmp.join("estop"),
            })),
            EntryRow::new("brain-native").with_config(json!({
                "brain_id": "fast",
                "wake": self.wake,
                "persona": PERSONA,
                "model": {"scripted": self.script},
                "max_steps": max_steps,
                "step_timeout_ms": step_timeout_ms,
                "oeh_home": self.home(),
                "robot_id": "g1",
                // 明确给一组受保护前缀：默认值按进程 cwd 推，而测试的 cwd 是 crate 目录。
                "protected_paths": [tmp.join("guarded")],
                "review": review,
            })),
        ]
    }
}

struct Booted {
    h: KernelHandle,
    run: tokio::task::JoinHandle<Result<(), oeh_kernel::BootError>>,
    log: Arc<WorldLog>,
    brains: Arc<Brains>,
    model: Arc<ScriptedModel>,
}

fn boot(rows: Vec<EntryRow>) -> Booted {
    let k = Kernel::boot_with(rows, &|id| {
        oeh_kernel::find_factory(id).map(|f| (f.build)())
    })
    .unwrap();
    assert!(k.report().failed.is_empty(), "{:?}", k.report());
    assert!(k.report().pending.is_empty(), "{:?}", k.report());
    let reg = k.registry().clone();
    let log = reg
        .get::<WorldLog>(&ServiceName::from("world-log"))
        .unwrap();
    let brains = reg.get::<Brains>(&ServiceName::from("brains")).unwrap();
    let model = reg
        .get::<ScriptedModel>(&ServiceName::from("brain-native.fast.model"))
        .unwrap();
    let h = k.handle();
    let run = tokio::spawn(k.run());
    Booted {
        h,
        run,
        log,
        brains,
        model,
    }
}

impl Booted {
    fn wake(&self) {
        self.h
            .emit(EventDraft::new(
                EventKind::new("obs/gesture").unwrap(),
                "test",
                json!({"gesture": "wave"}),
            ))
            .unwrap();
    }

    fn events(&self) -> Vec<WorldEvent> {
        self.log.since(0).unwrap_or_default()
    }

    async fn stop(self) {
        self.h.shutdown().unwrap();
        self.run.await.unwrap().unwrap();
    }
}

/// 等世界日志里出现第一条满足 `pred` 的事件（注册观察者之前已经进去的也算）。
async fn wait_for(
    log: &Arc<WorldLog>,
    pred: impl Fn(&WorldEvent) -> bool + Send + Sync + 'static,
) -> WorldEvent {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let pred = Arc::new(pred);
    let p = pred.clone();
    let dispose = log.observe(Arc::new(move |e: &WorldEvent| {
        if p(e) {
            let _ = tx.send(e.clone());
        }
    }));
    let found = match log
        .since(0)
        .unwrap_or_default()
        .into_iter()
        .find(|e| pred(e))
    {
        Some(e) => e,
        None => tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("timed out waiting for a world event")
            .expect("world log observer channel closed"),
    };
    dispose();
    found
}

/// 等到世界日志里累计出现 `n` 条 `brain/turn_end`。判别信号是事件本身，不是时间。
async fn turn_ends(log: &Arc<WorldLog>, n: usize) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let dispose = log.observe(Arc::new(move |e: &WorldEvent| {
        if e.kind.as_str() == "brain/turn_end" {
            let _ = tx.send(e.seq);
        }
    }));
    let mut seen: Vec<u64> = log
        .since(0)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.kind.as_str() == "brain/turn_end")
        .map(|e| e.seq)
        .collect();
    while seen.len() < n {
        let seq = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("timed out waiting for brain/turn_end")
            .expect("world log observer channel closed");
        // 观察者与上面那遍 `since(0)` 会重叠，按 seq 去重。
        if !seen.contains(&seq) {
            seen.push(seq);
        }
    }
    dispose();
}

/// 在 `window` 内反复确认某件事没有发生。判别信号仍然是"事件出现即失败"。
async fn stays_absent(log: &Arc<WorldLog>, what: &str, kind: &str, window: Duration) {
    let deadline = tokio::time::Instant::now() + window;
    loop {
        if let Some(e) = log
            .since(0)
            .unwrap_or_default()
            .into_iter()
            .find(|e| e.kind.as_str() == kind)
        {
            panic!("{what} 不该出现，却有一条 {} (seq {})", e.kind, e.seq);
        }
        if tokio::time::Instant::now() >= deadline {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn fiber_id(log: &Arc<WorldLog>, plugin: &str) -> FiberId {
    let want = plugin.to_string();
    let ev = wait_for(log, move |e| {
        e.kind.as_str() == "plugin/state" && e.payload["plugin"] == want.as_str()
    })
    .await;
    FiberId(ev.payload["fiber_id"].as_u64().expect("fiber_id") as u32)
}

/// 按出现顺序的 turn trace_id。
fn traces(evs: &[WorldEvent]) -> Vec<String> {
    evs.iter()
        .filter(|e| e.kind.as_str() == "brain/turn_start")
        .filter_map(|e| e.trace_id.clone())
        .collect()
}

fn of<'a>(evs: &'a [WorldEvent], trace: &str, kind: &str) -> Vec<&'a WorldEvent> {
    evs.iter()
        .filter(|e| e.trace_id.as_deref() == Some(trace) && e.kind.as_str() == kind)
        .collect()
}

fn only<'a>(evs: &'a [WorldEvent], kind: &str) -> &'a WorldEvent {
    let hits: Vec<&WorldEvent> = evs.iter().filter(|e| e.kind.as_str() == kind).collect();
    assert_eq!(hits.len(), 1, "{kind} 应该恰好一条，实际 {}", hits.len());
    hits[0]
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|x| x.as_str().map(str::to_string))
        .collect()
}

/// 剧本模型收到过的请求，分成"普通 turn 的"和"review 的"——review 的工具表里有
/// `skill_manage`，普通 turn 的没有，这正是 (a) 要断言的那条边界。
fn split_requests(m: &ScriptedModel) -> (Vec<ModelRequest>, Vec<ModelRequest>) {
    m.requests()
        .into_iter()
        .partition(|r| !r.tools.iter().any(|t| t.name == "skill_manage"))
}

// ---------------------------------------------------------------- (a)

const GREETING: &str = "# greeting\n\n见到人先打招呼。\n";

/// 两个普通 turn 之后，第二个 turn 的 Maintenance 相位里跑一次 review：skill 落盘、
/// 账本一行、`created_by` 是 `review:fast`；第三个 turn 的前缀因此不同，而它的工具表
/// 里没有 `skill_manage`。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_two_turns_the_review_writes_a_skill_and_the_next_prefix_changes() {
    let fx = Fixture::new(&format!(
        "turns:\n  \
           - text: \"第一个 turn\"\n  \
           - text: \"第二个 turn\"\n  \
           - text: \"第三个 turn\"\n\
         review:\n  \
           - text: \"把打招呼记成一个 skill\"\n    \
             tool_calls:\n      \
               - name: skill_manage\n        \
                 args: {{ op: create, path: {SKILL}, content: \"{}\" }}\n  \
           - text: \"记完了\"\n",
        GREETING.replace('\n', "\\n")
    ));
    let b = boot(fx.rows(
        json!({"every_turns": 2, "on_session_end": false, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    b.wake();
    turn_ends(&b.log, 2).await;
    b.wake();
    turn_ends(&b.log, 3).await;
    // 维护相位收了尾：turn 关掉之后脑回到 Idle，而不是卡在 Maintenance 上。
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));

    let evs = b.events();
    let model = b.model.clone();
    b.stop().await;

    // review 恰好一条，挂在**第二个** turn 的 trace 上，而且排在那个 turn 的
    // `brain/turn_end` 前面（它跑在 done 之前的 Maintenance 相位里）。
    let tr = traces(&evs);
    assert_eq!(tr.len(), 3, "应该三个 turn: {tr:?}");
    let review = only(&evs, "brain/review");
    assert_eq!(review.trace_id.as_deref(), Some(tr[1].as_str()));
    assert_eq!(review.payload["trace_id"], tr[1].as_str());
    assert_eq!(review.payload["brain_id"], "fast");
    assert_eq!(review.payload["trigger"], "every_turns");
    assert_eq!(review.payload["steps"], 2);
    assert_eq!(strs(&review.payload["writes"]), vec![SKILL.to_string()]);
    assert_eq!(review.payload["ledger_seq"], 1);
    assert_eq!(review.salience, 100);
    let end2 = of(&evs, &tr[1], "brain/turn_end");
    assert_eq!(end2.len(), 1);
    assert!(review.seq < end2[0].seq, "review 跑在 done 之前");
    assert_eq!(end2[0].payload["reason"], "end_turn");

    // review 里的工具调用照常记账。
    assert_eq!(of(&evs, &tr[1], "brain/tool_call").len(), 1);
    let res = of(&evs, &tr[1], "brain/tool_result");
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].payload["name"], "skill_manage");
    assert_eq!(res[0].payload["ok"], true);

    // 磁盘：文件、所有权、账本。
    let store = fx.store();
    let doc = store.read(SKILL).expect("skill 该落盘了");
    assert_eq!(doc.text, GREETING);
    assert_eq!(doc.created_by.as_deref(), Some("review:fast"));
    let ledger = store.ledger().unwrap();
    assert_eq!(ledger.len(), 1, "{ledger:?}");
    assert_eq!(ledger[0].actor, "review:fast");
    assert_eq!(ledger[0].op, Op::Create);
    assert_eq!(ledger[0].rel, SKILL);
    assert_eq!(ledger[0].new_sha256, sha256_hex(GREETING.as_bytes()));

    // 第三个 turn 看不见 review 的工具。
    let reqs = of(&evs, &tr[2], "brain/request");
    assert_eq!(reqs.len(), 1);
    let names = strs(&reqs[0].payload["tool_names"]);
    assert!(
        !names
            .iter()
            .any(|n| n == "skill_manage" || n == "memory_manage" || n == "read"),
        "普通 turn 不该看得见 review 的工具: {names:?}"
    );

    // 前缀：前两个 turn 逐字节相同，第三个因为 `# Skills` 段变了而不同。
    let (turns, reviews) = split_requests(&model);
    assert_eq!(turns.len(), 3, "三个普通 turn 各一次模型调用");
    assert_eq!(reviews.len(), 2, "review 跑了两步");
    assert_eq!(turns[0].system, turns[1].system, "什么都没改时前缀字节稳定");
    assert_ne!(turns[0].system, turns[2].system, "写过之后前缀必须变");
    assert!(!turns[0].system.contains("# Skills"), "{}", turns[0].system);
    assert!(
        turns[2]
            .system
            .contains(&format!("\n# Skills\n{SKILL}: # greeting\n")),
        "{}",
        turns[2].system
    );

    // review 的工具表只有三个，一个 `robot_*` / `ask_*` 都没有。
    let mut rn: Vec<String> = reviews[0].tools.iter().map(|t| t.name.clone()).collect();
    rn.sort();
    assert_eq!(rn, vec!["memory_manage", "read", "skill_manage"]);
    assert!(
        reviews[0].system.contains("DO NOT capture"),
        "{}",
        reviews[0].system
    );
}

// ---------------------------------------------------------------- (b)

/// 写 store 外面的路径：工具错误 + `safety/tamper_attempt`，一个字节都不落盘。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_review_write_that_escapes_the_store_is_refused_and_reported_as_tampering() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n  \
           - text: \"试着改 bundle\"\n    \
             tool_calls:\n      \
               - name: skill_manage\n        \
                 args: { op: create, path: \"../bundles/x.yaml\", content: \"boom\" }\n  \
           - text: \"被挡了\"\n",
    );
    let b = boot(fx.rows(
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    let evs = b.events();
    b.stop().await;

    let tr = traces(&evs);
    let res = of(&evs, &tr[0], "brain/tool_result");
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].payload["ok"], false);
    let err = res[0].payload["error"].as_str().unwrap_or_default();
    assert!(err.contains("bad path"), "{err}");

    let tamper = only(&evs, "safety/tamper_attempt");
    assert_eq!(tamper.payload["actor"], "review:fast");
    assert_eq!(tamper.payload["path"], "../bundles/x.yaml");
    assert_eq!(tamper.payload["op"], "write");

    let review = only(&evs, "brain/review");
    assert_eq!(review.payload["trigger"], "every_turns");
    assert!(strs(&review.payload["writes"]).is_empty());
    assert_eq!(review.payload["ledger_seq"], 0);
    assert!(
        review.payload.get("error").is_none(),
        "工具失败不是 review 失败"
    );

    // 一个字节都没落盘。
    assert!(!fx.home().join("robots/bundles/x.yaml").exists());
    assert!(fx.store().ledger().unwrap().is_empty());
}

// ---------------------------------------------------------------- (c)

const V2: &str = "# greeting\n\n见到人先打招呼，然后报名字。\n";

/// read-before-write：不带哈希的 `update` 是 `StaleRead`；先 `read` 拿到哈希，再带着它
/// 写才成功。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_update_without_the_hash_is_stale_and_the_one_after_a_read_succeeds() {
    let v1_sha = sha256_hex(GREETING.as_bytes());
    let fx = Fixture::new(&format!(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n  \
           - tool_calls:\n      \
               - name: skill_manage\n        \
                 args: {{ op: create, path: {SKILL}, content: \"{g}\" }}\n  \
           - tool_calls:\n      \
               - name: skill_manage\n        \
                 args: {{ op: update, path: {SKILL}, content: \"{v2}\" }}\n  \
           - tool_calls:\n      \
               - name: read\n        \
                 args: {{ path: {SKILL} }}\n  \
           - tool_calls:\n      \
               - name: skill_manage\n        \
                 args: {{ op: update, path: {SKILL}, content: \"{v2}\", expect_sha256: {v1_sha} }}\n",
        g = GREETING.replace('\n', "\\n"),
        v2 = V2.replace('\n', "\\n"),
    ));
    let b = boot(fx.rows(
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    let evs = b.events();
    b.stop().await;

    let tr = traces(&evs);
    let res = of(&evs, &tr[0], "brain/tool_result");
    assert_eq!(res.len(), 4, "四步各一次工具调用");
    assert_eq!(res[0].payload["ok"], true, "create");
    assert_eq!(res[1].payload["ok"], false, "不带哈希的 update");
    let err = res[1].payload["error"].as_str().unwrap_or_default();
    assert!(err.contains("stale read"), "{err}");
    assert_eq!(res[2].payload["ok"], true, "read");
    assert_eq!(res[3].payload["ok"], true, "带哈希的 update");

    let review = only(&evs, "brain/review");
    assert_eq!(review.payload["steps"], 4);
    assert_eq!(
        strs(&review.payload["writes"]),
        vec![SKILL.to_string()],
        "同一个文件写两次只算一条"
    );
    assert_eq!(review.payload["ledger_seq"], 2);

    let store = fx.store();
    assert_eq!(store.read(SKILL).unwrap().text, V2);
    let ledger = store.ledger().unwrap();
    assert_eq!(ledger.len(), 2);
    assert_eq!(ledger[0].op, Op::Create);
    assert_eq!(ledger[1].op, Op::Update);
    assert_eq!(ledger[1].prev_sha256.as_deref(), Some(v1_sha.as_str()));
}

// ---------------------------------------------------------------- (d)

const MEMO: &str = "- 会话结束时记一条。\n";

/// detach 时距上次 review 还有一个 turn → 补一次 `trigger: "session_end"` 的 review。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn detaching_runs_one_last_review_when_turns_are_pending() {
    let fx = Fixture::new(&format!(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n  \
           - tool_calls:\n      \
               - name: memory_manage\n        \
                 args: {{ op: append, path: memory/MEMORY.md, content: \"{}\" }}\n  \
           - text: \"记完了\"\n",
        MEMO.replace('\n', "\\n")
    ));
    let b = boot(fx.rows(
        json!({"every_turns": 0, "on_session_end": true, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    // `every_turns: 0` = 关掉周期 review，所以此刻一条都不该有。
    assert!(!b.events().iter().any(|e| e.kind.as_str() == "brain/review"));

    // 撤销脑那一行：ack 回来 = detach 已经发生（会话末 review 由那一下派生出来）。
    let fid = fiber_id(&b.log, "brain-native").await;
    b.h.dispose_fiber(fid).unwrap().await.unwrap();
    let review = wait_for(&b.log, |e| e.kind.as_str() == "brain/review").await;
    assert_eq!(review.payload["trigger"], "session_end");
    assert_eq!(review.payload["brain_id"], "fast");
    assert_eq!(
        strs(&review.payload["writes"]),
        vec!["memory/MEMORY.md".to_string()]
    );
    assert_eq!(review.payload["ledger_seq"], 1);
    b.stop().await;

    let store = fx.store();
    assert_eq!(store.read("memory/MEMORY.md").unwrap().text, MEMO);
    let ledger = store.ledger().unwrap();
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].actor, "review:fast");
}

/// 两个触发器都关掉 → 一条 `brain/review` 都不该有，detach 也不该补。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_both_triggers_off_no_review_ever_runs() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n  - text: \"这一步永远不该被请求\"\n",
    );
    let b = boot(fx.rows(
        json!({"every_turns": 0, "on_session_end": false, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    let fid = fiber_id(&b.log, "brain-native").await;
    b.h.dispose_fiber(fid).unwrap().await.unwrap();
    stays_absent(
        &b.log,
        "两个触发器都关掉时的 brain/review",
        "brain/review",
        Duration::from_millis(200),
    )
    .await;
    let model = b.model.clone();
    let evs = b.events();
    b.stop().await;
    assert!(!evs.iter().any(|e| e.kind.as_str() == "brain/review"));
    // review 的剧本一步都没被弹出来。
    assert!(split_requests(&model).1.is_empty());
    assert!(fx.store().ledger().unwrap().is_empty());
}

// ---------------------------------------------------------------- (e)

/// review 的模型流断掉：`brain/review{error}`，而触发它的那个 turn 照常 `end_turn`。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_broken_model_stream_inside_the_review_never_fails_the_turn() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n  - text: \"开始 review\"\n    stream_error: connection reset\n",
    );
    let b = boot(fx.rows(
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let evs = b.events();
    b.stop().await;

    let tr = traces(&evs);
    let review = only(&evs, "brain/review");
    assert_eq!(review.trace_id.as_deref(), Some(tr[0].as_str()));
    assert_eq!(review.payload["trigger"], "every_turns");
    assert_eq!(review.payload["steps"], 1);
    assert!(strs(&review.payload["writes"]).is_empty());
    let err = review.payload["error"].as_str().unwrap_or_default();
    assert!(err.contains("connection reset"), "{err}");

    // 维护相位的模型调用在日志里说得出自己是哪个相位。review **不记**
    // `brain/request`，所以少了这个字段，一条 `brain/request-error{step: 0}` 读起来和
    // turn 第一步断流一模一样——两者要查的地方完全不同。
    let req_errs = of(&evs, &tr[0], "brain/request-error");
    assert_eq!(req_errs.len(), 1, "{req_errs:?}");
    assert_eq!(req_errs[0].payload["phase"], "review");
    assert_eq!(req_errs[0].payload["step"], 0);

    let end = of(&evs, &tr[0], "brain/turn_end");
    assert_eq!(end.len(), 1);
    assert_eq!(
        end[0].payload["reason"], "end_turn",
        "review 出错不带走 turn"
    );
    assert!(review.seq < end[0].seq);
    assert!(fx.store().ledger().unwrap().is_empty());
}

// ---------------------------------------------------------------- (f)

/// **整段 review 有界，而且那道界不依赖 `step_timeout_ms`。**
///
/// 配的是 `step_timeout_ms: 0`（单步不设限）加一个永不产出也永不结束的模型流。内核的
/// `turn_timeout_ms` 扫描只看 `Running` 的脑，而 review 跑在 `Maintenance` 上——所以在
/// `review.timeout_ms` 之前，这个组合会让 turn 永远停在维护相位里：没有 `done`、没有
/// `brain/turn_end`、没有 `brain/error`，此后每一次唤醒都只 latch 不开 turn，只有一次
/// 定向 `interrupt` 救得回来。
///
/// 顺带钉死**超时保留已经跑出来的那部分**：第一步真的写成了一个 skill，第二步才挂住。
/// 那次写已经落盘入账了，所以 `writes` 必须说得出它——否则一次超时会让日志谎称什么都
/// 没写，而且 `# Skills` 前缀不会重建，模型下一个 turn 看不见自己刚写下的东西。
///
/// 判别信号是那条 `brain/review{error:"timeout"}`（等事件，不是等时钟）；`timeout_ms`
/// 只决定这个用例跑多久。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_review_that_never_returns_is_cut_off_and_keeps_what_it_already_wrote() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n\
         \x20 - tool_calls:\n\
         \x20     - name: skill_manage\n\
         \x20       args: { op: create, path: skills/greeting/SKILL.md, content: \"# greeting\\n\" }\n\
         \x20 - hang: true\n",
    );
    let b = boot(fx.rows_with(
        // `timeout_ms` 要比"第一步写完一个文件（含 fsync）"宽裕得多：这一条判的是第二步
        // 挂住之后有没有人来收场，不是第一步跑得快不快。余量小了会在负载高的机器上
        // 变成偶发红——1200 ms 在 2 核的 GitHub runner 上就红过一次（develop 4d66647 的
        // 合并后 CI：`writes` 是空的，第一步还没落盘就被截了）。4 s 让这条用例多花约 3 s。
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 4, "timeout_ms": 4000}),
        4,
        0, // 单步不设限：这道界只能来自 `review.timeout_ms`
    ));
    b.wake();
    // turn 关得掉，这本身就是断言：没有这道界，维护相位会把它永远挂在那里。
    turn_ends(&b.log, 1).await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let evs = b.events();
    b.stop().await;

    let tr = traces(&evs);
    let review = only(&evs, "brain/review");
    assert_eq!(review.payload["error"], "timeout");
    assert_eq!(review.payload["trigger"], "every_turns");
    // 第一步跑完了（第二步挂在模型调用里，没能记上 steps）。
    assert_eq!(review.payload["steps"], 1);
    // 已经落盘的那次写没有被超时抹掉。
    assert_eq!(
        strs(&review.payload["writes"]),
        vec!["skills/greeting/SKILL.md".to_string()]
    );
    assert_eq!(review.payload["ledger_seq"], 1);

    // turn 照常收尾，而且排在 review 之后。
    let end = of(&evs, &tr[0], "brain/turn_end");
    assert_eq!(end.len(), 1);
    assert_eq!(end[0].payload["reason"], "end_turn");
    assert!(review.seq < end[0].seq);

    // 磁盘和账本上都是那一次写。
    let doc = fx.store().read("skills/greeting/SKILL.md").unwrap();
    assert_eq!(doc.text, "# greeting\n");
    assert_eq!(doc.created_by.as_deref(), Some("review:fast"));
    assert_eq!(fx.store().ledger().unwrap().len(), 1);
}

// ---------------------------------------------------------------- (g)

/// review 里的 panic 不带走 turn，也不把脑留在 `Maintenance` 上。
///
/// 维护相位跑在 `run_turn` 那层 `catch_unwind` 的**外面**（那层只包 step 循环），所以
/// 一个漏出去的 panic 会同时吃掉 `leave_maintenance()` 和 `done`——症状恰恰是"什么都
/// 没发生"：turn 挂着、没有 `brain/turn_end`、而超时扫描只管 `Running`，谁也不会来收
/// 尾。这条用例的判别信号就是那条 `brain/turn_end`。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_panic_inside_the_review_still_leaves_maintenance_and_closes_the_turn() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n  - text: \"开始 review\"\n    panic: boom\n",
    );
    let b = boot(fx.rows(
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 4}),
        4,
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    // 退出了维护相位：脑回到 Idle，而不是永远停在 Maintenance 上只 latch 不开 turn。
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let evs = b.events();
    b.stop().await;

    let tr = traces(&evs);
    let review = only(&evs, "brain/review");
    assert_eq!(review.payload["error"], "panicked");
    assert_eq!(review.payload["trigger"], "every_turns");
    assert!(strs(&review.payload["writes"]).is_empty());

    let end = of(&evs, &tr[0], "brain/turn_end");
    assert_eq!(end.len(), 1);
    assert_eq!(end[0].payload["reason"], "end_turn", "panic 不带走 turn");
    assert!(review.seq < end[0].seq);
    assert!(fx.store().ledger().unwrap().is_empty());
}

// ---------------------------------------------------------------- (h)

/// `review.max_steps` 真的是 review **自己的**步数上限，不是 turn 的那个。
///
/// 剧本给了五步，`review.max_steps: 2` 而 turn 的 `max_steps: 8`：跑满两步就停，所以只
/// 有前两个 skill 落盘。没有这条用例，一个错用了 `s.cfg.max_steps`（甚至完全不设限）的
/// 实现会悄悄通过——一份刚好跑完的剧本加上"剧本用完就 `EndTurn`"会把它盖住。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_review_stops_at_its_own_max_steps_not_the_turns() {
    let step = |n: u32| {
        format!(
            "\x20 - tool_calls:\n\
             \x20     - name: skill_manage\n\
             \x20       args: {{ op: create, path: skills/s{n}/SKILL.md, content: \"# s{n}\\n\" }}\n"
        )
    };
    let script = format!(
        "turns:\n  - text: \"一个 turn\"\nreview:\n{}",
        (1..=5).map(step).collect::<String>()
    );
    let fx = Fixture::new(&script);
    let b = boot(fx.rows(
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 2}),
        8, // turn 的上限故意开得比 review 的大
    ));
    b.wake();
    turn_ends(&b.log, 1).await;
    let evs = b.events();
    let remaining = b.model.remaining();
    b.stop().await;

    let review = only(&evs, "brain/review");
    assert_eq!(review.payload["steps"], 2, "review 该在自己的第 2 步停下");
    assert!(review.payload.get("error").is_none(), "停在上限不是错误");
    assert_eq!(
        strs(&review.payload["writes"]),
        vec![
            "skills/s1/SKILL.md".to_string(),
            "skills/s2/SKILL.md".to_string()
        ]
    );
    // 剧本剩下的三步一步都没跑：账本两行，s3 根本不存在。
    assert_eq!(fx.store().ledger().unwrap().len(), 2);
    assert!(fx.store().read("skills/s3/SKILL.md").is_err());
    assert_eq!(remaining, 3, "剧本里该还剩三步没被弹出");
}

// ---------------------------------------------------------------- (i)

/// **review 真的跑在 `Maintenance` 相位里**，所以内核的 turn 超时扫描够不着它。
///
/// 这是整套用例里唯一配 `turn_timeout_ms` **非 0** 的一条，也是唯一能看见这件事的一
/// 条：别的用例全配 0，而那恰好把唯一的观测机制关掉了——`MaintenanceGuard` 一度只做了
/// "出"那一半（构造时没有 `enter_maintenance()`），整个套件一条红的都没有。
///
/// 场景：一个瞬间跑完的 turn，后面跟一段**合法地**比 `turn_timeout_ms` 还长、但远在
/// `review.timeout_ms` 之内、并且干净收尾的 review。
///
/// - 不进 Maintenance 的话：`sweep_timeouts` 只看 `status == Running`，这个 turn 在
///   200 ms 之后就会被强行关掉，留下 `brain/error{reason:"turn_timeout"}` 和一条
///   `brain/turn_end{reason:"timeout"}`——一个其实已经做完、只是在记笔记的 turn 被诬告
///   成脑异常。
/// - 进了 Maintenance：扫描跳过它，review 跑完，turn 照常 `end_turn`。
///
/// 判别信号是事件（`brain/error` 出现即失败 / `turn_end.reason`），而最强的那条和时钟
/// 完全无关：`status == Maintenance`。`delay_ms` 只是**刺激**——它决定这一条跑多久，不
/// 决定它怎么判。中间还同步调了一次 `sweep_now()`：那是一个确定的同步点（"现在就扫一
/// 遍"），不用等定时器。
///
/// 两个数字的余量是有意留宽的：turn 自己的 step 循环必须稳稳跑在 `turn_timeout_ms`
/// （1000 ms）之内，而 review（~1800 ms）必须明显长过它。余量小了，一次负载尖峰会让这
/// 一条在 step 循环那一段就被误杀，变成偶发红——这不是被测系统的问题，是用例的问题。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_review_longer_than_the_turn_timeout_is_not_force_closed() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n\
         \x20 - tool_calls:\n\
         \x20     - name: skill_manage\n\
         \x20       args: { op: create, path: skills/greeting/SKILL.md, content: \"# greeting\\n\" }\n\
         \x20 - text: \"记完了\"\n    delay_ms: 1800\n",
    );
    let b = boot(fx.rows_full(
        json!({"every_turns": 1, "on_session_end": false, "max_steps": 4, "timeout_ms": 30_000}),
        4,
        0,    // 单步不设限：这一条要测的是内核那道扫描，不是 step 预算
        1000, // turn 自己的 step 循环要稳稳跑在这之内，而 review 要明显长过它
    ));
    b.wake();

    // 同步点：review 的第一步写成了 skill，也就是说它**正在**维护相位里跑。
    wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/tool_result" && e.payload["name"] == "skill_manage"
    })
    .await;
    // 这就是那条一直缺着的直接断言。
    assert_eq!(
        b.brains.status("fast"),
        Some(BrainStatus::Maintenance),
        "review 必须跑在 Maintenance 相位里，否则内核的超时扫描够得着它"
    );
    // 立刻同步扫一遍（生产路径上那个定时任务调的是同一个函数）。review 还有 ~1800 ms
    // 要跑，其间定时任务（周期 250 ms）还会扫好几次，全都在 turn 的 1000 ms 预算之外。
    b.brains.sweep_now();

    turn_ends(&b.log, 1).await;
    assert_eq!(b.brains.status("fast"), Some(BrainStatus::Idle));
    let evs = b.events();
    b.stop().await;

    // 一条 `brain/error` 都不该有：这个 turn 没有超时，它只是在记笔记。
    let errs: Vec<&WorldEvent> = evs
        .iter()
        .filter(|e| e.kind.as_str() == "brain/error")
        .collect();
    assert!(errs.is_empty(), "turn 被超时扫描误杀了: {errs:?}");

    let tr = traces(&evs);
    let end = of(&evs, &tr[0], "brain/turn_end");
    assert_eq!(end.len(), 1);
    assert_eq!(
        end[0].payload["reason"], "end_turn",
        "不是 timeout——这个 turn 自己收的尾"
    );

    // review 干净跑完（没有 error），而且它的写落了盘。
    let review = only(&evs, "brain/review");
    assert!(
        review.payload.get("error").is_none(),
        "review 该干净收尾: {:?}",
        review.payload
    );
    assert_eq!(
        strs(&review.payload["writes"]),
        vec!["skills/greeting/SKILL.md".to_string()]
    );
    assert!(review.seq < end[0].seq);
    assert_eq!(fx.store().ledger().unwrap().len(), 1);
}

// ---------------------------------------------------------------- (j)

/// **detach 撞上一次在飞的周期 review：至少还有一条 `brain/review`，不会一条都没有。**
///
/// `Drop` 先 `abort_current()` 再派生会话末那一次，而 `abort` 只是**预约**取消。此前的
/// 写法在这里会两头落空：在飞的那一个还攥着 `ReviewGuard`，于是会话末那次判定"有人在
/// 跑"直接跳过；而在飞的那一个随后在它的下一个 await 点被丢掉，永远走不到
/// `finish_review`。结果是**零**条 `brain/review`——比这个守卫本来要防的"两条"更糟，
/// 而且前缀也不会重建。
///
/// 现在会话末那一次先 `await` 那个被 abort 的句柄（任务真的停了、令牌还回来了）再开
/// 始，所以这里必然留下一条。
///
/// 同步点是那条 `brain/tool_result`：它证明周期 review 已经在飞了，此刻才 dispose。
///
/// 这一条当场抓住了第一版修复的一个洞：`abort_current()` 曾经把句柄**返回**给调用方，
/// 而 detach 这条路上它会被调用**两次**（插件的 `on_dispose` 先跑一步，`Drop` 随后），
/// 于是 `Drop` 永远拿到 `None`——等于根本没等。症状正是这一条断言的反面：十秒之内一条
/// `brain/review` 都没有。句柄现在留在 `Shared::aborted` 里，谁 abort 的都一样。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn detaching_on_top_of_a_running_review_still_leaves_one_review_behind() {
    let fx = Fixture::new(
        "turns:\n  - text: \"一个 turn\"\n\
         review:\n\
         \x20 - tool_calls:\n\
         \x20     - name: skill_manage\n\
         \x20       args: { op: create, path: skills/greeting/SKILL.md, content: \"# greeting\\n\" }\n\
         \x20 - text: \"还在想\"\n    delay_ms: 1200\n\
         \x20 - tool_calls:\n\
         \x20     - name: memory_manage\n\
         \x20       args: { op: append, path: memory/MEMORY.md, content: \"- noted\\n\" }\n",
    );
    let b = boot(fx.rows(
        json!({"every_turns": 1, "on_session_end": true, "max_steps": 4, "timeout_ms": 10_000}),
        4,
    ));
    b.wake();
    // 周期 review 的第一步写成了：它正在飞。
    wait_for(&b.log, |e| {
        e.kind.as_str() == "brain/tool_result" && e.payload["name"] == "skill_manage"
    })
    .await;

    // 就在这时候撤销脑那一行。
    let fid = fiber_id(&b.log, "brain-native").await;
    b.h.dispose_fiber(fid).unwrap().await.unwrap();

    // 至少一条 `brain/review`——判别信号是它出现，不是等多久。
    let review = wait_for(&b.log, |e| e.kind.as_str() == "brain/review").await;
    let evs = b.events();
    b.stop().await;

    let reviews: Vec<&WorldEvent> = evs
        .iter()
        .filter(|e| e.kind.as_str() == "brain/review")
        .collect();
    assert!(
        !reviews.is_empty(),
        "detach 撞上在飞的 review 之后一条记录都没有"
    );
    assert_eq!(review.payload["brain_id"], "fast");

    // 第一步那次写是**确实**落了盘的，所以无论最后是哪一次 review 收的尾，账本上都有
    // 它，而 `writes` 不能否认它（`finish_review` 会拿账本对一遍）。
    let ledger = fx.store().ledger().unwrap();
    assert!(!ledger.is_empty(), "第一步那次写没落盘");
    assert_eq!(ledger[0].actor, "review:fast");
    assert_eq!(ledger[0].rel, "skills/greeting/SKILL.md");
    let reported: Vec<String> = reviews
        .iter()
        .flat_map(|e| strs(&e.payload["writes"]))
        .collect();
    assert!(
        reported.contains(&"skills/greeting/SKILL.md".to_string()),
        "账本上有这一行，`brain/review.writes` 却否认它: {reported:?}"
    );
}
