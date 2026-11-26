use oeh_kernel::observation::{Observation, WakeReason};
use oeh_kernel::session_log::{read_session, SessionEntry, SessionLog};
use oeh_kernel::wake::WakeAction;
use std::collections::BTreeMap;

fn obs(seq: u64) -> Observation {
    Observation {
        t_kernel: 5,
        wake_reason: WakeReason {
            kind: "obs/person".into(),
            seq,
            action: WakeAction::Wake,
        },
        state: BTreeMap::new(),
        events: vec![],
        source_seq_range: (1, seq),
        truncated: 0,
    }
}

#[test]
fn records_and_reads_back_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let log = SessionLog::open(dir.path(), "fast").unwrap();
    assert!(log.path().ends_with("fast.session.jsonl"));
    log.record(&SessionEntry::TurnStart {
        turn: 1,
        t_kernel: 5,
    })
    .unwrap();
    log.record(&SessionEntry::Observation {
        turn: 1,
        observation: obs(7),
    })
    .unwrap();
    log.record(&SessionEntry::Say {
        turn: 1,
        text: "hi".into(),
    })
    .unwrap();
    log.record(&SessionEntry::TurnEnd {
        turn: 1,
        t_kernel: 9,
        reason: "completed".into(),
    })
    .unwrap();
    log.record(&SessionEntry::Attached {
        epoch: 3,
        brain_id: "fast".into(),
        t_kernel: 4,
    })
    .unwrap();
    // `record` 只入队（写线程负责落盘），所以读文件之前要排空一次。
    log.flush().unwrap();
    let back = read_session(log.path()).unwrap();
    assert_eq!(back.len(), 5);
    assert!(
        matches!(&back[4], SessionEntry::Attached { epoch: 3, brain_id, t_kernel: 4 } if brain_id == "fast")
    );
    assert!(
        matches!(&back[1], SessionEntry::Observation { turn: 1, observation } if observation.source_seq_range == (1, 7))
    );
    let text = std::fs::read_to_string(log.path()).unwrap();
    assert!(text
        .lines()
        .next()
        .unwrap()
        .contains(r#""type":"turn_start""#));
    // 再次 open 是追加
    let log2 = SessionLog::open(dir.path(), "fast").unwrap();
    log2.record(&SessionEntry::TurnStart {
        turn: 2,
        t_kernel: 10,
    })
    .unwrap();
    log2.flush().unwrap();
    assert_eq!(read_session(log.path()).unwrap().len(), 6);
}

#[test]
fn parse_error_reports_line() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("x.session.jsonl");
    std::fs::write(
        &p,
        "{\"type\":\"turn_start\",\"turn\":1,\"t_kernel\":1}\nnot json\n",
    )
    .unwrap();
    let err = read_session(&p).unwrap_err();
    assert!(matches!(
        err,
        oeh_kernel::session_log::SessionReadError::Parse { line: 2 }
    ));
}

/// 一个**已经存在、但这个进程 chmod 不动**的目录，不能让 `brain/attach` 起不来。
///
/// 容器里的常规做法：日志/会话目录由 root（或者别的 uid）预先建好挂进来，内核以一个普通
/// 用户跑。收紧权限是纵深防御（见 `oeh_kernel` 的 `fsutil`），不是打开会话日志的前置条件
/// ——把它当成前置条件的话，一个 `set_permissions` 的 `EPERM` 会变成"每一次
/// `brain/attach` 都是 `io: Operation not permitted`"。
///
/// 怎么造出"chmod 不动"：`/tmp` 属于 root 且带 sticky 位，一个普通用户写得进去、却改不了
/// 它的权限位。两道闸门保证这条测试永远不会真的去改什么：以 root 跑时直接跳过（那种进程
/// 什么都 chmod 得动，这个场景不存在），`/tmp` 万一属于当前用户也跳过。
#[test]
#[cfg(unix)]
fn a_foreign_owned_directory_does_not_break_session_log_open() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    unsafe extern "C" {
        fn geteuid() -> u32;
    }
    let me = unsafe { geteuid() };
    if me == 0 {
        return;
    }
    let foreign = std::path::Path::new("/tmp");
    let Ok(md) = std::fs::metadata(foreign) else {
        return;
    };
    if md.uid() == me {
        return;
    }
    let before = md.permissions().mode();

    let brain = format!("oeh-foreign-{}", std::process::id());
    let log =
        SessionLog::open(foreign, &brain).expect("a foreign-owned session dir must still open");
    log.record(&SessionEntry::TurnStart {
        turn: 1,
        t_kernel: 5,
    })
    .unwrap();
    let path = log.path().to_path_buf();
    drop(log);
    let _ = std::fs::remove_file(&path);

    // 而且我们确实没有动那个目录。
    assert_eq!(
        std::fs::metadata(foreign).unwrap().permissions().mode(),
        before
    );
}

/// `record` 不在调用线程上写盘：2000 条一口气记完（队列 1024）不阻塞，`flush()` 之后
/// 落在文件里的那些必须**保序**、从第 0 条起，丢掉的那些数得出来。
///
/// 钉住的语义是"**有序，但可能有缺口**"，不是"连续的前缀"：`try_send` 丢的是"入队的那
/// 一刻队列恰好满"的那一条，写线程腾出一格之后下一条又进得去，所以缺口可以落在中间。
/// 三条断言正好是这个语义的判据——第一条一定在（入队时队列是空的）、严格递增（单消费者
/// FIFO），而 `len + dropped == 2000` 说明每一条要么落盘要么被计数，没有第三种下场。
#[test]
fn record_does_not_block_and_flush_drains_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let log = SessionLog::open(dir.path(), "fast").unwrap();
    for i in 0..2000u64 {
        log.record(&SessionEntry::ModelOutput {
            turn: 1,
            step: i,
            text: format!("out {i}"),
        })
        .unwrap();
    }
    log.flush().unwrap();
    let entries = read_session(log.path()).unwrap();
    let outs: Vec<u64> = entries
        .iter()
        .filter_map(|e| match e {
            SessionEntry::ModelOutput { step, .. } => Some(*step),
            _ => None,
        })
        .collect();
    assert_eq!(outs[0], 0);
    assert!(outs.windows(2).all(|w| w[0] < w[1]), "out of order");
    assert_eq!(outs.len() as u64 + log.dropped(), 2000);
}

/// Plan 4 的三个新条目里的两个：一次工具调用和它的结果，按 `call_id` 配对，原样回来。
#[test]
fn tool_call_and_result_entries_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let log = SessionLog::open(dir.path(), "fast").unwrap();
    log.record(&SessionEntry::ToolCall {
        turn: 1,
        step: 0,
        call_id: "c1".into(),
        name: "robot_walk".into(),
        args: serde_json::json!({"vx": 0.1}),
    })
    .unwrap();
    log.record(&SessionEntry::ToolResult {
        turn: 1,
        step: 0,
        call_id: "c1".into(),
        ok: true,
        result: serde_json::json!({"outcome": "executed"}),
    })
    .unwrap();
    log.flush().unwrap();
    let entries = read_session(log.path()).unwrap();
    assert!(
        matches!(&entries[0], SessionEntry::ToolCall { name, args, call_id, .. }
            if name == "robot_walk" && args["vx"] == 0.1 && call_id == "c1")
    );
    assert!(matches!(
        entries[1],
        SessionEntry::ToolResult { ok: true, .. }
    ));
    // 线格式是 snake_case 的 `type`，和已有条目一个风格。
    let text = std::fs::read_to_string(log.path()).unwrap();
    assert!(text
        .lines()
        .next()
        .unwrap()
        .contains(r#""type":"tool_call""#));
    assert!(text
        .lines()
        .nth(1)
        .unwrap()
        .contains(r#""type":"tool_result""#));
}
