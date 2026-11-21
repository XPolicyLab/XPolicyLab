//! `mock-codex`：一个按剧本演的 Codex app-server 替身。
//!
//! 线格式就是真的那一套（[`oeh_brain::codex::wire`] 里的方法名，JSON-RPC 2.0 行分隔，
//! stdin 进 stdout 出），所以 `brain-codex` 那边一个字都不用改就能对着它跑端到端。
//!
//! 剧本由环境变量 `MOCK_CODEX_SCRIPT` 指向一个 YAML 文件：**一个 turn 一个列表**，
//! 元素按顺序演：
//!
//! ```yaml
//! - - say: "hi"                                           # 第 1 个 turn
//!   - tool: { name: robot_walk, arguments: { vx: 0.1 } }
//!   - say: "done"
//! - - say: "second turn"                                  # 第 2 个 turn
//! ```
//!
//! 六种元素：
//!
//! - `say: <文本>` —— 发 `item/agentMessage/delta` + `item/completed{agentMessage}`；
//! - `tool: {name, arguments}` —— 发 `item/tool/call` 请求，**等**对面回答；
//! - `approval: {command, cwd?}` —— 发 `item/commandExecution/requestApproval`，等回答；
//! - `stray: <文本>` —— 一条**点名了别的 turn** 的增量：脑必须丢掉它并数一次；
//! - `status: <串>` —— 这个 turn 的 `turn/completed.status`（默认 `completed`）；
//! - `exit: true` —— 当场 `exit(9)`，不发 `turn/completed`：模拟子进程中途死掉。
//!
//! 控制面：`thread/start` → `{thread_id:"t1"}`；`turn/start` → `{turn_id:"u<n>"}` 之后
//! 才开演；`turn/steer` 记下那句话；`turn/interrupt` 掐掉正在演的那个 turn 并回一条
//! `turn/completed{status:"interrupted"}`。
//!
//! **收到的控制消息与审批答复会攒成"笔记"，在下一句 `say` 里原样回显。** 这是测试看见
//! "`turn/steer` / `turn/interrupt` / `decline` 真的到了子进程手里"的唯一途径——它们在
//! 世界日志里没有对应的事件，而没有回显就只能靠 sleep 去猜。

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use oeh_brain::codex::wire;
use oeh_proto::rpc::{code, Id, Message, Notification, Request, Response, RpcError};
use oeh_rpc::codec::{decode, encode};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

const THREAD_ID: &str = "t1";

/// 剧本里的一个元素。写成"四个都可选的结构"而不是一个 enum：`serde_yaml_ng` 的外部
/// tag 走的是 YAML 的 `!Say` 标签，而剧本里自然的写法是 `{say: "hi"}` 这样的 map；
/// 而且这样写，最常见的错法（拼错 key）能得到一句点名的错误。
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Step {
    #[serde(default)]
    say: Option<String>,
    #[serde(default)]
    tool: Option<ToolStep>,
    #[serde(default)]
    approval: Option<ApprovalStep>,
    /// 一条点名了别的 turn 的增量。
    #[serde(default)]
    stray: Option<String>,
    /// 这个 turn 收尾时报的 `status`。
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    exit: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolStep {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovalStep {
    command: Value,
    #[serde(default)]
    cwd: Option<String>,
}

/// 写线程要做的事。`Flush` 是一道屏障：它被处理到，就说明排在它前面的每一行都已经
/// 写出去并 flush 过了。`exit` 那一步需要它——`std::process::exit` 不会等任何队列，
/// 少了这道屏障，"说完一句话再崩" 会变成 "什么都没说就崩"，而那是另一个测试场景。
enum Wire {
    Line(String),
    Flush(oneshot::Sender<()>),
}

struct Mock {
    out: mpsc::UnboundedSender<Wire>,
    /// 还没演的 turn，从头取。演完了再来一个 `turn/start` = 一个空 turn（立刻 completed）。
    script: Mutex<VecDeque<Vec<Step>>>,
    /// 收到的控制消息 / 审批答复，下一句 `say` 一并回显。
    notes: Mutex<Vec<String>>,
    /// 我们发出去的请求在等的答复。
    pending: Mutex<HashMap<String, oneshot::Sender<Value>>>,
    next_id: AtomicU64,
    turns: AtomicU64,
    /// 正在演的那个 turn。`turn/interrupt` 把它掐掉。
    current: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Mock {
    fn send(&self, m: &Message) {
        let _ = self.out.send(Wire::Line(encode(m)));
    }

    /// 等排在前面的每一行都真的写出去了。
    async fn flush(&self) {
        let (tx, rx) = oneshot::channel();
        if self.out.send(Wire::Flush(tx)).is_ok() {
            let _ = rx.await;
        }
    }

    fn notify(&self, method: &str, params: Value) {
        self.send(&Message::Notification(Notification::new(method, params)));
    }

    fn reply(&self, id: Id, result: Value) {
        self.send(&Message::Response(Response::ok(id, result)));
    }

    fn reply_err(&self, id: Id, message: &str) {
        self.send(&Message::Response(Response::err(
            id,
            RpcError::new(code::METHOD_NOT_FOUND, message),
        )));
    }

    /// 发一个请求，等对面回答。对面没答（连接断了）就还 `null`。
    async fn request(&self, method: &str, params: Value) -> Value {
        let id = format!("m{}", self.next_id.fetch_add(1, Ordering::SeqCst));
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id.clone(), tx);
        self.send(&Message::Request(Request::new(Id::Str(id), method, params)));
        rx.await.unwrap_or(Value::Null)
    }

    fn resolve(&self, resp: Response) {
        let Id::Str(key) = resp.id else { return };
        let tx = self.pending.lock().remove(&key);
        if let Some(tx) = tx {
            let _ = tx.send(resp.result.unwrap_or(Value::Null));
        }
    }

    fn note(&self, what: String) {
        self.notes.lock().push(what);
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let script = load_script();
    let (out, mut out_rx) = mpsc::unbounded_channel::<Wire>();
    tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(w) = out_rx.recv().await {
            match w {
                Wire::Line(line) => {
                    if stdout.write_all(line.as_bytes()).await.is_err()
                        || stdout.write_all(b"\n").await.is_err()
                        || stdout.flush().await.is_err()
                    {
                        break;
                    }
                }
                Wire::Flush(tx) => {
                    let _ = tx.send(());
                }
            }
        }
    });
    let m = Arc::new(Mock {
        out,
        script: Mutex::new(script),
        notes: Mutex::new(Vec::new()),
        pending: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(1),
        turns: AtomicU64::new(0),
        current: Mutex::new(None),
    });

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    // 读循环**从不阻塞在剧本上**：`turn/steer` / `turn/interrupt` 必须在一个 turn 正演
    // 到一半（比如卡在 `item/tool/call` 的答复上）的时候也能被处理。
    while let Ok(Some(line)) = lines.next_line().await {
        match decode(&line) {
            Ok(Message::Request(req)) => on_request(&m, req),
            Ok(Message::Response(resp)) => m.resolve(resp),
            Ok(Message::Notification(_)) => {}
            Err(_) => {}
        }
    }
}

fn on_request(m: &Arc<Mock>, req: Request) {
    match req.method.as_str() {
        wire::THREAD_START => m.reply(req.id, json!({ "thread_id": THREAD_ID })),
        wire::TURN_START => {
            let n = m.turns.fetch_add(1, Ordering::SeqCst) + 1;
            let turn_id = format!("u{n}");
            // 先回 turn_id，再开演：剧本发的第一条通知必须排在这条答复后面。
            m.reply(req.id, json!({ "turn_id": turn_id }));
            let steps = m.script.lock().pop_front().unwrap_or_default();
            let mm = m.clone();
            let task = tokio::spawn(run_turn(mm, turn_id, steps));
            if let Some(old) = m.current.lock().replace(task) {
                old.abort();
            }
        }
        wire::TURN_STEER => {
            let text = req.params["text"].as_str().unwrap_or_default().to_string();
            m.note(format!("steer:{text}"));
            m.reply(req.id, json!({}));
        }
        wire::TURN_INTERRUPT => {
            let turn_id = req.params["turn_id"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            m.note(format!("interrupt:{turn_id}"));
            if let Some(t) = m.current.lock().take() {
                t.abort();
            }
            m.notify(
                wire::TURN_COMPLETED,
                json!({"thread_id": THREAD_ID, "turn_id": turn_id, "status": "interrupted"}),
            );
            m.reply(req.id, json!({}));
        }
        other => m.reply_err(req.id, &format!("mock-codex: no method `{other}`")),
    }
}

async fn run_turn(m: Arc<Mock>, turn_id: String, steps: Vec<Step>) {
    let mut status = String::from("completed");
    for step in steps {
        // 当场死掉：不发 `turn/completed`，也不收尾——正是"子进程中途崩了"。
        // 先把已经说出口的话真的写出去（见 [`Wire::Flush`]）。
        if step.exit == Some(true) {
            m.flush().await;
            std::process::exit(9);
        }
        if let Some(text) = step.say {
            say(&m, &turn_id, text);
        }
        if let Some(text) = step.stray {
            // 点名一个**不存在**的 turn：脑那边必须丢掉它，而且不能静悄悄地丢。
            m.notify(
                wire::AGENT_MESSAGE_DELTA,
                json!({"thread_id": THREAD_ID, "turn_id": "u-nobody", "delta": text}),
            );
        }
        if let Some(s) = step.status {
            status = s;
        }
        if let Some(t) = step.tool {
            let call_id = format!("c{}", m.next_id.fetch_add(1, Ordering::SeqCst));
            let _ = m
                .request(
                    wire::TOOL_CALL,
                    json!({"name": t.name, "arguments": t.arguments, "call_id": call_id}),
                )
                .await;
        }
        if let Some(a) = step.approval {
            let item_id = format!("i{}", m.next_id.fetch_add(1, Ordering::SeqCst));
            let reply = m
                .request(
                    wire::REQUEST_APPROVAL,
                    json!({"item_id": item_id, "command": a.command, "cwd": a.cwd}),
                )
                .await;
            let decision = reply["decision"].as_str().unwrap_or("none").to_string();
            m.note(format!("approval:{decision}"));
        }
    }
    m.notify(
        wire::TURN_COMPLETED,
        json!({"thread_id": THREAD_ID, "turn_id": turn_id, "status": status}),
    );
}

/// 一句话：正文一条增量，攒下的笔记再一条，最后一条 `item/completed` 收尾。
/// 分两条增量不是为了好看——它顺手证明了对面真的在**累加**，而不是只留最后一条。
///
/// 笔记那一条**故意不带 `turn_id`**：真的 app-server 不保证每条增量都带它，而脑那边把
/// "没说属于谁"当成当前 turn 才是安全的默认（当成"不属于我"会把全部输出静默丢掉）。
fn say(m: &Mock, turn_id: &str, text: String) {
    m.notify(
        wire::AGENT_MESSAGE_DELTA,
        json!({"thread_id": THREAD_ID, "turn_id": turn_id, "delta": text}),
    );
    let notes: Vec<String> = m.notes.lock().drain(..).collect();
    if !notes.is_empty() {
        m.notify(
            wire::AGENT_MESSAGE_DELTA,
            json!({"thread_id": THREAD_ID, "delta": format!("\n{}", notes.join("\n"))}),
        );
    }
    m.notify(
        wire::ITEM_COMPLETED,
        json!({"item": {"type": wire::ITEM_AGENT_MESSAGE}}),
    );
}

fn load_script() -> VecDeque<Vec<Step>> {
    let Ok(path) = std::env::var("MOCK_CODEX_SCRIPT") else {
        return VecDeque::new();
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => die(&format!("{path}: {e}")),
    };
    match serde_yaml_ng::from_str::<Vec<Vec<Step>>>(&text) {
        Ok(v) => v.into(),
        Err(e) => die(&format!("{path}: {e}")),
    }
}

/// 剧本读不了就当场退出并说清楚为什么。默默演一个空剧本只会让端到端测试以一个
/// 看不出病因的超时收场。
fn die(msg: &str) -> ! {
    eprintln!("mock-codex: {msg}");
    std::process::exit(2)
}
