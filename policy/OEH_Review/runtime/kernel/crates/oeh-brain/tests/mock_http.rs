//! 传输测试的替身：一个最小 HTTP/1.1 服务器（[`MockHttp`]）和一个最小 WebSocket 服务器
//! （[`ws::MockWs`]）。
//!
//! 手写而不是引一个 web 框架：这两个东西要做的只有"把预置的字节按顺序写出去、把收到的
//! 请求原样记下来"，而传输测试要断言的恰恰是**字节**——SSE 的分片边界、请求体的形状。
//! 框架会把这两件事都藏起来（自动分块、自动反序列化），正好藏掉了被测的那一层。
//!
//! 全部绑在 `127.0.0.1:0` 上（内核分配端口），不碰外网，也不和别的测试抢固定端口。
//!
//! 这个文件同时也是一个**空的测试目标**（`tests/*.rs` 每个都是一个 crate）：里面的东西
//! 只在被 `mod mock_http;` 引进来的那些测试里用得上，所以整文件 `allow(dead_code)`。

#![allow(dead_code)]

use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// 服务器收到的一次请求。`body` 解不出 JSON 就是 [`Value::Null`]——断言会因此失败并把
/// 整个请求打出来，比在替身里 `unwrap` 炸掉有用。
#[derive(Clone, Debug)]
pub struct Recorded {
    pub path: String,
    /// header 名全部小写（HTTP 的 header 名大小写不敏感，断言不该跟着客户端的拼法走）。
    pub headers: BTreeMap<String, String>,
    pub body: Value,
}

/// 一次请求的回答。
pub enum Reply {
    /// `200` + `text/event-stream`，按给定的分片依次写出。**分片边界可以落在一行 SSE 的
    /// 中间**：传输必须自己把跨分片的行拼回来。
    Sse(Vec<String>),
    /// 一个状态码 + 文本 body（重试语义的用例要 5xx / 4xx）。
    Status(u16, String),
}

impl Reply {
    /// 一整段 SSE 一次写完。
    pub fn sse(body: &str) -> Reply {
        Reply::Sse(vec![body.to_string()])
    }
}

pub struct MockHttp {
    addr: SocketAddr,
    reqs: Arc<Mutex<Vec<Recorded>>>,
    task: tokio::task::JoinHandle<()>,
}

impl MockHttp {
    /// 起一个服务器：第 n 个连接拿 `replies[n]`，用完之后一律 503。
    pub async fn start(replies: Vec<Reply>) -> MockHttp {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let reqs = Arc::new(Mutex::new(Vec::new()));
        let r = reqs.clone();
        let mut queue: VecDeque<Reply> = replies.into();
        let task = tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                let reply = queue.pop_front();
                // 一个连接一次请求（回答都带 `Connection: close`），串行处理够用，
                // 而且让"第 n 个请求拿第 n 份回答"是确定的。
                if let Err(e) = serve(sock, &r, reply).await {
                    eprintln!("mock http: {e}");
                }
            }
        });
        MockHttp { addr, reqs, task }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.reqs.lock().clone()
    }

    /// 最后一次请求。没有请求就 panic——断言的前提没满足时，越早炸越好懂。
    pub fn last(&self) -> Recorded {
        self.reqs.lock().last().cloned().expect("没有收到任何请求")
    }
}

impl Drop for MockHttp {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(
    mut sock: TcpStream,
    reqs: &Mutex<Vec<Recorded>>,
    reply: Option<Reply>,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i + 4;
        }
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await?;
        if n == 0 {
            // 对方没发完就关了（reqwest 的连接探测就是这样）：不记账，也不算错。
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let start = lines.next().unwrap_or_default();
    let path = start.split(' ').nth(1).unwrap_or("/").to_string();
    let mut headers = BTreeMap::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end..].to_vec();
    while body.len() < len {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    reqs.lock().push(Recorded {
        path,
        headers,
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
    });
    match reply {
        Some(Reply::Sse(chunks)) => {
            sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            )
            .await?;
            for c in chunks {
                sock.write_all(c.as_bytes()).await?;
                sock.flush().await?;
            }
        }
        Some(Reply::Status(code, text)) => {
            let head = format!(
                "HTTP/1.1 {code} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                reason(code),
                text.len()
            );
            sock.write_all(head.as_bytes()).await?;
            sock.write_all(text.as_bytes()).await?;
        }
        None => {
            sock.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await?;
        }
    }
    sock.shutdown().await
}

fn reason(code: u16) -> &'static str {
    match code {
        400 => "Bad Request",
        401 => "Unauthorized",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// realtime 传输的替身。
pub mod ws {
    use std::collections::BTreeMap;
    use std::net::SocketAddr;
    use std::sync::Arc;

    use futures_util::{SinkExt, StreamExt};
    use parking_lot::Mutex;
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    use tokio_tungstenite::tungstenite::Message;

    pub struct MockWs {
        addr: SocketAddr,
        got: Arc<Mutex<Vec<Value>>>,
        headers: Arc<Mutex<BTreeMap<String, String>>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl MockWs {
        /// 接一个连接：收到的每条文本消息都记下来，看到 `response.create` 之后把 `script`
        /// 按顺序发出去，然后关闭。
        ///
        /// "等 `response.create` 再发"不是装饰：realtime 的事件顺序就是这样，而这也让
        /// 测试不需要任何 sleep——服务器的第一条消息严格在客户端发完请求之后。
        pub async fn start(script: Vec<Value>) -> MockWs {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("local_addr");
            let got = Arc::new(Mutex::new(Vec::new()));
            let headers = Arc::new(Mutex::new(BTreeMap::new()));
            let (g, h) = (got.clone(), headers.clone());
            let task = tokio::spawn(async move {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let hs = h.clone();
                // 错误类型是 tungstenite 的 `ErrorResponse`（一整个 HTTP 响应），这个
                // 回调的签名由它的 `Callback` trait 定死，不是我们能改小的。
                #[allow(clippy::result_large_err)]
                let cb = move |req: &Request, resp: Response| {
                    let mut m = hs.lock();
                    for (k, v) in req.headers() {
                        m.insert(
                            k.as_str().to_ascii_lowercase(),
                            String::from_utf8_lossy(v.as_bytes()).into_owned(),
                        );
                    }
                    Ok(resp)
                };
                let Ok(mut stream) = tokio_tungstenite::accept_hdr_async(sock, cb).await else {
                    return;
                };
                while let Some(Ok(msg)) = stream.next().await {
                    let Message::Text(t) = msg else { continue };
                    let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                    let go = v["type"] == "response.create";
                    g.lock().push(v);
                    if !go {
                        continue;
                    }
                    for m in &script {
                        if stream.send(Message::text(m.to_string())).await.is_err() {
                            return;
                        }
                    }
                    let _ = stream.close(None).await;
                    return;
                }
            });
            MockWs {
                addr,
                got,
                headers,
                task,
            }
        }

        /// 带路径的 URL：真实的 realtime 端点是 `wss://…/v1/realtime`，而一个没有路径
        /// 的 ws URL 连请求行都拼不出来。
        pub fn url(&self) -> String {
            format!("ws://{}/v1/realtime", self.addr)
        }

        pub fn received(&self) -> Vec<Value> {
            self.got.lock().clone()
        }

        pub fn handshake_headers(&self) -> BTreeMap<String, String> {
            self.headers.lock().clone()
        }
    }

    impl Drop for MockWs {
        fn drop(&mut self) {
            self.task.abort();
        }
    }
}
