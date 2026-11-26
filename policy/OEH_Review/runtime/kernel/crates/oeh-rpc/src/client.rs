//! 最小 Rust 客户端：连测试和 `oeh rpc` 用，不是给插件/大脑 SDK 用的（那是 Plan 3 的
//! Python SDK）。跟 `conn.rs` 是镜像关系——同一套 codec，同一套"出站队列 + pending
//! oneshot + 读写各一个任务、用一个 `watch` 互相知会关闭"套路，只是这边发 Request、收
//! Response，还额外要处理内核发起的反向请求（`on_request`）和内核通知的广播
//! （`notifications`）。

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::proposal::BoxFuture;
use oeh_proto::rpc::{code, Id, Message, Notification, Request, Response, RpcError};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::net::UnixStream;
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::codec::{decode, encode, LineTransport, Transport};
use crate::conn::{Closed, SEND_TIMEOUT};

/// 内核反向请求（`approval/request`、`capability/execute`……）的处理函数。
pub type RequestHandler = Arc<dyn Fn(Value) -> BoxFuture<Result<Value, RpcError>> + Send + Sync>;

const OUT_QUEUE_CAPACITY: usize = 4096;
const NOTIFICATIONS_CAPACITY: usize = 1024;

/// `Client::call` 没显式 [`Client::with_timeout`] 时用的默认值（Fix round 1，M4）。
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(60);

struct ClientInner {
    out: mpsc::Sender<String>,
    pending: Mutex<HashMap<Id, oneshot::Sender<Result<Value, RpcError>>>>,
    next_id: AtomicU64,
    handlers: Mutex<HashMap<String, RequestHandler>>,
    notifications: broadcast::Sender<Notification>,
    closed: AtomicBool,
    /// 读/写任务互相通知关闭；也是 `close()` 等收工用的底子（`wait_for` 不怕先关后等，
    /// 见 `conn::Conn::shutdown` 上的注释——这里是同一个理由）。
    shutdown: watch::Sender<bool>,
    /// 读任务、写任务各自的 `JoinHandle`；`close()` 第一个调用者取走它们，真的等两个
    /// 任务都退出（也就是都 drop 掉了自己那一半 transport，对端才会看到 EOF）；并发/
    /// 重复调用 `close()` 的人拿不到（已经被取走），退化成只等 `shutdown` 变 `true`
    /// （`mark_closed` 已经保证这一步不会漏）。
    join: Mutex<Option<(JoinHandle<()>, JoinHandle<()>)>>,
}

impl ClientInner {
    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// 已关闭就直接拒绝，绝不再碰 `out`（跟 `Conn::try_enqueue` 一样的理由：见那边关于
    /// H2 的注释——这里虽然没有 `WorldLog` 观察者链，但同一个短路没有坏处）。
    fn try_enqueue(&self, line: String) -> Result<(), Closed> {
        if self.is_closed() {
            return Err(Closed);
        }
        match self.out.try_send(line) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.mark_closed();
                Err(Closed)
            }
        }
    }

    fn drain_pending(&self) {
        let pending: Vec<_> = std::mem::take(&mut *self.pending.lock())
            .into_iter()
            .collect();
        for (_, tx) in pending {
            let _ = tx.send(Err(RpcError::new(code::REFUSED, "connection closed")));
        }
    }

    fn mark_closed(&self) -> bool {
        let first = !self.closed.swap(true, Ordering::SeqCst);
        if first {
            // `send_replace`, not `send`: `send` is a no-op (drops the value) with zero
            // current receivers, and `send_replace` always stores it regardless (Fix round 2,
            // REG-2 — mirrors `conn.rs`'s `mark_closed`, see the comment there).
            self.shutdown.send_replace(true);
        }
        self.drain_pending();
        first
    }

    fn resolve_pending(&self, resp: Response) {
        // 先把 guard 落到局部变量再 drop：`tx.send` 唤醒等待者的任务，不能带着锁跑过去
        // （Fix round 1，L4）。
        let tx = self.pending.lock().remove(&resp.id);
        let Some(tx) = tx else { return };
        let result = match (resp.result, resp.error) {
            (_, Some(e)) => Err(e),
            (Some(v), None) => Ok(v),
            (None, None) => Err(RpcError::new(
                code::INTERNAL,
                "response without result or error",
            )),
        };
        let _ = tx.send(result);
    }

    fn handler_for(&self, method: &str) -> Option<RequestHandler> {
        self.handlers.lock().get(method).cloned()
    }
}

/// 读任务/写任务收工时的兜底，跟 `conn::CloseOnDrop` 一个道理：不管是正常 `break` 还是
/// 任务体内部意外 panic，`mark_closed()` 都会执行到。
struct CloseOnDrop(Arc<ClientInner>);
impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.mark_closed();
    }
}

/// 一个到内核的连接。`Clone`：内部状态是 `Arc`，克隆只是多一个句柄（超时是每个 `Client`
/// 值自己的设置，不跟着共享状态走）。
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
    default_timeout: Duration,
}

impl Client {
    pub async fn connect_uds(path: &Path) -> io::Result<Client> {
        let stream = UnixStream::connect(path).await?;
        let (r, w) = tokio::io::split(stream);
        Ok(Client::from_transport(LineTransport::new(r, w)))
    }

    pub fn from_transport<T: Transport>(transport: T) -> Client {
        let (read_half, write_half) = Box::new(transport).into_split();
        let (out_tx, out_rx) = mpsc::channel::<String>(OUT_QUEUE_CAPACITY);
        let (notif_tx, _) = broadcast::channel(NOTIFICATIONS_CAPACITY);
        let (shutdown, _) = watch::channel(false);
        let inner = Arc::new(ClientInner {
            out: out_tx,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            handlers: Mutex::new(HashMap::new()),
            notifications: notif_tx,
            closed: AtomicBool::new(false),
            shutdown,
            join: Mutex::new(None),
        });

        let read_inner = inner.clone();
        let mut read_shutdown = read_inner.shutdown.subscribe();
        let read_task = tokio::spawn(async move {
            let mut read_half = read_half;
            let _guard = CloseOnDrop(read_inner.clone());
            loop {
                if *read_shutdown.borrow() {
                    break;
                }
                tokio::select! {
                    line = read_half.recv() => {
                        match line {
                            None | Some(Err(_)) => break,
                            Some(Ok(raw)) => dispatch_line(&read_inner, raw),
                        }
                    }
                    _ = read_shutdown.changed() => break,
                }
            }
        });

        let write_inner = inner.clone();
        let mut write_shutdown = write_inner.shutdown.subscribe();
        let write_task = tokio::spawn(async move {
            let mut write_half = write_half;
            let mut out_rx = out_rx;
            let _guard = CloseOnDrop(write_inner);
            loop {
                if *write_shutdown.borrow() {
                    break;
                }
                tokio::select! {
                    outgoing = out_rx.recv() => {
                        match outgoing {
                            Some(line) => {
                                match tokio::time::timeout(SEND_TIMEOUT, write_half.send(line)).await {
                                    Ok(Ok(())) => {}
                                    _ => break,
                                }
                            }
                            None => break,
                        }
                    }
                    _ = write_shutdown.changed() => break,
                }
            }
        });

        *inner.join.lock() = Some((read_task, write_task));

        Client {
            inner,
            default_timeout: DEFAULT_CALL_TIMEOUT,
        }
    }

    /// 覆盖 [`Client::call`] 的超时（默认 [`DEFAULT_CALL_TIMEOUT`]，60s）。
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, RpcError> {
        let idx = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let id = Id::Num(idx);
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().insert(id.clone(), tx);
        let line = encode(&Message::Request(Request::new(id.clone(), method, params)));
        let _ = self.inner.try_enqueue(line);
        if self.inner.is_closed() {
            self.inner.drain_pending();
        }
        match tokio::time::timeout(self.default_timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(RpcError::new(code::REFUSED, "connection closed")),
            Err(_) => {
                self.inner.pending.lock().remove(&id);
                Err(RpcError::new(code::TIMEOUT, "request timed out"))
            }
        }
    }

    pub fn notify(&self, method: &str, params: Value) -> Result<(), Closed> {
        let line = encode(&Message::Notification(Notification::new(method, params)));
        self.inner.try_enqueue(line)
    }

    /// 处理内核的反向请求（`approval/request`、`capability/execute`……）。同一个方法
    /// 名注册第二次会覆盖第一次。一个 panic 的 handler 会被答复成
    /// `RpcError{INTERNAL, "handler panicked"}`，而不是让内核那头的 `Conn::request`
    /// 干等到超时。
    pub fn on_request(&self, method: &str, f: RequestHandler) {
        self.inner.handlers.lock().insert(method.to_string(), f);
    }

    /// 内核通知（`event`/`brain/turn`/`plugin/state`……）。每个订阅者都是独立的广播接收端；
    /// 跟不上的订阅者会丢消息（`broadcast::error::RecvError::Lagged`），不会拖慢连接。
    pub fn notifications(&self) -> broadcast::Receiver<Notification> {
        self.inner.notifications.subscribe()
    }

    /// 只等，不发起关闭（跟 `Conn::wait_closed` 一个道理）：`wait_for` 先 `borrow()`
    /// 探一次，就算调用的时候连接已经关了也不会永远等下去。用来在服务端主动
    /// `Conn::close()`（比如 `Disposer` 广播完 `shutdown` 之后）时，客户端确认自己
    /// 那头也看到了断连，而不是自己发起关闭。
    pub async fn wait_closed(&self) {
        let mut rx = self.inner.shutdown.subscribe();
        let _ = rx.wait_for(|closed| *closed).await;
    }

    /// 幂等、可以并发调用：请求两个驱动任务退出，第一个调用者真的等它们退出（也就是
    /// 真的 drop 掉了 transport，对端才会看到 EOF）；后来者/并发者等不到 `JoinHandle`
    /// 就退化成只等 `shutdown` 状态（已经由 `mark_closed` 保证置位）。
    pub async fn close(&self) {
        self.inner.mark_closed();
        let handles = self.inner.join.lock().take();
        match handles {
            Some((r, w)) => {
                let _ = tokio::join!(r, w);
            }
            None => {
                let mut rx = self.inner.shutdown.subscribe();
                let _ = rx.wait_for(|v| *v).await;
            }
        }
    }
}

fn dispatch_line(inner: &Arc<ClientInner>, raw: String) {
    match decode(&raw) {
        Ok(Message::Response(resp)) => inner.resolve_pending(resp),
        Ok(Message::Notification(n)) => {
            let _ = inner.notifications.send(n);
        }
        Ok(Message::Request(req)) => {
            let handler = inner.handler_for(&req.method);
            let id = req.id;
            let method = req.method;
            let params = req.params;
            match handler {
                Some(f) => {
                    // 单独任务跑 handler：panic 变成 JoinError 而不是直接杀掉读任务
                    // （对称于 `conn.rs` 里 `router.call` 的处理）。
                    let handle: JoinHandle<Result<Value, RpcError>> =
                        tokio::spawn(async move { f(params).await });
                    let inner = inner.clone();
                    tokio::spawn(async move {
                        let resp = match handle.await {
                            Ok(Ok(v)) => Response::ok(id, v),
                            Ok(Err(e)) => Response::err(id, e),
                            Err(join_err) if join_err.is_panic() => {
                                Response::err(id, RpcError::new(code::INTERNAL, "handler panicked"))
                            }
                            Err(_cancelled) => return,
                        };
                        let _ = inner.try_enqueue(encode(&Message::Response(resp)));
                    });
                }
                None => {
                    let resp = Response::err(id, RpcError::new(code::METHOD_NOT_FOUND, method));
                    let _ = inner.try_enqueue(encode(&Message::Response(resp)));
                }
            }
        }
        Err(_) => {
            // 内核不该发出解不了的行；没有 id 可用来回错误，忽略即可（连接不受影响）。
        }
    }
}
