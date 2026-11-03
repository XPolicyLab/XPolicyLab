use std::io;
use std::sync::Arc;
use std::time::Duration;

use oeh_kernel::clock::ManualClock;
use oeh_kernel::proposal::BoxFuture;
use oeh_kernel::world_log::WorldLog;
use oeh_proto::rpc::RpcError;
use oeh_rpc::client::Client;
use oeh_rpc::codec::MAX_LINE;
use oeh_rpc::conn::Conn;
use oeh_rpc::router::Router;
use oeh_rpc::transport::{spawn_listeners, Admit, ConnRegistry, Listeners};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Echo;
impl Router for Echo {
    fn call(&self, _c: Arc<Conn>, _m: String, p: Value) -> BoxFuture<Result<Value, RpcError>> {
        Box::pin(async move { Ok(p) })
    }
    fn notify(&self, _c: Arc<Conn>, _m: String, _p: Value) {}
}

/// `Result::unwrap_err` needs `T: Debug`, and `Disposer` (`Box<dyn FnOnce() + Send>`) isn't
/// one — a small stand-in that only needs `E: Debug`.
fn expect_err<T, E: std::fmt::Debug>(r: Result<T, E>) -> E {
    match r {
        Ok(_) => panic!("expected an Err"),
        Err(e) => e,
    }
}

#[tokio::test]
async fn uds_and_ws_serve_the_same_router() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let reg = Arc::new(ConnRegistry::default());
    let ws_port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dispose = spawn_listeners(
        Listeners {
            uds: Some(sock.clone()),
            ws: Some(format!("127.0.0.1:{ws_port}").parse().unwrap()),
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log,
        reg.clone(),
    )
    .unwrap();

    let c1 = Client::connect_uds(&sock).await.unwrap();
    assert_eq!(
        c1.call("echo", json!({"via": "uds"})).await.unwrap(),
        json!({"via": "uds"})
    );

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{ws_port}"))
        .await
        .unwrap();
    let c2 = Client::from_transport(oeh_rpc::transport::WsTransport::client(ws));
    assert_eq!(
        c2.call("echo", json!({"via": "ws"})).await.unwrap(),
        json!({"via": "ws"})
    );
    assert_eq!(reg.len(), 2);

    // 先订阅、再 dispose：`Client::notifications()` 是一个 `broadcast::Receiver`，
    // 只收得到订阅**之后**发出的消息（`broadcast::Sender` 对零接收端的发送直接丢弃）。
    // 反过来写的话，这个断言靠的是"dispose() 返回之后通知才被投递"这个时序巧合——而
    // `close_all_connections` 的整个用意正是让通知在 `dispose()` 返回**之前**就送到
    // （Task 3 Fix round 1，H2）。它做得越好，这个测试越容易莫名其妙地失败。
    let mut n = c1.notifications();
    dispose();
    // 关闭后客户端拿到 shutdown 通知然后断连
    let got = tokio::time::timeout(std::time::Duration::from_secs(2), n.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got.method, "shutdown");
    tokio::time::timeout(std::time::Duration::from_secs(2), c1.wait_closed())
        .await
        .unwrap();
    assert!(!sock.exists());
}

#[tokio::test]
async fn stale_socket_file_is_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    // A genuinely *stale* socket file — what a crashed process's leftover UDS socket really
    // looks like on disk (`bind_uds`'s cleanup now only ever removes an actual `AF_UNIX`
    // socket-typed file, never an arbitrary file sitting at a typo'd `uds:` path; Task 3 Fix
    // round 1, L6). Binding a listener and dropping it without unlinking leaves exactly
    // that: closing the fd does not remove the pathname entry.
    {
        let dead = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        drop(dead);
    }
    assert!(sock.exists());
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let dispose = spawn_listeners(
        Listeners {
            uds: Some(sock.clone()),
            ws: None,
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log,
        Arc::new(ConnRegistry::default()),
    )
    .unwrap();
    let c = Client::connect_uds(&sock).await.unwrap();
    assert_eq!(c.call("x", json!(1)).await.unwrap(), json!(1));
    dispose();
}

/// Task 3 Fix round 1, L5: the *other* branch of `bind_uds` — a socket that is genuinely
/// live must not be replaced, and its file must survive a second, refused `spawn_listeners`.
#[tokio::test]
async fn spawn_listeners_refuses_to_replace_a_live_socket() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    // Simulates another process already holding this socket: bound, and (unlike the stale
    // case) never dropped.
    let _live = std::os::unix::net::UnixListener::bind(&sock).unwrap();

    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let err = expect_err(spawn_listeners(
        Listeners {
            uds: Some(sock.clone()),
            ws: None,
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log,
        Arc::new(ConnRegistry::default()),
    ));
    assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
    assert!(sock.exists(), "a live socket's file must not be deleted");
    // The original listener is still exactly as usable as before the refused rebind.
    assert!(std::os::unix::net::UnixStream::connect(&sock).is_ok());
}

/// Task 3 Fix round 1, H1: a later bind failure (ws port taken) must not leave the UDS
/// listener bound in this call running and answering RPCs, and must not leave its socket
/// file on disk — both binds happen before anything is spawned (see `spawn_listeners`'s
/// `Building` guard).
#[tokio::test]
async fn ws_bind_failure_leaves_no_orphaned_uds_listener_or_socket_file() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    // Occupy the ws port so the bind fails.
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let ws_addr = occupied.local_addr().unwrap();

    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let err = expect_err(spawn_listeners(
        Listeners {
            uds: Some(sock.clone()),
            ws: Some(ws_addr),
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log,
        Arc::new(ConnRegistry::default()),
    ));
    assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
    assert!(
        !sock.exists(),
        "the uds socket file must not be left behind when a later bind fails"
    );
    // No leaked, still-answering uds listener from this call either.
    assert!(
        Client::connect_uds(&sock).await.is_err(),
        "there must be no live uds listener left over from the failed spawn_listeners call"
    );
    drop(occupied);
}

/// Task 3 Fix round 1, M1: connections that land concurrently with `dispose()` must not
/// escape both notification and closing — a `closing` flag on `ConnRegistry` refuses late
/// inserts, and `spawn_conn` closes the just-built `Conn` outright when that happens.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn connections_racing_dispose_are_still_closed() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let reg = Arc::new(ConnRegistry::default());
    let dispose = spawn_listeners(
        Listeners {
            uds: Some(sock.clone()),
            ws: None,
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log,
        reg.clone(),
    )
    .unwrap();

    let sock_for_clients = sock.clone();
    let attempts = tokio::spawn(async move {
        let mut clients = Vec::new();
        for _ in 0..40 {
            if let Ok(c) = Client::connect_uds(&sock_for_clients).await {
                clients.push(c);
            }
        }
        clients
    });
    // Give the connect loop a couple of iterations' head start so some connections land
    // *before* dispose(), and some race it.
    tokio::time::sleep(Duration::from_millis(3)).await;
    dispose();
    let clients = attempts.await.unwrap();

    for c in &clients {
        let _ = tokio::time::timeout(Duration::from_secs(2), c.wait_closed()).await;
    }
    // A bounded retry (20 ms steps, 2 s cap) rather than one generous settle window: the
    // last `on_close` hook to run is what this waits for, and a fixed guess either wastes
    // the difference or — on a loaded machine — is simply too short.
    for _ in 0..100 {
        if reg.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        reg.is_empty(),
        "no connection — whether accepted before or racing dispose() — should survive it"
    );
}

/// Task 3 Fix round 1, H2 (per the controller's ruling and answer A): the listener
/// `Disposer` must not just `tokio::spawn` the notify-then-close work and walk away — the
/// caller (`oeh run`'s shutdown path) has nothing left to `.await` afterwards, and
/// `#[tokio::main]` drops the runtime the moment `main` returns, cancelling any still-
/// pending spawned task along with it. This reproduces that shape directly: a multi-thread
/// *server* runtime, a real UDS client on its own, independent runtime/thread (standing in
/// for a separate client process), `dispose()` called synchronously from within the server
/// runtime, and the server runtime dropped immediately afterward.
#[test]
fn shutdown_notification_survives_immediate_runtime_drop() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    let sock_for_client = sock.clone();

    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let (result_tx, result_rx) = std::sync::mpsc::channel::<bool>();
    let client_thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let mut client = None;
            for _ in 0..200 {
                match Client::connect_uds(&sock_for_client).await {
                    Ok(c) => {
                        client = Some(c);
                        break;
                    }
                    Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                }
            }
            let client = client.expect("server socket never came up");
            // Subscribe *before* signalling ready: the server's notify() only reaches
            // receivers that already exist at send time (mirrors `oeh rpc tail`'s own
            // subscribe-before-call ordering, for the same reason).
            let mut notes = client.notifications();
            let _ = ready_tx.send(());
            let got = tokio::time::timeout(Duration::from_secs(2), notes.recv()).await;
            let delivered = matches!(got, Ok(Ok(n)) if n.method == "shutdown");
            let _ = result_tx.send(delivered);
        });
    });

    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let reg = Arc::new(ConnRegistry::default());
    let dispose = server_rt.block_on(async {
        let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
        spawn_listeners(
            Listeners {
                uds: Some(sock.clone()),
                ws: None,
                stdio: false,
                ..Default::default()
            },
            Arc::new(Echo),
            log,
            reg.clone(),
        )
        .unwrap()
    });

    ready_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("client never connected");
    // `Client::connect_uds`'s underlying `connect()` can return as soon as the OS queues the
    // connection in the listening socket's backlog — *before* the server's accept loop task
    // has actually been scheduled to `accept()` it and register it in `reg`. Wait for that
    // registration directly rather than inferring it from the client's own connect success,
    // or `dispose()` can run against a still-empty registry and never notify anyone.
    server_rt.block_on(async {
        for _ in 0..200 {
            if !reg.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    assert_eq!(
        reg.len(),
        1,
        "server never registered the accepted connection"
    );
    // Mirrors `oeh run`: `dispose()` is the last real work done inside the runtime, and
    // nothing after it awaits anything more.
    server_rt.block_on(async {
        dispose();
    });
    drop(server_rt);

    let delivered = result_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("client thread never reported a result");
    client_thread.join().unwrap();
    assert!(
        delivered,
        "the shutdown notification must survive the server runtime being dropped \
         immediately after dispose() returns"
    );
}

/// Minimal hand-rolled RFC 6455 client framing — just enough to send frames tungstenite's
/// own `Message` API cannot: multiple, individually small, unmasked-content (zero mask key,
/// so payload bytes are unchanged) frames that only exceed a size limit once reassembled.
mod raw_ws {
    use super::*;

    pub async fn handshake(port: u16) -> tokio::net::TcpStream {
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        let req = format!(
            "GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n\
             Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
             Sec-WebSocket-Version: 13\r\n\r\n"
        );
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut tail = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            stream.read_exact(&mut byte).await.unwrap();
            tail.push(byte[0]);
            if tail.ends_with(b"\r\n\r\n") {
                break;
            }
        }
        stream
    }

    /// `fin`/`opcode` per RFC 6455 (`0x1` text, `0x0` continuation); always sets the
    /// client-required mask bit, with an all-zero key so the payload on the wire is the
    /// payload the server receives (no XOR bookkeeping needed for a test frame).
    pub async fn write_frame(
        stream: &mut tokio::net::TcpStream,
        fin: bool,
        opcode: u8,
        payload: &[u8],
    ) {
        let mut header = vec![(if fin { 0x80 } else { 0x00 }) | opcode];
        let len = payload.len();
        if len < 126 {
            header.push(0x80 | len as u8);
        } else if len <= 0xFFFF {
            header.push(0x80 | 126);
            header.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            header.push(0x80 | 127);
            header.extend_from_slice(&(len as u64).to_be_bytes());
        }
        header.extend_from_slice(&[0u8; 4]); // masking key: all-zero
        stream.write_all(&header).await.unwrap();
        stream.write_all(payload).await.unwrap();
    }
}

/// Task 3 Fix round 1, M3: a message fragmented across several frames — each comfortably
/// under `MAX_LINE` individually — whose *reassembled total* exceeds it must be rejected,
/// not accepted, parsed and answered on an connection left open for another attempt. This is
/// exactly the gap the review found: the default `WebSocketConfig` bounds a single *frame*
/// to 16 MiB but a *message* to 64 MiB, four times looser than the UDS/stdio `MAX_LINE`.
#[tokio::test]
async fn ws_rejects_a_fragmented_message_over_max_line() {
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    let reg = Arc::new(ConnRegistry::default());
    let ws_port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dispose = spawn_listeners(
        Listeners {
            uds: None,
            ws: Some(format!("127.0.0.1:{ws_port}").parse().unwrap()),
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log,
        reg.clone(),
    )
    .unwrap();

    let mut stream = raw_ws::handshake(ws_port).await;
    // Three frames, each under MAX_LINE on its own, whose *sum* is (just) over it.
    let chunk = vec![b'x'; (MAX_LINE / 3) + 4096];
    raw_ws::write_frame(&mut stream, false, 0x1, &chunk).await; // Text, not final
    raw_ws::write_frame(&mut stream, false, 0x0, &chunk).await; // Continuation, not final
    raw_ws::write_frame(&mut stream, true, 0x0, &chunk).await; // Continuation, final

    // The server must not keep this connection open and answer it — it must tear the
    // connection down instead of assembling, copying and dispatching an over-limit message.
    let mut buf = [0u8; 8];
    let n = tokio::time::timeout(Duration::from_secs(10), stream.read(&mut buf))
        .await
        .expect("server must close the connection rather than hang on an over-limit message")
        .unwrap();
    assert_eq!(
        n, 0,
        "server must close the connection rather than answer an over-limit fragmented message"
    );
    // Same bounded retry as above: the teardown is what we wait for, not a fixed window.
    for _ in 0..100 {
        if reg.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        reg.is_empty(),
        "an over-limit ws message must not leave a live, retriable connection behind"
    );

    dispose();
}

/// Plan 3 最终评审 I2: the number of concurrent connections is bounded, and a connection
/// refused for that reason is told why before it is closed.
///
/// The bound protects the *kernel task*, for the same reason `plugin/register`'s six guard
/// rails do: every connection may register `max_fibers_per_conn` (16) remote fibers, and
/// `FiberTree::settle()` — O(rows × injects), once per `Op::AddFiber` — runs on the one task
/// that also serialises `Op::IngestFrame`. Without a connection bound the global fiber count
/// is `16 × unbounded`. Each connection is also a 4096-slot outbound queue plus two tasks.
///
/// A raw socket, not a `Client`: the refusal is one line written directly onto the transport
/// (a refused connection is never `serve`d, so nothing would ever drain its outbound queue),
/// and `Client`'s notifications are a broadcast — subscribing a moment too late would miss
/// it and make this test a coin flip. `read_to_end` asserts both halves at once: the reason
/// arrives, *and* the peer is then closed rather than left hanging.
#[tokio::test]
async fn connections_over_the_limit_are_told_why_and_closed() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("oeh.sock");
    let log = Arc::new(WorldLog::new(Arc::new(ManualClock::new(0))));
    // `with_max(0)` would be `with_max(1)` anyway — the floor (R4) is asserted below.
    let reg = Arc::new(ConnRegistry::with_max(0));
    assert_eq!(reg.max(), 1, "max_connections has a floor of 1");
    let dispose = spawn_listeners(
        Listeners {
            uds: Some(sock.clone()),
            ws: None,
            stdio: false,
            ..Default::default()
        },
        Arc::new(Echo),
        log.clone(),
        reg.clone(),
    )
    .unwrap();

    let c1 = Client::connect_uds(&sock).await.unwrap();
    assert_eq!(
        c1.call("echo", json!({"n": 1})).await.unwrap(),
        json!({"n": 1})
    );

    let mut raw = tokio::net::UnixStream::connect(&sock).await.unwrap();
    let mut got = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), raw.read_to_end(&mut got))
        .await
        .expect("a refused connection must be closed, not left hanging")
        .unwrap();
    let text = String::from_utf8(got).unwrap();
    let notice: Value = serde_json::from_str(text.trim()).unwrap_or_else(|e| {
        panic!("a refused connection must say why before EOF; got {text:?} ({e})")
    });
    assert_eq!(notice["method"], "shutdown", "{notice}");
    assert_eq!(
        notice["params"]["reason"], "too many connections",
        "{notice}"
    );

    // The limit refuses the newcomer, never the connection that is already there.
    assert_eq!(reg.len(), 1);
    assert_eq!(
        c1.call("echo", json!({"n": 2})).await.unwrap(),
        json!({"n": 2})
    );

    // And an operator reading the log afterwards can see it happened (R3). The event is
    // appended *before* the refusal line is written, so having read that line above is
    // enough to know the append has happened — no sleep, no retry loop.
    let events = log.since(0).unwrap();
    let refused = events
        .iter()
        .find(|e| e.kind.as_str() == "health/rpc_connection_refused")
        .unwrap_or_else(|| panic!("no refusal event in {events:?}"));
    assert_eq!(refused.payload["limit"], 1, "{:?}", refused.payload);
    assert_eq!(refused.payload["refused"], 1, "{:?}", refused.payload);

    dispose();
}

/// Task 4 fix (parked finding (d)): `ConnRegistry::insert`'s "is it closing?" check and its
/// insertion have to happen under the *same* `conns` lock.
///
/// The invariant, stated as the assertion below: **if `insert` returned `true`, that
/// connection is in the snapshot `close_all_connections` takes.** With check-and-insert
/// under one lock it follows directly — an `insert` that saw `closing == false` held the
/// lock before `set_closing` could take it, so its write happened-before the later
/// `iter()`. With the check as a bare atomic load outside the lock, the accepting thread
/// could read `false`, stall on the lock while `set_closing` and the snapshot both went
/// past it, and only then insert itself into a map nobody will ever walk again — exactly
/// the unreachable, never-closed connection M1 exists to prevent.
#[test]
fn an_accepted_connection_is_never_missed_by_the_closing_snapshot() {
    // A barrier, not a bare `spawn`: thread startup costs tens of microseconds while
    // `set_closing` + `iter()` cost tens of nanoseconds, so without one the racer always
    // arrives after the flag is already up, `insert` returns `false`, and the assertion is
    // vacuously true on every iteration — a test that can only pass. Releasing both sides
    // from the same barrier makes the order an actual coin flip, and over this many rounds
    // both outcomes show up.
    let mut accepted_at_least_once = false;
    for i in 0..2_000u64 {
        let reg = Arc::new(ConnRegistry::default());
        let racer = reg.clone();
        let gate = Arc::new(std::sync::Barrier::new(2));
        let racer_gate = gate.clone();
        let inserting = std::thread::spawn(move || {
            racer_gate.wait();
            racer.insert(Conn::new(i, None)) == Admit::Ok
        });
        gate.wait();
        reg.set_closing();
        let snapshot = reg.iter();
        let accepted = inserting.join().unwrap();
        accepted_at_least_once |= accepted;
        assert!(
            !accepted || !snapshot.is_empty(),
            "insert() accepted a connection that the closing snapshot did not see"
        );
        // And the flag really is one-way: nothing gets in afterwards.
        assert_eq!(reg.insert(Conn::new(i + 1_000_000, None)), Admit::Closing);
    }
    assert!(
        accepted_at_least_once,
        "the race never went the interesting way; this test proved nothing"
    );
}
