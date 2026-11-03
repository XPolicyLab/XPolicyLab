use std::sync::Arc;

use oeh_kernel::proposal::BoxFuture;
use oeh_proto::rpc::{code, RpcError};
use oeh_rpc::client::Client;
use oeh_rpc::codec::LineTransport;
use oeh_rpc::conn::{serve, Conn};
use oeh_rpc::router::{parse_params, Router};
use serde_json::{json, Value};

struct Echo;
impl Router for Echo {
    fn call(
        &self,
        conn: Arc<Conn>,
        method: String,
        params: Value,
    ) -> BoxFuture<Result<Value, RpcError>> {
        Box::pin(async move {
            match method.as_str() {
                "echo" => Ok(params),
                "slow" => {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Ok(json!("slow done"))
                }
                "ask_back" => {
                    // 反向请求：让客户端回答
                    let v = conn.request("client/answer", json!({"q": 1})).await?;
                    Ok(json!({"client_said": v}))
                }
                "typed" => {
                    #[derive(serde::Deserialize)]
                    struct P {
                        n: u64,
                    }
                    let p: P = parse_params(&params)?;
                    Ok(json!(p.n + 1))
                }
                _ => Err(RpcError::new(code::METHOD_NOT_FOUND, method)),
            }
        })
    }
    fn notify(&self, conn: Arc<Conn>, method: String, params: Value) {
        if method == "ping" {
            let _ = conn.notify("pong", params);
        }
    }
}

fn pair() -> (Arc<Conn>, Client, tokio::task::JoinHandle<()>) {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    let conn = Conn::new(1, None);
    let server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(Echo),
    ));
    let client = Client::from_transport(LineTransport::new(br, bw));
    (conn, client, server)
}

#[tokio::test]
async fn request_response_and_typed_params() {
    let (_c, client, _s) = pair();
    assert_eq!(
        client.call("echo", json!({"a": 1})).await.unwrap(),
        json!({"a": 1})
    );
    assert_eq!(
        client.call("typed", json!({"n": 41})).await.unwrap(),
        json!(42)
    );
    let e = client.call("typed", json!({"n": "x"})).await.unwrap_err();
    assert_eq!(e.code, code::INVALID_PARAMS);
    let e = client.call("nope", json!({})).await.unwrap_err();
    assert_eq!(e.code, code::METHOD_NOT_FOUND);
}

#[tokio::test]
async fn slow_requests_do_not_block_fast_ones() {
    // Fix round 1, L2: the brief's original assertion (`join!` finishes under 600ms) would
    // also pass if `router.call` were awaited inline in the read loop (slow finishes at
    // ~200ms, fast immediately after, well under 600ms) — it never actually pinned down
    // that `fast` completes *while `slow` is still in flight*. Record completion instants
    // and assert that directly.
    let (_c, client, _s) = pair();
    let t0 = std::time::Instant::now();
    let slow = async {
        let r = client.call("slow", json!({})).await;
        (r, t0.elapsed())
    };
    let fast = async {
        let r = client.call("echo", json!(1)).await;
        (r, t0.elapsed())
    };
    let ((slow_result, slow_at), (fast_result, fast_at)) = tokio::join!(slow, fast);
    assert_eq!(fast_result.unwrap(), json!(1));
    assert_eq!(slow_result.unwrap(), json!("slow done"));
    assert!(
        fast_at < std::time::Duration::from_millis(100),
        "fast call took {fast_at:?}, expected well under the 200ms slow handler"
    );
    assert!(
        fast_at < slow_at,
        "fast ({fast_at:?}) did not complete before slow ({slow_at:?})"
    );
    assert!(t0.elapsed() < std::time::Duration::from_millis(600));
}

#[tokio::test]
async fn reverse_request_reaches_client_handler() {
    let (_c, client, _s) = pair();
    client.on_request(
        "client/answer",
        Arc::new(|p: Value| Box::pin(async move { Ok(json!({"echo": p["q"]})) })),
    );
    let v = client.call("ask_back", json!({})).await.unwrap();
    assert_eq!(v, json!({"client_said": {"echo": 1}}));
}

#[tokio::test]
async fn notifications_flow_both_ways() {
    let (_c, client, _s) = pair();
    let mut rx = client.notifications();
    client.notify("ping", json!({"x": 2})).unwrap();
    let n = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(n.method, "pong");
    assert_eq!(n.params, json!({"x": 2}));
}

#[tokio::test]
async fn closing_the_client_runs_on_close_hooks_and_fails_pending_reverse_requests() {
    let (conn, client, server) = pair();
    let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let r2 = ran.clone();
    conn.on_close(Box::new(move || {
        r2.store(true, std::sync::atomic::Ordering::SeqCst)
    }));
    let pending = conn.request("client/never", json!({}));
    client.close().await;
    server.await.unwrap();
    assert!(conn.is_closed());
    assert!(ran.load(std::sync::atomic::Ordering::SeqCst));
    let e = pending.await.unwrap_err();
    assert_eq!(e.code, code::REFUSED);
}

#[tokio::test]
async fn malformed_lines_get_an_error_response_and_do_not_kill_the_connection() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (mut br, mut bw) = tokio::io::split(b);
    let conn = Conn::new(2, None);
    let _server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(Echo),
    ));
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    bw.write_all(b"not json\n{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"echo\",\"params\":3}\n")
        .await
        .unwrap();
    let mut lines = tokio::io::BufReader::new(&mut br).lines();
    let l1 = lines.next_line().await.unwrap().unwrap();
    let v1: Value = serde_json::from_str(&l1).unwrap();
    assert_eq!(v1["error"]["code"], code::PARSE);
    let l2 = lines.next_line().await.unwrap().unwrap();
    let v2: Value = serde_json::from_str(&l2).unwrap();
    assert_eq!(v2["id"], 5);
    assert_eq!(v2["result"], 3);
}

// ---- Fix round 1 regression tests -----------------------------------------------------

/// C1 repro from the review, reproduced against the *fixed* crate: a request line split
/// across two writes must still be answered even when the kernel has something else to
/// write (an outbound notification) while the line is incomplete. Under the old
/// single-`select!`-loop design, the write arm winning would cancel the in-flight
/// `read_line` and destroy the already-consumed bytes; the split read/write task design
/// makes that structurally impossible (the read task never shares a `select!` arm with the
/// write side).
#[tokio::test]
async fn a_request_split_across_two_writes_is_still_answered_even_with_a_notification_in_between() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (mut br, mut bw) = tokio::io::split(b);
    let conn = Conn::new(10, None);
    let _server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(Echo),
    ));
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    // First segment: no newline yet, so the read task's `pending` buffer holds a partial
    // line.
    bw.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"echo\",\"par")
        .await
        .unwrap();
    bw.flush().await.unwrap();

    // Give the kernel something to write while the request line is still incomplete.
    conn.notify("tick", json!(1)).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Second segment completes the line.
    bw.write_all(b"ams\":3}\n").await.unwrap();

    let mut lines = tokio::io::BufReader::new(&mut br).lines();
    let l1 = lines.next_line().await.unwrap().unwrap();
    let v1: Value = serde_json::from_str(&l1).unwrap();
    assert_eq!(v1["method"], "tick");

    let l2 = lines.next_line().await.unwrap().unwrap();
    let v2: Value = serde_json::from_str(&l2).unwrap();
    assert_eq!(v2["id"], 7);
    assert_eq!(v2["result"], 3);
}

/// H1/H2: a full outbound queue closes the connection and records the backpressure event
/// exactly once (no re-fire, so an `event`-fan-out observer wired up in Task 3 can never
/// recurse into a stack overflow).
#[tokio::test]
async fn backpressure_closes_the_connection_and_logs_exactly_once() {
    let clock: Arc<dyn oeh_kernel::Clock> = Arc::new(oeh_kernel::ManualClock::new(0));
    let log = Arc::new(oeh_kernel::WorldLog::new(clock));
    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count2 = count.clone();
    let _disposer = log.observe(Arc::new(move |ev: &oeh_proto::WorldEvent| {
        if ev.kind.as_str() == "health/rpc_backpressure" {
            count2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }));

    // No `serve()` is running, so nothing ever drains the outbound queue: past its 4096
    // capacity, every further `notify` hits `TrySendError::Full`.
    let conn = Conn::new(11, Some(log));
    for i in 0..4200u32 {
        let _ = conn.notify("spam", json!(i));
    }

    assert!(conn.is_closed());
    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// M1: a line over `MAX_LINE` bytes with no newline closes the connection rather than being
/// buffered without bound.
#[tokio::test]
async fn oversized_line_closes_the_connection() {
    let (a, b) = tokio::io::duplex(1 << 20);
    let (ar, aw) = tokio::io::split(a);
    let (_br, mut bw) = tokio::io::split(b);
    let conn = Conn::new(12, None);
    let server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(Echo),
    ));
    use tokio::io::AsyncWriteExt;
    let huge = vec![b'x'; oeh_rpc::codec::MAX_LINE + 10];
    for chunk in huge.chunks(1 << 16) {
        if bw.write_all(chunk).await.is_err() {
            break;
        }
    }
    let _ = bw.write_all(b"\n").await;
    tokio::time::timeout(std::time::Duration::from_secs(10), server)
        .await
        .expect("serve() should finish once the oversized line is detected")
        .unwrap();
    assert!(conn.is_closed());
}

/// Brief rule 4: `Conn::request`'s reverse-request ids are `Id::Str("k{n}")`, sequential
/// from 0, regardless of whether anyone ever answers them.
#[tokio::test]
async fn kernel_reverse_request_ids_are_sequential() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (mut br, bw) = tokio::io::split(b);
    let conn = Conn::new(13, None);
    let _server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(Echo),
    ));
    let _f0 = conn.request("client/never", json!({}));
    let _f1 = conn.request("client/never", json!({}));

    use tokio::io::AsyncBufReadExt;
    let mut lines = tokio::io::BufReader::new(&mut br).lines();
    let l0 = lines.next_line().await.unwrap().unwrap();
    let v0: Value = serde_json::from_str(&l0).unwrap();
    assert_eq!(v0["id"], "k0");
    let l1 = lines.next_line().await.unwrap().unwrap();
    let v1: Value = serde_json::from_str(&l1).unwrap();
    assert_eq!(v1["id"], "k1");
    drop(bw);
}

/// `on_close` hooks run in reverse registration order.
#[tokio::test]
async fn on_close_hooks_run_in_reverse_registration_order() {
    let (conn, client, server) = pair();
    let order = Arc::new(std::sync::Mutex::new(Vec::<i32>::new()));
    for i in [1, 2, 3] {
        let order = order.clone();
        conn.on_close(Box::new(move || order.lock().unwrap().push(i)));
    }
    client.close().await;
    server.await.unwrap();
    assert_eq!(*order.lock().unwrap(), vec![3, 2, 1]);
}

struct PanicOnNotify;
impl Router for PanicOnNotify {
    fn call(
        &self,
        _conn: Arc<Conn>,
        _method: String,
        _params: Value,
    ) -> BoxFuture<Result<Value, RpcError>> {
        Box::pin(async move { Ok(Value::Null) })
    }
    fn notify(&self, _conn: Arc<Conn>, _method: String, _params: Value) {
        panic!("router.notify boom");
    }
}

/// H3: a panic inside `router.notify` still closes the connection and runs `on_close`
/// hooks, instead of unwinding past the teardown and leaking the connection forever.
#[tokio::test]
async fn panicking_notify_handler_still_closes_the_connection_and_runs_hooks() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    let conn = Conn::new(14, None);
    let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ran2 = ran.clone();
    conn.on_close(Box::new(move || {
        ran2.store(true, std::sync::atomic::Ordering::SeqCst)
    }));
    let server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(PanicOnNotify),
    ));
    let client = Client::from_transport(LineTransport::new(br, bw));
    client.notify("anything", json!({})).unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(5), conn.wait_closed())
        .await
        .unwrap();
    assert!(conn.is_closed());
    assert!(ran.load(std::sync::atomic::Ordering::SeqCst));
    tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}

struct NeverRead;
impl oeh_rpc::codec::TransportRead for NeverRead {
    fn recv(&mut self) -> oeh_rpc::codec::TransportFuture<'_, Option<std::io::Result<String>>> {
        Box::pin(std::future::pending())
    }
}
struct NeverWrite;
impl oeh_rpc::codec::TransportWrite for NeverWrite {
    fn send(&mut self, _line: String) -> oeh_rpc::codec::TransportFuture<'_, std::io::Result<()>> {
        Box::pin(std::future::pending())
    }
}
struct NeverTransport;
impl oeh_rpc::codec::Transport for NeverTransport {
    fn into_split(
        self: Box<Self>,
    ) -> (
        Box<dyn oeh_rpc::codec::TransportRead>,
        Box<dyn oeh_rpc::codec::TransportWrite>,
    ) {
        (Box::new(NeverRead), Box::new(NeverWrite))
    }
}

/// H4: a write half that never completes (peer stopped reading, or a wedged transport)
/// closes the connection once `SEND_TIMEOUT` elapses, rather than parking the write task
/// (and the read task, and `Client::close()`) forever.
#[tokio::test(start_paused = true)]
async fn a_write_that_never_completes_closes_the_connection_after_send_timeout() {
    let conn = Conn::new(15, None);
    let _server = tokio::spawn(serve(conn.clone(), NeverTransport, Arc::new(Echo)));
    conn.notify("x", json!(1)).unwrap();

    // Paused time auto-advances once the runtime has nothing progressable left except
    // timers, straight to the next one due — which is the write task's `SEND_TIMEOUT`.
    // The outer timeout here is a generous multiple of `SEND_TIMEOUT` (not an equal
    // duration — an equal one raced against it and was a coin flip on which fired first).
    tokio::time::timeout(oeh_rpc::conn::SEND_TIMEOUT * 4, conn.wait_closed())
        .await
        .expect("connection should close once SEND_TIMEOUT elapses");
    assert!(conn.is_closed());
}

struct NeverAnswers;
impl Router for NeverAnswers {
    fn call(
        &self,
        _conn: Arc<Conn>,
        _method: String,
        _params: Value,
    ) -> BoxFuture<Result<Value, RpcError>> {
        Box::pin(std::future::pending())
    }
    fn notify(&self, _conn: Arc<Conn>, _method: String, _params: Value) {}
}

/// M4: `Client::call` times out with `code::TIMEOUT` rather than hanging forever when the
/// kernel never answers.
#[tokio::test(start_paused = true)]
async fn client_call_times_out_when_the_kernel_never_answers() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    let conn = Conn::new(16, None);
    let _server = tokio::spawn(serve(
        conn,
        LineTransport::new(ar, aw),
        Arc::new(NeverAnswers),
    ));
    let client = Client::from_transport(LineTransport::new(br, bw))
        .with_timeout(std::time::Duration::from_millis(200));

    let call = tokio::spawn(async move { client.call("whatever", json!({})).await });
    tokio::time::advance(std::time::Duration::from_millis(250)).await;
    let e = call.await.unwrap().unwrap_err();
    assert_eq!(e.code, code::TIMEOUT);
}

// ---- Fix round 2 regression tests -----------------------------------------------------

/// REG-1: cancelling `serve()` (a listener racing it in a `select!` against a shutdown
/// signal, or — as here — wrapping it in a deadline it doesn't meet) must not leak the
/// connection. Before the fix, dropping `serve()`'s future only dropped the two
/// `JoinHandle`s, which *detaches* rather than stops the spawned read/write tasks: the
/// connection kept answering RPCs forever with `is_closed() == false` and the `on_close`
/// hooks never ran.
#[tokio::test]
async fn cancelling_serve_still_closes_the_connection_and_runs_hooks_once() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    let conn = Conn::new(20, None);
    let hook_runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hook_runs2 = hook_runs.clone();
    conn.on_close(Box::new(move || {
        hook_runs2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }));
    let client = Client::from_transport(LineTransport::new(br, bw));

    // Nothing here ever gives `serve()` a reason to finish on its own (no EOF, no
    // backpressure, no panic) — the 100ms deadline elapsing is the only thing that ends it,
    // and it ends it by dropping the `serve()` future, exactly like a listener's
    // `tokio::select! { _ = serve(..) => {}, _ = shutdown.changed() => {} }` would.
    let result = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        serve(conn.clone(), LineTransport::new(ar, aw), Arc::new(Echo)),
    )
    .await;
    assert!(
        result.is_err(),
        "serve() should not have finished on its own"
    );

    assert!(conn.is_closed());
    assert_eq!(hook_runs.load(std::sync::atomic::Ordering::SeqCst), 1);

    // The abort dropped both of the server's transport halves; the client's own driver
    // should notice (EOF) and fail rather than the RPC just silently still working.
    let e = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        client.call("echo", json!(1)),
    )
    .await
    .expect("client.call should not hang once the server side is gone")
    .unwrap_err();
    assert_eq!(e.code, code::REFUSED);
}

/// REG-2: `watch::Sender::send` is a documented no-op (drops the value) with zero current
/// receivers. A `Conn` that closes via backpressure before `serve()` ever runs (or between
/// `serve()` calls) has zero receivers at that moment, so the round-1 fix's `send(true)`
/// left the watch at `false` forever — `wait_closed()` calls made afterward hung, and
/// starting `serve()` on that `Conn` would have served it as if still open.
#[tokio::test]
async fn a_close_with_no_subscribers_is_still_observed_by_a_later_wait_closed_and_serve() {
    let conn = Conn::new(21, None);
    // Nobody has subscribed to the shutdown watch yet (no `serve()`, no `wait_closed()`
    // caller in flight) when this fills the queue and closes the connection.
    for i in 0..4200u32 {
        let _ = conn.notify("spam", json!(i));
    }
    assert!(conn.is_closed());

    // Subscribes *after* the close already happened.
    tokio::time::timeout(std::time::Duration::from_millis(100), conn.wait_closed())
        .await
        .expect("wait_closed() should observe the already-closed state, not hang");

    // Starting `serve()` on an already-closed `Conn` must not actually serve anything: both
    // tasks subscribe fresh and should see the stored `true` immediately.
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    drop(b);
    tokio::time::timeout(
        std::time::Duration::from_millis(100),
        serve(conn.clone(), LineTransport::new(ar, aw), Arc::new(Echo)),
    )
    .await
    .expect("serve() on an already-closed Conn should exit immediately");
}

/// REG-4: `Conn::request_timeout` must not leak its `pending` entry when the timeout fires —
/// `Client::call` already cleaned up on timeout (round 1); this brings the two in line.
#[tokio::test]
async fn request_timeout_removes_its_pending_entry_on_elapse() {
    let conn = Conn::new(22, None);
    let e = conn
        .request_timeout(
            "client/never",
            json!({}),
            std::time::Duration::from_millis(20),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, code::TIMEOUT);
    assert_eq!(conn.pending_len(), 0);
}

/// Task 5: the same entry must also go when the **caller** drops the future before it
/// resolves — which is the normal way a reverse request ends, not an exotic one:
/// `Admission` wraps `RpcApproval::decide` in its own `ask_timeout_ms` and drops it when
/// that elapses, and an aborted execution drops an in-flight `capability/execute`. Neither
/// path runs a single line of the future's body, so only `Drop` can clean up.
#[tokio::test]
async fn a_dropped_request_future_removes_its_pending_entry_too() {
    let conn = Conn::new(23, None);
    let fut = conn.request_timeout(
        "client/never",
        json!({}),
        std::time::Duration::from_secs(60),
    );
    assert_eq!(conn.pending_len(), 1, "the request is registered up front");
    drop(fut);
    assert_eq!(
        conn.pending_len(),
        0,
        "dropping the future must not leak it"
    );
}

/// Task 3 Fix round 1, M2 (the `--rpc-stdio` "write one request, close stdin" repro): the
/// peer can hang up (here: drop its write half, exactly what a closed `stdin` pipe looks
/// like from the read task's side) *before* a request it already sent has finished being
/// processed. The read task's EOF must not race the still-in-flight response — it has to
/// win. This went through two designs before landing here (see `conn.rs`'s module doc and
/// the write task's drain-loop comment): a fixed idle-window guess was too short under real
/// load, and even after replacing it with an exact `Conn::in_flight` counter, the response
/// was still silently dropped by `try_enqueue`'s own `draining` check — `Conn::
/// try_enqueue_response` (bypassing that check, since the response is for work already
/// counted, not new work) is what actually closes the gap.
#[tokio::test]
async fn eof_while_a_request_is_still_being_handled_still_delivers_the_response() {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let conn = Conn::new(30, None);
    let server = tokio::spawn(serve(
        conn.clone(),
        LineTransport::new(ar, aw),
        Arc::new(Echo),
    ));

    let (br, mut bw) = tokio::io::split(b);
    // `Echo`'s "slow" method sleeps 200ms before answering — long enough that the read
    // task's next `recv()` (returning EOF, since the write half below is dropped right
    // after this) reliably runs *while* the response is still being computed, not after.
    let request = r#"{"jsonrpc":"2.0","id":1,"method":"slow","params":{}}"#;
    tokio::io::AsyncWriteExt::write_all(&mut bw, request.as_bytes())
        .await
        .unwrap();
    tokio::io::AsyncWriteExt::write_all(&mut bw, b"\n")
        .await
        .unwrap();
    // Half-close immediately: from the server's read task, this looks exactly like a peer
    // that wrote one request and hung up (or, for `--rpc-stdio`, a pipe whose writer already
    // exited) before the response was ready. Merely `drop`ping `bw` would not do it — the
    // generic `tokio::io::split` shares the one underlying `DuplexStream` between both
    // halves behind a lock, so the object (and thus the connection) stays alive as long as
    // `br` is; an explicit `shutdown()` is what actually signals EOF to the peer.
    tokio::io::AsyncWriteExt::shutdown(&mut bw).await.unwrap();
    drop(bw);

    let mut received = Vec::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        tokio::io::AsyncReadExt::read_to_end(&mut { br }, &mut received),
    )
    .await
    .expect("reading the response must not hang")
    .unwrap();
    let text = String::from_utf8(received).unwrap();
    assert!(
        text.contains(r#""result":"slow done""#),
        "the response to a request that was still in flight when the peer hung up must \
         still be delivered before the connection closes; got {text:?}"
    );

    tokio::time::timeout(std::time::Duration::from_secs(2), server)
        .await
        .expect("serve() must still terminate")
        .unwrap();
    assert!(conn.is_closed());
}

/// 排水中的连接上发起一次反向请求，必须**当场**被拒，而不是干等到它自己的超时。
///
/// `close_after_drain` 的定义就是"不再收新工作"：`try_enqueue` 会拒掉这一行，对端
/// 永远看不到这条请求，也就永远不会答复。这里等的是一个机器人动作的裁决
/// （`approval/request`、`capability/execute`），默认超时 60 s——把那 60 秒的静默换成
/// 一个当场的 `REFUSED`，脑才能改走别的路。
///
/// 判据不是"快"而是"不靠超时"：请求本身给的超时是 60 s，外面只等 2 s。旧实现里这条
/// 测试会卡满那 2 s 然后失败。
#[tokio::test]
async fn a_reverse_request_on_a_draining_connection_is_refused_at_once() {
    let conn = Conn::new(31, None);
    // 没有 `serve()` 在跑，所以排水只会在下面那次 `close()` 时收尾。
    let draining = tokio::spawn({
        let conn = conn.clone();
        async move {
            conn.close_after_drain(std::time::Duration::from_secs(5))
                .await
        }
    });
    // 有界重试等排水置位（最多 2 s），不是一个固定的 settle 窗口。
    for _ in 0..100 {
        if conn.is_draining() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(conn.is_draining(), "close_after_drain never set draining");

    let e = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        conn.request_timeout(
            "client/answer",
            json!({}),
            std::time::Duration::from_secs(60),
        ),
    )
    .await
    .expect("a reverse request on a draining connection must not wait for its own timeout")
    .unwrap_err();
    assert_eq!(e.code, code::REFUSED);
    assert_eq!(
        conn.pending_len(),
        0,
        "a refused request leaves nothing behind"
    );

    conn.close();
    draining.await.unwrap();
}
