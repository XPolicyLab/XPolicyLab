//! 线格式：一行一个 JSON-RPC 2.0 信封（`Message`），和收发“一行”的传输抽象。
//!
//! `decode`/`encode` 只管一行文本 ↔ `Message`；谁来提供这一行（UDS、ws、stdio、内存管道）
//! 由 [`Transport`] 抽象，`Conn`/`Client` 的驱动各跑一个读任务、一个写任务，各自独占一半
//! （[`TransportRead`]/[`TransportWrite`]）——两个任务不共享 `&mut`，`select!` 就不可能取消
//! 掉一个正在攒的半行（Fix round 1，C1/H4）。
use std::future::Future;
use std::io;
use std::pin::Pin;

use oeh_proto::rpc::{code, Message, RpcError, JSONRPC};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

/// 一行内容（不含换行符）的最大长度（字节），闭区间——内容正好 `MAX_LINE` 字节合法，
/// `MAX_LINE + 1` 才不合法（Fix round 2，REG-3：round 1 曾经因为拿 `pending.len()`
/// 而不是"换行符在哪"来判断，把这个边界错记成开区间）。超过就当传输错误处理（关闭
/// 连接），不是协议错误。
pub const MAX_LINE: usize = 16 * 1024 * 1024;

/// 把一行文本解析成信封。几类失败分开编码，方便客户端按 `code` 分支：
/// - 空行 → `INVALID_REQUEST`；
/// - 不是合法 JSON → `PARSE`；
/// - `jsonrpc` 不是 `"2.0"` → `INVALID_REQUEST`；
/// - 有 `id` 字段但既不是能塞进 `u64` 的整数也不是字符串（负数、浮点数、`null`……）→
///   `INVALID_REQUEST`——不这么做的话，这种 id 会因为 `Id` 只认 `Num(u64)|Str(String)`、
///   `Message` 是 untagged 枚举、`Notification` 又不管未知字段，静静地被当成
///   Notification 解出来，调用方等一个永远不会来的 Response（Fix round 1，M3）；
/// - 合法 JSON、`jsonrpc` 对、`id` 形状也对，但整体形状既不是 Request/Response/Notification
///   → `INVALID_REQUEST`。
pub fn decode(line: &str) -> Result<Message, RpcError> {
    if line.trim().is_empty() {
        return Err(RpcError::new(code::INVALID_REQUEST, "empty line"));
    }
    let value: Value = serde_json::from_str(line)
        .map_err(|e| RpcError::new(code::PARSE, format!("invalid JSON: {e}")))?;
    match value.get("jsonrpc") {
        Some(Value::String(v)) if v == JSONRPC => {}
        _ => {
            return Err(RpcError::new(
                code::INVALID_REQUEST,
                "jsonrpc must be \"2.0\"",
            ))
        }
    }
    if let Some(id) = value.get("id") {
        let representable = match id {
            Value::Number(n) => n.as_u64().is_some(),
            Value::String(_) => true,
            _ => false,
        };
        if !representable {
            return Err(RpcError::new(
                code::INVALID_REQUEST,
                "id must be an integer or string",
            ));
        }
    }
    serde_json::from_value(value)
        .map_err(|e| RpcError::new(code::INVALID_REQUEST, format!("invalid envelope: {e}")))
}

/// 把信封编码成单行 JSON（不含换行符——加不加 `\n` 是 `TransportWrite::send` 的事）。
pub fn encode(m: &Message) -> String {
    serde_json::to_string(m).expect("Message always serializes")
}

/// `Transport`/`TransportRead`/`TransportWrite` 用的 future 别名：借着 `&mut self`，生命
/// 周期不能是 `'static`（`oeh_kernel::proposal::BoxFuture` 是拥有型、`'static` 的，这里的
/// future 不是）。
pub type TransportFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// 一条连接收方向的一半：只会被一个任务（读任务）独占驱动。
pub trait TransportRead: Send + 'static {
    /// 读下一行（不含换行符）。`None` = 对端干净地关闭（EOF，且没有半行残留）。
    fn recv(&mut self) -> TransportFuture<'_, Option<io::Result<String>>>;
}

/// 一条连接发方向的一半：只会被一个任务（写任务）独占驱动。
pub trait TransportWrite: Send + 'static {
    /// 发一行（自己负责补换行符）。
    fn send(&mut self, line: String) -> TransportFuture<'_, io::Result<()>>;
}

/// 任何能一行一行收发文本、且能拆成独立收发两半的东西：UDS/TCP 流、ws 消息、stdio、
/// `tokio::io::duplex`……`serve`/`Client::from_transport` 各起一个读任务、一个写任务，
/// 分别独占一半，两个任务之间不共享 `&mut`。
pub trait Transport: Send + 'static {
    fn into_split(self: Box<Self>) -> (Box<dyn TransportRead>, Box<dyn TransportWrite>);
}

/// 基于 `AsyncRead`/`AsyncWrite` 的行传输。读半持有一个持久化的字节缓冲区，用
/// `AsyncBufReadExt::fill_buf`/`consume` 一块一块攒，从不用 `read_line`——`fill_buf` 不消费、
/// `consume` 在拷贝之后同步发生，中间没有可以被 `select!` 取消掉、从而丢字节的 `.await`
/// 点（Fix round 1，C1）。`MAX_LINE` 按累计长度增量检查，不必等一整行都进了内存才发现超限
/// （Fix round 1，M1）。写半就是 `W::write_all(line + "\n")`。
pub struct LineTransport<R, W> {
    reader: BufReader<R>,
    writer: W,
}

impl<R, W> LineTransport<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    pub fn new(r: R, w: W) -> Self {
        Self {
            reader: BufReader::new(r),
            writer: w,
        }
    }
}

impl<R, W> Transport for LineTransport<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    fn into_split(self: Box<Self>) -> (Box<dyn TransportRead>, Box<dyn TransportWrite>) {
        let this = *self;
        (
            Box::new(LineReader {
                reader: this.reader,
                pending: Vec::new(),
                scanned: 0,
            }),
            Box::new(LineWriter {
                writer: this.writer,
            }),
        )
    }
}

struct LineReader<R> {
    reader: BufReader<R>,
    /// 还没凑成一整行的字节；跨 `recv()` 调用持久化，`select!` 取消这个 future 不会丢
    /// 已经从底层 reader `consume` 掉的字节——它们已经在这里了，不在 future 自己的栈上。
    pending: Vec<u8>,
    /// `pending[..scanned]` 已经确认过不含 `\n`；下一轮只扫 `pending[scanned..]`。没有
    /// 这个游标的话，每次 `fill_buf` 都要把 `pending` 从头扫一遍找换行符，累计是
    /// O(n²)——一行 16MB 的输入会把 CPU 耗在无谓的重复扫描上（曾经在这实测卡到两分钟）。
    scanned: usize,
}

impl<R: AsyncRead + Unpin + Send + 'static> TransportRead for LineReader<R> {
    fn recv(&mut self) -> TransportFuture<'_, Option<io::Result<String>>> {
        Box::pin(async move {
            loop {
                // Scan for the newline *first*. `pending` can (and, once a peer sends more
                // than one message per `fill_buf` chunk, routinely does) hold the head of the
                // *next* line past the one we're about to return — bounding on
                // `pending.len()` instead of on where the terminator actually is rejects a
                // legal, in-range line just because something after it pushed the buffer's
                // total past `MAX_LINE` (Fix round 2, REG-3; this doesn't need
                // `tokio::io::duplex` to hide it — any reader whose chunk boundaries don't
                // happen to align with a `\n` can trigger it).
                if let Some(rel) = self.pending[self.scanned..]
                    .iter()
                    .position(|&b| b == b'\n')
                {
                    let pos = self.scanned + rel;
                    // `pos` is the newline's 0-based index, i.e. exactly the content length
                    // in bytes — so content of exactly `MAX_LINE` bytes (`pos == MAX_LINE`)
                    // is legal, and this still rejects the round-1 case a single `fill_buf`
                    // chunk both crosses `MAX_LINE` *and* carries the terminator: `pos` would
                    // be `> MAX_LINE` right here, before any of the line gets extracted.
                    if pos > MAX_LINE {
                        return Some(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("line exceeds MAX_LINE ({MAX_LINE} bytes)"),
                        )));
                    }
                    let mut line: Vec<u8> = self.pending.drain(..=pos).collect();
                    line.pop(); // '\n'
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    self.scanned = 0;
                    return Some(
                        String::from_utf8(line)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
                    );
                }
                // No newline yet: only *now* does the accumulated-but-still-incomplete
                // prefix get bounded, so a peer that never sends a terminator can't grow
                // `pending` without limit while we wait for one.
                self.scanned = self.pending.len();
                if self.pending.len() > MAX_LINE {
                    return Some(Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("line exceeds MAX_LINE ({MAX_LINE} bytes)"),
                    )));
                }
                let used = match self.reader.fill_buf().await {
                    Ok([]) => 0,
                    Ok(chunk) => {
                        self.pending.extend_from_slice(chunk);
                        chunk.len()
                    }
                    Err(e) => return Some(Err(e)),
                };
                if used == 0 {
                    return if self.pending.is_empty() {
                        None
                    } else {
                        Some(Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "connection closed with a partial line pending",
                        )))
                    };
                }
                self.reader.consume(used);
            }
        })
    }
}

struct LineWriter<W> {
    writer: W,
}

impl<W: AsyncWrite + Unpin + Send + 'static> TransportWrite for LineWriter<W> {
    fn send(&mut self, line: String) -> TransportFuture<'_, io::Result<()>> {
        Box::pin(async move {
            self.writer.write_all(line.as_bytes()).await?;
            self.writer.write_all(b"\n").await?;
            self.writer.flush().await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oeh_proto::rpc::{Id, Notification, Request};
    use serde_json::json;

    #[test]
    fn empty_line_is_invalid_request() {
        let e = decode("").unwrap_err();
        assert_eq!(e.code, code::INVALID_REQUEST);
        let e = decode("   ").unwrap_err();
        assert_eq!(e.code, code::INVALID_REQUEST);
    }

    #[test]
    fn non_json_is_parse_error() {
        let e = decode("not json").unwrap_err();
        assert_eq!(e.code, code::PARSE);
    }

    #[test]
    fn wrong_jsonrpc_version_is_invalid_request() {
        let e = decode(r#"{"jsonrpc":"1.0","id":1,"method":"x","params":{}}"#).unwrap_err();
        assert_eq!(e.code, code::INVALID_REQUEST);
    }

    #[test]
    fn exotic_ids_are_rejected_rather_than_silently_becoming_notifications() {
        for line in [
            r#"{"jsonrpc":"2.0","id":-1,"method":"echo","params":3}"#,
            r#"{"jsonrpc":"2.0","id":1.5,"method":"echo","params":3}"#,
            r#"{"jsonrpc":"2.0","id":null,"method":"echo","params":3}"#,
        ] {
            let e = decode(line).unwrap_err();
            assert_eq!(e.code, code::INVALID_REQUEST, "line: {line}");
        }
    }

    #[test]
    fn round_trips_request_and_notification() {
        let req = Message::Request(Request::new(Id::Num(1), "echo", json!({"a": 1})));
        let line = encode(&req);
        assert!(!line.contains('\n'));
        assert_eq!(decode(&line).unwrap(), req);

        let note = Message::Notification(Notification::new("ping", json!(null)));
        let line = encode(&note);
        assert_eq!(decode(&line).unwrap(), note);
    }

    /// An `AsyncRead` that hands back at most 8000 bytes per `poll_read` — deliberately not
    /// aligned with `BufReader`'s default 8192-byte internal buffer (unlike
    /// `tokio::io::duplex`, whose chunking happens to hide REG-3: the newline can land on
    /// either side of an 8192-aligned `MAX_LINE` boundary depending on exactly where a real
    /// socket's packets split).
    struct RaggedReader {
        data: Vec<u8>,
        pos: usize,
    }

    impl AsyncRead for RaggedReader {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<io::Result<()>> {
            let this = self.get_mut();
            let remaining = this.data.len() - this.pos;
            if remaining > 0 {
                let n = remaining.min(8000).min(buf.remaining());
                let start = this.pos;
                buf.put_slice(&this.data[start..start + n]);
                this.pos += n;
            }
            std::task::Poll::Ready(Ok(()))
        }
    }

    fn ragged_reader(data: Vec<u8>) -> LineTransport<RaggedReader, tokio::io::Sink> {
        LineTransport::new(RaggedReader { data, pos: 0 }, tokio::io::sink())
    }

    #[tokio::test]
    async fn content_of_exactly_max_line_bytes_is_legal_and_the_next_line_still_arrives() {
        let content = vec![b'x'; MAX_LINE];
        let mut data = content.clone();
        data.push(b'\n');
        data.extend_from_slice(b"short\n");
        let (mut read, _write) = Box::new(ragged_reader(data)).into_split();

        let line1 = read.recv().await.unwrap().unwrap();
        assert_eq!(line1.len(), MAX_LINE);
        assert_eq!(line1.as_bytes(), content.as_slice());

        let line2 = read.recv().await.unwrap().unwrap();
        assert_eq!(line2, "short");
    }

    #[tokio::test]
    async fn content_one_byte_over_max_line_is_rejected() {
        let mut data = vec![b'x'; MAX_LINE + 1];
        data.push(b'\n');
        let (mut read, _write) = Box::new(ragged_reader(data)).into_split();

        let err = read.recv().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
