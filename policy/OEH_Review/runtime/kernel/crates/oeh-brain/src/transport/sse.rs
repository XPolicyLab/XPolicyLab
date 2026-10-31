//! SSE 切分：HTTP 的字节流 → 一条条 `event:` / `data:`。
//!
//! 三家 HTTP 厂商都用 Server-Sent Events 发流，格式是同一套：若干行 `字段: 值`，空行
//! 分事件，`data` 可以有多行（按规范用 `\n` 连起来）。
//!
//! **分片边界和行边界没有任何关系。** 一次 `bytes_stream()` 的分片可能是半行、也可能
//! 是三条半事件——这是网络的事实，不是厂商的选择。所以这里是一个带缓冲的增量解析器：
//! [`SseDecoder::push`] 只吐出"已经完整"的事件，剩下的半行留在缓冲里等下一片。这一段
//! 正是最容易写成"按 `\n\n` split 一下"然后在生产环境里偶尔丢半个工具调用的地方。
//!
//! **两个上限。** 缓冲是按对面发的字节长的，而"对面"可能是一个坏掉的网关：一条永远
//! 不换行的 `data:`、或者没完没了的 `data:` 续行，都会让缓冲无限涨到进程被 OOM 杀掉。
//! 失败路径这边早就有界了（`transport::MAX_ERROR_BODY`，理由是"对面可能回一整页
//! HTML"），成功路径没有理由更宽松。超界就是一条 [`ModelError::Protocol`]：瀑布看见
//! 协议错不重试、直接降级，这个 turn 以 `done("error: …")` 收尾——fail closed。

use crate::model::ModelError;

/// 一条**行**在缓冲里最多攒这么多字节（还没等到 `\n`）。
pub const MAX_SSE_LINE: usize = 256 * 1024;

/// 一条**事件**的 `data` 最多攒这么多字节（多行 `data:` 拼起来之后）。
pub const MAX_SSE_EVENT: usize = 256 * 1024;

/// 一条 SSE 事件。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// `event:` 行。Anthropic 与 Responses 用它分类型（正文里一般也有 `type` 字段），
    /// OpenAI 的 chat_completions 根本不发这一行。
    pub event: Option<String>,
    /// `data:` 行的正文，多行按规范用 `\n` 连接。
    pub data: String,
}

#[derive(Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    event: Option<String>,
    data: String,
    got_data: bool,
}

impl SseDecoder {
    pub fn new() -> SseDecoder {
        SseDecoder::default()
    }

    /// 喂一片字节，拿回这一片让它**完整**了的那些事件（可能是 0 条，也可能是好几条）。
    ///
    /// 超过 [`MAX_SSE_LINE`] / [`MAX_SSE_EVENT`] 就是一条 [`ModelError::Protocol`]，这个
    /// 流到此为止（调用方 [`crate::transport::stream_of`] 收到 `Err` 就不再喂了）。
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ModelError> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(i) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=i).collect();
            if let Some(ev) = self.line(&line[..line.len() - 1])? {
                out.push(ev);
            }
        }
        // 剩下的是半行。它只会越攒越长——一条不换行的流在这里被截住，而不是在 OOM 那里。
        if self.buf.len() > MAX_SSE_LINE {
            return Err(too_large("sse line", self.buf.len(), MAX_SSE_LINE));
        }
        Ok(out)
    }

    /// 流结束了：缓冲里剩下的半行按一行处理，没等到空行的那条事件也交出来。
    ///
    /// 真实的流一定以空行收尾，但"连接被中途关掉"时最后一条事件常常就差那个空行——
    /// 丢掉它等于丢掉整个回合的最后一句话。
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, ModelError> {
        let rest: Vec<u8> = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        if !rest.is_empty() {
            if let Some(ev) = self.line(&rest)? {
                out.push(ev);
            }
        }
        if let Some(ev) = self.take() {
            out.push(ev);
        }
        Ok(out)
    }

    /// 一行（不含行尾的 `\n`）。空行 = 发事件。
    fn line(&mut self, raw: &[u8]) -> Result<Option<SseEvent>, ModelError> {
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        // 线上数据不 `unwrap`：非法 UTF-8 用替换字符顶上，整条流不该因为一个坏字节断掉。
        let line = String::from_utf8_lossy(raw);
        if line.is_empty() {
            return Ok(self.take());
        }
        // `:` 开头是注释（很多厂商拿它当心跳）。
        if line.starts_with(':') {
            return Ok(None);
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line.as_ref(), ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => {
                if self.got_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.got_data = true;
                // 多行 `data:` 是规范允许的，"无穷多行 `data:`"不是。
                if self.data.len() > MAX_SSE_EVENT {
                    return Err(too_large("sse event", self.data.len(), MAX_SSE_EVENT));
                }
            }
            // `id:` / `retry:` 与未知字段：SSE 规范说忽略。
            _ => {}
        }
        Ok(None)
    }

    fn take(&mut self) -> Option<SseEvent> {
        if !self.got_data {
            // 只有 `event:` 没有 `data:` 的空事件不往上传——厂商用它做心跳。
            self.event = None;
            return None;
        }
        self.got_data = false;
        Some(SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data),
        })
    }
}

/// 超界的那条协议错。字节数进 `reason`，正文不进——它会一路走到世界日志。
fn too_large(what: &str, got: usize, max: usize) -> ModelError {
    ModelError::Protocol(format!("{what} too large ({got} bytes, max {max})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(event: Option<&str>, data: &str) -> SseEvent {
        SseEvent {
            event: event.map(str::to_string),
            data: data.to_string(),
        }
    }

    #[test]
    fn a_whole_body_splits_into_events() {
        let mut d = SseDecoder::new();
        let out = d
            .push(b"event: a\ndata: {\"x\":1}\n\ndata: [DONE]\n\n")
            .unwrap();
        assert_eq!(out, vec![ev(Some("a"), "{\"x\":1}"), ev(None, "[DONE]")]);
    }

    /// 被测的那件事：**每一个字节**单独到达，事件还是那两条。
    #[test]
    fn an_event_split_across_chunks_is_still_one_event() {
        let body = "event: a\ndata: {\"x\":\ndata: 1}\n\ndata: second\n\n";
        let mut d = SseDecoder::new();
        let mut out = Vec::new();
        for b in body.as_bytes() {
            out.extend(d.push(&[*b]).unwrap());
        }
        // 两行 `data:` 按规范用 `\n` 连起来。
        assert_eq!(out, vec![ev(Some("a"), "{\"x\":\n1}"), ev(None, "second")]);
    }

    #[test]
    fn crlf_comments_and_unknown_fields_do_not_show_up() {
        let mut d = SseDecoder::new();
        let out = d
            .push(b": ping\r\nid: 7\r\nretry: 100\r\ndata: hi\r\n\r\n")
            .unwrap();
        assert_eq!(out, vec![ev(None, "hi")]);
    }

    #[test]
    fn a_stream_that_ends_without_a_blank_line_still_yields_its_last_event() {
        let mut d = SseDecoder::new();
        assert!(d.push(b"data: tail").unwrap().is_empty());
        assert_eq!(d.finish().unwrap(), vec![ev(None, "tail")]);
    }

    #[test]
    fn an_event_line_without_data_is_not_an_event() {
        let mut d = SseDecoder::new();
        assert!(d.push(b"event: ping\n\n").unwrap().is_empty());
        assert!(d.finish().unwrap().is_empty());
    }

    /// 一条永远不换行的 `data:`：缓冲被截住，不是涨到进程被杀。
    #[test]
    fn a_line_that_never_ends_is_a_protocol_error() {
        let mut d = SseDecoder::new();
        let chunk = vec![b'x'; 64 * 1024];
        let mut err = None;
        // 一片 64 KiB，喂到超过 256 KiB 为止。真实的分片就是这个量级。
        for _ in 0..8 {
            if let Err(e) = d.push(&chunk) {
                err = Some(e);
                break;
            }
        }
        let e = err.expect("一条不换行的流必须在某一片上被截住");
        assert!(matches!(e, ModelError::Protocol(_)), "{e:?}");
        assert!(e.to_string().contains("sse line too large"), "{e}");
    }

    /// 没完没了的 `data:` 续行：每一行都合法，攒起来的那条事件不合法。
    #[test]
    fn endless_data_continuations_are_a_protocol_error() {
        let mut d = SseDecoder::new();
        let line = format!("data: {}\n", "y".repeat(1024));
        let mut err = None;
        for _ in 0..1024 {
            if let Err(e) = d.push(line.as_bytes()) {
                err = Some(e);
                break;
            }
        }
        let e = err.expect("续行攒过 MAX_SSE_EVENT 必须被截住");
        assert!(e.to_string().contains("sse event too large"), "{e}");
    }

    /// 正常大小的一条大事件照常通过：上限不是在挡厂商真的会发的东西。
    #[test]
    fn a_large_but_bounded_event_still_gets_through() {
        let mut d = SseDecoder::new();
        let body = format!("data: {}\n\n", "z".repeat(MAX_SSE_EVENT / 2));
        let out = d.push(body.as_bytes()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data.len(), MAX_SSE_EVENT / 2);
    }
}
