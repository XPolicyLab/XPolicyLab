//! 老师调用记录要的 Python 字节：`oeh_policy.contracts.encoded/digest`、`json.dumps` 的默认格式、
//! `teacher.redact`，以及 `base64`。
//!
//! 调用记录（`teacher/call-NNNN.json`）和 `model_call` 事件里的 `full_input_sha256` / `input_sha256`
//! 是**字节**的哈希：评测器拿它们做"输入已哈希"的检查，复审时按它们对照记录。所以这里不能
//! "语义相同"，必须和 Python 写出来的每个字节一样：
//!
//! - `encoded(v)` = `json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"`；
//! - `dumps_default(pairs)` = `json.dumps(dict(pairs))`（默认分隔符 `", "` / `": "`，保留插入序）——
//!   老师重问时那段错误 JSON 是**发给模型的文本**，线上一致要求它逐字一样；
//! - 两者都是 `ensure_ascii=True`：非 ASCII 写成 `\uXXXX`（小写十六进制，astral 平面拆代理对）；
//! - 浮点按 Python `float.__repr__`：最短可回读的数字，`1e-4 <= |x| < 1e16` 用定点（至少一位
//!   小数），否则科学计数（`1e-05`、`1.5e+16`）。

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

/// `contracts.encoded`：排序键、紧凑分隔符、ASCII 转义，末尾一个换行。
pub fn encoded(value: &Value) -> Result<Vec<u8>, String> {
    let mut out = String::new();
    write_value(&mut out, value, Style::Compact)?;
    out.push('\n');
    Ok(out.into_bytes())
}

/// `contracts.digest`：`encoded` 的 sha256（小写十六进制）。
pub fn digest(value: &Value) -> Result<String, String> {
    Ok(hex(&Sha256::digest(encoded(value)?)))
}

/// `json.dumps(dict(pairs))`：默认分隔符、按给定顺序（插入序）。嵌套的对象按 serde 的顺序
/// （排序）写出——调用方只在值是标量时用它。
pub fn dumps_default(pairs: &[(&str, Value)]) -> Result<String, String> {
    let mut out = String::from("{");
    for (i, (key, value)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_str(&mut out, key);
        out.push_str(": ");
        write_value(&mut out, value, Style::Default)?;
    }
    out.push('}');
    Ok(out)
}

/// Python `repr(float)`（也是 `json.dumps` 写浮点的方式）。非有限值返回 `None`
/// （`allow_nan=False` 会拒绝它们）。
pub fn py_float_repr(x: f64) -> Option<String> {
    if !x.is_finite() {
        return None;
    }
    if x == 0.0 {
        return Some(if x.is_sign_negative() { "-0.0" } else { "0.0" }.into());
    }
    // `{:e}` 给最短可回读的十进制数字：`1.2345e-5`、`1e16`、`-3.5e0`。
    let sci = format!("{x:e}");
    let (sign, rest) = match sci.strip_prefix('-') {
        Some(r) => ("-", r),
        None => ("", sci.as_str()),
    };
    let (mantissa, exp) = rest.split_once('e')?;
    let exp: i32 = exp.parse().ok()?;
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let text = if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let head = &digits[..1];
        let tail = &digits[1..];
        let mant = if tail.is_empty() {
            head.to_owned()
        } else {
            format!("{head}.{tail}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{esign}{:02}", exp.abs())
    };
    Some(format!("{sign}{text}"))
}

/// Python 打印一个 JSON 数：整数原样，浮点走 [`py_float_repr`]。
pub fn py_number(n: &Number) -> Result<String, String> {
    if n.is_i64() || n.is_u64() {
        return Ok(n.to_string());
    }
    let f = n.as_f64().ok_or("number is not representable")?;
    py_float_repr(f).ok_or_else(|| "Out of range float values are not JSON compliant".into())
}

#[derive(Clone, Copy, PartialEq)]
enum Style {
    /// `sort_keys=True, separators=(",", ":")`
    Compact,
    /// 默认分隔符（`", "` / `": "`）；对象按 serde 的顺序。
    Default,
}

fn write_value(out: &mut String, value: &Value, style: Style) -> Result<(), String> {
    let (item_sep, key_sep) = match style {
        Style::Compact => (",", ":"),
        Style::Default => (", ", ": "),
    };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&py_number(n)?),
        Value::String(s) => write_str(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_value(out, item, style)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in sorted(map).into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_str(out, key);
                out.push_str(key_sep);
                write_value(out, item, style)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Python 的 `sort_keys` 比的是 str（按码点），和 Rust `str` 的字节序在 UTF-8 下一致。
fn sorted(map: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut items: Vec<_> = map.iter().collect();
    items.sort_by(|a, b| a.0.cmp(b.0));
    items
}

/// `json.dumps` 的字符串转义，`ensure_ascii=True`。
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            // Python 的 ESCAPE_ASCII 是 `[^\ -~]`：空格到 `~` 以外的一律 `\uXXXX`（含 0x7f）。
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `base64.b64encode`：标准字母表，带 `=` 补齐。
pub fn b64encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `base64.b64decode(s)`（`validate=False`）：字母表以外的字符丢掉，`=` 处停。
pub fn b64decode(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        let Some(v) = B64.iter().position(|&x| x == c) else {
            continue;
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    out
}

const PNG_URL: &str = "data:image/png;base64,";

/// `teacher.redact`：记录保留文本与结构；图片字节换成它的 sha256 与长度，加密推理和签名
/// 换成它们的哈希。
pub fn redact(value: &Value) -> Result<Value, String> {
    Ok(match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, item) in map {
                let redacted = match (key.as_str(), item) {
                    ("image_url" | "url", Value::String(s)) if s.starts_with(PNG_URL) => {
                        png(&b64decode(&s[PNG_URL.len()..]))
                    }
                    ("data", Value::String(s))
                        if map.get("type").and_then(Value::as_str) == Some("base64")
                            && map
                                .get("media_type")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .starts_with("image/") =>
                    {
                        png(&b64decode(s))
                    }
                    ("image_url" | "encrypted_content" | "signature", Value::String(s)) => {
                        let mut wrapped = Map::new();
                        wrapped.insert("value".into(), Value::String(s.clone()));
                        serde_json::json!({
                            "sha256": digest(&Value::Object(wrapped))?,
                            "chars": s.chars().count(),
                        })
                    }
                    _ => redact(item)?,
                };
                out.insert(key.clone(), redacted);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(redact).collect::<Result<_, _>>()?),
        other => other.clone(),
    })
}

fn png(data: &[u8]) -> Value {
    serde_json::json!({"png_sha256": hex(&Sha256::digest(data)), "bytes": data.len()})
}

/// `str(x)[:n]`：按字符截。
pub fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_print_like_python_repr() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1.5, "1.5"),
            (123.456, "123.456"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.25e-7, "1.25e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (-2.5, "-2.5"),
            (1234567.0, "1234567.0"),
            (0.30000000000000004, "0.30000000000000004"),
            (1e300, "1e+300"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(x).unwrap(), want, "{x}");
        }
        assert!(py_float_repr(f64::NAN).is_none());
    }

    #[test]
    fn encoded_is_sorted_compact_and_ascii() {
        let v = json!({"b": [1, 2.0, "é😀\n\u{1}\u{7f}"], "a": {"z": null, "y": true}});
        assert_eq!(
            String::from_utf8(encoded(&v).unwrap()).unwrap(),
            "{\"a\":{\"y\":true,\"z\":null},\"b\":[1,2.0,\"\\u00e9\\ud83d\\ude00\\n\\u0001\\u007f\"]}\n"
        );
        // python3 -c 'import json,hashlib;print(hashlib.sha256((json.dumps({"a":[1,"x"]},sort_keys=True,
        //   separators=(",",":"))+"\n").encode()).hexdigest())'
        assert_eq!(
            digest(&json!({"a": [1, "x"]})).unwrap(),
            hex(&Sha256::digest(b"{\"a\":[1,\"x\"]}\n"))
        );
    }

    #[test]
    fn dumps_default_keeps_the_given_order() {
        let text = dumps_default(&[
            ("error", json!("bad \"x\"")),
            ("no_execution", json!(true)),
            ("request_id", json!("r1")),
        ])
        .unwrap();
        assert_eq!(
            text,
            "{\"error\": \"bad \\\"x\\\"\", \"no_execution\": true, \"request_id\": \"r1\"}"
        );
    }

    #[test]
    fn base64_round_trips() {
        for data in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            assert_eq!(b64decode(&b64encode(data)), data);
        }
        assert_eq!(b64encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(b64encode(b"fo"), "Zm8=");
    }

    #[test]
    fn redact_hashes_images_and_reasoning() {
        let url = format!("{PNG_URL}{}", b64encode(b"png!"));
        let v = redact(
            &json!([{"type": "image", "url": url}, {"encrypted_content": "abc"},
                               {"type": "text", "text": "keep"}]),
        )
        .unwrap();
        assert_eq!(v[0]["url"]["bytes"], 4);
        assert_eq!(v[0]["url"]["png_sha256"], hex(&Sha256::digest(b"png!")));
        assert_eq!(v[1]["encrypted_content"]["chars"], 3);
        assert_eq!(v[2]["text"], "keep");
    }
}
