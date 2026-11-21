use crate::brains::directed_action;
use crate::wake::{WakeAction, WakePolicy};
use oeh_proto::{FrameRef, WorldEvent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationBudget {
    #[serde(default = "d_events")]
    pub max_events: usize,
    #[serde(default = "d_bytes")]
    pub max_bytes: usize,
}
fn d_events() -> usize {
    32
}
fn d_bytes() -> usize {
    8192
}
impl Default for ObservationBudget {
    fn default() -> Self {
        Self {
            max_events: d_events(),
            max_bytes: d_bytes(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WakeReason {
    pub kind: String,
    pub seq: u64,
    pub action: WakeAction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollapsedEvent {
    pub kind: String,
    pub collapse_key: String,
    pub first_seq: u64,
    pub last_seq: u64,
    pub count: u64,
    pub salience: u8,
    pub first: Value,
    pub last: Value,
    /// 首末两条事件各自的帧引用：折叠掉中间那些之后，只有这两端还能和帧存储 join 上。
    /// 只留 last 会让“这段是从哪一帧开始的”无从回答。
    #[serde(default)]
    pub first_frame_refs: Vec<FrameRef>,
    #[serde(default)]
    pub last_frame_refs: Vec<FrameRef>,
}

/// 脑在一个 turn 开始时看到的全部内容。`source_seq_range` 让它能从世界日志重建。
/// `state` 在字节预算下可能被整体省略为 `{"_elided": true, "keys": <n>}`。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub t_kernel: u64,
    pub wake_reason: WakeReason,
    pub state: BTreeMap<String, Value>,
    pub events: Vec<CollapsedEvent>,
    pub source_seq_range: (u64, u64),
    pub truncated: usize,
}

/// 这条事件是不是**点名给 `brain_id`** 的定向事件。
///
/// `Some(Ignore)` 是"定向给别的脑"——对这个脑来说它只是一条普通的 `brain/*` 事件，照常
/// 走 policy。
fn directed_at(ev: &WorldEvent, brain_id: &str) -> bool {
    matches!(
        directed_action(ev, brain_id),
        Some(WakeAction::Steer) | Some(WakeAction::Interrupt)
    )
}

/// 把一段世界事件折叠成 `brain_id` 这个脑看到的那份观测。
///
/// `brain_id` 只用来认出**点名给这个脑的定向事件**（`Brains::steer`/`interrupt`/`inject`
/// 写的那三种，判据和 `route_one` 用的是同一个 [`crate::brains::directed_action`]）：它们
/// 不受下面那道 policy 过滤的约束。
///
/// 为什么必须破例：定向事件绕过 policy 到达这个脑（那正是"定向"的意思），但观测这一遍
/// 还在按 policy 滤——于是一份带着 `brain/*: ignore` 的策略（三个 shipped 策略都带）会把
/// 那条事件整条丢掉，脑收到的是一个 `wake_reason` 说着 `brain/inject`、正文里却什么都
/// 没有的观测。人说的那句话就这么没了，而且没有任何报错。
///
/// 破例只给**点名给这个脑**的那一条。`directed_action` 要求 `source == "brains"`，所以
/// 一条伪造的 `brain/inject`（`event/emit` 的 source 永远是 `rpc:<name>`）仍然老老实实
/// 落回 policy——`brain/*: ignore` 对它还是 ignore。
pub fn build_observation(
    brain_id: &str,
    events: &[WorldEvent],
    state: BTreeMap<String, Value>,
    wake_reason: WakeReason,
    t_kernel: u64,
    policy: &WakePolicy,
    budget: &ObservationBudget,
) -> Observation {
    let range = match (events.first(), events.last()) {
        (Some(a), Some(b)) => (a.seq, b.seq),
        _ => (0, 0),
    };
    // 折叠的同时**在插入点**记下哪一条覆盖了唤醒证据（`bool` 随它一起排序、一起搬动）。
    // 事后靠 kind + seq 区间反推会认错：同 kind 的另一条折叠项也可能把那个 seq 罩进区间。
    let mut collapsed: Vec<(CollapsedEvent, bool)> = Vec::new();
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for ev in events {
        if policy.action_for(&ev.kind) == WakeAction::Ignore && !directed_at(ev, brain_id) {
            continue;
        }
        let key = ev
            .collapse_key
            .clone()
            .unwrap_or_else(|| ev.kind.as_str().to_string());
        let is_wake = ev.seq == wake_reason.seq;
        match index.get(&key) {
            Some(&i) => {
                let (c, w) = &mut collapsed[i];
                c.last_seq = ev.seq;
                c.count += 1;
                c.last = ev.payload.clone();
                c.salience = c.salience.max(ev.salience);
                c.last_frame_refs = ev.frame_refs.clone();
                *w |= is_wake;
            }
            None => {
                index.insert(key.clone(), collapsed.len());
                collapsed.push((
                    CollapsedEvent {
                        kind: ev.kind.as_str().to_string(),
                        collapse_key: key,
                        first_seq: ev.seq,
                        last_seq: ev.seq,
                        count: 1,
                        salience: ev.salience,
                        first: ev.payload.clone(),
                        last: ev.payload.clone(),
                        first_frame_refs: ev.frame_refs.clone(),
                        last_frame_refs: ev.frame_refs.clone(),
                    },
                    is_wake,
                ));
            }
        }
    }
    collapsed.sort_by(|(a, _), (b, _)| {
        b.salience
            .cmp(&a.salience)
            .then(a.last_seq.cmp(&b.last_seq))
    });

    let total = collapsed.len();
    let mut wake_pos = collapsed.iter().position(|(_, w)| *w);

    // Pass 1: truncate to max_events, but protect the wake-reason collapsed event from being
    // dropped even if its sorted position falls beyond the cutoff. Relative order is preserved.
    if collapsed.len() > budget.max_events {
        match wake_pos {
            Some(pos) if pos >= budget.max_events => {
                let wake_event = collapsed.remove(pos);
                collapsed.truncate(budget.max_events.saturating_sub(1));
                collapsed.push(wake_event);
                wake_pos = Some(collapsed.len() - 1);
            }
            _ => collapsed.truncate(budget.max_events),
        }
    }

    let mut obs = Observation {
        t_kernel,
        wake_reason,
        state,
        events: collapsed.into_iter().map(|(c, _)| c).collect(),
        source_seq_range: range,
        truncated: 0,
    };

    // Pass 2: drop from the tail (skipping the wake-reason event) until within the byte budget,
    // or until only the wake-reason event (or a single event) remains. `wake_pos` is maintained
    // by hand: every removal below it shifts it down by one.
    loop {
        let too_big = serde_json::to_vec(&obs)
            .map(|v| v.len())
            .unwrap_or(usize::MAX)
            > budget.max_bytes;
        if !too_big || obs.events.len() <= 1 {
            break;
        }
        let remove_idx = (0..obs.events.len()).rev().find(|i| Some(*i) != wake_pos);
        match remove_idx {
            Some(idx) => {
                obs.events.remove(idx);
                if let Some(w) = wake_pos {
                    if idx < w {
                        wake_pos = Some(w - 1);
                    }
                }
            }
            None => break,
        }
    }

    // Pass 3: events are down to one (or the wake-reason event alone) and it is still over
    // budget — the state snapshot itself is what does not fit. Replace it wholesale with an
    // explicit marker: a brain that knows its state was elided can ask for it, one that silently
    // receives an oversized observation cannot.
    let over = serde_json::to_vec(&obs)
        .map(|v| v.len())
        .unwrap_or(usize::MAX)
        > budget.max_bytes;
    if over && !obs.state.is_empty() {
        let keys = obs.state.len();
        obs.state = BTreeMap::from([
            ("_elided".to_string(), Value::Bool(true)),
            ("keys".to_string(), Value::from(keys)),
        ]);
    }

    obs.truncated = total - obs.events.len();
    obs
}
