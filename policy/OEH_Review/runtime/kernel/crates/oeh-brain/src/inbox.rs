//! 三 lane 的收件箱（规格 §7.2）。
//!
//! 一个 turn 在跑的时候，外面还会有东西进来：人插一句话、别的脑发来消息、上一个 turn
//! 留下的待办。它们不能都立刻打断当前的 step，也不能全被吞掉——所以分三条 lane：
//!
//! | lane | 什么时候生效 | 会不会唤醒 |
//! |---|---|---|
//! | [`Lane::Steer`] | 下一个 step | 会（idle 时由内核的唤醒策略开 turn） |
//! | [`Lane::Inject`] | 下一个 step | 不会（只是捎带上的上下文） |
//! | [`Lane::Followup`] | 下一个 **turn** | —— |
//!
//! claim 规则只有一条：**全部 next-step + 至多一条 next-turn**。为什么 followup 一次
//! 只放一条：followup 是"这个 turn 结束之后还要接着做的事"，一次全倒进同一个 step 就
//! 等于把后面几件事的上下文提前塞进来，模型会在一个 step 里试着一起做完。一次一条，
//! 剩下的留给下一个 turn（[`Inbox::has_followup`] 告诉调用方还要不要再开一个 turn）。

/// 一条收件的 lane。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// next-turn：下一个 turn 才处理。
    Followup,
    /// next-step：下一个 step 处理，并且会唤醒。
    Steer,
    /// next-step：下一个 step 处理，但不唤醒（不为它单独开一个 turn）。
    Inject,
}

impl Lane {
    /// 这条 lane 是不是"下一个 step 就生效"。
    pub fn is_next_step(self) -> bool {
        matches!(self, Lane::Steer | Lane::Inject)
    }
}

/// 收件箱里的一条。`seq` 是投递序（由调用方给，单调递增）：claim 出来的一批按 `seq`
/// 排序，模型看到的顺序就是事情发生的顺序，和 lane 无关。
#[derive(Clone, Debug, PartialEq)]
pub struct InboxItem {
    pub lane: Lane,
    pub text: String,
    pub seq: u64,
}

/// 三 lane 收件箱。没有内部锁：它属于一个 turn 的循环，由持有它的那一方（Task 2 的
/// `NativeBrain`，用它自己的 `Mutex`）串行访问。
#[derive(Default, Debug)]
pub struct Inbox {
    /// 未认领的条目，按投递顺序。
    items: Vec<InboxItem>,
}

impl Inbox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, item: InboxItem) {
        self.items.push(item);
    }

    /// 认领下一个 step 要带上的条目：全部 next-step（[`Lane::Steer`] + [`Lane::Inject`]）
    /// 加**最早的一条** [`Lane::Followup`]。返回按 `seq` 升序；认领走的从收件箱里移除。
    ///
    /// 选 followup 用的是**下标**而不是 `seq` 值：`seq` 由调用方给，两条 followup 撞号
    /// 时按 `seq` 挑会把两条都认领走，"至多一条 next-turn"这条规则就破了。撞号时取先
    /// 投进来的那条。
    pub fn claim_for_step(&mut self) -> Vec<InboxItem> {
        let followup = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, i)| i.lane == Lane::Followup)
            .min_by_key(|(idx, i)| (i.seq, *idx))
            .map(|(idx, _)| idx);
        let mut claimed = Vec::new();
        let mut kept = Vec::with_capacity(self.items.len());
        for (idx, item) in self.items.drain(..).enumerate() {
            if item.lane.is_next_step() || Some(idx) == followup {
                claimed.push(item);
            } else {
                kept.push(item);
            }
        }
        self.items = kept;
        // 稳定排序：`seq` 撞号时保持投递顺序。
        claimed.sort_by_key(|i| i.seq);
        claimed
    }

    /// turn 收尾时问一句：还有没有 followup 在等——有就该再开一个 turn。
    pub fn has_followup(&self) -> bool {
        self.items.iter().any(|i| i.lane == Lane::Followup)
    }

    /// 还没被认领的条目数。
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(lane: Lane, seq: u64) -> InboxItem {
        InboxItem {
            lane,
            text: format!("{lane:?}-{seq}"),
            seq,
        }
    }

    #[test]
    fn claim_on_an_empty_inbox_is_empty_and_has_no_followup() {
        let mut ib = Inbox::new();
        assert!(ib.claim_for_step().is_empty());
        assert!(!ib.has_followup());
        assert!(ib.is_empty());
    }

    #[test]
    fn the_earliest_followup_goes_first_even_if_it_was_pushed_last() {
        let mut ib = Inbox::new();
        ib.push(item(Lane::Followup, 9));
        ib.push(item(Lane::Followup, 2));
        let claimed = ib.claim_for_step();
        assert_eq!(claimed.iter().map(|i| i.seq).collect::<Vec<_>>(), vec![2]);
        assert!(ib.has_followup());
    }

    #[test]
    fn followups_that_tie_on_seq_still_claim_exactly_one() {
        let mut ib = Inbox::new();
        ib.push(InboxItem {
            lane: Lane::Followup,
            text: "first".into(),
            seq: 4,
        });
        ib.push(InboxItem {
            lane: Lane::Followup,
            text: "second".into(),
            seq: 4,
        });
        let claimed = ib.claim_for_step();
        // "至多一条 next-turn" 不能靠调用方的 seq 自律。
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].text, "first");
        assert!(ib.has_followup());
        assert_eq!(ib.len(), 1);
    }

    #[test]
    fn inject_and_steer_are_both_next_step_and_never_left_behind() {
        let mut ib = Inbox::new();
        ib.push(item(Lane::Inject, 1));
        ib.push(item(Lane::Steer, 2));
        assert_eq!(ib.claim_for_step().len(), 2);
        assert!(ib.is_empty());
        assert!(!ib.has_followup());
    }
}
