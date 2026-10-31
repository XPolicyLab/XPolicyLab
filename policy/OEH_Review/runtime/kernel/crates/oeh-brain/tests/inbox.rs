//! §7.2 的三 lane Inbox：claim 规则是"全部 next-step + 至多一条 next-turn"。

use oeh_brain::inbox::{Inbox, InboxItem, Lane};

fn item(lane: Lane, seq: u64) -> InboxItem {
    InboxItem {
        lane,
        text: format!("{lane:?}-{seq}"),
        seq,
    }
}

#[test]
fn claim_takes_all_next_step_and_at_most_one_followup_in_seq_order() {
    let mut ib = Inbox::default();
    ib.push(item(Lane::Followup, 5));
    ib.push(item(Lane::Steer, 3));
    ib.push(item(Lane::Inject, 4));
    ib.push(item(Lane::Followup, 6));
    let claimed = ib.claim_for_step();
    let seqs: Vec<u64> = claimed.iter().map(|i| i.seq).collect();
    assert_eq!(seqs, vec![3, 4, 5]);
    assert!(ib.has_followup());
    assert_eq!(ib.len(), 1);
    assert_eq!(
        ib.claim_for_step()
            .iter()
            .map(|i| i.seq)
            .collect::<Vec<_>>(),
        vec![6]
    );
    assert!(ib.is_empty());
}
