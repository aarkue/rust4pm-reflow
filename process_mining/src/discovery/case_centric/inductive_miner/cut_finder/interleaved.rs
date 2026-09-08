//! Detection of interleaved (`↔`) cuts.

use crate::core::process_models::process_tree::OperatorType;

use super::super::log::{ActivityID, ActivityLog};
use super::super::structures::union_find::UnionFind;
use super::cut::Cut;

/// Finds the maximal interleaved cut, if there is one.
///
/// `↔(a, b)` and `∧(a, b)` have the same directly-follows graph (Requirement Cb.4), so this reads
/// the traces: an interleaved part runs to completion before the next starts, and its activities
/// therefore occupy one uninterrupted block. Two activities share a part when their blocks overlap
/// in some trace, or when their order never varies, which is a sequence rather than an
/// interleaving. Merging can create new overlaps, so this repeats until it settles.
pub fn interleaved_cut(log: &ActivityLog) -> Option<Cut> {
    let alphabet_size = log.alphabet_size();
    let mut parts = UnionFind::new(alphabet_size);
    let mut last_position = vec![0usize; alphabet_size];

    loop {
        let mut merged = false;
        for variant in log.variants() {
            merged |= merge_blocks(&variant.activities, &mut parts, &mut last_position);
        }
        // Only worth looking at once the blocks no longer move.
        if !merged && !merge_fixed_order(log, &mut parts) {
            break;
        }
    }

    let cut = Cut::new(OperatorType::Interleaving, groups_of(log, &mut parts));
    cut.is_non_trivial().then_some(cut)
}

/// Merges the parts whose order never varies, parts that never occur together included, and
/// reports whether that changed anything.
///
/// Only pairs that share no part are merged in one round, since merging is itself what can make an
/// order vary: for `⟨a,c,d⟩, ⟨b,d⟩, ⟨d,b,a,b,a⟩` both `{a,b}, {c}` and `{c}, {d}` are ordered, but
/// merging all three gives up the cut `↔({a,b,c}, {d})` that merging only the first pair reaches.
fn merge_fixed_order(log: &ActivityLog, parts: &mut UnionFind) -> bool {
    let groups = groups_of(log, parts);
    if groups.len() < 2 {
        return false;
    }

    let mut part_of = vec![usize::MAX; log.alphabet_size()];
    for (index, group) in groups.iter().enumerate() {
        for &activity in group {
            part_of[activity] = index;
        }
    }

    let count = groups.len();
    let mut before = vec![false; count * count];
    let mut order: Vec<usize> = Vec::new();
    for variant in log.variants() {
        order.clear();
        for &activity in &variant.activities {
            let part = part_of[activity];
            if order.last() != Some(&part) {
                order.push(part);
            }
        }
        for (position, &left) in order.iter().enumerate() {
            for &right in &order[position + 1..] {
                before[left * count + right] = true;
            }
        }
    }

    let mut merged = false;
    let mut taken = vec![false; count];
    for left in 0..count {
        for right in left + 1..count {
            let fixed = !before[left * count + right] || !before[right * count + left];
            if fixed && !taken[left] && !taken[right] {
                merged |= parts.union(groups[left][0], groups[right][0]);
                taken[left] = true;
                taken[right] = true;
            }
        }
    }
    merged
}

/// The activities of the log grouped by part, each group in ascending order.
fn groups_of(log: &ActivityLog, parts: &mut UnionFind) -> Vec<Vec<ActivityID>> {
    let mut group_of = vec![usize::MAX; log.alphabet_size()];
    let mut groups: Vec<Vec<ActivityID>> = Vec::new();
    for activity in log.activities() {
        let root = parts.find(activity);
        if group_of[root] == usize::MAX {
            group_of[root] = groups.len();
            groups.push(Vec::new());
        }
        groups[group_of[root]].push(activity);
    }
    groups
}

/// Merges the parts of the activities sharing a block of one trace, and reports whether that
/// changed anything.
///
/// A block reaches at least as far as the last occurrence of every activity in it, so extending the
/// end while sweeping divides the trace into the smallest blocks that do not overlap.
fn merge_blocks(trace: &[ActivityID], parts: &mut UnionFind, last_position: &mut [usize]) -> bool {
    let roots: Vec<usize> = trace.iter().map(|&a| parts.find(a)).collect();
    for (position, &root) in roots.iter().enumerate() {
        last_position[root] = position;
    }

    let mut merged = false;
    let mut block_start = 0;
    let mut block_end = 0;
    for (position, &root) in roots.iter().enumerate() {
        block_end = block_end.max(last_position[root]);
        merged |= parts.union(roots[block_start], root);
        if position == block_end {
            block_start = position + 1;
        }
    }
    merged
}

#[cfg(test)]
mod test_interleaved_cut {
    use super::super::test_utils::parts;
    use super::*;

    fn cut(traces: &[&[&str]]) -> Option<Vec<Vec<String>>> {
        let (labels, log) = super::super::super::log::test_utils::log_of(traces);
        interleaved_cut(&log).map(|cut| {
            cut.partitions()
                .iter()
                .map(|part| part.iter().map(|&a| labels[a].clone()).collect())
                .collect()
        })
    }

    #[test]
    fn test_blocks_in_any_order() {
        // (a b) and c never overlap, but their order varies. No directly-follows cut applies.
        assert_eq!(
            cut(&[&["a", "b", "c"], &["c", "a", "b"]]),
            parts(&[&["a", "b"], &["c"]])
        );
    }

    #[test]
    fn test_overlapping_blocks_are_merged() {
        // b sits inside the block of a, so the two cannot be separated.
        assert_eq!(cut(&[&["a", "b", "a"]]), None);
        // Merging a with b makes the block of a, b overlap the one of c.
        assert_eq!(cut(&[&["a", "c", "b"], &["a", "b", "a"]]), None);
    }

    #[test]
    fn test_parts_in_a_fixed_order_are_merged() {
        // A sequence is not an interleaving: a is always before b, so both stay together and the
        // sequence cut handles them.
        assert_eq!(cut(&[&["a", "b"]]), None);
        // Activities that never occur together are not an interleaving either.
        assert_eq!(cut(&[&["a"], &["b"]]), None);
    }

    #[test]
    fn test_no_cut() {
        assert_eq!(cut(&[&["a"]]), None);
        assert_eq!(cut(&[]), None);
        assert_eq!(cut(&[&[]]), None);
    }
}
