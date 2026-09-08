//! Detection of sequence (`→`) cuts.

use crate::core::process_models::process_tree::OperatorType;

use super::super::dfg::ActivityDfg;
use super::super::log::ActivityLog;
use super::super::structures::complement_components::complement_components;
use super::super::structures::reachability::Reachability;
use super::cut::Cut;

/// Finds the maximal sequence cut, if there is one.
///
/// Requirement →.1 is that for `i < j`, every `a` in part `i` reaches every `b` in part `j` and no
/// `b` reaches back. Two activities may therefore only be separated if exactly one of them reaches
/// the other; mutually reachable ones (a loop) and mutually unreachable ones (a choice or
/// concurrency) have to be merged. Reachability then orders the parts.
///
/// Almost every pair has to be merged on an unstructured log, so the parts are computed on the
/// sparse complement of the merge relation.
pub fn sequence_cut(dfg: &ActivityDfg) -> Option<Cut> {
    let reaches = Reachability::of(dfg);
    let may_be_separated = |a: usize, b: usize| reaches.reaches(a, b) != reaches.reaches(b, a);

    let mut groups = complement_components(dfg.len(), may_be_separated);
    if groups.len() <= 1 {
        return None;
    }

    // Every activity of a part reaches the same activities outside it, so comparing any two
    // representatives orders the parts.
    groups.sort_by(|left, right| {
        let (a, b) = (left[0], right[0]);
        reaches.reaches(b, a).cmp(&reaches.reaches(a, b))
    });

    let cut = Cut::new(OperatorType::Sequence, super::to_partitions(dfg, &groups));
    cut.is_non_trivial().then_some(cut)
}

/// Merges the neighbouring parts of a sequence cut that the log never shows apart.
///
/// Requirement →.1 only looks at the graph, so the maximal cut can separate parts that always occur
/// together. For `⟨c,c,c,a,c⟩, ⟨b,b,d,a⟩, ⟨b,c⟩` it separates `{b}` from `{d}`, and the resulting
/// tree accepts a `d` without a preceding `b`, which no trace does. The thesis names the refinement
/// and declines it "for efficiency considerations" (§6.1, discussion of L77); `ProM` and `PM4Py`
/// apply it by default.
///
/// Merging changes which parts are neighbours, so this repeats until it settles. Returns `None` if
/// nothing is left to cut.
pub fn strict_sequence_cut(log: &ActivityLog, cut: Cut) -> Option<Cut> {
    let mut parts = cut.partitions().to_vec();

    while let Some(merge) = merges(log, &parts) {
        let mut merged: Vec<Vec<usize>> = Vec::with_capacity(parts.len());
        for (index, part) in parts.into_iter().enumerate() {
            match merged.last_mut() {
                Some(previous) if merge[index - 1] => previous.extend(part),
                _ => merged.push(part),
            }
        }
        for part in &mut merged {
            part.sort_unstable();
        }
        parts = merged;
    }

    let cut = Cut::new(OperatorType::Sequence, parts);
    cut.is_non_trivial().then_some(cut)
}

/// Which neighbouring parts have to be merged, or `None` if none have.
///
/// Two neighbours stay separated only if the log shows every combination of "this part occurs" and
/// "it does not" that the split allows. Neither of them occurring is not such a combination:
/// whether everything may be skipped is settled by the empty traces before cut detection runs.
fn merges(log: &ActivityLog, parts: &[Vec<usize>]) -> Option<Vec<bool>> {
    let mut part_of = vec![usize::MAX; log.alphabet_size()];
    for (index, part) in parts.iter().enumerate() {
        for &activity in part {
            part_of[activity] = index;
        }
    }

    let mut occurs = vec![[false; 2]; parts.len()];
    let mut neighbours = vec![[false; 4]; parts.len() - 1];
    let mut occurring = vec![false; parts.len()];

    for variant in log.variants() {
        occurring.fill(false);
        for &activity in &variant.activities {
            let part = part_of[activity];
            if part != usize::MAX {
                occurring[part] = true;
            }
        }
        for (part, &present) in occurring.iter().enumerate() {
            occurs[part][present as usize] = true;
        }
        for (index, pair) in occurring.windows(2).enumerate() {
            neighbours[index][pair[0] as usize * 2 + pair[1] as usize] = true;
        }
    }

    let merge: Vec<bool> = neighbours
        .iter()
        .enumerate()
        .map(|(index, shown)| {
            [(0, 1), (1, 0), (1, 1)].iter().any(|&(left, right)| {
                occurs[index][left] && occurs[index + 1][right] && !shown[left * 2 + right]
            })
        })
        .collect();
    merge.iter().any(|&m| m).then_some(merge)
}

#[cfg(test)]
mod test_sequence_cut {
    use super::super::test_utils::{cut_of, parts};
    use super::*;

    fn cut(traces: &[&[&str]]) -> Option<Vec<Vec<String>>> {
        cut_of(traces, sequence_cut)
    }

    #[test]
    fn test_simple_sequence() {
        assert_eq!(cut(&[&["a", "b", "c"]]), parts(&[&["a"], &["b"], &["c"]]));
        assert_eq!(cut(&[&["a"]]), None);
    }

    #[test]
    fn test_parts_that_have_to_be_merged() {
        // b and c never reach each other (a choice), and B and C reach each other (concurrency).
        assert_eq!(
            cut(&[&["a", "b", "d"], &["a", "c", "d"]]),
            parts(&[&["a"], &["b", "c"], &["d"]])
        );
        assert_eq!(
            cut(&[&["A", "B", "C", "D"], &["A", "C", "B", "D"]]),
            parts(&[&["A"], &["B", "C"], &["D"]])
        );
        assert_eq!(
            cut(&[&["a", "c", "d"], &["b", "c", "e"]]),
            parts(&[&["a", "b"], &["c"], &["d", "e"]])
        );
        assert_eq!(
            cut(&[&["A", "C"], &["B", "C", "D"], &["B", "D"]]),
            parts(&[&["A", "B"], &["C"], &["D"]])
        );
    }

    #[test]
    fn test_skipped_activity_still_sequences() {
        assert_eq!(
            cut(&[&["a", "b", "c"], &["a", "c"]]),
            parts(&[&["a"], &["b"], &["c"]])
        );
    }

    #[test]
    fn test_strict_merges_parts_that_never_occur_apart() {
        let strict = |traces: &[&[&str]]| -> Option<Vec<Vec<String>>> {
            let (labels, log) = super::super::super::log::test_utils::log_of(traces);
            let dfg = ActivityDfg::discover(&log);
            sequence_cut(&dfg)
                .and_then(|cut| strict_sequence_cut(&log, cut))
                .map(|cut| {
                    cut.partitions()
                        .iter()
                        .map(|part| part.iter().map(|&a| labels[a].clone()).collect())
                        .collect()
                })
        };

        // No trace has d without b, so separating the two would allow a d on its own.
        assert_eq!(
            strict(&[
                &["c", "c", "c", "a", "c"],
                &["b", "b", "d", "a"],
                &["b", "c"]
            ]),
            parts(&[&["b", "d"], &["a", "c"]])
        );
        // Both parts are seen with and without the other, so the cut stays as it is.
        assert_eq!(
            strict(&[&["a", "b"], &["b", "c"]]),
            parts(&[&["a"], &["b"], &["c"]])
        );
        assert_eq!(
            strict(&[&["a", "b", "c"], &["a", "c"]]),
            parts(&[&["a"], &["b"], &["c"]])
        );
        // Skipping both parts is what the empty traces decide, so it does not merge them.
        assert_eq!(
            strict(&[&["a", "b"], &["a"], &["b"]]),
            parts(&[&["a"], &["b"]])
        );
    }

    #[test]
    fn test_no_cut() {
        // Concurrency, a choice (handled by the exclusive choice cut first), and a loop.
        assert_eq!(cut(&[&["B", "C"], &["C", "B"]]), None);
        assert_eq!(cut(&[&["a", "b", "c"], &["d"]]), None);
        assert_eq!(
            cut(&[
                &["B", "C"],
                &["C", "B"],
                &["B", "C", "E", "F", "B", "C"],
                &["C", "B", "E", "F", "C", "B"],
            ]),
            None
        );
    }
}
