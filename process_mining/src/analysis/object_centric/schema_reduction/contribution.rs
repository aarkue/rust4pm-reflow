use std::collections::HashSet;

use super::{novelty::Pair, schema::ObjectTypeIndex};

/// What one object type adds to the model's behaviour that nothing else does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contribution {
    /// The type.
    pub object_type: ObjectTypeIndex,
    /// Activity pairs it orders.
    pub orders: usize,
    /// Of those, the ones no **other** surviving type orders. Zero exactly when the type
    /// leaves the flow layer.
    pub unique: usize,
    /// Whether it survives as a type that draws arcs.
    pub drawn: bool,
    /// A surviving type that orders one of its pairs, when this type does not survive.
    pub covered_by: Option<ObjectTypeIndex>,
}

/// Reason (b), threshold-free: a type stops drawing arcs when every activity pair it
/// orders is already ordered by some other type that is drawn.
///
/// The rule has no constant and no denominator, is computable on the recorded log without
/// saturation, and is the mirror image of expansion's novelty rule.
///
/// Types leave the flow layer one at a time, re-checked after each. Two types that order
/// exactly the same pairs are each redundant given the other, and taking both out at once
/// would lose the behaviour they share. Exactly one of a mutually redundant class survives, chosen
/// by `rep` (the lexicographically least name in the refinement class), not by iteration
/// order.
///
/// The order in which types leave: fewest ordered pairs first, then a type that is not its class
/// representative before one that is, then the lexicographically greater name, so the
/// least name is the one left standing. The result is a function of the log.
pub fn marginal_contribution(
    per_type: &[HashSet<Pair>],
    rep: &[ObjectTypeIndex],
    types: &[String],
) -> Vec<Contribution> {
    let n = per_type.len();
    let mut drawn: Vec<bool> = (0..n).map(|t| !per_type[t].is_empty()).collect();

    type Key<'a> = (usize, bool, std::cmp::Reverse<&'a String>, std::cmp::Reverse<usize>);
    loop {
        let mut best: Option<Key<'_>> = None;
        for t in 0..n {
            if !drawn[t] {
                continue;
            }
            let redundant = per_type[t]
                .iter()
                .all(|p| (0..n).any(|s| s != t && drawn[s] && per_type[s].contains(p)));
            if !redundant {
                continue;
            }
            let key = (
                per_type[t].len(),
                rep[t] == t,
                std::cmp::Reverse(&types[t]),
                std::cmp::Reverse(t),
            );
            if best.as_ref().is_none_or(|b| key < *b) {
                best = Some(key);
            }
        }
        match best {
            Some((_, _, _, std::cmp::Reverse(t))) => drawn[t] = false,
            None => break,
        }
    }

    (0..n)
        .map(|t| {
            let unique = per_type[t]
                .iter()
                .filter(|p| !(0..n).any(|s| s != t && drawn[s] && per_type[s].contains(*p)))
                .count();
            let covered_by = if drawn[t] {
                None
            } else {
                per_type[t].iter().find_map(|p| {
                    (0..n).find(|s| *s != t && drawn[*s] && per_type[*s].contains(p))
                })
            };
            Contribution {
                object_type: t,
                orders: per_type[t].len(),
                unique: if drawn[t] { unique } else { 0 },
                drawn: drawn[t],
                covered_by,
            }
        })
        .collect()
}

/// The types that survive as arc-drawing types.
pub fn drawing_types(rows: &[Contribution]) -> HashSet<ObjectTypeIndex> {
    rows.iter().filter(|c| c.drawn).map(|c| c.object_type).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("t{i}")).collect()
    }

    /// `Case_R` in miniature: one ordering, and another type already has it.
    #[test]
    fn a_type_whose_every_ordering_another_type_has_stops_drawing() {
        let per_type = vec![
            [(0, 1), (1, 2), (0, 2)].into_iter().collect::<HashSet<Pair>>(),
            [(0, 1)].into_iter().collect(),
        ];
        let rows = marginal_contribution(&per_type, &[0, 1], &names(2));
        assert!(rows[0].drawn, "the type carrying three orderings stays");
        assert!(!rows[1].drawn, "the one restating one of them goes");
        assert_eq!(rows[1].covered_by, Some(0));
        assert_eq!(rows[0].unique, 3);
    }

    /// Two types ordering exactly the same pairs are each redundant given the other.
    /// Taking both out of the flow layer loses the behaviour; exactly one survives, chosen by `rep`.
    #[test]
    fn mutual_redundancy_keeps_one_and_rep_says_which() {
        let same: HashSet<Pair> = [(0, 1), (1, 2)].into_iter().collect();
        let per_type = vec![same.clone(), same];
        let types = vec!["Workflow".to_string(), "Application".to_string()];
        let rows = marginal_contribution(&per_type, &[0, 1], &types);
        assert_eq!(rows.iter().filter(|c| c.drawn).count(), 1);
        assert!(rows[1].drawn, "`Application` sorts before `Workflow`");
    }

    /// A type that orders nothing was never drawing anything to begin with.
    #[test]
    fn a_type_that_orders_nothing_is_not_drawn_and_is_not_a_demotion() {
        let per_type = vec![[(0, 1)].into_iter().collect::<HashSet<Pair>>(), HashSet::new()];
        let rows = marginal_contribution(&per_type, &[0, 1], &names(2));
        assert!(rows[0].drawn && !rows[1].drawn);
        assert_eq!(rows[1].covered_by, None);
    }

    /// Nothing leaves the flow layer when every type carries a pair of its own.
    #[test]
    fn disjoint_behaviour_is_never_redundant() {
        let per_type = vec![
            [(0, 1)].into_iter().collect::<HashSet<Pair>>(),
            [(2, 3)].into_iter().collect(),
            [(4, 5)].into_iter().collect(),
        ];
        let rows = marginal_contribution(&per_type, &[0, 1, 2], &names(3));
        assert!(rows.iter().all(|c| c.drawn && c.unique == 1));
    }
}
