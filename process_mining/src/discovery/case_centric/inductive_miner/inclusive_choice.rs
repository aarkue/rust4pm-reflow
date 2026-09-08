//! Rewriting concurrency over skippable children as an inclusive choice.

use crate::core::process_models::process_tree::{LeafLabel, Node, OperatorType};

/// Rewrites `∧(T₁, …, Tₙ)` into `∨` over those children that accept the empty trace, leaving the
/// rest next to it as in `∧(c, ∨(×(τ, a), ×(τ, b)))`.
///
/// Language-preserving: once two children can both do nothing, the concurrency already allows
/// running only some of them, which is what an inclusive choice says. Cut detection cannot find `∨`
/// (Requirement Cb.5), so this runs on the finished tree.
pub fn to_inclusive_choice(node: Node) -> Node {
    let Node::Operator(mut operator) = node else {
        return node;
    };
    operator.children = operator
        .children
        .into_iter()
        .map(to_inclusive_choice)
        .collect();

    if operator.operator_type != OperatorType::Concurrency {
        return Node::Operator(operator);
    }
    let skippable = operator
        .children
        .iter()
        .filter(|child| accepts_empty_trace(child))
        .count();
    if skippable < 2 {
        return Node::Operator(operator);
    }

    let (skippable, rest): (Vec<Node>, Vec<Node>) =
        operator.children.into_iter().partition(accepts_empty_trace);

    let mut choice = Node::new_operator(OperatorType::InclusiveChoice);
    for child in skippable {
        choice.add_child(child);
    }
    if rest.is_empty() {
        return choice;
    }

    let mut concurrency = Node::new_operator(OperatorType::Concurrency);
    concurrency.add_child(choice);
    for child in rest {
        concurrency.add_child(child);
    }
    concurrency
}

/// Whether the tree accepts the empty trace.
fn accepts_empty_trace(node: &Node) -> bool {
    match node {
        Node::Leaf(leaf) => leaf.activity_label == LeafLabel::Tau,
        Node::Operator(operator) => {
            let children = || operator.children.iter();
            match operator.operator_type {
                // Every child has to do nothing.
                OperatorType::Sequence | OperatorType::Concurrency | OperatorType::Interleaving => {
                    children().all(accepts_empty_trace)
                }
                // One is enough.
                OperatorType::ExclusiveChoice | OperatorType::InclusiveChoice => {
                    children().any(accepts_empty_trace)
                }
                // The body runs at least once, the redo parts need not run at all.
                OperatorType::Loop => operator.children.first().is_some_and(accepts_empty_trace),
            }
        }
    }
}

#[cfg(test)]
mod test_inclusive_choice {
    use super::super::log::test_utils::log_of;
    use super::super::{discover, InductiveMinerOptions};
    use super::*;

    fn mine(traces: &[&[&str]]) -> String {
        let (labels, log) = log_of(traces);
        // On top of the thesis version, so that only the rewrite can introduce a `∨`.
        let options = InductiveMinerOptions {
            rewrite_inclusive_choice: true,
            ..InductiveMinerOptions::im_thesis()
        };
        discover(&labels, &log, options).to_string()
    }

    #[test]
    fn test_concurrency_over_skippable_children() {
        // Both branches may be skipped, so the concurrency is an inclusive choice. The skipping
        // stays where it was: the rewrite does not change the language, and this tree still
        // accepts the empty trace, as the concurrency it replaces did.
        assert_eq!(
            mine(&[&["a", "b"], &["b", "a"], &["a"], &["b"]]),
            "∨(X(tau, a), X(tau, b))"
        );
        // Only some children are skippable: the rest stays concurrent to the choice.
        assert_eq!(
            mine(&[&["a", "b", "c"], &["c", "b", "a"], &["a", "c"], &["b", "c"]]),
            "∧(c, ∨(X(tau, a), X(tau, b)))"
        );
    }

    #[test]
    fn test_untouched_where_nothing_may_be_skipped() {
        assert_eq!(mine(&[&["a", "b"], &["b", "a"]]), "∧(a, b)");
        assert_eq!(mine(&[&["a", "b", "c"]]), "→(a, b, c)");
    }

    #[test]
    fn test_accepts_empty_trace() {
        let empty = |traces: &[&[&str]]| {
            let (labels, log) = log_of(traces);
            let tree = discover(&labels, &log, InductiveMinerOptions::im_thesis());
            accepts_empty_trace(&tree.root)
        };
        assert!(empty(&[&[], &["a"]]));
        assert!(empty(&[&[], &["a", "b"]]));
        assert!(!empty(&[&["a"]]));
        assert!(!empty(&[&["a", "b"], &["b", "a"]]));
        // A loop repeats its body at least once.
        assert!(!empty(&[&["a", "a"]]));
    }
}
