//! Two events of one object at the same instant assert no ordering fact.
//!
//! An ordering fact is a claim about every object of a type, and a pair the clock puts at
//! one instant does not establish it in either direction. A non-strict bounds comparison
//! reads a tie as a precedence, so the rule lives here and every site that asks about facts
//! calls it.
//!
//! The directly-follows side does not use this. An arc is one linearisation of what the log
//! recorded, so [`trace_variants`](super::trace_variants) sorts by (timestamp, activity,
//! event id) and chains adjacent events: a tie is broken by the key.
//!
//! Milliseconds are the unit of simultaneity, here and in [`Bounds`](super::Bounds).

/// Whether an object witnessing `x` no earlier than `xmin` and `y` no later than `ymax`
/// witnesses `x` before `y`.
///
/// Strict: `xmin == ymax` forces every `x` and every `y` of this object into one instant,
/// which is a tie. Applied in both directions a tie lands in neither `ordering` nor `role`,
/// which is why [`Facts`](super::Facts) counts it as a third outcome instead of concurrency.
pub fn precedes(xmin: i64, ymax: i64) -> bool {
    xmin < ymax
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::analysis::object_centric::schema_reduction::{
        facts_from, trace_variants, ActivityIndexing, Bounds, CellGrid, SchemaClosure,
        StructuralSchema, TraceVariants,
    };
    use crate::core::chrono::DateTime;
    use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;

    fn at(ms: i64) -> crate::core::chrono::DateTime<crate::core::chrono::FixedOffset> {
        DateTime::from_timestamp_millis(ms).unwrap().fixed_offset()
    }

    /// One type, and per object one event per activity in `plan`, at the given instant.
    fn log(object_type: &str, n: usize, plan: &[(&str, i64)]) -> SlimLinkedOCEL {
        let mut ocel = SlimLinkedOCEL::new();
        ocel.add_object_type(object_type, Vec::new());
        let mut seen: Vec<&str> = plan.iter().map(|(a, _)| *a).collect();
        seen.sort_unstable();
        seen.dedup();
        for a in seen {
            ocel.add_event_type(a, Vec::new());
        }
        for k in 0..n {
            let o = ocel
                .add_object(object_type, Some(format!("o{k}")), Vec::new(), Vec::new())
                .unwrap();
            for (a, ms) in plan {
                ocel.add_event(
                    a,
                    at(1_600_000_000_000 + *ms),
                    Some(format!("{a}-{k}")),
                    Vec::new(),
                    vec![("part".into(), o)],
                );
            }
        }
        ocel
    }

    fn prepared(ocel: &SlimLinkedOCEL) -> (StructuralSchema, CellGrid, ActivityIndexing) {
        let schema = StructuralSchema::discover(ocel);
        let closure = SchemaClosure::build(ocel, &schema);
        let grid = CellGrid::build(ocel, &schema, &closure);
        let acts = ActivityIndexing::build(ocel, &grid);
        (schema, grid, acts)
    }

    /// Every object has `start` and `complete` on one instant: no ordering fact in either
    /// direction, not concurrency, but still the one directly-follows arc the sort key gives.
    #[test]
    fn a_pair_of_simultaneous_events_asserts_nothing_and_draws_one_arc() {
        let ocel = log("towers", 3, &[("start", 0), ("complete", 0)]);
        let (schema, grid, acts) = prepared(&ocel);
        let ix = |a: &str| grid.activities.iter().position(|x| x == a).unwrap();
        let (t, start, complete) = (0usize, ix("start"), ix("complete"));

        let facts = facts_from(&Bounds::build(&ocel, &schema, &acts), &grid.cells, 0.0);
        assert!(facts.ordering.is_empty(), "a tie is not an ordering");
        assert!(facts.asserted.is_empty());
        assert!(facts.role.is_empty(), "the two activities do share objects");
        // A tie must stay distinguishable from parallelism.
        assert_eq!(
            facts.tied,
            HashSet::from([(t, start.min(complete), start.max(complete))])
        );

        // Sorting by (timestamp, activity, event id) puts `complete` first by name.
        let arcs = TraceVariants::build(&ocel, &schema, &acts).arc_set(&grid.cells);
        assert_eq!(arcs, HashSet::from([(t, complete, start)]));
    }

    /// A tie in the middle of a trace is chained through in activity order, and still
    /// asserts nothing across itself.
    #[test]
    fn a_tie_inside_a_trace_is_chained_but_asserts_nothing() {
        let ocel = log("parts", 2, &[("a", 0), ("b", 1000), ("c", 1000), ("d", 2000)]);
        let (schema, grid, acts) = prepared(&ocel);
        let ix = |a: &str| grid.activities.iter().position(|x| x == a).unwrap();
        let (t, a, b, c, d) = (0usize, ix("a"), ix("b"), ix("c"), ix("d"));

        // The tied `b` and `c` are ordered by activity, so `b -> c` is drawn and the
        // cross-product `a -> c` and `b -> d` are not.
        let arcs = TraceVariants::build(&ocel, &schema, &acts).arc_set(&grid.cells);
        assert_eq!(arcs, HashSet::from([(t, a, b), (t, b, c), (t, c, d)]));

        let facts = facts_from(&Bounds::build(&ocel, &schema, &acts), &grid.cells, 0.0);
        assert!(!facts.asserted.contains(&(t, b, c)));
        assert!(!facts.asserted.contains(&(t, c, b)));
        assert!(facts.tied.contains(&(t, b.min(c), b.max(c))));
        assert!(facts.asserted.contains(&(t, a, d)));
    }

    /// The sort key itself: time first, then activity, and a trace stays flat.
    #[test]
    fn a_trace_is_one_chain_in_sort_order() {
        let ocel = log("parts", 1, &[("d", 2000), ("c", 1000), ("b", 1000), ("a", 0)]);
        let (schema, grid, acts) = prepared(&ocel);
        let ix = |a: &str| grid.activities.iter().position(|x| x == a).unwrap();
        let variants = trace_variants(&ocel, &schema, &acts);
        assert_eq!(variants.len(), 1);
        assert_eq!(
            variants[0].activities,
            vec![ix("a"), ix("b"), ix("c"), ix("d")]
        );

        assert!(!precedes(1000, 1000));
        assert!(precedes(0, 1000));
    }
}
