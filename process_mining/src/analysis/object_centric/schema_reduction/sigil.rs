use std::collections::{HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::QualifierIdx, LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    arcs::ActivityIndexing,
    cells::{ActivityIndex, Cell},
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// The `nonflow` tag: a prefix on an event-to-object qualifier saying the tuple is recorded
/// and draws no arcs.
///
/// The encoding is decided and documented in [`tagged`](super::tagged). The tuple stays in
/// the log and only its qualifier changes, which is what makes the full projection give the
/// input back with no side artifact.
pub const NOT_FLOWING: char = '!';

/// Prefix saying an expansion wrote this tuple and the extraction did not record it.
///
/// Not a tag: it marks provenance, so that removing the marked tuples and dropping the tags
/// returns the input log. Reflow never writes object-to-object edges, so this only ever
/// appears on an event-to-object qualifier.
pub const WRITTEN: char = '+';

/// What a qualifier's prefixes say about the relationship carrying it.
///
/// Prefixes compose, in this order: `+!q` is an expanded participation tagged non-flow.
/// Qualifiers are free strings in OCEL 2.0, so a tagged log is a valid OCEL 2.0 log and a
/// tool that does not know the convention sees a non-flow tuple as an ordinary
/// participation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Marks {
    /// An expansion wrote this tuple; the extraction did not record it.
    pub written: bool,
    /// The tuple is tagged `nonflow`, i.e. the qualifier carries [`NOT_FLOWING`].
    pub not_flowing: bool,
}

impl Marks {
    /// Whether schema discovery may read a relationship carrying these marks.
    ///
    /// Asymmetric on purpose. A `+` never reaches an object-to-object edge, and a `+` on an
    /// event-to-object tuple was derived from the schema, so reading it back is circular.
    /// This keeps the schema invariant under any number of expansions.
    pub fn counts_for_discovery(&self, e2o: bool) -> bool {
        !(e2o && self.written)
    }
}

/// Split a qualifier into its prefixes and the name underneath.
pub fn decode(qualifier: &str) -> (Marks, &str) {
    let mut marks = Marks::default();
    let mut rest = qualifier;
    if let Some(r) = rest.strip_prefix(WRITTEN) {
        marks.written = true;
        rest = r;
    }
    if let Some(r) = rest.strip_prefix(NOT_FLOWING) {
        marks.not_flowing = true;
        rest = r;
    }
    (marks, rest)
}

/// Put the prefixes back on a name.
pub fn encode(marks: Marks, base: &str) -> String {
    let mut out = String::with_capacity(base.len() + 2);
    if marks.written {
        out.push(WRITTEN);
    }
    if marks.not_flowing {
        out.push(NOT_FLOWING);
    }
    out.push_str(base);
    out
}

/// Whether a relationship with this qualifier draws arcs.
pub fn flows(qualifier: &str) -> bool {
    !decode(qualifier).0.not_flowing
}

/// Whether ReFlow wrote a relationship with this qualifier.
pub fn written(qualifier: &str) -> bool {
    decode(qualifier).0.written
}

/// The same qualifier with [`NOT_FLOWING`] set or cleared, or [`None`] when it already is
/// what was asked for.
fn retagged(qualifier: &str, not_flowing: bool) -> Option<String> {
    let (mut marks, base) = decode(qualifier);
    if marks.not_flowing == not_flowing {
        return None;
    }
    marks.not_flowing = not_flowing;
    Some(encode(marks, base))
}

/// Native object type index to the sorted [`ObjectTypeIndex`] the schema uses.
fn object_type_indexing(locel: &SlimLinkedOCEL, schema: &StructuralSchema) -> Vec<ObjectTypeIndex> {
    let ix: HashMap<&str, ObjectTypeIndex> = schema
        .types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    locel.get_ob_types().map(|t| ix[t]).collect()
}

/// Tag the participations of a set of cells `nonflow`, or `flow`.
///
/// Only the qualifier changes. Returns the number of relationships rewritten. The pipeline
/// entry point is [`tag`](super::tagged::tag), which builds the whole tagged log.
pub fn set_flow(
    locel: &mut SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    cells: &HashSet<Cell>,
    flowing: bool,
) -> usize {
    if cells.is_empty() {
        return 0;
    }
    let ot = object_type_indexing(locel, schema);
    let act_of: Vec<ActivityIndex> = acts.act_of.clone();
    // The replacement is a function of the qualifier alone, so the table is resolved once
    // before the pass.
    let names: Vec<Option<String>> = locel
        .qualifiers()
        .iter()
        .map(|q| retagged(q, !flowing))
        .collect();
    let table: Vec<Option<QualifierIdx>> = names
        .into_iter()
        .map(|n| n.map(|s| locel.intern_qualifier(&s)))
        .collect();
    locel.retag_e2o_by(|et, obt, q| {
        if !cells.contains(&(act_of[et], ot[obt])) {
            return None;
        }
        table.get(q.into_inner() as usize).copied().flatten()
    })
}

/// The non-flow cells of a tagged log, read off its qualifiers alone.
///
/// A cell counts as non-flow when every participation it carries is tagged so. A
/// half-tagged cell reads as flowing and is reported by [`partially_marked_cells`].
pub fn non_flow_cells(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
) -> HashSet<Cell> {
    let (marked, unmarked) = mark_tally(locel, schema, acts);
    marked
        .keys()
        .filter(|c| !unmarked.contains_key(*c))
        .copied()
        .collect()
}

/// Cells some but not all of whose participations carry [`NOT_FLOWING`].
///
/// Empty on anything Reflow produced, since it tags per cell. A non-empty answer means the
/// log was tagged by something else, or by hand.
pub fn partially_marked_cells(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
) -> HashSet<Cell> {
    let (marked, unmarked) = mark_tally(locel, schema, acts);
    marked
        .keys()
        .filter(|c| unmarked.contains_key(*c))
        .copied()
        .collect()
}

/// Per cell, how many of its participations are marked and how many are not.
fn mark_tally(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
) -> (HashMap<Cell, usize>, HashMap<Cell, usize>) {
    let (mut marked, mut unmarked): (HashMap<Cell, usize>, HashMap<Cell, usize>) =
        (HashMap::new(), HashMap::new());
    for e in locel.get_all_evs() {
        let a = acts.act_of[e.get_ev(locel).event_type];
        for (q, o) in e.get_e2o_q(locel) {
            let cell = (a, schema.type_of[o]);
            let side = if flows(q) { &mut unmarked } else { &mut marked };
            *side.entry(cell).or_default() += 1;
        }
    }
    (marked, unmarked)
}

/// Participations an expansion wrote, per cell: the `+`-marked event-to-object tuples.
pub fn written_cells(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
) -> HashMap<Cell, usize> {
    let mut out: HashMap<Cell, usize> = HashMap::new();
    for e in locel.get_all_evs() {
        let a = acts.act_of[e.get_ev(locel).event_type];
        for (q, o) in e.get_e2o_q(locel) {
            if written(q) {
                *out.entry((a, schema.type_of[o])).or_default() += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::{
        canonical::Fingerprint, cells::CellGrid, closure::SchemaClosure, tagged::tests::toy,
    };

    /// Tag every `orders` cell non-flow, read the tags back, and undo them.
    #[test]
    fn tagging_a_cell_changes_only_its_qualifier_and_undoes_exactly() {
        let ocel = toy();
        let before = Fingerprint::build(&ocel);
        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let grid = CellGrid::build(&ocel, &schema, &closure);
        let acts = ActivityIndexing::build(&ocel, &grid);
        let orders = schema.types.iter().position(|t| t == "orders").unwrap();
        let cells: HashSet<Cell> = grid.cells.iter().filter(|(_, t)| *t == orders).copied().collect();
        assert_eq!(cells.len(), 2);

        let mut marked = ocel.clone();
        assert_eq!(set_flow(&mut marked, &schema, &acts, &cells, false), 4);
        assert_eq!(non_flow_cells(&marked, &schema, &acts), cells);
        assert!(partially_marked_cells(&marked, &schema, &acts).is_empty());

        let e2o = |l: &SlimLinkedOCEL| l.get_all_evs().map(|e| e.get_e2o(l).count()).sum::<usize>();
        assert_eq!(e2o(&marked), e2o(&ocel));
        assert!(marked
            .get_all_evs()
            .flat_map(|e| e.get_e2o_q(&marked).map(|(q, _)| q.to_string()).collect::<Vec<_>>())
            .any(|q| q == "!order"));

        // `!` is a rendering mark, not evidence: discovery reads marked tuples like any other.
        let after = StructuralSchema::discover(&marked);
        assert_eq!(after.pairs(), schema.pairs());

        set_flow(&mut marked, &schema, &acts, &cells, true);
        assert_eq!(Fingerprint::build(&marked), before);
    }

    /// A `+`-marked participation must not witness a map, or the schema grows on what an
    /// upward step asserted.
    #[test]
    fn discovery_ignores_written_participations() {
        let ocel = toy();
        let base = StructuralSchema::discover(&ocel);
        let mut widened = ocel.clone();
        // Every `place` event gets the other order too, marked as written. Unmarked this
        // destroys `items -> orders`; marked it must change nothing.
        let evs: Vec<_> = widened.get_evs_of_type("place").copied().collect();
        let obs: Vec<_> = widened.get_all_obs().collect();
        let orders: Vec<_> = obs
            .iter()
            .filter(|o| o.get_ob_type(&widened) == "orders")
            .copied()
            .collect();
        for e in &evs {
            for o in &orders {
                widened.add_e2o(*e, *o, "+order".to_string());
            }
        }
        let derived = |s: &StructuralSchema| -> HashSet<(usize, usize)> {
            s.derived.iter().map(|m| (m.source, m.target)).collect()
        };
        assert_eq!(derived(&StructuralSchema::discover(&widened)), derived(&base));

        let mut naive = ocel;
        for e in &evs {
            for o in &orders {
                naive.add_e2o(*e, *o, "order".to_string());
            }
        }
        assert_ne!(derived(&StructuralSchema::discover(&naive)), derived(&base));
    }

    #[test]
    fn prefixes_compose_and_round_trip() {
        for q in ["", "item", "!item", "+item", "+!item"] {
            let (m, base) = decode(q);
            assert_eq!(encode(m, base), q, "round trip of {q:?}");
        }
        assert_eq!(decode("+!q").0, Marks { written: true, not_flowing: true });
        assert!(!flows("!q") && flows("+q") && written("+q") && !written("!q"));
    }

    #[test]
    fn discovery_reads_all_o2o_and_only_unwritten_e2o() {
        let written = Marks { written: true, not_flowing: false };
        assert!(!written.counts_for_discovery(true));
        assert!(written.counts_for_discovery(false));
        let marked = Marks { written: false, not_flowing: true };
        assert!(marked.counts_for_discovery(true) && marked.counts_for_discovery(false));
    }
}
