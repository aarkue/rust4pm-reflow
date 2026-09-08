//! The tagged OCEL and its two projections.
//!
//! A tagged OCEL is the input log with every event-to-object tuple carrying a tag,
//! `flow` or `nonflow`. It has the same events, objects, attributes and object-to-object
//! relation as the input, and the same tuples; only the tags are added. The *flow
//! projection* keeps the `flow` tuples and is what discovery reads; the *full projection*
//! drops the tags and is what every other analysis reads.
//!
//! # The tag encoding
//!
//! **This is the one place the encoding is decided.** A tag is a prefix on the tuple's
//! event-to-object qualifier: [`sigil::NOT_FLOWING`] (`!`) means `nonflow`, no prefix
//! means `flow`. Qualifiers are free strings in OCEL 2.0, so a tagged log exports and
//! imports as an ordinary OCEL 2.0 log, and a tool that does not know the convention reads
//! a `nonflow` tuple as an ordinary participation -- it draws arcs that Reflow would have
//! left out, which loses nothing.
//!
//! Nothing else records the tagging: no side file, no derived object-to-object edge, no
//! distinction in the log between an implied and an involved cell. Which non-flow cells
//! are implied and which are involved is recomputed by [`annotation`] from the tagged log
//! alone.
//!
//! The second prefix, [`sigil::WRITTEN`] (`+`), is not a tag. It marks a tuple expansion
//! added that the extraction never recorded, so that removing the `+` tuples and dropping
//! the tags returns the input log. The two compose as `+!q`.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::{EventIndex, ObjectIndex, QualifierIdx},
    LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    annotate::{annotate, Determination},
    arcs::ActivityIndexing,
    assignment::{assign, Assignment, CellState},
    canonical::Fingerprint,
    cells::{Cell, CellGrid, DETERMINATION_THETA},
    closure::SchemaClosure,
    expansion::EXPANSION_QUALIFIER,
    schema::StructuralSchema,
    sigil,
};

/// Tag a log with a flow layer: the tuples of every cell in `flow` carry `flow` and those
/// of every other recorded cell carry `nonflow`.
///
/// `added` are the tuples an expansion writes, i.e. tuples of cells the extraction never
/// recorded. They carry [`EXPANSION_QUALIFIER`] under the [`sigil::WRITTEN`] prefix and
/// are tagged `flow`, an expanded cell being in the flow layer by definition. Everything
/// else is untouched: same events, same objects, same attributes, same object-to-object
/// relation.
pub fn tag(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    flow: &HashSet<Cell>,
    added: &[(EventIndex, ObjectIndex)],
) -> SlimLinkedOCEL {
    let mut out = locel.clone();
    let ot = object_type_indexing(locel, schema);
    let act_of = acts.act_of.clone();
    // The replacement depends on the qualifier alone, so the table is interned once rather
    // than once per tuple.
    let table: Vec<Option<QualifierIdx>> = locel
        .qualifiers()
        .iter()
        .map(|q| {
            let (mut marks, base) = sigil::decode(q);
            (!marks.not_flowing).then(|| {
                marks.not_flowing = true;
                sigil::encode(marks, base)
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|n| n.map(|s| out.intern_qualifier(&s)))
        .collect();
    out.retag_e2o_by(|et, obt, q| {
        if flow.contains(&(act_of[et], ot[obt])) {
            return None;
        }
        table.get(q.into_inner() as usize).copied().flatten()
    });
    if !added.is_empty() {
        let q = sigil::encode(
            sigil::Marks {
                written: true,
                not_flowing: false,
            },
            EXPANSION_QUALIFIER,
        );
        for (e, o) in added {
            out.add_e2o(*e, *o, q.clone());
        }
    }
    out
}

/// Whether any tuple of the log carries a tag other than `flow`.
///
/// Read off the interned qualifier table, so it costs one pass over the distinct
/// qualifiers rather than over the relationships.
pub fn is_tagged(locel: &SlimLinkedOCEL) -> bool {
    locel.qualifiers().iter().any(|q| !sigil::flows(q))
}

/// The flow projection: the plain OCEL holding the `flow` tuples of a tagged log.
///
/// This is what discovery reads. An untagged log is its own flow projection, and is
/// returned borrowed.
pub fn flow_projection(locel: &SlimLinkedOCEL) -> Cow<'_, SlimLinkedOCEL> {
    if !is_tagged(locel) {
        return Cow::Borrowed(locel);
    }
    let doomed: Vec<(EventIndex, ObjectIndex)> = locel
        .get_all_evs()
        .flat_map(|e| {
            e.get_e2o_q(locel)
                .filter(|(q, _)| !sigil::flows(q))
                .map(move |(_, o)| (e, *o))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut out = locel.clone();
    out.delete_e2o_bulk(&doomed);
    Cow::Owned(out)
}

/// The full projection: every tuple of a tagged log, with the tags dropped.
///
/// This is what filtering, counting and performance analysis read. An untagged log is its
/// own full projection, and is returned borrowed.
pub fn full_projection(locel: &SlimLinkedOCEL) -> Cow<'_, SlimLinkedOCEL> {
    if !is_tagged(locel) {
        return Cow::Borrowed(locel);
    }
    let mut out = locel.clone();
    let table: Vec<Option<QualifierIdx>> = locel
        .qualifiers()
        .iter()
        .map(|q| {
            let (mut marks, base) = sigil::decode(q);
            marks.not_flowing.then(|| {
                marks.not_flowing = false;
                sigil::encode(marks, base)
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|n| n.map(|s| out.intern_qualifier(&s)))
        .collect();
    out.retag_e2o_by(|_, _, q| table.get(q.into_inner() as usize).copied().flatten());
    Cow::Owned(out)
}

/// Whether `tagged` tags `input`, i.e. whether dropping the tags gives `input` back
/// exactly.
///
/// False for an expanded log, which holds tuples the input does not.
pub fn tags(tagged: &SlimLinkedOCEL, input: &SlimLinkedOCEL) -> bool {
    Fingerprint::build(&full_projection(tagged)) == Fingerprint::build(input)
}

/// What a tagged log says about one cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotatedCell {
    /// The activity.
    pub activity: String,
    /// The object type.
    pub object_type: String,
    /// Whether the cell flows, is implied, or is involved.
    pub state: CellState,
    /// The reconstruction that recovers an implied cell, and [`None`] for the other two
    /// states.
    pub determined_by: Option<Determination>,
}

/// The annotation of a tagged log: which non-flow cells are implied and by what, and which
/// are involved.
#[derive(Debug, Clone, Default)]
pub struct Annotation {
    /// One entry per recorded cell, sorted by activity and then object type.
    pub cells: Vec<AnnotatedCell>,
}

impl Annotation {
    /// Cells at each of the three states.
    pub fn tally(&self) -> (usize, usize, usize) {
        let mut out = (0, 0, 0);
        for c in &self.cells {
            match c.state {
                CellState::Flow => out.0 += 1,
                CellState::Involvement => out.1 += 1,
                CellState::Implied => out.2 += 1,
            }
        }
        out
    }

    /// Implied cells whose route does not return the recorded participations exactly.
    pub fn inexact(&self) -> usize {
        self.cells
            .iter()
            .filter_map(|c| c.determined_by.as_ref())
            .filter(|d| !d.residuals.is_empty())
            .count()
    }
}

/// Recompute the annotation from a tagged log alone.
///
/// The object schema is derived from the full projection, the flow layer is read off the
/// tags, and every non-flow recorded cell is implied when the flow layer determines it
/// through a route and involved otherwise.
pub fn annotation(tagged: &SlimLinkedOCEL, theta: f64) -> Annotation {
    let full = full_projection(tagged);
    let schema = StructuralSchema::discover(&full);
    let closure = SchemaClosure::build(&full, &schema);
    let grid = CellGrid::build_with_theta(&full, &schema, &closure, theta);
    let acts = ActivityIndexing::build(&full, &grid);
    let non_flow = sigil::non_flow_cells(tagged, &schema, &acts);
    let flow: HashSet<Cell> = grid
        .cells
        .iter()
        .filter(|c| !non_flow.contains(c))
        .copied()
        .collect();
    let a: Assignment = assign(&grid, &flow, &HashSet::new());
    let record = annotate(&full, &schema, &closure, &grid, &flow, &HashSet::new());

    let mut cells: Vec<AnnotatedCell> = a
        .flow
        .iter()
        .filter(|c| grid.cells.contains(*c))
        .map(|(x, t)| AnnotatedCell {
            activity: grid.activities[*x].clone(),
            object_type: schema.types[*t].clone(),
            state: CellState::Flow,
            determined_by: None,
        })
        .collect();
    for m in record.cells {
        cells.push(AnnotatedCell {
            activity: m.activity,
            object_type: m.object_type,
            state: m.state,
            determined_by: m.determined_by,
        });
    }
    cells.sort_by(|a, b| {
        (&a.activity, &a.object_type).cmp(&(&b.activity, &b.object_type))
    });
    Annotation { cells }
}

/// The annotation at the default determination threshold.
pub fn annotation_of(tagged: &SlimLinkedOCEL) -> Annotation {
    annotation(tagged, DETERMINATION_THETA)
}

/// Native object type index to the sorted [`ObjectTypeIndex`] the schema uses.
///
/// [`ObjectTypeIndex`]: super::schema::ObjectTypeIndex
fn object_type_indexing(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
) -> Vec<super::schema::ObjectTypeIndex> {
    let ix: std::collections::HashMap<&str, super::schema::ObjectTypeIndex> = schema
        .types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    locel.get_ob_types().map(|t| ix[t]).collect()
}

#[cfg(test)]
pub(in crate::analysis::object_centric::schema_reduction) mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::FoldDirection;
    use crate::core::chrono::DateTime;

    pub(in crate::analysis::object_centric::schema_reduction) fn toy() -> SlimLinkedOCEL {
        let mut ocel = SlimLinkedOCEL::new();
        ocel.add_object_type("items", Vec::new());
        ocel.add_object_type("orders", Vec::new());
        ocel.add_event_type("place", Vec::new());
        ocel.add_event_type("pay", Vec::new());

        let o1 = ocel.add_object("orders", Some("o1".into()), Vec::new(), Vec::new()).unwrap();
        let o2 = ocel.add_object("orders", Some("o2".into()), Vec::new(), Vec::new()).unwrap();
        let i1 = ocel
            .add_object("items", Some("i1".into()), Vec::new(), vec![("of".into(), o1)])
            .unwrap();
        let i2 = ocel
            .add_object("items", Some("i2".into()), Vec::new(), vec![("of".into(), o2)])
            .unwrap();

        let t = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        for (n, (item, order)) in [(i1, o1), (i2, o2)].into_iter().enumerate() {
            for (k, act) in ["place", "pay"].into_iter().enumerate() {
                ocel.add_event(
                    act,
                    t(1_600_000_000_000 + (n * 2 + k) as i64 * 1000),
                    Some(format!("{act}-{n}")),
                    Vec::new(),
                    vec![("item".into(), item), ("order".into(), order)],
                );
            }
        }
        ocel
    }


    fn prepared(locel: &SlimLinkedOCEL) -> (StructuralSchema, SchemaClosure, CellGrid, ActivityIndexing) {
        let schema = StructuralSchema::discover(locel);
        let closure = SchemaClosure::build(locel, &schema);
        let grid = CellGrid::build(locel, &schema, &closure);
        let acts = ActivityIndexing::build(locel, &grid);
        (schema, closure, grid, acts)
    }

    fn e2o(l: &SlimLinkedOCEL) -> usize {
        l.get_all_evs().map(|e| e.get_e2o_q(l).count()).sum()
    }

    #[test]
    fn tagging_keeps_every_tuple_and_the_full_projection_gives_the_input_back() {
        let ocel = toy();
        let (schema, closure, grid, acts) = prepared(&ocel);
        let keep = grid.canonical_keepset(&closure, FoldDirection::Finest);

        let tagged = tag(&ocel, &schema, &acts, &keep.kept, &[]);
        assert_eq!(e2o(&tagged), e2o(&ocel), "tagging removes no tuple");
        assert!(tags(&tagged, &ocel), "dropping the tags returns the input");

        let flow = flow_projection(&tagged);
        assert_eq!(e2o(&flow), 4, "only the item cells flow");
        assert_eq!(flow.get_all_evs().count(), ocel.get_all_evs().count());
        assert_eq!(flow.get_all_obs().count(), ocel.get_all_obs().count());
    }

    #[test]
    fn an_untagged_log_is_its_own_projection() {
        let ocel = toy();
        assert!(!is_tagged(&ocel));
        assert!(matches!(flow_projection(&ocel), Cow::Borrowed(_)));
        assert!(matches!(full_projection(&ocel), Cow::Borrowed(_)));
        assert!(tags(&ocel, &ocel));
    }

    #[test]
    fn the_annotation_reads_the_tagged_log_alone() {
        let ocel = toy();
        let (schema, closure, grid, acts) = prepared(&ocel);
        let keep = grid.canonical_keepset(&closure, FoldDirection::Finest);
        let tagged = tag(&ocel, &schema, &acts, &keep.kept, &[]);

        let ann = annotation_of(&tagged);
        assert_eq!(ann.cells.len(), 4);
        let (flow, involved, implied) = ann.tally();
        assert_eq!((flow, involved), (2, 0));
        assert_eq!(implied, 2, "orders are implied by items at both activities");
        for c in ann.cells.iter().filter(|c| c.state == CellState::Implied) {
            assert_eq!(c.object_type, "orders");
            assert_eq!(
                c.determined_by.as_ref().map(|d| d.source_type.as_str()),
                Some("items")
            );
        }
    }

    #[test]
    fn the_object_to_object_relation_is_untouched() {
        let ocel = toy();
        let (schema, closure, grid, acts) = prepared(&ocel);
        let keep = grid.canonical_keepset(&closure, FoldDirection::Finest);
        let tagged = tag(&ocel, &schema, &acts, &keep.kept, &[]);
        let o2o = |l: &SlimLinkedOCEL| l.get_all_obs().map(|o| o.get_o2o_q(l).count()).sum::<usize>();
        assert_eq!(o2o(&tagged), o2o(&ocel));
        assert_eq!(o2o(&flow_projection(&tagged)), o2o(&ocel));
    }
}
