use std::collections::{HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::EventIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    cells::{ActivityIndex, Cell, CellGrid},
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// A coloured directly-follows arc: an object type and the two activities it orders.
pub type Arc = (ObjectTypeIndex, ActivityIndex, ActivityIndex);

/// The activity index of every event, and the sorted activity names it indexes into.
///
/// Everything that reads an event's type must go through this, or activity identities
/// silently permute between runs.
#[derive(Debug, Clone)]
pub struct ActivityIndexing {
    /// Sorted activity names.
    pub activities: Vec<String>,
    /// Activity index per event, in importer event order.
    pub act_of: Vec<ActivityIndex>,
}

impl ActivityIndexing {
    /// Build the indexing, using the same sorted activity order as a [`CellGrid`].
    pub fn build(locel: &SlimLinkedOCEL, grid: &CellGrid) -> Self {
        let ix: HashMap<&str, ActivityIndex> = grid
            .activities
            .iter()
            .enumerate()
            .map(|(i, a)| (a.as_str(), i))
            .collect();
        let act_of = locel.get_ev_types().map(|a| ix[a]).collect();
        Self {
            activities: grid.activities.clone(),
            act_of,
        }
    }

    /// Number of activities.
    pub fn len(&self) -> usize {
        self.activities.len()
    }

    /// Whether the log has no activity at all.
    pub fn is_empty(&self) -> bool {
        self.activities.is_empty()
    }
}

/// Coloured arcs of the object-centric directly-follows graph a keep-set induces.
///
/// A type's flow is its trace restricted to the activities whose cell is kept (activity
/// projection), not the induced subgraph: removing a cell splices the trace, it does not
/// delete the object's later behaviour.
///
/// Events are ordered by (timestamp, activity, event id) and adjacent ones are chained.
/// Activity and event id break timestamp ties, so the graph is a function of the log and
/// not of the import order. A directly-follows arc is one linearisation of the log, not an
/// ordering fact; the strict test is [`precedes`](super::precedes).
pub fn arc_set(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    kept: &HashSet<Cell>,
) -> HashSet<Arc> {
    TraceVariants::build(locel, schema, acts).arc_set(kept)
}

/// One distinct activity sequence, and how many objects of the type walk it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceVariant {
    /// The object type whose objects walk this sequence.
    pub object_type: ObjectTypeIndex,
    /// The activities in order, unprojected.
    pub activities: Vec<ActivityIndex>,
    /// Objects of the type with exactly this sequence.
    pub objects: usize,
}

/// The distinct unprojected activity sequences per object type.
///
/// Projecting a keep-set onto these gives the same arcs as walking the objects, because an
/// arc is a set membership and duplicate sequences contribute nothing new. A client that
/// holds the variants can evaluate any keep-set itself.
///
/// The sort key is (timestamp, activity index, event id), with the activity index from
/// [`ActivityIndexing`], so the key is a function of the log and not of the import order.
pub fn trace_variants(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
) -> Vec<TraceVariant> {
    trace_variants_with(locel, schema, acts, &[])
}

/// The same, over the log plus a set of participations it does not record.
///
/// A flow layer may hold cells the extraction did not record, and scoring those against
/// the recorded variants alone reads their arcs as missing. `extra` is what expansion
/// wrote; the sort key is unchanged, so a written event takes its place in the object's
/// trace by its own timestamp.
pub fn trace_variants_with(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    extra: &[(EventIndex, crate::core::event_data::object_centric::linked_ocel::slim_linked_ocel::ObjectIndex)],
) -> Vec<TraceVariant> {
    let mut written: HashMap<
        crate::core::event_data::object_centric::linked_ocel::slim_linked_ocel::ObjectIndex,
        Vec<EventIndex>,
    > = HashMap::new();
    for (e, o) in extra {
        written.entry(*o).or_default().push(*e);
    }
    let mut counts: HashMap<(ObjectTypeIndex, Vec<ActivityIndex>), usize> = HashMap::new();
    for o in locel.get_all_obs() {
        let t = schema.type_of[&o];
        let mut evs: Vec<(i64, ActivityIndex, EventIndex)> = o
            .get_e2o_rev(locel)
            .copied()
            .chain(written.get(&o).into_iter().flatten().copied())
            .map(|e| {
                (
                    e.get_time(locel).timestamp_millis(),
                    acts.act_of[e.get_ev(locel).event_type],
                    e,
                )
            })
            .collect();
        evs.sort_by(|(ta, aa, a), (tb, ab, b)| {
            ta.cmp(tb)
                .then_with(|| aa.cmp(ab))
                .then_with(|| a.get_ev(locel).id.cmp(&b.get_ev(locel).id))
        });
        let trace: Vec<ActivityIndex> = evs.into_iter().map(|(_, a, _)| a).collect();
        *counts.entry((t, trace)).or_default() += 1;
    }
    let mut out: Vec<TraceVariant> = counts
        .into_iter()
        .map(|((object_type, activities), objects)| TraceVariant {
            object_type,
            activities,
            objects,
        })
        .collect();
    out.sort_by(|a, b| (a.object_type, &a.activities).cmp(&(b.object_type, &b.activities)));
    out
}

/// The trace variants of one log, held so a keep-set can be projected onto them.
///
/// Per-activity bounds cannot recover directly-follows arcs (an object visiting `a, b, a`
/// draws two arcs no pair of bounds recovers), so every object's events are sorted once,
/// the distinct sequences kept, and a keep-set evaluated by projecting them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceVariants {
    /// The distinct sequences, sorted by `(type, activities)`.
    pub variants: Vec<TraceVariant>,
    /// Activities the log has, which the union-find of [`Self::arcs_and_components`] spans.
    pub n_activities: usize,
}

impl TraceVariants {
    /// Read the variants off the log, once.
    pub fn build(
        locel: &SlimLinkedOCEL,
        schema: &StructuralSchema,
        acts: &ActivityIndexing,
    ) -> Self {
        Self {
            variants: trace_variants(locel, schema, acts),
            n_activities: acts.len(),
        }
    }

    /// The same, over the log plus what expansion wrote. See [`trace_variants_with`].
    pub fn build_with(
        locel: &SlimLinkedOCEL,
        schema: &StructuralSchema,
        acts: &ActivityIndexing,
        extra: &[(EventIndex, crate::core::event_data::object_centric::linked_ocel::slim_linked_ocel::ObjectIndex)],
    ) -> Self {
        Self {
            variants: trace_variants_with(locel, schema, acts, extra),
            n_activities: acts.len(),
        }
    }

    /// The coloured arcs a keep-set induces.
    ///
    /// Projection drops the activities whose cell is cut and the arc then runs from the
    /// surviving activity before to the surviving activity after.
    pub fn arc_set(&self, kept: &HashSet<Cell>) -> HashSet<Arc> {
        let mut arcs = HashSet::new();
        for v in &self.variants {
            let t = v.object_type;
            let mut prev: Option<ActivityIndex> = None;
            for a in v.activities.iter().filter(|a| kept.contains(&(**a, t))) {
                if let Some(p) = prev {
                    arcs.insert((t, p, *a));
                }
                prev = Some(*a);
            }
        }
        arcs
    }

    /// How many arcs one type's row contributes, given the activities kept for it.
    ///
    /// An [`Arc`] names its own type, so the total over a keep-set is the sum of this over
    /// the types, and a search can score a type's row independently of the others.
    pub fn arcs_of_row(&self, t: ObjectTypeIndex, row: &HashSet<ActivityIndex>) -> usize {
        let mut arcs: HashSet<(ActivityIndex, ActivityIndex)> = HashSet::new();
        for v in self.variants.iter().filter(|v| v.object_type == t) {
            let mut prev: Option<ActivityIndex> = None;
            for a in v.activities.iter().filter(|a| row.contains(*a)) {
                if let Some(p) = prev {
                    arcs.insert((p, *a));
                }
                prev = Some(*a);
            }
        }
        arcs.len()
    }

    /// Arcs and the number of connected components of the activity graph they induce.
    pub fn arcs_and_components(&self, kept: &HashSet<Cell>) -> (usize, usize) {
        let arcs = self.arc_set(kept);
        let mut parent: Vec<usize> = (0..self.n_activities).collect();
        for (_, a, b) in &arcs {
            let (ra, rb) = (find(&mut parent, *a), find(&mut parent, *b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let live: HashSet<ActivityIndex> = kept.iter().map(|(a, _)| *a).collect();
        let comps: HashSet<usize> = live.iter().map(|a| find(&mut parent, *a)).collect();
        (arcs.len(), comps.len())
    }

    /// Compare the arcs a keep-set induces against the full model's over the same cells.
    pub fn df_fidelity(&self, grid: &CellGrid, kept: &HashSet<Cell>) -> DfFidelity {
        let full = self.arc_set(&grid.cells);
        let reduced = self.arc_set(kept);
        let kept_types: HashSet<ObjectTypeIndex> = kept.iter().map(|(_, t)| *t).collect();
        let full_on_kept: HashSet<Arc> = full
            .iter()
            .filter(|(t, a, b)| {
                kept_types.contains(t) && kept.contains(&(*a, *t)) && kept.contains(&(*b, *t))
            })
            .copied()
            .collect();
        DfFidelity {
            reduced_arcs: reduced.len(),
            spurious: reduced.difference(&full_on_kept).count(),
            missing: full_on_kept.difference(&reduced).count(),
        }
    }
}

/// Arcs and the number of connected components of the activity graph they induce.
pub fn arcs_and_components(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    kept: &HashSet<Cell>,
) -> (usize, usize) {
    TraceVariants::build(locel, schema, acts).arcs_and_components(kept)
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut x = x;
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Components of the activity-type incidence graph: two activities are linked when they
/// keep a common object type.
///
/// This is the artifact-independent notion of connectivity. Repairing the directly-follows
/// graph does not imply the constraint graph is connected, while two activities sharing a
/// kept type generally carry a constraint between them, so repairing this graph repairs
/// both without a discovery run.
pub fn incidence_components(kept: &HashSet<Cell>) -> usize {
    let mut by_type: HashMap<ObjectTypeIndex, Vec<ActivityIndex>> = HashMap::new();
    for (a, t) in kept {
        by_type.entry(*t).or_default().push(*a);
    }
    let acts: Vec<ActivityIndex> = kept
        .iter()
        .map(|(a, _)| *a)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut par: HashMap<ActivityIndex, ActivityIndex> = acts.iter().map(|a| (*a, *a)).collect();
    for group in by_type.values() {
        for w in group.windows(2) {
            let (ra, rb) = (find_map(&mut par, w[0]), find_map(&mut par, w[1]));
            if ra != rb {
                par.insert(ra, rb);
            }
        }
    }
    acts.iter()
        .map(|a| find_map(&mut par, *a))
        .collect::<HashSet<_>>()
        .len()
}

fn find_map(par: &mut HashMap<ActivityIndex, ActivityIndex>, x: ActivityIndex) -> ActivityIndex {
    let mut x = x;
    while par[&x] != x {
        let g = par[&par[&x]];
        par.insert(x, g);
        x = g;
    }
    x
}

/// Add cut cells back, in a fixed order, until every activity is linked to every other
/// through a shared kept type. Returns the repaired keep-set and how many cells it added.
pub fn repair_incidence(kept: &HashSet<Cell>, cut: &[Cell]) -> (HashSet<Cell>, usize) {
    let mut kept = kept.clone();
    let mut added = 0;
    while incidence_components(&kept) > 1 {
        let mut best: Option<(usize, Cell)> = None;
        for c in cut {
            if kept.contains(c) {
                continue;
            }
            let mut trial = kept.clone();
            trial.insert(*c);
            let n = incidence_components(&trial);
            if best.is_none_or(|(bn, _)| n < bn) {
                best = Some((n, *c));
            }
        }
        match best {
            Some((n, c)) if n < incidence_components(&kept) => {
                kept.insert(c);
                added += 1;
            }
            _ => break,
        }
    }
    (kept, added)
}

/// Add cut cells back, in a fixed order, until the reduced model has no more components
/// than `target_comps`.
///
/// Deterministic, which keeps the uniqueness argument intact once connectivity is
/// required. Quadratic in the number of cut cells, so bound `cut` on a large log.
pub fn repair_connectivity(
    variants: &TraceVariants,
    kept: &HashSet<Cell>,
    cut: &[Cell],
    target_comps: usize,
) -> (HashSet<Cell>, usize) {
    let mut kept = kept.clone();
    let mut added = 0;
    loop {
        let (_, comps) = variants.arcs_and_components(&kept);
        if comps <= target_comps {
            break;
        }
        let mut best: Option<(usize, Cell)> = None;
        for c in cut {
            if kept.contains(c) {
                continue;
            }
            let mut trial = kept.clone();
            trial.insert(*c);
            let (_, tc) = variants.arcs_and_components(&trial);
            if tc < comps && best.is_none_or(|(bc, _)| tc < bc) {
                best = Some((tc, *c));
            }
        }
        match best {
            Some((_, c)) => {
                kept.insert(c);
                added += 1;
            }
            None => break,
        }
    }
    (kept, added)
}

/// Directly-follows fidelity of a keep-set (Prop. 4.4): arcs the reduced model asserts
/// that no full trace supports, and arcs of the full model between two kept cells that the
/// reduced model drops.
///
/// `missing` is zero by construction, since projection preserves adjacency, so a non-zero
/// value is a bug. `spurious` counts the arcs the splice introduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfFidelity {
    /// Arcs of the reduced model.
    pub reduced_arcs: usize,
    /// Arcs the reduced model asserts that no unprojected trace supports.
    pub spurious: usize,
    /// Arcs between kept cells that the reduction lost. Always zero.
    pub missing: usize,
}

/// Compare the arcs a keep-set induces against the full model's arcs over the same cells.
pub fn df_fidelity(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    grid: &CellGrid,
    kept: &HashSet<Cell>,
) -> DfFidelity {
    TraceVariants::build(locel, schema, acts).df_fidelity(grid, kept)
}
