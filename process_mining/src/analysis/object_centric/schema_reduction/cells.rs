use std::collections::{HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    closure::{ObjectFibre, SchemaClosure},
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// Index into the log's activities, sorted by name.
///
/// Sorted, like object types, because the connectivity repair walks cut cells in index order
/// and the result has to be the same between runs.
pub type ActivityIndex = usize;

/// One entry of the type-level event-to-object relation: an activity and an object type.
pub type Cell = (ActivityIndex, ObjectTypeIndex);

/// The share of an activity's events at which a route must reproduce the target objects for
/// the cell to count as determined.
///
/// The schema layer's only threshold. At `1` every participation of a determined cell is
/// recomputed by the kept cells, so emptying it is lossless by construction. The default is
/// below 1 so that a few contradicted objects cannot hide a relationship that holds of the
/// rest.
pub const DETERMINATION_THETA: f64 = 0.95;

/// Objects of each type carried by one event.
type PerType = HashMap<ObjectTypeIndex, HashSet<ObjectIndex>>;

/// Which end of the refinement order is decided first.
///
/// Deciding the finest first cuts the coarse types but loses them as witnesses for anything
/// they alone explain. Folding out the coarse type keeps the same information in a
/// differently scoped model. Cell count cannot see that difference, arc count can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FoldDirection {
    /// Keep the finest type, so every reconstruction is a function application and the
    /// annotation needs only badges. The rule the paper recommends.
    #[default]
    Finest,
    /// Keep the coarse type and derive the finer one by fibre expansion (group folding).
    Coarsest,
}

/// How a cut cell is put back.
///
/// Part of the result: a reader recovers an implied cell by reading the route in the model's
/// legend, and a tool recovers it by evaluating the route over the flow participations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconRoute {
    /// The union of the images under a subset of the recorded qualifiers, named here.
    QualifiedUnion {
        /// The qualifiers whose image never exceeds the target at any event.
        qualifiers: Vec<String>,
    },
    /// A function application: `obj^T(e) = f[obj^S(e)]`.
    Function {
        /// Which of the type pair's composed functions ([`SchemaClosure::maps`]) it is.
        ///
        /// A pair carries several different functions, one per qualifier and one per
        /// composition route, and only some reconstruct at a given activity. Inverting the
        /// step needs the one that was used.
        ///
        /// [`SchemaClosure::maps`]: super::SchemaClosure::maps
        witness: usize,
    },
    /// A fibre expansion: `obj^T(e) = f^-1[obj^S(e)]` for a map `f: T -> S`.
    Fibre {
        /// Which of `(T, S)`'s functions was inverted. See [`ReconRoute::Function`].
        witness: usize,
    },
}

/// The object types present at one activity, with the reconstruction relation between them.
#[derive(Debug, Clone)]
pub struct ActivityCells {
    /// Object types this activity's events name, sorted.
    pub present: Vec<ObjectTypeIndex>,
    /// Recorded participations per present type, in the order of `present`.
    pub counts: Vec<usize>,
    /// `recon[i][j]`: how the cell of `present[j]` is reconstructed, at every event of
    /// this activity, from the cell of `present[i]`; `None` when it is not.
    pub recon: Vec<Vec<Option<ReconRoute>>>,
}

impl ActivityCells {
    /// Position of a type in `present`.
    pub fn slot(&self, t: ObjectTypeIndex) -> Option<usize> {
        self.present.iter().position(|x| *x == t)
    }

    /// Which cells at this activity a set of kept cells determines, allowing chains.
    ///
    /// Reconstruction is transitive, so determinacy is a closure: keeping `packages` recovers
    /// `items` by fibre expansion, and `items` then recovers `products` by function
    /// application. The relation itself is [`reconstructs`], checked when the grid was built;
    /// this only walks it.
    pub fn determined_by(&self, kept: &[ObjectTypeIndex]) -> Vec<bool> {
        let k = self.present.len();
        let mut have = vec![false; k];
        for t in kept {
            if let Some(i) = self.slot(*t) {
                have[i] = true;
            }
        }
        loop {
            let mut grew = false;
            let sources: Vec<usize> = (0..k).filter(|i| have[*i]).collect();
            for i in sources {
                let row = &self.recon[i];
                for (j, reached) in have.iter_mut().enumerate() {
                    if !*reached && row[j].is_some() {
                        *reached = true;
                        grew = true;
                    }
                }
            }
            if !grew {
                return have;
            }
        }
    }

    /// The same walk as [`Self::determined_by`], keeping the hop each cell was reached by.
    ///
    /// `out[j] = Some((i, route))` means the cell of `present[j]` is put back from the cell
    /// of `present[i]` by `route`. A kept cell, and a cell nothing reaches, are both `None`;
    /// the two are told apart by whether the cell is in `kept`.
    ///
    /// Cells are reached in rounds and, inside a round, from the lowest-numbered source that
    /// reaches them, so the recorded route is a function of the grid alone.
    pub fn determining_routes(&self, kept: &[ObjectTypeIndex]) -> Vec<Option<(usize, ReconRoute)>> {
        let k = self.present.len();
        let mut have = vec![false; k];
        for t in kept {
            if let Some(i) = self.slot(*t) {
                have[i] = true;
            }
        }
        let mut via: Vec<Option<(usize, ReconRoute)>> = vec![None; k];
        loop {
            let mut grew = false;
            for j in 0..k {
                if have[j] {
                    continue;
                }
                let Some(i) = (0..k).find(|i| have[*i] && self.recon[*i][j].is_some()) else {
                    continue;
                };
                via[j] = Some((i, self.recon[i][j].clone().unwrap()));
                grew = true;
            }
            for (j, v) in via.iter().enumerate() {
                if v.is_some() {
                    have[j] = true;
                }
            }
            if !grew {
                return via;
            }
        }
    }

    /// Which cells reconstruct which, as a plain matrix.
    pub fn recon_matrix(&self) -> Vec<Vec<bool>> {
        self.recon
            .iter()
            .map(|row| row.iter().map(Option::is_some).collect())
            .collect()
    }
}

/// The cell grid of a log: every (activity, object type) the log records, with the
/// per-activity reconstruction relation the reduction decides from.
///
/// Feasibility factorises per activity: a cell may be cut only against cells kept at the
/// same activity, and reconstruction is checked over the events of that activity alone.
#[derive(Debug, Clone)]
pub struct CellGrid {
    /// Activity names, sorted. Every [`ActivityIndex`] refers to this vector.
    pub activities: Vec<String>,
    /// Per activity, in the order of `activities`.
    pub per_activity: Vec<ActivityCells>,
    /// Every recorded cell.
    pub cells: HashSet<Cell>,
    /// Recorded participations over the whole log.
    pub e2o_total: usize,
}

impl CellGrid {
    /// Build the grid and check, per activity, which cells reconstruct which.
    pub fn build(
        locel: &SlimLinkedOCEL,
        schema: &StructuralSchema,
        closure: &SchemaClosure,
    ) -> Self {
        Self::build_with_theta(locel, schema, closure, DETERMINATION_THETA)
    }

    /// [`Self::build`] at a chosen determination threshold. See [`DETERMINATION_THETA`].
    pub fn build_with_theta(
        locel: &SlimLinkedOCEL,
        schema: &StructuralSchema,
        closure: &SchemaClosure,
        theta: f64,
    ) -> Self {
        let mut activities: Vec<String> = locel.get_ev_types().map(str::to_string).collect();
        activities.sort();

        let mut per_activity = Vec::with_capacity(activities.len());
        let mut cells: HashSet<Cell> = HashSet::new();
        let mut e2o_total = 0usize;

        for (aix, act) in activities.iter().enumerate() {
            let evs: Vec<PerType> = locel
                .get_evs_of_type(act)
                .map(|e| {
                    let mut per: PerType = HashMap::new();
                    for o in e.get_e2o(locel) {
                        per.entry(schema.type_of[o]).or_default().insert(*o);
                    }
                    per
                })
                .collect();

            let mut present: Vec<ObjectTypeIndex> = evs
                .iter()
                .flat_map(|p| p.keys().copied())
                .collect::<HashSet<ObjectTypeIndex>>()
                .into_iter()
                .collect();
            present.sort();

            let counts: Vec<usize> = present
                .iter()
                .map(|t| evs.iter().map(|p| p.get(t).map_or(0, HashSet::len)).sum())
                .collect();
            e2o_total += counts.iter().sum::<usize>();
            for t in &present {
                cells.insert((aix, *t));
            }

            let k = present.len();
            let recon: Vec<Vec<Option<ReconRoute>>> = (0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| {
                            if i == j {
                                None
                            } else {
                                reconstructs(&evs, present[i], present[j], closure, theta)
                            }
                        })
                        .collect()
                })
                .collect();

            per_activity.push(ActivityCells {
                present,
                counts,
                recon,
            });
        }

        Self {
            activities,
            per_activity,
            cells,
            e2o_total,
        }
    }

    /// Number of recorded cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the log records no cell at all.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The canonical keep-set: at each activity, keep a cell unless a cell already kept there
    /// reconstructs it.
    ///
    /// A cell is cut only when a KEPT cell reconstructs it, since a cut justified by a cell
    /// that is itself cut would leave nothing to reconstruct from. Deciding strictly finer
    /// types first, and a class representative before the rest of its class, makes one pass
    /// sufficient, since a finer cell is never cut by a coarser one.
    pub fn canonical_keepset(&self, closure: &SchemaClosure, dir: FoldDirection) -> KeepSet {
        let coarsest_first = dir == FoldDirection::Coarsest;
        let mut kept: HashSet<Cell> = HashSet::new();
        let mut cut: Vec<CutDecision> = Vec::new();

        for (aix, cellset) in self.per_activity.iter().enumerate() {
            let present = &cellset.present;
            let k = present.len();
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by_key(|i| {
                let t = present[*i];
                let coarser = (0..k)
                    .filter(|j| closure.strictly_finer(t, present[*j]))
                    .count();
                let key = if coarsest_first { k - coarser } else { coarser };
                (std::cmp::Reverse(key), closure.rep[t] != t, t)
            });

            let mut kept_here = vec![false; k];
            for &j in &order {
                let t = present[j];
                let witness = (0..k).find(|i| {
                    let s = present[*i];
                    if !kept_here[*i] || cellset.recon[*i][j].is_none() || s == t {
                        return false;
                    }
                    if coarsest_first {
                        // Group folding: a kept witness in either direction justifies the
                        // cut, since the coarse types are decided first and the fine ones
                        // are folded away.
                        true
                    } else {
                        closure.strictly_finer(s, t) || (closure.rep[t] == s && closure.rep[s] == s)
                    }
                });
                match witness {
                    Some(i) => cut.push(CutDecision {
                        cell: (aix, t),
                        witness: present[i],
                        participations: cellset.counts[j],
                    }),
                    None => {
                        kept_here[j] = true;
                        kept.insert((aix, t));
                    }
                }
            }
        }
        KeepSet { kept, cut }
    }

    /// Keep every cell of every type that is not eliminated outright.
    ///
    /// A type is dropped only where it is derivable at *every* activity it occurs at, so no
    /// surviving type is partially cut and no kept type's flow is spliced. Every arc of every
    /// surviving type is exactly the arc the full model has (Prop. 4.3).
    pub fn type_closed_keepset(&self, closure: &SchemaClosure, dir: FoldDirection) -> HashSet<Cell> {
        let keep = self.canonical_keepset(closure, dir);
        let surviving: HashSet<ObjectTypeIndex> = keep.kept.iter().map(|(_, t)| *t).collect();
        self.cells
            .iter()
            .filter(|(_, t)| surviving.contains(t))
            .copied()
            .collect()
    }

    /// The smallest keep-set at every activity, ignoring which end of the refinement
    /// order a witness sits at.
    ///
    /// Not unique: if two types are mutually total, keeping either is optimal, so two runs
    /// can both be optimal and disagree.
    pub fn maximum_keepset(&self, closure: &SchemaClosure) -> HashSet<Cell> {
        let mut out = HashSet::new();
        for (aix, cellset) in self.per_activity.iter().enumerate() {
            let present = &cellset.present;
            let k = present.len();
            // Coarseness at this activity: how many of the other present types reach it
            // in the refinement order. `orders` is coarser than `items`.
            let coarseness: Vec<usize> = (0..k)
                .map(|i| {
                    (0..k)
                        .filter(|j| *j != i && closure.reach[present[*j]][present[i]])
                        .count()
                })
                .collect();
            for i in max_reduction(k, &cellset.recon_matrix(), &coarseness) {
                out.insert((aix, present[i]));
            }
        }
        out
    }
}

/// One cut, and the kept cell that justifies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutDecision {
    /// The cell removed.
    pub cell: Cell,
    /// The object type at the same activity whose cell reconstructs it.
    pub witness: ObjectTypeIndex,
    /// Participations the cut removes.
    pub participations: usize,
}

/// A keep-set, with the reason for every cut it makes.
#[derive(Debug, Clone)]
pub struct KeepSet {
    /// The cells that survive.
    pub kept: HashSet<Cell>,
    /// The cells removed, each with its witness.
    pub cut: Vec<CutDecision>,
}

impl KeepSet {
    /// Participations the cuts remove.
    pub fn participations_cut(&self) -> usize {
        self.cut.iter().map(|c| c.participations).sum()
    }

    /// The type pairs used as reconstruction witnesses. The annotation only needs the
    /// generators of these.
    pub fn witness_pairs(&self) -> HashSet<(ObjectTypeIndex, ObjectTypeIndex)> {
        self.cut.iter().map(|c| (c.witness, c.cell.1)).collect()
    }

    /// Object types no cell of the keep-set mentions: the types that leave the model.
    pub fn types_eliminated(&self, n_types: usize) -> Vec<ObjectTypeIndex> {
        let kept_types: HashSet<ObjectTypeIndex> = self.kept.iter().map(|(_, t)| *t).collect();
        (0..n_types).filter(|t| !kept_types.contains(t)).collect()
    }
}

/// Can the cell `(a,T)` be reconstructed from the cell `(a,S)` at the events of `a`?
///
/// Three routes, checked as set equality per event and accepted at a `theta` share of the
/// events (see [`share_at_least`]).
///
/// - (R-union) `obj^T(e)` is the union of the images of `obj^S(e)` under a subset of the
///   qualifiers. The subset is not searched: a qualifier is *admissible* if its image never
///   exceeds `obj^T(e)` at any event, and the union of the admissible ones is the largest
///   safe image, so if any subset works, that one does.
/// - (R-fun) `obj^T(e) = f[obj^S(e)]` for a map `f: S -> T`.
/// - (R-fib) `obj^T(e) = f^-1[obj^S(e)]` for a map `f: T -> S`.
fn reconstructs(
    events: &[PerType],
    s: ObjectTypeIndex,
    t: ObjectTypeIndex,
    closure: &SchemaClosure,
    theta: f64,
) -> Option<ReconRoute> {
    if let Some(by_qual) = closure.relations.get(&(s, t)) {
        let admissible: Vec<(&String, &ObjectFibre)> = by_qual
            .iter()
            .filter(|(_, r)| {
                events.iter().all(|per| match (per.get(&s), per.get(&t)) {
                    (Some(os), Some(ot)) => os
                        .iter()
                        .filter_map(|x| r.get(x))
                        .flat_map(|v| v.iter())
                        .all(|y| ot.contains(y)),
                    (Some(os), None) => os.iter().all(|x| r.get(x).is_none_or(HashSet::is_empty)),
                    _ => true,
                })
            })
            .collect();
        if !admissible.is_empty() {
            let ok = share_at_least(events, theta, |per| {
                let (Some(os), Some(ot)) = (per.get(&s), per.get(&t)) else {
                    return per.get(&s).is_none() && per.get(&t).is_none();
                };
                let image: HashSet<ObjectIndex> = admissible
                    .iter()
                    .flat_map(|(_, r)| os.iter().filter_map(move |x| r.get(x)))
                    .flat_map(|v| v.iter().copied())
                    .collect();
                &image == ot
            });
            if ok {
                return Some(ReconRoute::QualifiedUnion {
                    qualifiers: admissible.iter().map(|(q, _)| (*q).clone()).collect(),
                });
            }
        }
    }

    for (k, f) in closure.maps.get(&(s, t)).into_iter().flatten().enumerate() {
        let ok = share_at_least(events, theta, |per| {
            let (Some(os), Some(ot)) = (per.get(&s), per.get(&t)) else {
                // Both sides present or both missing: a T object with no S object at the
                // same event cannot be produced by a function of the S objects.
                return per.get(&s).is_none() && per.get(&t).is_none();
            };
            // f must be defined on every S object present, or the image is not the image
            // of obj^S(e) and the reconstruction is not exact.
            os.iter().all(|x| f.contains_key(x)) && {
                let image: HashSet<ObjectIndex> =
                    os.iter().filter_map(|x| f.get(x).copied()).collect();
                &image == ot
            }
        });
        if ok {
            return Some(ReconRoute::Function { witness: k });
        }
    }

    for (k, fib) in closure.fibres.get(&(t, s)).into_iter().flatten().enumerate() {
        let ok = share_at_least(events, theta, |per| {
            let (Some(os), Some(ot)) = (per.get(&s), per.get(&t)) else {
                return per.get(&s).is_none() && per.get(&t).is_none();
            };
            let expanded: HashSet<ObjectIndex> = os
                .iter()
                .filter_map(|x| fib.get(x))
                .flat_map(|v| v.iter().copied())
                .collect();
            &expanded == ot
        });
        if ok {
            return Some(ReconRoute::Fibre { witness: k });
        }
    }
    None
}

/// Whether `pred` holds at at least a `theta` share of the events.
///
/// At `theta = 1` this is "at every event", the setting under which the route returns the
/// recorded participations exactly. Below 1 a cell counts as determined while some of its
/// participations cannot be recomputed.
fn share_at_least(
    events: &[PerType],
    theta: f64,
    mut pred: impl FnMut(&PerType) -> bool,
) -> bool {
    if events.is_empty() {
        return true;
    }
    let good = events.iter().filter(|e| pred(e)).count();
    good as f64 >= theta * events.len() as f64
}

/// Smallest keep-set at one activity: every cut cell must be reconstructible from a KEPT
/// cell. Exponential in the number of types at the activity.
fn max_reduction(n: usize, recon: &[Vec<bool>], coarseness: &[usize]) -> Vec<usize> {
    if n > 20 {
        // The mask enumeration below is exponential; keep everything past 20 types.
        return (0..n).collect();
    }
    for size in 1..=n {
        // The minimum is not unique. Prefer the coarsest types, then the lexicographically
        // first, so the choice does not depend on enumeration order.
        let mut best: Option<(usize, Vec<usize>)> = None;
        for mask in 0u32..(1 << n) {
            if mask.count_ones() as usize != size || !derives_all(n, mask, recon) {
                continue;
            }
            let kept: Vec<usize> = (0..n).filter(|i| mask >> i & 1 == 1).collect();
            let score: usize = kept.iter().map(|i| coarseness[*i]).sum();
            if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                best = Some((score, kept));
            }
        }
        if let Some((_, kept)) = best {
            return kept;
        }
    }
    (0..n).collect()
}

/// Does the keep-set reach every cell at this activity, allowing chains?
///
/// Reconstruction is transitive, so determinacy is a closure: keeping `packages` recovers
/// `items` by fibre expansion, and `items` then recovers `products` by function application.
fn derives_all(n: usize, mask: u32, recon: &[Vec<bool>]) -> bool {
    let mut have = mask;
    loop {
        let mut next = have;
        for t in 0..n {
            if next >> t & 1 == 1 {
                continue;
            }
            if (0..n).any(|s| have >> s & 1 == 1 && recon[s][t]) {
                next |= 1 << t;
            }
        }
        if next == have {
            return have == (1u32 << n) - 1;
        }
        have = next;
    }
}
