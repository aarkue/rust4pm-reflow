use std::collections::{HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::{EventIndex, ObjectIndex},
    SlimLinkedOCEL,
};

use super::{
    arcs::ActivityIndexing,
    bounds::Bounds,
    cells::{ActivityIndex, Cell, CellGrid},
    expansion::{
        expansion_candidates, expansion_work, ExpansionCandidate, ExpansionDirection, Route,
        EXPANSION_WORK_BUDGET,
    },
    facts::asserted_by_type,
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// `max`: the log plus every candidate cell `theta` admits, written whole.
///
/// The coverage constraint is stated against `max` instead of the recorded log, so that
/// the objective does not depend on how generously the extraction recorded E2O. `max` is
/// not a normal form: it is only what the schema derives and `theta` admits.
///
/// Everything downstream reads `cells`, `bounds` and `cost` and never the log again:
/// coverage, novelty and both searches are activity-pair arithmetic over the bounds.
#[derive(Debug, Clone)]
pub struct Saturation {
    /// The share of a cell's tuples that had to be inside their own object's lifetime.
    pub theta: f64,
    /// Which way candidates were derived.
    pub direction: ExpansionDirection,
    /// Route-object checks the enumeration needs, from [`expansion_work`].
    pub work: u64,
    /// Whether the enumeration ran. When it did not, `max` is the recorded log and every
    /// number derived from it is a recorded-log number.
    pub within_budget: bool,
    /// Every candidate cell, admitted or not, in the order [`expansion_candidates`]
    /// returns them.
    pub candidates: Vec<ExpansionCandidate>,
    /// Indices into [`candidates`](Self::candidates) of the cells `theta` admitted.
    pub admitted: Vec<usize>,
    /// Cells of `max`: recorded plus admitted.
    pub cells: HashSet<Cell>,
    /// The participations the admitted cells write, sorted and deduplicated.
    pub written: Vec<(EventIndex, ObjectIndex)>,
    /// Per-object per-activity bounds over `max`, which is what every coverage question
    /// is answered from.
    pub bounds: Bounds,
    /// Participations per cell of `max`: recorded where the cell is recorded, recorded
    /// plus written where both. The tie-break of the objective.
    pub cost: HashMap<Cell, usize>,
}

impl Saturation {
    /// Saturate, or report that the enumeration is over budget and return the recorded
    /// log as `max`.
    ///
    /// `bounds` are the recorded ones. The saturated ones are these widened by what was
    /// written. `routes` must be the [`agreed_routes`](super::agreed_routes), not the raw
    /// chains: a pair two chains contradict determines nothing, and unioning them would
    /// inflate `max` with pairs no single map derives.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        locel: &SlimLinkedOCEL,
        schema: &StructuralSchema,
        grid: &CellGrid,
        acts: &ActivityIndexing,
        routes: &[Route],
        bounds: &Bounds,
        theta: f64,
        dir: ExpansionDirection,
    ) -> Self {
        let work = expansion_work(routes, grid, dir);
        let within_budget = work <= EXPANSION_WORK_BUDGET;
        let candidates = if within_budget {
            expansion_candidates(locel, routes, schema, grid, acts, theta, dir)
        } else {
            Vec::new()
        };
        let admitted: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.admissible)
            .map(|(i, _)| i)
            .collect();

        let mut written: Vec<(EventIndex, ObjectIndex)> = admitted
            .iter()
            .flat_map(|i| candidates[*i].tuples.iter().copied())
            .collect();
        written.sort();
        written.dedup();

        let mut cells = grid.cells.clone();
        cells.extend(admitted.iter().map(|i| candidates[*i].cell));

        let mut cost: HashMap<Cell, usize> = HashMap::new();
        for (a, here) in grid.per_activity.iter().enumerate() {
            for (j, t) in here.present.iter().enumerate() {
                cost.insert((a, *t), here.counts[j]);
            }
        }
        for i in &admitted {
            *cost.entry(candidates[*i].cell).or_default() += candidates[*i].tuples.len();
        }

        let sat_bounds = bounds.plus(locel, acts, &written);
        Self {
            theta,
            direction: dir,
            work,
            within_budget,
            candidates,
            admitted,
            cells,
            written,
            bounds: sat_bounds,
            cost,
        }
    }

    /// The admitted candidate cells.
    pub fn admitted_cells(&self) -> impl Iterator<Item = &ExpansionCandidate> {
        self.admitted.iter().map(|i| &self.candidates[*i])
    }

    /// The coverage target: every activity pair some type orders in `max`.
    ///
    /// Types are dropped from the pair on purpose. A pair delivered by a different type
    /// from the one that first ordered it is still delivered, so the target is a set of
    /// activity pairs, not of (type, pair) triples.
    pub fn target_pairs(&self) -> HashSet<(ActivityIndex, ActivityIndex)> {
        asserted_by_type(&self.bounds, &self.cells)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Participations a cell set carries in `max`.
    pub fn participations(&self, cells: &HashSet<Cell>) -> usize {
        cells.iter().map(|c| self.cost.get(c).copied().unwrap_or(0)).sum()
    }

    /// Cells of `max` that the extraction did not record.
    pub fn expanded_cells(&self) -> HashSet<Cell> {
        self.admitted_cells().map(|c| c.cell).collect()
    }

    /// Objects per type, for the "coarsest" tie-break the handoff constraint reads.
    pub fn objects_per_type(schema: &StructuralSchema) -> Vec<usize> {
        let mut n = vec![0usize; schema.types.len()];
        for t in schema.type_of.values() {
            n[*t] += 1;
        }
        n
    }

    /// Types that order at least one activity pair in `max`.
    ///
    /// The flow-eligibility guard, threshold-free: a type that orders nothing cannot
    /// carry a phase.
    pub fn flow_eligible(&self, n_types: usize) -> HashSet<ObjectTypeIndex> {
        let per_type = asserted_by_type(&self.bounds, &self.cells);
        (0..n_types)
            .filter(|t| per_type.get(*t).is_some_and(|s| !s.is_empty()))
            .collect()
    }
}
