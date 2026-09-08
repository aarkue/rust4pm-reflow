use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::{EventIndex, ObjectIndex},
    SlimLinkedOCEL,
};

use super::{
    arcs::ActivityIndexing,
    bounds::Bounds,
    cells::{ActivityIndex, Cell},
    facts::{asserted_by_type, asserted_of_type},
    saturation::Saturation,
    schema::ObjectTypeIndex,
};

/// An activity pair some type orders.
pub type Pair = (ActivityIndex, ActivityIndex);

/// The cells of one type and the participations they jointly write.
type PerType = BTreeMap<ObjectTypeIndex, (Vec<Cell>, Vec<(EventIndex, ObjectIndex)>)>;

/// What one written cell teaches the model.
///
/// Expansion has two filters. `theta` is correctness: do not write an object that was not
/// there. Novelty is value: only write a cell if the model learns an ordering it did not
/// have.
#[derive(Debug, Clone, PartialEq)]
pub struct Novelty {
    /// The cell that would be written.
    pub cell: Cell,
    /// Participations it writes.
    pub tuples: usize,
    /// Distinct objects it names.
    pub objects: usize,
    /// Activity pairs its type orders once it is written that no type orders on the
    /// recorded log.
    pub novel_pairs: Vec<Pair>,
    /// Pairs its type orders that some type already ordered.
    pub echo: usize,
    /// Participations per distinct object. A diagnostic, not a rule.
    pub fanout: f64,
}

impl Novelty {
    /// Pairs the model gains.
    pub fn novel(&self) -> usize {
        self.novel_pairs.len()
    }

    /// Whether the model learns anything at all. The admission rule.
    pub fn admissible(&self) -> bool {
        !self.novel_pairs.is_empty()
    }
}

/// Every activity pair some type orders, over a given log and cell set.
///
/// The baseline novelty is measured against. It must be measured on the recorded log:
/// every pair the expansion asserts is in the saturation by construction, so against the
/// saturation nothing would ever be novel.
pub fn ordered_pairs(bounds: &Bounds, cells: &HashSet<Cell>) -> HashSet<Pair> {
    asserted_by_type(bounds, cells).into_iter().flatten().collect()
}

/// Per admitted cell, what writing it alone would teach the model.
///
/// Each cell is scored alone against the recorded log, not against the other admitted
/// cells. Scoring the second of two cells that order the same new pair against the first
/// would make the answer depend on the enumeration order. [`novelty_by_type`] aggregates
/// per type instead.
pub fn novelty(
    locel: &SlimLinkedOCEL,
    acts: &ActivityIndexing,
    recorded_bounds: &Bounds,
    recorded_cells: &HashSet<Cell>,
    max: &Saturation,
) -> Vec<Novelty> {
    let before = ordered_pairs(recorded_bounds, recorded_cells);
    max.admitted_cells()
        .map(|c| {
            let mut cells = recorded_cells.clone();
            cells.insert(c.cell);
            let bounds = recorded_bounds.plus(locel, acts, &c.tuples);
            let own = asserted_of_type(&bounds, &cells, c.cell.1);
            score(c.cell, &c.tuples, &own, &before)
        })
        .collect()
}

/// Novelty of a whole type at once: write every admitted cell of the type and ask what
/// the type then orders.
///
/// Not the sum of the per-cell answers. Two cells of one type can order a new pair
/// together and none apart, because an ordering needs both of its endpoints.
pub fn novelty_by_type(
    locel: &SlimLinkedOCEL,
    acts: &ActivityIndexing,
    recorded_bounds: &Bounds,
    recorded_cells: &HashSet<Cell>,
    max: &Saturation,
) -> Vec<(ObjectTypeIndex, usize, Novelty)> {
    let before = ordered_pairs(recorded_bounds, recorded_cells);
    let mut per_type: PerType = BTreeMap::new();
    for c in max.admitted_cells() {
        let e = per_type.entry(c.cell.1).or_default();
        e.0.push(c.cell);
        e.1.extend(c.tuples.iter().copied());
    }
    let per_cell_fanout: HashMap<Cell, f64> = max
        .admitted_cells()
        .map(|c| {
            let objects = c.tuples.iter().map(|(_, o)| *o).collect::<HashSet<_>>().len();
            (c.cell, c.tuples.len() as f64 / objects.max(1) as f64)
        })
        .collect();
    per_type
        .into_iter()
        .map(|(t, (cells_of_type, mut tuples))| {
            tuples.sort();
            tuples.dedup();
            let mut cells = recorded_cells.clone();
            cells.extend(cells_of_type.iter().copied());
            let bounds = recorded_bounds.plus(locel, acts, &tuples);
            let own = asserted_of_type(&bounds, &cells, t);
            let mut n = score((0, t), &tuples, &own, &before);
            // Fan-out is averaged over the type's cells, not the type's tuples over the
            // type's objects: the union reuses the same objects at each activity.
            n.fanout = cells_of_type
                .iter()
                .filter_map(|c| per_cell_fanout.get(c))
                .sum::<f64>()
                / cells_of_type.len().max(1) as f64;
            (t, cells_of_type.len(), n)
        })
        .collect()
}

fn score(
    cell: Cell,
    tuples: &[(EventIndex, ObjectIndex)],
    own: &HashSet<Pair>,
    before: &HashSet<Pair>,
) -> Novelty {
    let mut novel_pairs: Vec<Pair> = own.difference(before).copied().collect();
    novel_pairs.sort_unstable();
    let objects = tuples.iter().map(|(_, o)| *o).collect::<HashSet<_>>().len();
    Novelty {
        cell,
        tuples: tuples.len(),
        objects,
        echo: own.len() - novel_pairs.len(),
        fanout: tuples.len() as f64 / objects.max(1) as f64,
        novel_pairs,
    }
}

/// The admitted cells that teach the model something, keyed for lookup.
pub fn novel_cells(rows: &[Novelty]) -> HashMap<Cell, usize> {
    rows.iter()
        .filter(|n| n.admissible())
        .map(|n| (n.cell, n.novel()))
        .collect()
}
