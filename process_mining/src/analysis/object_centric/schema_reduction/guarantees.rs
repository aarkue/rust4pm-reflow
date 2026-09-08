use std::collections::HashSet;

use super::{
    arcs::incidence_components,
    cells::{ActivityIndex, Cell, CellGrid},
    schema::ObjectTypeIndex,
};

/// What a keep-set guarantees a downstream technique.
///
/// Each field is a property of the keep-set and the schema alone, checked before any
/// discovery runs. They are the conditions of the ICPM 2027 paper's Props. 4.3 to 4.5,
/// in the same order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guarantees {
    /// Every object type is kept at all of its cells or at none of them.
    ///
    /// Then per-type models (a per-type directly-follows graph, a flattened log, anything
    /// discovered one type at a time) are identical to the full model's for the surviving
    /// types.
    pub type_closed: bool,
    /// Types the keep-set drops entirely.
    pub types_dropped: Vec<ObjectTypeIndex>,
    /// Types kept at some activities and cut at others. Non-empty exactly when
    /// `type_closed` is false.
    pub types_split: Vec<ObjectTypeIndex>,
    /// Types kept at every activity that has any kept cell.
    ///
    /// One such type makes declarative constraints recoverable: it witnesses every cut
    /// cell, so the same map serves at both ends of a constraint. The condition fails
    /// exactly when a keep-set hands the process from one type to another.
    pub spanning_types: Vec<ObjectTypeIndex>,
    /// Components of the activity-type incidence graph. More than one means the model
    /// falls into parts no kept type relates.
    pub incidence_components: usize,
    /// Activities the keep-set leaves with no cell at all.
    pub activities_emptied: Vec<ActivityIndex>,
}

impl Guarantees {
    /// Evaluate a keep-set against the grid it was taken from.
    ///
    /// Eventually-follows is exact for any keep-set (Prop. 4.4) and so is not a field.
    /// The directly-follows over-approximation needs the log and is measured by
    /// [`df_fidelity`](super::df_fidelity).
    pub fn evaluate(grid: &CellGrid, kept: &HashSet<Cell>, n_types: usize) -> Self {
        let mut types_dropped = Vec::new();
        let mut types_split = Vec::new();
        for t in 0..n_types {
            let total = grid.cells.iter().filter(|(_, ty)| *ty == t).count();
            if total == 0 {
                continue;
            }
            let held = kept.iter().filter(|(_, ty)| *ty == t).count();
            if held == 0 {
                types_dropped.push(t);
            } else if held < total {
                types_split.push(t);
            }
        }

        let live: HashSet<ActivityIndex> = kept.iter().map(|(a, _)| *a).collect();
        let spanning_types: Vec<ObjectTypeIndex> = (0..n_types)
            .filter(|t| {
                !live.is_empty() && live.iter().all(|a| kept.contains(&(*a, *t)))
            })
            .collect();

        let activities_emptied: Vec<ActivityIndex> = (0..grid.activities.len())
            .filter(|a| {
                !live.contains(a) && grid.cells.iter().any(|(ca, _)| ca == a)
            })
            .collect();

        Self {
            type_closed: types_split.is_empty(),
            types_dropped,
            types_split,
            spanning_types,
            incidence_components: incidence_components(kept),
            activities_emptied,
        }
    }

    /// Whether declarative constraints are recoverable modulo the schema: some type
    /// survives at every activity.
    pub fn constraints_preserved(&self) -> bool {
        !self.spanning_types.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::cells::ActivityCells;

    /// Two activities, two types: type 0 at both, type 1 only at the first.
    fn grid() -> CellGrid {
        CellGrid {
            activities: vec!["a".into(), "b".into()],
            per_activity: vec![
                ActivityCells {
                    present: vec![0, 1],
                    counts: vec![1, 1],
                    recon: vec![vec![None, None], vec![None, None]],
                },
                ActivityCells {
                    present: vec![0],
                    counts: vec![1],
                    recon: vec![vec![None]],
                },
            ],
            cells: HashSet::from([(0, 0), (0, 1), (1, 0)]),
            e2o_total: 3,
        }
    }

    #[test]
    fn dropping_a_whole_type_is_type_closed() {
        let g = grid();
        let kept = HashSet::from([(0, 0), (1, 0)]);
        let got = Guarantees::evaluate(&g, &kept, 2);
        assert!(got.type_closed);
        assert_eq!(got.types_dropped, vec![1]);
        assert!(got.types_split.is_empty());
        assert_eq!(got.spanning_types, vec![0]);
        assert!(got.constraints_preserved());
    }

    #[test]
    fn cutting_one_cell_of_a_type_is_not_type_closed() {
        let g = grid();
        let kept = HashSet::from([(0, 0), (0, 1)]);
        let got = Guarantees::evaluate(&g, &kept, 2);
        assert!(!got.type_closed);
        assert_eq!(got.types_split, vec![0]);
        assert_eq!(got.activities_emptied, vec![1]);
    }

    #[test]
    fn a_handoff_keeps_no_spanning_type() {
        let g = grid();
        // type 1 at activity a, type 0 at activity b: the process is handed over.
        let kept = HashSet::from([(0, 1), (1, 0)]);
        let got = Guarantees::evaluate(&g, &kept, 2);
        assert!(got.spanning_types.is_empty());
        assert!(!got.constraints_preserved());
        assert_eq!(got.incidence_components, 2);
    }
}
