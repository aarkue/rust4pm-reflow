use std::collections::HashSet;

use super::{
    cells::{Cell, CellGrid},
    schema::ObjectTypeIndex,
};

/// Where one cell ends up.
///
/// Three states of one fact: the cell flows, or it is non-flow and the annotation of the
/// tagged log then tells an implied cell from an involved one. Every recorded tuple stays
/// in the log in all three, so the state is a reading of the tags, not a deletion.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum CellState {
    /// Tagged flow: the tuples contribute arcs.
    Flow,
    /// Tagged non-flow and nothing at the activity determines it. A badge in the activity
    /// node carrying counts.
    Involvement,
    /// Tagged non-flow and a flow cell at the activity determines it through a route, so
    /// the annotation draws it as a map.
    Implied,
}

impl CellState {
    /// The name used in the census and in the paper.
    pub fn label(&self) -> &'static str {
        match self {
            CellState::Flow => "flow",
            CellState::Involvement => "involvement",
            CellState::Implied => "implied",
        }
    }
}

/// The end-to-end result: every recorded cell placed, and what the flow layer added.
#[derive(Debug, Clone, Default)]
pub struct Assignment {
    /// Cells that draw arcs. The search's output, which may include cells the extraction
    /// did not record.
    pub flow: Vec<Cell>,
    /// Recorded cells that no flow cell at their activity determines. Nothing recomputes
    /// them, so they cannot be deleted and stay as a badge.
    pub involvement: Vec<Cell>,
    /// Recorded cells a flow cell at their activity determines. Their tuples stay in the
    /// log, tagged non-flow, and the annotation names the route that recomputes them.
    pub implied: Vec<Cell>,
    /// Flow cells the extraction did not record. Written, and marked as written.
    pub expanded: Vec<Cell>,
}

impl Assignment {
    /// How many recorded cells land in each state, plus the expansions.
    pub fn tally(&self) -> (usize, usize, usize, usize) {
        (
            self.flow.len() - self.expanded.len(),
            self.involvement.len(),
            self.implied.len(),
            self.expanded.len(),
        )
    }

    /// The state of one recorded cell.
    pub fn state_of(&self, c: Cell) -> Option<CellState> {
        if self.flow.contains(&c) {
            Some(CellState::Flow)
        } else if self.involvement.contains(&c) {
            Some(CellState::Involvement)
        } else if self.implied.contains(&c) {
            Some(CellState::Implied)
        } else {
            None
        }
    }
}

/// Place every recorded cell, given the flow layer and whatever the analyst forced.
///
/// A non-flow cell is implied when a flow cell at its activity determines it and involved
/// otherwise (Def. Annotation).
///
/// `dropped` overrules that: a cell named there is reported implied whether or not anything
/// determines it.
///
/// Determinacy is the per-event set equality the grid checked when it was built, walked
/// transitively over the activity ([`ActivityCells::determined_by`]), with no slack.
///
/// [`ActivityCells::determined_by`]: super::ActivityCells::determined_by
pub fn assign(grid: &CellGrid, flow: &HashSet<Cell>, dropped: &HashSet<Cell>) -> Assignment {
    let mut out = Assignment {
        flow: flow.iter().copied().collect(),
        ..Default::default()
    };
    out.flow.sort_unstable();
    out.expanded = out
        .flow
        .iter()
        .filter(|c| !grid.cells.contains(*c))
        .copied()
        .collect();

    for (a, here) in grid.per_activity.iter().enumerate() {
        let kept: Vec<ObjectTypeIndex> = here
            .present
            .iter()
            .filter(|t| flow.contains(&(a, **t)))
            .copied()
            .collect();
        let reached = here.determined_by(&kept);
        for (j, t) in here.present.iter().enumerate() {
            if flow.contains(&(a, *t)) {
                continue;
            }
            if reached[j] || dropped.contains(&(a, *t)) {
                out.implied.push((a, *t));
            } else {
                out.involvement.push((a, *t));
            }
        }
    }
    out.involvement.sort_unstable();
    out.implied.sort_unstable();
    out
}

/// Participations a set of cells carries, over the recorded log.
///
/// Counted in participations, not cells, since cells differ in size by orders of
/// magnitude.
pub fn participations(grid: &CellGrid, cells: &[Cell]) -> usize {
    cells
        .iter()
        .filter_map(|(a, t)| {
            let here = grid.per_activity.get(*a)?;
            here.slot(*t).map(|j| here.counts[j])
        })
        .sum()
}
