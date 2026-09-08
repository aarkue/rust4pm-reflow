//! Coverage report of an assignment.
//!
//! An assignment is preserving when its kept cells cover every assertion of the log. This
//! module answers, for a cell the assignment did not keep at flow, which of its type's
//! assertions at that activity the kept cells still show, and which they leave unshown:
//! the residual its leaving the flow layer produces.
//!
//! Three mechanisms deliver an assertion: [`drawn`] (a kept type orders it directly, both
//! endpoints kept), [`Chained`] (composed through a third activity), and
//! [`route_delivered`] (an object-to-object map carries it).

use std::collections::{HashMap, HashSet};

use super::{
    bounds::Bounds,
    cells::Cell,
    coverage::{drawn, Chained},
    expansion::Route,
    facts::Facts,
    novelty::Pair,
    route_delivery::route_delivered,
};

/// How a covered assertion reaches the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoveredBy {
    /// A kept type orders the pair directly, kept at both endpoints.
    Drawn,
    /// Reached by composing drawn pairs through a third activity.
    Chained,
    /// Delivered by an object-to-object route, not by activity composition.
    Routed,
}

/// One non-flow cell's assertions, split by whether the keep-set still shows them.
///
/// The assertions considered are those of `cell`'s type naming `cell`'s activity as an
/// endpoint, the pairs a reader loses exactly when this cell is the one taken out of the
/// flow layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellResidual {
    /// The non-flow cell these assertions are named against.
    pub cell: Cell,
    /// Assertions still shown, and how.
    pub covered: Vec<(Pair, CoveredBy)>,
    /// Assertions the keep-set would leave unshown: the residual.
    pub residual: Vec<Pair>,
}

impl CellResidual {
    /// Whether every assertion this cell names is still shown.
    pub fn is_preserving(&self) -> bool {
        self.residual.is_empty()
    }
}

/// The coverage report of one assignment: every ordering assertion of the full log, how
/// many the keep-set delivers and by which mechanism, and the per-cell breakdown for
/// cells outside flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualReport {
    /// Assertions of every cell the keep-set left outside flow, split covered/uncovered.
    pub cells: Vec<CellResidual>,
    /// Ordering assertions of the full log (`full.ordering.len()`).
    pub total: usize,
    /// Assertions a kept type orders directly, both endpoints kept.
    pub drawn: usize,
    /// Assertions delivered only by composing drawn pairs through a third activity.
    pub chained: usize,
    /// Assertions delivered only by an object-to-object route.
    pub routed: usize,
    /// Assertions covered by none of the three mechanisms.
    pub residual: usize,
}

impl ResidualReport {
    /// Whether the assignment is preserving: the residual is empty.
    pub fn is_preserving(&self) -> bool {
        self.residual == 0
    }

    /// Cells whose leaving the flow layer is not preserving, i.e. left a non-empty residual.
    pub fn non_preserving_cells(&self) -> Vec<Cell> {
        self.cells
            .iter()
            .filter(|c| !c.is_preserving())
            .map(|c| c.cell)
            .collect()
    }
}

/// Compute the coverage report of `kept` against `full`, the full log's facts.
///
/// `full` must come from the maximal keep-set (`facts_from(&max.bounds, &max.cells,
/// tau)`), the same basis a search's `target` is stated against. Facts from `kept` itself
/// would only describe what the assignment already shows.
pub fn residual_report(
    bounds: &Bounds,
    full: &Facts,
    kept: &HashSet<Cell>,
    routes: &[Route],
    n_activities: usize,
) -> ResidualReport {
    let drawn_pairs = drawn(bounds, kept);
    let chain = Chained::of(&drawn_pairs, n_activities);
    let routed_pairs = route_delivered(bounds, kept, routes);

    let classify = |p: Pair| -> Option<CoveredBy> {
        if drawn_pairs.contains(&p) {
            Some(CoveredBy::Drawn)
        } else if chain.holds(p.0, p.1) {
            Some(CoveredBy::Chained)
        } else if routed_pairs.contains(&p) {
            Some(CoveredBy::Routed)
        } else {
            None
        }
    };

    type PerCell = HashMap<Cell, (Vec<(Pair, CoveredBy)>, Vec<Pair>)>;

    let mut counts = [0usize; 4]; // drawn, chained, routed, residual
    let mut per_cell: PerCell = HashMap::new();

    let mut ordering: Vec<_> = full.ordering.iter().collect();
    ordering.sort_unstable();
    for (t, x, y) in ordering {
        let p = (*x, *y);
        let by = classify(p);
        counts[match by {
            Some(CoveredBy::Drawn) => 0,
            Some(CoveredBy::Chained) => 1,
            Some(CoveredBy::Routed) => 2,
            None => 3,
        }] += 1;
        for a in [*x, *y] {
            let cell = (a, *t);
            if kept.contains(&cell) {
                continue;
            }
            let entry = per_cell.entry(cell).or_default();
            match by {
                Some(cb) => entry.0.push((p, cb)),
                None => entry.1.push(p),
            }
        }
    }

    let mut cells: Vec<CellResidual> = per_cell
        .into_iter()
        .map(|(cell, (mut covered, mut residual))| {
            covered.sort_unstable();
            residual.sort_unstable();
            CellResidual {
                cell,
                covered,
                residual,
            }
        })
        .collect();
    cells.sort_by_key(|c| c.cell);

    ResidualReport {
        cells,
        total: full.ordering.len(),
        drawn: counts[0],
        chained: counts[1],
        routed: counts[2],
        residual: counts[3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::{
        bounds::ObjectBounds, facts::facts_from,
    };

    /// Three types over three activities (pick=0, create=1, ship=2): `case1` draws
    /// `0 < 1`, `case2` draws `1 < 2`, `shortcut` draws `0 < 2` directly, the same pair
    /// `case1` and `case2` chain together.
    fn scenario() -> (Bounds, HashSet<Cell>) {
        let ob = |object: u32, at: &[(usize, i64, i64)]| ObjectBounds {
            object: object.into(),
            at: at.to_vec(),
            times: at.iter().map(|(a, _, _)| (*a, 1)).collect(),
        };
        let bounds = Bounds {
            per_type: vec![
                vec![ob(0, &[(0, 10, 10), (1, 20, 20)])], // case1: pick < create
                vec![ob(1, &[(1, 20, 20), (2, 30, 30)])], // case2: create < ship
                vec![ob(2, &[(0, 10, 10), (2, 30, 30)])], // shortcut: pick < ship
            ],
        };
        let full_kept: HashSet<Cell> =
            HashSet::from([(0, 0), (1, 0), (1, 1), (2, 1), (0, 2), (2, 2)]);
        (bounds, full_kept)
    }

    #[test]
    fn a_non_flow_cell_the_chain_still_covers_has_an_empty_residual() {
        let (bounds, full_kept) = scenario();
        let full = facts_from(&bounds, &full_kept, 0.0);
        // Take `shortcut` out of the flow layer everywhere: case1 + case2 still chain-cover
        // its own pair.
        let kept: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (1, 1), (2, 1)]);

        let report = residual_report(&bounds, &full, &kept, &[], 3);
        assert!(report.is_preserving(), "{report:?}");
        assert_eq!(report.chained, 1);

        let shortcut_at_pick = report.cells.iter().find(|c| c.cell == (0, 2)).unwrap();
        assert!(shortcut_at_pick.is_preserving());
        assert_eq!(shortcut_at_pick.covered, vec![((0, 2), CoveredBy::Chained)]);
    }

    #[test]
    fn forcing_the_sole_carrier_out_leaves_exactly_its_pair_as_residual() {
        let (bounds, full_kept) = scenario();
        let full = facts_from(&bounds, &full_kept, 0.0);
        // Take `case2` out of the flow layer: nothing else touches activity 1 (create) and
        // activity 2 (ship)
        // together, so `1 < 2` is lost outright.
        let kept: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (0, 2), (2, 2)]);

        let report = residual_report(&bounds, &full, &kept, &[], 3);
        assert!(!report.is_preserving());
        assert_eq!(report.residual, 1);
        assert_eq!(report.non_preserving_cells(), vec![(1, 1), (2, 1)]);

        for cell in [(1, 1), (2, 1)] {
            let row = report.cells.iter().find(|c| c.cell == cell).unwrap();
            assert_eq!(row.residual, vec![(1, 2)]);
            assert!(row.covered.is_empty());
        }
    }

    #[test]
    fn a_flow_cell_is_never_reported_as_a_non_flow_residual() {
        let (bounds, full_kept) = scenario();
        let full = facts_from(&bounds, &full_kept, 0.0);
        let report = residual_report(&bounds, &full, &full_kept, &[], 3);
        // Every cell is kept, so there is nothing outside flow to report on.
        assert!(report.cells.is_empty());
        assert!(report.is_preserving());
        assert_eq!(report.drawn, 3);
    }

    /// The search enforces coverage, so the residual of every cell it takes out of the flow
    /// layer must be empty.
    #[test]
    fn the_default_search_assignment_never_leaves_a_residual() {
        use crate::analysis::object_centric::schema_reduction::{
            agreed_routes, best_of_two, novel_cells, novelty, ActivityIndexing, CellGrid,
            ExpansionDirection, Saturation, SchemaClosure, SearchInput, StructuralSchema,
            TraceVariants, DEFAULT_NOISE_THRESHOLD, DEFAULT_THETA,
        };
        use crate::core::chrono::DateTime;
        use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;

        let mut ocel = SlimLinkedOCEL::new();
        for t in ["items", "packages"] {
            ocel.add_object_type(t, Vec::new());
        }
        for a in ["pick item", "create package", "ship package"] {
            ocel.add_event_type(a, Vec::new());
        }
        let ts = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        let mut clock = 1_700_000_000_000i64;
        for n in 0..3 {
            let item = ocel
                .add_object("items", Some(format!("i{n}")), Vec::new(), Vec::new())
                .unwrap();
            let package = ocel
                .add_object(
                    "packages",
                    Some(format!("p{n}")),
                    Vec::new(),
                    vec![("packed".into(), item)],
                )
                .unwrap();
            clock += 1000;
            ocel.add_event(
                "pick item",
                ts(clock),
                Some(format!("pick-{n}")),
                Vec::new(),
                vec![("item".into(), item)],
            );
            clock += 1000;
            ocel.add_event(
                "create package",
                ts(clock),
                Some(format!("create-{n}")),
                Vec::new(),
                vec![("item".into(), item), ("package".into(), package)],
            );
            clock += 1000;
            ocel.add_event(
                "ship package",
                ts(clock),
                Some(format!("ship-{n}")),
                Vec::new(),
                vec![("package".into(), package)],
            );
        }

        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let grid = CellGrid::build(&ocel, &schema, &closure);
        let acts = ActivityIndexing::build(&ocel, &grid);
        let bounds = Bounds::build(&ocel, &schema, &acts);
        let (routes, _clashes) = agreed_routes(&schema);
        let max = Saturation::build(
            &ocel,
            &schema,
            &grid,
            &acts,
            &routes,
            &bounds,
            DEFAULT_THETA,
            ExpansionDirection::default(),
        );
        let variants = TraceVariants::build_with(&ocel, &schema, &acts, &max.written);
        let objects = Saturation::objects_per_type(&schema);
        let novel = novel_cells(&novelty(&ocel, &acts, &bounds, &grid.cells, &max));
        let mut allowed = grid.cells.clone();
        allowed.extend(novel.keys().copied());
        let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
        target.sort_unstable();
        let n_activities = grid.activities.len();

        let input = SearchInput {
            max: &max,
            variants: &variants,
            allowed: &allowed,
            recorded: &grid.cells,
            target: &target,
            objects_per_type: &objects,
            rep: &closure.rep,
            types: &schema.types,
            n_activities,
        };
        let (best, _) = best_of_two(&input);
        assert!(best.ran);
        assert!(best.coverage.complete(), "search must cover its own target");

        let full = facts_from(&max.bounds, &max.cells, DEFAULT_NOISE_THRESHOLD);
        let report = residual_report(&max.bounds, &full, &best.cells, &routes, n_activities);
        assert!(
            report.is_preserving(),
            "search enforces coverage: {report:?}"
        );
    }
}
