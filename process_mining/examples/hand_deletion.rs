//! The hand-deletion baseline: what ReFlow gives up when it may only act on whole object
//! types.
//!
//! Removing whole object types before discovery is common practice
//! (`Liss2025ObjectCentricCausalNets`, `Christfort2024DiscoveryOfObjectCentricDeclarativeModels`
//! and `Kusters2025OCDECLARE` all do it). Rather than reimplementing another method, this runs
//! ReFlow itself under the restriction that practice imposes.
//!
//! A keep-set is *type-closed* when, for every object type, either all of its recorded cells
//! flow or none of them do. The best type-closed keep-set is found by enumeration over subsets
//! of the flow-eligible types, scored on exactly the objective `search.rs` uses: coverage of the
//! log's asserted orderings first, then arcs over `max`, then participations, then cell count.
//! Enumeration is exhaustive up to [`MAX_ENUM_TYPES`] types and the run says so when it is not,
//! since a baseline that silently sampled would understate what hand deletion can do.
//!
//! Reported per log: what the per-cell oracle keeps and covers, what the best type-closed
//! keep-set keeps and covers, and the orderings the restriction loses. That last number is the
//! one the introduction's claim needs.
//!
//! Usage: `cargo run --release --example hand_deletion -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, coverage, delivered_facts, facts_from, handoff, ActivityIndexing, Bounds,
        Cell, CellGrid, ExpansionDirection, Facts, Pair, Saturation, SchemaClosure, SearchInput,
        StructuralSchema, TraceVariants, DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};

/// Above this many flow-eligible types the subset enumeration is skipped rather than sampled.
///
/// `2^16` scored keep-sets is seconds; the corpus tops out at 12 types (Hinge), so every log
/// here is enumerated exhaustively and the guard never fires. It exists so that a larger log
/// reports "not enumerated" instead of quietly returning the best of an arbitrary subset.
const MAX_ENUM_TYPES: usize = 16;

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

struct LogReport {
    log: String,
    recorded_cells: usize,
    target_pairs: usize,
    types_eligible: usize,
    enumerated: bool,
    /// The per-cell oracle: the flow layer `handoff` returns.
    cell_cells: usize,
    cell_covered: usize,
    cell_arcs: usize,
    cell_orderings: usize,
    cell_roles: usize,
    cell_drawn: usize,
    cell_bare_orderings: usize,
    cell_bare_roles: usize,
    /// The best type-closed keep-set found.
    closed_types: Vec<String>,
    closed_cells: usize,
    closed_covered: usize,
    closed_arcs: usize,
    closed_orderings: usize,
    closed_roles: usize,
    closed_drawn: usize,
    closed_bare_orderings: usize,
    closed_bare_roles: usize,
    /// Facts the whole log makes, the denominator for both pairs above.
    full_orderings: usize,
    full_roles: usize,
}

impl LogReport {
    /// Orderings the per-cell layer shows that no type-closed keep-set does.
    fn lost(&self) -> usize {
        self.cell_covered.saturating_sub(self.closed_covered)
    }

    /// Role facts, i.e. activity pairs no object of a type ever attends both of, that the
    /// per-cell layer still delivers and the best type-closed keep-set does not.
    ///
    /// This is the half an ordering measure cannot see, and it is what a resource type
    /// carries. Hand deletion removes a type outright, so its roles go with it.
    fn roles_lost(&self) -> usize {
        self.cell_roles.saturating_sub(self.closed_roles)
    }
}

fn run(path: &str) -> LogReport {
    let stem = log_stem(path);
    println!("\n{}\n{stem}\n{}", "=".repeat(72), "=".repeat(72));

    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);

    // Identical to the pipeline every other example runs, so the only difference measured is
    // the restriction to whole types.
    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);
    let bounds = Bounds::build(&locel, &schema, &acts);
    let routes = agreed_routes(&schema).0;
    let max = Saturation::build(
        &locel,
        &schema,
        &grid,
        &acts,
        &routes,
        &bounds,
        DEFAULT_THETA,
        ExpansionDirection::default(),
    );

    let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
    target.sort_unstable();
    let variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
    let objects = Saturation::objects_per_type(&schema);
    let allowed = grid.cells.clone();
    let input = SearchInput {
        max: &max,
        variants: &variants,
        allowed: &allowed,
        recorded: &grid.cells,
        target: &target,
        objects_per_type: &objects,
        rep: &closure.rep,
        types: &schema.types,
        n_activities: grid.activities.len(),
    };

    let n_acts = grid.activities.len();
    let per_cell = handoff(&input);
    let cell_cov = coverage(&max.bounds, &per_cell.cells, &target, n_acts);
    let (cell_covered, cell_drawn) = (cell_cov.chained, cell_cov.drawn);

    // Only types that order something can carry a phase, so a type outside this set is dead
    // weight in the enumeration and its cells never help coverage.
    let eligible: Vec<usize> = {
        let mut v: Vec<usize> = max.flow_eligible(schema.types.len()).into_iter().collect();
        v.sort_unstable();
        v
    };
    let cells_of: HashMap<usize, Vec<Cell>> = eligible
        .iter()
        .map(|&t| (t, grid.cells.iter().filter(|(_, tt)| *tt == t).copied().collect()))
        .collect();

    let enumerated = eligible.len() <= MAX_ENUM_TYPES;
    let mut best: Option<(usize, usize, usize, usize, Vec<usize>)> = None;
    if enumerated {
        for mask in 0u32..(1u32 << eligible.len()) {
            let chosen: Vec<usize> = eligible
                .iter()
                .enumerate()
                .filter(|(i, _)| mask >> i & 1 == 1)
                .map(|(_, &t)| t)
                .collect();
            let kept: HashSet<Cell> =
                chosen.iter().flat_map(|t| cells_of[t].iter().copied()).collect();
            let cov = coverage(&max.bounds, &kept, &target, n_acts).chained;
            let arcs = variants.arc_set(&kept).len();
            let parts = max.participations(&kept);
            // Same ranking as `best_of_two`: coverage first, then arcs, then participations,
            // then size. Whole types are the only thing that changed.
            let key = (std::cmp::Reverse(cov), arcs, parts, kept.len());
            let better = match &best {
                None => true,
                Some((c, a, p, n, _)) => key < (std::cmp::Reverse(*c), *a, *p, *n),
            };
            if better {
                best = Some((cov, arcs, parts, kept.len(), chosen));
            }
        }
    }

    let (closed_covered, closed_arcs, _, closed_cells, closed_type_ids) =
        best.clone().unwrap_or((0, 0, 0, 0, Vec::new()));
    let closed_kept_early: HashSet<Cell> = closed_type_ids
        .iter()
        .flat_map(|t| cells_of[t].iter().copied())
        .collect();
    // Drawn alone, reported beside drawn-plus-chained because the closure is the step that
    // saturates: one broad type kept at all of its activities chains to nearly everything, and
    // that is exactly the keep-set a whole-type restriction produces. If the two columns
    // diverge, the no-loss reading of the chained column is an artefact of the closure.
    let closed_drawn = coverage(&max.bounds, &closed_kept_early, &target, n_acts).drawn;
    let closed_types: Vec<String> =
        closed_type_ids.iter().map(|&t| schema.types[t].clone()).collect();

    // The half an ordering measure cannot see. `Facts` is orderings plus roles, a role being
    // an activity pair no object of the type ever attends both of, which is what a resource
    // type carries. `delivered_facts` credits a fact the keep-set still pushes forward through
    // a map, so a non-flow type whose carrier determines it does not count as a loss.
    let n_types = schema.types.len();
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
    let delivered = |kept: &HashSet<Cell>| -> Facts {
        let reduced = facts_from(&max.bounds, kept, 0.0);
        delivered_facts(&grid, &full_facts, &reduced, kept, n_types)
    };
    let cell_delivered = delivered(&per_cell.cells);
    // Without map credit. `delivered_facts` pushes a deleted type's fact forward through the
    // schema, which hands the baseline this paper's own machinery: hand deletion as practiced
    // has no schema and no way to recompute anything. These columns give neither side that
    // credit, so they are the comparison against the practice rather than against the framework.
    let bare = |kept: &HashSet<Cell>| facts_from(&max.bounds, kept, 0.0);
    let cell_bare = bare(&per_cell.cells);
    let closed_kept: HashSet<Cell> = closed_type_ids
        .iter()
        .flat_map(|t| cells_of[t].iter().copied())
        .collect();
    let closed_delivered = delivered(&closed_kept);
    let closed_bare = bare(&closed_kept);

    println!(
        "  cells {}   target {}   eligible types {}{}",
        grid.cells.len(),
        target.len(),
        eligible.len(),
        if enumerated { "" } else { "  (NOT ENUMERATED)" }
    );
    println!(
        "  per-cell    cells {:>3}   shows {:>5} of {:<5}  arcs {:>6}",
        per_cell.cells.len(),
        cell_covered,
        target.len(),
        per_cell.arcs
    );
    println!(
        "  type-closed cells {:>3}   shows {:>5} of {:<5}  arcs {:>6}   types [{}]",
        closed_cells,
        closed_covered,
        target.len(),
        closed_arcs,
        closed_types.join(", ")
    );
    println!(
        "  facts       per-cell    orderings {:>5}  roles {:>5}   of {} / {}",
        cell_delivered.ordering.len(),
        cell_delivered.role.len(),
        full_facts.ordering.len(),
        full_facts.role.len()
    );
    println!(
        "              type-closed orderings {:>5}  roles {:>5}",
        closed_delivered.ordering.len(),
        closed_delivered.role.len()
    );
    println!(
        "  drawn only  per-cell {:>5}   type-closed {:>5}   (chained: {} vs {})",
        cell_drawn, closed_drawn, cell_covered, closed_covered
    );
    println!(
        "  no map credit  per-cell orderings {:>5}  roles {:>5}   type-closed {:>5} / {:>5}",
        cell_bare.ordering.len(),
        cell_bare.role.len(),
        closed_bare.ordering.len(),
        closed_bare.role.len()
    );
    println!(
        "  the whole-type restriction loses: {} shown, {} drawn, {} ordering facts, {} role facts, {} bare roles",
        cell_covered.saturating_sub(closed_covered),
        cell_drawn.saturating_sub(closed_drawn),
        cell_delivered.ordering.len().saturating_sub(closed_delivered.ordering.len()),
        cell_delivered.role.len().saturating_sub(closed_delivered.role.len()),
        cell_bare.role.len().saturating_sub(closed_bare.role.len())
    );

    LogReport {
        log: stem,
        recorded_cells: grid.cells.len(),
        target_pairs: target.len(),
        types_eligible: eligible.len(),
        enumerated,
        cell_cells: per_cell.cells.len(),
        cell_covered,
        cell_arcs: per_cell.arcs,
        cell_orderings: cell_delivered.ordering.len(),
        cell_roles: cell_delivered.role.len(),
        cell_drawn,
        cell_bare_orderings: cell_bare.ordering.len(),
        cell_bare_roles: cell_bare.role.len(),
        closed_types,
        closed_cells,
        closed_covered,
        closed_arcs,
        closed_orderings: closed_delivered.ordering.len(),
        closed_roles: closed_delivered.role.len(),
        closed_drawn,
        closed_bare_orderings: closed_bare.ordering.len(),
        closed_bare_roles: closed_bare.role.len(),
        full_orderings: full_facts.ordering.len(),
        full_roles: full_facts.role.len(),
    }
}

fn main() {
    let paths = corpus::logs_or_args();

    let reports: Vec<LogReport> = paths.iter().map(|p| run(p)).collect();
    let losing = reports.iter().filter(|r| r.lost() > 0).count();
    let total_lost: usize = reports.iter().map(|r| r.lost()).sum();
    let losing_roles = reports.iter().filter(|r| r.roles_lost() > 0).count();
    let total_roles_lost: usize = reports.iter().map(|r| r.roles_lost()).sum();

    println!("\n{}\ncorpus\n{}", "=".repeat(72), "=".repeat(72));
    println!(
        "  whole-type restriction loses orderings on {} of {} logs, {} orderings in total",
        losing,
        reports.len(),
        total_lost
    );
    println!(
        "  whole-type restriction loses role facts on {} of {} logs, {} role facts in total",
        losing_roles,
        reports.len(),
        total_roles_lost
    );

    let logs: Vec<serde_json::Value> = reports
        .iter()
        .map(|r| {
            serde_json::json!({
                "log": r.log,
                "recorded_cells": r.recorded_cells,
                "target_pairs": r.target_pairs,
                "types_eligible": r.types_eligible,
                "enumerated_exhaustively": r.enumerated,
                "facts_total": {"orderings": r.full_orderings, "roles": r.full_roles},
                "per_cell": {
                    "cells": r.cell_cells,
                    "orderings_shown": r.cell_covered,
                    "arcs_over_max": r.cell_arcs,
                    "facts_delivered": {"orderings": r.cell_orderings, "roles": r.cell_roles},
                    "orderings_drawn": r.cell_drawn,
                    "facts_no_map_credit": {"orderings": r.cell_bare_orderings, "roles": r.cell_bare_roles},
                },
                "type_closed": {
                    "cells": r.closed_cells,
                    "orderings_shown": r.closed_covered,
                    "arcs_over_max": r.closed_arcs,
                    "types_kept": r.closed_types,
                    "facts_delivered": {"orderings": r.closed_orderings, "roles": r.closed_roles},
                    "orderings_drawn": r.closed_drawn,
                    "facts_no_map_credit": {"orderings": r.closed_bare_orderings, "roles": r.closed_bare_roles},
                },
                "orderings_lost": r.lost(),
                "role_facts_lost": r.roles_lost(),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "logs": logs,
        "corpus_totals": {
            "logs": reports.len(),
            "logs_losing_orderings": losing,
            "orderings_lost": total_lost,
            "logs_losing_role_facts": losing_roles,
            "role_facts_lost": total_roles_lost,
        },
    });

    let stats = corpus::stats_dir();
    std::fs::create_dir_all(&stats).expect("create stats dir");
    let out = stats.join("hand_deletion.json");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).expect("serialise"))
        .expect("write report");
    println!("\nstats written to {}", out.display());
}
