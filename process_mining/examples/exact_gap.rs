//! How far the constructions are from the minimum, under each objective.
//!
//! The searches are heuristics and nothing so far says how good. This runs the exact
//! enumeration beside them on the corpus and reports the gap, per log and per objective.
//!
//! **Why an exact answer is affordable here.** Both the objective and the coverage
//! constraint decompose per object type: a coloured arc names its own type, and
//! `Abstraction::asserts` reads only its own type's cells. So a flow layer is a choice of one
//! row per type, the objective is the sum of what those rows cost, and the only coupling is
//! that the union of what they assert has to cover the requirement once chained. The corpus
//! has between 18 and 40 cells over 3 to 12 types, which is a few thousand rows per type
//! before the dominance prune and far fewer after it.
//!
//! **What the gap means.** It is measured against the same requirement the constructions
//! face, so a construction at gap zero is optimal and not merely close. A gap is reported in
//! the objective's own primary quantity; the other two are printed beside it because a layer
//! that ties on arcs can still differ in cells and participations.
//!
//! Usage: `cargo run --release --example exact_gap -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{collections::HashSet, path::PathBuf, time::Instant};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, exact_cells_per_type, exact_with_objective, fact_repair, facts_from,
        flow_projection, greedy_with_objective, handoff_with_objective, tag, ActivityIndexing,
        Bounds, CellGrid, ExpansionDirection, FlowLayer, LogAbstraction, Objective, Pair,
        Saturation, SchemaClosure, SearchInput, StructuralSchema, TraceVariants, DEFAULT_THETA,
    },
    core::{
        event_data::object_centric::linked_ocel::SlimLinkedOCEL,
        process_models::object_centric::{
            ocdfg::{discover_dfg_from_ocel, OCDirectlyFollowsGraph},
            ocpn::ObjectCentricPetriNet,
        },
    },
    discovery::{
        case_centric::inductive_miner::InductiveMinerOptions,
        object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions},
    },
    Importable,
};

/// Matches `examples/cross_instantiation.rs`, so the model sizes here are Table 1's.
const IMF_THRESHOLD: f64 = 0.2;

/// Collapse silent structure that constrains nothing, then count arcs. Same convention as
/// every other arc figure in the evaluation.
fn ocpn_arcs(net: &ObjectCentricPetriNet) -> usize {
    let mut net = net.clone();
    for component in net.nets.values_mut() {
        component.simplify_silent();
    }
    net.nets.values().map(|n| n.arcs.len()).sum()
}

fn ocdfg_arcs(g: &OCDirectlyFollowsGraph) -> usize {
    g.object_type_to_dfg.values().map(|d| d.directly_follows_relations.len()).sum()
}

const OBJECTIVES: &[Objective] = &[Objective::Arcs, Objective::Cells, Objective::Participations];

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

/// The objective's own primary quantity, for the gap.
fn primary(obj: Objective, f: &FlowLayer) -> usize {
    obj.key(f).0
}

struct Run {
    strategy: &'static str,
    layer: FlowLayer,
    seconds: f64,
    /// Arcs of the models actually discovered from the log this layer reduces to, so the
    /// search objective can be told apart from what a reader of the model sees.
    ocpn: usize,
    ocdfg: usize,
}

fn main() {
    let paths = corpus::logs_or_args();

    let mut reports = Vec::new();
    for path in &paths {
        let stem = log_stem(path);
        println!("\n=== {stem} ===");
        let ocel = match SlimLinkedOCEL::import_from_path(&PathBuf::from(path)) {
            Ok(o) => o,
            Err(e) => {
                println!("  skipped: {e}");
                continue;
            }
        };

        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let theta: f64 = std::env::var("DET_THETA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let grid = CellGrid::build_with_theta(&ocel, &schema, &closure, theta);
        let acts = ActivityIndexing::build(&ocel, &grid);
        let bounds = Bounds::build(&ocel, &schema, &acts);
        let (routes, _) = agreed_routes(&schema);
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
        // `cross_instantiation.rs`, which produces Table 1, searches over the recorded cells
        // alone; expansion is a separate run there. Allowing expansion cells here would price
        // a different question and the two numbers could not be set beside each other.
        let allowed = grid.cells.clone();
        let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
        target.sort_unstable();

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
        let lens = LogAbstraction::over(&max.bounds, &max.cells);
        let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);

        let types: HashSet<usize> = allowed.iter().map(|(_, t)| *t).collect();
        println!(
            "  {} activities, {} types, {} cells allowed, {} required assertions",
            grid.activities.len(),
            types.len(),
            allowed.len(),
            target.len()
        );
        println!("  allowed cells per type: {:?}", exact_cells_per_type(&input));

        let mut per_objective = Vec::new();
        for obj in OBJECTIVES {
            let timed = |f: &dyn Fn() -> FlowLayer| {
                let t = Instant::now();
                let layer = f();
                (layer, t.elapsed().as_secs_f64())
            };
            let (g, gs) = timed(&|| greedy_with_objective(&input, &lens, *obj));
            let (h, hs) = timed(&|| handoff_with_objective(&input, &lens, *obj));
            let (e, es) = timed(&|| exact_with_objective(&input, &lens, *obj));
            let runs: Vec<Run> = [("greedy", g, gs), ("handoff", h, hs), ("exact", e, es)]
                .into_iter()
                .map(|(strategy, mut layer, seconds)| {
                    // The same `fact_repair` pass `cross_instantiation` applies before it measures, so
                    // the model sizes here are the ones Table 1 reports.
                    if layer.ran {
                        fact_repair(&grid, &full_facts, &mut layer.cells);
                    }
                    let (ocpn, ocdfg) = if layer.ran {
                        let reduced = flow_projection(&tag(&ocel, &schema, &acts, &layer.cells, &[])).into_owned();
                        let opts = ObjectCentricDiscoveryOptions::new(
                            InductiveMinerOptions::imf(IMF_THRESHOLD),
                        );
                        (
                            ocpn_arcs(&discover_ocpn(&reduced, opts)),
                            ocdfg_arcs(&discover_dfg_from_ocel(&reduced)),
                        )
                    } else {
                        (0, 0)
                    };
                    Run { strategy, layer, seconds, ocpn, ocdfg }
                })
                .collect();

            println!("  -- minimising {} --", obj.label());
            println!(
                "     {:<9}{:>7}{:>7}{:>7}{:>8}{:>13}{:>6}{:>9}",
                "strategy", "cells", "obj", "OCPN", "OC-DFG", "coverage", "gap", "seconds"
            );
            let opt = runs
                .iter()
                .find(|r| r.strategy == "exact" && r.layer.ran)
                .map(|r| primary(*obj, &r.layer));
            for r in &runs {
                if !r.layer.ran {
                    println!("     {:<9}{:>7}", r.strategy, "did not run");
                    continue;
                }
                let l = &r.layer;
                let gap = opt.map(|o| primary(*obj, l).saturating_sub(o));
                println!(
                    "     {:<9}{:>7}{:>7}{:>7}{:>8}{:>8}/{:<4}{:>6}{:>9.3}",
                    r.strategy,
                    l.cells.len(),
                    primary(*obj, l),
                    r.ocpn,
                    r.ocdfg,
                    l.coverage.chained,
                    l.coverage.target,
                    gap.map(|g| g.to_string()).unwrap_or_else(|| "-".into()),
                    r.seconds,
                );
            }

            per_objective.push(serde_json::json!({
                "objective": obj.label(),
                "optimum": opt,
                "runs": runs.iter().map(|r| serde_json::json!({
                    "strategy": r.strategy,
                    "ran": r.layer.ran,
                    "cells": r.layer.cells.len(),
                    "arcs": r.layer.arcs,
                    "participations": r.layer.participations,
                    "ocpn_arcs": r.ocpn,
                    "ocdfg_arcs": r.ocdfg,
                    "covered": r.layer.coverage.chained,
                    "target": r.layer.coverage.target,
                    "complete": r.layer.coverage.complete(),
                    "gap": opt.map(|o| primary(*obj, &r.layer).saturating_sub(o)),
                    "seconds": r.seconds,
                })).collect::<Vec<_>>(),
            }));
        }

        reports.push(serde_json::json!({
            "log": stem,
            "path": path,
            "activities": grid.activities.len(),
            "types": types.len(),
            "cells_allowed": allowed.len(),
            "required": target.len(),
            "objectives": per_objective,
        }));
        report(&reports, false);
    }

    report(&reports, true);
}

/// Write what has been measured so far.
///
/// Called after every log and not only at the end. The exact enumeration is minutes on the
/// logs it can afford and does not terminate on the ones it cannot, so a single write at the
/// end means a run that stalls on its last log saves nothing at all -- which is how the first
/// corpus run left `exact_gap.json` holding one log while the paper quoted nine.
fn report(reports: &[serde_json::Value], final_call: bool) {
    let json = serde_json::json!({
        "complete": final_call,
        "logs": reports,
    });
    let out = corpus::stats_dir().join("exact_gap.json");
    match std::fs::write(&out, serde_json::to_string_pretty(&json).unwrap()) {
        Ok(()) if final_call => println!("\nwrote {}", out.display()),
        Ok(()) => {}
        Err(e) => println!("  could not write {}: {e}", out.display()),
    }
}
