//! What the determination threshold buys, and what it costs, per log across its range.
//!
//! `theta` is the share of an activity's events at which a route must reproduce the target
//! objects for the cell to count as determined. It is the schema layer's only threshold: a
//! map carries no admission bar of its own, so whether a relationship is good enough is asked
//! once, about the cell it is meant to reconstruct, rather than twice.
//!
//! The two sides of the trade are reported together, because either alone is misleading:
//!
//! - the OCPN and OC-DFG a discovery run finds on the flow projection;
//! - **what it costs**, the event-to-object tuples that do not survive a round trip. At
//!   `theta = 1` this is zero by construction and the reduction is lossless. Below 1 a cell
//!   counts as determined while some of its participations cannot be recomputed, and this is
//!   the count of them.
//!
//! Coverage of the behavioural requirement is *not* part of the trade: it is enforced by the
//! abstraction and does not move with `theta`. It is reported per row so that stays visible.
//!
//! Usage: `cargo run --release --example determination_sweep -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{collections::HashSet, path::PathBuf};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, annotate, assign, fact_repair, facts_from, flow_projection, novel_cells,
        novelty, reflow_layer_with, tag, ActivityIndexing, Bounds, Cell, CellGrid,
        ExpansionDirection, LogAbstraction, Pair, Saturation, SchemaClosure, SearchInput,
        StructuralSchema, TraceVariants, DEFAULT_THETA,
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

/// Matches every other arc figure in the evaluation.
const IMF_THRESHOLD: f64 = 0.2;

const THETAS: &[f64] = &[1.0, 0.99, 0.95, 0.9, 0.8, 0.5];

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


fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

fn main() {
    let paths = corpus::logs_or_args();
    let out = corpus::stats_dir().join("determination_sweep.json");

    let mut reports = Vec::new();
    for path in &paths {
        let stem = log_stem(path);
        println!("\n=== {stem} ===");
        let Ok(ocel) = SlimLinkedOCEL::import_from_path(&PathBuf::from(path)) else {
            println!("  skipped");
            continue;
        };
        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        println!(
            "  {} recorded maps, {} derived maps",
            schema.recorded.len(),
            schema.derived.len()
        );
        println!(
            "     {:>6}{:>8}{:>8}{:>8}{:>10}{:>12}{:>12}",
            "theta", "flow", "implied", "OCPN", "OC-DFG", "coverage", "residuals"
        );

        let mut rows = Vec::new();
        for theta in THETAS {
            let grid = CellGrid::build_with_theta(&ocel, &schema, &closure, *theta);
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
            let _ = novel_cells(&novelty(&ocel, &acts, &bounds, &grid.cells, &max));
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
            let mut layer = reflow_layer_with(&input, &lens);
            if !layer.ran {
                println!("     {theta:>6.2}  search did not run");
                continue;
            }
            let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
            fact_repair(&grid, &full_facts, &mut layer.cells);

            let reduced_log = flow_projection(&tag(&ocel, &schema, &acts, &layer.cells, &[])).into_owned();
            let opts = ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(IMF_THRESHOLD));
            let ocpn = ocpn_arcs(&discover_ocpn(&reduced_log, opts));
            let dfg = ocdfg_arcs(&discover_dfg_from_ocel(&reduced_log));

            let assignment = assign(&grid, &layer.cells, &Default::default());
            let annotation = annotate(
                &ocel,
                &schema,
                &closure,
                &grid,
                &layer.cells,
                &HashSet::new(),
            );
            // What a reader of the flow projection alone gives up: objects an implied cell's
            // route does not put back. Zero at theta = 1.
            let lost: usize = annotation
                .cells
                .iter()
                .filter_map(|c| c.determined_by.as_ref())
                .map(|d| d.residuals.len())
                .sum();

            if std::env::var("ROUTES").is_ok() {
                for m in &annotation.cells {
                    if let Some(d) = &m.determined_by {
                        println!(
                            "       {} / {}  <- {}  residuals {}",
                            m.activity,
                            m.object_type,
                            d.source_type,
                            d.residuals.len()
                        );
                    }
                }
            }
            if std::env::var("CELLS").is_ok() {
                let name = |c: &Cell| {
                    format!("{} / {}", grid.activities[c.0], schema.types[c.1])
                };
                let mut inv: Vec<String> = assignment.involvement.iter().map(name).collect();
                let mut abs: Vec<String> = assignment.implied.iter().map(name).collect();
                inv.sort();
                abs.sort();
                println!("       theta {theta}: involved {inv:?}");
                println!("       theta {theta}: implied  {abs:?}");
            }
            println!(
                "     {:>6.2}{:>8}{:>8}{:>8}{:>10}{:>7}/{:<4}{:>12}",
                theta,
                layer.cells.len(),
                assignment.implied.len(),
                ocpn,
                dfg,
                layer.coverage.chained,
                layer.coverage.target,
                lost,
            );
            rows.push(serde_json::json!({
                "theta": theta,
                "flow_cells": layer.cells.len(),
                "implied_cells": assignment.implied.len(),
                "ocpn_arcs": ocpn,
                "ocdfg_arcs": dfg,
                "covered": layer.coverage.chained,
                "target": layer.coverage.target,
                "residual_objects": lost,
            }));
        }
        reports.push(serde_json::json!({"log": stem, "path": corpus::label(&path), "thetas": rows}));
        let json = serde_json::json!({"logs": reports});
        let _ = std::fs::write(&out, serde_json::to_string_pretty(&json).unwrap());
    }
    println!("\nwrote {}", out.display());
}
