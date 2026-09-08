//! What the expansion threshold buys, per log, across its whole range.
//!
//! `theta` is the share of the tuples a candidate cell would write whose event falls inside
//! the recorded lifetime of the object receiving them. An admitted cell is written whole, so
//! `theta = 1` is the only setting under which expansion writes no participation outside its
//! own object's lifetime, and `theta = 0` admits every cell the schema determines.
//!
//! The question this answers is whether the parameter is load-bearing. If the admitted cells
//! and the orderings they add are the same at `1` as at `0.5`, the threshold can be fixed at
//! `1` and stops being a parameter at all.
//!
//! Reported per log and per theta: cells the saturation admits beyond the recorded ones, the
//! orderings the saturated target holds that the recorded log does not assert, and the tuples
//! written outside their own object's lifetime.
//!
//! Usage: `cargo run --release --example theta_sweep -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{collections::HashSet, path::PathBuf};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, ActivityIndexing, Bounds, CellGrid, ExpansionDirection,
        Pair, Saturation, SchemaClosure, StructuralSchema,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};

const THETAS: &[f64] = &[0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0];

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

fn main() {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    println!(
        "{:<20} {:>6} {:>14} {:>16} {:>18}",
        "log", "theta", "expanded_cells", "added_orderings", "outside_lifetime"
    );

    let paths = corpus::logs_or_args();
    for path in &paths {
        let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
        let locel = SlimLinkedOCEL::from_ocel(ocel);
        let schema = StructuralSchema::discover(&locel);
        let closure = SchemaClosure::build(&locel, &schema);
        let grid = CellGrid::build(&locel, &schema, &closure);
        let acts = ActivityIndexing::build(&locel, &grid);
        let bounds = Bounds::build(&locel, &schema, &acts);
        let routes = agreed_routes(&schema).0;

        let recorded_target: HashSet<Pair> =
            asserted_by_type(&bounds, &grid.cells).into_iter().flatten().collect();

        let stem = log_stem(path);
        for &theta in THETAS {
            let max = Saturation::build(
                &locel,
                &schema,
                &grid,
                &acts,
                &routes,
                &bounds,
                theta,
                ExpansionDirection::default(),
            );
            let expanded_cells = max.cells.difference(&grid.cells).count();
            let saturated: HashSet<Pair> = max.target_pairs().into_iter().collect();
            let added = saturated.difference(&recorded_target).count();

            // Tuples expansion writes whose event falls outside the receiving object's own
            // recorded lifetime. Zero at theta = 1 by construction, reported so the cost of
            // every lower setting is visible next to its result.
            let outside = max.candidates.iter().filter(|c| c.admissible).map(|c| c.extrapolated()).sum::<usize>();

            println!(
                "{:<20} {:>6.2} {:>14} {:>16} {:>18}",
                stem, theta, expanded_cells, added, outside
            );
            let mut cell_names: Vec<String> = max
                .cells
                .difference(&grid.cells)
                .map(|(a, t)| format!("{}|{}", grid.activities[*a], schema.types[*t]))
                .collect();
            cell_names.sort();
            let mut pair_names: Vec<String> = saturated
                .difference(&recorded_target)
                .map(|(x, y)| format!("{} < {}", grid.activities[*x], grid.activities[*y]))
                .collect();
            pair_names.sort();
            rows.push(serde_json::json!({
                "log": stem,
                "theta": theta,
                "cells": cell_names,
                "pairs": pair_names,
                "expanded_cells": expanded_cells,
                "added_orderings": added,
                "tuples_outside_lifetime": outside,
            }));
        }
    }

    let doc = serde_json::json!({ "rows": rows });
    let out_path = corpus::stats_dir().join("theta_sweep.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).expect("serialize"))
        .expect("write stats json");
    println!("\nstats written to {}", out_path.display());
}
