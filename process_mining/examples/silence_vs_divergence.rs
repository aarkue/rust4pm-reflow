//! Neutrality against divergence, per recorded cell across the corpus.
//!
//! `silent` is the paper's neutral cell: the type asserts no ordering pair involving the
//! activity, read off the recorded bounds exactly as Def. Orders does. `divergent` is the
//! per-cell divergence of `type_deletion.rs`: some object of the type attends two events
//! of the activity whose other objects differ. The introduction quotes the two
//! off-diagonal counts (cells divergent but not silent, and silent but not divergent).
//!
//! Usage: `cargo run --release --example silence_vs_divergence -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    path::PathBuf,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        asserted_by_type, ActivityIndexing, Bounds, CellGrid, SchemaClosure, StructuralSchema,
    },
    core::event_data::object_centric::linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL},
    Importable,
};

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

/// Divergent cells by name. One bit per object: a cell stops being tracked once it is
/// known to diverge, so BPIC2017's 433k events fit in memory.
fn divergent_cells(l: &SlimLinkedOCEL) -> HashSet<(String, String)> {
    let mut seen: HashMap<(String, String), HashMap<String, u64>> = HashMap::new();
    let mut diverges: HashSet<(String, String)> = HashSet::new();
    for e in l.get_all_evs() {
        let act = e.get_ev_type(l).clone();
        let mut by_type: HashMap<&str, Vec<&str>> = HashMap::new();
        for o in e.get_e2o(l) {
            by_type.entry(o.get_ob_type(l).as_str()).or_default().push(o.get_ob(l).id.as_str());
        }
        for (t, mine) in &by_type {
            let cell = (act.clone(), (*t).to_string());
            if diverges.contains(&cell) {
                continue;
            }
            let mut others: Vec<&str> = by_type
                .iter()
                .filter(|(tt, _)| *tt != t)
                .flat_map(|(_, v)| v.iter().copied())
                .collect();
            others.sort_unstable();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            others.hash(&mut h);
            let sig = h.finish();
            let per_obj = seen.entry(cell.clone()).or_default();
            for ob in mine {
                match per_obj.get(*ob) {
                    None => {
                        per_obj.insert((*ob).to_string(), sig);
                    }
                    Some(prev) if *prev != sig => {
                        diverges.insert(cell.clone());
                        break;
                    }
                    Some(_) => {}
                }
            }
        }
    }
    diverges
}

fn main() {
    let paths = corpus::logs_or_args();

    let mut per_log = Vec::new();
    let (mut t_cells, mut t_both, mut t_sil, mut t_div, mut t_nei) = (0, 0, 0, 0, 0);
    for path in &paths {
        let stem = log_stem(path);
        let Ok(ocel) = SlimLinkedOCEL::import_from_path(&PathBuf::from(path)) else {
            println!("{stem}: skipped");
            continue;
        };
        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let grid = CellGrid::build(&ocel, &schema, &closure);
        let acts = ActivityIndexing::build(&ocel, &grid);
        let bounds = Bounds::build(&ocel, &schema, &acts);
        let asserted = asserted_by_type(&bounds, &grid.cells);
        let divergent = divergent_cells(&ocel);

        let (mut both, mut sil_only, mut div_only, mut neither) = (0, 0, 0, 0);
        let mut sil_cells = Vec::new();
        let mut div_cells = Vec::new();
        for &(a, t) in &grid.cells {
            let silent = !asserted[t].iter().any(|&(x, y)| x == a || y == a);
            let name = (grid.activities[a].clone(), schema.types[t].clone());
            let diverges = divergent.contains(&name);
            match (silent, diverges) {
                (true, true) => both += 1,
                (true, false) => {
                    sil_only += 1;
                    sil_cells.push(name);
                }
                (false, true) => {
                    div_only += 1;
                    div_cells.push(name);
                }
                (false, false) => neither += 1,
            }
        }
        sil_cells.sort();
        div_cells.sort();
        println!(
            "{stem}: {} cells, both {both}, silent_only {sil_only}, divergent_only {div_only}, neither {neither}",
            grid.cells.len()
        );
        t_cells += grid.cells.len();
        t_both += both;
        t_sil += sil_only;
        t_div += div_only;
        t_nei += neither;
        per_log.push(serde_json::json!({
            "log": format!("{stem}.xml"),
            "cells": grid.cells.len(),
            "both": both,
            "silent_only": sil_only,
            "divergent_only": div_only,
            "neither": neither,
            "silent_only_cells": sil_cells,
            "divergent_only_cells": div_cells,
        }));
    }
    let doc = serde_json::json!({
        "per_log": per_log,
        "total": {
            "cells": t_cells,
            "both": t_both,
            "silent_only": t_sil,
            "divergent_only": t_div,
            "neither": t_nei,
        },
    });
    let out = corpus::stats_dir().join("silence_vs_divergence.json");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("write stats");
    println!(
        "total: {t_cells} cells, both {t_both}, silent_only {t_sil}, divergent_only {t_div}, neither {t_nei}"
    );
    println!("wrote {}", out.display());
}
