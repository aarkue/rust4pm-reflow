//! The model-based behavioural instantiation: per (activity, object type) cell, whether the
//! type's mined process tree orders the activity, compared against the log-based instantiation
//! ([`asserted_of_type`]).
//!
//! Ports `model_silence_constructs.py` from the ICPM 2027 schema-reduction paper's Python
//! reference, which mined with pm4py and turned out hash-seed nondeterministic. This mines with
//! rust4pm's own inductive miner instead, and records the deciding construct of every verdict the
//! same way: a cut (`xor_cut`, `sequence_cut`, `concurrent_cut`, `loop_cut`, ...) or a fall
//! through (`activity_once_per_trace`, `strict_tau_loop`, `flower`, ...).
//!
//! Usage: `cargo run --release --example model_silence -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the directory
//! `model_silence_constructs_rust.json` is written to.

mod corpus;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

use process_mining::{
    analysis::object_centric::schema_reduction::{
        asserted_of_type, model_abstraction, ActivityIndex, ActivityIndexing, Bounds, CellGrid,
        ObjectTypeIndex, SchemaClosure, StructuralSchema, TypeModel, MODEL_NOISE_THRESHOLD,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};
use serde::Serialize;

#[derive(Serialize)]
struct CellReport {
    #[serde(rename = "type")]
    object_type: String,
    activity: String,
    verdict: &'static str,
    constructs: Vec<String>,
}

#[derive(Serialize)]
struct PairReport {
    pair: (String, String),
    constructs: Vec<String>,
}

#[derive(Serialize)]
struct DisagreementReport {
    #[serde(rename = "type")]
    object_type: String,
    activity: String,
    direction: &'static str,
    pairs: Vec<PairReport>,
}

#[derive(Serialize)]
struct LogReport {
    log: String,
    cells: Vec<CellReport>,
    disagreements: Vec<DisagreementReport>,
}

/// Activities of `activities` that no pair of `pairs` mentions.
fn silent_of(
    activities: &BTreeSet<ActivityIndex>,
    pairs: &HashSet<(ActivityIndex, ActivityIndex)>,
) -> BTreeSet<ActivityIndex> {
    let ordered: BTreeSet<ActivityIndex> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    activities.difference(&ordered).copied().collect()
}

/// The constructs deciding one activity's cell, mirroring the pm4py reference's `cons`
/// computation: the union of the deciding constructs over every pair touching the activity, on
/// whichever side (asserting or silent) decided its verdict, with `single_activity` and
/// `dropped_by_noise_filter` as the two readings a pair-based lookup cannot produce.
fn cell_constructs(model: &TypeModel, a: ActivityIndex, silent: bool) -> Vec<String> {
    let mut constructs: BTreeSet<String> = BTreeSet::new();
    if !silent {
        for &(x, y) in &model.ordered {
            if x == a || y == a {
                constructs.extend(model.pair_labels(x, y));
            }
        }
    } else if model.activities.len() == 1 {
        constructs.insert("single_activity".to_string());
    } else if !model.tree_activities.contains(&a) {
        constructs.insert("dropped_by_noise_filter".to_string());
    } else {
        for &x in &model.activities {
            if x != a {
                constructs.extend(model.pair_labels(a, x));
            }
        }
    }
    constructs.into_iter().collect()
}

fn pair_report(model: &TypeModel, acts: &ActivityIndexing, x: ActivityIndex, y: ActivityIndex) -> PairReport {
    PairReport {
        pair: (acts.activities[x].clone(), acts.activities[y].clone()),
        constructs: model.pair_labels(x, y),
    }
}

fn run(path: &str) -> LogReport {
    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);
    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);
    let bounds = Bounds::build(&locel, &schema, &acts);

    let models = model_abstraction(&locel, &schema, &acts, &grid.cells, MODEL_NOISE_THRESHOLD);
    let by_type: HashMap<ObjectTypeIndex, &TypeModel> =
        models.iter().map(|m| (m.object_type, m)).collect();

    let mut type_order: Vec<ObjectTypeIndex> = by_type.keys().copied().collect();
    type_order.sort_by_key(|&t| schema.types[t].clone());

    let mut cells = Vec::new();
    let mut disagreements = Vec::new();

    for t in type_order {
        let model = by_type[&t];
        let lpairs = asserted_of_type(&bounds, &grid.cells, t);
        let model_silent = model.silent();
        let log_silent = silent_of(&model.activities, &lpairs);

        let mut acts_sorted: Vec<ActivityIndex> = model.activities.iter().copied().collect();
        acts_sorted.sort_by_key(|&a| acts.activities[a].clone());

        for a in acts_sorted {
            let silent = model_silent.contains(&a);
            cells.push(CellReport {
                object_type: schema.types[t].clone(),
                activity: acts.activities[a].clone(),
                verdict: if silent { "silent" } else { "asserting" },
                constructs: cell_constructs(model, a, silent),
            });

            if silent != log_silent.contains(&a) {
                let direction = if silent {
                    "log orders, tree does not"
                } else {
                    "tree orders, log does not"
                };
                let mut pairs: Vec<(ActivityIndex, ActivityIndex)> = if silent {
                    lpairs.iter().filter(|&&(x, y)| x == a || y == a).copied().collect()
                } else {
                    model
                        .ordered
                        .iter()
                        .filter(|p| !lpairs.contains(*p))
                        .filter(|&&(x, y)| x == a || y == a)
                        .copied()
                        .collect()
                };
                pairs.sort();
                disagreements.push(DisagreementReport {
                    object_type: schema.types[t].clone(),
                    activity: acts.activities[a].clone(),
                    direction,
                    pairs: pairs.into_iter().map(|(x, y)| pair_report(model, &acts, x, y)).collect(),
                });
            }
        }
    }

    let stem = log_stem(path);
    let n_silent = cells.iter().filter(|c| c.verdict == "silent").count();
    println!(
        "{stem:<28} cells {:>4}  silent {:>4}  disagreements {:>3}",
        cells.len(),
        n_silent,
        disagreements.len()
    );
    for d in &disagreements {
        println!(
            "    [{}] {}: {}",
            d.object_type, d.activity, d.direction
        );
        for p in &d.pairs {
            println!(
                "        {} -> {}  via {}",
                p.pair.0,
                p.pair.1,
                p.constructs.join(", ")
            );
        }
    }

    LogReport {
        log: stem,
        cells,
        disagreements,
    }
}

fn main() {
    let paths = corpus::logs_or_args();

    let reports: Vec<LogReport> = paths.iter().map(|p| run(p)).collect();

    let total_cells: usize = reports.iter().map(|r| r.cells.len()).sum();
    let total_silent: usize = reports
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| c.verdict == "silent")
        .count();
    let total_dis: usize = reports.iter().map(|r| r.disagreements.len()).sum();
    println!(
        "\ncorpus: {total_cells} cells, {total_silent} silent, {total_dis} disagreements"
    );

    let doc = serde_json::json!({ "logs": reports });
    let json = serde_json::to_string_pretty(&doc).expect("serialise report");
    let out_path = corpus::stats_dir().join("model_silence_constructs_rust.json");
    std::fs::write(&out_path, json).expect("write report");
    println!("written {}", out_path.display());
}

/// The log's name with its compression and format extensions stripped, so that
/// `bpic2017-no-W.xml.gz` and `bpic2017-no-W.xml` both key the same report files.
fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}
