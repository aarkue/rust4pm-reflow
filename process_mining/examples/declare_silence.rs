//! The OC-DECLARE behavioural instantiation: per (activity, object type) cell, whether the
//! type's discovered OC-DECLARE constraints order the activity, compared against the log-based
//! instantiation ([`asserted_of_type`]).
//!
//! Mirrors `model_silence.rs`'s report shape and corpus, one instantiation over.
//!
//! Usage: `cargo run --release --example declare_silence -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the directory
//! `declare_silence.json` is written to.

mod corpus;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use process_mining::{
    analysis::object_centric::schema_reduction::{
        asserted_of_type, declare_abstraction, ActivityIndex, ActivityIndexing, Bounds, CellGrid,
        DeclareTypeModel, ObjectTypeIndex, SchemaClosure, StructuralSchema,
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
    templates: Vec<String>,
}

#[derive(Serialize)]
struct PairReport {
    pair: (String, String),
    templates: Vec<String>,
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
    /// Wall-clock time of `declare_abstraction`, summed over every recorded type: the runtime
    /// column of the cross-instantiation table.
    declare_seconds: f64,
    /// The same, broken down per type, sorted by type name for a stable diff.
    declare_seconds_by_type: Vec<(String, f64)>,
}

/// Activities of `activities` that no pair of `pairs` mentions.
fn silent_of(
    activities: &BTreeSet<ActivityIndex>,
    pairs: &HashSet<(ActivityIndex, ActivityIndex)>,
) -> BTreeSet<ActivityIndex> {
    let ordered: BTreeSet<ActivityIndex> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    activities.difference(&ordered).copied().collect()
}

fn pair_report(
    model: &DeclareTypeModel,
    acts: &ActivityIndexing,
    x: ActivityIndex,
    y: ActivityIndex,
) -> PairReport {
    PairReport {
        pair: (acts.activities[x].clone(), acts.activities[y].clone()),
        templates: model
            .pair_templates(x, y)
            .unwrap_or_default()
            .into_iter()
            .map(str::to_string)
            .collect(),
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

    let total_start = Instant::now();
    let models = declare_abstraction(&locel, &schema, &acts, &grid.cells);
    let declare_seconds = total_start.elapsed().as_secs_f64();

    let by_type: HashMap<ObjectTypeIndex, &DeclareTypeModel> =
        models.iter().map(|m| (m.object_type, m)).collect();

    let mut type_order: Vec<ObjectTypeIndex> = by_type.keys().copied().collect();
    type_order.sort_by_key(|&t| schema.types[t].clone());

    let mut cells = Vec::new();
    let mut disagreements = Vec::new();
    let mut declare_seconds_by_type: Vec<(String, f64)> = models
        .iter()
        .map(|m| (schema.types[m.object_type].clone(), m.discovery_seconds))
        .collect();
    declare_seconds_by_type.sort_by(|a, b| a.0.cmp(&b.0));

    for t in type_order {
        let model = by_type[&t];
        let lpairs = asserted_of_type(&bounds, &grid.cells, t);
        let declare_neutral = model.neutral();
        let log_neutral = silent_of(&model.activities, &lpairs);

        let mut acts_sorted: Vec<ActivityIndex> = model.activities.iter().copied().collect();
        acts_sorted.sort_by_key(|&a| acts.activities[a].clone());

        for a in acts_sorted {
            let neutral = declare_neutral.contains(&a);
            let mut templates: BTreeSet<String> = BTreeSet::new();
            for &(x, y) in &model.ordered {
                if x == a || y == a {
                    templates.extend(model.pair_templates(x, y).unwrap_or_default().into_iter().map(str::to_string));
                }
            }
            cells.push(CellReport {
                object_type: schema.types[t].clone(),
                activity: acts.activities[a].clone(),
                verdict: if neutral { "neutral" } else { "asserting" },
                templates: templates.into_iter().collect(),
            });

            if neutral != log_neutral.contains(&a) {
                let direction = if neutral {
                    "log orders, OC-DECLARE does not"
                } else {
                    "OC-DECLARE orders, log does not"
                };
                let mut pairs: Vec<(ActivityIndex, ActivityIndex)> = if neutral {
                    lpairs
                        .iter()
                        .filter(|&&(x, y)| x == a || y == a)
                        .copied()
                        .collect()
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
                    pairs: pairs
                        .into_iter()
                        .map(|(x, y)| pair_report(model, &acts, x, y))
                        .collect(),
                });
            }
        }
    }

    let stem = log_stem(path);
    let n_neutral = cells.iter().filter(|c| c.verdict == "neutral").count();
    println!(
        "{stem:<28} cells {:>4}  neutral {:>4}  disagreements {:>3}  declare {:>7.3}s",
        cells.len(),
        n_neutral,
        disagreements.len(),
        declare_seconds
    );
    for d in &disagreements {
        println!("    [{}] {}: {}", d.object_type, d.activity, d.direction);
        for p in &d.pairs {
            println!(
                "        {} -> {}  via {}",
                p.pair.0,
                p.pair.1,
                if p.templates.is_empty() {
                    "log-only".to_string()
                } else {
                    p.templates.join(", ")
                }
            );
        }
    }

    LogReport {
        log: stem,
        cells,
        disagreements,
        declare_seconds,
        declare_seconds_by_type,
    }
}

fn main() {
    let paths = corpus::logs_or_args();

    let reports: Vec<LogReport> = paths.iter().map(|p| run(p)).collect();

    let total_cells: usize = reports.iter().map(|r| r.cells.len()).sum();
    let total_neutral: usize = reports
        .iter()
        .flat_map(|r| &r.cells)
        .filter(|c| c.verdict == "neutral")
        .count();
    let total_dis: usize = reports.iter().map(|r| r.disagreements.len()).sum();
    let total_seconds: f64 = reports.iter().map(|r| r.declare_seconds).sum();
    println!(
        "\ncorpus: {total_cells} cells, {total_neutral} neutral, {total_dis} disagreements, {total_seconds:.3}s discovery"
    );

    let doc = serde_json::json!({ "logs": reports });
    let json = serde_json::to_string_pretty(&doc).expect("serialise report");
    let out_path = corpus::stats_dir().join("declare_silence.json");
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
