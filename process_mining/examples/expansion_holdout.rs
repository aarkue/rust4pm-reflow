//! Hold-out ablation for the expansion direction: does it write back what it never saw?
//!
//! Counting the orderings a log gains after expansion proves nothing on its own. Expansion
//! writes participations, so of course the resulting log asserts more, and the count measures
//! our own construction rather than whether the construction is right. Model precision cannot
//! settle it either: against the expanded log the question is circular, against the recorded
//! log an expanded model necessarily permits more, and per-type flattening changes shape once
//! a type gains participations at activities where it had none, so the before and after are
//! not the same measurement.
//!
//! This asks the question the counts dodge, with truth known by construction. For each
//! recorded cell:
//!
//! 1. delete its participations, giving an ablated log,
//! 2. **rediscover the schema on the ablated log**, so nothing the deleted cell witnessed
//!    leaks back in,
//! 3. run the expansion candidate enumeration on it,
//! 4. compare what it would write against what was deleted.
//!
//! Rediscovery in step 2 is the whole point. Reusing the original schema would hand the
//! expansion a map the deleted cell helped witness, and the test would pass by leakage.
//!
//! Per cell the outcome is one of:
//!
//! | outcome | meaning |
//! |---|---|
//! | undetermined | the ablated schema derives no candidate for the cell, so expansion would not write it. A miss, not an error |
//! | inadmissible | a candidate exists but `theta` rejects it on lifetime grounds |
//! | exact | the written participations are exactly the deleted ones |
//! | partial | written and deleted overlap but differ, reported as precision and recall |
//!
//! Only `partial` with precision below one is expansion asserting something the log denies.
//! That is the number the expansion claim rests on.
//!
//! **Scope, stated rather than buried.** This ablates cells the extraction *did* record, while
//! expansion in the wild acts on cells it never recorded. A cell can be unrecorded for reasons
//! that also make it harder to derive, so this is evidence about the mechanism and not a
//! sample of the population expansion actually meets.
//!
//! Usage: `cargo run --release --example expansion_holdout -- <log> ...`
//! Env: `THETA` (default 0.5), `MAX_CELLS` to cap the ablations per log, `REFLOW_LOGS` for
//! the corpus directory, `REFLOW_STATS` for the output directory.

mod corpus;

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    path::PathBuf,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, expansion_candidates, flow_projection, tag, ActivityIndexing, Cell,
        CellGrid, ExpansionDirection, SchemaClosure, StructuralSchema, DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL},
    Importable, OCEL,
};

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

/// Participations of one cell, as the identifier pairs that survive a re-import.
///
/// Indices cannot be compared across two `SlimLinkedOCEL`s, since the ablated log is built
/// and linked afresh, so the comparison runs on the event and object ids the OCEL carries.
fn tuples_of(
    locel: &SlimLinkedOCEL,
    acts: &ActivityIndexing,
    schema: &StructuralSchema,
    cell: Cell,
) -> BTreeSet<(String, String)> {
    let (a, t) = cell;
    let mut out = BTreeSet::new();
    for e in locel.get_all_evs() {
        if acts.act_of[e.get_ev(locel).event_type] != a {
            continue;
        }
        for o in e.get_e2o(locel) {
            if schema.type_of[&o] == t {
                out.insert((e.get_ev(locel).id.clone(), o.get_ob(locel).id.clone()));
            }
        }
    }
    out
}

#[derive(Default)]
struct Tally {
    undetermined: usize,
    inadmissible: usize,
    exact: usize,
    partial: usize,
    /// Written tuples that were not there, summed over partial outcomes.
    false_written: usize,
    /// Deleted tuples the expansion did not write back, summed over partial outcomes.
    missed: usize,
    /// Deleted tuples in total, over cells that produced an admissible candidate.
    truth: usize,
    /// Cells where the ablation itself manufactured the map expansion then used.
    ///
    /// The recorded log may refuse a map into the cell's type because the cell's own
    /// participations contradict it. Deleting the cell deletes the contradiction, so the
    /// ablated schema accepts a map the full log rejects and expansion applies it. Those
    /// cells measure that exposure, not expansion's accuracy where a map already held, so
    /// the two populations are tallied apart.
    manufactured: usize,
    man_false: usize,
    man_missed: usize,
    man_truth: usize,
    /// Cells whose ablation orphaned every object the cell named.
    ///
    /// Removing a cell removes its participations, and where a type's objects take part in
    /// that activity and nowhere else, the ablated log holds no trace of them at all.
    /// Expansion cannot write back an object the log no longer mentions, so scoring the
    /// attempt would measure the ablation and not the method. Counted apart and excluded.
    orphaned: usize,
}

fn run(path: &str, theta: f64, cap: usize) -> (String, Tally, Vec<String>) {
    let stem = log_stem(path);
    println!("\n{}\n{stem}\n{}", "=".repeat(72), "=".repeat(72));

    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);
    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);

    let mut cells: Vec<Cell> = grid.cells.iter().copied().collect();
    cells.sort_unstable();
    let total = cells.len();
    if cells.len() > cap {
        cells.truncate(cap);
    }

    let mut tally = Tally::default();
    let mut notes = Vec::new();
    for &c in &cells {
        let (a, t) = c;
        let name = format!("{} / {}", grid.activities[a], schema.types[t]);
        let truth = tuples_of(&locel, &acts, &schema, c);
        if truth.is_empty() {
            continue;
        }

        // Ablate: the same log without this cell's participations.
        let kept: HashSet<Cell> = grid.cells.iter().copied().filter(|&x| x != c).collect();
        // The projection clones and strips the links, so event and object indices
        // survive. The schema and activity indexing below are rebuilt and theirs do not,
        // which is why the comparison runs on ids.
        let ablated = flow_projection(&tag(&locel, &schema, &acts, &kept, &[])).into_owned();

        // Rediscover everything. Nothing the deleted cell witnessed may leak back in.
        let schema2 = StructuralSchema::discover(&ablated);
        let closure2 = SchemaClosure::build(&ablated, &schema2);
        let grid2 = CellGrid::build(&ablated, &schema2, &closure2);
        let acts2 = ActivityIndexing::build(&ablated, &grid2);
        let routes2 = agreed_routes(&schema2).0;
        let cands = expansion_candidates(
            &ablated,
            &routes2,
            &schema2,
            &grid2,
            &acts2,
            theta,
            ExpansionDirection::default(),
        );

        // Find the candidate for this cell, matched by name since indices were rebuilt.
        let hit = cands.iter().find(|k| {
            grid2.activities.get(k.cell.0).map(String::as_str) == Some(grid.activities[a].as_str())
                && schema2.types.get(k.cell.1).map(String::as_str)
                    == Some(schema.types[t].as_str())
        });
        let Some(k) = hit else {
            tally.undetermined += 1;
            continue;
        };
        if !k.admissible {
            tally.inadmissible += 1;
            continue;
        }

        // Is the test even winnable? If ablation left none of the cell's own objects visible
        // anywhere, no derivation could name them and the comparison is void.
        let truth_objs: BTreeSet<&String> = truth.iter().map(|(_, o)| o).collect();
        let mut alive_objs: BTreeSet<String> = BTreeSet::new();
        for e in ablated.get_all_evs() {
            for o in e.get_e2o(&ablated) {
                let id = o.get_ob(&ablated).id.clone();
                if truth_objs.contains(&id) {
                    alive_objs.insert(id);
                }
            }
        }
        if alive_objs.is_empty() {
            tally.orphaned += 1;
            continue;
        }

        // Did the ablation invent the map? A source accepted into this type after ablation
        // that the recorded schema rejected is the contradiction having been deleted with
        // the cell.
        let accepted_into = |sc: &StructuralSchema, ty: &str| -> BTreeSet<String> {
            let Some(ti) = sc.types.iter().position(|x| x == ty) else {
                return BTreeSet::new();
            };
            sc.recorded
                .iter()
                .chain(sc.derived.iter())
                .filter(|m| m.target == ti)
                .map(|m| sc.types[m.source].clone())
                .collect()
        };
        let before = accepted_into(&schema, &schema.types[t]);
        let after = accepted_into(&schema2, &schema.types[t]);
        let manufactured = !after.difference(&before).collect::<Vec<_>>().is_empty();

        let written: BTreeSet<(String, String)> = k
            .tuples
            .iter()
            .map(|(e, o)| (e.get_ev(&ablated).id.clone(), o.get_ob(&ablated).id.clone()))
            .collect();
        let fp = written.difference(&truth).count();
        let fnn = truth.difference(&written).count();
        if manufactured {
            tally.manufactured += 1;
            tally.man_truth += truth.len();
            tally.man_false += fp;
            tally.man_missed += fnn;
        } else {
            tally.truth += truth.len();
            tally.false_written += fp;
            tally.missed += fnn;
        }
        if written == truth {
            tally.exact += 1;
        } else {
            tally.partial += 1;
            let prec = 1.0 - fp as f64 / written.len().max(1) as f64;
            let rec = 1.0 - fnn as f64 / truth.len() as f64;
            notes.push(format!(
                "{name}{}: precision {prec:.3}, recall {rec:.3} ({} written, {} true, {fp} false, {fnn} missed)",
                if manufactured { " [map manufactured by ablation]" } else { "" },
                written.len(),
                truth.len()
            ));
        }
    }

    println!(
        "  {} of {} ablated   undet {}   inadm {}   orphaned {}   exact {}   partial {}",
        cells.len(),
        total,
        tally.undetermined,
        tally.inadmissible,
        tally.orphaned,
        tally.exact,
        tally.partial
    );
    if tally.truth > 0 {
        println!(
            "  over the {} cells expansion would write: {} tuples false, {} missed, of {} true",
            tally.exact + tally.partial,
            tally.false_written,
            tally.missed,
            tally.truth
        );
    }
    for n in notes.iter().take(6) {
        println!("    {n}");
    }
    (stem, tally, notes)
}

fn main() {
    let paths = corpus::logs_or_args();
    let theta: f64 = std::env::var("THETA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THETA);
    let cap: usize = std::env::var("MAX_CELLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    let mut per_log: BTreeMap<String, (Tally, Vec<String>)> = BTreeMap::new();
    for p in &paths {
        let (stem, t, n) = run(p, theta, cap);
        per_log.insert(stem, (t, n));
    }

    let sum = |f: fn(&Tally) -> usize| per_log.values().map(|(t, _)| f(t)).sum::<usize>();
    let (exact, partial) = (sum(|t| t.exact), sum(|t| t.partial));
    let (fp, fnn, truth) = (sum(|t| t.false_written), sum(|t| t.missed), sum(|t| t.truth));
    println!("\n{}\ncorpus\n{}", "=".repeat(72), "=".repeat(72));
    println!(
        "  undetermined {}   inadmissible {}   orphaned {}   exact {}   partial {}",
        sum(|t| t.undetermined),
        sum(|t| t.inadmissible),
        sum(|t| t.orphaned),
        exact,
        partial
    );
    let (mf, mm, mt) = (sum(|t| t.man_false), sum(|t| t.man_missed), sum(|t| t.man_truth));
    if truth > 0 {
        println!(
            "  map already held: precision {:.4}, recall {:.4} over {} tuples",
            1.0 - fp as f64 / (truth - fnn + fp).max(1) as f64,
            1.0 - fnn as f64 / truth as f64,
            truth
        );
    }
    if mt > 0 {
        println!(
            "  map manufactured by the ablation: precision {:.4}, recall {:.4} over {} tuples, {} cells",
            1.0 - mf as f64 / (mt - mm + mf).max(1) as f64,
            1.0 - mm as f64 / mt as f64,
            mt,
            sum(|t| t.manufactured)
        );
    }

    let logs: Vec<serde_json::Value> = per_log
        .iter()
        .map(|(k, (t, n))| {
            serde_json::json!({
                "log": k,
                "undetermined": t.undetermined,
                "inadmissible": t.inadmissible,
                "orphaned": t.orphaned,
                "exact": t.exact,
                "partial": t.partial,
                "manufactured_cells": t.manufactured,
                "manufactured_tuples_true": t.man_truth,
                "manufactured_tuples_false": t.man_false,
                "manufactured_tuples_missed": t.man_missed,
                "tuples_true": t.truth,
                "tuples_false_written": t.false_written,
                "tuples_missed": t.missed,
                "partial_detail": n,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "theta": theta,
        "logs": logs,
        "corpus_totals": {
            "exact": exact, "partial": partial,
            "undetermined": sum(|t| t.undetermined), "inadmissible": sum(|t| t.inadmissible),
            "orphaned": sum(|t| t.orphaned),
            "tuples_true": truth, "tuples_false_written": fp, "tuples_missed": fnn,
            "manufactured_cells": sum(|t| t.manufactured),
            "manufactured_tuples_true": mt, "manufactured_tuples_false": mf,
            "manufactured_tuples_missed": mm,
        },
    });
    let stats = corpus::stats_dir();
    std::fs::create_dir_all(&stats).ok();
    let out = stats.join("expansion_holdout.json");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).expect("serialise")).expect("write");
    println!("\nstats written to {}", out.display());
}
