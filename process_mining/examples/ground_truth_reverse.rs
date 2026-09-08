//! The CPN ground-truth check run in the reverse direction, and the expansion direction
//! adjudicated against the same generators.
//!
//! `code/footprint_check.py` asks whether the reduced flow layer still shows what the
//! generating net enforces. It never asks the opposite question: does the reduced flow layer
//! show an ordering the generator does not enforce? Splicing a non-flow activity out of a
//! type's trace is what would produce one, so the question is worth measuring rather than
//! arguing. Both logs published together with their CPN~Tools models are run here.
//!
//! Reverse reduction, three populations read off the recorded bounds so expansion cannot
//! confound them:
//!
//! - the orderings the recorded flow layer shows, and the ones the reduced flow layer shows,
//!   each drawn by a kept type, composed by [`Chained`], or delivered by a schema route,
//! - the directly-follows arcs each projection draws, per object type,
//! - every ordering either layer shows whose reverse the generator enforces.
//!
//! Expansion: the pairs the saturated target holds that the recorded log does not assert,
//! which is exactly the `+exp` column of the paper's cross-instantiation table. Each is
//! adjudicated against the generator: *confirmed* when the footprint enforces it,
//! *contradicted* when the footprint enforces its reverse, *unadjudicated* when the net
//! constrains neither direction. Only the second is a defect. A footprint lists what the net
//! forces, not everything it permits, so the third bucket is the expected majority.
//!
//! Enforced pairs are read as a union over the footprint's types and also under transitive
//! closure. The closure is the wider reading and is reported apart, because composing
//! enforced pairs across types is the same step [`Chained`] takes and carries the same
//! weaker guarantee. Pairs the footprint marks `uncertain` are excluded from both.
//!
//! Usage: `cargo run --release --example ground_truth_reverse`
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory; the
//! footprints are read from `ground_truth/` beside it.

mod corpus;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, asserted_of_route, drawn, fact_repair, facts_from,
        reflow_layer, ActivityIndexing, Bounds, Cell, CellGrid, Chained, ExpansionDirection, Pair,
        Route, Saturation, SchemaClosure, SearchInput, StructuralSchema, TraceVariants,
        DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};

/// The two corpus logs published together with the CPN model that generated them.
const GROUND_TRUTH: &[(&str, &str)] = &[
    ("order-management.xml", "order-management.footprint.json"),
    ("ContainerLogistics.xml", "container-logistics.footprint.json"),
];

fn footprint_dir() -> PathBuf {
    corpus::stats_dir().join("../ground_truth")
}

/// Enforced orderings read off one hand-derived footprint, as activity index pairs.
struct Footprint {
    enforced: HashSet<Pair>,
    closed: HashSet<Pair>,
    /// Footprint activities the log's own activity set does not contain, which would silently
    /// shrink the enforced set if left unreported.
    unmatched: Vec<String>,
}

fn transitive_closure(pairs: &HashSet<Pair>) -> HashSet<Pair> {
    let mut out = pairs.clone();
    loop {
        let mut add = Vec::new();
        for &(a, b) in &out {
            for &(c, d) in &out {
                if b == c && a != d && !out.contains(&(a, d)) {
                    add.push((a, d));
                }
            }
        }
        if add.is_empty() {
            return out;
        }
        out.extend(add);
    }
}

fn read_footprint(path: &PathBuf, activities: &[String]) -> Footprint {
    let text = std::fs::read_to_string(path).expect("read footprint");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("parse footprint");
    let index: HashMap<&str, usize> =
        activities.iter().enumerate().map(|(i, a)| (a.as_str(), i)).collect();

    let mut enforced = HashSet::new();
    let mut unmatched = Vec::new();
    for spec in doc["types"].as_object().expect("footprint types").values() {
        for pair in spec["enforced"].as_array().into_iter().flatten() {
            let a = pair[0].as_str().expect("footprint activity");
            let b = pair[1].as_str().expect("footprint activity");
            match (index.get(a), index.get(b)) {
                (Some(x), Some(y)) => {
                    enforced.insert((*x, *y));
                }
                _ => unmatched.push(format!("{a} < {b}")),
            }
        }
    }
    unmatched.sort();
    unmatched.dedup();
    let closed = transitive_closure(&enforced);
    Footprint { enforced, closed, unmatched }
}

/// Everything the pipeline builds for one log, up to the repaired flow layers of both
/// directions.
struct Run {
    activities: Vec<String>,
    types: Vec<String>,
    recorded_cells: HashSet<Cell>,
    reduced_cells: HashSet<Cell>,
    recorded_bounds: Bounds,
    routes: Vec<Route>,
    variants: TraceVariants,
    recorded_target: HashSet<Pair>,
    saturated_target: HashSet<Pair>,
    expanded_shown: HashSet<Pair>,
}

fn build(path: &str) -> Run {
    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);

    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);
    let bounds = Bounds::build(&locel, &schema, &acts);
    let (routes, _) = agreed_routes(&schema);
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
    let search_variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
    let objects = Saturation::objects_per_type(&schema);
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);

    let allowed = grid.cells.clone();
    let mut best = reflow_layer(&SearchInput {
        max: &max,
        variants: &search_variants,
        allowed: &allowed,
        recorded: &grid.cells,
        target: &target,
        objects_per_type: &objects,
        rep: &closure.rep,
        types: &schema.types,
        n_activities: grid.activities.len(),
    });
    fact_repair(&grid, &full_facts, &mut best.cells);

    // The expansion direction differs in one field: the search may buy every cell the
    // saturation admits instead of the recorded ones alone.
    let allowed_exp = max.cells.clone();
    let mut exp = reflow_layer(&SearchInput {
        max: &max,
        variants: &search_variants,
        allowed: &allowed_exp,
        recorded: &grid.cells,
        target: &target,
        objects_per_type: &objects,
        rep: &closure.rep,
        types: &schema.types,
        n_activities: grid.activities.len(),
    });
    fact_repair(&grid, &full_facts, &mut exp.cells);

    let n = grid.activities.len();
    let expanded_shown = shown(&max.bounds, &exp.cells, &routes, n);

    let recorded_target: HashSet<Pair> =
        asserted_by_type(&bounds, &grid.cells).into_iter().flatten().collect();

    Run {
        activities: grid.activities.clone(),
        types: schema.types.clone(),
        recorded_cells: grid.cells.clone(),
        reduced_cells: best.cells.clone(),
        recorded_bounds: bounds,
        routes,
        variants: TraceVariants::build(&locel, &schema, &acts),
        recorded_target,
        saturated_target: target.into_iter().collect(),
        expanded_shown,
    }
}

/// Every ordering a flow layer shows: drawn by a kept type, composed by the chained closure,
/// or delivered object by object along a schema route.
fn shown(bounds: &Bounds, cells: &HashSet<Cell>, routes: &[Route], n: usize) -> HashSet<Pair> {
    let drawn_set = drawn(bounds, cells);
    let chained = Chained::of(&drawn_set, n);
    let mut out: HashSet<Pair> = HashSet::new();
    for route in routes {
        out.extend(asserted_of_route(bounds, cells, route));
    }
    for x in 0..n {
        for y in 0..n {
            if x != y && (drawn_set.contains(&(x, y)) || chained.holds(x, y)) {
                out.insert((x, y));
            }
        }
    }
    out
}

fn names(run: &Run, pairs: &[Pair]) -> Vec<serde_json::Value> {
    pairs
        .iter()
        .map(|(a, b)| serde_json::json!([run.activities[*a], run.activities[*b]]))
        .collect()
}

fn sorted(set: impl IntoIterator<Item = Pair>) -> Vec<Pair> {
    let mut v: Vec<Pair> = set.into_iter().collect();
    v.sort_unstable();
    v
}

/// Orderings a layer shows whose reverse the generator enforces.
fn contradicting(shown: &HashSet<Pair>, enforced: &HashSet<Pair>) -> Vec<Pair> {
    sorted(shown.iter().copied().filter(|(x, y)| enforced.contains(&(*y, *x))))
}

fn report(path: &str, footprint_file: &str) -> serde_json::Value {
    let stem = log_stem(path);
    println!("\n{}\n{stem}\n{}", "=".repeat(72), "=".repeat(72));

    let run = build(path);
    let fp = read_footprint(&footprint_dir().join(footprint_file), &run.activities);
    let n = run.activities.len();

    // --- reduction, reverse direction ---
    let recorded_shown = shown(&run.recorded_bounds, &run.recorded_cells, &run.routes, n);
    let reduced_shown = shown(&run.recorded_bounds, &run.reduced_cells, &run.routes, n);
    let added = sorted(reduced_shown.difference(&recorded_shown).copied());

    let recorded_arcs = run.variants.arc_set(&run.recorded_cells);
    let reduced_arcs = run.variants.arc_set(&run.reduced_cells);
    let recorded_on_kept: HashSet<_> = recorded_arcs
        .iter()
        .filter(|(t, a, b)| {
            run.reduced_cells.contains(&(*a, *t)) && run.reduced_cells.contains(&(*b, *t))
        })
        .copied()
        .collect();
    let mut spliced: Vec<_> = reduced_arcs.difference(&recorded_on_kept).copied().collect();
    spliced.sort_unstable();
    let spliced_named: Vec<serde_json::Value> = spliced
        .iter()
        .map(|(t, a, b)| {
            serde_json::json!([run.types[*t], run.activities[*a], run.activities[*b]])
        })
        .collect();

    let contra_reduced = contradicting(&reduced_shown, &fp.enforced);
    let contra_recorded = contradicting(&recorded_shown, &fp.enforced);
    let contra_reduced_closed = contradicting(&reduced_shown, &fp.closed);
    let contra_recorded_closed = contradicting(&recorded_shown, &fp.closed);

    println!(
        "  enforced {} ({} under closure), unmatched activities {:?}",
        fp.enforced.len(),
        fp.closed.len(),
        fp.unmatched
    );
    println!(
        "  orderings shown: recorded {}, reduced {}, added by the reduction {}",
        recorded_shown.len(),
        reduced_shown.len(),
        added.len()
    );
    println!(
        "  directly-follows arcs: recorded {}, reduced {}, spliced {}",
        recorded_arcs.len(),
        reduced_arcs.len(),
        spliced.len()
    );
    println!(
        "  contradicting the generator: recorded {} / reduced {} (closure: {} / {})",
        contra_recorded.len(),
        contra_reduced.len(),
        contra_recorded_closed.len(),
        contra_reduced_closed.len()
    );

    // --- expansion, adjudicated against the generator ---
    let expansion_added =
        sorted(run.saturated_target.difference(&run.recorded_target).copied());
    let realized = expansion_added.iter().filter(|p| run.expanded_shown.contains(p)).count();
    let confirmed: Vec<Pair> =
        expansion_added.iter().copied().filter(|p| fp.enforced.contains(p)).collect();
    let contradicted: Vec<Pair> = expansion_added
        .iter()
        .copied()
        .filter(|(x, y)| fp.enforced.contains(&(*y, *x)))
        .collect();
    let confirmed_closed: Vec<Pair> =
        expansion_added.iter().copied().filter(|p| fp.closed.contains(p)).collect();
    let contradicted_closed: Vec<Pair> = expansion_added
        .iter()
        .copied()
        .filter(|(x, y)| fp.closed.contains(&(*y, *x)))
        .collect();
    let unadjudicated = expansion_added.len() - confirmed.len() - contradicted.len();

    println!(
        "  expansion adds {} orderings, {} of them realized by the expansion search",
        expansion_added.len(),
        realized
    );
    println!(
        "    against the generator: confirmed {}, contradicted {}, unadjudicated {} \
         (closure: confirmed {}, contradicted {})",
        confirmed.len(),
        contradicted.len(),
        unadjudicated,
        confirmed_closed.len(),
        contradicted_closed.len()
    );
    for (a, b) in &expansion_added {
        let verdict = if fp.enforced.contains(&(*a, *b)) {
            "confirmed"
        } else if fp.enforced.contains(&(*b, *a)) {
            "CONTRADICTED"
        } else if fp.closed.contains(&(*a, *b)) {
            "confirmed (closure)"
        } else if fp.closed.contains(&(*b, *a)) {
            "CONTRADICTED (closure)"
        } else {
            "unconstrained"
        };
        println!(
            "      {:<30} < {:<30} {verdict}",
            run.activities[*a], run.activities[*b]
        );
    }

    serde_json::json!({
        "log": stem,
        "footprint": footprint_file,
        "enforced": fp.enforced.len(),
        "enforced_under_closure": fp.closed.len(),
        "footprint_activities_not_in_log": fp.unmatched,
        "reduction_reverse": {
            "orderings_shown_recorded": recorded_shown.len(),
            "orderings_shown_reduced": reduced_shown.len(),
            "orderings_added_by_reduction": names(&run, &added),
            "directly_follows_arcs_recorded": recorded_arcs.len(),
            "directly_follows_arcs_reduced": reduced_arcs.len(),
            "spliced_arcs": spliced_named,
            "contradicting_generator_recorded": names(&run, &contra_recorded),
            "contradicting_generator_reduced": names(&run, &contra_reduced),
            "contradicting_generator_recorded_closure": names(&run, &contra_recorded_closed),
            "contradicting_generator_reduced_closure": names(&run, &contra_reduced_closed),
        },
        "expansion": {
            "added": names(&run, &expansion_added),
            "realized_by_expansion_search": realized,
            "confirmed": names(&run, &confirmed),
            "contradicted": names(&run, &contradicted),
            "unadjudicated": unadjudicated,
            "confirmed_under_closure": names(&run, &confirmed_closed),
            "contradicted_under_closure": names(&run, &contradicted_closed),
        }
    })
}

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

fn main() {
    let logs_dir = corpus::logs_dir();
    let logs: Vec<serde_json::Value> = GROUND_TRUTH
        .iter()
        .map(|(log, fp)| report(&logs_dir.join(log).to_string_lossy(), fp))
        .collect();

    let doc = serde_json::json!({ "logs": logs });
    let out_dir = corpus::stats_dir();
    std::fs::create_dir_all(&out_dir).expect("create stats dir");
    let out_path = out_dir.join("ground_truth_reverse.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).expect("serialize"))
        .expect("write stats json");
    println!("\nstats written to {}", out_path.display());
}
