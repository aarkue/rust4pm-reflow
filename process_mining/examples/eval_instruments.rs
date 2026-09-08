//! Evaluation instruments for the schema-reduction paper: ordering preservation, spliced
//! directly-follows, and performance/size, per corpus log.
//!
//! Ordering preservation classifies every activity pair the full log asserts
//! ([`Saturation::target_pairs`], the same target `search.rs` covers against) by how the
//! default-pipeline flow layer delivers it: drawn by a kept type, covered by the chained
//! closure ([`Chained`]), or carried by a schema map ([`asserted_of_route`]). A pair in none
//! of the three is printed explicitly -- the default pipeline's own `fact_repair` promotes
//! cells until every fact is drawn or determined, but determination is a fourth mechanism
//! this instrument does not credit, so an uncovered pair here is not necessarily a bug.
//!
//! Spliced directly-follows reuses the same arithmetic as [`TraceVariants::df_fidelity`],
//! bucketed per object type instead of summed over the log, because "which type pays for
//! the projection" is the number the paper reports.
//!
//! Performance times import, the reduction pipeline, and OCPN/OC-DFG discovery on the
//! recorded log and the flow projection, each as the median of three fresh runs. Size reports E2O
//! participations and exported file bytes, recorded against reduced.
//!
//! Usage: `cargo run --release --example eval_instruments -- <log> ...`
//! With no arguments, runs the corpus. Env: `REFLOW_LOGS` names the corpus directory,
//! `REFLOW_STATS` the output directory, `EXPORT_DIR` where flow projections are exported.

mod corpus;

use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    time::Instant,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, asserted_of_route, drawn, fact_repair, facts_from,
        flow_projection, reflow_layer, tag, ActivityIndexing, Bounds, CellGrid, Chained,
        ExpansionDirection, FlowLayer, Route, Saturation, SchemaClosure, SearchInput,
        StructuralSchema, TraceVariants, DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL},
    core::process_models::object_centric::ocdfg::discover_dfg_from_ocel,
    discovery::case_centric::inductive_miner::InductiveMinerOptions,
    discovery::object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions},
    Exportable, Importable, OCEL,
};

/// The noise threshold every discovery run in this file uses, so recorded and reduced are
/// discovered under the same setting and only the log differs.
const IMF_THRESHOLD: f64 = 0.2;

/// Run `f` three times, freshly, and return the median wall-clock time and the last result.
/// Only one result is ever alive at once -- each iteration's value replaces the last -- so
/// timing a heavy artifact (a discovered net, an imported log) three times costs no extra
/// peak memory over running it once.
fn median3<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut times = [0.0f64; 3];
    let mut last = None;
    for t in &mut times {
        let t0 = Instant::now();
        let r = f();
        *t = t0.elapsed().as_secs_f64();
        last = Some(r);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[1], last.unwrap())
}

/// What the default reduction pipeline produces for one log: the same sequence of calls
/// `examples/ocpn_quality.rs` makes (schema, closure, grid, activities, bounds, agreed
/// routes, saturation, handoff, fact repair), plus the tagged log and its flow projection.
struct Reduction {
    schema: StructuralSchema,
    grid: CellGrid,
    acts: ActivityIndexing,
    routes: Vec<Route>,
    /// Bounds over the log as extracted, before expansion widened them.
    recorded_bounds: Bounds,
    max: Saturation,
    best: FlowLayer,
    /// The flow projection of the tagged log the pipeline produces.
    reduced_log: SlimLinkedOCEL,
}

fn build_reduction(locel: &SlimLinkedOCEL) -> Reduction {
    let schema = StructuralSchema::discover(locel);
    let closure = SchemaClosure::build(locel, &schema);
    let grid = CellGrid::build(locel, &schema, &closure);
    let acts = ActivityIndexing::build(locel, &grid);
    let bounds = Bounds::build(locel, &schema, &acts);
    let (routes, _disagreements) = agreed_routes(&schema);
    let max = Saturation::build(
        locel,
        &schema,
        &grid,
        &acts,
        &routes,
        &bounds,
        DEFAULT_THETA,
        ExpansionDirection::default(),
    );

    let mut target: Vec<_> = max.target_pairs().into_iter().collect();
    target.sort_unstable();
    let variants = TraceVariants::build_with(locel, &schema, &acts, &max.written);
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
    let mut best = reflow_layer(&input);
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
    fact_repair(&grid, &full_facts, &mut best.cells);

    // What expansion wrote that the repaired flow layer now keeps: materialising needs
    // these written the same way `ocpn_quality.rs` writes them, or a type flowing at an
    // expanded cell has no participations behind it in the flow projection.
    let added: Vec<_> = max
        .written
        .iter()
        .filter(|(e, o)| {
            let a = acts.act_of[e.get_ev(locel).event_type];
            let t = schema.type_of[o];
            best.cells.contains(&(a, t)) && !grid.cells.contains(&(a, t))
        })
        .copied()
        .collect();
    let reduced_log = flow_projection(&tag(locel, &schema, &acts, &best.cells, &added)).into_owned();

    Reduction {
        schema,
        grid,
        acts,
        routes,
        recorded_bounds: bounds,
        max,
        best,
        reduced_log,
    }
}

/// Instrument 1: how the reduced flow layer delivers every activity pair the full log
/// asserts.
struct OrderingPreservation {
    target: usize,
    drawn: usize,
    chained: usize,
    routed: usize,
    /// Delivered only by composing a route hop with other delivered hops.
    ///
    /// Counted apart from `routed` because the two carry different evidence. A single-hop
    /// route pair is witnessed by concrete objects with concrete timestamps, which a reader
    /// of the flow projection re-checks directly. A multi-hop one is witnessed the way
    /// [`Chained`] pairs are, by composing at the level of activities, so it inherits that
    /// step's weaker guarantee and has to be reported as its own row.
    routed_multi: usize,
    uncovered: Vec<(String, String)>,
    chain_check: ChainCheck,
}

/// The chained case put through the soundness check the routed case passes by construction.
///
/// A routed pair is witnessed by concrete objects with concrete timestamps, and a reader of
/// the flow projection re-checks it against the log's own object-to-object edges.
/// A chained pair is composed at the level of activities: the two hops name different events
/// of the middle activity, possibly in the opposite time order, and the closure never looks.
/// Three conditions, all read off the bounds, on each pair the flow layer carries by
/// chaining alone.
///
/// - *timed*: the earliest event of `x` is strictly before the latest event of `y`, so some
///   pair of events in the log realises the composed ordering. The same strictness
///   [`precedes`] applies within a type, applied to the activities instead.
/// - *unreversed*: no type orders `(y, x)`, and the closure does not deliver it either. A
///   pair the picture delivers both ways tells a reader nothing.
/// - *recorded*: some type orders the pair on the log as extracted, so the composition rests
///   on participations the log holds and not only on ones expansion wrote.
///
/// Containment in the recorded closure needs no measurement. Taking a cell out of the flow
/// layer removes pairs
/// from what [`asserted_by_type`] scans and changes the forward and backward witnesses of no
/// remaining pair, so the drawn relation shrinks with the keep-set and its closure with it.
/// Chaining under a reduction therefore delivers a subset of what chaining on the full log
/// delivers, whatever these three say.
struct ChainCheck {
    /// Pairs carried by chaining alone, the population the three conditions run over.
    carried: usize,
    untimed: Vec<(String, String)>,
    reversed: Vec<(String, String)>,
    only_expanded: Vec<(String, String)>,
}

/// Earliest and latest timestamp of each activity over every object that takes part in it.
fn activity_span(bounds: &Bounds, n_activities: usize) -> Vec<Option<(i64, i64)>> {
    let mut out = vec![None; n_activities];
    for objs in &bounds.per_type {
        for ob in objs {
            for (a, first, last) in &ob.at {
                let span = out[*a].get_or_insert((*first, *last));
                span.0 = span.0.min(*first);
                span.1 = span.1.max(*last);
            }
        }
    }
    out
}

fn ordering_preservation(r: &Reduction) -> OrderingPreservation {
    let mut target: Vec<(usize, usize)> = r.max.target_pairs().into_iter().collect();
    target.sort_unstable();

    let kept = &r.best.cells;
    let drawn_set = drawn(&r.max.bounds, kept);
    let chained = Chained::of(&drawn_set, r.grid.activities.len());
    let mut route_pairs: HashSet<(usize, usize)> = HashSet::new();
    for route in &r.routes {
        route_pairs.extend(asserted_of_route(&r.max.bounds, kept, route));
    }

    // Multi-hop: close drawn and route-delivered pairs together, so a route hop may be
    // followed by a drawn hop and the other way round. Reported separately from both.
    let mut with_routes = drawn_set.clone();
    with_routes.extend(route_pairs.iter().copied());
    let multi = Chained::of(&with_routes, r.grid.activities.len());

    let (mut drawn_n, mut chained_n, mut routed_n, mut multi_n) = (0usize, 0usize, 0usize, 0usize);
    let mut uncovered: Vec<(usize, usize)> = Vec::new();
    let mut chain_carried: Vec<(usize, usize)> = Vec::new();
    for p in &target {
        if drawn_set.contains(p) {
            drawn_n += 1;
        } else if chained.holds(p.0, p.1) {
            chained_n += 1;
            chain_carried.push(*p);
        } else if route_pairs.contains(p) {
            routed_n += 1;
        } else if multi.holds(p.0, p.1) {
            multi_n += 1;
        } else {
            uncovered.push(*p);
        }
    }
    uncovered.sort_unstable();
    let name = |(a, b): &(usize, usize)| {
        (r.grid.activities[*a].clone(), r.grid.activities[*b].clone())
    };
    let uncovered: Vec<(String, String)> = uncovered.iter().map(name).collect();

    let span = activity_span(&r.max.bounds, r.grid.activities.len());
    let target_set: HashSet<(usize, usize)> = target.iter().copied().collect();
    let recorded: HashSet<(usize, usize)> =
        asserted_by_type(&r.recorded_bounds, &r.grid.cells).into_iter().flatten().collect();
    let timed = |&(x, y): &(usize, usize)| match (span[x], span[y]) {
        (Some((first_x, _)), Some((_, last_y))) => first_x < last_y,
        _ => false,
    };
    let chain_check = ChainCheck {
        carried: chain_carried.len(),
        untimed: chain_carried.iter().filter(|p| !timed(p)).map(name).collect(),
        reversed: chain_carried
            .iter()
            .filter(|(x, y)| target_set.contains(&(*y, *x)) || chained.holds(*y, *x))
            .map(name)
            .collect(),
        only_expanded: chain_carried
            .iter()
            .filter(|p| !recorded.contains(*p))
            .map(name)
            .collect(),
    };

    OrderingPreservation {
        target: target.len(),
        drawn: drawn_n,
        chained: chained_n,
        routed: routed_n,
        routed_multi: multi_n,
        uncovered,
        chain_check,
    }
}

/// Instrument 2: directly-follows pairs the reduced flow layer's projected traces contain
/// that no original (unprojected) trace of that type supports, per type.
struct Splicing {
    reduced_arcs: usize,
    spurious: usize,
    missing: usize,
    worst: Option<(String, usize)>,
    per_type: Vec<(String, usize, usize, usize)>,
}

fn splicing(locel: &SlimLinkedOCEL, r: &Reduction) -> Splicing {
    let variants = TraceVariants::build(locel, &r.schema, &r.acts);
    let full = variants.arc_set(&r.grid.cells);
    let reduced = variants.arc_set(&r.best.cells);
    let kept_types: HashSet<usize> = r.best.cells.iter().map(|(_, t)| *t).collect();
    let full_on_kept: HashSet<_> = full
        .iter()
        .filter(|(t, a, b)| {
            kept_types.contains(t) && r.best.cells.contains(&(*a, *t)) && r.best.cells.contains(&(*b, *t))
        })
        .copied()
        .collect();

    let mut per: BTreeMap<usize, (usize, usize, usize)> = BTreeMap::new();
    for arc in &reduced {
        per.entry(arc.0).or_default().0 += 1;
    }
    for arc in reduced.difference(&full_on_kept) {
        per.entry(arc.0).or_default().1 += 1;
    }
    for arc in full_on_kept.difference(&reduced) {
        per.entry(arc.0).or_default().2 += 1;
    }

    let per_type: Vec<(String, usize, usize, usize)> = per
        .into_iter()
        .map(|(t, (ra, sp, mi))| (r.schema.types[t].clone(), ra, sp, mi))
        .collect();

    // Highest spurious count first; alphabetically first on a tie, so the answer does not
    // depend on hash or insertion order.
    let worst = per_type
        .iter()
        .filter(|(_, _, sp, _)| *sp > 0)
        .min_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)))
        .map(|(name, _, sp, _)| (name.clone(), *sp));

    Splicing {
        reduced_arcs: reduced.len(),
        spurious: per_type.iter().map(|(_, _, sp, _)| sp).sum(),
        missing: per_type.iter().map(|(_, _, _, mi)| mi).sum(),
        worst,
        per_type,
    }
}

/// Distinct `(event, object)` participations, deduplicated across qualifiers the same way
/// [`CellGrid::build`]'s `e2o_total` is.
fn participations(locel: &SlimLinkedOCEL) -> usize {
    locel
        .get_all_evs()
        .map(|e| e.get_e2o(locel).collect::<HashSet<_>>().len())
        .sum()
}

struct LogReport {
    log: String,
    path: String,
    ordering: OrderingPreservation,
    splice: Splicing,
    import_s: f64,
    reduction_s: f64,
    ocpn_recorded_s: f64,
    ocpn_reduced_s: f64,
    ocdfg_recorded_s: f64,
    ocdfg_reduced_s: f64,
    e2o_recorded: usize,
    e2o_reduced: usize,
    file_bytes_recorded: u64,
    file_bytes_reduced: u64,
}

fn process_log(path: &Path, export_dir: &Path) -> LogReport {
    let stem = log_stem(&path.to_string_lossy());
    println!("\n{}\nprocessing {}\n{}", "=".repeat(72), path.display(), "=".repeat(72));

    let (import_s, ocel) = median3(|| OCEL::import_from_path(path).expect("import log"));
    println!("  import        median {import_s:.3}s   {} events / {} objects", ocel.events.len(), ocel.objects.len());
    let locel = SlimLinkedOCEL::from_ocel(ocel);

    let (reduction_s, reduction) = median3(|| build_reduction(&locel));
    println!(
        "  reduction     median {reduction_s:.3}s   strategy {}   flow cells {}",
        reduction.best.strategy.label(),
        reduction.best.cells.len()
    );

    let opts = || ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(IMF_THRESHOLD));
    let (ocpn_recorded_s, _net) = median3(|| discover_ocpn(&locel, opts()));
    let (ocpn_reduced_s, _net) = median3(|| discover_ocpn(&reduction.reduced_log, opts()));
    println!("  ocpn discovery recorded median {ocpn_recorded_s:.3}s   reduced median {ocpn_reduced_s:.3}s");

    let (ocdfg_recorded_s, _g) = median3(|| discover_dfg_from_ocel(&locel));
    let (ocdfg_reduced_s, _g) = median3(|| discover_dfg_from_ocel(&reduction.reduced_log));
    println!("  ocdfg discovery recorded median {ocdfg_recorded_s:.3}s   reduced median {ocdfg_reduced_s:.3}s");

    let ordering = ordering_preservation(&reduction);
    let splice = splicing(&locel, &reduction);

    let e2o_recorded = participations(&locel);
    let e2o_reduced = participations(&reduction.reduced_log);
    let file_bytes_recorded = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let export_path = export_dir.join(format!("{stem}.reduced.xml"));
    reduction
        .reduced_log
        .export_to_path(&export_path)
        .expect("export flow projection");
    let file_bytes_reduced = std::fs::metadata(&export_path).map(|m| m.len()).unwrap_or(0);

    println!(
        "  ordering      target {:<7} drawn {:<7} chained {:<7} routed {:<7} routed* {:<7} uncovered {}",
        ordering.target, ordering.drawn, ordering.chained, ordering.routed,
        ordering.routed_multi, ordering.uncovered.len()
    );
    if !ordering.uncovered.is_empty() {
        println!("    uncovered pairs (expected none by construction):");
        for (a, b) in &ordering.uncovered {
            println!("      {a} -> {b}");
        }
    }
    let cc = &ordering.chain_check;
    println!(
        "  chain check   carried {:<7} untimed {:<7} reversed {:<7} only in expansion {}",
        cc.carried,
        cc.untimed.len(),
        cc.reversed.len(),
        cc.only_expanded.len()
    );
    for (label, pairs) in [
        ("untimed", &cc.untimed),
        ("reversed", &cc.reversed),
        ("only in expansion", &cc.only_expanded),
    ] {
        for (a, b) in pairs.iter().take(5) {
            println!("      {label:<18} {a} -> {b}");
        }
    }
    println!(
        "  splicing      reduced_arcs {:<7} spurious {:<7} missing {:<7} worst {}",
        splice.reduced_arcs,
        splice.spurious,
        splice.missing,
        splice
            .worst
            .as_ref()
            .map(|(t, n)| format!("{t} ({n})"))
            .unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  size          E2O recorded {e2o_recorded:<9} reduced {e2o_reduced:<9}   file bytes recorded {file_bytes_recorded:<10} reduced {file_bytes_reduced}"
    );

    LogReport {
        log: stem,
        path: path.to_string_lossy().into_owned(),
        ordering,
        splice,
        import_s,
        reduction_s,
        ocpn_recorded_s,
        ocpn_reduced_s,
        ocdfg_recorded_s,
        ocdfg_reduced_s,
        e2o_recorded,
        e2o_reduced,
        file_bytes_recorded,
        file_bytes_reduced,
    }
}

fn pairs_json(pairs: &[(String, String)]) -> Vec<serde_json::Value> {
    pairs.iter().map(|(a, b)| serde_json::json!([a, b])).collect()
}

fn to_json(r: &LogReport) -> serde_json::Value {
    let uncovered = pairs_json(&r.ordering.uncovered);
    let per_type: Vec<serde_json::Value> = r
        .splice
        .per_type
        .iter()
        .map(|(t, ra, sp, mi)| {
            serde_json::json!({
                "type": t, "reduced_arcs": ra, "spurious": sp, "missing": mi,
            })
        })
        .collect();
    serde_json::json!({
        "log": r.log,
        "path": corpus::label(&r.path),
        "ordering_preservation": {
            "target_pairs": r.ordering.target,
            "drawn": r.ordering.drawn,
            "chained": r.ordering.chained,
            "routed": r.ordering.routed,
            "routed_multi_hop": r.ordering.routed_multi,
            "uncovered": r.ordering.uncovered.len(),
            "uncovered_pairs": uncovered,
            "chain_check": {
                "carried": r.ordering.chain_check.carried,
                "untimed": pairs_json(&r.ordering.chain_check.untimed),
                "reversed": pairs_json(&r.ordering.chain_check.reversed),
                "only_in_expansion": pairs_json(&r.ordering.chain_check.only_expanded),
            },
        },
        "spliced_directly_follows": {
            "reduced_arcs": r.splice.reduced_arcs,
            "spurious": r.splice.spurious,
            "missing": r.splice.missing,
            "worst_type": r.splice.worst.as_ref().map(|(t, n)| serde_json::json!({"type": t, "spurious": n})),
            "per_type": per_type,
        },
        "performance_seconds_median3": {
            "import": r.import_s,
            "reduction": r.reduction_s,
            "ocpn_discovery_recorded": r.ocpn_recorded_s,
            "ocpn_discovery_reduced": r.ocpn_reduced_s,
            "ocdfg_discovery_recorded": r.ocdfg_recorded_s,
            "ocdfg_discovery_reduced": r.ocdfg_reduced_s,
        },
        "size": {
            "e2o_participations_recorded": r.e2o_recorded,
            "e2o_participations_reduced": r.e2o_reduced,
            "file_bytes_recorded": r.file_bytes_recorded,
            "file_bytes_reduced": r.file_bytes_reduced,
        },
    })
}

fn print_summary_tables(reports: &[LogReport]) {
    println!("\n{}\nordering preservation\n{}", "=".repeat(90), "=".repeat(90));
    println!(
        "  {:<16}{:>9}{:>9}{:>9}{:>9}{:>9}{:>11}{:>9}{:>10}{:>10}",
        "log", "target", "drawn", "chained", "routed", "routed*", "uncovered", "untimed",
        "reversed", "expanded"
    );
    let mut chain_totals = [0usize; 4];
    for r in reports {
        let cc = &r.ordering.chain_check;
        for (slot, n) in chain_totals.iter_mut().zip([
            cc.carried,
            cc.untimed.len(),
            cc.reversed.len(),
            cc.only_expanded.len(),
        ]) {
            *slot += n;
        }
        println!(
            "  {:<16}{:>9}{:>9}{:>9}{:>9}{:>9}{:>11}{:>9}{:>10}{:>10}",
            r.log, r.ordering.target, r.ordering.drawn, r.ordering.chained, r.ordering.routed,
            r.ordering.routed_multi, r.ordering.uncovered.len(), cc.untimed.len(),
            cc.reversed.len(), cc.only_expanded.len()
        );
    }
    println!(
        "  chain check: {} carried by chaining alone, {} untimed, {} reversed, {} only in expansion",
        chain_totals[0], chain_totals[1], chain_totals[2], chain_totals[3]
    );

    println!("\n{}\nspliced directly-follows\n{}", "=".repeat(90), "=".repeat(90));
    println!(
        "  {:<16}{:>13}{:>11}{:>10}  {}",
        "log", "reduced_arcs", "spurious", "missing", "worst type (spurious)"
    );
    for r in reports {
        println!(
            "  {:<16}{:>13}{:>11}{:>10}  {}",
            r.log,
            r.splice.reduced_arcs,
            r.splice.spurious,
            r.splice.missing,
            r.splice
                .worst
                .as_ref()
                .map(|(t, n)| format!("{t} ({n})"))
                .unwrap_or_else(|| "-".to_string())
        );
    }

    println!("\n{}\nperformance (median of 3, seconds) and size\n{}", "=".repeat(110), "=".repeat(110));
    println!(
        "  {:<16}{:>8}{:>10}{:>12}{:>12}{:>12}{:>12}{:>14}{:>14}{:>13}{:>13}",
        "log", "import", "reduce", "ocpn(rec)", "ocpn(red)", "dfg(rec)", "dfg(red)",
        "E2O(rec)", "E2O(red)", "bytes(rec)", "bytes(red)"
    );
    for r in reports {
        println!(
            "  {:<16}{:>8.3}{:>10.3}{:>12.3}{:>12.3}{:>12.3}{:>12.3}{:>14}{:>14}{:>13}{:>13}",
            r.log, r.import_s, r.reduction_s, r.ocpn_recorded_s, r.ocpn_reduced_s,
            r.ocdfg_recorded_s, r.ocdfg_reduced_s, r.e2o_recorded, r.e2o_reduced,
            r.file_bytes_recorded, r.file_bytes_reduced
        );
    }
}

fn main() {
    let paths: Vec<PathBuf> = corpus::logs_or_args().iter().map(PathBuf::from).collect();

    let export_dir = std::env::var("EXPORT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| corpus::stats_dir().join("../exports"));
    std::fs::create_dir_all(&export_dir).expect("create export dir");

    let mut reports: Vec<LogReport> = paths
        .iter()
        .map(|p| process_log(p, &export_dir))
        .collect();
    reports.sort_by(|a, b| a.log.cmp(&b.log));

    print_summary_tables(&reports);

    let logs: Vec<serde_json::Value> = reports.iter().map(to_json).collect();
    let doc = serde_json::json!({ "logs": logs });
    let out_dir = corpus::stats_dir();
    std::fs::create_dir_all(&out_dir).expect("create stats dir");
    let out_path = out_dir.join("eval_instruments.json");
    let text = serde_json::to_string_pretty(&doc).expect("serialize json");
    std::fs::write(&out_path, text).expect("write stats json");
    println!("\nstats written to {}", out_path.display());
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
