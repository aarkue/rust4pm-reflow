//! Cross-instantiation comparison for the schema-reduction paper: the same schema discovery
//! and the same search (`reflow_layer`), run per corpus log under each behavioural abstraction,
//! swapping only what a type asserts and when kept cells cover it:
//!
//! - **log**: today's default -- [`asserted_of_type`] read off the recorded log's bounds.
//! - **model**: [`TreeAbstraction`] over [`model_abstraction_with`], a per-type `IMf` tree at
//!   the preset OCPN discovery uses ([`IMF_THRESHOLD`]): sequence cuts as orderings, exclusive
//!   choice cuts as exclusions, loop cuts and sequences inside loops as alternations, the latter
//!   two read only above every concurrent cut and covered only by a kept type showing the same
//!   cut.
//! - **declare**: [`declare_abstraction`], the crate's own OC-DECLARE discovery restricted to
//!   one type at a time, orderings and coexistence both blocking. The paper's `dec` column.
//! - **declare~**: the same with coexistence only reported, not blocking. Priced, not proposed.
//!
//! The swap is the [`Abstraction`] the search runs under: `search.rs`'s `cover`/`Drawn` read
//! what a type asserts through it instead of calling `asserted_of_type` directly, and the
//! `log` lens is exactly [`LogAbstraction`] over `max`'s own bounds -- the same call `reflow_layer`
//! always made, so its answer is checked against `results/stats/<log>.keepsets.json` (written
//! by `examples/ocpn_quality.rs` from the unparameterised code) and the run aborts rather than
//! writing a report if any log's flow/involved/implied sets differ.
//!
//! Everything else about the pipeline is shared across the three lenses: the same schema,
//! saturation, search and `fact_repair`. What differs between the runs is the
//! requirement. Each lens states its own from its own model over `max.cells`, and OC-DECLARE
//! brings its own covering rule as well, so the lens now moves the reduction and not only the
//! verdict.
//!
//! That makes arcs removed useless as a cross-lens ranking, since a lens asserting less
//! reduces more for free. `against_recorded_log.orderings_shown` is the column that asks all
//! three the same question: of the orderings the recorded log itself asserts, how many does
//! this lens's model still show. It is drawn plus chained with no route credit, so it
//! is a lower bound to be read across lenses and never as an absolute preservation rate.
//!
//! Per log and lens this reports: neutral cells (recorded activities the lens's pairs never
//! touch), cells the coverage step alone lets leave the flow layer (recorded cells outside
//! the pre-repair flow layer), final state counts after repair (flow/involved/implied), arcs
//! removed in OCPN and OC-DFG discovered from the lens's flow projection (against the same recorded-log
//! models), and the lens's own runtime. It also counts, per pair of lenses, how many recorded
//! cells the two lenses give a different asserting-vs-neutral verdict, and of those, how many
//! end up in a different final state ([`CellState`]) once each lens's search has run.
//!
//! Usage: `cargo run --release --example cross_instantiation -- <log> ...` (defaults to the
//! corpus). Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the stats directory
//! (keepsets are read from it, `cross_instantiation.json` written to it unless `CROSS_OUT`
//! names another file), `KEEPSET_DIR` optionally receives each lens's flow layer.

mod corpus;

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    path::PathBuf,
    time::Instant,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, asserted_of_route, asserted_of_type, assign,
        declare_abstraction, drawn, fact_repair, facts_from, flow_projection,
        model_abstraction_with, novel_cells, novelty, reflow_layer_with, tag, Abstraction,
        ActivityIndex, ActivityIndexing, Assertion, Bounds, Cell, CellGrid, CellState, Chained,
        DeclareAbstraction, ExpansionDirection, Facts, LogAbstraction, Pair, Route, Saturation,
        SchemaClosure, SearchInput, StructuralSchema, TraceVariants, TreeAbstraction,
        DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    core::process_models::object_centric::{
        ocdfg::{discover_dfg_from_ocel, OCDirectlyFollowsGraph},
        ocpn::ObjectCentricPetriNet,
    },
    discovery::case_centric::inductive_miner::InductiveMinerOptions,
    discovery::object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions},
    Importable, OCEL,
};

/// The `IMf` threshold OCPN discovery uses for the arcs-removed columns, matching
/// `examples/eval_instruments.rs`'s `IMF_THRESHOLD` so recorded and reduced are discovered
/// under the same setting.
const IMF_THRESHOLD: f64 = 0.2;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Lens {
    Log,
    Model,
    Declare,
    /// OC-DECLARE with coexistence only reported instead of blocking a cell from leaving the
    /// flow layer. Priced, not
    /// proposed: it is what dropping coexistence from the requirement would buy. Excluded from
    /// the pairwise disagreement counts, which stay over the three primary instantiations.
    DeclareReported,
}

impl Lens {
    fn label(&self) -> &'static str {
        match self {
            Lens::Log => "log",
            Lens::Model => "model",
            Lens::Declare => "declare",
            Lens::DeclareReported => "declare~",
        }
    }
}

/// Activities of `activities` that no pair of `pairs` touches -- the neutral verdict, read
/// the same way for every lens: [`super::TypeModel::silent`] and
/// [`super::DeclareTypeModel::neutral`] compute exactly this over their own `ordered`, and
/// the log lens gets nothing built in, so it is repeated here once.
fn silent_of(activities: &BTreeSet<ActivityIndex>, pairs: &HashSet<Pair>) -> BTreeSet<ActivityIndex> {
    let touched: BTreeSet<ActivityIndex> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    activities.difference(&touched).copied().collect()
}

/// Collapse silent structure that constrains nothing, then count arcs.
///
/// `examples/ocpn_quality.rs` does this before every arc figure it prints, and the paper's
/// arc counts are that convention: dead `p -> tau -> q` chains, single-branch choice
/// skeletons and emptied types' husks are construction verbosity, not model size. Without it
/// Order Management reads 238 recorded arcs against the 163 the paper prints, so an arc
/// column measured the other way cannot be set beside any other number in the evaluation.
fn ocpn_arcs(net: &ObjectCentricPetriNet) -> usize {
    let mut net = net.clone();
    for component in net.nets.values_mut() {
        component.simplify_silent();
    }
    net.nets.values().map(|n| n.arcs.len()).sum()
}

fn ocdfg_arcs(g: &OCDirectlyFollowsGraph) -> usize {
    g.object_type_to_dfg
        .values()
        .map(|d| d.directly_follows_relations.len())
        .sum()
}

/// Compare the `log` lens's final per-cell states against
/// `results/stats/<stem>.keepsets.json`, written by the unparameterised
/// `examples/ocpn_quality.rs`. Panics with a diff rather than returning an error: a mismatch
/// means the additive parameterisation changed the default path's behaviour, and the run must
/// stop instead of reporting numbers built on that.
fn verify_default_path(
    stem: &str,
    states: &HashMap<Cell, CellState>,
    grid: &CellGrid,
    schema: &StructuralSchema,
) {
    let path = corpus::stats_dir().join(format!("{stem}.keepsets.json")).to_string_lossy().into_owned();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("keepset verification: cannot read {path}: {e}"));
    let doc: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("keepset verification: cannot parse {path}: {e}"));
    let want = |key: &str| -> BTreeSet<(String, String)> {
        doc[key]
            .as_array()
            .unwrap_or_else(|| panic!("keepset verification: {path} has no \"{key}\" array"))
            .iter()
            .map(|p| {
                let p = p.as_array().expect("keepset entry is a 2-element array");
                (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string())
            })
            .collect()
    };

    let named_state = |want: CellState| -> BTreeSet<(String, String)> {
        states
            .iter()
            .filter(|(_, s)| **s == want)
            .map(|(&(a, t), _)| (grid.activities[a].clone(), schema.types[t].clone()))
            .collect()
    };
    let got_flow = named_state(CellState::Flow);
    let got_inv = named_state(CellState::Involvement);
    let got_implied = named_state(CellState::Implied);
    // `absent` is the key the stored keepset files use for the implied cells; the file
    // format is a contract with the published results and does not change.
    let (want_flow, want_inv, want_implied) = (want("flow"), want("inv"), want("absent"));

    let mut problems = Vec::new();
    for (label, got, want) in [
        ("flow", &got_flow, &want_flow),
        ("inv", &got_inv, &want_inv),
        ("absent", &got_implied, &want_implied),
    ] {
        if got != want {
            let missing: Vec<_> = want.difference(got).collect();
            let extra: Vec<_> = got.difference(want).collect();
            problems.push(format!(
                "  {label}: missing {missing:?}  extra {extra:?}"
            ));
        }
    }
    if !problems.is_empty() {
        panic!(
            "KEEPSET MISMATCH for {stem} against {path} -- the default path's behaviour \
             changed; stopping instead of proceeding:\n{}",
            problems.join("\n")
        );
    }
    println!(
        "  [keepset check] {stem}: OK ({} flow, {} inv, {} implied, matches {path})",
        got_flow.len(),
        got_inv.len(),
        got_implied.len()
    );
}

/// What one lens's search produced for one log, plus the per-cell verdicts needed for the
/// cross-lens disagreement count.
struct LensRun {
    lens: Lens,
    runtime_seconds: f64,
    neutral_cells: usize,
    asserting_cells: usize,
    raw_flow_cells: usize,
    demotions_permitted: usize,
    fact_repair_added: usize,
    final_flow: usize,
    final_involvement: usize,
    final_absent: usize,
    search_arcs_objective: usize,
    coverage_drawn: usize,
    coverage_chained: usize,
    coverage_target: usize,
    /// This lens's own requirement, split by assertion kind. Only OC-DECLARE supplies a
    /// second kind today, so `required_together` is what "`AS` counts" is measured by.
    required_orders: usize,
    required_together: usize,
    required_never: usize,
    required_looped: usize,
    /// Orderings of the recorded log that this lens's final flow layer still shows, drawn or
    /// chained, out of `log_orderings_total`.
    ///
    /// The only cross-lens instrument that is not apples to oranges. Each lens covers its own
    /// requirement, so a lens asserting less reduces more for free, and comparing reduction
    /// alone rewards the lens that sees least. This asks every lens the same question instead:
    /// of what the log itself orders, how much does the reduced model still show?
    ///
    /// **A lower bound, and not the paper's preservation figure.** It is
    /// [`coverage`](super::coverage), which counts drawn and chained only. Route delivery is
    /// not credited, so BPIC2017 reads 65 of 122 here where `eval_instruments` reports the
    /// pair as delivered. Read this column across lenses, never as an absolute rate.
    log_orderings_shown: usize,
    log_orderings_total: usize,
    /// The same, against what the recorded log asserts rather than the saturation.
    rec_orderings_shown: usize,
    rec_orderings_total: usize,
    /// Orderings the search shows when it may also buy the cells expansion admits.
    ///
    /// The reduction run buys recorded cells only, so the orderings the schema would
    /// reveal by writing determined participations stay potential. This runs the same
    /// search over `max.cells` instead, which is the expansion direction, and the number
    /// is realized rather than licensed. It depends on the abstraction, because the
    /// abstraction decides which of those cells the search has any reason to buy.
    exp_orderings_shown: usize,
    exp_flow: usize,
    exp_involvement: usize,
    exp_absent: usize,
    exp_expanded: usize,
    exp_ocpn_arcs: usize,
    exp_ocdfg_arcs: usize,
    /// Assertions this lens makes that do not block a cell from leaving the flow layer, and
    /// how many the final flow
    /// layer still shows. The difference is the residual an override would leave unshown.
    residual_covered: usize,
    residual_total: usize,
    ocpn_arcs_reduced: usize,
    ocpn_arcs_removed: usize,
    ocdfg_arcs_reduced: usize,
    ocdfg_arcs_removed: usize,
    /// Per recorded cell: does this lens assert something touching it?
    asserts: HashMap<Cell, bool>,
    /// Per recorded cell: its final state under this lens's repaired assignment.
    states: HashMap<Cell, CellState>,
}

struct PairwiseDisagreement {
    a: Lens,
    b: Lens,
    disagreements: usize,
    survive_into_different_state: usize,
}

struct LogReport {
    log: String,
    recorded_cells: usize,
    target_pairs: usize,
    ocpn_arcs_recorded: usize,
    ocdfg_arcs_recorded: usize,
    lenses: Vec<LensRun>,
    pairwise: Vec<PairwiseDisagreement>,
}

#[allow(clippy::too_many_arguments)]
fn run_lens(
    stem: &str,
    lens: Lens,
    abstraction: &dyn Abstraction,
    runtime_seconds: f64,
    asserts: HashMap<Cell, bool>,
    input: &SearchInput<'_>,
    grid: &CellGrid,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    locel: &SlimLinkedOCEL,
    full_facts: &Facts,
    net_recorded: &ObjectCentricPetriNet,
    dfg_recorded: &OCDirectlyFollowsGraph<'_>,
    routes: &[Route],
    recorded_bounds: &Bounds,
    input_exp: &SearchInput<'_>,
    ocpn_opts: &dyn Fn() -> ObjectCentricDiscoveryOptions,
) -> LensRun {
    let recorded_cells = grid.cells.len();
    let neutral_cells = asserts.values().filter(|v| !**v).count();
    let asserting_cells = asserts.values().filter(|v| **v).count();

    let required = abstraction.required();
    let required_orders = required.iter().filter(|s| matches!(s, Assertion::Order(..))).count();
    let required_together =
        required.iter().filter(|s| matches!(s, Assertion::Together(..))).count();
    let required_never = required.iter().filter(|s| matches!(s, Assertion::Never(..))).count();
    let required_looped = required.iter().filter(|s| matches!(s, Assertion::Looped(..))).count();

    let mut best = reflow_layer_with(input, abstraction);
    let raw_flow_cells = best.cells.len();
    let demotions_permitted = recorded_cells.saturating_sub(raw_flow_cells);
    let search_arcs_objective = best.arcs;
    let coverage_drawn = best.coverage.drawn;
    let coverage_chained = best.coverage.chained;
    let coverage_target = best.coverage.target;

    let fact_repair_added = fact_repair(grid, full_facts, &mut best.cells).len();
    if let Ok(dir) = std::env::var("KEEPSET_DIR") {
        let named: Vec<(String, String)> = best
            .cells
            .iter()
            .map(|&(a, t)| (grid.activities[a].clone(), schema.types[t].clone()))
            .collect();
        let out = format!("{dir}/{stem}.{}.keepset.json", lens.label());
        std::fs::write(&out, serde_json::to_string(&named).unwrap()).expect("write keepset");
    }
    // Same reading `examples/eval_instruments.rs` takes, so the two agree: an ordering of the
    // target counts as still shown when the flow layer draws it, when kept orderings compose to
    // it, or when a schema route delivers it object by object. Counting drawn and chained only
    // understates every abstraction alike, and on BPIC2017 it understates by 56 of 122.
    let drawn_set = drawn(&input.max.bounds, &best.cells);
    let chained = Chained::of(&drawn_set, input.n_activities);
    let mut route_pairs: HashSet<Pair> = HashSet::new();
    for r in routes {
        route_pairs.extend(asserted_of_route(&input.max.bounds, &best.cells, r));
    }
    let orderings_shown = input
        .target
        .iter()
        .filter(|p| drawn_set.contains(*p) || chained.holds(p.0, p.1) || route_pairs.contains(*p))
        .count();

    // The same reading against what the *recorded* log asserts, which is the denominator the
    // reduction claim needs. `input.target` is stated over the saturation, so on a log where
    // expansion admits cells it counts orderings that exist only because expansion wrote them.
    // The search here may buy recorded cells only, so measuring it against the saturated target
    // charges it for pairs it was never allowed to reach. BPIC2017 is the whole of the gap.
    let rec_target: Vec<Pair> = {
        let mut v: Vec<Pair> = asserted_by_type(recorded_bounds, &grid.cells)
            .into_iter()
            .flatten()
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let rec_drawn = drawn(recorded_bounds, &best.cells);
    let rec_chained = Chained::of(&rec_drawn, input.n_activities);
    let mut rec_routes: HashSet<Pair> = HashSet::new();
    for r in routes {
        rec_routes.extend(asserted_of_route(recorded_bounds, &best.cells, r));
    }
    let rec_shown = rec_target
        .iter()
        .filter(|p| {
            rec_drawn.contains(*p) || rec_chained.holds(p.0, p.1) || rec_routes.contains(*p)
        })
        .count();

    // The expansion direction under the same abstraction: the search may now buy the cells
    // `theta` admitted, so the orderings the schema licenses can actually be shown.
    let mut exp = reflow_layer_with(input_exp, abstraction);
    fact_repair(grid, full_facts, &mut exp.cells);
    let exp_drawn = drawn(&input_exp.max.bounds, &exp.cells);
    let exp_chained = Chained::of(&exp_drawn, input_exp.n_activities);
    let mut exp_routes: HashSet<Pair> = HashSet::new();
    for r in routes {
        exp_routes.extend(asserted_of_route(&input_exp.max.bounds, &exp.cells, r));
    }
    let exp_orderings_shown = input_exp
        .target
        .iter()
        .filter(|p| {
            exp_drawn.contains(*p) || exp_chained.holds(p.0, p.1) || exp_routes.contains(*p)
        })
        .count();
    let exp_assignment = assign(grid, &exp.cells, &Default::default());
    let (exp_flow, exp_involvement, exp_absent, exp_expanded) = exp_assignment.tally();
    let added_exp: Vec<_> = input_exp
        .max
        .written
        .iter()
        .filter(|(e, o)| {
            let a = acts.act_of[e.get_ev(locel).event_type];
            let t = schema.type_of[o];
            exp.cells.contains(&(a, t)) && !grid.cells.contains(&(a, t))
        })
        .copied()
        .collect();
    let exp_log = flow_projection(&tag(locel, schema, acts, &exp.cells, &added_exp)).into_owned();
    let exp_ocpn_arcs = ocpn_arcs(&discover_ocpn(&exp_log, ocpn_opts()));
    let exp_ocdfg_arcs = ocdfg_arcs(&discover_dfg_from_ocel(&exp_log));
    let (residual_covered, residual_total) = abstraction.residual(&best.cells, input.n_activities);
    let assignment = assign(grid, &best.cells, &Default::default());
    let (final_flow, final_involvement, final_absent, _expanded) = assignment.tally();

    let mut states: HashMap<Cell, CellState> = HashMap::new();
    for &c in &grid.cells {
        if let Some(s) = assignment.state_of(c) {
            states.insert(c, s);
        }
    }

    let reduced_log = flow_projection(&tag(locel, schema, acts, &best.cells, &[])).into_owned();
    let net_reduced = discover_ocpn(&reduced_log, ocpn_opts());
    let ocpn_arcs_reduced = ocpn_arcs(&net_reduced);
    let ocpn_arcs_removed = ocpn_arcs(net_recorded).saturating_sub(ocpn_arcs_reduced);

    let dfg_reduced = discover_dfg_from_ocel(&reduced_log);
    let ocdfg_arcs_reduced = ocdfg_arcs(&dfg_reduced);
    let ocdfg_arcs_removed = ocdfg_arcs(dfg_recorded).saturating_sub(ocdfg_arcs_reduced);

    LensRun {
        lens,
        runtime_seconds,
        neutral_cells,
        asserting_cells,
        raw_flow_cells,
        demotions_permitted,
        fact_repair_added,
        final_flow,
        final_involvement,
        final_absent,
        search_arcs_objective,
        coverage_drawn,
        coverage_chained,
        coverage_target,
        required_orders,
        required_together,
        required_never,
        required_looped,
        log_orderings_shown: orderings_shown,
        log_orderings_total: input.target.len(),
        rec_orderings_shown: rec_shown,
        rec_orderings_total: rec_target.len(),
        exp_orderings_shown,
        exp_flow,
        exp_involvement,
        exp_absent,
        exp_expanded,
        exp_ocpn_arcs,
        exp_ocdfg_arcs,
        residual_covered,
        residual_total,
        ocpn_arcs_reduced,
        ocpn_arcs_removed,
        ocdfg_arcs_reduced,
        ocdfg_arcs_removed,
        asserts,
        states,
    }
}

fn run(path: &str) -> LogReport {
    let stem = log_stem(path);
    println!("\n{}\n{stem}\n{}", "=".repeat(72), "=".repeat(72));

    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);

    // Schema discovery, shared verbatim across the three lenses.
    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);
    let bounds = Bounds::build(&locel, &schema, &acts); // recorded bounds, for the log lens's own neutral-cell reading
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
    let target_pairs = target.len();
    let variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
    let objects = Saturation::objects_per_type(&schema);
    // Recorded cells only, matching `ocpn_quality.rs`/`eval_instruments.rs`'s default run: the
    // search never buys a cell the extraction did not record.
    let allowed = grid.cells.clone();
    // The expansion direction differs in exactly one field: the search may also buy the cells
    // `theta` admitted that are novel, i.e. carry participations the recorded log does not.
    let mut allowed_exp = grid.cells.clone();
    allowed_exp.extend(novel_cells(&novelty(&locel, &acts, &bounds, &grid.cells, &max)).keys().copied());
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
    let input_exp = SearchInput {
        max: &max,
        variants: &variants,
        allowed: &allowed_exp,
        recorded: &grid.cells,
        target: &target,
        objects_per_type: &objects,
        rep: &closure.rep,
        types: &schema.types,
        n_activities: grid.activities.len(),
    };

    // `fact_repair` is log-based for every abstraction: it is not the search being
    // swapped, it is the correctness floor underneath it, same as `ocpn_quality.rs` applies
    // regardless of strategy.
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);

    let ocpn_opts = || ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(IMF_THRESHOLD));
    let net_recorded = discover_ocpn(&locel, ocpn_opts());
    let ocpn_arcs_recorded = ocpn_arcs(&net_recorded);
    let dfg_recorded = discover_dfg_from_ocel(&locel);
    let ocdfg_arcs_recorded = ocdfg_arcs(&dfg_recorded);

    let mut lenses = Vec::new();

    // --- log lens: today's default, AssertionSource::Log over max's own bounds. ---
    {
        let mut asserts: HashMap<Cell, bool> = HashMap::new();
        for t in 0..schema.types.len() {
            let activities: BTreeSet<ActivityIndex> =
                grid.cells.iter().filter(|(_, tt)| *tt == t).map(|(a, _)| *a).collect();
            if activities.is_empty() {
                continue;
            }
            let lpairs = asserted_of_type(&bounds, &grid.cells, t);
            let silent = silent_of(&activities, &lpairs);
            for a in &activities {
                asserts.insert((*a, t), !silent.contains(a));
            }
        }
        let abstraction = LogAbstraction::over(&max.bounds, &max.cells);
        let lr = run_lens(
            &stem,
            Lens::Log,
            &abstraction,
            0.0, // shared discovery work (bounds), not a lens-specific cost
            asserts,
            &input,
            &grid,
            &schema,
            &acts,
            &locel,
            &full_facts,
            &net_recorded,
            &dfg_recorded,
            &routes,
            &bounds,
            &input_exp,
            &ocpn_opts,
        );
        // The default path must be behaviourally identical to the unparameterised code: the
        // exact per-cell states this lens's search produced, checked against the file
        // `ocpn_quality.rs` wrote before this parameterisation existed.
        verify_default_path(&stem, &lr.states, &grid, &schema);
        lenses.push(lr);
    }

    // --- model lens: per-type IMf trees at the discovery preset, every cut an assertion. ---
    {
        let t0 = Instant::now();
        let models = model_abstraction_with(
            &locel,
            &schema,
            &acts,
            &grid.cells,
            InductiveMinerOptions::imf(IMF_THRESHOLD),
        );
        let runtime_seconds = t0.elapsed().as_secs_f64();

        let mut asserts: HashMap<Cell, bool> = HashMap::new();
        for m in &models {
            let neutral = m.neutral();
            for a in &m.activities {
                asserts.insert((*a, m.object_type), !neutral.contains(a));
            }
        }
        let abstraction = TreeAbstraction::over(&models, &max.cells);
        lenses.push(run_lens(
            &stem,
            Lens::Model,
            &abstraction,
            runtime_seconds,
            asserts,
            &input,
            &grid,
            &schema,
            &acts,
            &locel,
            &full_facts,
            &net_recorded,
            &dfg_recorded,
            &routes,
            &bounds,
            &input_exp,
            &ocpn_opts,
        ));
    }

    // --- declare lens: per-type OC-DECLARE discovery. ---
    {
        let t0 = Instant::now();
        let dmodels = declare_abstraction(&locel, &schema, &acts, &grid.cells);
        let runtime_seconds = t0.elapsed().as_secs_f64();

        let mut per_type: Vec<HashSet<Pair>> = vec![HashSet::new(); schema.types.len()];
        let mut asserts: HashMap<Cell, bool> = HashMap::new();
        for m in &dmodels {
            per_type[m.object_type] = m.ordered.clone();
            let neutral = m.neutral();
            for a in &m.activities {
                asserts.insert((*a, m.object_type), !neutral.contains(a));
            }
        }
        let abstraction = DeclareAbstraction::strict(&dmodels, &max.cells);
        lenses.push(run_lens(
            &stem,
            Lens::Declare,
            &abstraction,
            runtime_seconds,
            asserts.clone(),
            &input,
            &grid,
            &schema,
            &acts,
            &locel,
            &full_facts,
            &net_recorded,
            &dfg_recorded,
            &routes,
            &bounds,
            &input_exp,
            &ocpn_opts,
        ));

        // The same discovery, priced with coexistence dropped from blocking to reported.
        let reported = DeclareAbstraction::over(&dmodels, &max.cells);
        lenses.push(run_lens(
            &stem,
            Lens::DeclareReported,
            &reported,
            runtime_seconds,
            asserts,
            &input,
            &grid,
            &schema,
            &acts,
            &locel,
            &full_facts,
            &net_recorded,
            &dfg_recorded,
            &routes,
            &bounds,
            &input_exp,
            &ocpn_opts,
        ));
    }

    // Cell-level verdict disagreements, and how many of them land in a different final state.
    let by_lens: HashMap<Lens, &LensRun> = lenses.iter().map(|l| (l.lens, l)).collect();
    let mut pairwise = Vec::new();
    for &(a, b) in &[
        (Lens::Log, Lens::Model),
        (Lens::Log, Lens::Declare),
        (Lens::Model, Lens::Declare),
    ] {
        let (la, lb) = (by_lens[&a], by_lens[&b]);
        let mut disagreements = 0usize;
        let mut survive = 0usize;
        for &c in &grid.cells {
            let (va, vb) = (la.asserts.get(&c), lb.asserts.get(&c));
            if let (Some(va), Some(vb)) = (va, vb) {
                if va != vb {
                    disagreements += 1;
                    if la.states.get(&c) != lb.states.get(&c) {
                        survive += 1;
                    }
                }
            }
        }
        pairwise.push(PairwiseDisagreement {
            a,
            b,
            disagreements,
            survive_into_different_state: survive,
        });
    }

    print_log_summary(&stem, grid.cells.len(), target_pairs, &lenses, &pairwise);

    LogReport {
        log: stem,
        recorded_cells: grid.cells.len(),
        target_pairs,
        ocpn_arcs_recorded,
        ocdfg_arcs_recorded,
        lenses,
        pairwise,
    }
}

fn print_log_summary(
    _stem: &str,
    recorded_cells: usize,
    target_pairs: usize,
    lenses: &[LensRun],
    pairwise: &[PairwiseDisagreement],
) {
    println!(
        "  recorded cells {recorded_cells}   target pairs {target_pairs}"
    );
    println!(
        "  {:<10}{:>9}{:>9}{:>9}{:>12}{:>7}{:>7}{:>7}{:>9}{:>9}{:>8}{:>8}{:>9}{:>9}{:>11}{:>9}{:>12}",
        "lens", "neutral", "assert", "raw_flow", "nonflow", "flow", "inv", "implied",
        "ocpn_rmv", "dfg_rmv", "req_ord", "req_tog", "cov_d", "cov_c", "vs_log", "resid",
        "runtime(s)"
    );
    for l in lenses {
        println!(
            "  {:<10}{:>9}{:>9}{:>9}{:>12}{:>7}{:>7}{:>7}{:>9}{:>9}{:>8}{:>8}{:>9}{:>9}{:>11}{:>9}{:>12.4}",
            l.lens.label(),
            l.neutral_cells,
            l.asserting_cells,
            l.raw_flow_cells,
            l.demotions_permitted,
            l.final_flow,
            l.final_involvement,
            l.final_absent,
            l.ocpn_arcs_removed,
            l.ocdfg_arcs_removed,
            l.required_orders,
            l.required_together,
            l.coverage_drawn,
            l.coverage_chained,
            format!("{}/{}", l.log_orderings_shown, l.log_orderings_total),
            format!("{}/{}", l.residual_covered, l.residual_total),
            l.runtime_seconds
        );
    }
    for p in pairwise {
        println!(
            "  disagreements {} vs {}: {}  (survive into a different final state: {})",
            p.a.label(),
            p.b.label(),
            p.disagreements,
            p.survive_into_different_state
        );
    }
}

fn lens_json(l: &LensRun) -> serde_json::Value {
    // The keys below are the stored format of `results/stats/cross_instantiation.json`,
    // which the paper's tables read: `demotions_permitted` counts the cells the coverage step
    // lets leave the flow layer, and `absent` counts the implied ones.
    serde_json::json!({
        "lens": l.lens.label(),
        "runtime_seconds": l.runtime_seconds,
        "neutral_cells": l.neutral_cells,
        "asserting_cells": l.asserting_cells,
        "raw_flow_cells": l.raw_flow_cells,
        "demotions_permitted": l.demotions_permitted,
        "fact_repair_added": l.fact_repair_added,
        "final_state": {
            "flow": l.final_flow,
            "involvement": l.final_involvement,
            "absent": l.final_absent,
        },
        "search_arcs_objective": l.search_arcs_objective,
        "coverage": {
            "drawn": l.coverage_drawn,
            "chained": l.coverage_chained,
            "target": l.coverage_target,
        },
        "required": {
            "total": l.required_orders + l.required_together + l.required_never + l.required_looped,
            "orders": l.required_orders,
            "together": l.required_together,
            "never": l.required_never,
            "looped": l.required_looped,
        },
        "against_saturated_target": {
            "orderings_shown": l.log_orderings_shown,
            "orderings_total": l.log_orderings_total,
        },
        "against_recorded_log": {
            "orderings_shown": l.rec_orderings_shown,
            "orderings_total": l.rec_orderings_total,
        },
        "under_expansion": {
            "orderings_shown": l.exp_orderings_shown,
            "final_state": {"flow": l.exp_flow, "involvement": l.exp_involvement, "absent": l.exp_absent, "expanded": l.exp_expanded},
            "ocpn_arcs": l.exp_ocpn_arcs,
            "ocdfg_arcs": l.exp_ocdfg_arcs,
        },
        "reported_not_blocking": {
            "shown": l.residual_covered,
            "total": l.residual_total,
        },
        "ocpn_arcs": {
            "reduced": l.ocpn_arcs_reduced,
            "removed": l.ocpn_arcs_removed,
        },
        "ocdfg_arcs": {
            "reduced": l.ocdfg_arcs_reduced,
            "removed": l.ocdfg_arcs_removed,
        },
    })
}

fn log_json(r: &LogReport) -> serde_json::Value {
    let lenses: Vec<serde_json::Value> = r.lenses.iter().map(lens_json).collect();
    let pairwise: Vec<serde_json::Value> = r
        .pairwise
        .iter()
        .map(|p| {
            serde_json::json!({
                "a": p.a.label(),
                "b": p.b.label(),
                "cell_verdict_disagreements": p.disagreements,
                "survive_into_different_final_state": p.survive_into_different_state,
            })
        })
        .collect();
    let total_disagreements: usize = r.pairwise.iter().map(|p| p.disagreements).sum();
    let total_survive: usize = r.pairwise.iter().map(|p| p.survive_into_different_state).sum();
    serde_json::json!({
        "log": r.log,
        "recorded_cells": r.recorded_cells,
        "target_pairs": r.target_pairs,
        "ocpn_arcs_recorded": r.ocpn_arcs_recorded,
        "ocdfg_arcs_recorded": r.ocdfg_arcs_recorded,
        "keepset_check": "ok",
        "lenses": lenses,
        "pairwise_disagreements": pairwise,
        "pairwise_disagreements_total": {
            "cell_verdict_disagreements": total_disagreements,
            "survive_into_different_final_state": total_survive,
        },
    })
}

fn main() {
    let paths = corpus::logs_or_args();

    let reports: Vec<LogReport> = paths.iter().map(|p| run(p)).collect();

    let corpus_totals = serde_json::json!({
        "cell_verdict_disagreements": reports.iter().flat_map(|r| &r.pairwise).map(|p| p.disagreements).sum::<usize>(),
        "survive_into_different_final_state": reports.iter().flat_map(|r| &r.pairwise).map(|p| p.survive_into_different_state).sum::<usize>(),
    });

    let logs: Vec<serde_json::Value> = reports.iter().map(log_json).collect();
    let doc = serde_json::json!({
        "corpus": paths.iter().map(|p| corpus::label(p)).collect::<Vec<_>>(),
        "imf_threshold_for_ocpn_discovery": IMF_THRESHOLD,
        "model_lens": "tree constructs at the discovery preset (orders, never, looped); exclusions and alternations read above concurrent cuts only",
        "logs": logs,
        "corpus_totals": corpus_totals,
    });

    println!("\n{}\ncorpus totals\n{}", "=".repeat(72), "=".repeat(72));
    println!("{}", serde_json::to_string_pretty(&corpus_totals).unwrap());

    let stats = corpus::stats_dir();
    std::fs::create_dir_all(&stats).expect("create stats dir");
    let out_path = std::env::var("CROSS_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| stats.join("cross_instantiation.json"));
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
