//! One log under the three IMf readings: the log lens, the orderings of the per-type trees
//! only (`PairAbstraction`, the pre-2026-09-02 `imf` column) and `TreeAbstraction`, whose XOR
//! and loop cuts block a cell from leaving the flow layer too (the paper's `imf` column).
//! Prints every type's tree and
//! assertions, the state counts and arcs per lens, and the cells whose state differs.
//! `KEEPSET_DIR=<dir>` also writes each lens's flow layer for `ocpn_quality`'s
//! `KEEPSET_OVERRIDE`.
//!
//! Usage: `cargo run --release --example tree_abstraction -- [<log>]` (defaults to Order Management
//! in the `REFLOW_LOGS` corpus directory).

mod corpus;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::PathBuf,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, assign, fact_repair, facts_from, flow_projection, model_abstraction, tag,
        reflow_layer_with, Abstraction, ActivityIndexing, Assertion, Bounds, Cell, CellGrid,
        CellState, ExpansionDirection, LogAbstraction, Pair, PairAbstraction, Saturation,
        SchemaClosure, SearchInput, StructuralSchema, TraceVariants, TreeAbstraction,
        DEFAULT_THETA, MODEL_NOISE_THRESHOLD,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    core::process_models::object_centric::{
        ocdfg::discover_dfg_from_ocel, ocpn::ObjectCentricPetriNet,
    },
    discovery::case_centric::inductive_miner::InductiveMinerOptions,
    discovery::object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions},
    Importable,
};

const IMF_THRESHOLD: f64 = 0.2;

fn ocpn_arcs(net: &ObjectCentricPetriNet) -> usize {
    let mut net = net.clone();
    for component in net.nets.values_mut() {
        component.simplify_silent();
    }
    net.nets.values().map(|n| n.arcs.len()).sum()
}

fn show(s: Assertion, acts: &[String]) -> String {
    match s {
        Assertion::Order(x, y) => format!("{} < {}", acts[x], acts[y]),
        Assertion::Never(x, y) => format!("{} xor {}", acts[x], acts[y]),
        Assertion::Looped(x, y) => format!("{} loop {}", acts[x], acts[y]),
        Assertion::Together(x, y) => format!("{} with {}", acts[x], acts[y]),
        Assertion::Repeats(a) => format!("{} repeats", acts[a]),
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        corpus::logs_dir().join("order-management.xml").to_string_lossy().into_owned()
    });
    let locel = SlimLinkedOCEL::import_from_path(PathBuf::from(&path)).expect("import log");

    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);
    let bounds = Bounds::build(&locel, &schema, &acts);
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
    let variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
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
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
    let ocpn_opts = || ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(IMF_THRESHOLD));
    let net_recorded = discover_ocpn(&locel, ocpn_opts());
    let rec_ocpn = ocpn_arcs(&net_recorded);
    let rec_dfg: usize = discover_dfg_from_ocel(&locel)
        .object_type_to_dfg
        .values()
        .map(|d| d.directly_follows_relations.len())
        .sum();
    println!("{path}\n  {} cells, recorded OCPN {rec_ocpn} arcs, OC-DFG {rec_dfg} arcs", grid.cells.len());

    let models = model_abstraction(&locel, &schema, &acts, &grid.cells, MODEL_NOISE_THRESHOLD);
    println!("\nper-type trees and what they assert");
    for m in &models {
        println!("\n  {}: {}", schema.types[m.object_type], m.tree);
        let mut by_kind: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for s in m.assertions() {
            let k = match s {
                Assertion::Order(..) => "order",
                Assertion::Never(..) => "xor",
                Assertion::Looped(..) => "loop",
                _ => "other",
            };
            by_kind.entry(k).or_default().push(show(s, &acts.activities));
        }
        for (k, mut v) in by_kind {
            v.sort();
            println!("    {k:5} {}", v.join(", "));
        }
        let neutral: Vec<&str> = m.neutral().iter().map(|a| acts.activities[*a].as_str()).collect();
        println!("    neutral: {}", if neutral.is_empty() { "-".to_string() } else { neutral.join(", ") });
    }

    let per_type: Vec<HashSet<Pair>> = {
        let mut v = vec![HashSet::new(); schema.types.len()];
        for m in &models {
            v[m.object_type] = m.ordered.clone();
        }
        v
    };
    let log_lens = LogAbstraction::over(&max.bounds, &max.cells);
    let pair_lens = PairAbstraction::over(&per_type, &max.cells);
    let tree_lens = TreeAbstraction::over(&models, &max.cells);
    let lenses: Vec<(&str, &dyn Abstraction)> =
        vec![("log", &log_lens), ("imf-pairs", &pair_lens), ("imf-tree", &tree_lens)];

    let mut states_by_lens: Vec<(&str, HashMap<Cell, CellState>)> = Vec::new();
    println!("\nlens        required  flow  inv  abs   OCPN arcs   OC-DFG arcs");
    for (name, lens) in &lenses {
        let mut best = reflow_layer_with(&input, *lens);
        fact_repair(&grid, &full_facts, &mut best.cells);
        let assignment = assign(&grid, &best.cells, &Default::default());
        let (flow, inv, abs, _) = assignment.tally();
        let reduced = flow_projection(&tag(&locel, &schema, &acts, &best.cells, &[])).into_owned();
        let arcs = ocpn_arcs(&discover_ocpn(&reduced, ocpn_opts()));
        let dfg: usize = discover_dfg_from_ocel(&reduced)
            .object_type_to_dfg
            .values()
            .map(|d| d.directly_follows_relations.len())
            .sum();
        let mut states = HashMap::new();
        for &c in &grid.cells {
            if let Some(s) = assignment.state_of(c) {
                states.insert(c, s);
            }
        }
        println!(
            "{name:10}  {:8}  {flow:4}  {inv:3}  {abs:3}   {arcs:9}   {dfg:11}",
            lens.required().len()
        );
        if let Ok(dir) = std::env::var("KEEPSET_DIR") {
            let stem = std::path::Path::new(&path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let named: Vec<(String, String)> = best
                .cells
                .iter()
                .map(|&(a, t)| (grid.activities[a].clone(), schema.types[t].clone()))
                .collect();
            let out = format!("{dir}/{stem}.{name}.keepset.json");
            std::fs::write(&out, serde_json::to_string(&named).unwrap()).expect("write keepset");
        }
        states_by_lens.push((name, states));
    }

    println!("\ncells whose state differs between lenses (activity, type: log / imf-pairs / imf-tree)");
    let mut cells: Vec<Cell> = grid.cells.iter().copied().collect();
    cells.sort_by_key(|&(a, t)| (t, a));
    let label = |s: Option<&CellState>| match s {
        Some(CellState::Flow) => "flow",
        Some(CellState::Involvement) => "inv",
        Some(CellState::Implied) => "imp",
        _ => "?",
    };
    let mut seen_types: BTreeSet<usize> = BTreeSet::new();
    for c in cells {
        let ss: Vec<&str> = states_by_lens.iter().map(|(_, m)| label(m.get(&c))).collect();
        if ss.iter().all(|s| *s == ss[0]) {
            continue;
        }
        if seen_types.insert(c.1) {
            println!("  {}", schema.types[c.1]);
        }
        println!("    {:32} {}", grid.activities[c.0], ss.join(" / "));
    }
}
