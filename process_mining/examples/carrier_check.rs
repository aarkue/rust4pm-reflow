//! Who actually carries a non-flow type's orderings.
//!
//! Aggregate coverage is satisfied when *any* kept type orders the pair, related to the
//! non-flow type or not. That lets a cell leave the flow layer on the strength of a
//! coincidence: two
//! unrelated types happening to order the same two activities. This asks, per per-type
//! ordering fact `(T, x, y)`, which of four things is true after the reduction.
//!
//! - **own_drawn**: `T` still flows at `x` and `y`, so `T` draws its own fact.
//! - **own_readable**: `T` is kept at both, at least one at `inv`, so the fact is computable
//!   from the marked log but no arc of `T` shows it.
//! - **map**: `T` does not carry it, and a schema route relates objects of a kept type at `x`
//!   to objects of a kept type at `y`, so something object-level stands behind the pair.
//! - **coincidence**: none of the above, and the pair survives only because some unrelated
//!   kept type orders the same two activities.
//!
//! The last bucket is the one worth knowing the size of: it is where "the ordering is
//! preserved" means only that the activity pair is still ordered somewhere in the model.
//!
//! Usage: `cargo run --release --example carrier_check -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{collections::HashSet, path::PathBuf};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_of_route, assign, drawn, fact_repair, facts_from, handoff,
        ActivityIndexing, Bounds, Cell, CellGrid, Chained, ExpansionDirection, Pair, Saturation,
        SchemaClosure, SearchInput, StructuralSchema, TraceVariants, DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

fn main() {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let (mut c_own, mut c_read, mut c_map, mut c_coin) = (0usize, 0usize, 0usize, 0usize);
    println!(
        "{:<20} {:>7} {:>10} {:>13} {:>6} {:>13}",
        "log", "facts", "own_drawn", "own_readable", "map", "coincidence"
    );

    for path in &corpus::logs_or_args() {
        let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
        let locel = SlimLinkedOCEL::from_ocel(ocel);
        let schema = StructuralSchema::discover(&locel);
        let closure = SchemaClosure::build(&locel, &schema);
        let grid = CellGrid::build(&locel, &schema, &closure);
        let acts = ActivityIndexing::build(&locel, &grid);
        let bounds = Bounds::build(&locel, &schema, &acts);
        let routes = agreed_routes(&schema).0;
        let max = Saturation::build(
            &locel, &schema, &grid, &acts, &routes, &bounds, DEFAULT_THETA,
            ExpansionDirection::default(),
        );

        let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
        target.sort_unstable();
        let variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
        let objects = Saturation::objects_per_type(&schema);
        let allowed = grid.cells.clone();
        let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
        let mut best = handoff(&SearchInput {
            max: &max,
            variants: &variants,
            allowed: &allowed,
            recorded: &grid.cells,
            target: &target,
            objects_per_type: &objects,
            rep: &closure.rep,
            types: &schema.types,
            n_activities: grid.activities.len(),
        });
        fact_repair(&grid, &full_facts, &mut best.cells);

        let a = assign(&grid, &best.cells, &HashSet::new());
        let flow: HashSet<Cell> = a.flow.iter().copied().collect();
        let readable: HashSet<Cell> = a.flow.iter().chain(a.involvement.iter()).copied().collect();

        // What a schema route delivers object-to-object over the kept layer.
        let mut route_pairs: HashSet<Pair> = HashSet::new();
        for r in &routes {
            route_pairs.extend(asserted_of_route(&max.bounds, &flow, r));
        }
        let drawn_set = drawn(&max.bounds, &flow);
        let chained = Chained::of(&drawn_set, grid.activities.len());

        let (mut own, mut read, mut map, mut coin) = (0usize, 0usize, 0usize, 0usize);
        let mut examples: Vec<String> = Vec::new();
        for (t, x, y) in &full_facts.ordering {
            if flow.contains(&(*x, *t)) && flow.contains(&(*y, *t)) {
                own += 1;
            } else if readable.contains(&(*x, *t)) && readable.contains(&(*y, *t)) {
                read += 1;
            } else if route_pairs.contains(&(*x, *y)) {
                map += 1;
            } else {
                coin += 1;
                if examples.len() < 4 {
                    let by: Vec<&str> = (0..schema.types.len())
                        .filter(|u| {
                            flow.contains(&(*x, *u))
                                && flow.contains(&(*y, *u))
                                && drawn(&max.bounds, &flow).contains(&(*x, *y))
                        })
                        .map(|u| schema.types[u].as_str())
                        .collect();
                    examples.push(format!(
                        "{} : {} < {}  [shown by {}{}]",
                        schema.types[*t],
                        grid.activities[*x],
                        grid.activities[*y],
                        if by.is_empty() { "chaining only".to_string() } else { by.join(", ") },
                        if chained.holds(*x, *y) && by.is_empty() { "" } else { "" }
                    ));
                }
            }
        }

        let stem = log_stem(path);
        println!(
            "{:<20} {:>7} {:>10} {:>13} {:>6} {:>13}",
            stem,
            full_facts.ordering.len(),
            own,
            read,
            map,
            coin
        );
        for e in &examples {
            println!("      {e}");
        }
        c_own += own;
        c_read += read;
        c_map += map;
        c_coin += coin;
        rows.push(serde_json::json!({
            "log": stem,
            "per_type_ordering_facts": full_facts.ordering.len(),
            "own_drawn": own,
            "own_readable_not_drawn": read,
            "map_backed": map,
            "coincidence_only": coin,
            "examples": examples,
        }));
    }

    println!(
        "\n{:<20} {:>7} {:>10} {:>13} {:>6} {:>13}",
        "CORPUS",
        c_own + c_read + c_map + c_coin,
        c_own,
        c_read,
        c_map,
        c_coin
    );

    let doc = serde_json::json!({ "logs": rows });
    let out_path = corpus::stats_dir().join("carrier_check.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).expect("serialize"))
        .expect("write stats json");
    println!("\nstats written to {}", out_path.display());
}
