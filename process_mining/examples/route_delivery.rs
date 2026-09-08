//! Does composing along the schema map deliver what activity-level chaining delivers,
//! without inventing what it invents?
//!
//! Three readings of "the reduced model delivers `x < y`", scored against the same target:
//!
//! | column | composition | sound? |
//! |---|---|---|
//! | drawn | none: one kept type orders the pair and is kept at both ends | yes, by projection |
//! | chained | transitive closure of drawn over activities | no -- measured here |
//! | route | one kept type orders `x`, a map carries its objects to a kept type at `y` | measured here |
//!
//! The ground truth is the facts of the log: the pairs some type orders, **plus** the pairs
//! some route orders, both taken over every cell of `max`. Route facts belong in it because
//! they are facts -- "every item at `pick item` has its package delivered later" is a
//! statement about the log, not an inference off a picture -- and scoring route delivery
//! against a single-type ground truth would count them as invention.
//!
//! Usage: `cargo run --release --example route_delivery -- <log> ...`
//! Env: `THETA`, `SEARCHES=1` to also score the keep-sets the searches build, `SHOW=n` to
//! print n invented pairs per reading.

use std::{
    collections::HashSet,
    env,
    path::PathBuf,
    time::Instant,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, asserted_of_route, best_of_two, novel_cells, novelty,
        route_delivered, ActivityIndexing, Bounds, Cell, CellGrid, Chained, ExpansionDirection,
        Pair, Route, Saturation, SchemaClosure, SearchInput, StructuralSchema, TraceVariants,
        DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};

/// The pairs every route asserts under a keep-set, without the single-type ones.
fn route_only(bounds: &Bounds, kept: &HashSet<Cell>, routes: &[Route]) -> HashSet<Pair> {
    let mut out: HashSet<Pair> = HashSet::new();
    for r in routes {
        out.extend(asserted_of_route(bounds, kept, r));
    }
    out
}

fn pct(part: usize, whole: usize) -> String {
    if whole == 0 {
        return "  n/a".into();
    }
    format!("{:5.1}%", 100.0 * part as f64 / whole as f64)
}

fn show_some(label: &str, ps: &HashSet<Pair>, names: &[String], show: usize) {
    if show == 0 || ps.is_empty() {
        return;
    }
    let mut v: Vec<Pair> = ps.iter().copied().collect();
    v.sort_unstable();
    for (x, y) in v.iter().take(show) {
        println!("      {label:<10} {} -> {}", names[*x], names[*y]);
    }
}

fn run(path: &str) {
    let t0 = Instant::now();
    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);
    let schema = StructuralSchema::discover(&locel);
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let acts = ActivityIndexing::build(&locel, &grid);
    let bounds = Bounds::build(&locel, &schema, &acts);
    let theta: f64 = env::var("THETA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THETA);
    let show: usize = env::var("SHOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (routes, _clashes) = agreed_routes(&schema);
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
    let n = grid.activities.len();
    let names = &grid.activities;

    println!("\n{}\n{path}\n{}", "=".repeat(78), "=".repeat(78));
    println!(
        "{n} activities, {} types, {} routes, {} cells recorded, {} in max  ({:.1}s)",
        schema.types.len(),
        routes.len(),
        grid.cells.len(),
        max.cells.len(),
        t0.elapsed().as_secs_f64()
    );

    // The facts of the log, over every cell of `max`: what a type orders, plus what a route
    // orders. This is the ground truth both compositions are scored against.
    let target: HashSet<Pair> = max.target_pairs();
    let route_facts = route_only(&max.bounds, &max.cells, &routes);
    let facts: HashSet<Pair> = target.union(&route_facts).copied().collect();
    println!(
        "facts: {} ordered by a type, {} by a route ({} of those by no type) => {} total",
        target.len(),
        route_facts.len(),
        route_facts.difference(&target).count(),
        facts.len()
    );

    let score = |label: &str, kept: &HashSet<Cell>| {
        let drawn: HashSet<Pair> = asserted_by_type(&max.bounds, kept)
            .into_iter()
            .flatten()
            .collect();
        let route = route_delivered(&max.bounds, kept, &routes);
        let ch = Chained::of(&drawn, n);

        let inv_route: HashSet<Pair> = route.difference(&facts).copied().collect();
        let mut inv_chain: HashSet<Pair> = HashSet::new();
        for x in 0..n {
            for y in 0..n {
                if ch.holds(x, y) && !facts.contains(&(x, y)) {
                    inv_chain.insert((x, y));
                }
            }
        }
        let chained_hits = target.iter().filter(|p| ch.holds(p.0, p.1)).count();

        println!(
            "  {label:<20}{:>5} cells | drawn {:>7} {} | chained {:>7} {} inv {:>8} \
             | route {:>7} {} inv {:>8}",
            kept.len(),
            drawn.intersection(&target).count(),
            pct(drawn.intersection(&target).count(), target.len()),
            chained_hits,
            pct(chained_hits, target.len()),
            inv_chain.len(),
            route.intersection(&target).count(),
            pct(route.intersection(&target).count(), target.len()),
            inv_route.len(),
        );
        show_some("chain-inv", &inv_chain, names, show);
        show_some("route-inv", &inv_route, names, show);
    };

    println!("\nscored against {} target pairs:", target.len());
    score("recorded", &grid.cells);
    score("max", &max.cells);

    if env::var("SEARCHES").is_err() {
        return;
    }

    let mut target_v: Vec<Pair> = target.iter().copied().collect();
    target_v.sort_unstable();
    let sat_variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
    let objects = Saturation::objects_per_type(&schema);
    let novel = novel_cells(&novelty(&locel, &acts, &bounds, &grid.cells, &max));
    let mut allowed = grid.cells.clone();
    allowed.extend(novel.keys().copied());
    let input = SearchInput {
        max: &max,
        variants: &sat_variants,
        allowed: &allowed,
        recorded: &grid.cells,
        target: &target_v,
        objects_per_type: &objects,
        rep: &closure.rep,
        types: &schema.types,
        n_activities: n,
    };
    let (_best, both) = best_of_two(&input);
    for f in &both {
        if !f.ran {
            println!("  {:<20} SKIPPED (over budget)", f.strategy.label());
            continue;
        }
        score(f.strategy.label(), &f.cells);
    }
}

fn main() {
    for path in env::args().skip(1) {
        run(&path);
    }
}
