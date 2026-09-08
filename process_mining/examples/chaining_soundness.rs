//! Is the chained closure of the drawn relation contained in the ordering relation?
//!
//! Coverage is scored on `Chained`: the drawn pairs closed under composition, forgetting
//! which type supplied each hop. That direction is a *completeness* measure -- every pair
//! some type orders is delivered. This example measures the other direction, which nothing
//! in the coverage check: how many pairs the closure delivers that no type orders, and
//! what the log says about them.
//!
//! Four outcomes per invented pair, and they are not equally bad:
//!
//! | class | meaning |
//! |---|---|
//! | self | `(x, x)`: the closure ran round a cycle. Never an ordering fact |
//! | reversal | some type orders `(y, x)`. The picture asserts the opposite of a fact |
//! | witnessed | some object does `x` before `y`, but another does the reverse, so no type orders it. The closure picks a side of a genuine parallelism |
//! | tie | objects take part in both, none strictly before the other |
//! | unwitnessed | no object of any type takes part in both. The pair is a *role* fact |
//!
//! Usage: `cargo run --release --example chaining_soundness -- <log> ...`
//! Env: `THETA` (saturation admission rate), `SEARCHES=1` to also measure the keep-sets
//! the two searches build, `SHOW=n` to print n examples per class.

use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    time::Instant,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, best_of_two, novel_cells, novelty, ActivityIndexing,
        Bounds, Cell, CellGrid, Chained, ExpansionDirection, Pair, Saturation, SchemaClosure,
        SearchInput, StructuralSchema, TraceVariants, DEFAULT_THETA,
    },
    core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
    Importable, OCEL,
};

/// The raw witness relation and the co-participation relation, under a keep-set.
///
/// `asserted_of_type` throws both away -- it returns the strict orderings only -- and the
/// classification of an invented pair needs them: a pair no object takes part in both ends
/// of is a different failure from one two objects witness in opposite directions.
fn witnesses(bounds: &Bounds, kept: &HashSet<Cell>) -> (HashSet<Pair>, HashSet<Pair>) {
    let mut ef: HashSet<Pair> = HashSet::new();
    let mut co: HashSet<Pair> = HashSet::new();
    let mut live: Vec<(usize, i64, i64)> = Vec::new();
    for (t, objs) in bounds.per_type.iter().enumerate() {
        for ob in objs {
            live.clear();
            live.extend(ob.at.iter().filter(|(a, _, _)| kept.contains(&(*a, t))));
            for (x, xmin, _) in &live {
                for (y, _, ymax) in &live {
                    if x == y {
                        continue;
                    }
                    co.insert((*x, *y));
                    if *xmin < *ymax {
                        ef.insert((*x, *y));
                    }
                }
            }
        }
    }
    (ef, co)
}

struct Classified {
    total: usize,
    per_class: HashMap<&'static str, Vec<Pair>>,
}

fn classify(
    chained: &Chained,
    ord: &HashSet<Pair>,
    ef: &HashSet<Pair>,
    co: &HashSet<Pair>,
    n: usize,
) -> Classified {
    let mut per_class: HashMap<&'static str, Vec<Pair>> = HashMap::new();
    let mut total = 0;
    for x in 0..n {
        for y in 0..n {
            if !chained.holds(x, y) || ord.contains(&(x, y)) {
                continue;
            }
            total += 1;
            let class = if x == y {
                "self"
            } else if ord.contains(&(y, x)) {
                "reversal"
            } else if ef.contains(&(x, y)) {
                "witnessed"
            } else if co.contains(&(x, y)) {
                "tie"
            } else {
                "unwitnessed"
            };
            per_class.entry(class).or_default().push((x, y));
        }
    }
    Classified { total, per_class }
}

const CLASSES: [&str; 5] = ["self", "reversal", "witnessed", "tie", "unwitnessed"];

fn report(label: &str, c: &Classified, names: &[String], show: usize) {
    print!("  {label:<24}{:>8} invented ", c.total);
    for k in CLASSES {
        print!("{k} {:>6}  ", c.per_class.get(k).map_or(0, Vec::len));
    }
    println!();
    if show == 0 {
        return;
    }
    for k in CLASSES {
        let Some(ps) = c.per_class.get(k) else { continue };
        let mut ps = ps.clone();
        ps.sort_unstable();
        for (x, y) in ps.iter().take(show) {
            println!("      {k:<12} {} -> {}", names[*x], names[*y]);
        }
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
    let routes = agreed_routes(&schema);
    let max = Saturation::build(
        &locel,
        &schema,
        &grid,
        &acts,
        &routes.0,
        &bounds,
        theta,
        ExpansionDirection::default(),
    );
    let n = grid.activities.len();
    let names = &grid.activities;

    println!("\n{}\n{path}\n{}", "=".repeat(72), "=".repeat(72));
    println!(
        "{} activities, {} types, {} cells recorded, {} in max  ({:.1}s)",
        n,
        schema.types.len(),
        grid.cells.len(),
        max.cells.len(),
        t0.elapsed().as_secs_f64()
    );

    // Two readings of "what the log orders", because ReFlow uses both: the target is
    // stated over `max`, and novelty is stated over the recorded log.
    for (basis, bd, cells) in [
        ("recorded", &bounds, &grid.cells),
        ("max", &max.bounds, &max.cells),
    ] {
        let per_type = asserted_by_type(bd, cells);
        let ord: HashSet<Pair> = per_type.iter().flatten().copied().collect();
        let disagree = ord.iter().filter(|(x, y)| ord.contains(&(*y, *x))).count();
        let (ef, co) = witnesses(bd, cells);
        let ch = Chained::of(&ord, n);
        let c = classify(&ch, &ord, &ef, &co, n);
        println!(
            "\n{basis}: |ord| {} ({} of them a pair two types order both ways), \
             chained {} = ord + {} invented",
            ord.len(),
            disagree,
            ord.len() + c.total,
            c.total
        );
        report("full model", &c, names, show);
    }

    if env::var("SEARCHES").is_err() {
        return;
    }

    // The keep-sets the paper actually ships, scored the same way.
    let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
    target.sort_unstable();
    let target_set: HashSet<Pair> = target.iter().copied().collect();
    let (ef, co) = witnesses(&max.bounds, &max.cells);
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
        target: &target,
        objects_per_type: &objects,
        rep: &closure.rep,
        types: &schema.types,
        n_activities: n,
    };
    let (_best, both) = best_of_two(&input);
    println!("\nsearches: target {} pairs", target.len());
    // The recorded-log ordering, to say whether a chain-only target pair is a fact of the
    // log as extracted or only of its saturation.
    let ord_recorded: HashSet<Pair> = asserted_by_type(&bounds, &grid.cells)
        .into_iter()
        .flatten()
        .collect();
    for f in &both {
        if !f.ran {
            println!("  {:<24} SKIPPED (over budget)", f.strategy.label());
            continue;
        }
        let drawn: HashSet<Pair> = asserted_by_type(&max.bounds, &f.cells)
            .into_iter()
            .flatten()
            .collect();
        let ch = Chained::of(&drawn, n);
        let chain_only: Vec<Pair> = target
            .iter()
            .filter(|p| ch.holds(p.0, p.1) && !drawn.contains(*p))
            .copied()
            .collect();
        let not_recorded = chain_only
            .iter()
            .filter(|p| !ord_recorded.contains(*p))
            .count();
        println!(
            "  {:<24}{} cells, drawn {}/{}, chained {}/{}, chain-only {} ({} of those \
             ordered only in max)",
            f.strategy.label(),
            f.cells.len(),
            f.coverage.drawn,
            f.coverage.target,
            f.coverage.chained,
            f.coverage.target,
            chain_only.len(),
            not_recorded
        );
        let c = classify(&ch, &target_set, &ef, &co, n);
        report(f.strategy.label(), &c, names, show);
    }
}

fn main() {
    for path in env::args().skip(1) {
        run(&path);
    }
}
