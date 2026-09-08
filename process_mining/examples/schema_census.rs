//! Census of the structural schema of one or more OCEL 2.0 logs.
//!
//! The printed format is load-bearing: it is diffed against the Python reference oracle
//! that ships with the ICPM 2027 structure-based reduction paper, and that diff is the
//! only check the two implementations have on each other. Change the format and the
//! comparison has to be redone on every log.
//!
//! Usage: `cargo run --release --features ocel-sqlite --example schema_census -- <log> ...`

use std::{cell::LazyCell, collections::HashSet, env, path::PathBuf, time::Instant};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        activities_without_flow, agreed_routes, annotate, apply_o2o_reduction, assign, best_of_two,
        coverage, delivered_facts, demotable_types, event_attribute_cells, expand_o2o,
        expansion_candidates, expansion_work, fact_repair, facts_from, flow_projection, knobs,
        non_flow_cells, novel_cells, novelty, novelty_by_type, ordered_pairs,
        partially_marked_cells, participations, reduce_o2o, repair_connectivity, state_tally, tag,
        tags, type_densities, ActivityIndexing, Bounds, Cell, CellGrid, ExpansionCandidate,
        ExpansionDirection, Fingerprint, FoldDirection, Guarantees, ReconRoute, Saturation,
        SchemaClosure, SearchInput, StructuralSchema, TraceVariants, TypeState,
        DEFAULT_NOISE_THRESHOLD, DEFAULT_THETA, EXPANSION_WORK_BUDGET, SEARCH_CLOSURE_BUDGET,
    },
    bindings::schema_reduction_bindings::{schema_reduction_evaluate, schema_reduction_overview},
    core::event_data::object_centric::linked_ocel::{
        slim_linked_ocel::ObjectIndex as ObjIx, LinkedOCELAccess, SlimLinkedOCEL,
    },
    Importable, OCEL,
};

/// Cut cells above which the connectivity repair is skipped rather than run: it is
/// quadratic in that count, and a silent approximation would be worse than a stated gap.
const CONNECTIVITY_REPAIR_LIMIT: usize = 200;

fn census(path: &str) {
    let t0 = Instant::now();
    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);
    let t_load = t0.elapsed();

    let n_events = locel.get_all_evs().count();
    let e2o: usize = locel.get_all_evs().map(|e| e.get_e2o(&locel).count()).sum();

    let t0 = Instant::now();
    let schema = StructuralSchema::discover(&locel);
    let t_discover = t0.elapsed();

    println!("\n{}\n{path}\n{}", "=".repeat(72), "=".repeat(72));
    println!(
        "{n_events} events / {} objects / {} types  (load {:.2}s)",
        schema.type_of.len(),
        schema.types.len(),
        t_load.as_secs_f64()
    );

    let mut lines: Vec<String> = schema.recorded.iter().map(|m| m.line(&schema.types)).collect();
    lines.sort();
    println!("\nrecorded O2O maps: {}", schema.recorded.len());
    for l in &lines {
        println!("  {l}");
    }

    let mut lines: Vec<String> = schema.derived.iter().map(|m| m.line(&schema.types)).collect();
    lines.sort();
    println!(
        "\nco-participation maps: {}   ({:.2}s, {} updates, {:.1} ops per E2O tuple)",
        schema.derived.len(),
        t_discover.as_secs_f64(),
        schema.candidate_updates,
        schema.candidate_updates as f64 / e2o.max(1) as f64
    );
    for l in &lines {
        println!("  {l}");
    }

    let pairs = schema.pairs();
    let (gens, depth) = schema.generators();
    println!(
        "\ntype pairs covered: {}  generators: {}  max derivation depth: {depth}",
        pairs.len(),
        gens.len()
    );
    for (s, t) in &gens {
        println!("  gen: {} -> {}", schema.types[*s], schema.types[*t]);
    }

    let t0 = Instant::now();
    let closure = SchemaClosure::build(&locel, &schema);
    let grid = CellGrid::build(&locel, &schema, &closure);
    let canonical = grid.canonical_keepset(&closure, FoldDirection::default());
    let maximum = grid.maximum_keepset(&closure);

    // The two keep-set-independent structures, built at most once each and shared by every
    // block below. Both used to be rebuilt per call, which on BPIC2017 is a pass over 2.4M
    // participations a time.
    let acts = ActivityIndexing::build(&locel, &grid);
    let variants = LazyCell::new(|| TraceVariants::build(&locel, &schema, &acts));
    let bounds = LazyCell::new(|| Bounds::build(&locel, &schema, &acts));

    let tau: f64 = env::var("TAU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let theta: f64 = env::var("THETA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THETA);

    // `max` and the routes it is built from, shared by every block that reads either.
    // Both are keep-set independent and both cost a pass over the log, so a block that
    // rebuilds them is a block that pays for the whole saturation again.
    let routes = LazyCell::new(|| agreed_routes(&schema));
    let max = LazyCell::new(|| {
        Saturation::build(
            &locel,
            &schema,
            &grid,
            &acts,
            &routes.0,
            &bounds,
            theta,
            ExpansionDirection::default(),
        )
    });

    // The per-cell admissibility distribution, printed whatever `theta` is set to, because
    // the constant is exactly what the distribution is supposed to justify: a threshold read
    // off a band is only defensible while the band is there. Every rate the reduction would
    // decide on, sorted, so the gap (if any) is visible rather than asserted.
    if env::var("THETA_DIST").is_ok() {
        let (rts, _) = agreed_routes(&schema);
        for dir in [ExpansionDirection::Forward, ExpansionDirection::Backward] {
            let work = expansion_work(&rts, &grid, dir);
            println!("\nTHETA distribution, {dir:?}: {work} route-object checks");
            if work > EXPANSION_WORK_BUDGET {
                println!("  SKIPPED, over the {EXPANSION_WORK_BUDGET} budget");
                continue;
            }
            let t0 = Instant::now();
            let mut cand =
                expansion_candidates(&locel, &rts, &schema, &grid, &acts, theta, dir);
            cand.sort_by(|x, y| {
                y.admission_rate()
                    .partial_cmp(&x.admission_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let admitted = cand.iter().filter(|c| c.admissible).count();
            let tuples: usize = cand.iter().map(|c| c.tuples.len()).sum();
            let kept: usize = cand
                .iter()
                .filter(|c| c.admissible)
                .map(|c| c.tuples.len())
                .sum();
            let extrap: usize = cand
                .iter()
                .filter(|c| c.admissible)
                .map(ExpansionCandidate::extrapolated)
                .sum();
            println!(
                "  {} candidate cells, {admitted} admitted at theta={theta} ({kept} of {tuples} tuples, {extrap} extrapolated, {:.2}s)",
                cand.len(),
                t0.elapsed().as_secs_f64()
            );
            for c in &cand {
                println!(
                    "    {:>6.3}  {:<30} {:<18} {:>9} tuples  {}",
                    c.admission_rate(),
                    grid.activities[c.cell.0],
                    schema.types[c.cell.1],
                    c.tuples.len(),
                    if c.admissible { "admit" } else { "reject" }
                );
            }
            // The band the constant would be read off: the widest gap between an admitted
            // rate and the next one below it, over the cells this log offers.
            let rates: Vec<f64> = cand.iter().map(ExpansionCandidate::admission_rate).collect();
            let gap = rates
                .windows(2)
                .map(|w| (w[0] - w[1], w[1], w[0]))
                .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
            match gap {
                Some((w, lo, hi)) => println!("  widest gap: {w:.3}, from {lo:.3} to {hi:.3}"),
                None => println!("  widest gap: no two cells to compare"),
            }
        }
    }

    if env::var("FACTS").is_ok() {
        let full_facts = facts_from(&bounds, &grid.cells, tau);
        println!(
            "\nFACTS at tau={tau}: {} ordering, {} role, {} tied",
            full_facts.ordering.len(),
            full_facts.role.len(),
            full_facts.tied.len()
        );

        // The saturated log: every cell the schema determines and the extraction did not
        // record, written. It is the denominator of the second density, and it is NOT the
        // same measurement as the first -- covering pairs over recorded activities against
        // closure pairs over saturated ones.
        let t_sat = Instant::now();
        let clashes = &routes.1;
        println!(
            "  saturation: {} routes, {} route-object checks, {} objects dropped over {} disagreeing type pairs",
            routes.0.len(),
            max.work,
            clashes.iter().map(|d| d.objects).sum::<usize>(),
            clashes.len()
        );
        for d in clashes {
            println!(
                "    disagree: {} -> {}  {} of {} objects reached by two routes at once",
                schema.types[d.source], schema.types[d.target], d.objects, d.domain
            );
        }
        if max.within_budget {
            println!(
                "  max: {} cells ({} written), E2O {} recorded + {} written = {} ({:.2}s)",
                max.cells.len(),
                max.admitted.len(),
                grid.e2o_total,
                max.written.len(),
                grid.e2o_total + max.written.len(),
                t_sat.elapsed().as_secs_f64()
            );
        } else {
            println!(
                "  saturation SKIPPED, {} route-object checks over the {EXPANSION_WORK_BUDGET} budget; the max column is the recorded log",
                max.work
            );
        }
        let max_cells = max.cells.clone();
        let max_facts = facts_from(&max.bounds, &max_cells, tau);

        let rows = type_densities(
            schema.types.len(),
            &grid.cells,
            &full_facts,
            &max_cells,
            &max_facts,
            &schema.pairs(),
            &closure.rep,
            &schema.types,
        );
        let mut rows: Vec<_> = rows.into_iter().filter(|r| r.activities > 0).collect();
        rows.sort_by_key(|r| schema.types[r.object_type].clone());
        // `dens` and `densX` are printed and read by nothing: the classifier is `uniq`,
        // the marginal-contribution test. Both densities stay in the table so the retired
        // measure can be compared against the verdict rather than merely asserted to differ.
        println!(
            "  {:<22}{:>5}{:>7}{:>8}{:>6}{:>6}   {:>7}{:>6}   {:>5}{:>8}   {:<8}{:<12}state",
            "type", "acts", "order", "dens", "role", "tied", "asserts", "uniq", "actsX",
            "densX", "map?", "wants"
        );
        for r in &rows {
            println!(
                "  {:<22}{:>5}{:>7}{:>7.1}%{:>6}{:>6}   {:>7}{:>6}   {:>5}{:>7.1}%   {:<8}{:<12}{}{}",
                schema.types[r.object_type],
                r.activities,
                r.ordering,
                100.0 * r.density_recorded,
                r.role,
                r.tied,
                r.asserted,
                r.unique,
                r.activities_max,
                100.0 * r.density_max,
                if r.determined {
                    "apply"
                } else if r.determines {
                    "expand"
                } else {
                    "-"
                },
                r.wants.label(),
                r.state.label(),
                match r.covered_by {
                    Some(s) if r.asserted > 0 =>
                        format!("  (says nothing {} does not)", schema.types[s]),
                    _ => String::new(),
                }
            );
        }
        let (flow, involvement, implied) = state_tally(&rows);
        println!("  flow {flow}, involvement {involvement}, implied {implied}");
        let outside: usize = rows
            .iter()
            .filter(|r| r.state != TypeState::Flow)
            .map(|r| {
                grid.cells
                    .iter()
                    .filter(|(_, t)| *t == r.object_type)
                    .map(|(a, t)| {
                        let cs = &grid.per_activity[*a];
                        cs.slot(*t).map_or(0, |j| cs.counts[j])
                    })
                    .sum::<usize>()
            })
            .sum();
        println!(
            "  participations outside flow: {outside} of {} ({:.1}%)",
            grid.e2o_total,
            100.0 * outside as f64 / grid.e2o_total.max(1) as f64
        );
    }

    if env::var("NOVELTY").is_ok() {
        println!("\nNOVELTY at theta={theta}: what each written cell teaches the model");
        if !max.within_budget {
            println!("  saturation SKIPPED, nothing to score");
        } else {
            let before = ordered_pairs(&bounds, &grid.cells);
            println!("  activity pairs ordered by some type, recorded: {}", before.len());
            let rows = novelty(&locel, &acts, &bounds, &grid.cells, &max);
            let admits = rows.iter().filter(|n| n.admissible()).count();
            println!(
                "  {:<30} {:<18}{:>10}{:>8}{:>7}{:>9}",
                "activity", "type", "tuples", "novel", "echo", "fan-out"
            );
            for n in &rows {
                println!(
                    "    {:<28} {:<18}{:>10}{:>8}{:>7}{:>9.2}  {}",
                    grid.activities[n.cell.0],
                    schema.types[n.cell.1],
                    n.tuples,
                    n.novel(),
                    n.echo,
                    n.fanout,
                    if n.admissible() { "write" } else { "pointless" }
                );
            }
            println!(
                "  {admits} of {} admitted cells teach the model an ordering it did not have",
                rows.len()
            );
            println!("\n  by type (the shape the Python reference reports):");
            println!(
                "  {:<20}{:>6}{:>10}{:>7}{:>7}{:>9}",
                "type", "cells", "tuples", "novel", "echo", "fan-out"
            );
            for (t, cells, n) in novelty_by_type(&locel, &acts, &bounds, &grid.cells, &max) {
                println!(
                    "  {:<20}{cells:>6}{:>10}{:>7}{:>7}{:>9.2}",
                    schema.types[t],
                    n.tuples,
                    n.novel(),
                    n.echo,
                    n.fanout
                );
            }
        }
    }

    if env::var("COVER").is_ok() {
        let mut target: Vec<_> = max.target_pairs().into_iter().collect();
        target.sort_unstable();
        println!(
            "\nCOVER: {} target pairs ordered in max over {} activities",
            target.len(),
            grid.activities.len()
        );
        println!(
            "  {:<16}{:>6}{:>10}{:>10}{:>10}",
            "keep-set", "cells", "E2O", "drawn", "+chained"
        );
        let sets: Vec<(&str, HashSet<(usize, usize)>)> = vec![
            ("recorded", grid.cells.clone()),
            ("max", max.cells.clone()),
            ("canonical", canonical.kept.clone()),
            ("maximum", maximum.clone()),
        ];
        for (name, set) in &sets {
            // Coverage is read off `max`'s bounds throughout, so the comparison is between
            // keep-sets and not between two different logs.
            let c = coverage(&max.bounds, set, &target, grid.activities.len());
            println!(
                "  {name:<16}{:>6}{:>10}{:>6}/{:<3}{:>7}/{:<3}",
                set.len(),
                max.participations(set),
                c.drawn,
                c.target,
                c.chained,
                c.target
            );
        }
    }

    if env::var("SEARCH").is_ok() {
        let mut target: Vec<_> = max.target_pairs().into_iter().collect();
        target.sort_unstable();
        // Arcs are scored over `max`, not over the recorded log: a flow layer may hold
        // cells the extraction did not record, and the recorded variants read their arcs
        // as implied, which is the opposite of what writing them does.
        let sat_variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
        let objects = Saturation::objects_per_type(&schema);
        // Novelty gates which expansions a search may buy: theta says the objects were
        // there, novelty says the model learns something by saying so.
        let novel = novel_cells(&novelty(&locel, &acts, &bounds, &grid.cells, &max));
        let mut allowed = grid.cells.clone();
        allowed.extend(novel.keys().copied());
        println!(
            "\nSEARCH: {} target pairs, {} allowed cells ({} recorded + {} novel expansions of {} admitted)",
            target.len(),
            allowed.len(),
            grid.cells.len(),
            novel.len(),
            max.admitted.len()
        );
        let input = SearchInput {
            max: &max,
            variants: &sat_variants,
            allowed: &allowed,
            recorded: &grid.cells,
            target: &target,
            objects_per_type: &objects,
            rep: &closure.rep,
            types: &schema.types,
            n_activities: grid.activities.len(),
        };
        let t_s = Instant::now();
        let (best, both) = best_of_two(&input);
        println!(
            "  {:<10}{:>6}{:>8}{:>10}{:>8}{:>10}{:>7}",
            "strategy", "cells", "arcs", "E2O", "dropped", "chained", "comp"
        );
        for f in &both {
            if !f.ran {
                println!(
                    "  {:<10}  SKIPPED, over the {SEARCH_CLOSURE_BUDGET} chained-closure budget",
                    f.strategy.label()
                );
                continue;
            }
            println!(
                "  {:<10}{:>6}{:>8}{:>10}{:>8}{:>6}/{:<3}{:>7}",
                f.strategy.label(),
                f.cells.len(),
                f.arcs,
                f.participations,
                f.dropped,
                f.coverage.chained,
                f.coverage.target,
                f.incidence_components
            );
        }
        if best.ran {
            println!(
                "  WINNER: {} ({:.2}s)",
                best.strategy.label(),
                t_s.elapsed().as_secs_f64()
            );
            // Three-case rule: an ordering fact must be drawn by its own type or excused
            // by determination at each endpoint. The search optimises under type-agnostic
            // pair coverage, so its layer is repaired here, per strategy, and the winner
            // is re-ranked on the repaired arcs.
            let full_facts_r = facts_from(&bounds, &grid.cells, tau);
            let mut repaired: Vec<(&'static str, HashSet<Cell>, usize)> = Vec::new();
            for f in &both {
                if !f.ran {
                    continue;
                }
                let mut cells = f.cells.clone();
                let added = fact_repair(&grid, &full_facts_r, &mut cells);
                let arcs = input.variants.arc_set(&cells).len();
                println!(
                    "  {:<10} after fact repair: +{} cells -> {} cells / {} arcs",
                    f.strategy.label(),
                    added.len(),
                    cells.len(),
                    arcs
                );
                repaired.push((f.strategy.label(), cells, arcs));
            }
            let (best_label, best_cells, _) = repaired
                .iter()
                .min_by_key(|(_, c, arcs)| (*arcs, c.len()))
                .cloned()
                .expect("at least one strategy ran");
            println!("  RULE WINNER: {best_label}");
            let mut per: std::collections::BTreeMap<String, Vec<String>> = Default::default();
            for (a, t) in &best_cells {
                per.entry(schema.types[*t].clone()).or_default().push(format!(
                    "{}{}",
                    grid.activities[*a],
                    if grid.cells.contains(&(*a, *t)) { "" } else { " [+]" }
                ));
            }
            for (t, mut aa) in per {
                aa.sort();
                println!("    {t:<18} {aa:?}");
            }
            let orphans = activities_without_flow(&best_cells, grid.activities.len());
            if !orphans.is_empty() {
                println!(
                    "    activities with no flow cell: {:?}",
                    orphans.iter().map(|a| &grid.activities[*a]).collect::<Vec<_>>()
                );
            }

            // The end-to-end result. The two non-flow states are one rule, not two: a cell
            // no flow cell at its activity determines is involved, the rest are implied.
            let a = assign(&grid, &best_cells, &HashSet::new());
            let (flow, inv, abs, exp) = a.tally();
            println!(
                "\n  ASSIGNMENT ({}): {} recorded cells -> {flow} flow / {inv} involvement / {abs} implied, + {exp} expanded",
                best_label,
                grid.cells.len()
            );
            let name = |cs: &[(usize, usize)]| {
                let mut per: std::collections::BTreeMap<String, Vec<String>> = Default::default();
                for (a, t) in cs {
                    per.entry(schema.types[*t].clone())
                        .or_default()
                        .push(grid.activities[*a].clone());
                }
                per
            };
            for (label, cells) in [
                ("involvement", &a.involvement),
                ("implied", &a.implied),
            ] {
                for (t, mut aa) in name(cells) {
                    aa.sort();
                    println!("    {label:<12} {t:<18} {aa:?}");
                }
            }
            let p_flow = participations(&grid, &a.flow);
            let p_inv = participations(&grid, &a.involvement);
            let p_abs = participations(&grid, &a.implied);
            println!(
                "    participations: {p_flow} flow / {p_inv} involvement / {p_abs} implied of {} ({:.1}% outside flow)",
                grid.e2o_total,
                100.0 * (p_inv + p_abs) as f64 / grid.e2o_total.max(1) as f64
            );

            // Type-attributed facts under map-mediated delivery: a fact (T, x, y) counts
            // as delivered only when a kept type orders the pair AND T is readable from it
            // at both endpoints (determinacy closure). The search's coverage is
            // type-agnostic pairs, so this is the stricter accounting; a lost fact here is
            // a move out of the flow layer the three-case rule (map / nothing-to-say / keep)
            // would refuse.
            let flow_recorded: HashSet<Cell> = a
                .flow
                .iter()
                .filter(|c| grid.cells.contains(*c))
                .copied()
                .collect();
            let full_facts = facts_from(&bounds, &grid.cells, tau);
            let reduced_facts = facts_from(&bounds, &flow_recorded, tau);
            let del = delivered_facts(
                &grid,
                &full_facts,
                &reduced_facts,
                &flow_recorded,
                schema.types.len(),
            );
            let lost_ord: Vec<_> = full_facts.ordering.difference(&del.ordering).collect();
            let lost_role: Vec<_> = full_facts.role.difference(&del.role).collect();
            println!(
                "    FACT AUDIT: ordering {}/{} delivered, role {}/{} delivered",
                del.ordering.len(),
                full_facts.ordering.len(),
                del.role.len(),
                full_facts.role.len()
            );
            let mut lost_sorted: Vec<_> = lost_ord.iter().collect();
            lost_sorted.sort();
            for (t, x, y) in lost_sorted.iter().take(12) {
                println!(
                    "      lost ordering: {:<18} {} < {}",
                    schema.types[*t], grid.activities[*x], grid.activities[*y]
                );
            }
            let mut lost_role_sorted: Vec<_> = lost_role.iter().collect();
            lost_role_sorted.sort();
            for (t, x, y) in lost_role_sorted.iter().take(12) {
                println!(
                    "      lost role:     {:<18} {} | {}",
                    schema.types[*t], grid.activities[*x], grid.activities[*y]
                );
            }

            // What the annotation of the tagged log says, and what a reader of the flow
            // projection alone would have to rebuild.
            let t_p = Instant::now();
            let rec = annotate(
                &locel,
                &schema,
                &closure,
                &grid,
                &best_cells,
                &HashSet::new(),
            );
            let (recorded_w, coparticip_w) = rec.witness_split();
            let (involved, implied) = rec.split();
            let residuals: usize = rec
                .cells
                .iter()
                .filter_map(|m| m.determined_by.as_ref())
                .map(|d| d.residuals.len())
                .sum();
            println!(
                "\n  ANNOTATION ({:.2}s): {} non-flow cells, {involved} involved, {implied} implied, {} participations",
                t_p.elapsed().as_secs_f64(),
                rec.cells.len(),
                rec.participations()
            );
            println!(
                "    relations used: {recorded_w} with a recorded O2O witness / {coparticip_w} derived from co-participation"
            );
            println!(
                "    residual objects {residuals}; inexact cells {}; longest reconstruction chain {}",
                rec.inexact(),
                rec.max_route_depth()
            );
            let mut per_map: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
            for m in &rec.cells {
                let Some(d) = &m.determined_by else { continue };
                let e = per_map
                    .entry(format!(
                        "{} -> {} [{}]",
                        d.source_type,
                        m.object_type,
                        d.realisation.witness()
                    ))
                    .or_default();
                e.0 += 1;
                e.1 += m.participations;
            }
            for (k, (cells, parts)) in per_map {
                println!("    {k:<52} {cells} cells, {parts} participations");
            }

            // The headline claim, run rather than argued: the tagged log tags the input,
            // i.e. dropping the tags returns it. Held behind its own switch because it
            // materialises two more copies of the log.
            if env::var("ROUNDTRIP").is_ok() {
                let t_r = Instant::now();
                let tagged = tag(&locel, &schema, &acts, &best_cells, &[]);
                let read_back = non_flow_cells(&tagged, &schema, &acts);
                let partial = partially_marked_cells(&tagged, &schema, &acts);
                println!(
                    "\n  TAGGED LOG: {} cells read back off the qualifiers ({} expected, {} partially tagged), schema {}",
                    read_back.len(),
                    rec.cells.len(),
                    partial.len(),
                    if StructuralSchema::discover(&tagged).pairs() == pairs {
                        "unchanged"
                    } else {
                        "CHANGED"
                    }
                );
                let flow_only = flow_projection(&tagged);
                let e2o_after: usize =
                    flow_only.get_all_evs().map(|e| e.get_e2o(&*flow_only).count()).sum();
                let d = Fingerprint::build(&tagged).diff(&Fingerprint::build(&locel), 5);
                println!(
                    "  ROUND TRIP ({:.2}s): flow projection carries {e2o_after} participations, full projection {} L (tagged log differs from L in {} events)",
                    t_r.elapsed().as_secs_f64(),
                    if tags(&tagged, &locel) { "==" } else { "!=" },
                    d.events_differing.len()
                );
            }
        }
    }

    if env::var("O2OREDUCE").is_ok() {
        // The fourth operation. It never touches the log, so it is the only step that is
        // exactly reversible with no commitment at all.
        let t_o = Instant::now();
        let r = reduce_o2o(&locel, &schema);
        println!(
            "\nO2O REDUCTION ({:.2}s, {} composition checks): {} of {} qualified relations implied, {} of {} edges dropped ({:.1}%)",
            t_o.elapsed().as_secs_f64(),
            r.checks,
            r.implied.len(),
            r.relations_total,
            r.edges_dropped,
            r.edges_total,
            100.0 * r.edges_dropped as f64 / r.edges_total.max(1) as f64
        );
        for i in &r.implied {
            println!("    {}", i.line(&schema.types));
        }
        if !r.implied.is_empty() {
            let before = Fingerprint::build(&locel);
            let reduced = apply_o2o_reduction(&locel, &r);
            let back = expand_o2o(&reduced, &schema, &r);
            let d = Fingerprint::build(&back).diff(&before, 5);
            println!(
                "    inverse: {}",
                if d.is_empty() { "exact".to_string() } else { format!("DIFFERS -- {}", d.summary()) }
            );
        }
    }

    if env::var("ARCS").is_ok() {
        let full = variants.arc_set(&grid.cells);
        let mut per: std::collections::BTreeMap<usize, usize> = Default::default();
        for (t, _, _) in &full {
            *per.entry(*t).or_default() += 1;
        }
        println!("\narcs of the full model: {}", full.len());
        for (t, n) in &per {
            println!(
                "    {:<14} {n} arcs ({:.0}%)",
                schema.types[*t],
                100.0 * *n as f64 / full.len() as f64
            );
        }
    }

    if env::var("ATTRIBUTES").is_ok() {
        let full_facts = facts_from(&bounds, &grid.cells, tau);
        let dem = demotable_types(&schema, &closure, &grid, &full_facts);
        println!("\ntypes that could be an attribute instead of a type: {}", dem.len());
        for d in &dem {
            println!(
                "    {:<14} -> attribute of {:<14} {} values, {} objects, {} participations, {} activities, asserts {} orderings{}",
                schema.types[d.object_type],
                schema.types[d.carrier],
                d.distinct_values,
                d.objects_removed,
                d.participations_removed,
                d.activities,
                d.ordering_facts,
                if d.blocked.is_empty() {
                    String::new()
                } else {
                    format!(
                        "  BLOCKED: would lose {}",
                        d.blocked.iter().map(|u| schema.types[*u].as_str()).collect::<Vec<_>>().join(", ")
                    )
                }
            );
        }
    }

    if env::var("ATTR").is_ok() {
        let facts_a = facts_from(&bounds, &grid.cells, tau);
        let ea = event_attribute_cells(&locel, &schema, &grid, &acts, &facts_a);
        let free: Vec<_> = ea.iter().filter(|c| c.ordering_facts == 0).collect();
        println!(
            "\n  free of ordering cost: {} cells carrying {} participations",
            free.len(),
            free.iter().map(|c| c.participations_removed).sum::<usize>()
        );
        let irreducible: Vec<_> = ea.iter().filter(|c| !c.reducible).collect();
        println!(
            "\ncells that could be an event attribute: {} of {} ({} of them irreducible, so this is their only move)",
            ea.len(),
            grid.cells.len(),
            irreducible.len()
        );
        println!(
            "    participations they carry: {} of {}",
            ea.iter().map(|c| c.participations_removed).sum::<usize>(),
            grid.e2o_total
        );
        for c in ea.iter().take(8) {
            println!(
                "    {:<28} {:<14} {} events, {} distinct values{}",
                grid.activities[c.activity],
                schema.types[c.object_type],
                c.events,
                c.distinct_values,
                format!(
                    "{}{}",
                    if c.reducible { "" } else { ", irreducible" },
                    if c.ordering_facts == 0 {
                        ", asserts nothing".to_string()
                    } else {
                        format!(", costs {} orderings", c.ordering_facts)
                    }
                )
            );
        }
    }

    if env::var("GRID").is_ok() {
        let mut per_act: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        for (a, t) in &grid.cells {
            per_act.entry(grid.activities[*a].clone()).or_default().push(schema.types[*t].clone());
        }
        println!("\nrecorded cells per activity:");
        for (a, mut ts) in per_act {
            ts.sort();
            println!("    {a:<30} {}", ts.join(", "));
        }
    }
    let cells = grid.len();
    let cut_canonical = cells - canonical.kept.len();
    let e2o_cut = canonical.participations_cut();

    println!(
        "\nREDUCTION  ({:.2}s, closure {} pairs)",
        t0.elapsed().as_secs_f64(),
        closure.maps.len()
    );
    println!(
        "  cells {cells}  kept(canonical) {}  cut {cut_canonical} ({:.0}%)",
        canonical.kept.len(),
        100.0 * cut_canonical as f64 / cells.max(1) as f64
    );
    println!(
        "  cells {cells}  kept(max)       {}  cut {} ({:.0}%)",
        maximum.len(),
        cells - maximum.len(),
        100.0 * (cells - maximum.len()) as f64 / cells.max(1) as f64
    );
    println!(
        "  E2O instances removed (canonical): {e2o_cut} of {} ({:.0}%)",
        grid.e2o_total,
        100.0 * e2o_cut as f64 / grid.e2o_total.max(1) as f64
    );
    let eliminated: Vec<&String> = canonical
        .types_eliminated(schema.types.len())
        .iter()
        .map(|t| &schema.types[*t])
        .collect();
    println!("  types eliminated entirely: {eliminated:?}");

    let t1 = Instant::now();
    let (arcs_full, comps_full) = variants.arcs_and_components(&grid.cells);
    let (arcs_red, comps_red) = variants.arcs_and_components(&canonical.kept);
    let (arcs_max, comps_max) = variants.arcs_and_components(&maximum);
    println!(
        "  arcs  full {arcs_full} / {comps_full} comp   canonical {arcs_red} / {comps_red} comp   maximum {arcs_max} / {comps_max} comp   ({:.2}s)",
        t1.elapsed().as_secs_f64()
    );

    // Repair is quadratic in the number of cut cells, so it is skipped on very large logs
    // rather than silently approximated. Say so; do not print a bare number.
    if comps_max > comps_full && maximum.len() <= CONNECTIVITY_REPAIR_LIMIT {
        let mut cut_max: Vec<_> = grid.cells.difference(&maximum).copied().collect();
        cut_max.sort();
        let (rep_set, added) = repair_connectivity(&variants, &maximum, &cut_max, comps_full);
        let (a, c) = variants.arcs_and_components(&rep_set);
        println!(
            "  maximum + connectivity repair: +{added} cells -> kept {} , arcs {a} / {c} comp",
            rep_set.len()
        );
    }
    if comps_red > comps_full {
        let mut cut_list: Vec<_> = canonical.cut.iter().map(|c| c.cell).collect();
        if cut_list.len() <= CONNECTIVITY_REPAIR_LIMIT {
            cut_list.sort();
            let (repaired, added) =
                repair_connectivity(&variants, &canonical.kept, &cut_list, comps_full);
            let (a, c) = variants.arcs_and_components(&repaired);
            println!(
                "  connectivity repair: +{added} cells -> kept {} , arcs {a} / {c} comp",
                repaired.len()
            );
        } else {
            println!(
                "  connectivity repair: SKIPPED, {} cut cells exceeds the {CONNECTIVITY_REPAIR_LIMIT}-cell limit",
                cut_list.len()
            );
        }
    } else {
        println!("  connectivity repair: not needed");
    }

    for (name, set) in [("canonical", &canonical.kept), ("maximum", &maximum)] {
        let g = Guarantees::evaluate(&grid, set, schema.types.len());
        println!(
            "  guarantees ({name}): type-closed {}, spanning {:?}, incidence {} comp",
            g.type_closed,
            g.spanning_types
                .iter()
                .map(|t| &schema.types[*t])
                .collect::<Vec<_>>(),
            g.incidence_components
        );
        let fid = variants.df_fidelity(&grid, set);
        println!(
            "  DF fidelity ({name}): {} arcs, {} not supported by any full trace, {} missing",
            fid.reduced_arcs, fid.spurious, fid.missing
        );
    }

    if env::var("ROUTES").is_ok() {
        // What a cut cell costs the notation. A single function application is a badge on
        // the activity: one arrow from the kept type. A fibre or a union of qualifiers is
        // not, and has to be drawn as structure on the arcs.
        let levels: Vec<(&str, HashSet<(usize, usize)>)> = vec![
            ("finest-type", canonical.kept.clone()),
            ("maximum", maximum.clone()),
            ("wholetype", grid.type_closed_keepset(&closure, FoldDirection::default())),
        ];
        println!("\nROUTES: how each cut cell reconstructs");
        for (name, kept) in &levels {
            let (mut f, mut fib, mut un, mut none) = (0usize, 0usize, 0usize, 0usize);
            for (aix, cellset) in grid.per_activity.iter().enumerate() {
                for (j, t) in cellset.present.iter().enumerate() {
                    if kept.contains(&(aix, *t)) {
                        continue;
                    }
                    // The best route available from any kept type at this activity: a badge
                    // if one of them is a function, and the cheapest thing otherwise.
                    let mut best: Option<&ReconRoute> = None;
                    for (i, s) in cellset.present.iter().enumerate() {
                        if !kept.contains(&(aix, *s)) {
                            continue;
                        }
                        if let Some(r) = cellset.recon[i][j].as_ref() {
                            best = match (best, r) {
                                (_, ReconRoute::Function { .. }) => Some(r),
                                (None, _) => Some(r),
                                (b, _) => b,
                            };
                        }
                    }
                    match best {
                        Some(ReconRoute::Function { .. }) => f += 1,
                        Some(ReconRoute::Fibre { .. }) => fib += 1,
                        Some(ReconRoute::QualifiedUnion { .. }) => un += 1,
                        None => none += 1,
                    }
                }
            }
            let total = f + fib + un + none;
            println!(
                "  {name:<12} {total} cut: {f} function, {fib} fibre, {un} qualified union, {none} unreconstructed"
            );
        }
    }

    if env::var("STABLE").is_ok() {
        // Seidel et al., Def. 8: a type pair is a *stable many-to-one relationship* when
        // every object of the many-side co-occurs, over the whole log, with exactly one
        // object of the one-side. Ours asks that one object of the one-side is present at
        // every event of the many-side, which the union test refuses whenever a second
        // target appears at a single event. The union is also blind to the recorded
        // object-to-object relation, which Def. 8 never reads.
        let mut union: std::collections::HashMap<(usize, usize), std::collections::HashMap<ObjIx, std::collections::HashSet<ObjIx>>> = Default::default();
        for e in locel.get_all_evs() {
            let objs: Vec<ObjIx> = e.get_e2o(&locel).copied().collect();
            for a in &objs {
                for b in &objs {
                    if a == b {
                        continue;
                    }
                    let (s, t) = (schema.type_of[a], schema.type_of[b]);
                    if s == t {
                        continue;
                    }
                    union.entry((s, t)).or_default().entry(*a).or_default().insert(*b);
                }
            }
        }
        println!("\nSTABLE (Seidel Def. 8) against our maps:");
        let n_obj = |t: usize| schema.type_of.values().filter(|x| **x == t).count();
        for m in schema.recorded.iter().chain(schema.derived.iter()) {
            let key = (m.source, m.target);
            let per = union.get(&key);
            let (multi, missing) = match per {
                None => (0, n_obj(m.source)),
                Some(u) => (
                    u.values().filter(|v| v.len() > 1).count(),
                    n_obj(m.source) - u.len(),
                ),
            };
            println!(
                "  {:<28} {:<10} stable: {}   ({} of {} source objects co-occur with 2+, {} with none)",
                m.line(&schema.types).split(" (").next().unwrap_or(""),
                m.origin.label(),
                if multi == 0 && missing == 0 { "yes" } else { "NO" },
                multi,
                n_obj(m.source),
                missing
            );
        }
    }
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    // `--json` prints what the editor binding returns, which is how a UI fixture is made
    // from a real log rather than by hand.
    let json = args.iter().any(|a| a == "--json");
    args.retain(|a| a != "--json");
    // `--params` prints the parameter set alone, which is what `artifacts/results/params.json`
    // holds. It is the only place those constants are published.
    if args.iter().any(|a| a == "--params") {
        println!("{}", serde_json::to_string_pretty(&knobs()).expect("serialise knobs"));
        return;
    }
    if args.is_empty() {
        eprintln!("usage: schema_census [--json | --params] <log> [<log> ...]");
        return;
    }
    for path in &args {
        if json {
            let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
            let locel = SlimLinkedOCEL::from_ocel(ocel);
            let overview = schema_reduction_overview(&locel, None, None, Some(true));
            // The assignment the editor opens with is the computed one. There is no
            // menu of named levels to enumerate any more; a client edits cell by cell from
            // here, and this is what its first evaluation call returns.
            let flow = overview.recommendation.flow.clone();
            let evaluation =
                schema_reduction_evaluate(&locel, flow, Vec::new(), Vec::new(), None, None);
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "overview": overview,
                    "evaluation": evaluation,
                }))
                .unwrap()
            );
        } else {
            census(path);
        }
    }
}
