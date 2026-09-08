//! The type-deletion baseline: what removing whole object types before discovery costs.
//!
//! This is the practice the paper argues against, run as an external competitor rather than as
//! a restriction of our own oracle. Which types go is decided by a criterion, not by a search
//! over our own objective, because a search that optimises our coverage measure cannot lose
//! coverage and so cannot be a baseline at all.
//!
//! **The criterion.** A type is resource-like when it *diverges* at at least
//! [`DIVERGENCE_SHARE`] of the activities it attends and attends at least [`MIN_ACTIVITIES`] of
//! them. Divergence is the standard object-centric notion: an object of `T` attends two events
//! of one activity whose other objects differ. The second clause exists because a type with a
//! single cell that diverges there scores 1.0 for free.
//!
//! At 0.5 the criterion reproduces both published filterings we could check it against:
//! Liss et al. keep `Customer Order`, `Transport Document`, `Container` and `Handling Unit` on
//! Container Logistics, dropping `Truck`, `Forklift` and `Vehicle`; Christfort et al. keep
//! `orders`, `items` and `packages` on Order Management, dropping `customers`, `employees` and
//! `products`. The criterion drops exactly those sets. The two papers disagree with each other
//! on Container Logistics, where Christfort et al. drop nothing, which is itself the point:
//! hand filtering is a convention and not a stated condition.
//!
//! **The measurement.** Both sides are scored the same way and neither gets route credit. A
//! deleted type leaves no map behind, so an ordering it carried can only survive if a kept type
//! still draws it or a chain of kept orderings still reaches it. Giving our side the delivered
//! orderings and the baseline none would be the comparison answering itself.
//!
//! Usage: `cargo run --release --example type_deletion -- <log> ...` (defaults to the corpus).
//! Env: `REFLOW_LOGS` names the corpus directory, `REFLOW_STATS` the output directory.

mod corpus;

use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    path::PathBuf,
};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, asserted_by_type, assign, drawn, fact_repair, facts_from, flow_projection,
        reflow_layer, tag, ActivityIndexing, Bounds, Cell, CellGrid, Chained, ExpansionDirection,
        Pair, Saturation, SchemaClosure, SearchInput, StructuralSchema, TraceVariants,
        DEFAULT_THETA,
    },
    core::{
        event_data::object_centric::linked_ocel::SlimLinkedOCEL,
        process_models::object_centric::{
            ocdfg::{discover_dfg_from_ocel, OCDirectlyFollowsGraph},
            ocpn::ObjectCentricPetriNet,
        },
    },
    discovery::{
        case_centric::inductive_miner::InductiveMinerOptions,
        object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions},
    },
    Importable, OCEL,
};

/// Matches `examples/cross_instantiation.rs`, so arc counts are comparable with Table 1.
const IMF_THRESHOLD: f64 = 0.2;

/// Overridable so the sensitivity check is a rerun and not an edit.
fn divergence_share() -> f64 {
    std::env::var("DIV_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5)
}
const MIN_ACTIVITIES: usize = 2;

fn log_stem(path: &str) -> String {
    let p = path.strip_suffix(".gz").unwrap_or(path);
    std::path::Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

/// Silent transitions are an artefact of the miner, not of the participation, so they are
/// removed before counting. Without this the recorded Order Management net reads 238 arcs
/// instead of the 163 every other instrument reports.
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

/// Types the divergence criterion removes, by name.
///
/// Divergence is decided per (activity, type): some object of the type attends two events of
/// that activity whose *other* objects differ. Only one bit per object is kept, and a cell stops
/// being tracked once it is known to diverge, so BPIC2017's 433k events fit in memory.
fn divergent_types(ocel: &OCEL) -> (Vec<String>, HashMap<String, (usize, usize)>) {
    let type_of: HashMap<&str, &str> = ocel
        .objects
        .iter()
        .map(|o| (o.id.as_str(), o.object_type.as_str()))
        .collect();

    // (activity, type) -> object -> hash of the first other-set seen for it.
    let mut seen: HashMap<(&str, &str), HashMap<&str, u64>> = HashMap::new();
    let mut diverges: HashSet<(&str, &str)> = HashSet::new();
    let mut attends: HashSet<(&str, &str)> = HashSet::new();

    for e in &ocel.events {
        let mut by_type: HashMap<&str, Vec<&str>> = HashMap::new();
        for r in &e.relationships {
            if let Some(t) = type_of.get(r.object_id.as_str()) {
                by_type.entry(t).or_default().push(r.object_id.as_str());
            }
        }
        for (t, mine) in &by_type {
            let cell = (e.event_type.as_str(), *t);
            attends.insert(cell);
            if diverges.contains(&cell) {
                continue;
            }
            let mut others: Vec<&str> = by_type
                .iter()
                .filter(|(tt, _)| *tt != t)
                .flat_map(|(_, v)| v.iter().copied())
                .collect();
            others.sort_unstable();
            let mut h = DefaultHasher::new();
            others.hash(&mut h);
            let digest = h.finish();

            let per_obj = seen.entry(cell).or_default();
            for o in mine {
                match per_obj.get(o) {
                    Some(prev) if *prev != digest => {
                        diverges.insert(cell);
                    }
                    None => {
                        per_obj.insert(o, digest);
                    }
                    _ => {}
                }
            }
            if diverges.contains(&cell) {
                seen.remove(&cell);
            }
        }
    }

    let mut per_type: HashMap<String, (usize, usize)> = HashMap::new();
    for (_, t) in &attends {
        per_type.entry((*t).to_string()).or_insert((0, 0)).1 += 1;
    }
    for (_, t) in &diverges {
        per_type.entry((*t).to_string()).or_insert((0, 0)).0 += 1;
    }

    let mut dropped: Vec<String> = per_type
        .iter()
        .filter(|(_, (div, tot))| {
            *tot >= MIN_ACTIVITIES && (*div as f64) >= divergence_share() * (*tot as f64)
        })
        .map(|(t, _)| t.clone())
        .collect();
    dropped.sort();
    (dropped, per_type)
}

/// The log a practitioner is left with: the objects of the dropped types are gone, and with
/// them every event-to-object and object-to-object edge that named one. No map is written.
fn filter_ocel(ocel: &OCEL, dropped: &HashSet<String>) -> OCEL {
    let gone: HashSet<&str> = ocel
        .objects
        .iter()
        .filter(|o| dropped.contains(&o.object_type))
        .map(|o| o.id.as_str())
        .collect();

    let objects = ocel
        .objects
        .iter()
        .filter(|o| !dropped.contains(&o.object_type))
        .map(|o| {
            let mut o = o.clone();
            o.relationships.retain(|r| !gone.contains(r.object_id.as_str()));
            o
        })
        .collect();
    let events = ocel
        .events
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.relationships.retain(|r| !gone.contains(r.object_id.as_str()));
            e
        })
        .collect();

    OCEL {
        event_types: ocel.event_types.clone(),
        object_types: ocel
            .object_types
            .iter()
            .filter(|t| !dropped.contains(&t.name))
            .cloned()
            .collect(),
        events,
        objects,
    }
}

struct Report {
    log: String,
    dropped: Vec<String>,
    divergence: Vec<(String, usize, usize)>,
    recorded_cells: usize,
    recorded_orderings: usize,
    recorded_ocpn: usize,
    recorded_ocdfg: usize,
    /// The per-cell oracle, scored without route credit.
    reflow_cells: usize,
    reflow_shown: usize,
    reflow_ocpn: usize,
    reflow_ocdfg: usize,
    /// Whole-type deletion, scored the same way.
    baseline_cells: usize,
    baseline_shown: usize,
    baseline_ocpn: usize,
    baseline_ocdfg: usize,
    baseline_activities_lost: usize,
    activities: usize,
    fact_repair_added: usize,
    full_roles: usize,
    reflow_roles: usize,
    baseline_roles: usize,
    full_orderfacts: usize,
    reflow_orderfacts: usize,
    baseline_orderfacts: usize,
    parts_total: usize,
    parts_inv: usize,
    parts_abs: usize,
    parts_deleted: usize,
    del_removed: usize,
    del_recoverable: usize,
}

fn run(path: &str) -> Report {
    let stem = log_stem(path);
    println!("\n{}\n{stem}\n{}", "=".repeat(72), "=".repeat(72));

    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let (dropped_names, divergence) = divergent_types(&ocel);
    let dropped: HashSet<String> = dropped_names.iter().cloned().collect();

    let mut div_rows: Vec<(String, usize, usize)> = divergence
        .into_iter()
        .map(|(t, (d, n))| (t, d, n))
        .collect();
    div_rows.sort_by(|a, b| {
        (b.1 as f64 / b.2 as f64)
            .partial_cmp(&(a.1 as f64 / a.2 as f64))
            .unwrap()
            .then(a.0.cmp(&b.0))
    });
    for (t, d, n) in &div_rows {
        let mark = if dropped.contains(t) { "DROP" } else { "    " };
        println!("  {mark}  {t:26} diverges at {d:2}/{n:2}");
    }

    let locel = SlimLinkedOCEL::from_ocel(ocel.clone());
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
    let n_acts = grid.activities.len();

    // What the recorded log asserts. This, not the saturation, is the population both sides are
    // scored against: neither may buy an expansion cell here.
    let mut rec_target: Vec<Pair> = asserted_by_type(&bounds, &grid.cells)
        .into_iter()
        .flatten()
        .collect();
    rec_target.sort_unstable();
    rec_target.dedup();

    // Drawn or chained only. No route term, on either side.
    let shown_without_routes = |kept: &HashSet<Cell>| -> usize {
        let d = drawn(&bounds, kept);
        let c = Chained::of(&d, n_acts);
        rec_target
            .iter()
            .filter(|p| d.contains(*p) || c.holds(p.0, p.1))
            .count()
    };

    // The same `fact_repair` pass `cross_instantiation` runs, so the arc counts here are the ones
    // Table 1 reports. Without it the raw layer reads 44 arcs on Container Logistics and 64 on
    // LRMS P2P, because fact repair is exactly what buys back the cells carrying a role fact
    // nothing else delivers.
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
    let mut reflow = reflow_layer(&input);
    let raw_flow: HashSet<Cell> = reflow.cells.clone();
    let added = fact_repair(&grid, &full_facts, &mut reflow.cells);
    let repaired = added.len();
    if repaired > 0 {
        let readable = |set: &HashSet<(usize, usize, usize)>, kept: &HashSet<Cell>| {
            set.iter()
                .filter(|(t, x, y)| kept.contains(&(*x, *t)) && kept.contains(&(*y, *t)))
                .count()
        };
        println!("  fact repair added {repaired} cell(s):");
        for (a, t) in &added {
            println!("      {} / {}", grid.activities[*a], schema.types[*t]);
        }
        for (name, set) in [
            ("ordering", &full_facts.ordering),
            ("role", &full_facts.role),
            ("asserted", &full_facts.asserted),
            ("tied", &full_facts.tied),
        ] {
            println!(
                "      {name:9} total {:4}  raw {:4} -> repaired {:4}",
                set.len(),
                readable(set, &raw_flow),
                readable(set, &reflow.cells),
            );
        }
    }

    let baseline_cells: HashSet<Cell> = grid
        .cells
        .iter()
        .filter(|(_, t)| !dropped.contains(&schema.types[*t]))
        .copied()
        .collect();

    let ocpn_opts = || ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(IMF_THRESHOLD));
    let net_recorded = discover_ocpn(&locel, ocpn_opts());
    let dfg_recorded = discover_dfg_from_ocel(&locel);

    let reflow_log = flow_projection(&tag(&locel, &schema, &acts, &reflow.cells, &[])).into_owned();
    let net_reflow = discover_ocpn(&reflow_log, ocpn_opts());
    let dfg_reflow = discover_dfg_from_ocel(&reflow_log);

    let baseline_log = SlimLinkedOCEL::from_ocel(filter_ocel(&ocel, &dropped));
    let net_baseline = discover_ocpn(&baseline_log, ocpn_opts());
    let dfg_baseline = discover_dfg_from_ocel(&baseline_log);

    // An activity every one of whose types went is gone from the model entirely, which no arc
    // count and no ordering count can show.
    let surviving_acts: HashSet<usize> = baseline_cells.iter().map(|(a, _)| *a).collect();
    let activities_lost = n_acts - surviving_acts.len();

    // Role facts, i.e. activity pairs no object of a type ever attends both of. This is what a
    // resource type carries and what no ordering measure sees. Reflow keeps an involved cell's
    // participations in the marked log, so its facts stay computable even though the cell draws
    // no arc; a deleted type takes its facts with it.
    let a = assign(&grid, &reflow.cells, &HashSet::new());
    let reflow_readable: HashSet<Cell> = a
        .flow
        .iter()
        .chain(a.involvement.iter())
        .copied()
        .collect();
    // A role fact `(T, a, b)` is still checkable exactly when `T`'s participations survive at
    // both activities, since that is what it takes to see that no object of `T` attends both.
    // `delivered_facts` answers a different question, whether some *other* type delivers it,
    // and reads 0 on Order Management for both sides.
    let roles_of = |kept: &HashSet<Cell>| -> usize {
        full_facts
            .role
            .iter()
            .filter(|(t, x, y)| kept.contains(&(*x, *t)) && kept.contains(&(*y, *t)))
            .count()
    };
    let reflow_roles = roles_of(&reflow_readable);
    let baseline_roles = roles_of(&baseline_cells);
    let full_roles = full_facts.role.len();

    // A type's own eventually-follows footprint, which aggregate coverage does not see:
    // coverage is satisfied when *any* kept type draws the pair, this asks whether the type
    // that asserted it still shows it.
    let orderfacts_of = |kept: &HashSet<Cell>| -> usize {
        full_facts
            .ordering
            .iter()
            .filter(|(t, x, y)| kept.contains(&(*x, *t)) && kept.contains(&(*y, *t)))
            .count()
    };
    let full_orderfacts = full_facts.ordering.len();

    // Participations, counted on the log another tool actually receives. Our export is the
    // hard cut, so the involved cells lose their participations too, and those are involved
    // precisely because no map determines them: they come back only from the side record.
    let inv_cells: HashSet<Cell> = a.involvement.iter().copied().collect();
    let abs_cells: HashSet<Cell> = a.implied.iter().copied().collect();
    let parts_total = max.participations(&grid.cells);
    let parts_inv = max.participations(&inv_cells);
    let parts_abs = max.participations(&abs_cells);
    let parts_deleted = parts_total - max.participations(&baseline_cells);

    // The paper's own condition, turned on the baseline: of the cells whole-type deletion
    // removed, how many do the types it kept at that activity determine? Ours is all of them
    // by construction. What deletion removes without that check is destroyed, not compressed.
    let mut del_removed = 0usize;
    let mut del_recoverable = 0usize;
    for (act, t) in grid.cells.iter() {
        if baseline_cells.contains(&(*act, *t)) {
            continue;
        }
        del_removed += 1;
        let Some(here) = grid.per_activity.get(*act) else { continue };
        let kept: Vec<usize> = here
            .present
            .iter()
            .filter(|s| baseline_cells.contains(&(*act, **s)))
            .copied()
            .collect();
        let reached = here.determined_by(&kept);
        if here.slot(*t).map(|j| reached[j]).unwrap_or(false) {
            del_recoverable += 1;
        }
    }
    println!(
        "  participations {parts_total}: reflow hard cut drops {} involved (side record only) \
and {parts_abs} implied (schema recomputes); deletion drops {parts_deleted}",
        parts_inv
    );
    println!(
        "  cells removed by deletion {del_removed}, of which {del_recoverable} determined by \
what it kept"
    );
    let reflow_orderfacts = orderfacts_of(&reflow_readable);
    let baseline_orderfacts = orderfacts_of(&baseline_cells);

    let r = Report {
        log: stem,
        dropped: dropped_names,
        divergence: div_rows,
        recorded_cells: grid.cells.len(),
        recorded_orderings: rec_target.len(),
        recorded_ocpn: ocpn_arcs(&net_recorded),
        recorded_ocdfg: ocdfg_arcs(&dfg_recorded),
        reflow_cells: reflow.cells.len(),
        reflow_shown: shown_without_routes(&reflow.cells),
        reflow_ocpn: ocpn_arcs(&net_reflow),
        reflow_ocdfg: ocdfg_arcs(&dfg_reflow),
        baseline_cells: baseline_cells.len(),
        baseline_shown: shown_without_routes(&baseline_cells),
        baseline_ocpn: ocpn_arcs(&net_baseline),
        baseline_ocdfg: ocdfg_arcs(&dfg_baseline),
        baseline_activities_lost: activities_lost,
        activities: n_acts,
        fact_repair_added: repaired,
        full_roles,
        reflow_roles,
        baseline_roles,
        full_orderfacts,
        reflow_orderfacts,
        baseline_orderfacts,
        parts_total,
        parts_inv,
        parts_abs,
        parts_deleted,
        del_removed,
        del_recoverable,
    };

    println!(
        "  dropped {:?}\n  orderings {}: reflow {} / baseline {}\n  OCPN {} -> reflow {} / baseline {}\n  OC-DFG {} -> reflow {} / baseline {}\n  role facts {}: reflow {} / baseline {}\n  activities lost by baseline: {}",
        r.dropped,
        r.recorded_orderings,
        r.reflow_shown,
        r.baseline_shown,
        r.recorded_ocpn,
        r.reflow_ocpn,
        r.baseline_ocpn,
        r.recorded_ocdfg,
        r.reflow_ocdfg,
        r.baseline_ocdfg,
        r.full_roles,
        r.reflow_roles,
        r.baseline_roles,
        r.baseline_activities_lost,
    );
    r
}

fn main() {
    let paths = corpus::logs_or_args();

    let reports: Vec<Report> = paths.iter().map(|p| run(p)).collect();

    println!("\n{}", "=".repeat(72));
    println!(
        "{:<20} {:>9} {:>18} {:>18}",
        "log", "orderings", "OCPN arcs", "OC-DFG arcs"
    );
    println!(
        "{:<20} {:>9} {:>6}{:>6}{:>6} {:>6}{:>6}{:>6}",
        "", "rfl/bas", "rec", "rfl", "bas", "rec", "rfl", "bas"
    );
    let (mut t_ord, mut t_rfl, mut t_bas) = (0, 0, 0);
    for r in &reports {
        t_ord += r.recorded_orderings;
        t_rfl += r.reflow_shown;
        t_bas += r.baseline_shown;
        println!(
            "{:<20} {:>4}/{:<4} {:>6}{:>6}{:>6} {:>6}{:>6}{:>6}",
            r.log,
            r.reflow_shown,
            r.baseline_shown,
            r.recorded_ocpn,
            r.reflow_ocpn,
            r.baseline_ocpn,
            r.recorded_ocdfg,
            r.reflow_ocdfg,
            r.baseline_ocdfg,
        );
    }
    println!(
        "{:<20} {:>4}/{:<4}  of {}",
        "corpus", t_rfl, t_bas, t_ord
    );

    let json = serde_json::json!({
        "criterion": {
            "divergence_share": divergence_share(),
            "min_activities": MIN_ACTIVITIES,
            "note": "reproduces the filterings of Liss et al. (Container Logistics) and \
                     Christfort et al. (Order Management)",
        },
        "route_credit": false,
        "corpus_totals": {
            "recorded_orderings": t_ord,
            "reflow_shown": t_rfl,
            "baseline_shown": t_bas,
            "reflow_orderings_lost": t_ord - t_rfl,
            "baseline_orderings_lost": t_ord - t_bas,
        },
        "logs": reports.iter().map(|r| serde_json::json!({
            "log": r.log,
            "dropped_types": r.dropped,
            "divergence": r.divergence.iter().map(|(t, d, n)| serde_json::json!({
                "type": t, "diverges_at": d, "activities": n,
            })).collect::<Vec<_>>(),
            "activities": r.activities,
            "fact_repair_added": r.fact_repair_added,
            "role_facts": {
                "total": r.full_roles,
                "reflow_readable": r.reflow_roles,
                "type_deletion": r.baseline_roles,
            },
            // The key names are the stored format of `results/stats/type_deletion.json`:
            // `..._involved_...` counts the involved participations and `..._absent_...` the
            // implied ones. Neither leaves the tagged log; only the flow projection drops them.
            "participations": {
                "total": r.parts_total,
                "reflow_involved_side_record_only": r.parts_inv,
                "reflow_absent_schema_recomputes": r.parts_abs,
                "type_deletion_dropped": r.parts_deleted,
            },
            "cells_removed_by_deletion": {
                "removed": r.del_removed,
                "determined_by_what_it_kept": r.del_recoverable,
            },
            "per_type_ordering_facts": {
                "total": r.full_orderfacts,
                "reflow_readable": r.reflow_orderfacts,
                "type_deletion": r.baseline_orderfacts,
            },
            "recorded": {
                "cells": r.recorded_cells,
                "orderings": r.recorded_orderings,
                "ocpn_arcs": r.recorded_ocpn,
                "ocdfg_arcs": r.recorded_ocdfg,
            },
            "reflow": {
                "cells_flowing": r.reflow_cells,
                "orderings_shown": r.reflow_shown,
                "ocpn_arcs": r.reflow_ocpn,
                "ocdfg_arcs": r.reflow_ocdfg,
            },
            "type_deletion": {
                "cells_kept": r.baseline_cells,
                "orderings_shown": r.baseline_shown,
                "ocpn_arcs": r.baseline_ocpn,
                "ocdfg_arcs": r.baseline_ocdfg,
                "activities_lost": r.baseline_activities_lost,
            },
        })).collect::<Vec<_>>(),
    });
    let suffix = if divergence_share() == 0.5 { String::new() } else { format!(".div{}", divergence_share()) };
    let out = corpus::stats_dir().join(format!("type_deletion{suffix}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json).unwrap()).expect("write report");
    println!("\nwrote {}", out.display());
}
