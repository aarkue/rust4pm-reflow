//! Fitness and precision of the models discovered before and after ReFlow (Sect. 5 of the paper).
//!
//! An OCPN is one flat Petri net per object type, stitched on shared transition labels
//! ([`discover_ocpn`]), so the unit of comparison is the per-type flat net: each component is
//! mined from that type's flattening and replayed against it.
//!
//! Both models are replayed against the recorded log's flattening. Replaying the model
//! mined from the flow projection against that projection would score it against the
//! evidence the projection drops.
//!
//! A type with no flow cell has no component in that model and is reported as `absent`
//! (no component), not scored.
//!
//! Usage: `cargo run --release -- [<log> ...]`. With no arguments, runs the corpus in
//! `REFLOW_LOGS`. Writes `<log>.keepsets.json`, `<log>.scores.json` and `<log>.stats.json`
//! to `REFLOW_STATS`.

#[path = "../../../examples/corpus/mod.rs"]
mod corpus;

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        agreed_routes, assign, carriers_of, fact_repair, facts_from, flow_projection,
        involvement_clusters, reconstruct_ocpn, reflow_layer, tag, tags, AbsenceRender,
        ActivityIndexing, Bounds, CellGrid, ExpansionDirection, InvolvementRender, Saturation,
        SchemaClosure, SearchInput, StructuralSchema, TraceVariants, DEFAULT_THETA,
    },
    conformance::case_centric::alignments::{align_log, compute_fitness, AlignmentOptions},
    conformance::object_centric::{
        binding_semantics::ObjectRelations,
        oc_precision::{oc_conformance, OcConformanceOptions, OcEvent},
    },
    core::event_data::case_centric::utils::activity_projection::EventLogActivityProjection,
    core::event_data::object_centric::{
        linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL},
        utils::flatten::flatten_ocel_on,
    },
    core::process_models::case_centric::petri_net::{ArcType, Marking},
    core::process_models::object_centric::ocpn::ObjectCentricPetriNet,
    discovery::case_centric::inductive_miner::InductiveMinerOptions,
    discovery::object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions},
    Exportable, Importable, OCEL,
};

/// Places, transitions and arcs of the whole object-centric net.
fn size(net: &ObjectCentricPetriNet) -> (usize, usize, usize) {
    net.nets.values().fold((0, 0, 0), |(p, t, a), n| {
        (
            p + n.places.len(),
            t + n.transitions.len(),
            a + n.arcs.len(),
        )
    })
}

/// How a component scored, kept apart from each other because they mean different things.
#[derive(Clone)]
enum Score {
    /// Log fitness against the recorded flattening.
    Fit(f64),
    /// The component has no labelled transition left: no cell of this type flows.
    /// Every event is a log move, so "fitness 0" would be a fact about the arithmetic and
    /// not about the model.
    Emptied,
    /// The net has no component for this type at all. A property of the net, not a cell
    /// state: it is unrelated to the implied cells.
    Absent,
    /// Alignment did not finish. Not a score in either direction; the reason is carried
    /// because "too big" and "malformed net" need different fixes.
    Failed(String),
}

impl std::fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Score::Fit(x) => write!(f, "{x:.4}"),
            Score::Emptied => write!(f, "emptied"),
            Score::Absent => write!(f, "absent"),
            Score::Failed(_) => write!(f, "n/a"),
        }
    }
}

fn labelled(net: &process_mining::PetriNet) -> usize {
    net.transitions
        .values()
        .filter(|t| t.label.is_some())
        .count()
}

/// Per-trace A* state budget. 10M states on a trace that cannot be aligned (BPIC2017's
/// resource type: 145 objects, thousands of events each) is gigabytes of queue and hours
/// of swap before the same `Failed` verdict the default budget reaches in seconds.
fn max_states() -> usize {
    std::env::var("MAX_STATES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000_000)
}

fn score_of(net: Option<&process_mining::PetriNet>, log: &EventLogActivityProjection) -> Score {
    let Some(net) = net else { return Score::Absent };
    if labelled(net) == 0 {
        return Score::Emptied;
    }
    let opts = AlignmentOptions {
        max_states: Some(max_states()),
        ..Default::default()
    };
    let res = align_log(net, log, &opts);
    match compute_fitness(&res, net, &opts) {
        Ok(f) => Score::Fit(f.log_fitness),
        Err(e) => Score::Failed(format!("{e:?}").chars().take(22).collect()),
    }
}

/// Escaping-edges precision (Munoz-Gama and Carmona), on the alignment rather than on replay,
/// so it is defined for a log the net does not fit.
fn precision_of(
    net: Option<&process_mining::PetriNet>,
    log: &EventLogActivityProjection,
) -> Score {
    let Some(net) = net else { return Score::Absent };
    if labelled(net) == 0 {
        return Score::Emptied;
    }
    let opts = AlignmentOptions {
        max_states: Some(max_states()),
        ..Default::default()
    };
    match ee_precision::ee_precision_with(log, net, &opts, 200_000) {
        Ok(p) => Score::Fit(p.value),
        Err(e) => Score::Failed(format!("{e:?}").chars().take(22).collect()),
    }
}

/// A floor this close to 1 leaves no room to divide by: a type attending one activity has
/// a flower that permits exactly its log, so the adjusted scale is not reported.
const FLOOR_HEADROOM: f64 = 0.01;

/// The precision floor for a type: one place, every activity a self-loop on it.
///
/// Precision has no natural zero. A model that permits everything still scores well above
/// 0, and how far above depends on how many activities the type has and how the log's
/// prefixes distribute, so a bare percentage change between two models is not readable on
/// its own. The flower over the type's own activities is the "asserts nothing" end of the
/// scale, which is what Adams and van der Aalst calibrate against.
fn flower(log: &EventLogActivityProjection) -> process_mining::PetriNet {
    let mut net = process_mining::PetriNet::new();
    let p = net.add_place(None);
    for a in &log.activities {
        let t = net.add_transition(Some(a.clone()), None);
        net.add_arc(ArcType::place_to_transition(p, t), None);
        net.add_arc(ArcType::transition_to_place(t, p), None);
    }
    let mut m = Marking::default();
    m.insert(p, 1);
    net.initial_marking = Some(m.clone());
    net.final_markings = Some(vec![m]);
    net
}

/// Relative change against the recorded model, for the table.
fn relative_change(before: &Score, after: &Score) -> String {
    match (before, after) {
        (Score::Fit(b), Score::Fit(a)) if *b > 0.0 => format!("{:+.1}%", (a - b) / b * 100.0),
        _ => "--".to_string(),
    }
}

thread_local! {
    static PROBE: std::cell::RefCell<Option<ObjectCentricPetriNet>> =
        const { std::cell::RefCell::new(None) };
}
fn net_after_probe(ot: &str, label: &str) -> bool {
    PROBE.with(|p| {
        p.borrow()
            .as_ref()
            .and_then(|n| n.nets.get(ot))
            .map(|n| n.transitions.values().any(|t| t.label.as_deref() == Some(label)))
            .unwrap_or(false)
    })
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
    let routes = agreed_routes(&schema);
    let max = Saturation::build(
        &locel,
        &schema,
        &grid,
        &acts,
        &routes.0,
        &bounds,
        DEFAULT_THETA,
        ExpansionDirection::default(),
    );

    let mut target: Vec<_> = max.target_pairs().into_iter().collect();
    target.sort_unstable();
    let variants = TraceVariants::build_with(&locel, &schema, &acts, &max.written);
    let objects = Saturation::objects_per_type(&schema);
    // Recorded cells only. Expansion writes participations the extraction never had, and
    // both models here are scored against the *recorded* log -- so an expanded cell makes
    // the reduced model expect events that log cannot supply. Expansion is a separate
    // claim; mixing it in measures the two at once.
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
    // The reported layer, then the three-case repair: an ordering fact must
    // be drawn by its own type or excused by determination at each endpoint, so cells the
    // type-agnostic coverage took out of the flow layer come back where nothing licenses
    // their absence.
    let mut best = reflow_layer(&input);
    let full_facts = facts_from(&max.bounds, &grid.cells, 0.0);
    let repaired_added = fact_repair(&grid, &full_facts, &mut best.cells);
    if !repaired_added.is_empty() {
        println!(
            "  fact repair: +{} cells promoted to flow",
            repaired_added.len()
        );
        if std::env::var("REPAIRDBG").is_ok() {
            for (a, t) in &repaired_added {
                println!(
                    "    promoted: {} @ {}",
                    schema.types[*t], grid.activities[*a]
                );
            }
        }
    }
    if let Ok(p) = std::env::var("KEEPSET_OVERRIDE") {
        let txt = std::fs::read_to_string(&p).expect("read keepset override");
        let named: Vec<(String, String)> =
            serde_json::from_str(&txt).expect("parse keepset override");
        best.cells = named
            .iter()
            .filter_map(|(a, t)| {
                let ai = grid.activities.iter().position(|x| x == a)?;
                let ti = schema.types.iter().position(|x| x == t)?;
                Some((ai, ti))
            })
            .collect();
        println!(
            "  keepset override: {} of {} named cells resolved from {}",
            best.cells.len(),
            named.len(),
            p
        );
    }
    // The flow layer may contain expanded cells -- ones the extraction never recorded --
    // and the projection only removes. Writing them too is what makes the flow projection
    // actually support the layer: without it a type flows at an activity whose
    // participations are missing, the miner gives its component no transition there, and
    // anything carried by it has nothing to be drawn from.
    let added: Vec<_> = max
        .written
        .iter()
        .filter(|(e, o)| {
            let a = acts.act_of[e.get_ev(&locel).event_type];
            best.cells.contains(&(a, schema.type_of[o])) && !grid.cells.contains(&(a, schema.type_of[o]))
        })
        .copied()
        .collect();
    let tagged_log = tag(&locel, &schema, &acts, &best.cells, &added);
    let reduced_log = flow_projection(&tagged_log).into_owned();
    let assignment = assign(&grid, &best.cells, &Default::default());

    // Carriers and the population split both come from the library, so the drawn model
    // here is the one any consumer gets from `repair_ocpn`.
    let carriers = carriers_of(&grid, &max.bounds, &max.cells, &best.cells);
    if std::env::var("CARRIERS").is_ok() {
        let mut v: Vec<_> = carriers.iter().collect();
        v.sort();
        for ((a, t), s) in v {
            let has = net_after_probe(&schema.types[*s], &grid.activities[*a]);
            println!(
                "  carrier {:<22} at {:<28} <- {:<22} carrier_has_label={has}",
                schema.types[*t], grid.activities[*a], schema.types[*s]
            );
        }
    }

    // IMf at the noise threshold given by `IMF` (default 0.2), used for every configuration
    // so the three models differ only in the log they were mined from.
    let mut clusters = involvement_clusters(&max.bounds);
    // The population split is per-object discovery, not annotation, so it is opt-in like
    // the rest of the enrichment; the default is one place per type, the plain reading of
    // involvement. A separate flag from `ENRICH_REP` (which gates precedence places and
    // block edges inside `reconstruct_ocpn`), so the two enrichments can be measured apart
    // -- the paper reports the precision each buys on its own, not stacked.
    if std::env::var("POP_CLUSTERS").is_err() {
        clusters.clear();
    }
    if std::env::var("SETS").is_ok() {
        for t in 0..schema.types.len() {
            let mut per: std::collections::BTreeMap<Vec<usize>, usize> = Default::default();
            for ob in max.bounds.per_type[t].iter() {
                let mut a: Vec<usize> = ob.at.iter().map(|(x, _, _)| *x).collect();
                a.sort_unstable();
                a.dedup();
                *per.entry(a).or_default() += 1;
            }
            if per.len() < 2 || per.len() > 8 {
                continue;
            }
            println!("  sets {} ({} objects)", schema.types[t], max.bounds.per_type[t].len());
            for (a, n) in &per {
                let names: Vec<&str> = a.iter().map(|i| grid.activities[*i].as_str()).collect();
                println!("      {n:>5} obj: {names:?}");
            }
        }
    }
    if std::env::var("CLUSTERS").is_ok() {
        for (t, sets) in clusters.iter() {
            // Objects per population, and whether the sets are nested (a truncated
            // lifecycle) or genuinely disjoint (distinct kinds of object).
            let mut pop: Vec<usize> = vec![0; sets.len()];
            for ob in max.bounds.per_type[*t].iter() {
                let mut acts: Vec<usize> = ob.at.iter().map(|(a, _, _)| *a).collect();
                acts.sort_unstable();
                acts.dedup();
                if let Some(i) = sets.iter().position(|s| *s == acts) {
                    pop[i] += 1;
                }
            }
            let nested = sets.iter().any(|a| {
                sets.iter().any(|b| a != b && a.iter().all(|x| b.contains(x)))
            });
            pop.sort_unstable_by(|a, b| b.cmp(a));
            println!(
                "  clusters {:<22}{:>3} pops of {:>6} objects  nested={:<5} sizes={:?}",
                schema.types[*t],
                sets.len(),
                max.bounds.per_type[*t].len(),
                nested,
                &pop[..pop.len().min(6)]
            );
        }
    }

    let f: f64 = std::env::var("IMF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.2);
    let opts = ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(f));
    let mut net_before = discover_ocpn(&locel, opts.clone());
    let mut net_after = discover_ocpn(&reduced_log, opts);
    PROBE.with(|p| *p.borrow_mut() = Some(net_after.clone()));
    let (mut net_rebuilt, _place_roles) = reconstruct_ocpn(
        &net_after,
        &assignment,
        &carriers,
        &clusters,
        &max.bounds,
        &schema.types,
        &grid.activities,
        InvolvementRender::Connected,
        AbsenceRender::Mapped,
    );
    // Silent structure that constrains nothing is not size: dead `p -> tau -> q` chains,
    // single-branch choice skeletons, and emptied types' husks all collapse, on all three
    // nets alike, so the arc columns compare models rather than construction verbosity.
    for net in [&mut net_before, &mut net_after, &mut net_rebuilt] {
        for component in net.nets.values_mut() {
            component.simplify_silent();
        }
    }

    let (pb, tb, ab) = size(&net_before);
    let (pa, ta, aa) = size(&net_after);
    let (pr, tr, ar) = size(&net_rebuilt);
    println!("\n=== {path}   IMf {f}   ({:.1}s)", t0.elapsed().as_secs_f64());
    println!(
        "  strategy {}   flow cells {}   arcs(model-side objective) {}",
        best.strategy.label(),
        best.cells.len(),
        best.arcs
    );
    println!(
        "  OCPN size   before  {pb:>5} places {tb:>5} transitions {ab:>6} arcs   ({} types)",
        net_before.nets.len()
    );
    println!(
        "              after   {pa:>5} places {ta:>5} transitions {aa:>6} arcs   ({} types)",
        net_after.nets.len()
    );
    println!(
        "              rebuilt {pr:>5} places {tr:>5} transitions {ar:>6} arcs   ({} types)",
        net_rebuilt.nets.len()
    );

    // Per-type fitness, both nets against the RECORDED flattening.
    println!(
        "\n  {:<22}{:>9}{:>9}{:>9}{:>8}{:>9}{:>9}{:>8}{:>10}{:>10}{:>10}{:>8}{:>9}{:>9}{:>8}",
        "object type",
        "arc(rec)",
        "arc(red)",
        "arc(rep)",
        "d.arc",
        "fit(rec)",
        "fit(rep)",
        "d.fit",
        "prec(flr)",
        "prec(rec)",
        "prec(rep)",
        "d.prec",
        "adj(rec)",
        "adj(rep)",
        "d.adj"
    );
    let mut types: Vec<String> = schema.types.to_vec();
    types.sort();
    let count = |cells: &[(usize, usize)], t: usize| cells.iter().filter(|c| c.1 == t).count();
    // Fibre of the recovery map: how many carrier objects share one object of this type.
    // The projection copies the carrier's control flow, which is one object's path, so it
    // can only be right where that ratio is 1.
    let fibre = |t: usize| -> f64 {
        let mut worst: f64 = 0.0;
        for ((a, ct), s) in carriers.iter() {
            if *ct != t {
                continue;
            }
            let _ = a;
            for m in schema.recorded.iter().chain(schema.derived.iter()) {
                if m.source == *s && m.target == t && m.image > 0 {
                    worst = worst.max(m.f.len() as f64 / m.image as f64);
                }
            }
        }
        worst
    };
    #[allow(clippy::type_complexity)]
    let mut rows: BTreeMap<
        String,
        (usize, usize, usize, usize, usize, usize, Score, Score, Score),
    > = BTreeMap::new();
    let mut precs: BTreeMap<String, (Score, Score, Score)> = BTreeMap::new();
    let only: Option<Vec<String>> = std::env::var("TYPES")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    for ot in &types {
        if let Some(only) = &only {
            if !only.contains(ot) {
                continue;
            }
        }
        let flat = flatten_ocel_on(&locel, ot);
        let n_cases = flat.traces.len();
        if n_cases == 0 {
            continue;
        }
        let proj: EventLogActivityProjection = (&flat).into();
        let ti = schema.types.iter().position(|x| x == ot).unwrap();
        let tb = net_before.nets.get(ot).map_or(0, labelled);
        let tr = net_rebuilt.nets.get(ot).map_or(0, labelled);
        let before = score_of(net_before.nets.get(ot), &proj);
        let after = score_of(net_after.nets.get(ot), &proj);
        let rebuilt = score_of(net_rebuilt.nets.get(ot), &proj);
        let floor = flower(&proj);
        precs.insert(
            ot.clone(),
            (
                precision_of(Some(&floor), &proj),
                precision_of(net_before.nets.get(ot), &proj),
                precision_of(net_rebuilt.nets.get(ot), &proj),
            ),
        );
        rows.insert(
            ot.clone(),
            (
                n_cases,
                count(&assignment.flow, ti),
                count(&assignment.involvement, ti),
                count(&assignment.implied, ti),
                tb,
                tr,
                before,
                after,
                rebuilt,
            ),
        );
    }
    // Per-type record for the deposited JSON. The table reports object-weighted
    // aggregates only, which cannot be reopened into a per-type minimum after the
    // fact, so every quantity the loop computes is written out alongside it.
    let mut per_type_json: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let score_json = |s: &Score| match s {
        Score::Fit(v) => serde_json::json!(v),
        Score::Emptied => serde_json::json!("emptied"),
        Score::Absent => serde_json::json!("absent"),
        Score::Failed(e) => serde_json::json!({ "failed": e }),
    };
    for (ot, (n, fl, iv, ab, _tb, tr, b, a, r)) in &rows {
        let ti = schema.types.iter().position(|x| x == ot).unwrap();
        let _ = fibre(ti);
        // The four rendering classes. `abs` splits on whether the type orders anything:
        // with an ordering the carrier carries it, without one nothing is drawn.
        let with = carriers.keys().filter(|(_, t)| *t == ti).count();
        let none = ab - with;
        let fbs = none.to_string();
        let (pf, pb, pr) = precs
            .get(ot)
            .cloned()
            .unwrap_or((Score::Absent, Score::Absent, Score::Absent));
        // Precision with the floor taken out: 0 is a flower over the type's activities, 1 is
        // a model that permits exactly the log. Dividing by the room that is left rather
        // than by the miner's own margin keeps both models on one scale and stops a type
        // whose miner barely beat the flower from reporting a vast relative change.
        let adjusted = |s: &Score| match (&pf, s) {
            (Score::Fit(f), Score::Fit(v)) if 1.0 - f > FLOOR_HEADROOM => {
                format!("{:.4}", (v - f) / (1.0 - f))
            }
            _ => "--".to_string(),
        };
        let (ab_, ar_) = (adjusted(&pb), adjusted(&pr));
        let d_adj = match (&pf, &pb, &pr) {
            (Score::Fit(f), Score::Fit(b), Score::Fit(r)) if 1.0 - f > FLOOR_HEADROOM => {
                format!("{:+.3}", (r - b) / (1.0 - f))
            }
            _ => "--".to_string(),
        };
        let arc_rec = net_before.nets.get(ot).map_or(0, |x| x.arcs.len());
        let arc_rep = net_rebuilt.nets.get(ot).map_or(0, |x| x.arcs.len());
        let d_arc = if arc_rec > 0 {
            format!(
                "{:+.0}%",
                (arc_rep as f64 - arc_rec as f64) / arc_rec as f64 * 100.0
            )
        } else {
            "--".to_string()
        };
        println!(
            "  {ot:<22}{:>9}{:>9}{:>9}{:>8}{:>9}{:>9}{:>8}{:>10}{:>10}{:>10}{:>8}{:>9}{:>9}{:>8}",
            arc_rec,
            net_after.nets.get(ot).map_or(0, |x| x.arcs.len()),
            arc_rep,
            d_arc,
            b.to_string(),
            r.to_string(),
            relative_change(b, r),
            pf.to_string(),
            pb.to_string(),
            pr.to_string(),
            relative_change(&pb, &pr),
            ab_,
            ar_,
            d_adj
        );
        // The adjusted values again, as numbers rather than the table's strings.
        let adj_num = |s: &Score| match (&pf, s) {
            (Score::Fit(f), Score::Fit(v)) if 1.0 - f > FLOOR_HEADROOM => Some((v - f) / (1.0 - f)),
            _ => None,
        };
        per_type_json.insert(
            ot.clone(),
            serde_json::json!({
                "objects": n,
                // `absent` is the stored file format of the published results; the cells
                // it counts are the implied ones.
                "cells": { "flow": fl, "inv": iv, "absent": ab, "carried": with },
                "arcs": {
                    "recorded": arc_rec,
                    "reduced": net_after.nets.get(ot).map_or(0, |x| x.arcs.len()),
                    "rebuilt": arc_rep,
                },
                "fitness": { "recorded": score_json(b), "rebuilt": score_json(r) },
                "precision": {
                    "floor": score_json(&pf),
                    "recorded": score_json(&pb),
                    "rebuilt": score_json(&pr),
                },
                "adjusted_precision": {
                    "recorded": adj_num(&pb),
                    "rebuilt": adj_num(&pr),
                },
            }),
        );
        let _ = &a;
        let _ = (fbs, tr);
        if std::env::var("CELLS").is_ok() {
            println!("      cells: n={n} flow={fl} inv={iv} implied={ab} (carried {with})");
        }
    }
    {
        let dir = corpus::stats_dir().to_string_lossy().into_owned();
        let stem = log_stem(path);
        let _ = std::fs::create_dir_all(&dir);
        let out = format!("{dir}/{stem}.scores.json");
        // Aggregated here rather than by the reader: the floor is per type, so an
        // unweighted mean of adjusted precisions is not derivable from the weighted one.
        let adj_of = |k: &str, t: &serde_json::Value| t["adjusted_precision"][k].as_f64();
        let vals: Vec<&serde_json::Value> = per_type_json.values().collect();
        let unweighted = |k: &str| -> Option<f64> {
            let xs: Vec<f64> = vals.iter().filter_map(|t| adj_of(k, t)).collect();
            (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
        };
        let fits: Vec<f64> = vals
            .iter()
            .filter_map(|t| t["fitness"]["rebuilt"].as_f64())
            .collect();
        let drops: Vec<&String> = per_type_json
            .iter()
            .filter(|(_, t)| {
                matches!(
                    (t["fitness"]["recorded"].as_f64(), t["fitness"]["rebuilt"].as_f64()),
                    (Some(x), Some(y)) if y < x - 1e-9
                )
            })
            .map(|(k, _)| k)
            .collect();
        // The cells themselves, so the figures regenerate from the reported
        // assignment instead of from a keep-set file with no producer.
        let cells = |v: &[(usize, usize)]| -> Vec<[String; 2]> {
            let mut out: Vec<[String; 2]> = v
                .iter()
                .map(|(a, t)| [grid.activities[*a].clone(), schema.types[*t].clone()])
                .collect();
            out.sort();
            out
        };
        let (flow_c, inv_c, abs_c) = (
            cells(&assignment.flow),
            cells(&assignment.involvement),
            cells(&assignment.implied),
        );
        let mut kept = flow_c.clone();
        kept.extend(inv_c.iter().cloned());
        kept.sort();
        // A keep-set file in the shape make_cell_grid.py reads, from this run. The key
        // `absent` is the stored file format the published results and the figure scripts
        // already use; the cells it names are the implied ones.
        let ks = format!("{dir}/{stem}.keepsets.json");
        let ks_doc = serde_json::json!({
            "measured": kept, "flow": flow_c, "inv": inv_c, "absent": abs_c,
        });
        match serde_json::to_string_pretty(&ks_doc)
            .map_err(|e| e.to_string())
            .and_then(|t| std::fs::write(&ks, t).map_err(|e| e.to_string()))
        {
            Ok(()) => println!("  keepset {ks}"),
            Err(e) => println!("  keepset {ks}: {e}"),
        }
        let doc = serde_json::json!({
            "log": stem,
            "types": per_type_json,
            "assignment": {
                "flow": flow_c, "inv": inv_c, "absent": abs_c, "kept": kept,
            },
            "aggregates": {
                "min_fitness_rebuilt": fits.iter().cloned().fold(f64::INFINITY, f64::min),
                "types_whose_fitness_falls": drops,
                "mean_adjusted_precision_unweighted": {
                    "recorded": unweighted("recorded"),
                    "rebuilt": unweighted("rebuilt"),
                },
            },
        });
        match serde_json::to_string_pretty(&doc).map_err(|e| e.to_string()).and_then(|t| {
            std::fs::write(&out, t).map_err(|e| e.to_string())
        }) {
            Ok(()) => println!("  stats {out}"),
            Err(e) => println!("  stats {out}: {e}"),
        }
    }
    // Aggregate: totals for arcs, and means over the types both models score, weighted by
    // how many objects of the type there are.
    //
    // Unweighted, a component covering 15 customers counts as much as one covering 7,659
    // items, and on Order Management that alone decided the sign: the two smallest types in
    // the log carry the whole precision loss, and an unweighted mean read -13.3% where the
    // weighted one reads +1.5%. The question the table answers is whether the surviving model
    // is worse, and two components are not equal evidence for that when one explains 500
    // times more of the log.
    let mean = |pick: &dyn Fn(&String) -> Option<(f64, f64)>| -> Option<f64> {
        let vals: Vec<(f64, f64)> = rows.keys().filter_map(pick).collect();
        let total: f64 = vals.iter().map(|(w, _)| w).sum();
        (total > 0.0).then(|| vals.iter().map(|(w, v)| w * v).sum::<f64>() / total)
    };
    let paired = |f: &dyn Fn(&String) -> (Score, Score), first: bool| -> Option<f64> {
        mean(&|ot: &String| match f(ot) {
            (Score::Fit(b), Score::Fit(r)) => Some((rows[ot].0 as f64, if first { b } else { r })),
            _ => None,
        })
    };
    let fit_pair = |ot: &String| {
        let (_, _, _, _, _, _, b, _, r) = rows[ot].clone();
        (b, r)
    };
    let prec_pair = |ot: &String| {
        let (_, b, r) = precs
            .get(ot)
            .cloned()
            .unwrap_or((Score::Absent, Score::Absent, Score::Absent));
        (b, r)
    };
    let floor_pair = |ot: &String| {
        let (f, _, r) = precs
            .get(ot)
            .cloned()
            .unwrap_or((Score::Absent, Score::Absent, Score::Absent));
        (f, r)
    };
    let show = |v: Option<f64>| v.map_or("--".to_string(), |x| format!("{x:.4}"));
    let pct = |b: Option<f64>, a: Option<f64>| match (b, a) {
        (Some(b), Some(a)) if b > 0.0 => format!("{:+.1}%", (a - b) / b * 100.0),
        _ => "--".to_string(),
    };
    let (fb, fr) = (paired(&fit_pair, true), paired(&fit_pair, false));
    let (qb, qr) = (paired(&prec_pair, true), paired(&prec_pair, false));
    let qf = paired(&floor_pair, true);
    // The adjusted mean is taken per type and then weighted, not derived from the weighted
    // raw means: each type has its own floor, so there is no single floor to subtract.
    let adj_mean = |want_rep: bool| -> Option<f64> {
        mean(&|ot: &String| match precs.get(ot) {
            Some((Score::Fit(f), Score::Fit(b), Score::Fit(r))) if 1.0 - f > FLOOR_HEADROOM => Some((
                rows[ot].0 as f64,
                ((if want_rep { *r } else { *b }) - f) / (1.0 - f),
            )),
            _ => None,
        })
    };
    let (qab, qar) = (adj_mean(false), adj_mean(true));
    let d_adj = match (qab, qar) {
        (Some(b), Some(r)) => format!("{:+.3}", r - b),
        _ => "--".to_string(),
    };
    println!(
        "  {:<22}{:>9}{:>9}{:>9}{:>8}{:>9}{:>9}{:>8}{:>10}{:>10}{:>10}{:>8}{:>9}{:>9}{:>8}",
        "TOTAL / weighted",
        ab,
        aa,
        ar,
        if ab > 0 {
            format!("{:+.0}%", (ar as f64 - ab as f64) / ab as f64 * 100.0)
        } else {
            "--".to_string()
        },
        show(fb),
        show(fr),
        pct(fb, fr),
        show(qf),
        show(qb),
        show(qr),
        pct(qb, qr),
        show(qab),
        show(qar),
        d_adj
    );

    // Object-centric fitness and precision (Adams and van der Aalst), twice: with the net
    // alone, and with the schema's maps constraining which objects a binding may combine.
    // The difference is what the schema is worth, since an OCPN has no syntax for it.
    if let Ok(limit) = std::env::var("OCP") {
        let cap: usize = limit.parse().unwrap_or(2000);
        let events: Vec<OcEvent> = locel
            .get_all_evs()
            .take(cap)
            .map(|e| OcEvent {
                activity: e.get_ev_type(&locel).clone(),
                objects: e
                    .get_e2o(&locel)
                    .map(|o| {
                        (
                            o.get_ob(&locel).id.clone(),
                            schema.types[schema.type_of[&o]].clone(),
                        )
                    })
                    .collect(),
            })
            .collect();

        let mut relations = ObjectRelations::new();
        for m in schema.recorded.iter().chain(schema.derived.iter()) {
            let f = m
                .f
                .iter()
                .map(|(a, b)| (a.get_ob(&locel).id.clone(), b.get_ob(&locel).id.clone()))
                .collect();
            relations.add_map(
                schema.types[m.source].clone(),
                schema.types[m.target].clone(),
                f,
            );
        }

        println!("\n  object-centric, first {} events", events.len());
        println!(
            "  {:<12}{:<10}{:>10}{:>12}{:>10}",
            "model", "bindings", "fitness", "precision", "skipped"
        );
        for (tag, net) in [("recorded", &net_before), ("repaired", &net_rebuilt)] {
            for (kind, opts) in [
                ("net only", OcConformanceOptions::default()),
                (
                    "+ schema",
                    OcConformanceOptions::with_relations(relations.clone()),
                ),
            ] {
                let t = Instant::now();
                let _ = process_mining::conformance::object_centric::oc_precision::take_stats();
                let r = oc_conformance(net, &events, &opts);
                let st = process_mining::conformance::object_centric::oc_precision::take_stats();
                println!(
                    "      preset_edges={} contexts={} replay_fires={} advance_calls={} advance_states={} enabled_calls={}",
                    st[0], st[1], st[2], st[3], st[4], st[5]
                );
                println!(
                    "  {tag:<12}{kind:<10}{:>10.4}{:>12.4}{:>9.0}%   ({:.1}s)",
                    r.fitness,
                    r.precision,
                    r.skipped_share() * 100.0,
                    t.elapsed().as_secs_f64()
                );
            }
        }
    }

    // Object-centric fitness and precision (Adams and van der Aalst): scores the net as an
    // OCPN rather than as a stack of flat nets, so cross-type synchronisation counts. Read
    // against their scale, where a correct model reaches ~0.57, not against flat precision.
    if let Ok(limit) = std::env::var("OCP") {
        let cap: usize = limit.parse().unwrap_or(2000);
        let mut events: Vec<OcEvent> = locel
            .get_all_evs()
            .map(|e| OcEvent {
                activity: e.get_ev_type(&locel).clone(),
                objects: e
                    .get_e2o(&locel)
                    .map(|o| {
                        (
                            o.get_ob(&locel).id.clone(),
                            schema.types[schema.type_of[&o]].clone(),
                        )
                    })
                    .collect(),
            })
            .collect();
        events.truncate(cap);
        let opts = OcConformanceOptions::default();
        println!("\n  object-centric, first {} events", events.len());
        println!("  {:<12}{:>10}{:>12}{:>10}", "model", "fitness", "precision", "skipped");
        for (tag, net) in [("recorded", &net_before), ("repaired", &net_rebuilt)] {
            let t = Instant::now();
            let r = oc_conformance(net, &events, &opts);
            println!(
                "  {tag:<12}{:>10.4}{:>12.4}{:>9.0}%   ({:.1}s)",
                r.fitness,
                r.precision,
                r.skipped_share() * 100.0,
                t.elapsed().as_secs_f64()
            );
        }
    }

    // Round trip at the level of the log rather than the model: tag by the flow layer and
    // check that dropping the tags returns the log that went in. If this is exact then a
    // repair that has the log needs no drawing at all -- rediscovery returns the recorded
    // model -- and every loss the drawn repair reports is the price of refusing to look at
    // the log.
    if std::env::var("ROUNDTRIP").is_ok() {
        let e2o = |l: &SlimLinkedOCEL| -> std::collections::HashSet<(String, String)> {
            l.get_all_evs()
                .flat_map(|e| {
                    let ev = e.get_ev(l).id.clone();
                    e.get_e2o(l)
                        .map(|o| (ev.clone(), o.get_ob(l).id.clone()))
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let original = e2o(&locel);
        let projected = e2o(&reduced_log);
        println!(
            "  round trip   {} tuples -> {} in the flow projection -> full projection {} L",
            original.len(),
            projected.len(),
            if tags(&tagged_log, &locel) { "==" } else { "!=" }
        );
    }

    // Export for an external precision check. Only the types named in `EXPORT`, and only
    // the recorded and rebuilt models: the one mined from the flow projection has no
    // component for a type nothing flows, so there is nothing to score.
    // The whole object-centric net, in the exchange JSON, so the repaired model can be
    // opened and looked at rather than only scored. Written for all three configurations:
    // what the miner found, what the reduction leaves, and what the repair draws.
    if let Ok(dir) = std::env::var("EXPORT_DIR") {
        let stem = log_stem(path);
        let stem = stem.as_str();
        // The flow projection itself, not just the models discovered from it.
        let out = format!("{dir}/{stem}.reduced.ocel.xml.gz");
        match reduced_log.export_to_path(&out) {
            Ok(()) => println!("  log  {out}"),
            Err(e) => println!("  log  {out}: {e}"),
        }
        for (tag, net) in [
            ("recorded", &net_before),
            ("reduced", &net_after),
            ("repaired", &net_rebuilt),
        ] {
            let json = net.to_json_form();
            let out = format!("{dir}/{stem}.{tag}.ocpn.json");
            match serde_json::to_string_pretty(&json)
                .map_err(|e| e.to_string())
                .and_then(|t| std::fs::write(&out, t).map_err(|e| e.to_string()))
            {
                Ok(()) => println!("  ocpn {out}"),
                Err(e) => println!("  ocpn {out}: {e}"),
            }
        }
    }

    if let Ok(want) = std::env::var("EXPORT") {
        let dir = std::env::var("EXPORT_DIR").unwrap_or_else(|_| ".".to_string());
        for ot in want.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            let flat = flatten_ocel_on(&locel, ot);
            let stem = format!("{dir}/{}", ot.replace(' ', "_"));
            flat.export_to_path(format!("{stem}.xes")).expect("write xes");
            for (tag, net) in [("recorded", &net_before), ("rebuilt", &net_rebuilt)] {
                if let Some(n) = net.nets.get(ot) {
                    n.export_pnml(format!("{stem}.{tag}.pnml")).expect("write pnml");
                }
            }
            println!("  exported {stem}.xes + .recorded.pnml + .rebuilt.pnml");
        }
    }
}

fn main() {
    for p in corpus::logs_or_args() {
        run(&p);
    }
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
