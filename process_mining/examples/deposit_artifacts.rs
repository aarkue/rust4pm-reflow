//! The tagged log and its flow projection, measured side by side.
//!
//! One assignment, two files, and the difference between them is only visible from outside:
//! what a reader that does **not** know the tag convention draws from each. That reader is
//! simulated here by counting the arcs of every cell the artifact still records, which is what
//! the trace variants read -- they take a qualifier-blind view of the event-to-object relation
//! on purpose, so it is the same view an off-the-shelf discovery tool takes.
//!
//! Usage:
//! `cargo run --release --features ocel-sqlite --example deposit_artifacts -- <log> ...`
//!
//! `--write <dir>` also writes both artifacts out, which is how another tool gets to see one.

use std::{collections::HashSet, env, path::PathBuf};

use process_mining::{
    analysis::object_centric::schema_reduction::{
        arc_set, ActivityIndexing, CellGrid, SchemaClosure, StructuralSchema,
    },
    bindings::schema_reduction_bindings::{
        schema_reduction_apply, schema_reduction_evaluate, schema_reduction_overview, CellRef,
        CellStateKind, DepositMode, FlowProjectionCost,
    },
    core::event_data::object_centric::linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL},
    Exportable, Importable, OCEL,
};

/// Participations, and how many of them carry the non-flow tag.
fn e2o(locel: &SlimLinkedOCEL) -> (usize, usize) {
    let mut total = 0;
    let mut tagged = 0;
    for e in locel.get_all_evs() {
        for (q, _) in e.get_e2o_q(locel) {
            total += 1;
            if q.starts_with('!') || q.starts_with("+!") {
                tagged += 1;
            }
        }
    }
    (total, tagged)
}

/// What a reader that ignores the tag draws from an artifact: every cell it still records, at
/// flow.
fn arcs_a_blind_reader_draws(locel: &SlimLinkedOCEL) -> usize {
    let schema = StructuralSchema::discover(locel);
    let closure = SchemaClosure::build(locel, &schema);
    let grid = CellGrid::build(locel, &schema, &closure);
    let acts = ActivityIndexing::build(locel, &grid);
    let all: HashSet<_> = grid.cells.iter().copied().collect();
    arc_set(locel, &schema, &acts, &all).len()
}

fn report(path: &str, out_dir: Option<&str>) {
    let ocel = OCEL::import_from_path(PathBuf::from(path)).expect("import log");
    let locel = SlimLinkedOCEL::from_ocel(ocel);
    let name = PathBuf::from(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());

    let ov = schema_reduction_overview(&locel, None, None, Some(true));
    let r = &ov.recommendation;
    let flow: Vec<CellRef> = r.flow.clone();

    println!("\n{}\n{path}\n{}", "=".repeat(72), "=".repeat(72));
    println!(
        "{} cells recorded: {} flow / {} involvement / {} implied, {} expanded",
        ov.cells.len(),
        r.tally.flow_cells,
        r.tally.involvement_cells,
        r.tally.implied_cells,
        r.tally.expanded_cells
    );
    // Two counts of one relation: the grid counts distinct `(event, object)` pairs, and the
    // file carries one entry per qualifier, so a participation naming an employee as both the
    // forwarder and the shipper is one cell entry and two lines.
    println!(
        "recorded: {} participations over {} cell entries, {} arcs",
        e2o(&locel).0,
        ov.participations_total,
        ov.arcs_recorded
    );
    let say = |what: &str, c: FlowProjectionCost| {
        println!(
            "flow-projection cost {what}: {} non-flow cells / {} participations, \
             {} cells / {} participations come back, \
             {} cells / {} participations DO NOT",
            c.cells(),
            c.participations(),
            c.recoverable_cells,
            c.recoverable_participations,
            c.unrecoverable_cells,
            c.unrecoverable_participations,
        );
    };
    say("under the recommendation", r.flow_projection);
    // Everything left at involvement is there because nothing determines it, so that half of
    // the cost is one-way; the implied cells are the recoverable half.
    let held: Vec<CellRef> = r
        .assignment
        .iter()
        .filter(|c| c.state == CellStateKind::Implied && c.removal_recoverable)
        .map(|c| CellRef {
            activity: c.activity.clone(),
            object_type: c.object_type.clone(),
        })
        .collect();
    if !held.is_empty() {
        let ev = schema_reduction_evaluate(&locel, flow.clone(), held, Vec::new(), None, None);
        say("with every determined cell shown as involved", ev.flow_projection);
    }

    for (mode, label) in [
        (DepositMode::Tagged, "tagged"),
        (DepositMode::FlowProjection, "flowProjection"),
    ] {
        let out = schema_reduction_apply(
            &locel,
            flow.clone(),
            Vec::new(),
            Vec::new(),
            Some(mode),
            None,
        );
        let (total, tagged) = e2o(&out);
        println!(
            "  {label:15} E2O {total:>9}  ({tagged} non-flow)  arcs a tag-blind reader draws: {}",
            arcs_a_blind_reader_draws(&out)
        );
        if let Some(dir) = out_dir {
            let p = PathBuf::from(dir).join(format!("{name}.{label}.xml"));
            out.export_to_path(&p).expect("write artifact");
            println!("            {}", p.display());
        }
    }
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let out_dir = args
        .iter()
        .position(|a| a == "--write")
        .map(|i| args.splice(i..=i + 1, []).nth(1).expect("--write needs a directory"));
    if args.is_empty() {
        eprintln!("usage: deposit_artifacts [--write <dir>] <log> ...");
        return;
    }
    for path in &args {
        report(path, out_dir.as_deref());
    }
}
