use process_mining::core::event_data::case_centric::EventLogClassifier;
use process_mining::core::process_models::process_tree::ProcessTree;
use process_mining::discovery::case_centric::inductive_miner::{
    inductive_miner, inductive_miner_dfg, InductiveMinerDfgOptions, InductiveMinerOptions,
};
use process_mining::{EventLog, Exportable, Importable};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

/// Discovers a process tree with the Inductive Miner.
///
/// ```text
/// cargo run --release --example inductive_miner -- <event log> [noise threshold] [output.pnml]
///     [--json] [--dfg] [--thesis] [--like-prom] [--like-pm4py] [option flags]
/// ```
///
/// The noise threshold selects the variant: `0` (the default) is plain IM, anything above it is
/// IMf, which is usually what you want for a real-life log. Given an output path, the tree is also
/// converted to a Petri net and exported as PNML. With `--json` the tree is printed as JSON and
/// nothing else. `--dfg` runs the directly-follows variant IMd/IMfd.
///
/// `--thesis` starts from the presets that follow the thesis to the letter, `--like-prom` and
/// `--like-pm4py` from the ones that imitate those tools. The option flags below then toggle
/// single fields of [`InductiveMinerOptions`] resp. [`InductiveMinerDfgOptions`] on top of that.
const FLAGS: &[&str] = &[
    "--json",
    "--dfg",
    "--thesis",
    "--like-prom",
    "--like-pm4py",
    "--interleaved",
    "--rewrite-inclusive-choice",
    "--no-msd",
    "--no-strict-sequence",
    "--no-inclusive-choice",
    "--no-guard-empty-cut-parts",
    "--degenerate-tau-loops",
    "--no-fallthroughs",
    "--no-filter-single-activity",
    "--filter-fallthrough-probe",
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = env::args().collect();
    let set = |flag: &str| args.iter().any(|arg| arg == flag);

    let as_json = set("--json");
    let as_dfg = set("--dfg");
    let thesis = set("--thesis");
    let like_prom = set("--like-prom");
    let like_pm4py = set("--like-pm4py");
    let rewrite_inclusive_choice = set("--rewrite-inclusive-choice");
    let use_interleaved = set("--interleaved");
    let use_minimum_self_distance = !set("--no-msd");
    let strict_sequence = !set("--no-strict-sequence");
    let use_inclusive_choice = !set("--no-inclusive-choice");
    let guard_empty_cut_parts = !set("--no-guard-empty-cut-parts");
    let use_degenerate_tau_loops = set("--degenerate-tau-loops");
    let use_fallthroughs = !set("--no-fallthroughs");
    let filter_single_activity = !set("--no-filter-single-activity");
    let filter_activity_concurrent_probe = set("--filter-fallthrough-probe");

    // A silently ignored flag turns a careful comparison run into a default one.
    if let Some(unknown) = args
        .iter()
        .find(|arg| arg.starts_with("--") && !FLAGS.contains(&arg.as_str()))
    {
        eprintln!("Unknown flag {unknown}. Known flags: {}", FLAGS.join(" "));
        std::process::exit(2);
    }
    args.retain(|arg| !arg.starts_with("--"));

    if args.len() < 2 || args.len() > 4 {
        eprintln!(
            "Usage: {} <path_to_event_log> [noise_threshold] [output_pnml_path] {}",
            args[0],
            FLAGS.join(" ")
        );
        std::process::exit(1);
    }

    let log_path = PathBuf::from(&args[1]);
    let noise_threshold: f64 = match args.get(2) {
        Some(value) => value.parse()?,
        None => 0.0,
    };

    let base = match (thesis, like_prom, like_pm4py) {
        (true, _, _) => InductiveMinerOptions::imf_thesis(noise_threshold),
        (_, _, true) => InductiveMinerOptions::pm4py(noise_threshold),
        (_, true, _) => InductiveMinerOptions::prom(noise_threshold),
        _ => InductiveMinerOptions::imf(noise_threshold),
    };
    let options = InductiveMinerOptions {
        use_interleaved,
        rewrite_inclusive_choice,
        // Only override what was asked for, so a preset keeps its own value otherwise.
        use_minimum_self_distance: base.use_minimum_self_distance && use_minimum_self_distance,
        strict_sequence: base.strict_sequence && strict_sequence,
        use_inclusive_choice: base.use_inclusive_choice && use_inclusive_choice,
        guard_empty_cut_parts: base.guard_empty_cut_parts && guard_empty_cut_parts,
        use_degenerate_tau_loops: base.use_degenerate_tau_loops || use_degenerate_tau_loops,
        use_fallthroughs: base.use_fallthroughs && use_fallthroughs,
        filter_single_activity: base.filter_single_activity && filter_single_activity,
        filter_activity_concurrent_probe: base.filter_activity_concurrent_probe
            || filter_activity_concurrent_probe,
        ..base
    };
    let dfg_base = match (like_prom, like_pm4py) {
        (_, true) => InductiveMinerDfgOptions::pm4py(noise_threshold),
        (true, _) => InductiveMinerDfgOptions::prom(noise_threshold),
        _ => InductiveMinerDfgOptions::imfd(noise_threshold),
    };
    let discover = |log: &EventLog| -> ProcessTree {
        if as_dfg {
            let options = InductiveMinerDfgOptions {
                rewrite_inclusive_choice,
                ..dfg_base
            };
            inductive_miner_dfg(log, &EventLogClassifier::default(), options)
        } else {
            inductive_miner(log, &EventLogClassifier::default(), options)
        }
    };

    if as_json {
        let log = EventLog::import_from_path(&log_path)?;
        println!("{}", serde_json::to_string(&discover(&log))?);
        return Ok(());
    }

    println!("Importing event log from {log_path:?}");
    let log = EventLog::import_from_path(&log_path)?;
    println!(
        "Imported {} traces with {} events.",
        log.traces.len(),
        log.traces.iter().map(|t| t.events.len()).sum::<usize>()
    );

    println!(
        "Discovering a process tree with a noise threshold of {}...",
        options.noise_threshold
    );

    let started = Instant::now();
    let tree = discover(&log);
    println!("Done in {:?}.", started.elapsed());

    println!("\n{tree}\n");
    println!("The tree has {} leaves.", tree.find_all_leaves().len());

    if let Some(output_path) = args.get(3) {
        let output_path = PathBuf::from(output_path);
        println!("Exporting the corresponding Petri net to {output_path:?}");
        tree.to_petri_net().export_to_path(&output_path)?;
    }

    Ok(())
}
