use process_mining::core::event_data::object_centric::linked_ocel::IndexLinkedOCEL;
use process_mining::discovery::case_centric::inductive_miner::InductiveMinerOptions;
use process_mining::discovery::object_centric::ocpn::{
    discover_ocpn, ObjectCentricDiscoveryOptions, ObjectTypeFilter,
};
use process_mining::{Importable, OCEL};
use std::env;
use std::error::Error;
use std::time::Instant;

/// Discovers an object-centric Petri net with the Inductive Miner.
///
/// ```text
/// cargo run --release --example object_centric_discovery -- <ocel file> [noise threshold]
///     [variable arc tolerance] [--types a,b,c | --exclude a,b,c] [--json]
/// ```
///
/// The noise threshold selects the variant used for the per-type nets: `0` (the default) is plain
/// IM, anything above it is IMf. The tolerance says how many of an activity's executions may
/// deviate from "exactly one object" before an arc still counts as normal, `0` being the strict
/// reading. `--types` discovers only the given object types and `--exclude` all but them. With
/// `--json` the net is printed as JSON and nothing else.
fn main() -> Result<(), Box<dyn Error>> {
    let all: Vec<String> = env::args().collect();
    let as_json = all.iter().any(|arg| arg == "--json");

    // Split the flags off first, so the positional arguments keep their places.
    let mut args: Vec<String> = Vec::new();
    let mut rest = all.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--json" => {}
            "--types" | "--exclude" => {
                rest.next();
            }
            _ => args.push(arg.clone()),
        }
    }

    if args.len() < 2 || args.len() > 4 {
        eprintln!(
            "Usage: {} <path_to_ocel> [noise_threshold] [variable_arc_tolerance] \
             [--types a,b,c | --exclude a,b,c] [--json]",
            args[0]
        );
        std::process::exit(1);
    }

    let noise_threshold: f64 = match args.get(2) {
        Some(value) => value.parse()?,
        None => 0.0,
    };
    let tolerance: f64 = match args.get(3) {
        Some(value) => value.parse()?,
        None => 0.0,
    };

    let ocel = OCEL::import_from_path(&args[1])?;
    let events = ocel.events.len();
    let objects = ocel.objects.len();
    let locel = IndexLinkedOCEL::from(ocel);

    let started = Instant::now();
    let list = |flag: &str| {
        let at = all.iter().position(|arg| arg == flag)?;
        Some(
            all.get(at + 1)?
                .split(',')
                .map(str::trim)
                .collect::<Vec<_>>(),
        )
    };
    let object_types = match (list("--types"), list("--exclude")) {
        (Some(only), _) => ObjectTypeFilter::only(only),
        (None, Some(except)) => ObjectTypeFilter::except(except),
        (None, None) => ObjectTypeFilter::All,
    };

    let options = ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(noise_threshold))
        .with_variable_arc_tolerance(tolerance)
        .with_object_types(object_types);
    let ocpn = discover_ocpn(&locel, options);
    let elapsed = started.elapsed();

    if as_json {
        println!("{}", ocpn.to_json());
        return Ok(());
    }
    println!(
        "Imported {events} events and {objects} objects over {} object types.",
        ocpn.object_types().len()
    );
    println!(
        "Discovered an object-centric Petri net in {elapsed:?}: {} places, {} transitions, \
         {} arcs, {} activities.\n",
        ocpn.num_places(),
        ocpn.num_transitions(),
        ocpn.num_arcs(),
        ocpn.activities().len()
    );

    for object_type in ocpn.object_types() {
        let net = &ocpn.nets[object_type];
        let variable: Vec<&str> = ocpn
            .activities()
            .into_iter()
            .filter(|a| ocpn.is_variable_arc(object_type, a))
            .collect();
        println!(
            "  {object_type:<28} {:>4} places {:>4} transitions {:>4} arcs   variable arcs: {}",
            net.places.len(),
            net.transitions.len(),
            net.arcs.len(),
            if variable.is_empty() {
                "-".to_string()
            } else {
                variable.join(", ")
            }
        );
    }

    println!("\nActivities shared between object types (the stitching points):");
    for activity in ocpn.activities() {
        let types = ocpn.object_types_of(activity);
        if types.len() > 1 {
            println!("  {activity:<40} {}", types.join(", "));
        }
    }

    Ok(())
}
