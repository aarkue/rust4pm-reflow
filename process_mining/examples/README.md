# Examples

This folder contains example usages of the `process_mining` crate.

## Basic Usage

- **`event_log_stats.rs`**: Imports an XES event log and prints basic statistics (trace count, event count, etc.).
  ```bash
  cargo run --example event_log_stats -- <path_to_log.xes>
  ```

- **`process_discovery.rs`**: Imports an XES event log, discovers a Petri net using the Alpha+++ algorithm, and exports it to PNML.
  ```bash
  cargo run --example process_discovery -- <path_to_log.xes> <output_model.pnml>
  ```

- **`petri_net_import_export.rs`**: Imports a Petri net from PNML, prints stats, and exports it again.
  ```bash
  cargo run --example petri_net_import_export -- <input_model.pnml> <output_model.pnml>
  ```

## Conformance Checking

- **`calculate_alignments.rs`**: Imports an XES event log and a Petri net, computes optimal alignments for every trace variant, and prints the resulting fitness.
  ```bash
  cargo run --release --example calculate_alignments -- <path_to_log.xes> <path_to_model.pnml>
  ```

## Object-Centric Process Mining

- **`ocel_stats.rs`**: Imports an OCEL and prints basic statistics.
  ```bash
  cargo run --example ocel_stats -- <path_to_ocel.xml>
  ```

- **`ocel_csv_export.rs`**: Imports an OCEL and exports it to CSV format.
  ```bash
  cargo run --example ocel_csv_export -- <path_to_ocel.xml> [output.ocel.csv]
  ```

- **`ocel_duckdb_export.rs`**: Imports an OCEL and exports it to a DuckDB database.
  ```bash
  cargo run --example ocel_duckdb_export -- <path_to_ocel.xml>
  ```

- **`ocel_kuzudb_export.rs`**: Imports an OCEL and exports it to a KuzuDB graph database.
  ```bash
  cargo run --example ocel_kuzudb_export -- <path_to_folder_containing_ocel_files>
  ```

## ReFlow evaluation

The examples below produce the numbers of the ReFlow paper. Each takes log paths as arguments and, with none, runs the nine-log corpus from `REFLOW_LOGS` (default `artifacts/logs/`, see `artifacts/logs/README.md`). Results are written to `REFLOW_STATS` (default `artifacts/results/stats/`). Run them from `process_mining/` with `--release`; most need `--features "ocel-sqlite,token-based-replay"`.

| example | paper | writes |
|---|---|---|
| `../tools/ocpn-quality` | Table 1 assignment, fitness and precision (run first: it writes the keepsets the others read) | `<log>.keepsets.json`, `<log>.scores.json`, `<log>.stats.json` |
| `cross_instantiation` | Table 1 (cells, arcs, orderings under the three abstractions, expansion column) | `cross_instantiation.json` |
| `eval_instruments` | preserved orderings, sizes, runtimes | `eval_instruments.json` |
| `exact_gap` | distance to the optimum | `exact_gap.json` |
| `type_deletion` | whole-type deletion baseline (`DIV_SHARE=1` for the strict variant) | `type_deletion.json` |
| `ground_truth_reverse` | CPN ground truth for Order Management and Container Logistics | `ground_truth_reverse.json` |
| `theta_sweep`, `determination_sweep` | threshold sensitivity | `theta_sweep.json`, `determination_sweep.json` |
| `silence_vs_divergence` | neutral against divergent cells | `silence_vs_divergence.json` |
| `expansion_holdout` | hold-out check of expansion | `expansion_holdout.json` |
| `declare_silence`, `model_silence`, `tree_abstraction` | the OC-DECLARE and model-based abstractions per cell | `declare_silence.json`, `model_silence_constructs.json` |
| `carrier_check`, `hand_deletion`, `chaining_soundness`, `route_delivery` | supporting checks | `carrier_check.json`, `hand_deletion.json` |
| `schema_census` | inspect the schema, cells and routes of one log (`--features bindings`) | stdout |

```bash
cd process_mining
cargo run --release --features "ocel-sqlite,token-based-replay" --example cross_instantiation
cargo run --release --features "ocel-sqlite,token-based-replay" --example cross_instantiation -- path/to/one-log.xml
```
