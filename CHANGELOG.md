# Changelog

## Unreleased

- Alignments now search from both ends at once (`utils::dijkstra_search::search_bidirectional` + `ReversibleSearchProblem`), and detect unreachable final markings up front
- `search` takes `SearchLimits` instead of `Option<usize>`; its new `max_states_queued` bounds memory and reports `SearchError::QueuedLimitReached` (**Breaking**)
- `AlignmentOptions::max_states_queued` (default 10 million) bounds how many states a trace's search may hold at once; `alignments::states_in_memory` converts a memory budget into such a state count
- Fix parallel arcs between the same place and transition counting separately (previously their weights were not summed, which could underflow a place's token count)
- Arc weights and markings exceeding `TokenCount::MAX` now error with `SyncProdNetConstructionError::{ArcWeightTooLarge, MarkingTooLarge}` instead of being truncated (**Breaking**)
- Added extraction blueprints (`extraction-blueprint`): A declarative model for building an OCEL
  from relational data, with a row executor, a SQL-view compiler, a validator, and in-memory and
  `DuckDB` sinks
  - `extraction-dbcon` adds a `RowProvider` over SQLite, CSV and Parquet via `dbcon`, all readable
    from bytes and all buildable for `wasm32`
  - `extraction-dbcon-postgres` adds PostgreSQL, which needs `sqlx` and is native-only
- Added a pushdown query layer (`core::event_data::object_centric::query`): a binding-box query
  AST with an in-memory evaluator and a `DuckDB` SQL translator, which return the same rows
  - `Filter` covers relational, attribute and child-fold predicates in one list, composable with
    `Or`/`Not`/`Compare`, and readable as a value through `Expr::Satisfies`
  - `Query::emits` names extra per-node outputs: `evaluate_multi` counts every node's bindings in
    one enumeration, and `evaluate_node` replays a single node's rows on demand, paged
    (in-memory backends only)
- Added the OCEL 2.0 bundled CSV/Parquet format (`ocel-bundle`, plus `ocel-bundle-parquet` for
  Parquet storage): a `.ocel.zip` archive or a directory with the same layout, read and written
  through `Importable`/`Exportable` like any other OCEL format
  - `Exportable::export_to_path_as` writes to a path in an explicitly named format, for the cases
    where the path cannot carry one (a directory has no extension)
  - Snappy-compressed Parquet is now readable. It is what most other Parquet writers emit by
    default, and a container produced outside this crate failed to import without it
- OCEL 2.0 CSV: an object id or qualifier containing `/`, `#`, `{` or `\` is now escaped with a
  backslash on export and read back on import. Written raw, `a/b` came back as two references
  and `a#b` with a truncated id, both without an error. A backslash before anything else is
  still a backslash, so existing files read exactly as before
- OCEL 2.0 CSV: an `ot:<X>` header names the object type exactly, and an event attribute value
  is kept as written. Both were trimmed, which split a type whose name ends in a space away
  from itself and quietly rewrote padded values
- OCEL 2.0 CSV: a value is only read as a number or an instant when its text is the canonical
  spelling of one. `007`, `+7`, `1e3`, an integer too large for `i64`, and a timestamp with no
  timezone stay strings instead of coming back as `7`, `1000`, a rounded `f64`, or an instant at
  an offset the file never gave
- OCEL 2.0 CSV: `strict` now rejects an o2o row whose source object has no declared type. The
  check sat inside a `verbose` branch, so `strict` alone dropped the row silently
- Added the `ocel_dataset_crosscheck` example: reads a log from a `.ocel.zip`, from a `.ocel.csv`
  and from a baseline `.json`/`.xml`, and reports where the three disagree, separating a genuine
  difference in values from one only in the attribute variant or the recorded time
- `ocel_sql` exposes the `DuckDB` consolidated schema: `DuckDbLinkedOCEL`,
  `stream_ocel_file_to_duckdb` / `_with`, `DuckDbImportOptions`, and
  `read_consolidated_ocel_from_duckdb_path` / `read_consolidated_slim_ocel_from_duckdb_path`
- `SqlOcelImportOptions` with `import_ocel_sqlite_from_con_with_options` /
  `import_ocel_duckdb_from_con_with_options`. An object-type table without its
  `ocel_changed_field` column is now rejected by name instead of failing on the query that reads
  it, and `allow_missing_changed_field` reads such a file as initial state only
- `StreamImportOCEL` streams a reader or a path into any `AppendableOCEL` and finalizes it.
  `is_streaming_format` reports which formats it covers
- `TabularSource` holds the bytes of a tabular file (`SQLite`, CSV, Parquet, workbook) in the
  registry, so a binding can name a dropped file by id where there is no filesystem
- `OCELAttributeType::coalesce`: the narrowest type covering two others, used by CSV type inference
- `OCELEvent.time` accepts the non-RFC3339 timestamp formats the rest of the crate parses
- `AggregatedEventTimestamps::bin_width_ms` is the spacing of the bin centers, and `0` when there
  are no bins (was a rounded width that could disagree with the centers, and `1` for no bins)
  (**Breaking**)

- Implemented the Inductive Miner Algorithm (`discovery::case_centric::inductive_miner`)
  - Overall algorithm structure and initial implementation from @Kuhlkrypto. Thanks!
  - Added missing implementations parts (e.g., cut detection and log splitting left as stubs)
  - Added infrequency handling (IMf) via `InductiveMinerOptions::imf(f)`, filtering in cut detection, log splitting, base cases and fall throughs
  - IM(f) now uses an activity-projected log and an index-based directly-follows graph instead of cloning the `EventLog` per split
  - Sub-logs and fall-through candidates are mined in parallel
  - Added the directly-follows variant IMd/IMfd: `inductive_miner_dfg` passes over the log once and recurses on the graph alone, `inductive_miner_dfg_from_graph` mines an existing `DirectlyFollowsGraph`. Scales to logs the log recursion cannot handle; does not guarantee fitness
  - `InductiveMinerDfgOptions::{prom, pm4py}` imitate those tools' IMd for comparison runs, via the individual options `use_tau_loops`, `flower_accepts_empty`, `repeated_activity_to_flower` and `filter_single_activity`
  - `InductiveMinerOptions::{prom, pm4py}` do the same for the log-based IM, and `filter_single_activity` switches off the single-activity base case of IMf (§6.2.2.3), which `PM4Py` does not implement
  - IMf's activity-concurrent fall through probes the unfiltered directly-follows graph (§6.2.2.4); `filter_activity_concurrent_probe` makes it probe the filtered one, which on logs with heavy per-trace repetition accepts orderings the log does not support
- Added `PetriNet::reduce_silent_transitions` as language-preserving post-processing (removes silent transitions that only pass a token on)
- Object-centric Petri net discovery (`discovery::object_centric::ocpn`)
  - `with_object_types(ObjectTypeFilter::only([..]))` or `::except([..])` restricts discovery to some object types
- `flatten_ocel_on` handles objects involved multiple times in one event (e.g., through different qualifiers) differently now: It no longer repeats the event. This also affects `OCDirectlyFollowsGraph`.
- Fixed a regression in OC-DECLARE discovery/conformance runtime performance:
  - Now builds and construct a reverse-E2O index grouped by event type
  - OC-DECLARE internals are no longer public; their arguments could only be produced by other internals (**Breaking**)
    - `conformance::oc_declare::{get_evs_with_objs_perf, get_for_ev_perf, get_for_all_evs_perf, get_for_all_evs_perf_thresh}`
    - `discovery::object_centric::oc_declare::{get_oi_labels, combine_constraints}`
    - Per-arc checking is unchanged: `oc_declare_conformance`, `OCDeclareArc::violation_fraction`, `OCDeclareArc::satisfies_threshold`
  - `OCDeclareArc::get_for_all_evs_perf` -> `violation_fraction`, `OCDeclareArc::get_for_all_evs_perf_thresh` -> `satisfies_threshold` (**Breaking**)
  - Fixed `ANY` object involvements counting a target event once per referenced object instead of once

## 0.6.0

- Optimal Petri net alignments (`conformance::alignments`):
- Optimal alignments (`conformance::alignments`):
  - `align_log` / `align_projection` / `align_trace` / `align_empty_trace` compute cost-optimal alignments via Dijkstra over a synchronous product net
  - `compute_fitness` derives log/trace fitness for pre-computed alignments
  - Configurable `AlignmentOptions` (`CostFunction`, `max_states`)
  - Exposed as bindings
- Generic, reusable state-space search in `utils::dijkstra_search` (`SearchProblem` trait + `search`)
- `register_binding` macro accepts slice arguments (`&[T]` extracted as `Vec<T>`)
- New `ReadableOCEL` and `AppendableOCEL` traits; OCEL exporters (CSV, XML, SQL, JSON) and the JSON importer are generic over them
- `SlimLinkedOCEL` implements `AppendableOCEL`, with auto-declare types, auto-grow attributes, and value coercion via new `OCELAttributeValue::try_coerce_to`; misordered streams are buffered and resolved on `finalize`
- `import_ocel_json_into` and `import_ocel_xml_into` stream JSON / XML directly into any `AppendableOCEL` (e.g., `SlimLinkedOCEL`) without materializing an `OCEL` first; `SlimLinkedOCEL::import_from_*` uses the streaming paths automatically
- `ReadableOCEL::iter_events_of_type` / `iter_objects_of_type` (default filters; `SlimLinkedOCEL` overrides via per-type indices, avoiding a full scan per type in SQL export)
- New `OCELAttributeType::as_type_str` returns `&'static str` (cheaper than `to_type_string`)
- `EventIndex` / `ObjectIndex` are now `u32`-backed; `into_inner` returns `u32` (**Breaking**)
- `SlimLinkedOCEL`, `SlimOCELEvent`, `SlimOCELObject` no longer derive `Deserialize` (still serialize); construct via `Importable` or `AppendableOCEL` (**Breaking**)
- `SlimOCELEvent::relationships` and `SlimOCELObject::relationships` are now `Vec<(QualifierIdx, ObjectIndex)>` (was `Vec<(String, ObjectIndex)>`); resolve qualifier strings via `SlimLinkedOCEL::qualifier_str` (**Breaking**)
- CSV exporter streams rows instead of buffering all rows; tracks `(time, object_id)` pairs in a `HashSet` to skip redundant object-attribute rows
- `SlimLinkedOCEL::from_ocel` now goes through `AppendableOCEL`, so it shares attribute coercion, schema-grow, and duplicate-id detection with the streaming import paths. Behavior changes: events/objects with attribute names not in the declared type schema grow the schema (before they were silently dropped); duplicate event/object ids are skipped with a warning (before they were kept but unreachable); references to undeclared types auto-create the type (before they were dropped with warning)
- `try_json_to_ocel` returning `Result<OCEL, serde_json::Error>` added alongside the existing panicking `json_to_ocel`
- `SlimLinkedOCEL::get_o2o_rev_obs_of_obtype` / `get_e2o_rev_evs_of_evtype`: qualifier-optional reverse-relation getters filtered by type
- New public object-centric analysis functions (also exposed as bindings): per-event sojourn and synchronization times with optional `top_k` (`analysis::object_centric::oc_performance`), E2O `(event_type, object_type)` counts and `source -> target` conversion rate (`analysis::object_centric::oc_statistics`), and per-object-type directly-follows graph and activity-trace variants (`discovery::object_centric::dfg` / `variants`)
- Fix SQL export/import of floats and timestamps: floats are written as `DOUBLE PRECISION` (full f64 precision) and timestamps as naive UTC (avoids a double-applied timezone offset); import maps `DOUBLE` / `DOUBLE PRECISION` columns back to float, so round-trips no longer drop float attributes
- New direct dependency on `hashbrown` for the slim per-id hash tables

## 0.5.6

- Translate a `ProcessTree` into a `PetriNet`:
  - `ProcessTree::to_petri_net` returns a workflow net with initial and final marking set
  - `add_to_petri_net` on `Node` / `Operator` / `Leaf` for recursive insertion into an existing net
  - `From<ProcessTree> for PetriNet`
- Bump MSRV to 1.88
- Update dependencies (pin `cxx-build` to match `kuzu`'s `cxx`, refresh `cargo deny` license allowlist)
- Add Criterion + dhat benchmarks for event log import (+ DataFrame conversion)

## 0.5.5

- `SlimLinkedOCEL` and bindings:
  - `add_event` / `add_object` pad or truncate `attributes` to the declared length with a warning, instead of causing out-of-bounds panics later
  - `add_e2o` / `add_o2o` / `delete_e2o` / `delete_o2o` return `bool`; invalid indices warn and return `false` instead of panicking
  - Multiple qualifiers between the same `(event, object)` or `(from, to)` pair are kept, and reverse lookups return every qualifier
  - `fat_ev` / `fat_ob` fall back to `Null` / empty attribute on missing positional values
  - Drop `unwrap()`s on unknown type names in reverse-type lookups
  - Expanded docstrings on `slim_ocel_bindings` for the auto-generated Python API docs
- `Importable` / `Exportable` for `SlimLinkedOCEL` aligned with the `OCEL` versions
- OCEL import/export supports `.gz` for all formats
- Better defaults for OCEL 2.0 CSV export

## 0.5.4

- Add `stream_xes_bufread` function for streaming XES traces from a `BufRead` (supports gzipped input)
- Remove noisy `println!` in OCEL XML import for extended OCELs

## 0.5.3

- Parse XES version from log element
- Fix some missing unescapes in XML-based imports (XES, PNML)

## 0.5.2

- New `analysis` module with reusable analysis functions (also exposed as bindings ;))
  - `analysis::case_centric::dotted_chart`: Configurable multi-axis dotted chart generation (`DottedChartOptions`)
  - `analysis::case_centric::event_timestamp_histogram`: Aggregate event timestamps into bins grouped by activity (`EventTimestampOptions`)
  - `analysis::object_centric::object_attribute_changes`: Extract time-stamped attribute change history for an OCEL object

## 0.5.1
- Rename bindings function for SlimLinkedOCEL bindings (not breaking, as 0.5.0 bindings were not published yet)

## 0.5.0
- Fix SlimLinkedOCEL addObject function (previously did not correctly expand the reverse E2O/O2O reference array)
- Change error type of OCEL XML import to `OCELIOError` 
(**Breaking**), return error if XML does not contain any event or object types
  - Added related test (`test_xes_as_ocel_xml_import`) to ensure xes files are not correctly imported as OCEL
- Implement `Default` for SlimLinkedOCEL, add `new` function for SlimLinkedOCEL
- Expose SlimLinkedOCEL binding functions (e.g., for adding events/objects, getting relations, etc.)
- Implement From<...> for (bi-directional) conversion between (XES) `AttributeValue`s to `OCELAttributeValue`s
- Remove `Hash` derive from `OCELType` (**Breaking**)

## 0.4.4
- Fix version mismatch in macros crate

## 0.4.3
- Fix: typo in function name: `oc_declare_conformace` -> `oc_declare_conformance` (**Breaking**)

## 0.4.2
- **New OCEL CSV Format**
  - Added CSV format support for OCEL:
  - Added Importer/Exporter for CSV OCEL file format
  - Added CSV file format to OCEL io trait + known formats (as `.ocel.csv`)
- **Bindings Improvements**:
  - Exposed OC-DECLARE conformance function (`oc_declare_conformace`) to bindings
  - Renamed `discover_oc-declare` binding to `discover_oc_declare` (**Breaking for Bindings**)
  - Renamed `discover_dfg_from_locel` to `discover_dfg_from_ocel` (**Breaking**)
  - Added `SlimLinkedOCEL` <-> `OCEL` conversion support in bindings
  - Implemented `LinkedOCELAccess` trait support in bindings macro for more generic functions
  - Added `ocel_type_stats` binding to compute event/object type statistics
  - Exposed `flatten_ocel_on` function to bindings for flattening OCEL on object types
  - Exposed `add_init_exit_events_to_ocel` function to bindings
- **Other Fixes and Improvements**:
  - Fixed SQLite/DuckDB export to remove existing file before export (prevents UNIQUE constraint errors)
  - Combined/Deduped timestamp-related parsing functionality across files
  - Implemented `Null` as default `OCELAttributeValue`
- **Internal Improvements**:
  - Updated `rusqlite` and related dependencies
  - Improved CLI in `r4pm`

### Breaking Changes / Migration Guide
- The `From<OCELAttributeValue>` implementation for `OCELAttributeType` was removed. Instead, use the `get_type` function on `OCELAttributeValue` to retrieve its type.
- Updates related to io module for CSV parsing (e.g., new error variant in `OCELIOError`)
- Renamed binding `discover_oc-declare` to `discover_oc_declare`

## 0.4.1
- Added `verbose` option to `XESImportOptions`, defaulting to true
  - Note: Technically this is a breaking change, however the recommended way to use `XESImportOptions` is non-exhaustive with default fallback:
    - e.g., ```XESImportOptions {verbose: false, ..Default::default()}```

## 0.4.0

### Restructuring (Current)
- **Unified IO Traits**: Introduced `Importable` and `Exportable` traits in `process_mining::core::io` to standardize import and export operations across different data structures.
- **EventLog IO**: Implemented `Importable` and `Exportable` for `EventLog`, supporting JSON (`.json`), XES (`.xes`), and Gzipped XES (`.xes.gz`) formats.
- **PetriNet IO**: Implemented `Importable` and `Exportable` for `PetriNet`, supporting PNML (`.pnml`) format.
- **OCEL IO**: Implemented `Importable` and `Exportable` for Object-Centric Event Logs (OCEL), including support for SQLite and DuckDB (if features enabled).
- **Format Inference**: Added automatic format inference based on file extensions (e.g., `.xes`, `.xes.gz`, `.pnml`).
- **Auto-Bindings**: Added auto-binding functionality to facilitate Python bindings generation.
- **Module Restructuring**:
    - Moved Alpha+++ discovery to `process_mining::discovery`.
    - Moved Petri nets to `process_mining::core::process_models`.
    - Moved DFG discovery to `process_mining::discovery`.
- **API Simplification**: Users can now use generic `import_from_path` and `export_to_path` methods. These methods now strictly rely on file extension for format inference, removing the optional format argument.

### Features (Unreleased on crates.io)
- **KuzuDB Support**: Added initial support for OCEL export to KuzuDB.
- **DuckDB Support**: Added example for OCEL export to DuckDB.
- **Polars Export**: Added OCEL to Polars DataFrame export.
- **Object-Centric Process Trees**: Added implementation of object-centric process trees and abstraction-based conformance checking.
- **Token-Based Replay**: Implemented token-based replay on Petri nets.
- **Incidence Matrices**: Added incidence matrices for Petri nets.
- **Event Log Macros**: Implemented macros for easier event log creation.
- **OC-DECLARE**: Object-centric declarative process models, with discovery and conformance checking.

### Changed
- **Exposed Fields**: Exposed `OCLanguageAbstraction` fields.

### Migration Guide
- **Importing Event Logs**:
  - Old: `import_xes_file("log.xes")`
  - New: `EventLog::import_from_path("log.xes")`
- **Exporting Event Logs**:
  - New: `log.export_to_path("log.xes")`
- **Traits**: Ensure `process_mining::Importable` and `process_mining::Exportable` are in scope if you need to use the traits generically.
- **Format Specification**: If you need to specify a format explicitly (e.g., reading from a stream or non-standard extension), use `import_from_reader` or `export_to_writer` which still accept a format string.
