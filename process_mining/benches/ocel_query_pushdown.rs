//! Latency benchmark for the pushdown query layer: for each of three representative
//! algorithms, compares the existing direct implementation against the query-layer
//! rewrite (`query/algos.rs`) on the same in-memory backend, and against that rewrite
//! run on a `DuckDB` backend (SQL pushdown / streaming).
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use process_mining::analysis::object_centric::oc_statistics::locel_event_object_type_counts;
use process_mining::core::event_data::object_centric::linked_ocel::{
    LinkedOCELAccess, SlimLinkedOCEL,
};
use process_mining::core::event_data::object_centric::ocel_json::import_ocel_json_path;
use process_mining::core::event_data::object_centric::query::algos::{
    get_dfg_of_object_type_via_sequence, get_variants_of_object_type_via_sequence,
    type_counts_via_query,
};
use process_mining::discovery::object_centric::dfg::get_dfg_of_object_type;
use process_mining::discovery::object_centric::variants::get_variants_of_object_type;

mod common;

/// Representative object type: `items` is the largest object type in
/// `order-management.json` (7659 objects vs. 2000 `orders`), so its per-object trace
/// (dfg/variants) exercises the double-sort pushdown on a non-trivial row count.
const OB_TYPE: &str = "items";

fn bench_query_pushdown(c: &mut Criterion) {
    // --- Setup ONCE, outside all timed closures. ---
    // Both in-memory arms run on `slim`, so a group compares the algorithm, not the backend.
    let ocel = import_ocel_json_path(common::order_management("json")).expect("import OCEL json");
    let slim = SlimLinkedOCEL::from_ocel(ocel);

    let (_tmp_dir, db) = common::duckdb_from_json("query_pushdown.duckdb");

    assert!(
        LinkedOCELAccess::get_ob_types(&slim).any(|t| t == OB_TYPE),
        "expected object type {OB_TYPE} to exist in order-management.json"
    );

    // --- Group: type_counts ---
    let mut g = c.benchmark_group("type_counts");
    g.bench_function("direct", |b| {
        b.iter(|| black_box(locel_event_object_type_counts(&slim)))
    });
    g.bench_function("query_inmem", |b| {
        b.iter(|| black_box(type_counts_via_query(&slim).unwrap()))
    });
    g.sample_size(20);
    g.bench_function("query_duckdb", |b| {
        b.iter(|| black_box(type_counts_via_query(&db).unwrap()))
    });
    g.finish();

    // --- Group: dfg ---
    let mut g = c.benchmark_group("dfg");
    g.bench_function("direct", |b| {
        b.iter(|| black_box(get_dfg_of_object_type(&slim, OB_TYPE.to_string())))
    });
    g.bench_function("query_inmem", |b| {
        b.iter(|| {
            black_box(get_dfg_of_object_type_via_sequence(&slim, OB_TYPE.to_string()).unwrap())
        })
    });
    g.sample_size(20);
    g.measurement_time(Duration::from_secs(15));
    g.bench_function("query_duckdb", |b| {
        b.iter(|| black_box(get_dfg_of_object_type_via_sequence(&db, OB_TYPE.to_string()).unwrap()))
    });
    g.finish();

    // --- Group: variants ---
    let mut g = c.benchmark_group("variants");
    g.bench_function("direct", |b| {
        b.iter(|| black_box(get_variants_of_object_type(&slim, OB_TYPE.to_string())))
    });
    g.bench_function("query_inmem", |b| {
        b.iter(|| {
            black_box(get_variants_of_object_type_via_sequence(&slim, OB_TYPE.to_string()).unwrap())
        })
    });
    g.sample_size(20);
    g.measurement_time(Duration::from_secs(15));
    g.bench_function("query_duckdb", |b| {
        b.iter(|| {
            black_box(get_variants_of_object_type_via_sequence(&db, OB_TYPE.to_string()).unwrap())
        })
    });
    g.finish();
}

criterion_group!(benches, bench_query_pushdown);
criterion_main!(benches);
