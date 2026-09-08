//! Wall-clock latency benchmark: `DuckDbLinkedOCEL` (out-of-core, SQL point queries /
//! keyset paging) vs in-memory `IndexLinkedOCEL`. Complements the memory benchmark
//! with real time numbers for the same access patterns.
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use process_mining::core::event_data::object_centric::linked_ocel::{
    IndexLinkedOCEL, LinkedOCELAccess, QueryableOCEL,
};
use process_mining::core::event_data::object_centric::ocel_json::import_ocel_json_path;

mod common;

fn bench_latency(c: &mut Criterion) {
    let (_tmp_dir, db) = common::duckdb_from_json("latency.duckdb");

    let idx = IndexLinkedOCEL::from_ocel(
        import_ocel_json_path(common::order_management("json")).expect("import OCEL json"),
    );

    let ids: Vec<String> = db.get_all_evs().collect();
    assert!(!ids.is_empty(), "test OCEL has no events");
    let single_id = ids[0].clone();
    let batch: Vec<String> = ids[..1000.min(ids.len())].to_vec();

    // --- Group 1: scalar_by_id -- fair id-string -> event-type comparison ---
    let mut g = c.benchmark_group("scalar_by_id");
    g.bench_function("duckdb", |b| {
        b.iter(|| black_box(db.get_ev_type_of(&single_id)))
    });
    g.bench_function("in_memory", |b| {
        b.iter(|| {
            let ix = idx.get_ev_by_id(&single_id).unwrap();
            black_box(<IndexLinkedOCEL as LinkedOCELAccess>::get_ev_type_of(
                &idx, ix,
            ))
        })
    });
    g.finish();

    // --- Group 2: batch_1000_types -- 1000 id -> event-type lookups ---
    let mut g = c.benchmark_group("batch_1000_types");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(30));
    g.bench_function("duckdb_single", |b| {
        b.iter(|| {
            for id in &batch {
                black_box(db.get_ev_type_of(id));
            }
        })
    });
    g.bench_function("duckdb_batch", |b| {
        b.iter(|| black_box(db.get_ev_types_of_batch(&batch)))
    });
    g.bench_function("in_memory", |b| {
        b.iter(|| {
            for id in &batch {
                let ix = idx.get_ev_by_id(id).unwrap();
                black_box(<IndexLinkedOCEL as LinkedOCELAccess>::get_ev_type_of(
                    &idx, ix,
                ));
            }
        })
    });
    g.finish();

    // --- Group 3: full_traversal -- count all events ---
    let mut g = c.benchmark_group("full_traversal");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(30));
    g.bench_function("duckdb", |b| b.iter(|| black_box(db.get_all_evs().count())));
    // Plain `.count()` here lets LLVM see through the `Map<Range<usize>, _>` and
    // fold it to an O(1) length read (observed: ~580ps, i.e. optimized away).
    // `black_box` each element to force a real per-event iteration.
    g.bench_function("in_memory", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for ev in QueryableOCEL::get_all_evs(&idx) {
                black_box(ev);
                count += 1;
            }
            black_box(count)
        })
    });
    g.finish();
}

criterion_group!(benches, bench_latency);
criterion_main!(benches);
