//! GATE benchmark: `QueryableOCEL` (Cow/owned returns) vs `LinkedOCELAccess`
//! (reference returns), for the in-memory `IndexLinkedOCEL` and `SlimLinkedOCEL`
//! backends. Purpose: quantify whether the Cow/owned `QueryableOCEL` design
//! regresses performance vs the reference-returning `LinkedOCELAccess`.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use process_mining::core::event_data::object_centric::{
    linked_ocel::{IndexLinkedOCEL, LinkedOCELAccess, QueryableOCEL, SlimLinkedOCEL},
    ocel_json::import_ocel_json_path,
};
mod common;

fn load_ocel() -> process_mining::OCEL {
    import_ocel_json_path(common::order_management("json")).unwrap()
}

/// Sum the byte-lengths of event types matching a predicate. Isolates the
/// `Cow<str>`-wrapper cost of `QueryableOCEL::get_ev_type_of` (expected
/// `Cow::Borrowed` for in-memory backends -> near parity with `&str`).
fn scalar_peek_queryable<Q: QueryableOCEL>(ocel: &Q) -> usize {
    let mut total = 0usize;
    for ev in ocel.get_all_evs() {
        let ty = ocel.get_ev_type_of(&ev);
        if ty.len() > 3 {
            total += ty.len();
        }
    }
    total
}

fn scalar_peek_linked<'a, L: LinkedOCELAccess<'a>>(ocel: &'a L) -> usize {
    let mut total = 0usize;
    for ev in ocel.get_all_evs() {
        let ty = ocel.get_ev_type_of(&ev);
        if ty.len() > 3 {
            total += ty.len();
        }
    }
    total
}

/// Count all (qualifier, object) pairs across every event's E2O relations.
/// Neither `QueryableOCEL::get_e2o` (`(Cow<str>, Repr)`) nor `LinkedOCELAccess::get_e2o`
/// (`(&str, &Repr)`) allocates for these backends, so this measures what the `Cow` wrapper
/// and the by-value object repr cost on a relation-heavy traversal.
fn iter_traversal_queryable<Q: QueryableOCEL>(ocel: &Q) -> usize {
    let mut count = 0usize;
    for ev in ocel.get_all_evs() {
        for (qual, _ob) in ocel.get_e2o(&ev) {
            count += qual.len();
        }
    }
    count
}

fn iter_traversal_linked<'a, L: LinkedOCELAccess<'a>>(ocel: &'a L) -> usize {
    let mut count = 0usize;
    for ev in ocel.get_all_evs() {
        for (qual, _ob) in ocel.get_e2o(ev) {
            count += qual.len();
        }
    }
    count
}

/// Materialize all event types into a `Vec<String>`. Both sides allocate a
/// `String` per event, so this is the expected-parity baseline group.
fn collect_types_queryable<Q: QueryableOCEL>(ocel: &Q) -> Vec<String> {
    ocel.get_all_evs()
        .map(|ev| ocel.get_ev_type_of(&ev).into_owned())
        .collect()
}

fn collect_types_linked<'a, L: LinkedOCELAccess<'a>>(ocel: &'a L) -> Vec<String> {
    ocel.get_all_evs()
        .map(|ev| ocel.get_ev_type_of(&ev).to_string())
        .collect()
}

fn bench_index(c: &mut Criterion) {
    let ocel = load_ocel();
    let index_locel = IndexLinkedOCEL::from_ocel(ocel);

    let mut g = c.benchmark_group("scalar_peek/index");
    g.bench_function("queryable", |b| {
        b.iter(|| black_box(scalar_peek_queryable(&index_locel)))
    });
    g.bench_function("linked", |b| {
        b.iter(|| black_box(scalar_peek_linked(&index_locel)))
    });
    g.finish();

    let mut g = c.benchmark_group("iter_traversal/index");
    g.bench_function("queryable", |b| {
        b.iter(|| black_box(iter_traversal_queryable(&index_locel)))
    });
    g.bench_function("linked", |b| {
        b.iter(|| black_box(iter_traversal_linked(&index_locel)))
    });
    g.finish();

    let mut g = c.benchmark_group("collect_types/index");
    g.bench_function("queryable", |b| {
        b.iter(|| black_box(collect_types_queryable(&index_locel)))
    });
    g.bench_function("linked", |b| {
        b.iter(|| black_box(collect_types_linked(&index_locel)))
    });
    g.finish();
}

fn bench_slim(c: &mut Criterion) {
    let ocel = load_ocel();
    let slim_locel = SlimLinkedOCEL::from_ocel(ocel);

    let mut g = c.benchmark_group("scalar_peek/slim");
    g.bench_function("queryable", |b| {
        b.iter(|| black_box(scalar_peek_queryable(&slim_locel)))
    });
    g.bench_function("linked", |b| {
        b.iter(|| black_box(scalar_peek_linked(&slim_locel)))
    });
    g.finish();

    let mut g = c.benchmark_group("iter_traversal/slim");
    g.bench_function("queryable", |b| {
        b.iter(|| black_box(iter_traversal_queryable(&slim_locel)))
    });
    g.bench_function("linked", |b| {
        b.iter(|| black_box(iter_traversal_linked(&slim_locel)))
    });
    g.finish();

    let mut g = c.benchmark_group("collect_types/slim");
    g.bench_function("queryable", |b| {
        b.iter(|| black_box(collect_types_queryable(&slim_locel)))
    });
    g.bench_function("linked", |b| {
        b.iter(|| black_box(collect_types_linked(&slim_locel)))
    });
    g.finish();
}

criterion_group!(benches, bench_index, bench_slim);
criterion_main!(benches);
