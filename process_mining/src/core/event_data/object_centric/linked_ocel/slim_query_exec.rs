//! `SlimLinkedOCEL`-native fast executor for the pushdown `Query` AST.
//!
//! `SlimLinkedOCEL` overrides [`QueryableOCEL::run_query`]/`run_query_fold` to call the entry
//! points here before the generic evaluator. Both interpret the *generic* `Query` AST, but work
//! directly over Slim's integer-index fields instead of the trait's `Cow<str>` accessors.
//!
//! Two shapes are specialized. Every other shape returns `None`/`false` so the caller falls back
//! to the generic evaluator:
//! 1. **Aggregate scan + one-hop** (the `type_counts` shape): [`batched_aggregate_eligible`]'s
//!    childless, single-relational-filter box with a `Type(var)` `group_by` and `Count`-only
//!    aggregates.
//! 2. **Rows per-seed trace** (the dfg/variants shape): [`rows_streaming_seed`]'s shape with an
//!    `Id`/`Type`/`Time`-only projection.
//!
//! Results are identical to the generic evaluator. `String`s are materialized only at emit.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::slim_linked_ocel::{EventIndex, ObjectIndex, QualifierIdx, SlimLinkedOCEL};
use crate::core::event_data::object_centric::query::eval::{
    agg_column_name, batched_aggregate_eligible, column_name, compile_box, output_consumer_exprs,
    rows_streaming_seed, single_relational_filter, BoxPlan, Handle, PlanStep, QueryResult, Value,
};
use crate::core::event_data::object_centric::query::model::{
    Agg, AggSpec, Box as QBox, Dir, Expr, Output, Query, RowsSpec, TypeConstraint, VarDecl, VarId,
    VarKind,
};

/// Below this many seeds, rayon's per-task overhead outweighs the win: run sequentially (mirrors
/// `query::eval`'s `PAR_THRESHOLD`).
const PAR_THRESHOLD: usize = 4096;
/// Seed-chunk granularity for the parallel folds: each rayon task reuses one dedup set across its
/// whole chunk, not one per seed.
const PAR_CHUNK: usize = 2048;

/// Resolved qualifier constraint for one extend step.
enum QualFilter {
    /// No qualifier restriction (`qualifier: None`): every edge matches.
    Any,
    /// Restricted to this interned qualifier.
    Is(QualifierIdx),
    /// The restriction names a qualifier absent from this OCEL: no edge can ever match.
    NeverMatches,
}

impl QualFilter {
    fn resolve(slim: &SlimLinkedOCEL, qualifier: &Option<String>) -> Self {
        match qualifier {
            None => QualFilter::Any,
            Some(q) => match slim.qualifier_idx_of(q) {
                Some(idx) => QualFilter::Is(idx),
                None => QualFilter::NeverMatches,
            },
        }
    }
}

/// Resolved type constraint for a variable: `None` = `Any`, `Some(mask)` = allowed native type
/// indices (`mask[type_idx]`). Built once per query, not once per candidate.
type TypeMask = Option<Vec<bool>>;

fn build_ev_type_mask(slim: &SlimLinkedOCEL, tc: &TypeConstraint) -> TypeMask {
    match tc {
        TypeConstraint::Any => None,
        TypeConstraint::OneOf(tys) => {
            let mut mask = vec![false; slim.num_ev_types()];
            for t in tys {
                if let Some(i) = slim.ev_type_index(t) {
                    mask[i] = true;
                }
            }
            Some(mask)
        }
    }
}

fn build_ob_type_mask(slim: &SlimLinkedOCEL, tc: &TypeConstraint) -> TypeMask {
    match tc {
        TypeConstraint::Any => None,
        TypeConstraint::OneOf(tys) => {
            let mut mask = vec![false; slim.num_ob_types()];
            for t in tys {
                if let Some(i) = slim.ob_type_index(t) {
                    mask[i] = true;
                }
            }
            Some(mask)
        }
    }
}

#[inline]
fn ev_type_ok(slim: &SlimLinkedOCEL, mask: &TypeMask, ev: EventIndex) -> bool {
    match mask {
        None => true,
        Some(m) => m[ev.get_ev(slim).event_type],
    }
}

#[inline]
fn ob_type_ok(slim: &SlimLinkedOCEL, mask: &TypeMask, ob: ObjectIndex) -> bool {
    match mask {
        None => true,
        Some(m) => m[ob.get_ob(slim).object_type],
    }
}

/// The extended var's kind, and the concrete native traversal `step` compiles to.
enum ExtendKind {
    /// Extend events related to the (object) seed via reverse E2O (`object.e2o_rev`).
    E2ORev,
    /// Extend objects related to the (event) seed via forward E2O (`event.relationships`).
    E2OFwd,
    /// Extend "from" objects via reverse O2O (`object.o2o_rev`).
    O2ORev,
    /// Extend "to" objects via forward O2O (`object.relationships`).
    O2OFwd,
}

impl ExtendKind {
    fn ext_is_event(&self) -> bool {
        matches!(self, ExtendKind::E2ORev)
    }
}

/// Classify a compiled extend `step`. Returns `None` (caller falls back) when a *reverse*
/// traversal carries a qualifier restriction: `e2o_rev`/`o2o_rev` store no qualifier, so honoring
/// one would require the per-source relationship re-scan this path exists to avoid.
fn classify_step(step: &PlanStep) -> Option<(ExtendKind, VarId, &Option<String>)> {
    match step {
        PlanStep::ExtendE2O {
            var,
            var_is_event,
            qualifier,
            ..
        } => {
            if *var_is_event {
                if qualifier.is_some() {
                    return None;
                }
                Some((ExtendKind::E2ORev, *var, qualifier))
            } else {
                Some((ExtendKind::E2OFwd, *var, qualifier))
            }
        }
        PlanStep::ExtendO2O {
            var,
            var_is_from,
            qualifier,
            ..
        } => {
            if *var_is_from {
                if qualifier.is_some() {
                    return None;
                }
                Some((ExtendKind::O2ORev, *var, qualifier))
            } else {
                Some((ExtendKind::O2OFwd, *var, qualifier))
            }
        }
        PlanStep::Scan(_) => None,
    }
}

/// Iterate the distinct in-type neighbors of `seed` under `step`, calling `emit(neighbor_index)`
/// for each. `seen` is a caller-reused dedup set, cleared here rather than reallocated. Matches the
/// generic `batched_extend_fold`/`stream_extend_rows` distinct-neighbor semantics exactly.
#[allow(clippy::too_many_arguments)]
fn for_each_neighbor(
    slim: &SlimLinkedOCEL,
    kind: &ExtendKind,
    qual: &QualFilter,
    ext_mask: &TypeMask,
    seed_ix: u32,
    seen: &mut FxHashSet<u32>,
    mut emit: impl FnMut(u32),
) {
    if let QualFilter::NeverMatches = qual {
        return;
    }
    seen.clear();
    match kind {
        ExtendKind::E2ORev => {
            let ob = ObjectIndex::from(seed_ix);
            for ev in &ob.get_ob(slim).e2o_rev {
                let ei = ev.into_inner();
                if ev_type_ok(slim, ext_mask, *ev) && seen.insert(ei) {
                    emit(ei);
                }
            }
        }
        ExtendKind::O2ORev => {
            let ob = ObjectIndex::from(seed_ix);
            for o in &ob.get_ob(slim).o2o_rev {
                let oi = o.into_inner();
                if ob_type_ok(slim, ext_mask, *o) && seen.insert(oi) {
                    emit(oi);
                }
            }
        }
        ExtendKind::E2OFwd => {
            let ev = EventIndex::from(seed_ix);
            for (q, o) in &ev.get_ev(slim).relationships {
                if !qual_ok(qual, *q) {
                    continue;
                }
                let oi = o.into_inner();
                if ob_type_ok(slim, ext_mask, *o) && seen.insert(oi) {
                    emit(oi);
                }
            }
        }
        ExtendKind::O2OFwd => {
            let ob = ObjectIndex::from(seed_ix);
            for (q, o) in &ob.get_ob(slim).relationships {
                if !qual_ok(qual, *q) {
                    continue;
                }
                let oi = o.into_inner();
                if ob_type_ok(slim, ext_mask, *o) && seen.insert(oi) {
                    emit(oi);
                }
            }
        }
    }
}

#[inline]
fn qual_ok(qual: &QualFilter, q: QualifierIdx) -> bool {
    match qual {
        QualFilter::Any => true,
        QualFilter::Is(target) => *target == q,
        QualFilter::NeverMatches => false,
    }
}

/// Native scan candidates for a seed var declaration, as raw `u32` indices (no `Entity` wrapper).
/// Mirrors the generic `candidates_for_var` (including its no-dedup `flat_map` over `OneOf`).
fn seed_candidates(slim: &SlimLinkedOCEL, decl: &VarDecl) -> Vec<u32> {
    match (&decl.kind, &decl.types) {
        (VarKind::Event, TypeConstraint::Any) => {
            slim.all_evs_native().map(|e| e.into_inner()).collect()
        }
        (VarKind::Event, TypeConstraint::OneOf(tys)) => tys
            .iter()
            .flat_map(|t| slim.get_evs_of_type(t))
            .map(|e| e.into_inner())
            .collect(),
        (VarKind::Object, TypeConstraint::Any) => {
            slim.all_obs_native().map(|o| o.into_inner()).collect()
        }
        (VarKind::Object, TypeConstraint::OneOf(tys)) => tys
            .iter()
            .flat_map(|t| slim.get_obs_of_type(t))
            .map(|o| o.into_inner())
            .collect(),
    }
}

/// Compile `query`'s root box to a plan, returning `None` for anything but a single childless box
/// whose plan is exactly `[Scan(seed), Extend(other)]` (the shape both native paths need).
fn compile_single_hop(slim: &SlimLinkedOCEL, query: &Query) -> Option<(BoxPlan, VarId)> {
    if query.validate().is_err() {
        return None;
    }
    // One top-level relational filter and nothing else: that filter *is* the extend step below,
    // so a binding needs no further per-row check. See `eval::single_relational_filter`.
    if !query.root.children.is_empty() || !single_relational_filter(&query.root) {
        return None;
    }
    let total_vars = query.collect_vars().len();
    let consumer_exprs = output_consumer_exprs(&query.output);
    let mut cache = std::collections::HashMap::new();
    let plan = compile_box(
        slim,
        &query.root,
        0,
        &vec![false; total_vars],
        &consumer_exprs,
        &mut cache,
    );
    if plan.steps.len() != 2 || !matches!(plan.steps[0], PlanStep::Scan(_)) {
        return None;
    }
    let seed_var = plan.steps[0].var();
    Some((plan, seed_var))
}

// --- Aggregate (type_counts) native path -----------------------------------------------------

/// Which side of the binding a `group_by` `Type(var)` column reads, plus that side's entity kind
/// (fixed for the whole query, so the group name is resolved by kind at finalize).
struct GroupCol {
    from_seed: bool,
    is_event: bool,
}

/// Try to specialize `Output::Aggregate` for the `type_counts` shape. Returns `None` (fallback)
/// unless the box matches [`batched_aggregate_eligible`], `group_by` is all `Type(var)` over the
/// two bound vars, and every aggregate is `Count`.
fn native_aggregate(
    slim: &SlimLinkedOCEL,
    query: &Query,
    spec: &AggSpec,
    plan: &BoxPlan,
    seed_var: VarId,
) -> Option<QueryResult> {
    if !batched_aggregate_eligible(&query.root, plan) {
        return None;
    }
    if !spec.aggregates.iter().all(|a| matches!(a, Agg::Count)) {
        return None;
    }
    let (kind, ext_var, qualifier) = classify_step(&plan.steps[1])?;
    let box_ = &query.root;
    let seed_decl = &box_.new_vars[seed_var];
    let ext_decl = &box_.new_vars[ext_var];
    let seed_is_event = matches!(seed_decl.kind, VarKind::Event);
    let ext_is_event = kind.ext_is_event();

    // Each group_by must be Type(seed_var) or Type(ext_var). The group key packs into a `u128`
    // (four 32-bit type-index lanes) to avoid a per-binding `Vec`, so a wider `group_by` falls
    // back to the generic evaluator.
    if spec.group_by.len() > 4 {
        return None;
    }
    let mut group_cols: Vec<GroupCol> = Vec::with_capacity(spec.group_by.len());
    for e in &spec.group_by {
        let Expr::Type(v) = e else { return None };
        if *v == seed_var {
            group_cols.push(GroupCol {
                from_seed: true,
                is_event: seed_is_event,
            });
        } else if *v == ext_var {
            group_cols.push(GroupCol {
                from_seed: false,
                is_event: ext_is_event,
            });
        } else {
            return None;
        }
    }

    let qual = QualFilter::resolve(slim, qualifier);
    let ext_mask = if ext_is_event {
        build_ev_type_mask(slim, &ext_decl.types)
    } else {
        build_ob_type_mask(slim, &ext_decl.types)
    };

    let seeds = seed_candidates(slim, seed_decl);

    let fold_chunk = |chunk: &[u32], acc: &mut FxHashMap<u128, i64>, seen: &mut FxHashSet<u32>| {
        for &seed_ix in chunk {
            let seed_ty = if seed_is_event {
                EventIndex::from(seed_ix).get_ev(slim).event_type as u128
            } else {
                ObjectIndex::from(seed_ix).get_ob(slim).object_type as u128
            };
            for_each_neighbor(slim, &kind, &qual, &ext_mask, seed_ix, seen, |nbr_ix| {
                let ext_ty = if ext_is_event {
                    EventIndex::from(nbr_ix).get_ev(slim).event_type as u128
                } else {
                    ObjectIndex::from(nbr_ix).get_ob(slim).object_type as u128
                };
                let mut key: u128 = 0;
                for (i, gc) in group_cols.iter().enumerate() {
                    let ty = if gc.from_seed { seed_ty } else { ext_ty };
                    key |= ty << (32 * i);
                }
                *acc.entry(key).or_insert(0) += 1;
            });
        }
    };

    let counts: FxHashMap<u128, i64> = if seeds.len() < PAR_THRESHOLD {
        let mut acc = FxHashMap::default();
        let mut seen = FxHashSet::default();
        fold_chunk(&seeds, &mut acc, &mut seen);
        acc
    } else {
        seeds
            .par_chunks(PAR_CHUNK)
            .fold(
                || (FxHashMap::default(), FxHashSet::default()),
                |(mut acc, mut seen), chunk| {
                    fold_chunk(chunk, &mut acc, &mut seen);
                    (acc, seen)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(FxHashMap::default, merge_count_maps)
    };

    Some(finalize_aggregate(slim, spec, &group_cols, counts))
}

fn merge_count_maps(mut a: FxHashMap<u128, i64>, b: FxHashMap<u128, i64>) -> FxHashMap<u128, i64> {
    for (k, v) in b {
        *a.entry(k).or_insert(0) += v;
    }
    a
}

fn finalize_aggregate(
    slim: &SlimLinkedOCEL,
    spec: &AggSpec,
    group_cols: &[GroupCol],
    counts: FxHashMap<u128, i64>,
) -> QueryResult {
    let mut rows: Vec<Vec<Value>> = counts
        .into_iter()
        .map(|(key, count)| {
            let mut row: Vec<Value> = group_cols
                .iter()
                .enumerate()
                .map(|(i, gc)| {
                    let idx = ((key >> (32 * i)) & 0xffff_ffff) as usize;
                    let name = if gc.is_event {
                        slim.ev_type_name(idx)
                    } else {
                        slim.ob_type_name(idx)
                    };
                    Value::Str(name.to_string())
                })
                .collect();
            for _ in &spec.aggregates {
                row.push(Value::Int(count));
            }
            row
        })
        .collect();

    if spec.order_by.is_empty() {
        rows.sort();
    } else {
        rows.sort_by(|a, b| {
            for (idx, dir) in &spec.order_by {
                let ord = a[*idx].cmp(&b[*idx]);
                let ord = if matches!(dir, Dir::Desc) {
                    ord.reverse()
                } else {
                    ord
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    if let Some(n) = spec.limit {
        rows.truncate(n);
    }

    let columns = spec
        .group_by
        .iter()
        .map(column_name)
        .chain(spec.aggregates.iter().map(agg_column_name))
        .collect();
    QueryResult { columns, rows }
}

// --- Rows (dfg/variants trace) native path ---------------------------------------------------

/// A projection/order column reduced to a native read against `(seed, neighbor)`.
enum Col {
    SeedId,
    SeedType,
    SeedTime,
    ExtId,
    ExtType,
    ExtTime,
}

fn expr_to_col(e: &Expr, seed_var: VarId, ext_var: VarId) -> Option<Col> {
    let (v, want) = match e {
        Expr::Id(v) => (*v, 0u8),
        Expr::Type(v) => (*v, 1u8),
        Expr::Time(v) => (*v, 2u8),
        _ => return None,
    };
    if v == seed_var {
        Some(match want {
            0 => Col::SeedId,
            1 => Col::SeedType,
            _ => Col::SeedTime,
        })
    } else if v == ext_var {
        Some(match want {
            0 => Col::ExtId,
            1 => Col::ExtType,
            _ => Col::ExtTime,
        })
    } else {
        None
    }
}

fn col_value(
    slim: &SlimLinkedOCEL,
    col: &Col,
    seed_ix: u32,
    seed_is_event: bool,
    ext_ix: u32,
    ext_is_event: bool,
) -> Value {
    match col {
        Col::SeedId => {
            if seed_is_event {
                Value::Str(EventIndex::from(seed_ix).get_ev(slim).id.clone())
            } else {
                Value::Str(ObjectIndex::from(seed_ix).get_ob(slim).id.clone())
            }
        }
        Col::SeedType => {
            if seed_is_event {
                Value::Str(
                    slim.ev_type_name(EventIndex::from(seed_ix).get_ev(slim).event_type)
                        .to_string(),
                )
            } else {
                Value::Str(
                    slim.ob_type_name(ObjectIndex::from(seed_ix).get_ob(slim).object_type)
                        .to_string(),
                )
            }
        }
        Col::SeedTime => {
            if seed_is_event {
                Value::Time(EventIndex::from(seed_ix).get_ev(slim).time)
            } else {
                Value::Null
            }
        }
        Col::ExtId => {
            if ext_is_event {
                Value::Str(EventIndex::from(ext_ix).get_ev(slim).id.clone())
            } else {
                Value::Str(ObjectIndex::from(ext_ix).get_ob(slim).id.clone())
            }
        }
        Col::ExtType => {
            if ext_is_event {
                Value::Str(
                    slim.ev_type_name(EventIndex::from(ext_ix).get_ev(slim).event_type)
                        .to_string(),
                )
            } else {
                Value::Str(
                    slim.ob_type_name(ObjectIndex::from(ext_ix).get_ob(slim).object_type)
                        .to_string(),
                )
            }
        }
        Col::ExtTime => {
            if ext_is_event {
                Value::Time(EventIndex::from(ext_ix).get_ev(slim).time)
            } else {
                Value::Null
            }
        }
    }
}

/// Compiled native form of the trace query: the seed var, its extend, and the projection/order
/// columns pre-resolved to native reads.
struct RowsPlan {
    kind: ExtendKind,
    seed_var: VarId,
    seed_is_event: bool,
    ext_is_event: bool,
    qual: QualFilter,
    ext_mask: TypeMask,
    project: Vec<Col>,
    /// Only the `order_by` columns that read the *ext* (child) var. `rows_streaming_seed`
    /// guarantees the sole seed-referencing key is `Id(seed)`, which is constant within a seed and
    /// already the seeds' own sort key, so the intra-seed ordering reduces to these ext columns.
    order_ext: Vec<(Col, Dir)>,
}

fn compile_rows(
    slim: &SlimLinkedOCEL,
    query: &Query,
    spec: &RowsSpec,
    plan: &BoxPlan,
    seed_var: VarId,
) -> Option<RowsPlan> {
    if rows_streaming_seed(&query.root, plan, spec) != Some(seed_var) {
        return None;
    }
    let (kind, ext_var, qualifier) = classify_step(&plan.steps[1])?;
    let box_ = &query.root;
    let seed_is_event = matches!(box_.new_vars[seed_var].kind, VarKind::Event);
    let ext_is_event = kind.ext_is_event();
    let ext_decl = &box_.new_vars[ext_var];

    let mut project = Vec::with_capacity(spec.project.len());
    for e in &spec.project {
        project.push(expr_to_col(e, seed_var, ext_var)?);
    }
    let mut order_ext = Vec::new();
    for (e, dir) in &spec.order_by {
        let col = expr_to_col(e, seed_var, ext_var)?;
        if matches!(col, Col::ExtId | Col::ExtType | Col::ExtTime) {
            order_ext.push((col, dir.clone()));
        }
    }

    let qual = QualFilter::resolve(slim, qualifier);
    let ext_mask = if ext_is_event {
        build_ev_type_mask(slim, &ext_decl.types)
    } else {
        build_ob_type_mask(slim, &ext_decl.types)
    };
    Some(RowsPlan {
        kind,
        seed_var,
        seed_is_event,
        ext_is_event,
        qual,
        ext_mask,
        project,
        order_ext,
    })
}

/// Seeds, sorted by id string ascending, per the trace query's `order_by[0] = (Id(seed), Asc)`.
/// Ids are unique, so this is the whole cross-seed ordering, matching the generic global sort.
fn sorted_seeds(slim: &SlimLinkedOCEL, box_: &QBox, rp: &RowsPlan) -> Vec<(String, u32)> {
    let decl = &box_.new_vars[rp.seed_var];
    let mut seeds: Vec<(String, u32)> = seed_candidates(slim, decl)
        .into_iter()
        .map(|ix| {
            let id = if rp.seed_is_event {
                EventIndex::from(ix).get_ev(slim).id.clone()
            } else {
                ObjectIndex::from(ix).get_ob(slim).id.clone()
            };
            (id, ix)
        })
        .collect();
    seeds.sort_by(|a, b| a.0.cmp(&b.0));
    seeds
}

/// One seed's `Value` rows, sorted within the seed by `order_by` (stable). `Id(seed)` is constant
/// per seed, so this reproduces the generic path's tie-break exactly (see `eval_rows_stream_sorted`).
fn seed_rows(
    slim: &SlimLinkedOCEL,
    rp: &RowsPlan,
    seed_ix: u32,
    seen: &mut FxHashSet<u32>,
) -> Vec<Vec<Value>> {
    // Children arrive in adjacency (ascending index) order and the sort below is stable, so ties
    // resolve in that same order -- exactly as the generic per-seed stable sort does.
    let mut children: Vec<u32> = Vec::new();
    for_each_neighbor(
        slim,
        &rp.kind,
        &rp.qual,
        &rp.ext_mask,
        seed_ix,
        seen,
        |ext_ix| {
            children.push(ext_ix);
        },
    );
    if !rp.order_ext.is_empty() {
        children.sort_by(|&a, &b| cmp_ext_key(slim, rp, a, b));
    }
    children
        .into_iter()
        .map(|ext_ix| {
            rp.project
                .iter()
                .map(|c| col_value(slim, c, seed_ix, rp.seed_is_event, ext_ix, rp.ext_is_event))
                .collect()
        })
        .collect()
}

fn cmp_ext_key(slim: &SlimLinkedOCEL, rp: &RowsPlan, a: u32, b: u32) -> std::cmp::Ordering {
    for (col, dir) in &rp.order_ext {
        let va = col_value(slim, col, 0, rp.seed_is_event, a, rp.ext_is_event);
        let vb = col_value(slim, col, 0, rp.seed_is_event, b, rp.ext_is_event);
        let ord = va.cmp(&vb);
        let ord = if matches!(dir, Dir::Desc) {
            ord.reverse()
        } else {
            ord
        };
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// Produce the trace query's rows in final (seed-id, ext-order) order and stream them to `sink`,
/// honoring `limit`. Seeds are visited in id order. Within each `PAR_CHUNK` flush group the
/// per-seed extend+sort runs in parallel over sub-chunks, then the group's rows are flushed in
/// order, which bounds peak extra memory to one group.
fn stream_rows_ordered(
    slim: &SlimLinkedOCEL,
    rp: &RowsPlan,
    seeds: &[(String, u32)],
    limit: Option<usize>,
    sink: &mut dyn FnMut(Vec<Value>),
) {
    let mut remaining = limit;
    if remaining == Some(0) {
        return;
    }
    if seeds.len() < PAR_THRESHOLD {
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        for (_, seed_ix) in seeds {
            for row in seed_rows(slim, rp, *seed_ix, &mut seen) {
                sink(row);
                if let Some(r) = remaining.as_mut() {
                    *r -= 1;
                    if *r == 0 {
                        return;
                    }
                }
            }
        }
        return;
    }
    const SUBCHUNK: usize = 128;
    for group in seeds.chunks(PAR_CHUNK) {
        let group_rows: Vec<Vec<Vec<Value>>> = group
            .par_chunks(SUBCHUNK)
            .map(|sub| {
                let mut seen: FxHashSet<u32> = FxHashSet::default();
                let mut out: Vec<Vec<Value>> = Vec::new();
                for (_, seed_ix) in sub {
                    out.extend(seed_rows(slim, rp, *seed_ix, &mut seen));
                }
                out
            })
            .collect();
        for rows in group_rows {
            for row in rows {
                sink(row);
                if let Some(r) = remaining.as_mut() {
                    *r -= 1;
                    if *r == 0 {
                        return;
                    }
                }
            }
        }
    }
}

fn native_rows(
    slim: &SlimLinkedOCEL,
    query: &Query,
    spec: &RowsSpec,
    plan: &BoxPlan,
    seed_var: VarId,
) -> Option<QueryResult> {
    let rp = compile_rows(slim, query, spec, plan, seed_var)?;
    let seeds = sorted_seeds(slim, &query.root, &rp);
    let mut rows: Vec<Vec<Value>> = Vec::new();
    stream_rows_ordered(slim, &rp, &seeds, spec.limit, &mut |row| rows.push(row));
    let columns = spec.project.iter().map(column_name).collect();
    Some(QueryResult { columns, rows })
}

fn native_rows_fold(
    slim: &SlimLinkedOCEL,
    query: &Query,
    spec: &RowsSpec,
    plan: &BoxPlan,
    seed_var: VarId,
    f: &mut dyn FnMut(&[Value]),
) -> bool {
    let Some(rp) = compile_rows(slim, query, spec, plan, seed_var) else {
        return false;
    };
    let seeds = sorted_seeds(slim, &query.root, &rp);
    stream_rows_ordered(slim, &rp, &seeds, spec.limit, &mut |row| f(&row));
    true
}

/// Handle counterpart of [`col_value`]: reads `(seed, neighbor)` as a cheap [`Handle`] (native
/// index / interned type index) instead of materializing a `String`. Object-side `Time` columns
/// have no handle representation, so [`handles_representable`] rejects them upstream and those
/// arms are unreachable here.
fn col_handle(
    slim: &SlimLinkedOCEL,
    col: &Col,
    seed_ix: u32,
    seed_is_event: bool,
    ext_ix: u32,
    ext_is_event: bool,
) -> Handle<EventIndex, ObjectIndex, usize, usize> {
    match col {
        Col::SeedId => {
            if seed_is_event {
                Handle::Ev(EventIndex::from(seed_ix))
            } else {
                Handle::Ob(ObjectIndex::from(seed_ix))
            }
        }
        Col::SeedType => {
            if seed_is_event {
                Handle::EvType(EventIndex::from(seed_ix).get_ev(slim).event_type)
            } else {
                Handle::ObType(ObjectIndex::from(seed_ix).get_ob(slim).object_type)
            }
        }
        Col::SeedTime => {
            debug_assert!(seed_is_event, "handles_representable rejects object Time");
            Handle::Time(EventIndex::from(seed_ix).get_ev(slim).time)
        }
        Col::ExtId => {
            if ext_is_event {
                Handle::Ev(EventIndex::from(ext_ix))
            } else {
                Handle::Ob(ObjectIndex::from(ext_ix))
            }
        }
        Col::ExtType => {
            if ext_is_event {
                Handle::EvType(EventIndex::from(ext_ix).get_ev(slim).event_type)
            } else {
                Handle::ObType(ObjectIndex::from(ext_ix).get_ob(slim).object_type)
            }
        }
        Col::ExtTime => {
            debug_assert!(ext_is_event, "handles_representable rejects object Time");
            Handle::Time(EventIndex::from(ext_ix).get_ev(slim).time)
        }
    }
}

/// A projection is handle-representable unless it reads an *object*'s `Time`, which `col_value`
/// yields as `Value::Null` and [`Handle`] has no null for.
fn handles_representable(rp: &RowsPlan) -> bool {
    rp.project.iter().all(|c| match c {
        Col::SeedTime => rp.seed_is_event,
        Col::ExtTime => rp.ext_is_event,
        _ => true,
    })
}

// --- Public entry points (called from `SlimLinkedOCEL`'s `QueryableOCEL` impl) ----------------

/// Native `run_query` for the two specialized shapes. `None` = shape not covered (fall back).
pub(crate) fn native_run_query(slim: &SlimLinkedOCEL, query: &Query) -> Option<QueryResult> {
    let (plan, seed_var) = compile_single_hop(slim, query)?;
    match &query.output {
        Output::Aggregate(spec) => native_aggregate(slim, query, spec, &plan, seed_var),
        Output::Rows(spec) => native_rows(slim, query, spec, &plan, seed_var),
    }
}

/// Native `run_query_fold` for the Rows trace shape. `false` = not covered, and the caller falls
/// back to `run_query` + iterate.
pub(crate) fn native_run_query_fold(
    slim: &SlimLinkedOCEL,
    query: &Query,
    f: &mut dyn FnMut(&[Value]),
) -> bool {
    let Output::Rows(spec) = &query.output else {
        return false;
    };
    let Some((plan, seed_var)) = compile_single_hop(slim, query) else {
        return false;
    };
    native_rows_fold(slim, query, spec, &plan, seed_var, f)
}

type Hdl = Handle<EventIndex, ObjectIndex, usize, usize>;

/// Sort a seed's child indices by `order_by` (stable). Fast path for the dfg/variants trace shape,
/// which orders by exactly `Time(event) Asc`: `sort_by_key` on the raw timestamp avoids
/// `cmp_ext_key` rebuilding a `Value::Time` on every comparison. Both are stable and key on the
/// same time, so the tie-break is identical.
fn sort_children(slim: &SlimLinkedOCEL, rp: &RowsPlan, children: &mut [u32]) {
    if rp.order_ext.is_empty() {
        return;
    }
    if rp.ext_is_event
        && rp.order_ext.len() == 1
        && matches!(rp.order_ext[0], (Col::ExtTime, Dir::Asc))
    {
        children.sort_by_key(|&e| EventIndex::from(e).get_ev(slim).time);
    } else {
        children.sort_by(|&a, &b| cmp_ext_key(slim, rp, a, b));
    }
}

/// Per-seed reused scratch: the child-index buffer plus one `Vec<Handle>` per projected column.
/// `Handle` is `Copy`, so `clear()` keeps each column's capacity -- no per-seed allocation after
/// warmup.
struct SeedScratch {
    seen: FxHashSet<u32>,
    children: Vec<u32>,
    cols: Vec<Vec<Hdl>>,
}

impl SeedScratch {
    fn new(n_cols: usize) -> Self {
        SeedScratch {
            seen: FxHashSet::default(),
            children: Vec::new(),
            cols: (0..n_cols).map(|_| Vec::new()).collect(),
        }
    }
}

/// Fill `scratch`'s per-column buffers with one seed's ordered, projected handles (column-major).
fn seed_cols(slim: &SlimLinkedOCEL, rp: &RowsPlan, seed_ix: u32, scratch: &mut SeedScratch) {
    scratch.children.clear();
    for_each_neighbor(
        slim,
        &rp.kind,
        &rp.qual,
        &rp.ext_mask,
        seed_ix,
        &mut scratch.seen,
        |ext_ix| scratch.children.push(ext_ix),
    );
    sort_children(slim, rp, &mut scratch.children);
    for col in scratch.cols.iter_mut() {
        col.clear();
    }
    for &ext_ix in scratch.children.iter() {
        for (i, colspec) in rp.project.iter().enumerate() {
            scratch.cols[i].push(col_handle(
                slim,
                colspec,
                seed_ix,
                rp.seed_is_event,
                ext_ix,
                rp.ext_is_event,
            ));
        }
    }
}

/// Translate a single-`Sequence` aggregate into the equivalent per-object trace `RowsSpec`:
/// project `[group-key, of]`, order by `[(group-key, Asc)] ++ by`. `None` for any other aggregate
/// shape. Lets dfg/variants keep one declarative query definition and still fold in handle space.
fn sequence_as_trace_rows(spec: &AggSpec) -> Option<RowsSpec> {
    let [Agg::Sequence { of, by }] = spec.aggregates.as_slice() else {
        return None;
    };
    if spec.group_by.len() != 1 || !matches!(spec.group_by[0], Expr::Id(_)) {
        return None;
    }
    // The trace rewrite reorders by the group key and folds every group, so it cannot honour an
    // aggregate-level `order_by`/`limit`.
    if !spec.order_by.is_empty() || spec.limit.is_some() {
        return None;
    }
    let key = spec.group_by[0].clone();
    let mut order_by = vec![(key.clone(), Dir::Asc)];
    order_by.extend(by.iter().cloned());
    Some(RowsSpec {
        project: vec![key, of.clone()],
        order_by,
        limit: None,
    })
}

/// Parallel per-seed fused fold for the Rows trace shape: each seed's projected handles are built
/// column-major into reused buffers and folded into a thread-local `A`, then merged by `reduce`.
/// `None` means the shape is not covered and the caller falls back.
// The `reduce` closure is deliberately not passed by value: rayon's `reduce` requires `Send`,
// which `Reduce` is not bound by, so the wrapper closure borrows it instead.
#[allow(clippy::redundant_closure)]
pub(crate) fn native_fold_seeds<A, Init, FoldSeed, Reduce>(
    slim: &SlimLinkedOCEL,
    query: &Query,
    init: Init,
    fold_seed: FoldSeed,
    reduce: Reduce,
) -> Option<A>
where
    A: Send,
    Init: Fn() -> A + Sync,
    FoldSeed: Fn(&mut A, &[&[Hdl]]) + Sync,
    Reduce: Fn(A, A) -> A + Sync,
{
    // The Rows trace query and the declarative `Sequence` aggregate are the same per-object
    // ordered trace, so both fold identically here.
    let derived;
    let spec: &RowsSpec = match &query.output {
        Output::Rows(s) => s,
        Output::Aggregate(a) => {
            derived = sequence_as_trace_rows(a)?;
            &derived
        }
    };
    let (plan, seed_var) = compile_single_hop(slim, query)?;
    let rp = compile_rows(slim, query, spec, &plan, seed_var)?;
    if !handles_representable(&rp) {
        return None;
    }
    // A commutative fold+reduce is order-independent across seeds, so this path skips
    // `sorted_seeds`' id-string clone+sort. Only the intra-seed time order matters, and
    // `sort_children` preserves it.
    let seeds: Vec<u32> = seed_candidates(slim, &query.root.new_vars[seed_var]);
    let n_cols = rp.project.len();

    // Adaptive `par_iter().fold()`, not fixed-size `par_chunks`: object types with few but heavy
    // seeds still spread across cores, where a fixed-size chunk would leave them in one sequential
    // task. Each fold job keeps one `SeedScratch` and reuses it across the seeds it gets.
    let acc = seeds
        .par_iter()
        .fold(
            || (init(), SeedScratch::new(n_cols)),
            |(mut acc, mut scratch), &seed_ix| {
                seed_cols(slim, &rp, seed_ix, &mut scratch);
                // One slice pointer per column per seed, not a per-row `Handle` allocation.
                let col_refs: Vec<&[Hdl]> = scratch.cols.iter().map(|c| c.as_slice()).collect();
                fold_seed(&mut acc, &col_refs);
                (acc, scratch)
            },
        )
        .map(|(a, _)| a)
        .reduce(&init, |a, b| reduce(a, b));
    Some(acc)
}
