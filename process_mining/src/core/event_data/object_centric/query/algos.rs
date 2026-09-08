//! Backend-agnostic reimplementations of representative algorithms on the query layer.
//!
//! Each function is generic over [`QueryableOCEL`]: one query definition runs against an in-memory
//! or a `DuckDB` backend, with no branching on backend.

use std::collections::HashMap;

use crate::core::event_data::object_centric::linked_ocel::QueryableOCEL;
use crate::core::event_data::object_centric::query::eval::{Handle, Value};
use crate::core::event_data::object_centric::query::model::{
    Agg, AggSpec, Box, Dir, Expr, Filter, Query, TypeConstraint, VarDecl, VarKind,
};
use crate::core::event_data::object_centric::query::Output;

/// Query-layer counterpart of
/// [`locel_event_object_type_counts`](crate::analysis::object_centric::oc_statistics::locel_event_object_type_counts).
///
/// Counts distinct `(event, object)` bindings per type (OCPQ binding semantics): an event and
/// object linked by several qualifiers count once, not once per qualifier -- so on
/// multi-qualifier data this differs from `locel_event_object_type_counts`, which counts one per
/// relationship row. Row order is unspecified.
pub fn type_counts_via_query<Q: QueryableOCEL>(
    ocel: &Q,
) -> Result<Vec<(String, String, i64)>, String> {
    let query = Query {
        root: Box {
            new_vars: vec![
                VarDecl {
                    kind: VarKind::Event,
                    types: TypeConstraint::Any,
                },
                VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::Any,
                },
            ],
            filters: vec![Filter::E2O {
                event: 0,
                object: 1,
                qualifier: None,
            }],
            children: vec![],
        },
        output: Output::Aggregate(AggSpec {
            group_by: vec![Expr::Type(0), Expr::Type(1)],
            aggregates: vec![Agg::Count],
            having: vec![],
            order_by: vec![],
            limit: None,
        }),
        emits: Vec::new(),
    };
    let result = ocel.run_query(&query)?;
    result
        .rows
        .into_iter()
        .map(|row| {
            let mut it = row.into_iter();
            let (Some(Value::Str(ev_ty)), Some(Value::Str(ob_ty)), Some(Value::Int(count))) =
                (it.next(), it.next(), it.next())
            else {
                return Err("type_counts: backend did not project (str, str, int)".to_string());
            };
            Ok((ev_ty, ob_ty, count))
        })
        .collect()
}

/// One row per object of `ob_type`: `(object id, time-ordered list of related event types)`.
/// The primitive dfg/variants reduce over.
fn object_trace_seq_query(ob_type: &str) -> Query {
    Query {
        root: Box {
            new_vars: vec![
                VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec![ob_type.to_string()]),
                },
                VarDecl {
                    kind: VarKind::Event,
                    types: TypeConstraint::Any,
                },
            ],
            filters: vec![Filter::E2O {
                event: 1,
                object: 0,
                qualifier: None,
            }],
            children: vec![],
        },
        output: Output::Aggregate(AggSpec {
            group_by: vec![Expr::Id(0)],
            aggregates: vec![Agg::Sequence {
                of: Expr::Type(1),
                by: vec![(Expr::Time(1), Dir::Asc)],
            }],
            having: vec![],
            order_by: vec![],
            limit: None,
        }),
        emits: Vec::new(),
    }
}

/// Directly-follows edges: `((from_type, to_type), count)` pairs.
pub type DfgEdges = Vec<((String, String), usize)>;

/// dfg over [`object_trace_seq_query`], reduced with `windows(2)`: in generic `Q::EvTypeId`
/// handle space in-memory, or one trace at a time through `run_query_fold` otherwise.
pub fn get_dfg_of_object_type_via_sequence<Q: QueryableOCEL>(
    ocel: &Q,
    ob_type: String,
) -> Result<DfgEdges, String>
where
    Q::EvTypeId: Send,
{
    let query = object_trace_seq_query(&ob_type);

    // Fold `Q::EvTypeId` pairs, manifesting names only for the few resulting edges.
    let handle_counts = ocel.run_query_fold_seeds(
        &query,
        HashMap::<(Q::EvTypeId, Q::EvTypeId), usize>::new,
        |acc, cols| {
            let mut prev: Option<Q::EvTypeId> = None;
            for h in cols[1] {
                let Handle::EvType(t) = h else {
                    unreachable!("Sequence.of = Type(event) -> EvType handle in column 1")
                };
                if let Some(p) = prev {
                    *acc.entry((p, *t)).or_insert(0) += 1;
                }
                prev = Some(*t);
            }
        },
        |mut a: HashMap<(Q::EvTypeId, Q::EvTypeId), usize>, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        },
    )?;

    if let Some(counts) = handle_counts {
        let mut result: Vec<((String, String), usize)> = counts
            .into_iter()
            .map(|((a, b), c)| {
                (
                    (
                        ocel.resolve_ev_type(a).into_owned(),
                        ocel.resolve_ev_type(b).into_owned(),
                    ),
                    c,
                )
            })
            .collect();
        result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        return Ok(result);
    }

    // Fallback via `Value`. The fold closure cannot return an error, so an unexpected row shape
    // is recorded and surfaced afterwards.
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    let mut shape_err: Option<String> = None;
    ocel.run_query_fold(&query, |row| {
        if shape_err.is_some() {
            return;
        }
        let Value::List(trace) = &row[1] else {
            shape_err = Some("dfg: backend did not project a List in column 1".to_string());
            return;
        };
        for w in trace.windows(2) {
            let (Value::Str(a), Value::Str(b)) = (&w[0], &w[1]) else {
                shape_err = Some("dfg: trace elements are not Type(event) strings".to_string());
                return;
            };
            *counts.entry((a.clone(), b.clone())).or_insert(0) += 1;
        }
    })?;
    if let Some(e) = shape_err {
        return Err(e);
    }
    let mut result: Vec<((String, String), usize)> = counts.into_iter().collect();
    result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(result)
}

/// variants over [`object_trace_seq_query`], folding each trace into a variant key. Same
/// two-path structure as [`get_dfg_of_object_type_via_sequence`].
pub fn get_variants_of_object_type_via_sequence<Q: QueryableOCEL>(
    ocel: &Q,
    ob_type: String,
) -> Result<Vec<(Vec<String>, usize)>, String>
where
    Q::EvTypeId: Send,
{
    let query = object_trace_seq_query(&ob_type);

    let handle_counts = ocel.run_query_fold_seeds(
        &query,
        HashMap::<Vec<Q::EvTypeId>, usize>::new,
        |acc, cols| {
            let trace: Vec<Q::EvTypeId> = cols[1]
                .iter()
                .map(|h| {
                    let Handle::EvType(t) = h else {
                        unreachable!("Sequence.of = Type(event) -> EvType handle")
                    };
                    *t
                })
                .collect();
            if !trace.is_empty() {
                *acc.entry(trace).or_insert(0) += 1;
            }
        },
        |mut a: HashMap<Vec<Q::EvTypeId>, usize>, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        },
    )?;

    if let Some(counts) = handle_counts {
        let mut result: Vec<(Vec<String>, usize)> = counts
            .into_iter()
            .map(|(t, c)| {
                (
                    t.into_iter()
                        .map(|id| ocel.resolve_ev_type(id).into_owned())
                        .collect(),
                    c,
                )
            })
            .collect();
        result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        return Ok(result);
    }

    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
    let mut shape_err: Option<String> = None;
    ocel.run_query_fold(&query, |row| {
        if shape_err.is_some() {
            return;
        }
        let Value::List(trace) = &row[1] else {
            shape_err = Some("variants: backend did not project a List in column 1".to_string());
            return;
        };
        let mut t: Vec<String> = Vec::with_capacity(trace.len());
        for v in trace {
            let Value::Str(s) = v else {
                shape_err =
                    Some("variants: trace elements are not Type(event) strings".to_string());
                return;
            };
            t.push(s.clone());
        }
        if !t.is_empty() {
            *counts.entry(t).or_insert(0) += 1;
        }
    })?;
    if let Some(e) = shape_err {
        return Err(e);
    }
    let mut result: Vec<(Vec<String>, usize)> = counts.into_iter().collect();
    result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(result)
}

// `locel_conversion_rate` has no counterpart here: a two-hop correlated existence check is two
// `Aggregate(Count)` queries divided in Rust, not one query.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::oc_statistics::locel_event_object_type_counts;
    use crate::core::event_data::object_centric::linked_ocel::{IndexLinkedOCEL, LinkedOCELAccess};
    use crate::core::event_data::object_centric::ocel_json::import_ocel_json_path;
    use crate::discovery::object_centric::dfg::get_dfg_of_object_type;
    use crate::discovery::object_centric::variants::get_variants_of_object_type;
    use crate::test_utils::get_test_data_path;
    use std::collections::HashSet;

    fn src_path() -> std::path::PathBuf {
        get_test_data_path()
            .join("ocel")
            .join("order-management.json")
    }

    fn index_ocel() -> IndexLinkedOCEL {
        IndexLinkedOCEL::from_ocel(import_ocel_json_path(src_path()).unwrap())
    }

    fn slim_ocel() -> crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL {
        crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL::from_ocel(
            import_ocel_json_path(src_path()).unwrap(),
        )
    }

    fn as_set<T: Eq + std::hash::Hash>(v: Vec<T>) -> HashSet<T> {
        v.into_iter().collect()
    }

    /// Reference for `type_counts_via_query`'s distinct-`(event, object)`-pair-per-type
    /// semantics, computed independently of the query layer.
    fn expected_distinct_pair_type_counts(
        ocel: &IndexLinkedOCEL,
    ) -> HashSet<(String, String, i64)> {
        let mut counts: HashMap<(String, String), i64> = HashMap::new();
        for ev in LinkedOCELAccess::get_all_evs(ocel) {
            let ev_ty = LinkedOCELAccess::get_ev_type_of(ocel, ev).to_string();
            let mut seen = HashSet::new();
            for (_, ob) in LinkedOCELAccess::get_e2o(ocel, ev) {
                if seen.insert(*ob) {
                    let ob_ty = LinkedOCELAccess::get_ob_type_of(ocel, *ob).to_string();
                    *counts.entry((ev_ty.clone(), ob_ty)).or_insert(0) += 1;
                }
            }
        }
        counts.into_iter().map(|((e, o), c)| (e, o, c)).collect()
    }

    /// Whether any object of `ob_type` has two related events at the exact same timestamp.
    ///
    /// SQL `ORDER BY object_id, time` has no tie-break for equal `(object, time)` keys, unlike
    /// the in-memory path, whose stable sort reproduces `SlimLinkedOCEL`'s insertion order, so
    /// strict `DuckDB` assertions only hold for object types without such a tie.
    #[cfg(feature = "ocel-duckdb")]
    fn ob_type_has_timestamp_ties(
        slim: &crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL,
        ob_type: &str,
    ) -> bool {
        LinkedOCELAccess::get_obs_of_type(slim, ob_type).any(|ob| {
            let mut times: Vec<_> = ob.get_e2o_rev(slim).map(|e| *e.get_time(slim)).collect();
            times.sort();
            times.windows(2).any(|w| w[0] == w[1])
        })
    }

    #[test]
    fn type_counts_matches_existing_index_backend() {
        let ocel = index_ocel();
        let via_query = type_counts_via_query(&ocel).unwrap();
        let expected = expected_distinct_pair_type_counts(&ocel);
        assert!(!expected.is_empty());
        assert_eq!(as_set(via_query), expected);

        // The row-counting reference over-counts multi-qualifier pairs, but must not invent or
        // drop a (event_type, object_type) key.
        let existing_keys: HashSet<(String, String)> = locel_event_object_type_counts(&slim_ocel())
            .into_iter()
            .map(|(e, o, _)| (e, o))
            .collect();
        let expected_keys: HashSet<(String, String)> =
            expected.into_iter().map(|(e, o, _)| (e, o)).collect();
        assert_eq!(existing_keys, expected_keys);
    }

    // Covers both the handle path (Slim) and the `Value` fallback (Index has no handle executor).
    #[test]
    fn dfg_via_sequence_matches_existing() {
        let index = index_ocel();
        let slim = slim_ocel();
        for ob_type in LinkedOCELAccess::get_ob_types(&index) {
            let existing = get_dfg_of_object_type(&slim, ob_type.to_string());
            let seq_index =
                get_dfg_of_object_type_via_sequence(&index, ob_type.to_string()).unwrap();
            let seq_slim = get_dfg_of_object_type_via_sequence(&slim, ob_type.to_string()).unwrap();
            assert_eq!(
                as_set(seq_index),
                as_set(existing.clone()),
                "dfg seq(index) {ob_type}"
            );
            assert_eq!(
                as_set(seq_slim),
                as_set(existing),
                "dfg seq(slim) {ob_type}"
            );
        }
    }

    #[test]
    fn variants_via_sequence_matches_existing() {
        let index = index_ocel();
        let slim = slim_ocel();
        for ob_type in LinkedOCELAccess::get_ob_types(&index) {
            let existing = get_variants_of_object_type(&slim, ob_type.to_string());
            let seq_index =
                get_variants_of_object_type_via_sequence(&index, ob_type.to_string()).unwrap();
            let seq_slim =
                get_variants_of_object_type_via_sequence(&slim, ob_type.to_string()).unwrap();
            assert_eq!(
                as_set(seq_index),
                as_set(existing.clone()),
                "variants seq(index) {ob_type}"
            );
            assert_eq!(
                as_set(seq_slim),
                as_set(existing),
                "variants seq(slim) {ob_type}"
            );
        }
    }

    // On tie-affected object types only the total transition count is backend-independent, see
    // `ob_type_has_timestamp_ties`.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn dfg_via_sequence_matches_existing_duckdb() {
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };
        let out = get_test_data_path()
            .join("export")
            .join("algos-dfg-seq-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(src_path(), &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();
        let slim = slim_ocel();

        for ob_type in LinkedOCELAccess::get_ob_types(&index_ocel()) {
            let ob_type = ob_type.to_string();
            let via_seq = get_dfg_of_object_type_via_sequence(&db, ob_type.clone()).unwrap();
            let existing = get_dfg_of_object_type(&slim, ob_type.clone());
            if ob_type_has_timestamp_ties(&slim, &ob_type) {
                let sum_vq: usize = via_seq.iter().map(|(_, c)| *c).sum();
                let sum_ex: usize = existing.iter().map(|(_, c)| *c).sum();
                assert_eq!(
                    sum_vq, sum_ex,
                    "dfg total mismatch (duckdb) for tie type {ob_type}"
                );
            } else {
                assert_eq!(
                    as_set(via_seq),
                    as_set(existing),
                    "dfg mismatch (duckdb) {ob_type}"
                );
            }
        }
    }
}
