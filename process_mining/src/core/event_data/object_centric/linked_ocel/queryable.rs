//! Ownership-flexible OCEL access: no trait-level lifetime, `Cow`/owned returns, so an
//! owned-result backend such as `DuckDB` can implement it alongside in-memory backends.
use std::borrow::Cow;
use std::hash::Hash;

use chrono::{DateTime, FixedOffset};

use crate::core::event_data::object_centric::query::eval::{
    evaluate, try_stream_rows, Handle, QueryResult, Value,
};
use crate::core::event_data::object_centric::query::Query;
use crate::core::event_data::object_centric::OCELAttributeValue;

/// Ownership-flexible access to an OCEL, returning `Cow`/owned values so that both
/// in-memory and owned-result (e.g., `DuckDB`) backends can implement it.
pub trait QueryableOCEL {
    /// Return and argument type/representation for events. `Eq + Hash` lets `query::eval` key
    /// bindings on the cheap repr itself (e.g. an integer index) rather than an allocated id.
    type EventRepr: Clone + Eq + Hash;
    /// Return and argument type/representation for objects. See [`QueryableOCEL::EventRepr`].
    type ObjectRepr: Clone + Eq + Hash;

    /// Cheap, `Copy` handle for an event type, ideally the backend's own interned type index, so
    /// aggregate groups key on one machine word instead of re-hashing the type name per binding.
    /// A backend without such an index may return any correct handle.
    type EvTypeId: Copy + Eq + Hash;
    /// See [`QueryableOCEL::EvTypeId`].
    type ObTypeId: Copy + Eq + Hash;

    /// Get all events in the dataset
    fn get_all_evs(&self) -> impl Iterator<Item = Self::EventRepr> + '_;
    /// Get all objects in the dataset
    fn get_all_obs(&self) -> impl Iterator<Item = Self::ObjectRepr> + '_;

    /// Get the ID of an event
    fn get_ev_id(&self, ev: &Self::EventRepr) -> Cow<'_, str>;
    /// Get the ID of an object
    fn get_ob_id(&self, ob: &Self::ObjectRepr) -> Cow<'_, str>;
    /// Get the event type (i.e., activity) of an event
    fn get_ev_type_of(&self, ev: &Self::EventRepr) -> Cow<'_, str>;
    /// Get the object type of an object
    fn get_ob_type_of(&self, ob: &Self::ObjectRepr) -> Cow<'_, str>;
    /// Get the timestamp of an event
    fn get_ev_time(&self, ev: &Self::EventRepr) -> DateTime<FixedOffset>;

    /// Cheap handle for `ev`'s event type -- see [`QueryableOCEL::EvTypeId`].
    fn get_ev_type_id(&self, ev: &Self::EventRepr) -> Self::EvTypeId;
    /// Cheap handle for `ob`'s object type -- see [`QueryableOCEL::EvTypeId`].
    fn get_ob_type_id(&self, ob: &Self::ObjectRepr) -> Self::ObTypeId;
    /// Resolve an event-type handle back to its name. Called once per output group/row, never
    /// once per binding.
    fn resolve_ev_type(&self, id: Self::EvTypeId) -> Cow<'_, str>;
    /// Resolve an object-type handle back to its name. See [`QueryableOCEL::resolve_ev_type`].
    fn resolve_ob_type(&self, id: Self::ObTypeId) -> Cow<'_, str>;

    /// (qualifier, object) pairs for an event's E2O relationships.
    fn get_e2o(
        &self,
        ev: &Self::EventRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::ObjectRepr)> + '_;

    /// (qualifier, event) pairs for an object's reverse E2O relationships.
    fn get_e2o_rev(
        &self,
        ob: &Self::ObjectRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::EventRepr)> + '_;

    /// (qualifier, object) pairs for an object's O2O relationships.
    fn get_o2o(
        &self,
        ob: &Self::ObjectRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::ObjectRepr)> + '_;

    /// (qualifier, object) pairs for an object's reverse O2O relationships.
    fn get_o2o_rev(
        &self,
        ob: &Self::ObjectRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::ObjectRepr)> + '_;

    /// Get all objects of the given object type.
    fn get_obs_of_type(&self, ty: &str) -> impl Iterator<Item = Self::ObjectRepr> + '_;

    /// Get all events of the given event type (i.e., activity).
    fn get_evs_of_type(&self, ty: &str) -> impl Iterator<Item = Self::EventRepr> + '_;

    /// Get all event types (activities) present in the dataset.
    fn get_ev_types(&self) -> impl Iterator<Item = Cow<'_, str>> + '_;

    /// Get all object types present in the dataset.
    fn get_ob_types(&self) -> impl Iterator<Item = Cow<'_, str>> + '_;

    /// Get the value assigned to an event attribute (by name) for an event.
    fn get_ev_attr_val(&self, ev: &Self::EventRepr, name: &str) -> Option<OCELAttributeValue>;

    /// (time, value) pairs for one object attribute name (time-versioned).
    fn get_ob_attr_vals(
        &self,
        ob: &Self::ObjectRepr,
        name: &str,
    ) -> impl Iterator<Item = (DateTime<FixedOffset>, OCELAttributeValue)> + '_;

    /// Get the event types (i.e., activities) of a batch of events
    // Default loops the single-item method. `DuckDB` overrides it with one `WHERE id IN (...)`
    // query, which is what keeps id-only reprs viable there.
    fn get_ev_types_of_batch(&self, evs: &[Self::EventRepr]) -> Vec<Cow<'_, str>> {
        evs.iter().map(|e| self.get_ev_type_of(e)).collect()
    }

    /// Run a query, materializing all result rows. In-memory default; `DuckDB` overrides
    /// with SQL pushdown.
    fn run_query(&self, query: &Query) -> Result<QueryResult, String>
    where
        Self: Sized,
    {
        evaluate(query, self)
    }

    /// Run a query, streaming each result row to `f` (bounded memory on the `DuckDB` backend).
    /// The in-memory default tries [`try_stream_rows`]'s per-seed fast path, which needs no
    /// global sort, and otherwise falls back to `run_query` plus full materialization.
    fn run_query_fold(&self, query: &Query, mut f: impl FnMut(&[Value])) -> Result<(), String>
    where
        Self: Sized,
    {
        if try_stream_rows(query, self, &mut f)? {
            return Ok(());
        }
        let r = self.run_query(query)?;
        for row in &r.rows {
            f(row);
        }
        Ok(())
    }

    /// Parallel per-seed fused fold: each seed's ordered handle-rows are folded into a
    /// thread-local accumulator `A` (`init` per thread, `reduce` to merge), so traverse, sort and
    /// fold run in one parallel pass with no cross-seed row materialization.
    ///
    /// `fold_seed(acc, cols)` receives one seed's projected output **column-major** -- `cols[i]`
    /// is a row-aligned slice of `Handle`s for the `i`-th projected expression, already sorted by
    /// `order_by`. `Ok(None)` means the shape is not covered and the caller falls back.
    fn run_query_fold_seeds<A, Init, FoldSeed, Reduce>(
        &self,
        _query: &Query,
        _init: Init,
        _fold_seed: FoldSeed,
        _reduce: Reduce,
    ) -> Result<Option<A>, String>
    where
        Self: Sized,
        A: Send,
        Init: Fn() -> A + Sync,
        FoldSeed: Fn(
                &mut A,
                &[&[Handle<Self::EventRepr, Self::ObjectRepr, Self::EvTypeId, Self::ObTypeId>]],
            ) + Sync,
        Reduce: Fn(A, A) -> A + Sync,
    {
        Ok(None)
    }
}

// Delegates `QueryableOCEL` to `LinkedOCELAccess` for backends implementing
// `LinkedOCELAccess<'a>` for every `'a`. `IDLinkedOCEL<'a>` ties it to its own `'a`, so a fresh
// `&'s self` borrow cannot be delegated this way and gets a hand-written impl instead.
macro_rules! impl_queryable_from_linked {
    // `$evty`/`$obty`: the backend's native `EvTypeId`/`ObTypeId` (e.g. `usize`). The 4 clauses
    // below are `<self-ident>, <arg-ident> => <body>`. The idents are captured rather than
    // hardcoded because macro hygiene would keep a hardcoded `self`/`ev` here distinct from the
    // identically-spelled ones in a caller-supplied `:expr`.
    ($t:ty, $evty:ty, $obty:ty,
     $es:ident, $ev:ident => $ev_id:expr,
     $os:ident, $ob:ident => $ob_id:expr,
     $ers:ident, $eid:ident => $ev_resolve:expr,
     $ors:ident, $oid:ident => $ob_resolve:expr $(,)?) => {
        impl $crate::core::event_data::object_centric::linked_ocel::QueryableOCEL for $t {
            type EventRepr = <$t as $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess<'static>>::EventRepr;
            type ObjectRepr = <$t as $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess<'static>>::ObjectRepr;
            type EvTypeId = $evty;
            type ObTypeId = $obty;

            #[inline]
            fn get_ev_type_id(&$es, $ev: &Self::EventRepr) -> Self::EvTypeId {
                $ev_id
            }

            #[inline]
            fn get_ob_type_id(&$os, $ob: &Self::ObjectRepr) -> Self::ObTypeId {
                $ob_id
            }

            fn resolve_ev_type(&$ers, $eid: Self::EvTypeId) -> ::std::borrow::Cow<'_, str> {
                $ev_resolve
            }

            fn resolve_ob_type(&$ors, $oid: Self::ObTypeId) -> ::std::borrow::Cow<'_, str> {
                $ob_resolve
            }

            fn get_all_evs(&self) -> impl Iterator<Item = Self::EventRepr> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_all_evs(self)
            }

            fn get_all_obs(&self) -> impl Iterator<Item = Self::ObjectRepr> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_all_obs(self)
            }

            fn get_ev_id(&self, ev: &Self::EventRepr) -> ::std::borrow::Cow<'_, str> {
                ::std::borrow::Cow::Borrowed(
                    $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ev_id(self, ev),
                )
            }

            fn get_ob_id(&self, ob: &Self::ObjectRepr) -> ::std::borrow::Cow<'_, str> {
                ::std::borrow::Cow::Borrowed(
                    $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ob_id(self, ob),
                )
            }

            #[inline]
            fn get_ev_type_of(&self, ev: &Self::EventRepr) -> ::std::borrow::Cow<'_, str> {
                ::std::borrow::Cow::Borrowed(
                    $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ev_type_of(self, ev),
                )
            }

            #[inline]
            fn get_ob_type_of(&self, ob: &Self::ObjectRepr) -> ::std::borrow::Cow<'_, str> {
                ::std::borrow::Cow::Borrowed(
                    $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ob_type_of(self, ob),
                )
            }

            fn get_ev_time(&self, ev: &Self::EventRepr) -> ::chrono::DateTime<::chrono::FixedOffset> {
                *$crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ev_time(self, ev)
            }

            #[inline]
            fn get_e2o(
                &self,
                ev: &Self::EventRepr,
            ) -> impl Iterator<Item = (::std::borrow::Cow<'_, str>, Self::ObjectRepr)> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_e2o(self, *ev)
                    .map(|(qual, ob)| (::std::borrow::Cow::Borrowed(qual), *ob))
            }

            fn get_ob_attr_vals(
                &self,
                ob: &Self::ObjectRepr,
                name: &str,
            ) -> impl Iterator<
                Item = (
                    ::chrono::DateTime<::chrono::FixedOffset>,
                    $crate::core::event_data::object_centric::OCELAttributeValue,
                ),
            > + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ob_attr_vals(
                    self,
                    *ob,
                    name.to_string(),
                )
                .map(|(t, v)| (*t, v.clone()))
            }

            #[inline]
            fn get_e2o_rev(
                &self,
                ob: &Self::ObjectRepr,
            ) -> impl Iterator<Item = (::std::borrow::Cow<'_, str>, Self::EventRepr)> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_e2o_rev(self, *ob)
                    .map(|(qual, ev)| (::std::borrow::Cow::Borrowed(qual), *ev))
            }

            #[inline]
            fn get_o2o(
                &self,
                ob: &Self::ObjectRepr,
            ) -> impl Iterator<Item = (::std::borrow::Cow<'_, str>, Self::ObjectRepr)> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_o2o(self, *ob)
                    .map(|(qual, ob)| (::std::borrow::Cow::Borrowed(qual), *ob))
            }

            #[inline]
            fn get_o2o_rev(
                &self,
                ob: &Self::ObjectRepr,
            ) -> impl Iterator<Item = (::std::borrow::Cow<'_, str>, Self::ObjectRepr)> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_o2o_rev(self, *ob)
                    .map(|(qual, ob)| (::std::borrow::Cow::Borrowed(qual), *ob))
            }

            fn get_obs_of_type(&self, ty: &str) -> impl Iterator<Item = Self::ObjectRepr> + '_ {
                // `ty`'s short lifetime would otherwise leak into the returned opaque type via
                // RPITIT's implicit lifetime capture. Collecting into an owned `Vec` first
                // severs that dependency (reprs are `Copy`).
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_obs_of_type(self, ty)
                    .copied()
                    .collect::<::std::vec::Vec<_>>()
                    .into_iter()
            }

            fn get_evs_of_type(&self, ty: &str) -> impl Iterator<Item = Self::EventRepr> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_evs_of_type(self, ty)
                    .copied()
                    .collect::<::std::vec::Vec<_>>()
                    .into_iter()
            }

            fn get_ev_types(&self) -> impl Iterator<Item = ::std::borrow::Cow<'_, str>> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ev_types(self)
                    .map(::std::borrow::Cow::Borrowed)
            }

            fn get_ob_types(&self) -> impl Iterator<Item = ::std::borrow::Cow<'_, str>> + '_ {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ob_types(self)
                    .map(::std::borrow::Cow::Borrowed)
            }

            fn get_ev_attr_val(
                &self,
                ev: &Self::EventRepr,
                name: &str,
            ) -> Option<$crate::core::event_data::object_centric::OCELAttributeValue> {
                $crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess::get_ev_attr_val(self, *ev, name)
                    .cloned()
            }

            // `$t`'s reprs are `Send + Sync`, so use the rayon-parallel evaluator instead of the
            // trait's sequential default.
            fn run_query(
                &self,
                query: &$crate::core::event_data::object_centric::query::Query,
            ) -> Result<$crate::core::event_data::object_centric::query::eval::QueryResult, String>
            {
                $crate::core::event_data::object_centric::query::eval::evaluate_par(query, self)
            }

            // Likewise the parallel, flush-grouped streaming fast path.
            fn run_query_fold(
                &self,
                query: &$crate::core::event_data::object_centric::query::Query,
                mut f: impl FnMut(&[$crate::core::event_data::object_centric::query::eval::Value]),
            ) -> Result<(), String> {
                if $crate::core::event_data::object_centric::query::eval::try_stream_rows_par(query, self, &mut f)? {
                    return Ok(());
                }
                let r = self.run_query(query)?;
                for row in &r.rows {
                    f(row);
                }
                Ok(())
            }
        }
    };
}

pub(crate) use impl_queryable_from_linked;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_data::object_centric::linked_ocel::IndexLinkedOCEL;
    use crate::core::event_data::object_centric::ocel_json::import_ocel_json_path;
    use crate::core::event_data::object_centric::query::*;
    use crate::test_utils::get_test_data_path;

    fn reference() -> IndexLinkedOCEL {
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap())
    }

    // Orders and their directly-related events, projected as (event id, object id).
    fn rows_query() -> Query {
        Query {
            root: Box {
                new_vars: vec![
                    VarDecl {
                        kind: VarKind::Event,
                        types: TypeConstraint::Any,
                    },
                    VarDecl {
                        kind: VarKind::Object,
                        types: TypeConstraint::OneOf(vec!["orders".to_string()]),
                    },
                ],
                filters: vec![Filter::E2O {
                    event: 0,
                    object: 1,
                    qualifier: None,
                }],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    // (event type, object type) -> count, over all E2O relationships.
    #[cfg(feature = "ocel-duckdb")]
    fn aggregate_query() -> Query {
        Query {
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
        }
    }

    // Neither query below pins row order, so `run_query` and `run_query_fold` may emit rows in
    // different orders (e.g. DuckDB's hash-aggregate plan). Sort both sides before comparing.
    fn assert_fold_matches_run_query<Q: QueryableOCEL>(ocel: &Q, query: &Query) {
        let mut via_run_query = ocel.run_query(query).unwrap().rows;
        let mut folded = Vec::new();
        ocel.run_query_fold(query, |row| folded.push(row.to_vec()))
            .unwrap();
        assert!(!via_run_query.is_empty(), "sanity: expected rows");
        via_run_query.sort();
        folded.sort();
        assert_eq!(folded, via_run_query);
    }

    #[test]
    fn run_query_fold_matches_run_query_index_rows() {
        assert_fold_matches_run_query(&reference(), &rows_query());
    }
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn run_query_fold_matches_run_query_duckdb_aggregate() {
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };

        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let out = get_test_data_path()
            .join("export")
            .join("queryable-fold-parity-aggregate.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        assert_fold_matches_run_query(&db, &aggregate_query());
    }
}
