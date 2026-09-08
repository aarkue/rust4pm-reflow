//! In-memory evaluator for the pushdown query AST: binding enumeration, correlated
//! `Box::children`, `Output::Rows`, and `Output::Aggregate`.
//!
//! A binding is a variable assignment, not a relationship-junction row: extending a variable via
//! `E2O`/`O2O` dedups the related entities by id, so an entity linked to its neighbor under
//! several qualifiers is bound exactly once.
//!
//! Bindings are folded during enumeration rather than materialized, a sink being called at each
//! complete binding. A child feeding only a `Count`-vs-range filter stops enumerating once the
//! verdict is decided (`count_early_stop`).
//!
//! A box's filters run in two stages: the ones reading no child fold are checked at the complete
//! binding (`verify_binding`), the rest as each child they read is folded (`finish_one_binding`),
//! so an `AggRange` rejection still avoids folding the remaining children.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use chrono::{DateTime, FixedOffset};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::*;
use crate::core::event_data::object_centric::linked_ocel::QueryableOCEL;
use crate::core::event_data::object_centric::OCELAttributeValue;

/// Below this many bindings/candidates, rayon's per-task overhead outweighs the parallel win, so
/// the `_par` paths fall back to sequential code.
const PAR_THRESHOLD: usize = 4096;

/// Seed candidates per rayon task in [`eval_aggregate_batched_par`], sized so each task's
/// dedup-set/group-table reuse amortizes over many seeds rather than one.
const AGG_BATCH: usize = 2048;

/// A scalar produced by evaluating an [`Expr`] against a binding.
#[derive(Debug, Clone)]
pub enum Value {
    /// Absent or missing value.
    Null,
    /// A string value.
    Str(String),
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A timestamp value.
    Time(DateTime<FixedOffset>),
    /// An ordered list of values (produced by [`Agg::Sequence`], e.g. an object's activity trace).
    List(Vec<Value>),
}

impl Value {
    /// Ordering class. `Int` and `Float` deliberately share one class so numbers compare by
    /// magnitude rather than by variant, see [`Value::as_num`].
    fn rank(&self) -> u8 {
        match self {
            Value::Null => 0,
            Value::Bool(_) => 1,
            Value::Int(_) | Value::Float(_) => 2,
            Value::Str(_) => 3,
            Value::Time(_) => 4,
            Value::List(_) => 5,
        }
    }

    /// The canonical `f64` bits of a numeric value, or `None` for non-numerics.
    ///
    /// `Int` and `Float` are ordered, compared and hashed through this one representation, so
    /// `Int(3) == Float(3.0)`, matching SQL's single `DOUBLE` wide column. As on the SQL side,
    /// integers beyond 2^53 are not exactly representable, so two differing `Int`s can compare
    /// equal here.
    fn as_num(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(canon_f64(*i as f64)),
            Value::Float(f) => Some(canon_f64(*f)),
            _ => None,
        }
    }
}

/// NaN and `-0.0` folded to single representatives so `Eq`/`Hash`/`Ord` agree.
fn canon_f64(f: f64) -> f64 {
    if f.is_nan() {
        f64::NAN
    } else if f == 0.0 {
        0.0
    } else {
        f
    }
}

/// [`canon_f64`] as raw bits, for hashing.
fn canon_float_bits(f: f64) -> u64 {
    if f.is_nan() {
        f64::NAN.to_bits()
    } else if f == 0.0 {
        0.0_f64.to_bits()
    } else {
        f.to_bits()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank()
            .cmp(&other.rank())
            .then_with(|| match (self, other) {
                (Value::Null, Value::Null) => Ordering::Equal,
                (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
                // `total_cmp` on the canonicalized value orders by magnitude, where a bit
                // comparison would sort negatives after positives.
                (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => self
                    .as_num()
                    .expect("numeric rank")
                    .total_cmp(&other.as_num().expect("numeric rank")),
                (Value::Str(a), Value::Str(b)) => a.cmp(b),
                (Value::Time(a), Value::Time(b)) => a.cmp(b),
                (Value::List(a), Value::List(b)) => a.cmp(b),
                _ => unreachable!("Value::rank guarantees matching variants here"),
            })
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rank().hash(state);
        match self {
            Value::Null => {}
            Value::Bool(b) => b.hash(state),
            // Same canonical `f64` the ordering uses, so `Eq` and `Hash` agree on
            // `Int(3)`/`Float(3.0)`.
            Value::Int(_) | Value::Float(_) => {
                canon_float_bits(self.as_num().expect("numeric rank")).hash(state)
            }
            Value::Str(s) => s.hash(state),
            Value::Time(t) => t.hash(state),
            Value::List(v) => v.hash(state),
        }
    }
}

/// A projected column as a cheap backend handle rather than a materialized [`Value`], produced by
/// [`run_query_fold_handles`](crate::core::event_data::object_centric::linked_ocel::QueryableOCEL::run_query_fold_handles).
///
/// Lets a consumer fold in the backend's integer index space and materialize a `String` only when
/// resolving the finished result, never once per row. A caller writes
/// `Handle<Q::EventRepr, Q::ObjectRepr, Q::EvTypeId, Q::ObTypeId>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Handle<E, O, ET, OT> {
    /// An event, as the backend's own event repr (`QueryableOCEL::EventRepr`).
    Ev(E),
    /// An object, as the backend's own object repr (`QueryableOCEL::ObjectRepr`).
    Ob(O),
    /// An event type, as the backend's cheap event-type handle (`QueryableOCEL::EvTypeId`).
    EvType(ET),
    /// An object type, as the backend's cheap object-type handle (`QueryableOCEL::ObTypeId`).
    ObType(OT),
    /// A timestamp, from a `Time` projection.
    Time(DateTime<FixedOffset>),
}

/// Borrowing/handle counterpart of [`Value`], used only as the `Aggregate` fold's group key.
///
/// `Expr::Id`/`Expr::Type` key on the backend's own representations (an integer index in-memory)
/// rather than on an id or type-name `String`, so a group key needs neither an allocation nor a
/// string hash per binding. The `String` a [`Value`] ultimately needs is materialized once per
/// group at [`finalize_aggregate_fold`]. `Expr::Attr`/`Expr::ChildAgg` operands are per-binding
/// temporaries and still allocate, see [`key_part_from_value`].
enum KeyPart<'ocel, Q: QueryableOCEL> {
    Null,
    Str(Cow<'ocel, str>),
    Ev(Q::EventRepr),
    Ob(Q::ObjectRepr),
    EvType(Q::EvTypeId),
    ObType(Q::ObTypeId),
    Int(i64),
    Float(f64),
    Bool(bool),
    Time(DateTime<FixedOffset>),
}

// Manual, since a derive would add a blanket `Q: Clone` bound on the type param itself rather
// than on the associated types that actually need it.
impl<Q: QueryableOCEL> Clone for KeyPart<'_, Q> {
    fn clone(&self) -> Self {
        match self {
            KeyPart::Null => KeyPart::Null,
            KeyPart::Str(s) => KeyPart::Str(s.clone()),
            KeyPart::Ev(e) => KeyPart::Ev(e.clone()),
            KeyPart::Ob(o) => KeyPart::Ob(o.clone()),
            KeyPart::EvType(t) => KeyPart::EvType(*t),
            KeyPart::ObType(t) => KeyPart::ObType(*t),
            KeyPart::Int(i) => KeyPart::Int(*i),
            KeyPart::Float(f) => KeyPart::Float(*f),
            KeyPart::Bool(b) => KeyPart::Bool(*b),
            KeyPart::Time(t) => KeyPart::Time(*t),
        }
    }
}

impl<Q: QueryableOCEL> PartialEq for KeyPart<'_, Q> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (KeyPart::Null, KeyPart::Null) => true,
            (KeyPart::Bool(a), KeyPart::Bool(b)) => a == b,
            (KeyPart::Int(a), KeyPart::Int(b)) => a == b,
            (KeyPart::Float(a), KeyPart::Float(b)) => canon_float_bits(*a) == canon_float_bits(*b),
            (KeyPart::Str(a), KeyPart::Str(b)) => a == b,
            (KeyPart::Ev(a), KeyPart::Ev(b)) => a == b,
            (KeyPart::Ob(a), KeyPart::Ob(b)) => a == b,
            (KeyPart::EvType(a), KeyPart::EvType(b)) => a == b,
            (KeyPart::ObType(a), KeyPart::ObType(b)) => a == b,
            (KeyPart::Time(a), KeyPart::Time(b)) => a == b,
            _ => false,
        }
    }
}
impl<Q: QueryableOCEL> Eq for KeyPart<'_, Q> {}

impl<Q: QueryableOCEL> Hash for KeyPart<'_, Q> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            KeyPart::Null => {}
            KeyPart::Bool(b) => b.hash(state),
            KeyPart::Int(i) => i.hash(state),
            KeyPart::Float(f) => canon_float_bits(*f).hash(state),
            KeyPart::Str(s) => s.hash(state),
            KeyPart::Ev(e) => e.hash(state),
            KeyPart::Ob(o) => o.hash(state),
            KeyPart::EvType(t) => t.hash(state),
            KeyPart::ObType(t) => t.hash(state),
            KeyPart::Time(t) => t.hash(state),
        }
    }
}

/// An already-evaluated [`Value`] as a [`KeyPart`], for `Expr::Attr`/`Expr::ChildAgg` group-key
/// parts. Those have no backend-owned representation to borrow, so `Cow::Owned` is the best
/// available.
fn key_part_from_value<'ocel, Q: QueryableOCEL>(v: Value) -> KeyPart<'ocel, Q> {
    match v {
        Value::Null => KeyPart::Null,
        Value::Bool(b) => KeyPart::Bool(b),
        Value::Int(i) => KeyPart::Int(i),
        Value::Float(f) => KeyPart::Float(f),
        Value::Str(s) => KeyPart::Str(Cow::Owned(s)),
        Value::Time(t) => KeyPart::Time(t),
        // A `List` has no key representation. `Query::validate` rejects every position that
        // could reach here, so this only guards an unvalidated query.
        Value::List(_) => KeyPart::Null,
    }
}

/// Materializes a [`KeyPart`] into a [`Value`] for one output row. The one point a group key's
/// handle is resolved back to a `String`, once per group rather than once per binding.
fn key_part_into_value<Q: QueryableOCEL>(k: KeyPart<'_, Q>, ocel: &Q) -> Value {
    match k {
        KeyPart::Null => Value::Null,
        KeyPart::Bool(b) => Value::Bool(b),
        KeyPart::Int(i) => Value::Int(i),
        KeyPart::Float(f) => Value::Float(f),
        KeyPart::Str(s) => Value::Str(s.into_owned()),
        KeyPart::Ev(e) => Value::Str(ocel.get_ev_id(&e).into_owned()),
        KeyPart::Ob(o) => Value::Str(ocel.get_ob_id(&o).into_owned()),
        KeyPart::EvType(t) => Value::Str(ocel.resolve_ev_type(t).into_owned()),
        KeyPart::ObType(t) => Value::Str(ocel.resolve_ob_type(t).into_owned()),
        KeyPart::Time(t) => Value::Time(t),
    }
}

/// `Aggregate`'s group key, built once per binding by [`build_group_key`]. Inline storage up to
/// arity 4 keeps grouping a binding allocation-free, `Spill` covers the rest with a `Vec`.
///
/// `group_by`'s length is fixed for a whole evaluation, so every key a given `HashMap` sees is
/// the same variant and [`Hash`]/[`PartialEq`] need not separate variants from each other.
enum GroupKey<'ocel, Q: QueryableOCEL> {
    K0,
    K1(KeyPart<'ocel, Q>),
    K2(KeyPart<'ocel, Q>, KeyPart<'ocel, Q>),
    K3(KeyPart<'ocel, Q>, KeyPart<'ocel, Q>, KeyPart<'ocel, Q>),
    K4(
        KeyPart<'ocel, Q>,
        KeyPart<'ocel, Q>,
        KeyPart<'ocel, Q>,
        KeyPart<'ocel, Q>,
    ),
    Spill(Vec<KeyPart<'ocel, Q>>),
}

// Manual for the same reason as `KeyPart`'s `Clone`.
impl<Q: QueryableOCEL> PartialEq for GroupKey<'_, Q> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GroupKey::K0, GroupKey::K0) => true,
            (GroupKey::K1(a0), GroupKey::K1(b0)) => a0 == b0,
            (GroupKey::K2(a0, a1), GroupKey::K2(b0, b1)) => a0 == b0 && a1 == b1,
            (GroupKey::K3(a0, a1, a2), GroupKey::K3(b0, b1, b2)) => {
                a0 == b0 && a1 == b1 && a2 == b2
            }
            (GroupKey::K4(a0, a1, a2, a3), GroupKey::K4(b0, b1, b2, b3)) => {
                a0 == b0 && a1 == b1 && a2 == b2 && a3 == b3
            }
            (GroupKey::Spill(a), GroupKey::Spill(b)) => a == b,
            _ => false,
        }
    }
}
impl<Q: QueryableOCEL> Eq for GroupKey<'_, Q> {}

impl<Q: QueryableOCEL> Hash for GroupKey<'_, Q> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            GroupKey::K0 => {}
            GroupKey::K1(a0) => a0.hash(state),
            GroupKey::K2(a0, a1) => {
                a0.hash(state);
                a1.hash(state);
            }
            GroupKey::K3(a0, a1, a2) => {
                a0.hash(state);
                a1.hash(state);
                a2.hash(state);
            }
            GroupKey::K4(a0, a1, a2, a3) => {
                a0.hash(state);
                a1.hash(state);
                a2.hash(state);
                a3.hash(state);
            }
            GroupKey::Spill(v) => {
                for k in v {
                    k.hash(state);
                }
            }
        }
    }
}

/// Builds one binding's group key, allocating only in the `Spill` arm.
fn build_group_key<'ocel, Q: QueryableOCEL>(
    group_by: &[Expr],
    vars: &Vars<Q>,
    child_scalars: &[Vec<Value>],
    ocel: &'ocel Q,
) -> GroupKey<'ocel, Q> {
    let k = |e: &Expr| eval_expr_key(e, vars, child_scalars, ocel);
    match group_by {
        [] => GroupKey::K0,
        [a] => GroupKey::K1(k(a)),
        [a, b] => GroupKey::K2(k(a), k(b)),
        [a, b, c] => GroupKey::K3(k(a), k(b), k(c)),
        [a, b, c, d] => GroupKey::K4(k(a), k(b), k(c), k(d)),
        more => GroupKey::Spill(more.iter().map(k).collect()),
    }
}

/// Materializes a [`GroupKey`] into the finished row's leading columns, once per group.
fn group_key_into_values<Q: QueryableOCEL>(k: GroupKey<'_, Q>, ocel: &Q) -> Vec<Value> {
    match k {
        GroupKey::K0 => vec![],
        GroupKey::K1(a0) => vec![key_part_into_value(a0, ocel)],
        GroupKey::K2(a0, a1) => vec![key_part_into_value(a0, ocel), key_part_into_value(a1, ocel)],
        GroupKey::K3(a0, a1, a2) => vec![
            key_part_into_value(a0, ocel),
            key_part_into_value(a1, ocel),
            key_part_into_value(a2, ocel),
        ],
        GroupKey::K4(a0, a1, a2, a3) => vec![
            key_part_into_value(a0, ocel),
            key_part_into_value(a1, ocel),
            key_part_into_value(a2, ocel),
            key_part_into_value(a3, ocel),
        ],
        GroupKey::Spill(v) => v
            .into_iter()
            .map(|k| key_part_into_value(k, ocel))
            .collect(),
    }
}

/// Tabular result of evaluating a query with `Output::Rows`.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// Column names, one per projected expression.
    pub columns: Vec<String>,
    /// Row data, one entry per binding, in the same order as `columns`.
    pub rows: Vec<Vec<Value>>,
}

/// A variable's binding: either an event or an object of the backend's representation.
enum Entity<Q: QueryableOCEL> {
    /// Bound to an event.
    Ev(Q::EventRepr),
    /// Bound to an object.
    Ob(Q::ObjectRepr),
}

impl<Q: QueryableOCEL> Clone for Entity<Q> {
    fn clone(&self) -> Self {
        match self {
            Entity::Ev(e) => Entity::Ev(e.clone()),
            Entity::Ob(o) => Entity::Ob(o.clone()),
        }
    }
}

/// Raw variable slots of a binding, indexed by global [`VarId`]. Sized to the whole query
/// (all boxes), so ancestor/own/not-yet-bound descendant vars share one flat address space.
type Vars<Q> = Vec<Option<Entity<Q>>>;

/// One step of the binding-enumeration plan for a box.
///
/// `pub(crate)` so `linked_ocel::slim_query_exec` can interpret the steps directly over
/// `SlimLinkedOCEL`'s own fields instead of through `QueryableOCEL` dispatch.
pub(crate) enum PlanStep {
    /// Enumerate all entities of the variable's declared type(s).
    Scan(VarId),
    /// Enumerate the variable's candidates via an E2O relation from an already-bound neighbor.
    ExtendE2O {
        var: VarId,
        neighbor: VarId,
        var_is_event: bool,
        qualifier: Option<String>,
    },
    /// Enumerate the variable's candidates via an O2O relation from an already-bound neighbor.
    ExtendO2O {
        var: VarId,
        neighbor: VarId,
        var_is_from: bool,
        qualifier: Option<String>,
    },
}

impl PlanStep {
    pub(crate) fn var(&self) -> VarId {
        match self {
            PlanStep::Scan(v)
            | PlanStep::ExtendE2O { var: v, .. }
            | PlanStep::ExtendO2O { var: v, .. } => *v,
        }
    }
}

/// A box's compiled enumeration plan, plus its children's, mirroring `Box`'s tree shape.
///
/// Compiled once per query rather than once per parent binding: a box's plan is invariant across
/// every parent binding that reaches it, cardinality estimates included.
pub(crate) struct BoxPlan {
    pub(crate) steps: Vec<PlanStep>,
    children: Vec<ChildPlan>,
    /// Which [`Query::emits`] entry this node fills, if any. `None` throughout an ordinary
    /// single-output query, which keeps [`NodeCounts`] inert there.
    emit_idx: Option<usize>,
}

/// A compiled child box, plus the per-agg bounds a `Count` fold may stop early at (see
/// [`compile_box`]).
struct ChildPlan {
    plan: BoxPlan,
    /// Global [`VarId`] of the child's first own variable.
    start_id: VarId,
    /// One entry per `ChildBox::aggs`: `Some((min, max))` when that fold is an [`Agg::Count`]
    /// whose only reader is a single [`Filter::AggRange`], so enumerating past the point the
    /// range is decided cannot change any answer.
    bounded: Vec<Option<(Option<f64>, Option<f64>)>>,
}

/// Every `Expr` the caller evaluates directly against the root box's own bindings, which is what
/// [`compile_box`] checks a child's folded scalar against to tell whether anything besides one
/// range check reads it.
pub(crate) fn output_consumer_exprs(output: &Output) -> Vec<&Expr> {
    match output {
        Output::Rows(r) => r
            .project
            .iter()
            .chain(r.order_by.iter().map(|(e, _)| e))
            .collect(),
        Output::Aggregate(a) => a.group_by.iter().chain(agg_expr(&a.aggregates)).collect(),
    }
}

fn agg_expr(aggs: &[Agg]) -> impl Iterator<Item = &Expr> {
    aggs.iter().flat_map(|a| match a {
        Agg::Count => Vec::new(),
        Agg::CountDistinct(e) | Agg::Min(e) | Agg::Max(e) | Agg::Sum(e) | Agg::Avg(e) => vec![e],
        Agg::Sequence { of, by } => {
            let mut v = vec![of];
            v.extend(by.iter().map(|(e, _)| e));
            v
        }
    })
}

/// How many times `(child, agg_idx)` is read by `e`. Counts inside [`Expr::Satisfies`], which
/// consumes child scalars exactly as a bare `ChildAgg` does.
fn expr_reads(e: &Expr, child: ChildRef, agg_idx: usize) -> usize {
    match e {
        Expr::ChildAgg(c, i) => usize::from(*c == child && *i == agg_idx),
        Expr::Satisfies(f) => filter_reads(f, child, agg_idx),
        _ => 0,
    }
}

/// [`expr_reads`] for a filter. A [`Filter::AggRange`] counts as a read.
fn filter_reads(f: &Filter, child: ChildRef, agg_idx: usize) -> usize {
    match f {
        Filter::AggRange {
            child: c,
            agg_idx: i,
            ..
        } => usize::from(*c == child && *i == agg_idx),
        Filter::Compare { left, right, .. } => {
            expr_reads(left, child, agg_idx) + expr_reads(right, child, agg_idx)
        }
        Filter::Not(inner) => filter_reads(inner, child, agg_idx),
        Filter::Or(fs) => fs.iter().map(|f| filter_reads(f, child, agg_idx)).sum(),
        _ => 0,
    }
}

/// The bounds `(child, agg_idx)` may stop counting at: `Some` only when the fold is an
/// [`Agg::Count`] read by exactly one [`Filter::AggRange`] sitting at the top of `box_.filters`
/// (so its verdict prunes the binding outright) and by nothing else.
fn bounded_count_range(
    box_: &Box,
    consumer_exprs: &[&Expr],
    child: ChildRef,
    agg_idx: usize,
    child_emits: bool,
) -> Option<(Option<f64>, Option<f64>)> {
    // Stopping early would truncate the very binding set an emitting node reports.
    if child_emits {
        return None;
    }
    if !matches!(box_.children[child].aggs[agg_idx], Agg::Count) {
        return None;
    }
    if consumer_exprs
        .iter()
        .any(|e| expr_reads(e, child, agg_idx) > 0)
    {
        return None;
    }
    let reads: usize = box_
        .filters
        .iter()
        .map(|f| filter_reads(f, child, agg_idx))
        .sum();
    if reads != 1 {
        return None;
    }
    box_.filters.iter().find_map(|f| match f {
        Filter::AggRange {
            child: c,
            agg_idx: i,
            min,
            max,
        } if *c == child && *i == agg_idx => Some((*min, *max)),
        _ => None,
    })
}

/// Whether the node at `path`, or anything below it, is an emitting node.
fn subtree_emits(path: &[ChildRef], emits: &[NodeEmit]) -> bool {
    emits.iter().any(|e| e.path.starts_with(path))
}

/// Compile `box_`'s plan and, recursively, every descendant box's. `bound` are the already-bound
/// ancestor vars, `consumer_exprs` what reads this box's bindings (see [`output_consumer_exprs`]),
/// and `cache` memoizes per-type cardinality lookups across the whole query.
pub(crate) fn compile_box<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    start_id: VarId,
    bound: &[bool],
    consumer_exprs: &[&Expr],
    cache: &mut HashMap<(bool, String), usize>,
) -> BoxPlan {
    compile_box_at(ocel, box_, start_id, bound, consumer_exprs, cache, &[], &[])
}

/// [`compile_box`], also resolving which [`Query::emits`] entry each node fills. `path` is the
/// node's position in the box tree, `emits` the query's emit list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_box_at<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    start_id: VarId,
    bound: &[bool],
    consumer_exprs: &[&Expr],
    cache: &mut HashMap<(bool, String), usize>,
    path: &[ChildRef],
    emits: &[NodeEmit],
) -> BoxPlan {
    let emit_idx = emits.iter().position(|e| e.path == path);
    let steps = build_plan(ocel, box_, start_id, bound, cache);
    let n = box_.new_vars.len();
    let own_bound = own_bound_of(bound, start_id, n);
    let mut next_start = start_id + n;
    let mut children = Vec::with_capacity(box_.children.len());
    for (c, child) in box_.children.iter().enumerate() {
        let mut here = path.to_vec();
        here.push(c);
        let child_emits = subtree_emits(&here, emits);
        let bounded = (0..child.aggs.len())
            .map(|i| bounded_count_range(box_, consumer_exprs, c, i, child_emits))
            .collect();
        let child_consumer: Vec<&Expr> = agg_expr(&child.aggs).collect();
        let mut child_path = path.to_vec();
        child_path.push(c);
        let start_id = next_start;
        let plan = compile_box_at(
            ocel,
            &child.box_,
            start_id,
            &own_bound,
            &child_consumer,
            cache,
            &child_path,
            emits,
        );
        next_start += subtree_var_count(&child.box_);
        children.push(ChildPlan {
            plan,
            start_id,
            bounded,
        });
    }
    BoxPlan {
        steps,
        children,
        emit_idx,
    }
}

/// Build `box_`'s bind order by repeatedly picking the next var with [`best_step`].
fn build_plan<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    start_id: VarId,
    bound: &[bool],
    cache: &mut HashMap<(bool, String), usize>,
) -> Vec<PlanStep> {
    let n = box_.new_vars.len();
    let mut bound = bound.to_vec();
    let mut plan = Vec::with_capacity(n);
    while plan.len() < n {
        let step = best_step(ocel, box_, start_id, &bound, cache);
        bound[step.var()] = true;
        plan.push(step);
    }
    plan
}

/// Pick the next var to bind: `(reachable, ascending cardinality)` order, ties broken by
/// declaration order.
///
/// Cardinality only reorders a scan choice for a var with no `E2O`/`O2O` filter linking it to a
/// still-unbound sibling. Scanning a var that has one fixes which direction the other side is
/// extended through, and candidate-set size does not estimate that per-edge cost.
fn best_step<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    start_id: VarId,
    bound: &[bool],
    cache: &mut HashMap<(bool, String), usize>,
) -> PlanStep {
    let mut best: Option<(bool, usize, PlanStep)> = None;
    for v in (start_id..start_id + box_.new_vars.len()).filter(|v| !bound[*v]) {
        let connecting = find_connecting_predicate(box_, v, bound);
        let reachable = connecting.is_some();
        let eligible = reachable || !has_relation_predicate(box_, v);
        let decl = &box_.new_vars[v - start_id];
        let card = if eligible {
            var_cardinality(ocel, decl, cache)
        } else {
            usize::MAX
        };
        let step = connecting.unwrap_or(PlanStep::Scan(v));
        let better = match &best {
            None => true,
            Some((best_reachable, best_card, _)) => {
                (reachable, std::cmp::Reverse(card))
                    > (*best_reachable, std::cmp::Reverse(*best_card))
            }
        };
        if better {
            best = Some((reachable, card, step));
        }
    }
    best.expect("build_plan only calls this while unbound own vars remain")
        .2
}

/// Estimated candidate-set size for a variable declaration, memoized in `cache` by
/// `(is_event, type_name)`, with an empty string key for `Any`.
fn var_cardinality<Q: QueryableOCEL>(
    ocel: &Q,
    decl: &VarDecl,
    cache: &mut HashMap<(bool, String), usize>,
) -> usize {
    let is_event = matches!(decl.kind, VarKind::Event);
    match &decl.types {
        TypeConstraint::OneOf(tys) => tys
            .iter()
            .map(|ty| {
                *cache.entry((is_event, ty.clone())).or_insert_with(|| {
                    if is_event {
                        ocel.get_evs_of_type(ty).count()
                    } else {
                        ocel.get_obs_of_type(ty).count()
                    }
                })
            })
            .sum(),
        TypeConstraint::Any => *cache.entry((is_event, String::new())).or_insert_with(|| {
            if is_event {
                ocel.get_all_evs().count()
            } else {
                ocel.get_all_obs().count()
            }
        }),
    }
}

/// Top-level relational filters only: an `E2O` nested inside a [`Filter::Or`] or a
/// [`Filter::Not`] does not have to hold, so lifting it into a [`PlanStep`] would drop bindings
/// the composite keeps.
fn relational_filters(box_: &Box) -> impl Iterator<Item = &Filter> {
    box_.filters.iter().filter_map(Filter::as_relation)
}

fn find_connecting_predicate(box_: &Box, v: VarId, bound: &[bool]) -> Option<PlanStep> {
    relational_filters(box_).find_map(|p| match p {
        Filter::E2O {
            event,
            object,
            qualifier,
        } => {
            if *event == v && bound[*object] {
                Some(PlanStep::ExtendE2O {
                    var: v,
                    neighbor: *object,
                    var_is_event: true,
                    qualifier: qualifier.clone(),
                })
            } else if *object == v && bound[*event] {
                Some(PlanStep::ExtendE2O {
                    var: v,
                    neighbor: *event,
                    var_is_event: false,
                    qualifier: qualifier.clone(),
                })
            } else {
                None
            }
        }
        Filter::O2O {
            from,
            to,
            qualifier,
        } => {
            if *from == v && bound[*to] {
                Some(PlanStep::ExtendO2O {
                    var: v,
                    neighbor: *to,
                    var_is_from: true,
                    qualifier: qualifier.clone(),
                })
            } else if *to == v && bound[*from] {
                Some(PlanStep::ExtendO2O {
                    var: v,
                    neighbor: *from,
                    var_is_from: false,
                    qualifier: qualifier.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    })
}

/// Whether `v` is linked by a top-level `E2O`/`O2O` filter to any other var, bound or not.
fn has_relation_predicate(box_: &Box, v: VarId) -> bool {
    relational_filters(box_).any(|p| match p {
        Filter::E2O { event, object, .. } => *event == v || *object == v,
        Filter::O2O { from, to, .. } => *from == v || *to == v,
        _ => false,
    })
}

fn qualifier_matches(want: &Option<String>, got: &str) -> bool {
    match want {
        None => true,
        Some(w) => w == got,
    }
}

fn type_ok_ev<Q: QueryableOCEL>(ocel: &Q, ev: &Q::EventRepr, tc: &TypeConstraint) -> bool {
    match tc {
        TypeConstraint::Any => true,
        TypeConstraint::OneOf(tys) => tys.iter().any(|t| t == ocel.get_ev_type_of(ev).as_ref()),
    }
}

fn type_ok_ob<Q: QueryableOCEL>(ocel: &Q, ob: &Q::ObjectRepr, tc: &TypeConstraint) -> bool {
    match tc {
        TypeConstraint::Any => true,
        TypeConstraint::OneOf(tys) => tys.iter().any(|t| t == ocel.get_ob_type_of(ob).as_ref()),
    }
}

/// A variable's candidate entities, streamed. Boxed because the four constraint/kind combinations
/// have unrelated iterator types, and streamed because [`enumerate_step`]'s `Scan` arm sits inside
/// the per-binding recursion, where materializing the set once per outer binding dominates.
// Closures (not bare `Entity::Ev`) avoid a rustc ICE on a generic enum variant ctor as an fn value.
#[allow(clippy::redundant_closure)]
fn candidates_iter<'a, Q: QueryableOCEL>(
    ocel: &'a Q,
    decl: &'a VarDecl,
) -> std::boxed::Box<dyn Iterator<Item = Entity<Q>> + 'a> {
    match &decl.kind {
        VarKind::Event => match &decl.types {
            TypeConstraint::Any => {
                std::boxed::Box::new(ocel.get_all_evs().map(|e| Entity::Ev(e)))
            }
            TypeConstraint::OneOf(tys) => std::boxed::Box::new(
                tys.iter()
                    .flat_map(|t| ocel.get_evs_of_type(t))
                    .map(|e| Entity::Ev(e)),
            ),
        },
        VarKind::Object => match &decl.types {
            TypeConstraint::Any => {
                std::boxed::Box::new(ocel.get_all_obs().map(|o| Entity::Ob(o)))
            }
            TypeConstraint::OneOf(tys) => std::boxed::Box::new(
                tys.iter()
                    .flat_map(|t| ocel.get_obs_of_type(t))
                    .map(|o| Entity::Ob(o)),
            ),
        },
    }
}

fn candidates_for_var<Q: QueryableOCEL>(ocel: &Q, decl: &VarDecl) -> Vec<Entity<Q>> {
    candidates_iter(ocel, decl).collect()
}

fn ev_of<Q: QueryableOCEL>(vars: &Vars<Q>, v: VarId) -> &Q::EventRepr {
    match &vars[v] {
        Some(Entity::Ev(e)) => e,
        _ => unreachable!("var {v} expected to be bound as event"),
    }
}

fn ob_of<Q: QueryableOCEL>(vars: &Vars<Q>, v: VarId) -> &Q::ObjectRepr {
    match &vars[v] {
        Some(Entity::Ob(o)) => o,
        _ => unreachable!("var {v} expected to be bound as object"),
    }
}

fn own_bound_of(parent_bound: &[bool], start_id: VarId, n: usize) -> Vec<bool> {
    let mut own_bound = parent_bound.to_vec();
    for bound in own_bound.iter_mut().skip(start_id).take(n) {
        *bound = true;
    }
    own_bound
}

/// How many global `VarId`s `box_`'s subtree consumes.
fn subtree_var_count(box_: &Box) -> usize {
    box_.new_vars.len()
        + box_
            .children
            .iter()
            .map(|c| subtree_var_count(&c.box_))
            .sum::<usize>()
}

// Calls `leaf` at each verified complete binding. Returns `false` when `leaf` (or a nested call)
// asked to stop, unwinding the whole candidate loop stack immediately.
//
// `binding` is the caller's own slot vector, handed on to `leaf` so a child box enumerates into
// the same allocation. Every arm restores the slots it wrote before returning.
//
// Note the box's own declarations are indexed locally while step vars are global, so a step's
// decl is `box_.new_vars[var - start_id]`.
#[allow(clippy::too_many_arguments)]
fn enumerate_step<Q>(
    ocel: &Q,
    box_: &Box,
    start_id: VarId,
    plan: &[PlanStep],
    i: usize,
    binding: &mut Vars<Q>,
    leaf: &mut dyn FnMut(&mut Vars<Q>) -> bool,
) -> bool
where
    Q: QueryableOCEL,
{
    if i == plan.len() {
        return if verify_binding(ocel, box_, binding) {
            leaf(binding)
        } else {
            true
        };
    }
    match &plan[i] {
        PlanStep::Scan(v) => {
            let v = *v;
            for cand in candidates_iter(ocel, &box_.new_vars[v - start_id]) {
                binding[v] = Some(cand);
                if !enumerate_step(ocel, box_, start_id, plan, i + 1, binding, leaf) {
                    binding[v] = None;
                    return false;
                }
            }
            binding[v] = None;
        }
        PlanStep::ExtendE2O {
            var,
            neighbor,
            var_is_event,
            qualifier,
        } => {
            let (var, neighbor, var_is_event) = (*var, *neighbor, *var_is_event);
            let decl = &box_.new_vars[var - start_id];
            if var_is_event {
                let ob = ob_of(binding, neighbor).clone();
                // A binding is a variable assignment, not a relationship-junction row: an event
                // linked to `ob` under multiple qualifiers is enumerated once. Dedup by the
                // entity's own cheap `Eq + Hash` repr rather than by its `String` id.
                let mut seen: FxHashSet<Q::EventRepr> = FxHashSet::default();
                for (q, ev) in ocel.get_e2o_rev(&ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ev(ocel, &ev, &decl.types)
                        || !seen.insert(ev.clone())
                    {
                        continue;
                    }
                    binding[var] = Some(Entity::Ev(ev));
                    if !enumerate_step(ocel, box_, start_id, plan, i + 1, binding, leaf) {
                        binding[var] = None;
                        return false;
                    }
                }
            } else {
                let ev = ev_of(binding, neighbor).clone();
                let mut seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();
                for (q, ob) in ocel.get_e2o(&ev) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &decl.types)
                        || !seen.insert(ob.clone())
                    {
                        continue;
                    }
                    binding[var] = Some(Entity::Ob(ob));
                    if !enumerate_step(ocel, box_, start_id, plan, i + 1, binding, leaf) {
                        binding[var] = None;
                        return false;
                    }
                }
            }
            binding[var] = None;
        }
        PlanStep::ExtendO2O {
            var,
            neighbor,
            var_is_from,
            qualifier,
        } => {
            let (var, neighbor, var_is_from) = (*var, *neighbor, *var_is_from);
            let decl = &box_.new_vars[var - start_id];
            let neighbor_ob = ob_of(binding, neighbor).clone();
            // Dedup by repr, not id `String` -- see the `ExtendE2O` arm's comment above.
            let mut seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();
            // Iterate each direction's relation directly (no `.collect()`): the two branches'
            // iterator types differ (opaque `impl Iterator`s), so the loop body is duplicated
            // instead of unified via an intermediate `Vec`.
            if var_is_from {
                for (q, ob) in ocel.get_o2o_rev(&neighbor_ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &decl.types)
                        || !seen.insert(ob.clone())
                    {
                        continue;
                    }
                    binding[var] = Some(Entity::Ob(ob));
                    if !enumerate_step(ocel, box_, start_id, plan, i + 1, binding, leaf) {
                        binding[var] = None;
                        return false;
                    }
                }
            } else {
                for (q, ob) in ocel.get_o2o(&neighbor_ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &decl.types)
                        || !seen.insert(ob.clone())
                    {
                        continue;
                    }
                    binding[var] = Some(Entity::Ob(ob));
                    if !enumerate_step(ocel, box_, start_id, plan, i + 1, binding, leaf) {
                        binding[var] = None;
                        return false;
                    }
                }
            }
            binding[var] = None;
        }
    }
    true
}

fn in_range<T: PartialOrd>(v: T, min: Option<T>, max: Option<T>) -> bool {
    min.is_none_or(|m| v >= m) && max.is_none_or(|m| v <= m)
}

/// Check every filter that does not read a child fold. The rest are deferred to
/// [`finish_one_binding`], which runs them as each child they need becomes available.
fn verify_binding<Q: QueryableOCEL>(ocel: &Q, box_: &Box, binding: &Vars<Q>) -> bool {
    box_.filters
        .iter()
        .filter(|f| f.max_child_ref().is_none())
        .all(|f| verify_filter(ocel, f, binding, &[]))
}

fn verify_filter<Q: QueryableOCEL>(
    ocel: &Q,
    f: &Filter,
    binding: &Vars<Q>,
    child_scalars: &[Vec<Value>],
) -> bool {
    match f {
        Filter::E2O {
            event,
            object,
            qualifier,
        } => {
            let target = ob_of(binding, *object);
            ocel.get_e2o(ev_of(binding, *event))
                .any(|(q, o)| qualifier_matches(qualifier, q.as_ref()) && &o == target)
        }
        Filter::O2O {
            from,
            to,
            qualifier,
        } => {
            let target = ob_of(binding, *to);
            ocel.get_o2o(ob_of(binding, *from))
                .any(|(q, o)| qualifier_matches(qualifier, q.as_ref()) && &o == target)
        }
        Filter::TimeBetweenEvents {
            from,
            to,
            min_seconds,
            max_seconds,
        } => {
            let secs = (ocel.get_ev_time(ev_of(binding, *to))
                - ocel.get_ev_time(ev_of(binding, *from)))
            .num_milliseconds() as f64
                / 1000.0;
            in_range(secs, *min_seconds, *max_seconds)
        }
        Filter::EventAttr { event, name, vf } => value_filter_matches(
            vf,
            ocel.get_ev_attr_val(ev_of(binding, *event), name).as_ref(),
        ),
        Filter::ObjectAttr {
            object,
            name,
            at,
            vf,
        } => {
            let ob = ob_of(binding, *object);
            match at {
                FilterAt::Always => ocel
                    .get_ob_attr_vals(ob, name)
                    .all(|(_, v)| value_filter_matches(vf, Some(&v))),
                FilterAt::Sometime => ocel
                    .get_ob_attr_vals(ob, name)
                    .any(|(_, v)| value_filter_matches(vf, Some(&v))),
                FilterAt::AtEvent(ev_var) => {
                    let cutoff = ocel.get_ev_time(ev_of(binding, *ev_var));
                    let val = ocel
                        .get_ob_attr_vals(ob, name)
                        .filter(|(t, _)| *t <= cutoff)
                        .max_by(cmp_attr_versions)
                        .map(|(_, v)| v);
                    value_filter_matches(vf, val.as_ref())
                }
            }
        }
        Filter::AggRange {
            child,
            agg_idx,
            min,
            max,
        } => child_scalars
            .get(*child)
            .and_then(|s| s.get(*agg_idx))
            .and_then(value_as_f64)
            .is_some_and(|f| in_range(f, *min, *max)),
        Filter::Compare { left, op, right } => {
            let l = eval_expr(left, binding, child_scalars, ocel);
            let r = eval_expr(right, binding, child_scalars, ocel);
            compare_values(&l, *op, &r)
        }
        Filter::Not(inner) => !verify_filter(ocel, inner, binding, child_scalars),
        Filter::Or(fs) => fs
            .iter()
            .any(|f| verify_filter(ocel, f, binding, child_scalars)),
    }
}

/// `Null` on either side is false for every operator, matching SQL rather than [`Value`]'s own
/// total order -- the two backends have to agree, and SQL cannot be made to order `Null`.
///
/// Operands of differing [`Value::rank`] are unsatisfiable for the same reason. [`Value::cmp`]
/// orders across kinds so that sorting is total, but SQL compares a string against a number by
/// casting, which gives the opposite answer for `Str("2")` against `Int(10)`. `Int`/`Float` share
/// one rank, so numbers still compare by magnitude.
fn compare_values(l: &Value, op: CmpOp, r: &Value) -> bool {
    if matches!(l, Value::Null) || matches!(r, Value::Null) || l.rank() != r.rank() {
        return false;
    }
    let ord = l.cmp(r);
    match op {
        CmpOp::Eq => ord == Ordering::Equal,
        CmpOp::Ne => ord != Ordering::Equal,
        CmpOp::Lt => ord == Ordering::Less,
        CmpOp::Le => ord != Ordering::Greater,
        CmpOp::Gt => ord == Ordering::Greater,
        CmpOp::Ge => ord != Ordering::Less,
    }
}

fn value_filter_matches(vf: &ValueFilter, val: Option<&OCELAttributeValue>) -> bool {
    let Some(val) = val else { return false };
    match (vf, val) {
        (ValueFilter::Integer { min, max }, OCELAttributeValue::Integer(i)) => {
            in_range(*i, *min, *max)
        }
        (ValueFilter::Float { min, max }, OCELAttributeValue::Float(f)) => in_range(*f, *min, *max),
        (ValueFilter::Boolean { is }, OCELAttributeValue::Boolean(b)) => b == is,
        (ValueFilter::String { is_in }, OCELAttributeValue::String(s)) => is_in.contains(s),
        (ValueFilter::Time { from, to }, OCELAttributeValue::Time(t)) => in_range(*t, *from, *to),
        _ => false,
    }
}

/// [`Value`]'s ordering class for a stored attribute value. Mirrors [`Value::rank`].
fn attr_rank(v: &OCELAttributeValue) -> u8 {
    match v {
        OCELAttributeValue::Null => 0,
        OCELAttributeValue::Boolean(_) => 1,
        OCELAttributeValue::Integer(_) | OCELAttributeValue::Float(_) => 2,
        OCELAttributeValue::String(_) => 3,
        OCELAttributeValue::Time(_) => 4,
    }
}

/// [`Value`]'s ordering on two stored attribute values, without materializing either.
fn cmp_attr_vals(a: &OCELAttributeValue, b: &OCELAttributeValue) -> Ordering {
    let num = |v: &OCELAttributeValue| match v {
        OCELAttributeValue::Integer(i) => canon_f64(*i as f64),
        OCELAttributeValue::Float(f) => canon_f64(*f),
        _ => 0.0,
    };
    attr_rank(a).cmp(&attr_rank(b)).then_with(|| match (a, b) {
        (OCELAttributeValue::Boolean(x), OCELAttributeValue::Boolean(y)) => x.cmp(y),
        (
            OCELAttributeValue::Integer(_) | OCELAttributeValue::Float(_),
            OCELAttributeValue::Integer(_) | OCELAttributeValue::Float(_),
        ) => num(a).total_cmp(&num(b)),
        (OCELAttributeValue::String(x), OCELAttributeValue::String(y)) => x.cmp(y),
        (OCELAttributeValue::Time(x), OCELAttributeValue::Time(y)) => x.cmp(y),
        _ => Ordering::Equal,
    })
}

/// Order two recorded versions of one object attribute totally: by timestamp, then by the value
/// itself. Without the second key a tie at equal timestamps resolves by iteration order in memory
/// and arbitrarily in SQL, so the two backends could pick different versions.
fn cmp_attr_versions(
    a: &(DateTime<FixedOffset>, OCELAttributeValue),
    b: &(DateTime<FixedOffset>, OCELAttributeValue),
) -> Ordering {
    a.0.cmp(&b.0).then_with(|| cmp_attr_vals(&a.1, &b.1))
}

fn conv_attr(v: &OCELAttributeValue) -> Value {
    match v {
        OCELAttributeValue::Integer(i) => Value::Int(*i),
        OCELAttributeValue::Float(f) => Value::Float(*f),
        OCELAttributeValue::Boolean(b) => Value::Bool(*b),
        OCELAttributeValue::Time(t) => Value::Time(*t),
        OCELAttributeValue::String(s) => Value::Str(s.clone()),
        OCELAttributeValue::Null => Value::Null,
    }
}

fn eval_attr<Q: QueryableOCEL>(
    var: VarId,
    name: &str,
    at: &OutAt,
    binding: &Vars<Q>,
    ocel: &Q,
) -> Value {
    match &binding[var] {
        Some(Entity::Ev(ev)) => ocel
            .get_ev_attr_val(ev, name)
            .as_ref()
            .map(conv_attr)
            .unwrap_or(Value::Null),
        Some(Entity::Ob(ob)) => {
            let picked = match at {
                OutAt::Latest => ocel.get_ob_attr_vals(ob, name).max_by(cmp_attr_versions),
                OutAt::First => ocel.get_ob_attr_vals(ob, name).min_by(cmp_attr_versions),
                OutAt::AtEvent(ev_var) => {
                    let cutoff = ocel.get_ev_time(ev_of(binding, *ev_var));
                    ocel.get_ob_attr_vals(ob, name)
                        .filter(|(t, _)| *t <= cutoff)
                        .max_by(cmp_attr_versions)
                }
            };
            picked.map(|(_, v)| conv_attr(&v)).unwrap_or(Value::Null)
        }
        None => Value::Null,
    }
}

/// Evaluate `e` against a raw binding: `vars` (this box's own + ancestor var slots) and
/// `child_scalars` (this box's own children, already folded -- see `Expr::ChildAgg`'s doc).
fn eval_expr<Q: QueryableOCEL>(
    e: &Expr,
    vars: &Vars<Q>,
    child_scalars: &[Vec<Value>],
    ocel: &Q,
) -> Value {
    match e {
        Expr::Id(v) => match &vars[*v] {
            Some(Entity::Ev(ev)) => Value::Str(ocel.get_ev_id(ev).into_owned()),
            Some(Entity::Ob(ob)) => Value::Str(ocel.get_ob_id(ob).into_owned()),
            None => Value::Null,
        },
        Expr::Type(v) => match &vars[*v] {
            Some(Entity::Ev(ev)) => Value::Str(ocel.get_ev_type_of(ev).into_owned()),
            Some(Entity::Ob(ob)) => Value::Str(ocel.get_ob_type_of(ob).into_owned()),
            None => Value::Null,
        },
        Expr::Time(v) => match &vars[*v] {
            Some(Entity::Ev(ev)) => Value::Time(ocel.get_ev_time(ev)),
            _ => Value::Null,
        },
        Expr::Attr { var, name, at } => eval_attr(*var, name, at, vars, ocel),
        // Resolved against the box that owns `vars`/`child_scalars`: the root box's children when
        // called from output-level eval, a `ChildBox`'s own children when called while folding
        // that child's own nested `Agg`.
        Expr::ChildAgg(c, i) => child_scalars
            .get(*c)
            .and_then(|s| s.get(*i))
            .cloned()
            .unwrap_or(Value::Null),
        Expr::Satisfies(f) => Value::Bool(verify_filter(ocel, f, vars, child_scalars)),
    }
}

/// [`KeyPart`] counterpart of [`eval_expr`], used only to build an `Aggregate`'s group key (see
/// [`fold_into_group`]). `Id`/`Type` are the two `Expr` variants that key on the backend's own
/// cheap representations (repr / type handle) instead of a per-binding `String` -- see
/// [`KeyPart`]'s doc.
fn eval_expr_key<'ocel, Q: QueryableOCEL>(
    e: &Expr,
    vars: &Vars<Q>,
    child_scalars: &[Vec<Value>],
    ocel: &'ocel Q,
) -> KeyPart<'ocel, Q> {
    match e {
        Expr::Id(v) => match &vars[*v] {
            Some(Entity::Ev(ev)) => KeyPart::Ev(ev.clone()),
            Some(Entity::Ob(ob)) => KeyPart::Ob(ob.clone()),
            None => KeyPart::Null,
        },
        Expr::Type(v) => match &vars[*v] {
            Some(Entity::Ev(ev)) => KeyPart::EvType(ocel.get_ev_type_id(ev)),
            Some(Entity::Ob(ob)) => KeyPart::ObType(ocel.get_ob_type_id(ob)),
            None => KeyPart::Null,
        },
        Expr::Time(v) => match &vars[*v] {
            Some(Entity::Ev(ev)) => KeyPart::Time(ocel.get_ev_time(ev)),
            _ => KeyPart::Null,
        },
        // Per-binding temporaries, not backend-owned storage -- no borrow to reuse, see
        // `key_part_from_value`'s doc.
        Expr::Attr { var, name, at } => key_part_from_value(eval_attr(*var, name, at, vars, ocel)),
        Expr::ChildAgg(..) | Expr::Satisfies(_) => {
            key_part_from_value(eval_expr(e, vars, child_scalars, ocel))
        }
    }
}

pub(crate) fn column_name(e: &Expr) -> String {
    match e {
        Expr::Id(v) => format!("v{v}_id"),
        Expr::Type(v) => format!("v{v}_type"),
        Expr::Time(v) => format!("v{v}_time"),
        Expr::Attr { var, name, .. } => format!("v{var}_{name}"),
        Expr::ChildAgg(c, i) => format!("child{c}_{i}"),
        Expr::Satisfies(_) => "satisfies".to_string(),
    }
}

pub(crate) fn agg_column_name(a: &Agg) -> String {
    match a {
        Agg::Count => "count".to_string(),
        Agg::CountDistinct(e) => format!("count_distinct_{}", column_name(e)),
        Agg::Min(e) => format!("min_{}", column_name(e)),
        Agg::Max(e) => format!("max_{}", column_name(e)),
        Agg::Sum(e) => format!("sum_{}", column_name(e)),
        Agg::Avg(e) => format!("avg_{}", column_name(e)),
        Agg::Sequence { of, .. } => format!("seq_{}", column_name(of)),
    }
}

/// Coerces a numeric [`Value`] to `f64`. Non-numeric values (incl. `Null`) yield `None` and
/// are excluded by `Sum`/`Avg` and treated as "constraint not satisfiable" by `ChildBox::filter`.
fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

/// Incremental fold state for one [`Agg`], updated one binding at a time (never materializes the
/// bindings it folds).
///
/// `Null` operands are skipped by every fold, as SQL's aggregates skip `NULL`, so one missing
/// attribute cannot poison a group's `Min`/`Max` or add a phantom distinct value. `Sum` stays
/// `Int` unless a `Float` operand is seen, and skips non-numeric operands as `Avg` does. `finish`
/// yields `Value::Null` for an empty fold except `Count`/`CountDistinct`, which yield `0`.
enum AggAcc {
    Count(i64),
    CountDistinct(HashSet<Value>),
    Min(Option<Value>),
    Max(Option<Value>),
    Sum {
        int_sum: i64,
        float_sum: f64,
        saw_float: bool,
        count: usize,
    },
    Avg {
        sum: f64,
        count: usize,
    },
    /// Collected `(order-keys, value)` pairs, finalized by sorting on the keys and then
    /// extracting the values into a `Value::List`. See [`Agg::Sequence`].
    ///
    /// `dirs` mirrors the `by` directions so the sort matches the SQL pushdown's
    /// `ARRAY_AGG(x ORDER BY k DESC)`. The collected value is the implicit last key on both
    /// sides, so equal `by` keys cannot leave the order backend-dependent.
    Sequence {
        items: Vec<(Vec<Value>, Value)>,
        dirs: Vec<Dir>,
    },
}

impl AggAcc {
    fn new(agg: &Agg) -> Self {
        match agg {
            Agg::Count => AggAcc::Count(0),
            Agg::CountDistinct(_) => AggAcc::CountDistinct(HashSet::new()),
            Agg::Min(_) => AggAcc::Min(None),
            Agg::Max(_) => AggAcc::Max(None),
            Agg::Sum(_) => AggAcc::Sum {
                int_sum: 0,
                float_sum: 0.0,
                saw_float: false,
                count: 0,
            },
            Agg::Avg(_) => AggAcc::Avg { sum: 0.0, count: 0 },
            Agg::Sequence { by, .. } => AggAcc::Sequence {
                items: Vec::new(),
                dirs: by.iter().map(|(_, d)| d.clone()).collect(),
            },
        }
    }

    fn update<Q: QueryableOCEL>(
        &mut self,
        agg: &Agg,
        vars: &Vars<Q>,
        child_scalars: &[Vec<Value>],
        ocel: &Q,
    ) {
        match (self, agg) {
            (AggAcc::Count(c), Agg::Count) => *c += 1,
            (AggAcc::CountDistinct(set), Agg::CountDistinct(e)) => {
                let v = eval_expr(e, vars, child_scalars, ocel);
                if !matches!(v, Value::Null) {
                    set.insert(v);
                }
            }
            (AggAcc::Min(cur), Agg::Min(e)) => {
                let v = eval_expr(e, vars, child_scalars, ocel);
                if !matches!(v, Value::Null) && cur.as_ref().is_none_or(|c| v < *c) {
                    *cur = Some(v);
                }
            }
            (AggAcc::Max(cur), Agg::Max(e)) => {
                let v = eval_expr(e, vars, child_scalars, ocel);
                if !matches!(v, Value::Null) && cur.as_ref().is_none_or(|c| v > *c) {
                    *cur = Some(v);
                }
            }
            (
                AggAcc::Sum {
                    int_sum,
                    float_sum,
                    saw_float,
                    count,
                },
                Agg::Sum(e),
            ) => match eval_expr(e, vars, child_scalars, ocel) {
                Value::Int(i) => {
                    *count += 1;
                    if *saw_float {
                        *float_sum += i as f64;
                    } else {
                        *int_sum += i;
                    }
                }
                Value::Float(f) => {
                    *count += 1;
                    if !*saw_float {
                        *float_sum = *int_sum as f64;
                        *saw_float = true;
                    }
                    *float_sum += f;
                }
                _ => {}
            },
            (AggAcc::Avg { sum, count }, Agg::Avg(e)) => {
                if let Some(f) = value_as_f64(&eval_expr(e, vars, child_scalars, ocel)) {
                    *sum += f;
                    *count += 1;
                }
            }
            (AggAcc::Sequence { items, .. }, Agg::Sequence { of, by }) => {
                let keys = by
                    .iter()
                    .map(|(e, _)| eval_expr(e, vars, child_scalars, ocel))
                    .collect();
                let val = eval_expr(of, vars, child_scalars, ocel);
                items.push((keys, val));
            }
            _ => unreachable!("AggAcc variant always matches the Agg it was created from"),
        }
    }

    /// Combine two partial folds of the *same* `Agg` (rayon per-thread reduce).
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (AggAcc::Count(a), AggAcc::Count(b)) => AggAcc::Count(a + b),
            (AggAcc::CountDistinct(mut a), AggAcc::CountDistinct(b)) => {
                a.extend(b);
                AggAcc::CountDistinct(a)
            }
            (AggAcc::Min(a), AggAcc::Min(b)) => AggAcc::Min(match (a, b) {
                (None, x) | (x, None) => x,
                (Some(x), Some(y)) => Some(if x <= y { x } else { y }),
            }),
            (AggAcc::Max(a), AggAcc::Max(b)) => AggAcc::Max(match (a, b) {
                (None, x) | (x, None) => x,
                (Some(x), Some(y)) => Some(if x >= y { x } else { y }),
            }),
            (
                AggAcc::Sum {
                    int_sum: ia,
                    float_sum: fa,
                    saw_float: sa,
                    count: ca,
                },
                AggAcc::Sum {
                    int_sum: ib,
                    float_sum: fb,
                    saw_float: sb,
                    count: cb,
                },
            ) => {
                let count = ca + cb;
                if sa || sb {
                    let fa = if sa { fa } else { ia as f64 };
                    let fb = if sb { fb } else { ib as f64 };
                    AggAcc::Sum {
                        int_sum: 0,
                        float_sum: fa + fb,
                        saw_float: true,
                        count,
                    }
                } else {
                    AggAcc::Sum {
                        int_sum: ia + ib,
                        float_sum: 0.0,
                        saw_float: false,
                        count,
                    }
                }
            }
            (AggAcc::Avg { sum: sa, count: ca }, AggAcc::Avg { sum: sb, count: cb }) => {
                AggAcc::Avg {
                    sum: sa + sb,
                    count: ca + cb,
                }
            }
            (AggAcc::Sequence { items: mut a, dirs }, AggAcc::Sequence { items: b, .. }) => {
                // Concatenate; the final stable sort on the order-keys (in `finish`) fixes order
                // regardless of which partial fold each element came from.
                a.extend(b);
                AggAcc::Sequence { items: a, dirs }
            }
            _ => unreachable!("merge only combines two accs of the same Agg"),
        }
    }

    fn finish(self) -> Value {
        match self {
            AggAcc::Count(c) => Value::Int(c),
            AggAcc::CountDistinct(set) => Value::Int(set.len() as i64),
            AggAcc::Min(v) => v.unwrap_or(Value::Null),
            AggAcc::Max(v) => v.unwrap_or(Value::Null),
            AggAcc::Sum {
                int_sum,
                float_sum,
                saw_float,
                count,
            } => match (count, saw_float) {
                (0, _) => Value::Null,
                (_, true) => Value::Float(float_sum),
                (_, false) => Value::Int(int_sum),
            },
            AggAcc::Avg { sum, count } => {
                if count == 0 {
                    Value::Null
                } else {
                    Value::Float(sum / count as f64)
                }
            }
            AggAcc::Sequence { mut items, dirs } => {
                items.sort_by(|(ka, va), (kb, vb)| cmp_keys(ka, kb, &dirs).then_with(|| va.cmp(vb)));
                Value::List(items.into_iter().map(|(_, v)| v).collect())
            }
        }
    }
}

/// A sink called once per surviving binding, with that binding's child folds.
type BindingSink<'a, Q> = &'a mut dyn FnMut(&Vars<Q>, &[Vec<Value>]) -> bool;

/// One parallel chunk's contribution to an `Output::Rows` evaluation.
type RowsChunk<Q> = (Vec<(Vec<Value>, Vec<Value>)>, NodeCounts, Vec<Vars<Q>>);

/// Per-node binding counts for a multi-output evaluation, with the rollback the cascade needs.
///
/// A parent binding's survival is only known once its children are folded -- the deferred
/// `AggRange` filters need the fold, and the fold *is* the child's enumeration -- so a child's
/// count is incremented speculatively and undone if the parent is then rejected.
///
/// Inert unless the query has [`Query::emits`]: `counts` is empty, every `BoxPlan::emit_idx` is
/// `None`, and both `mark` and `rollback_to` return immediately.
#[derive(Debug, Default, Clone)]
struct NodeCounts {
    counts: Vec<u64>,
    /// One `counts.len()`-sized snapshot per open recursion frame, appended and truncated rather
    /// than allocated per binding.
    marks: Vec<u64>,
}

impl NodeCounts {
    fn new(n: usize) -> Self {
        Self {
            counts: vec![0; n],
            marks: Vec::new(),
        }
    }

    fn active(&self) -> bool {
        !self.counts.is_empty()
    }

    /// Snapshot the counters, returning the token [`Self::rollback_to`] takes.
    fn mark(&mut self) -> usize {
        let base = self.marks.len();
        if self.active() {
            self.marks.extend_from_slice(&self.counts);
        }
        base
    }

    /// Undo every increment made since `base`, then drop the snapshot.
    fn rollback_to(&mut self, base: usize) {
        if self.active() {
            let n = self.counts.len();
            self.counts.copy_from_slice(&self.marks[base..base + n]);
            self.marks.truncate(base);
        }
    }

    /// Keep the increments made since `base`, dropping only the snapshot.
    fn commit_to(&mut self, base: usize) {
        self.marks.truncate(base);
    }

    /// Count one binding of node `idx`. A no-op on an inert counter, which is what a replay
    /// ([`descend_to_node`]) passes: it re-reads bindings that were already counted.
    fn hit(&mut self, idx: usize) {
        if let Some(c) = self.counts.get_mut(idx) {
            *c += 1;
        }
    }

    /// Elementwise sum, for merging the per-task counters of a parallel run.
    fn merge(mut self, other: &Self) -> Self {
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
        self
    }
}

/// Given a running `Count` and a range, returns the final keep/drop verdict as soon as it is
/// already decided regardless of how much higher the count could still climb (`max` exceeded ->
/// reject forever; no `max` and `min` already met -> accept forever; neither bound -> always
/// accept). `None` while the verdict still depends on further counting. Bounds enumeration to
/// `max + 1` items in the common "at least/most N children" case.
fn count_early_stop(count: usize, min: Option<f64>, max: Option<f64>) -> Option<bool> {
    if let Some(m) = max {
        if count as f64 > m {
            return Some(false);
        }
    } else if let Some(mn) = min {
        if count as f64 >= mn {
            return Some(true);
        }
    } else {
        return Some(true);
    }
    None
}

/// Fold one correlated child box to one scalar per entry of its `aggs`, given the parent
/// binding's `vars`. Streams the child's own bindings through an [`AggAcc`] each (no
/// `Vec<Binding>` materialized).
///
/// A fold `ChildPlan::bounded` marks stops counting once [`count_early_stop`] decides its range,
/// but only when *every* fold of this child is so marked -- one unbounded fold still needs the
/// complete enumeration.
fn fold_child_scalars<Q: QueryableOCEL>(
    ocel: &Q,
    child: &ChildBox,
    child_plan: &ChildPlan,
    parent_vars: &mut Vars<Q>,
    counts: &mut NodeCounts,
) -> Vec<Value> {
    if child.aggs.is_empty() {
        return Vec::new();
    }
    let all_bounded = child_plan.bounded.iter().all(std::option::Option::is_some);

    let mut accs: Vec<AggAcc> = child.aggs.iter().map(AggAcc::new).collect();
    let decided = |accs: &[AggAcc]| {
        accs.iter()
            .zip(&child_plan.bounded)
            .all(|(acc, b)| match (acc, b) {
                (AggAcc::Count(c), Some((min, max))) => {
                    count_early_stop(*c as usize, *min, *max).is_some()
                }
                _ => false,
            })
    };

    if !(all_bounded && decided(&accs)) {
        enumerate_box_fold(
            ocel,
            &child.box_,
            &child_plan.plan,
            child_plan.start_id,
            parent_vars,
            counts,
            &mut |vars, nested| {
                for (acc, agg) in accs.iter_mut().zip(&child.aggs) {
                    acc.update(agg, vars, nested, ocel);
                }
                !(all_bounded && decided(&accs))
            },
        );
    }

    accs.into_iter().map(AggAcc::finish).collect()
}

/// Finish one leaf whose child-free filters already passed: fold each child, run the deferred
/// filters as soon as the child they read is available, and call `sink` with the complete
/// `(vars, child_scalars)` binding. Returns `sink`'s continue/stop signal, or `true` (continue --
/// this candidate simply is not part of the result) when a deferred filter rejected the binding.
fn finish_one_binding<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    vars: &mut Vars<Q>,
    counts: &mut NodeCounts,
    sink: BindingSink<'_, Q>,
) -> bool {
    let mark = counts.mark();
    let mut child_scalars: Vec<Vec<Value>> = Vec::with_capacity(box_.children.len());
    for (c, (child, child_plan)) in box_.children.iter().zip(&plan.children).enumerate() {
        child_scalars.push(fold_child_scalars(ocel, child, child_plan, vars, counts));
        // Every deferred filter that reads no child past this one can be decided now, which is
        // what keeps an `AggRange` rejection from folding the remaining children.
        let rejected = box_
            .filters
            .iter()
            .filter(|f| f.max_child_ref() == Some(c))
            .any(|f| !verify_filter(ocel, f, vars, &child_scalars));
        if rejected {
            // This binding is gone, and so is everything its subtree contributed.
            counts.rollback_to(mark);
            return true;
        }
    }
    if let Some(i) = plan.emit_idx {
        counts.hit(i);
    }
    counts.commit_to(mark);
    sink(vars, &child_scalars)
}

/// Enumerate every binding of `box_` (its own vars, given `parent_vars` for already-bound
/// ancestor vars), folding each complete binding (after child-folding/filtering) into `sink`
/// instead of collecting it. Returns `sink`'s continue/stop signal.
///
/// `parent_vars` is written in place and restored on return, so a whole box tree enumerates
/// through the one slot vector the outermost caller allocated.
fn enumerate_box_fold<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    start_id: VarId,
    parent_vars: &mut Vars<Q>,
    counts: &mut NodeCounts,
    sink: BindingSink<'_, Q>,
) -> bool {
    enumerate_step(
        ocel,
        box_,
        start_id,
        &plan.steps,
        0,
        parent_vars,
        &mut |vars| finish_one_binding(ocel, box_, plan, vars, counts, sink),
    )
}

/// If `plan`'s first step is an unconstrained [`PlanStep::Scan`] with at least [`PAR_THRESHOLD`]
/// candidates, return its seed var and candidates so the root's seed scan can be parallelized
/// (each candidate's whole subtree -- extension + child folding -- as one rayon task). Otherwise
/// `None` (root box doesn't start with a scan, or too few candidates for rayon to be worth it) --
/// callers fall back to the sequential path.
fn root_seed_candidates<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
) -> Option<(VarId, Vec<Entity<Q>>)> {
    let seed_var = match plan.steps.first() {
        Some(PlanStep::Scan(v)) => *v,
        _ => return None,
    };
    let cands = candidates_for_var(ocel, &box_.new_vars[seed_var]);
    if cands.len() < PAR_THRESHOLD {
        return None;
    }
    Some((seed_var, cands))
}

/// Compare two order-key tuples under per-key directions. Keys beyond `dirs` (which cannot
/// occur for a validated query) fall back to ascending.
fn cmp_keys(a: &[Value], b: &[Value], dirs: &[Dir]) -> Ordering {
    for (i, ka) in a.iter().enumerate() {
        let Some(kb) = b.get(i) else { break };
        let ord = ka.cmp(kb);
        let ord = if matches!(dirs.get(i), Some(Dir::Desc)) {
            ord.reverse()
        } else {
            ord
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn cmp_scored(
    order_by: &[(Expr, Dir)],
    a: &(Vec<Value>, Vec<Value>),
    b: &(Vec<Value>, Vec<Value>),
) -> Ordering {
    for (i, (_, dir)) in order_by.iter().enumerate() {
        let ord = a.0[i].cmp(&b.0[i]);
        let ord = if matches!(dir, Dir::Desc) {
            ord.reverse()
        } else {
            ord
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn finalize_rows(spec: &RowsSpec, scored: Vec<(Vec<Value>, Vec<Value>)>) -> QueryResult {
    let mut rows: Vec<Vec<Value>> = scored.into_iter().map(|(_, row)| row).collect();
    if let Some(n) = spec.limit {
        // Enumeration order is plan-dependent and has no SQL counterpart, so an unordered page is
        // taken from the rows' own total order, which the pushdown emits as an explicit `ORDER BY`.
        if spec.order_by.is_empty() {
            rows.sort();
        }
        rows.truncate(n);
    }
    let columns = spec.project.iter().map(column_name).collect();
    QueryResult { columns, rows }
}

fn score_leaf<Q: QueryableOCEL>(
    spec: &RowsSpec,
    vars: &Vars<Q>,
    child_scalars: &[Vec<Value>],
    ocel: &Q,
) -> (Vec<Value>, Vec<Value>) {
    let row = spec
        .project
        .iter()
        .map(|e| eval_expr(e, vars, child_scalars, ocel))
        .collect();
    let key = spec
        .order_by
        .iter()
        .map(|(e, _)| eval_expr(e, vars, child_scalars, ocel))
        .collect();
    (key, row)
}

fn eval_rows_fold<Q: QueryableOCEL>(
    spec: &RowsSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    total_vars: usize,
    counts: &mut NodeCounts,
) -> QueryResult {
    let mut scored: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
    let mut parent_vars = vec![None; total_vars];
    enumerate_box_fold(ocel, box_, plan, 0, &mut parent_vars, counts, &mut |vars, cs| {
        scored.push(score_leaf(spec, vars, cs, ocel));
        true
    });
    scored.sort_by(|a, b| cmp_scored(&spec.order_by, a, b));
    finalize_rows(spec, scored)
}

/// Parallel counterpart of [`eval_rows_fold`]: the root's seed-variable scan runs via
/// `par_iter`/fold+reduce (each candidate's subtree -- extension + child folding + row scoring --
/// as one rayon task), then `par_sort_by` the order-by key. Falls back to the sequential path
/// below [`PAR_THRESHOLD`] candidates or when the plan doesn't start with a scan.
fn eval_rows_par_fold<Q>(
    spec: &RowsSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    total_vars: usize,
    counts: &mut NodeCounts,
    seeds: &mut Vec<Vars<Q>>,
) -> QueryResult
where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
{
    let n_nodes = counts.counts.len();
    // Only a multi-output run needs the root bindings kept; collected before `finalize_rows`
    // applies `limit`, so paging the answer never changes what a node replay sees.
    let want_seeds = n_nodes > 0;
    let mut scored: Vec<(Vec<Value>, Vec<Value>)> = match root_seed_candidates(ocel, box_, plan) {
        Some((seed_var, cands)) => {
            // Chunked and order-preserving: each chunk folds into its own counters and row
            // buffer, and `collect` keeps chunk order, so a multi-output run's rows and counts do
            // not depend on how rayon scheduled the chunks.
            let parts: Vec<RowsChunk<Q>> = cands
                .par_chunks(AGG_BATCH)
                .map(|chunk| {
                    let mut acc = Vec::new();
                    let mut local = NodeCounts::new(n_nodes);
                    let mut local_seeds: Vec<Vars<Q>> = Vec::new();
                    let mut cur: Vars<Q> = vec![None; total_vars];
                    for cand in chunk {
                        cur[seed_var] = Some(cand.clone());
                        enumerate_step(ocel, box_, 0, &plan.steps, 1, &mut cur, &mut |vars| {
                            finish_one_binding(
                                ocel,
                                box_,
                                plan,
                                vars,
                                &mut local,
                                &mut |v, cs| {
                                    if want_seeds {
                                        local_seeds.push(v.clone());
                                    }
                                    acc.push(score_leaf(spec, v, cs, ocel));
                                    true
                                },
                            )
                        });
                    }
                    (acc, local, local_seeds)
                })
                .collect();
            let mut rows = Vec::new();
            for (mut part, local, mut ls) in parts {
                rows.append(&mut part);
                seeds.append(&mut ls);
                *counts = std::mem::take(counts).merge(&local);
            }
            rows
        }
        None => {
            let mut acc = Vec::new();
            let mut parent_vars = vec![None; total_vars];
            enumerate_box_fold(ocel, box_, plan, 0, &mut parent_vars, counts, &mut |vars, cs| {
                if want_seeds {
                    seeds.push(vars.clone());
                }
                acc.push(score_leaf(spec, vars, cs, ocel));
                true
            });
            acc
        }
    };

    if scored.len() >= PAR_THRESHOLD {
        scored.par_sort_by(|a, b| cmp_scored(&spec.order_by, a, b));
    } else {
        scored.sort_by(|a, b| cmp_scored(&spec.order_by, a, b));
    }
    finalize_rows(spec, scored)
}

fn new_group_accs(spec: &AggSpec) -> Vec<AggAcc> {
    spec.aggregates.iter().map(AggAcc::new).collect()
}

fn fold_into_group<'ocel, Q: QueryableOCEL>(
    spec: &AggSpec,
    groups: &mut FxHashMap<GroupKey<'ocel, Q>, Vec<AggAcc>>,
    vars: &Vars<Q>,
    child_scalars: &[Vec<Value>],
    ocel: &'ocel Q,
) {
    // No heap allocation for the (near-universal) `group_by.len() <= 4` case -- see `GroupKey`'s
    // doc. Borrows straight off `ocel`'s own storage for `Id`/`Type` group-by exprs (the common
    // case, e.g. `type_counts`'s `[Type(e), Type(o)]`) -- no `String` allocated per binding either,
    // just per group at `finalize_aggregate_fold` (see `KeyPart`'s doc).
    let key = build_group_key(&spec.group_by, vars, child_scalars, ocel);
    let accs = groups.entry(key).or_insert_with(|| new_group_accs(spec));
    for (acc, agg) in accs.iter_mut().zip(&spec.aggregates) {
        acc.update(agg, vars, child_scalars, ocel);
    }
}

fn merge_group_acc_maps<'ocel, Q: QueryableOCEL>(
    mut a: FxHashMap<GroupKey<'ocel, Q>, Vec<AggAcc>>,
    b: FxHashMap<GroupKey<'ocel, Q>, Vec<AggAcc>>,
) -> FxHashMap<GroupKey<'ocel, Q>, Vec<AggAcc>> {
    for (k, v) in b {
        match a.remove(&k) {
            None => {
                a.insert(k, v);
            }
            Some(existing) => {
                let merged: Vec<AggAcc> = existing
                    .into_iter()
                    .zip(v)
                    .map(|(x, y)| x.merge(y))
                    .collect();
                a.insert(k, merged);
            }
        }
    }
    a
}

fn finalize_aggregate_fold<Q: QueryableOCEL>(
    spec: &AggSpec,
    mut groups: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>>,
    ocel: &Q,
) -> QueryResult {
    // A global aggregate is one group whether or not anything bound, as SQL's `GROUP BY`-less
    // aggregate is one row over zero rows.
    if spec.group_by.is_empty() && groups.is_empty() {
        groups.insert(GroupKey::K0, new_group_accs(spec));
    }
    // The only point a group key's `String`/name is actually materialized -- once per group (see
    // `key_part_into_value`'s doc).
    let mut rows: Vec<Vec<Value>> = groups
        .into_iter()
        .map(|(key, accs)| {
            let mut row: Vec<Value> = group_key_into_values(key, ocel);
            row.extend(accs.into_iter().map(AggAcc::finish));
            row
        })
        .collect();

    if !spec.having.is_empty() {
        let offset = spec.group_by.len();
        rows.retain(|row| {
            spec.having.iter().all(|(idx, min, max)| {
                value_as_f64(&row[offset + idx]).is_some_and(|f| in_range(f, *min, *max))
            })
        });
    }

    if spec.order_by.is_empty() {
        // No explicit order: sort by group key so results are still deterministic.
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
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
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

/// True when `box_` carries exactly one filter and it is a top-level relational one -- the shape
/// both fast paths below need, since the plan step they run *is* that filter and nothing else is
/// left to check per binding.
pub(crate) fn single_relational_filter(box_: &Box) -> bool {
    box_.filters.len() == 1 && box_.filters[0].as_relation().is_some()
}

/// True when `box_`/`plan` match [`eval_aggregate_batched`]'s supported shape: a childless box
/// whose only filter is one relational filter the plan compiled into a connecting
/// `ExtendE2O`/`ExtendO2O` step -- "scan one var, extend the other via a single relationship hop"
/// (e.g. `type_counts`'s `bind e:Event Any, o:Object Any, E2O{e,o}`). Anything broader falls back
/// to [`eval_aggregate_fold`]/[`eval_aggregate_par_fold`]'s row-at-a-time path.
pub(crate) fn batched_aggregate_eligible(box_: &Box, plan: &BoxPlan) -> bool {
    box_.children.is_empty()
        && single_relational_filter(box_)
        && plan.steps.len() == 2
        && matches!(plan.steps[0], PlanStep::Scan(_))
        && matches!(
            plan.steps[1],
            PlanStep::ExtendE2O { .. } | PlanStep::ExtendO2O { .. }
        )
}

/// Extends one already-bound seed (`vars[step.var()]`'s neighbor) via `step` and folds every
/// distinct resulting binding into `groups` -- the inner loop [`eval_aggregate_batched`]/
/// [`eval_aggregate_batched_par`] call once per seed. `ev_seen`/`ob_seen` are the caller's
/// *reused* dedup sets (cleared here, not reallocated): the row-at-a-time path allocates a fresh
/// `FxHashSet` per seed candidate, which dominates on data with many seeds and few relationships
/// each. Semantics are otherwise identical to `enumerate_step`'s `ExtendE2O`/`ExtendO2O` arms.
#[allow(clippy::too_many_arguments)]
fn batched_extend_fold<'ocel, Q: QueryableOCEL>(
    ocel: &'ocel Q,
    box_: &Box,
    step: &PlanStep,
    vars: &mut Vars<Q>,
    spec: &AggSpec,
    groups: &mut FxHashMap<GroupKey<'ocel, Q>, Vec<AggAcc>>,
    ev_seen: &mut FxHashSet<Q::EventRepr>,
    ob_seen: &mut FxHashSet<Q::ObjectRepr>,
) {
    match step {
        PlanStep::ExtendE2O {
            var,
            neighbor,
            var_is_event,
            qualifier,
        } => {
            let ext_decl = &box_.new_vars[*var];
            if *var_is_event {
                let ob = ob_of(vars, *neighbor).clone();
                ev_seen.clear();
                for (q, ev) in ocel.get_e2o_rev(&ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ev(ocel, &ev, &ext_decl.types)
                        || !ev_seen.insert(ev.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ev(ev));
                    fold_into_group(spec, groups, vars, &[], ocel);
                }
            } else {
                let ev = ev_of(vars, *neighbor).clone();
                ob_seen.clear();
                for (q, ob) in ocel.get_e2o(&ev) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &ext_decl.types)
                        || !ob_seen.insert(ob.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ob(ob));
                    fold_into_group(spec, groups, vars, &[], ocel);
                }
            }
            vars[*var] = None;
        }
        PlanStep::ExtendO2O {
            var,
            neighbor,
            var_is_from,
            qualifier,
        } => {
            let ext_decl = &box_.new_vars[*var];
            let neighbor_ob = ob_of(vars, *neighbor).clone();
            ob_seen.clear();
            if *var_is_from {
                for (q, ob) in ocel.get_o2o_rev(&neighbor_ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &ext_decl.types)
                        || !ob_seen.insert(ob.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ob(ob));
                    fold_into_group(spec, groups, vars, &[], ocel);
                }
            } else {
                for (q, ob) in ocel.get_o2o(&neighbor_ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &ext_decl.types)
                        || !ob_seen.insert(ob.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ob(ob));
                    fold_into_group(spec, groups, vars, &[], ocel);
                }
            }
            vars[*var] = None;
        }
        PlanStep::Scan(_) => unreachable!("batched_aggregate_eligible requires steps[1] to extend"),
    }
}

/// Batched fast path for [`Output::Aggregate`] over [`batched_aggregate_eligible`]'s shape: scan
/// the seed var once, and for each seed extend+fold via the reused-dedup-set
/// [`batched_extend_fold`] instead of `enumerate_step`'s recursive `&mut dyn FnMut` dispatch
/// chain. Results are identical to [`eval_aggregate_fold`].
fn eval_aggregate_batched<Q: QueryableOCEL>(
    spec: &AggSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    total_vars: usize,
) -> QueryResult {
    let seed_var = plan.steps[0].var();
    let seed_decl = &box_.new_vars[seed_var];
    let mut groups: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> = FxHashMap::default();
    let mut vars: Vars<Q> = vec![None; total_vars];
    let mut ev_seen: FxHashSet<Q::EventRepr> = FxHashSet::default();
    let mut ob_seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();

    for seed in candidates_for_var(ocel, seed_decl) {
        vars[seed_var] = Some(seed);
        batched_extend_fold(
            ocel,
            box_,
            &plan.steps[1],
            &mut vars,
            spec,
            &mut groups,
            &mut ev_seen,
            &mut ob_seen,
        );
    }
    finalize_aggregate_fold(spec, groups, ocel)
}

/// Parallel counterpart of [`eval_aggregate_batched`]: seed candidates split into [`AGG_BATCH`]-
/// sized chunks distributed over rayon, each chunk folding into its own thread-local group table
/// (and its own reused dedup sets, cleared once per seed but allocated once per chunk-task) before
/// `reduce`-merging -- coarser, more uniform task granularity than [`eval_aggregate_par_fold`]'s
/// one-task-per-candidate split. Falls back to the sequential batched path below
/// [`PAR_THRESHOLD`] seed candidates (rayon overhead not worth it there).
fn eval_aggregate_batched_par<Q>(
    spec: &AggSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    total_vars: usize,
) -> QueryResult
where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
    Q::EvTypeId: Send + Sync,
    Q::ObTypeId: Send + Sync,
{
    let seed_var = plan.steps[0].var();
    let seed_decl = &box_.new_vars[seed_var];
    let seeds: Vec<Entity<Q>> = candidates_for_var(ocel, seed_decl);

    let groups: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> = if seeds.len() < PAR_THRESHOLD {
        let mut groups = FxHashMap::default();
        let mut vars: Vars<Q> = vec![None; total_vars];
        let mut ev_seen: FxHashSet<Q::EventRepr> = FxHashSet::default();
        let mut ob_seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();
        for seed in seeds {
            vars[seed_var] = Some(seed);
            batched_extend_fold(
                ocel,
                box_,
                &plan.steps[1],
                &mut vars,
                spec,
                &mut groups,
                &mut ev_seen,
                &mut ob_seen,
            );
        }
        groups
    } else {
        seeds
            .par_chunks(AGG_BATCH)
            .fold(FxHashMap::default, |mut acc, chunk| {
                let mut vars: Vars<Q> = vec![None; total_vars];
                let mut ev_seen: FxHashSet<Q::EventRepr> = FxHashSet::default();
                let mut ob_seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();
                for seed in chunk {
                    vars[seed_var] = Some(seed.clone());
                    batched_extend_fold(
                        ocel,
                        box_,
                        &plan.steps[1],
                        &mut vars,
                        spec,
                        &mut acc,
                        &mut ev_seen,
                        &mut ob_seen,
                    );
                }
                acc
            })
            .reduce(FxHashMap::default, merge_group_acc_maps)
    };
    finalize_aggregate_fold(spec, groups, ocel)
}

fn eval_aggregate_fold<Q: QueryableOCEL>(
    spec: &AggSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    total_vars: usize,
    counts: &mut NodeCounts,
) -> QueryResult {
    // The batched path skips `finish_one_binding`, so it cannot count nodes -- but it also
    // requires a childless box, which only an emitting root could be.
    if !counts.active() && batched_aggregate_eligible(box_, plan) {
        return eval_aggregate_batched(spec, ocel, box_, plan, total_vars);
    }
    let mut groups: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> = FxHashMap::default();
    let mut parent_vars = vec![None; total_vars];
    enumerate_box_fold(ocel, box_, plan, 0, &mut parent_vars, counts, &mut |vars, cs| {
        fold_into_group(spec, &mut groups, vars, cs, ocel);
        true
    });
    finalize_aggregate_fold(spec, groups, ocel)
}

/// Parallel counterpart of [`eval_aggregate_fold`]: per-thread `HashMap` fold over the root's
/// seed-variable scan, merged via `reduce` (mirrors `oc_statistics::locel_event_object_type_counts`'s
/// group-by pattern). Aggregation is order-independent, so this always yields the same groups as
/// the sequential fold. Falls back to the sequential path below [`PAR_THRESHOLD`] candidates or
/// when the plan doesn't start with a scan.
fn eval_aggregate_par_fold<Q>(
    spec: &AggSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    total_vars: usize,
    counts: &mut NodeCounts,
    seeds: &mut Vec<Vars<Q>>,
) -> QueryResult
where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
    Q::EvTypeId: Send + Sync,
    Q::ObTypeId: Send + Sync,
{
    // See [`eval_aggregate_fold`] for why the batched path is skipped when counting.
    if !counts.active() && batched_aggregate_eligible(box_, plan) {
        return eval_aggregate_batched_par(spec, ocel, box_, plan, total_vars);
    }
    let n_nodes = counts.counts.len();
    let want_seeds = n_nodes > 0;
    let groups: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> =
        match root_seed_candidates(ocel, box_, plan) {
            Some((seed_var, cands)) => {
                type AggChunk<'a, Q> = (
                    FxHashMap<GroupKey<'a, Q>, Vec<AggAcc>>,
                    NodeCounts,
                    Vec<Vars<Q>>,
                );
                let parts: Vec<AggChunk<'_, Q>> = cands
                    .par_chunks(AGG_BATCH)
                    .map(|chunk| {
                        let mut acc: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> = FxHashMap::default();
                        let mut local = NodeCounts::new(n_nodes);
                        let mut local_seeds: Vec<Vars<Q>> = Vec::new();
                        let mut cur: Vars<Q> = vec![None; total_vars];
                        for cand in chunk {
                            cur[seed_var] = Some(cand.clone());
                            enumerate_step(ocel, box_, 0, &plan.steps, 1, &mut cur, &mut |vars| {
                                finish_one_binding(
                                    ocel,
                                    box_,
                                    plan,
                                    vars,
                                    &mut local,
                                    &mut |v, cs| {
                                        if want_seeds {
                                            local_seeds.push(v.clone());
                                        }
                                        fold_into_group(spec, &mut acc, v, cs, ocel);
                                        true
                                    },
                                )
                            });
                        }
                        (acc, local, local_seeds)
                    })
                    .collect();
                let mut merged: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> = FxHashMap::default();
                for (part, local, mut ls) in parts {
                    merged = merge_group_acc_maps(merged, part);
                    seeds.append(&mut ls);
                    *counts = std::mem::take(counts).merge(&local);
                }
                merged
            }
            None => {
                let mut acc: FxHashMap<GroupKey<'_, Q>, Vec<AggAcc>> = FxHashMap::default();
                let mut parent_vars = vec![None; total_vars];
                enumerate_box_fold(ocel, box_, plan, 0, &mut parent_vars, counts, &mut |vars, cs| {
                    if want_seeds {
                        seeds.push(vars.clone());
                    }
                    fold_into_group(spec, &mut acc, vars, cs, ocel);
                    true
                });
                acc
            }
        };
    finalize_aggregate_fold(spec, groups, ocel)
}

fn entity_id<'ocel, Q: QueryableOCEL>(ocel: &'ocel Q, e: &Entity<Q>) -> Cow<'ocel, str> {
    match e {
        Entity::Ev(ev) => ocel.get_ev_id(ev),
        Entity::Ob(ob) => ocel.get_ob_id(ob),
    }
}

/// True when `box_`/`plan`/`spec` match the streaming-per-seed shape: a childless root box with
/// one relational filter, whose plan is exactly `[Scan(seed),
/// Extend{var: child, neighbor: seed, ..}]`, with `order_by` exactly `[(Id(seed), Asc),
/// (Time(child), Asc)]`. Returns the seed var on match; `None` otherwise (caller falls back to
/// the materialize-then-sort path).
pub(crate) fn rows_streaming_seed(box_: &Box, plan: &BoxPlan, spec: &RowsSpec) -> Option<VarId> {
    if !box_.children.is_empty() || !single_relational_filter(box_) {
        return None;
    }
    let [PlanStep::Scan(seed_var), step1] = plan.steps.as_slice() else {
        return None;
    };
    let child_var = *match step1 {
        PlanStep::ExtendE2O { var, neighbor, .. } | PlanStep::ExtendO2O { var, neighbor, .. }
            if neighbor == seed_var =>
        {
            var
        }
        _ => return None,
    };
    match spec.order_by.as_slice() {
        [(Expr::Id(o), Dir::Asc), (Expr::Time(t), Dir::Asc)]
            if *o == *seed_var && *t == child_var =>
        {
            Some(*seed_var)
        }
        _ => None,
    }
}

/// Extends one already-bound seed via `step` and scores every distinct resulting row (see
/// [`score_leaf`]) into `out` -- the streaming counterpart of [`batched_extend_fold`], reusing the
/// same caller-owned, cleared-not-reallocated dedup sets across seeds. No
/// `verify_binding`/child-folding call: [`rows_streaming_seed`]'s eligibility check (one
/// relational filter, exactly the one this extend just traversed; no children) makes both
/// trivially true.
#[allow(clippy::too_many_arguments)]
fn stream_extend_rows<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    step: &PlanStep,
    spec: &RowsSpec,
    vars: &mut Vars<Q>,
    out: &mut Vec<(Vec<Value>, Vec<Value>)>,
    ev_seen: &mut FxHashSet<Q::EventRepr>,
    ob_seen: &mut FxHashSet<Q::ObjectRepr>,
) {
    match step {
        PlanStep::ExtendE2O {
            var,
            neighbor,
            var_is_event,
            qualifier,
        } => {
            let ext_decl = &box_.new_vars[*var];
            if *var_is_event {
                let ob = ob_of(vars, *neighbor).clone();
                ev_seen.clear();
                for (q, ev) in ocel.get_e2o_rev(&ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ev(ocel, &ev, &ext_decl.types)
                        || !ev_seen.insert(ev.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ev(ev));
                    out.push(score_leaf(spec, vars, &[], ocel));
                }
            } else {
                let ev = ev_of(vars, *neighbor).clone();
                ob_seen.clear();
                for (q, ob) in ocel.get_e2o(&ev) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &ext_decl.types)
                        || !ob_seen.insert(ob.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ob(ob));
                    out.push(score_leaf(spec, vars, &[], ocel));
                }
            }
            vars[*var] = None;
        }
        PlanStep::ExtendO2O {
            var,
            neighbor,
            var_is_from,
            qualifier,
        } => {
            let ext_decl = &box_.new_vars[*var];
            let neighbor_ob = ob_of(vars, *neighbor).clone();
            ob_seen.clear();
            if *var_is_from {
                for (q, ob) in ocel.get_o2o_rev(&neighbor_ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &ext_decl.types)
                        || !ob_seen.insert(ob.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ob(ob));
                    out.push(score_leaf(spec, vars, &[], ocel));
                }
            } else {
                for (q, ob) in ocel.get_o2o(&neighbor_ob) {
                    if !qualifier_matches(qualifier, q.as_ref())
                        || !type_ok_ob(ocel, &ob, &ext_decl.types)
                        || !ob_seen.insert(ob.clone())
                    {
                        continue;
                    }
                    vars[*var] = Some(Entity::Ob(ob));
                    out.push(score_leaf(spec, vars, &[], ocel));
                }
            }
            vars[*var] = None;
        }
        PlanStep::Scan(_) => unreachable!("rows_streaming_seed requires steps[1] to extend"),
    }
}

/// Sequential entry point for [`try_stream_rows`]: collects [`rows_streaming_seed`]'s seed
/// candidates (not yet id-sorted) and hands off to [`eval_rows_stream_sorted`], which does the
/// actual streaming. See that function's doc for the shape/ordering/tie-break argument.
fn eval_rows_stream_by_seed<Q: QueryableOCEL>(
    spec: &RowsSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    seed_var: VarId,
    total_vars: usize,
    f: &mut dyn FnMut(&[Value]),
) {
    let seed_decl = &box_.new_vars[seed_var];
    let seeds: Vec<(String, Entity<Q>)> = candidates_for_var(ocel, seed_decl)
        .into_iter()
        .map(|e| (entity_id(ocel, &e).into_owned(), e))
        .collect();
    eval_rows_stream_sorted(spec, ocel, box_, plan, seed_var, total_vars, seeds, f);
}

/// Seed-count granularity [`eval_rows_stream_by_seed_par`] sub-chunks a flush group into: each
/// sub-chunk is one rayon task that reuses its own dedup sets across the sub-chunk's own seeds
/// (same reuse-across-many-seeds idea as [`AGG_BATCH`]/[`batched_extend_fold`]), so a flush
/// group's parallelism comes from having several such tasks in flight, not from allocating fresh
/// dedup sets per seed.
const ROWS_PAR_SUBCHUNK: usize = 128;

/// Parallel counterpart of [`eval_rows_stream_by_seed`]: seeds (pre-sorted by id) are visited in
/// [`AGG_BATCH`]-sized *flush groups*, sequentially group by group; within a group, seeds are
/// further split into [`ROWS_PAR_SUBCHUNK`]-sized rayon tasks (each task reuses one set of dedup
/// sets across its own seeds, avoiding a fresh `FxHashSet` per seed the way plain `par_iter` over
/// individual seeds would -- see [`eval_rows_stream_by_seed`]'s doc for why that cost matters
/// here), collected in order, then the whole group's rows are flushed to `f` before the next
/// group is computed. Bounds peak memory to one flush group's rows (`AGG_BATCH` seeds' worth, not
/// the full result) while still spreading the seed-extend/sort work over multiple cores. Falls
/// back to the sequential [`eval_rows_stream_sorted`] below [`PAR_THRESHOLD`] seeds.
fn eval_rows_stream_by_seed_par<Q>(
    spec: &RowsSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    seed_var: VarId,
    total_vars: usize,
    f: &mut dyn FnMut(&[Value]),
) where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
{
    let seed_decl = &box_.new_vars[seed_var];
    let mut seeds: Vec<(String, Entity<Q>)> = candidates_for_var(ocel, seed_decl)
        .into_iter()
        .map(|e| (entity_id(ocel, &e).into_owned(), e))
        .collect();
    if seeds.len() < PAR_THRESHOLD {
        return eval_rows_stream_sorted(spec, ocel, box_, plan, seed_var, total_vars, seeds, f);
    }
    seeds.par_sort_by(|a, b| a.0.cmp(&b.0));

    let mut remaining = spec.limit;
    if remaining == Some(0) {
        return;
    }
    let step1 = &plan.steps[1];
    'groups: for group in seeds.chunks(AGG_BATCH) {
        let group_rows: Vec<Vec<Vec<Value>>> = group
            .par_chunks(ROWS_PAR_SUBCHUNK)
            .map(|sub| {
                let mut vars: Vars<Q> = vec![None; total_vars];
                let mut buf: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
                let mut ev_seen: FxHashSet<Q::EventRepr> = FxHashSet::default();
                let mut ob_seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();
                let mut sub_rows: Vec<Vec<Value>> = Vec::new();
                for (_, seed) in sub {
                    vars[seed_var] = Some(seed.clone());
                    buf.clear();
                    stream_extend_rows(
                        ocel,
                        box_,
                        step1,
                        spec,
                        &mut vars,
                        &mut buf,
                        &mut ev_seen,
                        &mut ob_seen,
                    );
                    buf.sort_by(|a, b| cmp_scored(&spec.order_by, a, b));
                    sub_rows.extend(buf.drain(..).map(|(_, row)| row));
                }
                sub_rows
            })
            .collect();
        for rows in &group_rows {
            for row in rows {
                f(row);
                if let Some(r) = remaining.as_mut() {
                    *r -= 1;
                    if *r == 0 {
                        break 'groups;
                    }
                }
            }
        }
    }
}

/// Sequential streaming per-seed fast path for [`Output::Rows`] queries matching
/// [`rows_streaming_seed`]'s shape (the dfg/variants trace-query shape) -- shared tail of
/// [`eval_rows_stream_by_seed`] and [`eval_rows_stream_by_seed_par`]'s below-[`PAR_THRESHOLD`]
/// fallback. `seeds` (not yet id-sorted) are sorted, then visited in that ascending `Id(seed)`
/// order -- sorting `#seeds`, not `#rows` -- and for each seed extended via
/// [`stream_extend_rows`], stably sorted by `order_by` and streamed straight to `f`. No global row
/// buffer and no global sort: one seed's rows live in memory at a time.
///
/// Row sequence matches the materialize-then-sort path exactly: `Id(seed)` is unique per seed, so
/// the full `order_by` key can only tie *within* one seed's own rows -- ties across different
/// seeds are impossible. A stable sort within each seed therefore reproduces precisely the
/// tie-break a single stable global sort over all rows would have produced, since it would also
/// resolve those same intra-seed ties by original enumeration order (and every one of a seed's
/// rows is already contiguous, in that same order, in this function's own per-seed buffer).
#[allow(clippy::too_many_arguments)]
fn eval_rows_stream_sorted<Q: QueryableOCEL>(
    spec: &RowsSpec,
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    seed_var: VarId,
    total_vars: usize,
    mut seeds: Vec<(String, Entity<Q>)>,
    f: &mut dyn FnMut(&[Value]),
) {
    seeds.sort_by(|a, b| a.0.cmp(&b.0));

    let mut remaining = spec.limit;
    if remaining == Some(0) {
        return;
    }
    let step1 = &plan.steps[1];
    let mut vars: Vars<Q> = vec![None; total_vars];
    let mut buf: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
    let mut ev_seen: FxHashSet<Q::EventRepr> = FxHashSet::default();
    let mut ob_seen: FxHashSet<Q::ObjectRepr> = FxHashSet::default();

    'seeds: for (_, seed) in seeds {
        vars[seed_var] = Some(seed);
        buf.clear();
        stream_extend_rows(
            ocel,
            box_,
            step1,
            spec,
            &mut vars,
            &mut buf,
            &mut ev_seen,
            &mut ob_seen,
        );
        vars[seed_var] = None;
        buf.sort_by(|a, b| cmp_scored(&spec.order_by, a, b));
        for (_, row) in &buf {
            f(row);
            if let Some(r) = remaining.as_mut() {
                *r -= 1;
                if *r == 0 {
                    break 'seeds;
                }
            }
        }
    }
}

/// Compiles `query`'s root-box plan and checks it against [`rows_streaming_seed`]'s shape;
/// `Ok(None)` means either `query.output` isn't `Output::Rows` or the shape doesn't match --
/// either way the caller falls back to `run_query` + iterate. Shared by [`try_stream_rows`] and
/// [`try_stream_rows_par`] so the shape check and its plan-compile cost aren't duplicated between
/// them.
#[allow(clippy::type_complexity)]
fn compile_rows_stream_plan<Q: QueryableOCEL>(
    query: &Query,
    ocel: &Q,
) -> Result<Option<(BoxPlan, VarId, usize)>, String> {
    let Output::Rows(spec) = &query.output else {
        return Ok(None);
    };
    query.validate()?;
    let total_vars = query.collect_vars().len();
    let consumer_exprs = output_consumer_exprs(&query.output);
    let mut cache = HashMap::new();
    let plan = compile_box(
        ocel,
        &query.root,
        0,
        &vec![false; total_vars],
        &consumer_exprs,
        &mut cache,
    );
    Ok(rows_streaming_seed(&query.root, &plan, spec).map(|seed_var| (plan, seed_var, total_vars)))
}

/// Entry point for [`QueryableOCEL::run_query_fold`]'s in-memory default: attempts
/// [`eval_rows_stream_by_seed`]'s streaming per-seed fast path, avoiding both the full
/// `QueryResult` materialization and the one global sort the fallback (`run_query` + iterate)
/// would otherwise do. Returns `Ok(false)` (nothing streamed to `f`) when `query` doesn't match
/// the required shape -- the caller falls back to `run_query` + iterate; results are identical
/// either way, this only changes how -- and how much memory -- they're produced with.
pub fn try_stream_rows<Q: QueryableOCEL>(
    query: &Query,
    ocel: &Q,
    f: &mut dyn FnMut(&[Value]),
) -> Result<bool, String> {
    let Some((plan, seed_var, total_vars)) = compile_rows_stream_plan(query, ocel)? else {
        return Ok(false);
    };
    let Output::Rows(spec) = &query.output else {
        unreachable!("compile_rows_stream_plan only returns Some for Output::Rows")
    };
    eval_rows_stream_by_seed(spec, ocel, &query.root, &plan, seed_var, total_vars, f);
    Ok(true)
}

/// Parallel counterpart of [`try_stream_rows`], used by in-memory backends' `run_query_fold`
/// override (see `impl_queryable_from_linked!`): same shape detection, but
/// [`eval_rows_stream_by_seed_par`]'s flush-grouped rayon fast path instead of the fully
/// sequential one.
pub fn try_stream_rows_par<Q>(
    query: &Query,
    ocel: &Q,
    f: &mut dyn FnMut(&[Value]),
) -> Result<bool, String>
where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
{
    let Some((plan, seed_var, total_vars)) = compile_rows_stream_plan(query, ocel)? else {
        return Ok(false);
    };
    let Output::Rows(spec) = &query.output else {
        unreachable!("compile_rows_stream_plan only returns Some for Output::Rows")
    };
    eval_rows_stream_by_seed_par(spec, ocel, &query.root, &plan, seed_var, total_vars, f);
    Ok(true)
}

/// Evaluate `query` against `ocel`: compile the root box's plan once, enumerate its bindings
/// (recursively folding correlated `children` per binding), then project (`Output::Rows`) or
/// group/fold (`Output::Aggregate`) straight into a [`QueryResult`] as bindings are produced.
pub fn evaluate<Q: QueryableOCEL>(query: &Query, ocel: &Q) -> Result<QueryResult, String> {
    query.validate()?;
    let total_vars = query.collect_vars().len();
    let consumer_exprs = output_consumer_exprs(&query.output);
    let mut cache = HashMap::new();
    let plan = compile_box(
        ocel,
        &query.root,
        0,
        &vec![false; total_vars],
        &consumer_exprs,
        &mut cache,
    );

    let mut counts = NodeCounts::default();
    Ok(match &query.output {
        Output::Rows(spec) => {
            eval_rows_fold(spec, ocel, &query.root, &plan, total_vars, &mut counts)
        }
        Output::Aggregate(spec) => {
            eval_aggregate_fold(spec, ocel, &query.root, &plan, total_vars, &mut counts)
        }
    })
}

/// Parallel (rayon) counterpart of [`evaluate`]: same binding-box semantics and identical
/// results, but the root box's seed-variable scan, the `Aggregate` group-by fold, and `Rows`
/// projection/sort all run via `par_iter`/`par_sort_by`. Requires `Q: Sync` (and `Send + Sync`
/// entity reprs) to share `ocel` and bindings across threads -- in-memory backends
/// (`IndexLinkedOCEL`/`SlimLinkedOCEL`/`IDLinkedOCEL`) satisfy this and use it as their
/// `run_query` default; `DuckDbLinkedOCEL` overrides `run_query` with SQL pushdown instead (its
/// `duckdb::Connection` isn't `Sync`), so it never calls this.
pub fn evaluate_par<Q>(query: &Query, ocel: &Q) -> Result<QueryResult, String>
where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
    Q::EvTypeId: Send + Sync,
    Q::ObTypeId: Send + Sync,
{
    query.validate()?;
    let total_vars = query.collect_vars().len();
    let consumer_exprs = output_consumer_exprs(&query.output);
    let mut cache = HashMap::new();
    let plan = compile_box(
        ocel,
        &query.root,
        0,
        &vec![false; total_vars],
        &consumer_exprs,
        &mut cache,
    );

    let mut counts = NodeCounts::default();
    let mut seeds = Vec::new();
    Ok(match &query.output {
        Output::Rows(spec) => eval_rows_par_fold(
            spec,
            ocel,
            &query.root,
            &plan,
            total_vars,
            &mut counts,
            &mut seeds,
        ),
        Output::Aggregate(spec) => eval_aggregate_par_fold(
            spec,
            ocel,
            &query.root,
            &plan,
            total_vars,
            &mut counts,
            &mut seeds,
        ),
    })
}

// ------------------------------------------------------------------------------------------
// Multi-output: one count per tree node from a single enumeration, rows replayed per node on
// demand. See `Query::emits`.
// ------------------------------------------------------------------------------------------

/// A multi-output evaluation: the query's own answer plus one binding count per emitting node.
///
/// Rows for a node are not materialized here. [`evaluate_node`] replays a single node on demand,
/// which is what keeps a deep tree from building tables nobody opened -- and lets that replay be
/// paged, which materializing everything up front cannot be.
#[derive(Clone)]
pub struct MultiQueryResult<Q: QueryableOCEL> {
    /// [`Query::output`], exactly as [`evaluate`] produces it.
    pub output: QueryResult,
    /// One entry per [`Query::emits`] entry, in that order: how many bindings that node has,
    /// summed over every surviving ancestor binding.
    pub node_counts: Vec<u64>,
    /// The root box's surviving bindings, so [`evaluate_node`] can descend without re-running the
    /// root scan. Only meaningful against the same OCEL this was produced from.
    seeds: Vec<Vars<Q>>,
}

impl<Q: QueryableOCEL> std::fmt::Debug for MultiQueryResult<Q> {
    /// Omits `seeds`, whose entity handles are backend-internal and have no `Debug`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiQueryResult")
            .field("output", &self.output)
            .field("node_counts", &self.node_counts)
            .field("seeds", &self.seeds.len())
            .finish()
    }
}

/// Evaluate `query`, additionally counting the bindings of every node named in
/// [`Query::emits`].
///
/// One enumeration, not one per node: the child bindings a node reports are the ones
/// [`fold_child_scalars`] already walks. A node's count is incremented as its bindings are
/// produced and rolled back if an ancestor is later pruned, so the cascade an ancestor's
/// `AggRange` causes is reproduced exactly.
///
/// `Query::output`'s `order_by`/`limit` do **not** cascade: they page the query's own answer,
/// while a node count reports every binding that survived filtering.
///
/// # Errors
/// Returns the message from [`Query::validate`].
pub fn evaluate_multi<Q>(query: &Query, ocel: &Q) -> Result<MultiQueryResult<Q>, String>
where
    Q: QueryableOCEL + Sync,
    Q::EventRepr: Send + Sync,
    Q::ObjectRepr: Send + Sync,
    Q::EvTypeId: Send + Sync,
    Q::ObTypeId: Send + Sync,
{
    query.validate()?;
    let total_vars = query.collect_vars().len();
    let consumer_exprs = output_consumer_exprs(&query.output);
    let mut cache = HashMap::new();
    let plan = compile_box_at(
        ocel,
        &query.root,
        0,
        &vec![false; total_vars],
        &consumer_exprs,
        &mut cache,
        &[],
        &query.emits,
    );

    let mut counts = NodeCounts::new(query.emits.len());
    let mut seeds: Vec<Vars<Q>> = Vec::new();

    // The same parallel folds `evaluate_par` uses: they collect the root's surviving bindings
    // alongside the answer, so replaying a node later costs that node's subtree and not the root
    // scan as well. Chunks merge in index order, so counts and seeds do not depend on scheduling.
    let output = match &query.output {
        Output::Rows(spec) => eval_rows_par_fold(
            spec,
            ocel,
            &query.root,
            &plan,
            total_vars,
            &mut counts,
            &mut seeds,
        ),
        Output::Aggregate(spec) => eval_aggregate_par_fold(
            spec,
            ocel,
            &query.root,
            &plan,
            total_vars,
            &mut counts,
            &mut seeds,
        ),
    };

    Ok(MultiQueryResult {
        output,
        node_counts: counts.counts,
        seeds,
    })
}

/// How many rows of a node to skip and take, for a caller paging one node's table.
#[derive(Debug, Clone, Copy, Default)]
pub struct Page {
    /// Rows to skip.
    pub offset: usize,
    /// Rows to return, or every remaining row.
    pub limit: Option<usize>,
}

/// Materialize one node's rows, replaying it from `res`'s cached root bindings.
///
/// Costs that node's own subtree per surviving root binding -- the root scan is not repeated --
/// and stops as soon as `page` is filled, so showing the first screen of a large node does not
/// build the whole table.
///
/// `res` must come from [`evaluate_multi`] on this same `query` and `ocel`.
///
/// # Errors
/// Returns an error if `path` names no [`Query::emits`] entry, or the message from
/// [`Query::validate`].
pub fn evaluate_node<Q: QueryableOCEL>(
    res: &MultiQueryResult<Q>,
    query: &Query,
    ocel: &Q,
    path: &[ChildRef],
    page: Page,
) -> Result<QueryResult, String> {
    let emit_idx = query
        .emit_index_of(path)
        .ok_or_else(|| format!("no Query.emits entry for path {path:?}"))?;
    let project = &query.emits[emit_idx].project;

    let total_vars = query.collect_vars().len();
    let consumer_exprs = output_consumer_exprs(&query.output);
    let mut cache = HashMap::new();
    let plan = compile_box_at(
        ocel,
        &query.root,
        0,
        &vec![false; total_vars],
        &consumer_exprs,
        &mut cache,
        &[],
        &query.emits,
    );

    let mut rows: Vec<Vec<Value>> = Vec::new();
    let mut skipped = 0usize;
    let mut push = |vars: &Vars<Q>, cs: &[Vec<Value>]| -> bool {
        if skipped < page.offset {
            skipped += 1;
            return true;
        }
        rows.push(
            project
                .iter()
                .map(|e| eval_expr(e, vars, cs, ocel))
                .collect(),
        );
        page.limit.is_none_or(|n| rows.len() < n)
    };

    if path.is_empty() {
        // The root's own bindings are already in hand.
        for seed in &res.seeds {
            if !push(seed, &[]) {
                break;
            }
        }
    } else {
        for seed in &res.seeds {
            if !descend_to_node(ocel, &query.root, &plan, seed, path, &mut push) {
                break;
            }
        }
    }

    Ok(QueryResult {
        columns: project.iter().map(column_name).collect(),
        rows,
    })
}

/// Walk `path` down from a surviving binding of `box_`, calling `emit` at every binding of the
/// node it names. Returns `false` once `emit` asks to stop.
///
/// Reuses [`enumerate_box_fold`], so the bindings this yields are exactly the ones the counting
/// pass counted -- there is no second notion of "surviving" to drift from the first.
fn descend_to_node<Q: QueryableOCEL>(
    ocel: &Q,
    box_: &Box,
    plan: &BoxPlan,
    vars: &Vars<Q>,
    path: &[ChildRef],
    emit: BindingSink<'_, Q>,
) -> bool {
    let (&c, rest) = path.split_first().expect("non-empty path");
    let child = &box_.children[c];
    let child_plan = &plan.children[c];

    // Inert: a replay counts nothing, it only reads the bindings back out.
    let mut counts = NodeCounts::default();
    let mut cur = vars.clone();
    enumerate_box_fold(
        ocel,
        &child.box_,
        &child_plan.plan,
        child_plan.start_id,
        &mut cur,
        &mut counts,
        &mut |cvars, cs| {
            if rest.is_empty() {
                emit(cvars, cs)
            } else {
                descend_to_node(ocel, &child.box_, &child_plan.plan, cvars, rest, emit)
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_data::object_centric::linked_ocel::{IndexLinkedOCEL, LinkedOCELAccess};
    use crate::core::event_data::object_centric::ocel_json::import_ocel_json_path;
    use crate::test_utils::get_test_data_path;

    fn load() -> IndexLinkedOCEL {
        let ocel = import_ocel_json_path(
            get_test_data_path()
                .join("ocel")
                .join("order-management.json"),
        )
        .unwrap();
        IndexLinkedOCEL::from_ocel(ocel)
    }

    /// One event `e1` linked to object `o1` via two qualifiers (`q1`, `q2`) and to `o2` via one
    /// qualifier (`q1`) -- the exact shape (an event double-linked to the same object under
    /// different qualifiers) distinct-binding dedup must collapse to one binding per pair.
    fn multi_qualifier_synthetic_ocel() -> crate::core::event_data::object_centric::OCEL {
        use crate::core::event_data::object_centric::{
            OCELEvent, OCELObject, OCELRelationship, OCELType, OCEL,
        };

        let time = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap();
        OCEL {
            event_types: vec![OCELType {
                name: "et".to_string(),
                attributes: vec![],
            }],
            object_types: vec![OCELType {
                name: "ot".to_string(),
                attributes: vec![],
            }],
            objects: vec![
                OCELObject {
                    id: "o1".to_string(),
                    object_type: "ot".to_string(),
                    attributes: vec![],
                    relationships: vec![],
                },
                OCELObject {
                    id: "o2".to_string(),
                    object_type: "ot".to_string(),
                    attributes: vec![],
                    relationships: vec![],
                },
            ],
            events: vec![OCELEvent::new(
                "e1",
                "et",
                time,
                vec![],
                vec![
                    OCELRelationship::new("o1", "q1"),
                    OCELRelationship::new("o1", "q2"),
                    OCELRelationship::new("o2", "q1"),
                ],
            )],
        }
    }

    fn multi_qualifier_query(qualifier: Option<String>) -> Query {
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
                    qualifier,
                }],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![(Expr::Id(1), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    // e:Event Any (var 0), o:Object OneOf(["orders"]) (var 1), E2O{event:0, object:1}.
    fn base_box() -> Box {
        Box {
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
        }
    }

    #[test]
    fn bindings_match_direct_e2o_count() {
        let ocel = load();
        // Ambiguous between `LinkedOCELAccess` and `QueryableOCEL` (both in scope, same
        // method names) -> disambiguate via UFCS to compute the expectation independently.
        assert!(LinkedOCELAccess::get_ob_types(&ocel).any(|t| t == "orders"));

        let expected: usize = LinkedOCELAccess::get_all_evs(&ocel)
            .flat_map(|ev| {
                LinkedOCELAccess::get_e2o(&ocel, ev)
                    .map(|(_, o)| *o)
                    .collect::<Vec<_>>()
            })
            .filter(|o| LinkedOCELAccess::get_ob_type_of(&ocel, *o) == "orders")
            .count();
        assert!(expected > 0);

        let query = Query {
            root: base_box(),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let result = evaluate(&query, &ocel).unwrap();
        assert_eq!(result.rows.len(), expected);
        assert_eq!(result.columns, vec!["v0_id", "v1_id"]);

        for row in result.rows.iter().take(5) {
            let (Value::Str(ev_id), Value::Str(ob_id)) = (&row[0], &row[1]) else {
                panic!("expected string id columns")
            };
            let ev = ocel.get_ev_by_id(ev_id).expect("event exists");
            let ob = ocel.get_ob_by_id(ob_id).expect("object exists");
            assert_eq!(LinkedOCELAccess::get_ob_type_of(&ocel, ob), "orders");
            assert!(LinkedOCELAccess::get_e2o(&ocel, ev).any(|(_, o)| *o == ob));
        }
    }

    #[test]
    fn projection_order_and_limit() {
        let ocel = load();
        let query = Query {
            root: base_box(),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(1), Expr::Type(0), Expr::Time(0)],
                order_by: vec![(Expr::Time(0), Dir::Asc)],
                limit: Some(5),
            }),
            emits: Vec::new(),
        };
        let result = evaluate(&query, &ocel).unwrap();
        assert_eq!(result.rows.len(), 5);
        assert_eq!(result.columns, vec!["v1_id", "v0_type", "v0_time"]);

        let mut last: Option<DateTime<FixedOffset>> = None;
        for row in &result.rows {
            let Value::Time(t) = &row[2] else {
                panic!("expected time column")
            };
            if let Some(prev) = last {
                assert!(prev <= *t);
            }
            last = Some(*t);
        }
    }

    #[test]
    fn value_ordering_handles_nan_and_negative_zero() {
        assert_eq!(Value::Float(0.0), Value::Float(-0.0));
        assert_eq!(Value::Float(f64::NAN), Value::Float(f64::NAN));
        let mut values = [
            Value::Float(2.0),
            Value::Float(f64::NAN),
            Value::Float(-0.0),
            Value::Float(1.0),
        ];
        values.sort();
        assert_eq!(values.len(), 4);
        // Negatives must sort below positives. Comparing raw `f64` bits (as this did
        // before) puts them above, because the sign bit dominates the bit pattern.
        let mut signed = [
            Value::Float(5.0),
            Value::Float(-5.0),
            Value::Float(0.0),
            Value::Float(-1.0),
        ];
        signed.sort();
        assert_eq!(
            signed,
            [
                Value::Float(-5.0),
                Value::Float(-1.0),
                Value::Float(0.0),
                Value::Float(5.0),
            ]
        );
    }

    /// `Int` and `Float` are one ordering class, so numbers compare by magnitude and
    /// `Eq`/`Hash` agree with that. Previously every `Int` sorted before every `Float`,
    /// which made `Min`/`Max` over a mixed-typed attribute return the wrong value.
    #[test]
    fn int_and_float_compare_by_magnitude() {
        assert_eq!(Value::Int(3), Value::Float(3.0));
        assert!(Value::Float(36.7) < Value::Int(134));
        assert!(Value::Int(-2) < Value::Float(-1.5));

        let mut mixed = [
            Value::Int(134),
            Value::Float(36.7),
            Value::Int(-2),
            Value::Float(11241.55),
        ];
        mixed.sort();
        assert_eq!(
            mixed,
            [
                Value::Int(-2),
                Value::Float(36.7),
                Value::Int(134),
                Value::Float(11241.55),
            ]
        );

        // Eq/Hash must agree, or grouping and ordering would disagree.
        let mut set = HashSet::new();
        set.insert(Value::Int(3));
        assert!(set.contains(&Value::Float(3.0)));
        assert_eq!(set.len(), 1);
        set.insert(Value::Float(3.0));
        assert_eq!(set.len(), 1, "Int(3) and Float(3.0) must be one group");
    }

    // orders (0) -"comprises"-> items (1) child box, used by both children tests below.
    fn order_items_child(range: Option<(Option<f64>, Option<f64>)>) -> Box {
        Box {
            new_vars: vec![VarDecl {
                kind: VarKind::Object,
                types: TypeConstraint::OneOf(vec!["orders".to_string()]),
            }],
            filters: range
                .map(|(min, max)| {
                    vec![Filter::AggRange {
                        child: 0,
                        agg_idx: 0,
                        min,
                        max,
                    }]
                })
                .unwrap_or_default(),
            children: vec![ChildBox {
                box_: Box {
                    new_vars: vec![VarDecl {
                        kind: VarKind::Object,
                        types: TypeConstraint::OneOf(vec!["items".to_string()]),
                    }],
                    filters: vec![Filter::O2O {
                        from: 0,
                        to: 1,
                        qualifier: None,
                    }],
                    children: vec![],
                },
                aggs: vec![Agg::Count],
            }],
        }
    }

    /// Strong correctness test: `Output::Aggregate` type-counts must equal an independently
    /// computed distinct-`(event, object)`-pair count per type. `locel_event_object_type_counts`
    /// instead counts once per `(event, object, qualifier)` relationship row; order-management.json
    /// has ~78 events double-linked to the same object under two qualifiers (e.g.
    /// forwarder+shipper), so its counts differ from the query layer's on exactly those pairs --
    /// but the set of `(event_type, object_type)` keys touched is identical either way.
    #[test]
    fn aggregate_type_counts_matches_locel_algorithm() {
        use crate::analysis::object_centric::oc_statistics::locel_event_object_type_counts;
        use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;

        let raw = import_ocel_json_path(
            get_test_data_path()
                .join("ocel")
                .join("order-management.json"),
        )
        .unwrap();
        let ocel = SlimLinkedOCEL::from_ocel(raw);

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
        let result = evaluate(&query, &ocel).unwrap();
        assert_eq!(result.columns, vec!["v0_type", "v1_type", "count"]);

        let got: HashMap<(String, String), i64> = result
            .rows
            .iter()
            .map(|row| {
                let (Value::Str(ev_ty), Value::Str(ob_ty), Value::Int(c)) =
                    (&row[0], &row[1], &row[2])
                else {
                    panic!("unexpected row shape: {row:?}")
                };
                ((ev_ty.clone(), ob_ty.clone()), *c)
            })
            .collect();

        // Distinct (event, object) pairs per type -- one binding per pair regardless of
        // qualifier count, matching `type_counts_via_query`'s semantics exactly.
        let mut expected: HashMap<(String, String), i64> = HashMap::new();
        for ev in LinkedOCELAccess::get_all_evs(&ocel) {
            let ev_ty = LinkedOCELAccess::get_ev_type_of(&ocel, ev).to_string();
            let mut seen = HashSet::new();
            for (_, ob) in LinkedOCELAccess::get_e2o(&ocel, ev) {
                if seen.insert(*ob) {
                    let ob_ty = LinkedOCELAccess::get_ob_type_of(&ocel, *ob).to_string();
                    *expected.entry((ev_ty.clone(), ob_ty)).or_insert(0) += 1;
                }
            }
        }
        assert!(!expected.is_empty());
        assert_eq!(got, expected);

        // Sanity: the row-counting reference still touches the same (event_type, object_type)
        // key set (it only over-counts multi-qualifier pairs, never invents/drops a key).
        let existing_keys: HashSet<(String, String)> = locel_event_object_type_counts(&ocel)
            .into_iter()
            .map(|(e, o, _)| (e, o))
            .collect();
        let expected_keys: HashSet<(String, String)> = expected.keys().cloned().collect();
        assert_eq!(existing_keys, expected_keys);
    }

    #[test]
    fn children_filter_keeps_orders_with_at_least_n_items() {
        let ocel = load();
        const N: usize = 3;

        let query = Query {
            root: order_items_child(Some((Some(N as f64), None))),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let result = evaluate(&query, &ocel).unwrap();

        let got: HashSet<String> = result
            .rows
            .iter()
            .map(|row| match &row[0] {
                Value::Str(s) => s.clone(),
                other => panic!("expected id column, got {other:?}"),
            })
            .collect();

        let expected: HashSet<String> = LinkedOCELAccess::get_obs_of_type(&ocel, "orders")
            .filter(|ob| LinkedOCELAccess::get_o2o(&ocel, *ob).count() >= N)
            .map(|ob| LinkedOCELAccess::get_ob_id(&ocel, ob).to_string())
            .collect();

        assert!(!expected.is_empty());
        assert!(
            expected.len() < LinkedOCELAccess::get_obs_of_type(&ocel, "orders").count(),
            "sanity: filter should also exclude some orders"
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn child_agg_scalar_is_projectable_and_matches_brute_force() {
        let ocel = load();

        let query = Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let result = evaluate(&query, &ocel).unwrap();
        assert_eq!(result.columns, vec!["v0_id", "child0_0"]);

        let all_orders = LinkedOCELAccess::get_obs_of_type(&ocel, "orders").count();
        assert_eq!(result.rows.len(), all_orders);

        for row in &result.rows {
            let (Value::Str(order_id), Value::Int(count)) = (&row[0], &row[1]) else {
                panic!("unexpected row shape: {row:?}")
            };
            let ob = ocel.get_ob_by_id(order_id).expect("order exists");
            let expected = LinkedOCELAccess::get_o2o(&ocel, ob).count() as i64;
            assert_eq!(*count, expected, "child count mismatch for {order_id}");
        }
    }

    /// Closes a P4a gap: an `ObjectAttr` numeric-range filter (SQL `EXISTS`/`TRY_CAST`
    /// lowering) must select the same bindings as the in-memory evaluator.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn attr_filter_rows_match_duckdb_backend() {
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };

        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let reference = load();

        let out = get_test_data_path()
            .join("export")
            .join("eval-attr-filter-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        // e:Event Any (v0), o:Object OneOf(["orders"]) (v1), E2O{0,1}; keep orders with a
        // recorded price >= 5000.0 (order-management prices are mostly float-typed).
        let query = Query {
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
                },Filter::ObjectAttr {
                    object: 1,
                    name: "price".to_string(),
                    at: FilterAt::Sometime,
                    vf: ValueFilter::Float {
                        min: Some(5000.0),
                        max: None,
                    },
                }],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut eval_result = evaluate(&query, &reference).unwrap();
        let mut db_result = db.run_query(&query).unwrap();
        eval_result.rows.sort();
        db_result.rows.sort();

        assert!(
            !eval_result.rows.is_empty(),
            "sanity: filter should keep some rows"
        );
        assert_eq!(eval_result, db_result);
    }

    /// A binding is a variable assignment, not a relationship-junction row: `e1` is linked to
    /// `o1` via two qualifiers and to `o2` via one, so an unqualified `E2O` predicate must
    /// enumerate exactly 2 bindings (one per distinct object), not 3 junction rows.
    #[test]
    fn multi_qualifier_e2o_dedups_to_one_binding_per_pair() {
        let ocel = IndexLinkedOCEL::from_ocel(multi_qualifier_synthetic_ocel());

        let result = evaluate(&multi_qualifier_query(None), &ocel).unwrap();
        assert_eq!(
            result.rows.len(),
            2,
            "expected one binding per distinct object, got {:?}",
            result.rows
        );
        let ob_ids: Vec<&str> = result
            .rows
            .iter()
            .map(|row| match &row[1] {
                Value::Str(s) => s.as_str(),
                other => panic!("expected object id, got {other:?}"),
            })
            .collect();
        assert_eq!(ob_ids, vec!["o1", "o2"]);

        // A specific qualifier still selects exactly its own row -- dedup doesn't over-collapse.
        let q2_result = evaluate(&multi_qualifier_query(Some("q2".to_string())), &ocel).unwrap();
        assert_eq!(
            q2_result.rows.len(),
            1,
            "only o1 has a q2-qualified relationship"
        );
    }

    /// Cross-backend counterpart of [`multi_qualifier_e2o_dedups_to_one_binding_per_pair`]: the
    /// SQL translator's distinct-binding subquery wrap (`box_from_where_distinct`) must collapse
    /// the same duplicate-qualifier junction rows as the evaluator.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn multi_qualifier_e2o_distinct_binding_matches_duckdb_backend() {
        use crate::core::event_data::object_centric::ocel_json::export_ocel_json_to_path;
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };

        let raw = multi_qualifier_synthetic_ocel();
        let src = get_test_data_path()
            .join("export")
            .join("multi-qualifier-e2o.json");
        export_ocel_json_to_path(&raw, &src).unwrap();

        let out = get_test_data_path()
            .join("export")
            .join("multi-qualifier-e2o-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let ocel = IndexLinkedOCEL::from_ocel(raw);
        let query = multi_qualifier_query(None);

        let mut eval_result = evaluate(&query, &ocel).unwrap();
        let mut db_result = db.run_query(&query).unwrap();
        eval_result.rows.sort();
        db_result.rows.sort();

        assert_eq!(eval_result.rows.len(), 2);
        assert_eq!(eval_result, db_result);
    }

    /// Two events 200ms apart (`.900` and one second later at `.100`) straddling a
    /// second boundary -- the exact shape that used to trip the SQL side's
    /// `date_diff('second', ...)` boundary-crossing-count bug (it reported a distance of
    /// 1, not 0.2, for this pair).
    fn tbe_synthetic_ocel() -> crate::core::event_data::object_centric::OCEL {
        use crate::core::event_data::object_centric::{OCELEvent, OCELType, OCEL};

        let t0 = DateTime::parse_from_rfc3339("2024-01-01T00:00:00.900Z").unwrap();
        let t1 = DateTime::parse_from_rfc3339("2024-01-01T00:00:01.100Z").unwrap();
        OCEL {
            event_types: vec![OCELType {
                name: "et".to_string(),
                attributes: vec![],
            }],
            object_types: vec![],
            objects: vec![],
            events: vec![
                OCELEvent::new("e0", "et", t0, vec![], vec![]),
                OCELEvent::new("e1", "et", t1, vec![], vec![]),
            ],
        }
    }

    fn tbe_query(max_seconds: Option<f64>) -> Query {
        Query {
            root: Box {
                new_vars: vec![
                    VarDecl {
                        kind: VarKind::Event,
                        types: TypeConstraint::Any,
                    },
                    VarDecl {
                        kind: VarKind::Event,
                        types: TypeConstraint::Any,
                    },
                ],
                filters: vec![Filter::TimeBetweenEvents {
                    from: 0,
                    to: 1,
                    min_seconds: None,
                    max_seconds,
                }],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![(Expr::Id(0), Dir::Asc), (Expr::Id(1), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    /// True elapsed time for every ordered pair among 2 events is in `{-0.2, 0.0, 0.2}`
    /// seconds -- all within a `max_seconds: 0.5` bound, so all 4 pairs (including the two
    /// self-pairs) must be accepted.
    #[test]
    fn time_between_events_subsecond_precision_evaluator() {
        let ocel = IndexLinkedOCEL::from_ocel(tbe_synthetic_ocel());
        let query = tbe_query(Some(0.5));
        let result = evaluate(&query, &ocel).unwrap();
        assert_eq!(result.rows.len(), 4);
    }

    /// Cross-backend counterpart: under the old `date_diff('second', ...)` lowering, the
    /// boundary-crossing pair reported a distance of 1 (not 0.2) and would be wrongly
    /// rejected by the SQL backend while the evaluator (correctly) accepted it.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn time_between_events_subsecond_precision_matches_duckdb_backend() {
        use crate::core::event_data::object_centric::ocel_json::export_ocel_json_to_path;
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };

        let raw = tbe_synthetic_ocel();
        let src = get_test_data_path()
            .join("export")
            .join("tbe-subsecond.json");
        export_ocel_json_to_path(&raw, &src).unwrap();

        let out = get_test_data_path()
            .join("export")
            .join("tbe-subsecond-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let ocel = IndexLinkedOCEL::from_ocel(raw);
        let query = tbe_query(Some(0.5));

        let mut eval_result = evaluate(&query, &ocel).unwrap();
        let mut db_result = db.run_query(&query).unwrap();
        eval_result.rows.sort();
        db_result.rows.sort();

        assert_eq!(eval_result.rows.len(), 4);
        assert_eq!(eval_result, db_result);
    }

    /// `o1` has a recorded `price`; `o2` has zero recorded values for `price` at all --
    /// the vacuous-truth edge case for `FilterAt::Always` (`Iterator::all` on an empty
    /// iterator is `true`). Both backends must keep both objects.
    fn always_empty_history_synthetic_ocel() -> crate::core::event_data::object_centric::OCEL {
        use crate::core::event_data::object_centric::{
            OCELObject, OCELObjectAttribute, OCELType, OCEL,
        };

        let t = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap();
        OCEL {
            event_types: vec![],
            object_types: vec![OCELType {
                name: "ot".to_string(),
                attributes: vec![],
            }],
            objects: vec![
                OCELObject {
                    id: "o1".to_string(),
                    object_type: "ot".to_string(),
                    attributes: vec![OCELObjectAttribute::new("price", 100.0, t)],
                    relationships: vec![],
                },
                OCELObject {
                    id: "o2".to_string(),
                    object_type: "ot".to_string(),
                    attributes: vec![],
                    relationships: vec![],
                },
            ],
            events: vec![],
        }
    }

    fn always_empty_history_query() -> Query {
        Query {
            root: Box {
                new_vars: vec![VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec!["ot".to_string()]),
                }],
                filters: vec![Filter::ObjectAttr {
                    object: 0,
                    name: "price".to_string(),
                    at: FilterAt::Always,
                    vf: ValueFilter::Float {
                        min: Some(0.0),
                        max: None,
                    },
                }],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    #[test]
    fn always_filter_vacuous_true_on_empty_history_evaluator() {
        let ocel = IndexLinkedOCEL::from_ocel(always_empty_history_synthetic_ocel());
        let query = always_empty_history_query();
        let result = evaluate(&query, &ocel).unwrap();
        assert_eq!(
            result.rows.len(),
            2,
            "o2 has no recorded price at all -- Always is vacuously true for it"
        );
    }

    /// Cross-backend counterpart: the SQL `Always` lowering used to require at least one
    /// recorded value to exist (`AND EXISTS(...)`), dropping `o2` while the evaluator kept
    /// it.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn always_filter_vacuous_true_on_empty_history_matches_duckdb_backend() {
        use crate::core::event_data::object_centric::ocel_json::export_ocel_json_to_path;
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };

        let raw = always_empty_history_synthetic_ocel();
        let src = get_test_data_path()
            .join("export")
            .join("always-empty-history.json");
        export_ocel_json_to_path(&raw, &src).unwrap();

        let out = get_test_data_path()
            .join("export")
            .join("always-empty-history-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let ocel = IndexLinkedOCEL::from_ocel(raw);
        let query = always_empty_history_query();

        let mut eval_result = evaluate(&query, &ocel).unwrap();
        let mut db_result = db.run_query(&query).unwrap();
        eval_result.rows.sort();
        db_result.rows.sort();

        assert_eq!(eval_result.rows.len(), 2);
        assert_eq!(eval_result, db_result);
    }

    /// Open a `DuckDB` backend over order-management for a differential test.
    #[cfg(feature = "ocel-duckdb")]
    fn duckdb_backend(
        tag: &str,
    ) -> crate::core::event_data::object_centric::ocel_sql::DuckDbLinkedOCEL {
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let out = get_test_data_path()
            .join("export")
            .join(format!("eval-{tag}.duckdb"));
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        DuckDbLinkedOCEL::open(&out).unwrap()
    }

    /// Regression: `FilterAt::AtEvent` lowered through a raw string whose `\` line
    /// continuations became literal backslashes, so `DuckDB` rejected the SQL outright
    /// (a `syntax error` from `DuckDB`). No test covered this AST variant.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn object_attr_filter_at_event_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("attr-filter-at-event");

        let query = Query {
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
                },Filter::ObjectAttr {
                    object: 1,
                    name: "price".to_string(),
                    at: FilterAt::AtEvent(0),
                    vf: ValueFilter::Float {
                        min: Some(5000.0),
                        max: None,
                    },
                }],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut eval_result = evaluate(&query, &reference).unwrap();
        let mut db_result = db.run_query(&query).unwrap();
        eval_result.rows.sort();
        db_result.rows.sort();
        assert!(
            !eval_result.rows.is_empty(),
            "sanity: AtEvent filter should keep some rows"
        );
        assert_eq!(eval_result, db_result);
    }

    /// Regression: `AggAcc::Sequence` dropped the `by` directions and always sorted
    /// ascending, while the SQL path emitted `ARRAY_AGG(x ORDER BY k DESC)` -- so a `Desc`
    /// key returned reversed lists depending on the backend.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn sequence_desc_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("sequence-desc");

        let desc_query = |dir: Dir| Query {
            root: Box {
                new_vars: vec![
                    VarDecl {
                        kind: VarKind::Object,
                        types: TypeConstraint::OneOf(vec!["orders".to_string()]),
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
                    by: vec![(Expr::Time(1), dir)],
                }],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut asc_eval = evaluate(&desc_query(Dir::Asc), &reference).unwrap();
        let mut desc_eval = evaluate(&desc_query(Dir::Desc), &reference).unwrap();
        asc_eval.rows.sort();
        desc_eval.rows.sort();

        // The evaluator must actually honour Desc: some trace has to differ from its Asc form.
        let has_multi = asc_eval.rows.iter().any(|r| match &r[1] {
            Value::List(l) => l.len() > 1,
            _ => false,
        });
        assert!(has_multi, "sanity: need a trace with >1 event");
        assert_ne!(
            asc_eval, desc_eval,
            "Desc must not produce the same lists as Asc"
        );

        // ...and it must agree with what DuckDB's ARRAY_AGG(... ORDER BY ... DESC) returns.
        let mut desc_db = db.run_query(&desc_query(Dir::Desc)).unwrap();
        desc_db.rows.sort();
        assert_eq!(desc_eval, desc_db);
    }

    /// Regression: object attributes come back from the EAV table as `VARCHAR`, so a
    /// projected value was `Value::Str` on `DuckDB` but typed in-memory, and `Min`/`Max`
    /// compared lexicographically ("9" > "10") on one side and numerically on the other.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn object_attr_projection_and_minmax_match_duckdb_backend() {
        use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;
        let reference = SlimLinkedOCEL::from_ocel(
            import_ocel_json_path(
                get_test_data_path()
                    .join("ocel")
                    .join("order-management.json"),
            )
            .unwrap(),
        );
        let db = duckdb_backend("obj-attr-typing");

        let obs = || Box {
            new_vars: vec![VarDecl {
                kind: VarKind::Object,
                types: TypeConstraint::OneOf(vec!["orders".to_string()]),
            }],
            filters: vec![],
            children: vec![],
        };

        // (1) A projected object attribute must have the same `Value` variant on both sides.
        let proj = Query {
            root: obs(),
            output: Output::Rows(RowsSpec {
                project: vec![
                    Expr::Id(0),
                    Expr::Attr {
                        var: 0,
                        name: "price".to_string(),
                        at: OutAt::Latest,
                    },
                ],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let mut proj_eval = evaluate(&proj, &reference).unwrap();
        let mut proj_db = db.run_query(&proj).unwrap();
        proj_eval.rows.sort();
        proj_db.rows.sort();
        // `price` is declared float, and `SlimLinkedOCEL` reconciles each value to that
        // declaration on import, so the JSON literal `3083` becomes `Float(3083.0)` and
        // agrees exactly with the single `DOUBLE` the SQL side casts to.
        assert!(
            proj_eval
                .rows
                .iter()
                .any(|r| matches!(&r[1], Value::Float(_))),
            "sanity: prices are float-typed"
        );
        assert_eq!(proj_eval, proj_db);

        // (2) Min/Max over that attribute must agree numerically, not lexicographically.
        let minmax = Query {
            root: obs(),
            output: Output::Aggregate(AggSpec {
                group_by: vec![],
                aggregates: vec![
                    Agg::Min(Expr::Attr {
                        var: 0,
                        name: "price".to_string(),
                        at: OutAt::Latest,
                    }),
                    Agg::Max(Expr::Attr {
                        var: 0,
                        name: "price".to_string(),
                        at: OutAt::Latest,
                    }),
                ],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let eval_mm = evaluate(&minmax, &reference).unwrap();
        let db_mm = db.run_query(&minmax).unwrap();
        assert_eq!(eval_mm, db_mm);
    }

    // ---------------------------------------------------------------------------------------
    // Composite filters: the two backends have to agree, and negation is where they would
    // silently stop agreeing -- SQL is three-valued, this evaluator is not.
    // ---------------------------------------------------------------------------------------

    /// `orders` with the `price` attribute, so a filter over it has both matching and
    /// non-matching bindings and objects with no value at all.
    #[cfg(feature = "ocel-duckdb")]
    fn priced_orders() -> Box {
        Box {
            new_vars: vec![VarDecl {
                kind: VarKind::Object,
                types: TypeConstraint::OneOf(vec!["orders".to_string()]),
            }],
            filters: vec![],
            children: vec![],
        }
    }

    #[cfg(feature = "ocel-duckdb")]
    fn price_over(min: f64) -> Filter {
        Filter::ObjectAttr {
            object: 0,
            name: "price".to_string(),
            at: FilterAt::Sometime,
            vf: ValueFilter::Float {
                min: Some(min),
                max: None,
            },
        }
    }

    #[cfg(feature = "ocel-duckdb")]
    fn rows_of(root: Box, filters: Vec<Filter>) -> Query {
        let mut root = root;
        root.filters = filters;
        Query {
            root,
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    /// `Not` over an object-attribute filter: the objects with no `price` at all must be kept,
    /// which is what the two-valued reading means and what a bare SQL `NOT` would drop.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn negated_object_attr_filter_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("not-object-attr");

        let plain = rows_of(priced_orders(), vec![price_over(1000.0)]);
        let negated = rows_of(
            priced_orders(),
            vec![Filter::Not(std::boxed::Box::new(price_over(1000.0)))],
        );
        let all = rows_of(priced_orders(), vec![]);

        for q in [&plain, &negated, &all] {
            let mut e = evaluate(q, &reference).unwrap();
            let mut d = db.run_query(q).unwrap();
            e.rows.sort();
            d.rows.sort();
            assert_eq!(e, d);
        }

        // Two-valued negation partitions: nothing is lost to "unknown".
        let kept = evaluate(&plain, &reference).unwrap().rows.len();
        let dropped = evaluate(&negated, &reference).unwrap().rows.len();
        let total = evaluate(&all, &reference).unwrap().rows.len();
        assert!(kept > 0 && dropped > 0, "the fixture must exercise both sides");
        assert_eq!(kept + dropped, total);
    }

    /// An event type declaring an `amount` attribute that one of its events leaves unset, so the
    /// wide `events` column is `NULL` on that row -- the shape a negated `EventAttr` needs.
    #[cfg(feature = "ocel-duckdb")]
    fn event_attr_synthetic_ocel() -> crate::core::event_data::object_centric::OCEL {
        use crate::core::event_data::object_centric::ocel_struct::OCELAttributeType;
        use crate::core::event_data::object_centric::{
            OCELEvent, OCELEventAttribute, OCELType, OCELTypeAttribute, OCEL,
        };

        let t = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap();
        let ev = |id: &str, amount: Option<f64>| OCELEvent {
            id: id.to_string(),
            event_type: "et".to_string(),
            time: t,
            attributes: amount
                .map(|a| {
                    vec![OCELEventAttribute {
                        name: "amount".to_string(),
                        value: OCELAttributeValue::Float(a),
                    }]
                })
                .unwrap_or_default(),
            relationships: vec![],
        };
        OCEL {
            event_types: vec![OCELType {
                name: "et".to_string(),
                attributes: vec![OCELTypeAttribute::new(
                    "amount",
                    &OCELAttributeType::Float,
                )],
            }],
            object_types: vec![],
            objects: vec![],
            events: vec![ev("e1", Some(500.0)), ev("e2", Some(10.0)), ev("e3", None)],
        }
    }

    /// `Not` over an *event*-attribute filter, the path that genuinely needs the SQL guard: it
    /// compares a wide column that is `NULL` when the attribute is unset, which reads as false in
    /// `WHERE` position but as `NULL` under a bare `NOT`.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn negated_event_attr_filter_matches_duckdb_backend() {
        use crate::core::event_data::object_centric::ocel_json::export_ocel_json_to_path;
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };

        let raw = event_attr_synthetic_ocel();
        let src = get_test_data_path().join("export").join("event-attr.json");
        export_ocel_json_to_path(&raw, &src).unwrap();
        let out = get_test_data_path()
            .join("export")
            .join("event-attr-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();
        let reference = IndexLinkedOCEL::from_ocel(raw);

        let events = Box {
            new_vars: vec![VarDecl {
                kind: VarKind::Event,
                types: TypeConstraint::Any,
            }],
            filters: vec![],
            children: vec![],
        };
        let big = Filter::EventAttr {
            event: 0,
            name: "amount".to_string(),
            vf: ValueFilter::Float {
                min: Some(100.0),
                max: None,
            },
        };
        let plain = rows_of(events.clone(), vec![big.clone()]);
        let negated = rows_of(events.clone(), vec![Filter::Not(std::boxed::Box::new(big))]);
        let all = rows_of(events, vec![]);

        for q in [&plain, &negated, &all] {
            let mut e = evaluate(q, &reference).unwrap();
            let mut d = db.run_query(q).unwrap();
            e.rows.sort();
            d.rows.sort();
            assert_eq!(e, d);
        }

        // `e3` has no value at all: two-valued negation keeps it, a bare SQL `NOT` would not.
        assert_eq!(evaluate(&plain, &reference).unwrap().rows.len(), 1);
        assert_eq!(evaluate(&negated, &reference).unwrap().rows.len(), 2);
        assert_eq!(evaluate(&all, &reference).unwrap().rows.len(), 3);
    }

    /// A negated `AggRange`: the fold of an empty child is `Count = 0`, which the range rejects
    /// and the negation therefore keeps.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn negated_agg_range_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("not-agg-range");

        let range = Filter::AggRange {
            child: 0,
            agg_idx: 0,
            min: Some(3.0),
            max: None,
        };
        let mut root = order_items_child(None);
        root.filters = vec![Filter::Not(std::boxed::Box::new(range))];
        let query = Query {
            root,
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut e = evaluate(&query, &reference).unwrap();
        let mut d = db.run_query(&query).unwrap();
        e.rows.sort();
        d.rows.sort();
        assert!(!e.rows.is_empty(), "sanity: the query should bind rows");
        assert_eq!(e, d);
    }

    /// A relational filter under an `Or` cannot become a `PlanStep`, and lowers to an `EXISTS`
    /// semi-join rather than a `FROM`-list join.
    ///
    /// Runs against a small synthetic log on purpose: with no *top-level* relational filter the
    /// planner has nothing to extend through, so both backends enumerate the full cross product of
    /// the two variables. That is the honest cost of a disjunction, not a regression, but it makes
    /// this the one shape a large fixture must not be used for.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn a_disjunction_of_relational_filters_matches_duckdb_backend() {
        use crate::core::event_data::object_centric::ocel_json::export_ocel_json_to_path;
        use crate::core::event_data::object_centric::ocel_sql::{
            stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
        };
        use crate::core::event_data::object_centric::{
            OCELEvent, OCELObject, OCELRelationship, OCELType, OCEL,
        };

        let t = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap();
        let ev = |id: &str, q: &str, o: &str| OCELEvent {
            id: id.to_string(),
            event_type: "et".to_string(),
            time: t,
            attributes: vec![],
            relationships: vec![OCELRelationship::new(o, q)],
        };
        let ob = |id: &str| OCELObject {
            id: id.to_string(),
            object_type: "ot".to_string(),
            attributes: vec![],
            relationships: vec![],
        };
        let raw = OCEL {
            event_types: vec![OCELType {
                name: "et".to_string(),
                attributes: vec![],
            }],
            object_types: vec![OCELType {
                name: "ot".to_string(),
                attributes: vec![],
            }],
            objects: vec![ob("o1"), ob("o2"), ob("o3")],
            events: vec![
                ev("e1", "order", "o1"),
                ev("e2", "item", "o2"),
                // Neither qualifier: excluded by the disjunction.
                ev("e3", "other", "o3"),
            ],
        };

        let src = get_test_data_path().join("export").join("or-relational.json");
        export_ocel_json_to_path(&raw, &src).unwrap();
        let out = get_test_data_path()
            .join("export")
            .join("or-relational-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();
        let reference = IndexLinkedOCEL::from_ocel(raw);

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
                filters: vec![Filter::Or(vec![
                    Filter::E2O {
                        event: 0,
                        object: 1,
                        qualifier: Some("order".to_string()),
                    },
                    Filter::E2O {
                        event: 0,
                        object: 1,
                        qualifier: Some("item".to_string()),
                    },
                ])],
                children: vec![],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut e = evaluate(&query, &reference).unwrap();
        let mut d = db.run_query(&query).unwrap();
        e.rows.sort();
        d.rows.sort();
        assert_eq!(e.rows.len(), 2, "e3's qualifier matches neither branch");
        assert_eq!(e, d);
    }

    /// `Compare` between two expressions, the one expression-to-expression test in the model.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn expression_comparison_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("compare");

        let root = Box {
            new_vars: vec![
                VarDecl {
                    kind: VarKind::Event,
                    types: TypeConstraint::Any,
                },
                VarDecl {
                    kind: VarKind::Event,
                    types: TypeConstraint::Any,
                },
                VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec!["orders".to_string()]),
                },
            ],
            filters: vec![
                Filter::E2O {
                    event: 0,
                    object: 2,
                    qualifier: None,
                },
                Filter::E2O {
                    event: 1,
                    object: 2,
                    qualifier: None,
                },
                Filter::Compare {
                    left: Expr::Time(0),
                    op: CmpOp::Lt,
                    right: Expr::Time(1),
                },
            ],
            children: vec![],
        };
        let query = Query {
            root,
            output: Output::Aggregate(AggSpec {
                group_by: vec![Expr::Type(0), Expr::Type(1)],
                aggregates: vec![Agg::Count],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut e = evaluate(&query, &reference).unwrap();
        let mut d = db.run_query(&query).unwrap();
        e.rows.sort();
        d.rows.sort();
        assert!(!e.rows.is_empty(), "sanity: the query should bind rows");
        assert_eq!(e, d);
    }

    /// `Expr::Satisfies` annotates instead of pruning: every binding survives, carrying a boolean
    /// column. Also pins that a `Satisfies` reading a child fold is seen as a consumer, so the
    /// child's count is not truncated by the bounded-count optimisation.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn satisfies_column_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("satisfies");

        let query = Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![
                    Expr::Id(0),
                    Expr::Satisfies(std::boxed::Box::new(Filter::AggRange {
                        child: 0,
                        agg_idx: 0,
                        min: Some(3.0),
                        max: None,
                    })),
                ],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut e = evaluate(&query, &reference).unwrap();
        let mut d = db.run_query(&query).unwrap();
        e.rows.sort();
        d.rows.sort();
        assert!(!e.rows.is_empty(), "sanity: the query should bind rows");

        let pruning = rows_of(
            order_items_child(Some((Some(3.0), None))),
            vec![Filter::AggRange {
                child: 0,
                agg_idx: 0,
                min: Some(3.0),
                max: None,
            }],
        );
        let kept = evaluate(&pruning, &reference).unwrap().rows.len();
        let satisfied = e
            .rows
            .iter()
            .filter(|r| matches!(r[1], Value::Bool(true)))
            .count();
        assert_eq!(kept, satisfied, "annotating must not prune");
        assert!(satisfied < e.rows.len(), "the fixture must exercise both");
        assert_eq!(e, d);
    }

    /// One child folded two ways from one variable declaration, which is what `aggs: Vec<Agg>`
    /// buys over two `ChildBox` entries.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn one_child_folded_two_ways_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("multi-agg-child");

        let mut root = order_items_child(None);
        root.children[0].aggs = vec![Agg::Count, Agg::CountDistinct(Expr::Type(1))];
        let query = Query {
            root,
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::ChildAgg(0, 0), Expr::ChildAgg(0, 1)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let mut e = evaluate(&query, &reference).unwrap();
        let mut d = db.run_query(&query).unwrap();
        e.rows.sort();
        d.rows.sort();
        assert!(!e.rows.is_empty(), "sanity: the query should bind rows");
        assert_eq!(e, d);
    }

    /// `having` cuts groups after aggregation, on both backends and before `limit`.
    #[cfg(feature = "ocel-duckdb")]
    #[test]
    fn having_matches_duckdb_backend() {
        let reference = load();
        let db = duckdb_backend("having");

        let spec = |having: Vec<(usize, Option<f64>, Option<f64>)>| Query {
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
                having,
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let unfiltered = spec(vec![]);
        let filtered = spec(vec![(0, Some(500.0), None)]);

        let mut e = evaluate(&filtered, &reference).unwrap();
        let mut d = db.run_query(&filtered).unwrap();
        e.rows.sort();
        d.rows.sort();
        assert_eq!(e, d);

        let all = evaluate(&unfiltered, &reference).unwrap();
        assert!(!e.rows.is_empty() && e.rows.len() < all.rows.len());
    }

    // ---------------------------------------------------------------------------------------
    // Multi-output: one count per node from a single enumeration, rows replayed on demand.
    // ---------------------------------------------------------------------------------------

    /// Counts every node, replays each node's rows, and pins the two properties the design rests
    /// on: a node's count equals the number of rows its replay yields, and a parent's `AggRange`
    /// cascades into its child's count.
    #[test]
    fn multi_output_counts_match_replayed_rows_and_cascade() {
        let reference = load();

        let emits = vec![
            NodeEmit {
                path: vec![],
                project: vec![Expr::Id(0)],
            },
            NodeEmit {
                path: vec![0],
                project: vec![Expr::Id(0), Expr::Id(1)],
            },
        ];

        let unfiltered = Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: emits.clone(),
        };

        let res = evaluate_multi(&unfiltered, &reference).unwrap();
        assert_eq!(res.node_counts.len(), 2);

        // The root's count is its own binding count, and matches its output.
        assert_eq!(res.node_counts[0] as usize, res.output.rows.len());

        // Every node's count equals what replaying it yields.
        for (i, emit) in unfiltered.emits.iter().enumerate() {
            let rows = evaluate_node(
                &res,
                &unfiltered,
                &reference,
                &emit.path,
                Page::default(),
            )
            .unwrap();
            assert_eq!(
                rows.rows.len() as u64,
                res.node_counts[i],
                "node {:?} count must equal its replayed row count",
                emit.path
            );
        }
        assert!(res.node_counts[1] > res.node_counts[0], "items fan out");

        // Now prune the parent: orders with fewer than 3 items are dropped, and the child's count
        // must drop with them rather than counting items of a parent that no longer exists.
        let filtered = Query {
            root: order_items_child(Some((Some(3.0), None))),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits,
        };
        let res2 = evaluate_multi(&filtered, &reference).unwrap();
        assert!(
            res2.node_counts[0] < res.node_counts[0],
            "the range must actually prune parents"
        );
        assert!(
            res2.node_counts[1] < res.node_counts[1],
            "a pruned parent's items must not be counted"
        );
        for (i, emit) in filtered.emits.iter().enumerate() {
            let rows =
                evaluate_node(&res2, &filtered, &reference, &emit.path, Page::default()).unwrap();
            assert_eq!(rows.rows.len() as u64, res2.node_counts[i]);
        }
    }

    /// `Query::output`'s `limit` pages the query's answer and must not cascade into node counts.
    #[test]
    fn a_root_limit_does_not_cascade_into_node_counts() {
        let reference = load();
        let emits = vec![NodeEmit {
            path: vec![0],
            project: vec![Expr::Id(1)],
        }];
        let mk = |limit| Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit,
            }),
            emits: emits.clone(),
        };

        let all = evaluate_multi(&mk(None), &reference).unwrap();
        let capped = evaluate_multi(&mk(Some(3)), &reference).unwrap();

        assert_eq!(capped.output.rows.len(), 3);
        assert!(all.output.rows.len() > 3);
        assert_eq!(
            all.node_counts, capped.node_counts,
            "a limit pages the answer, it does not change what a node contains"
        );
    }

    /// Paging a node replays only as far as it must.
    #[test]
    fn a_node_replay_can_be_paged() {
        let reference = load();
        let query = Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![(Expr::Id(0), Dir::Asc)],
                limit: None,
            }),
            emits: vec![NodeEmit {
                path: vec![0],
                project: vec![Expr::Id(0), Expr::Id(1)],
            }],
        };
        let res = evaluate_multi(&query, &reference).unwrap();
        let full = evaluate_node(&res, &query, &reference, &[0], Page::default()).unwrap();

        let page = evaluate_node(
            &res,
            &query,
            &reference,
            &[0],
            Page {
                offset: 5,
                limit: Some(10),
            },
        )
        .unwrap();
        assert_eq!(page.rows.len(), 10);
        assert_eq!(page.rows, full.rows[5..15]);
    }

    /// A `Query` with no `emits` must behave exactly as before, including keeping the batched
    /// aggregate fast path.
    #[test]
    fn multi_output_is_inert_without_emits() {
        let reference = load();
        let mut query = query_type_counts_multi();
        query.emits = Vec::new();
        let single = evaluate(&query, &reference).unwrap();
        let multi = evaluate_multi(&query, &reference).unwrap();
        assert_eq!(single, multi.output);
        assert!(multi.node_counts.is_empty());
    }

    fn query_type_counts_multi() -> Query {
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

    #[test]
    fn tmp_min_null_poison_repro() {
        use crate::core::event_data::object_centric::linked_ocel::IndexLinkedOCEL;
        use crate::core::event_data::object_centric::{
            OCELAttributeValue, OCELObject, OCELObjectAttribute, OCELType, OCEL,
        };
        let time = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap();
        let ocel = OCEL {
            event_types: vec![],
            object_types: vec![OCELType {
                name: "ot".to_string(),
                attributes: vec![],
            }],
            objects: vec![
                OCELObject {
                    id: "o1".to_string(),
                    object_type: "ot".to_string(),
                    attributes: vec![OCELObjectAttribute::new(
                        "n",
                        OCELAttributeValue::Float(5.0),
                        time,
                    )],
                    relationships: vec![],
                },
                OCELObject {
                    id: "o2".to_string(),
                    object_type: "ot".to_string(),
                    attributes: vec![], // no "n" attribute at all -> Value::Null
                    relationships: vec![],
                },
            ],
            events: vec![],
        };
        let ocel = IndexLinkedOCEL::from_ocel(ocel);
        let query = Query {
            root: Box {
                new_vars: vec![VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::Any,
                }],
                filters: vec![],
                children: vec![],
            },
            output: Output::Aggregate(AggSpec {
                group_by: vec![],
                aggregates: vec![Agg::Min(Expr::Attr {
                    var: 0,
                    name: "n".to_string(),
                    at: OutAt::Latest,
                })],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let result = evaluate(&query, &ocel).unwrap();
        eprintln!("MIN result row: {:?}", result.rows[0]);
        // Expect the true minimum (5.0), ignoring the missing/Null attribute -- like Sum/Avg do.
        assert_eq!(result.rows[0][0], Value::Float(5.0));
    }
}
