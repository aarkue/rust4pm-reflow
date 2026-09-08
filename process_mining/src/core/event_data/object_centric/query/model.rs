use std::collections::BTreeSet;

use chrono::{DateTime, FixedOffset};

/// Global index: pre-order position of a variable across all boxes' `new_vars`.
pub type VarId = usize;
/// Index of a [`ChildBox`] within its parent [`Box`]'s `children`.
pub type ChildRef = usize;

/// A pushdown query: a root binding box plus how to project/aggregate its bindings.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Query {
    /// The root binding box.
    pub root: Box,
    /// How bindings of the root box are turned into output rows.
    pub output: Output,
    /// Extra per-node projections, for a caller showing one table per tree node rather than only
    /// the query's answer. Empty for an ordinary single-output query.
    ///
    /// Declares what a node projects, not whether it is materialized:
    /// [`evaluate_multi`](super::eval::evaluate_multi) only counts each node's bindings, and
    /// [`evaluate_node`](super::eval::evaluate_node) replays one node's rows on demand.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub emits: Vec<NodeEmit>,
}

/// One node's projection, addressed by its position in the box tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeEmit {
    /// Path from the root: `[]` is the root box, `[0]` is `root.children[0].box_`, `[0, 1]` is
    /// that box's `children[1].box_`, and so on.
    pub path: Vec<ChildRef>,
    /// What one row of that node's own bindings projects, in its own scope.
    pub project: Vec<Expr>,
}

/// A binding box: variables it introduces, constraints on them, and correlated child boxes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Box {
    /// Variables newly declared (bound) by this box.
    pub new_vars: Vec<VarDecl>,
    /// Predicates every binding of this box must satisfy, implicitly conjoined.
    pub filters: Vec<Filter>,
    /// Correlated child boxes, each folded to one or more scalars for use by this box.
    pub children: Vec<ChildBox>,
}

/// Declaration of a single variable bound by a [`Box`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VarDecl {
    /// Whether the variable ranges over events or objects.
    pub kind: VarKind,
    /// Which OCEL types the variable may be bound to.
    pub types: TypeConstraint,
}

/// The kind of entity a variable ranges over.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VarKind {
    /// The variable ranges over events.
    Event,
    /// The variable ranges over objects.
    Object,
}

/// A constraint on the OCEL type(s) a variable may be bound to.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TypeConstraint {
    /// No type restriction.
    Any,
    /// Restricted to one of the given type names.
    OneOf(Vec<String>),
}

/// A boolean predicate over one binding, evaluated against a complete binding of the owning box.
///
/// Two-valued: an unresolvable check (an absent attribute, a `Null` operand) is `false`, never
/// "unknown", so [`Filter::Not`] over it is `true`. The SQL translation emits the guards that keep
/// three-valued SQL in step with that.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Filter {
    /// Event-to-object relation, optionally restricted to a qualifier.
    E2O {
        /// The event-side variable.
        event: VarId,
        /// The object-side variable.
        object: VarId,
        /// Optional qualifier restriction.
        qualifier: Option<String>,
    },
    /// Object-to-object relation, optionally restricted to a qualifier.
    O2O {
        /// The source-side variable.
        from: VarId,
        /// The target-side variable.
        to: VarId,
        /// Optional qualifier restriction.
        qualifier: Option<String>,
    },
    /// Constraint on the time distance between two events.
    TimeBetweenEvents {
        /// The earlier (reference) event variable.
        from: VarId,
        /// The later event variable.
        to: VarId,
        /// Minimum allowed distance in seconds, if any.
        min_seconds: Option<f64>,
        /// Maximum allowed distance in seconds, if any.
        max_seconds: Option<f64>,
    },
    /// Filter on an event attribute.
    EventAttr {
        /// The event variable.
        event: VarId,
        /// The attribute name.
        name: String,
        /// The value constraint.
        vf: ValueFilter,
    },
    /// Filter on an object attribute (possibly time-varying).
    ObjectAttr {
        /// The object variable.
        object: VarId,
        /// The attribute name.
        name: String,
        /// Which version of the attribute value to check.
        at: FilterAt,
        /// The value constraint.
        vf: ValueFilter,
    },
    /// Keep the binding iff a child's folded scalar falls in `[min, max]`.
    AggRange {
        /// Which child of the box owning this filter.
        child: ChildRef,
        /// Which of that child's [`ChildBox::aggs`].
        agg_idx: usize,
        /// Inclusive lower bound, if any.
        min: Option<f64>,
        /// Inclusive upper bound, if any.
        max: Option<f64>,
    },
    /// Compare two expressions. `Null` on either side is `false` for every operator, matching SQL
    /// rather than [`Expr`] evaluation's total ordering.
    Compare {
        /// Left operand.
        left: Expr,
        /// The comparison.
        op: CmpOp,
        /// Right operand.
        right: Expr,
    },
    /// Negation, two-valued: true when the inner filter did not match, absence included.
    Not(std::boxed::Box<Filter>),
    /// Disjunction. Empty is `false`.
    Or(Vec<Filter>),
}

/// The comparison a [`Filter::Compare`] performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

/// Which version of a (possibly time-varying) object attribute a filter applies to.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FilterAt {
    /// The value must satisfy the filter at all recorded versions.
    Always,
    /// The value must satisfy the filter at some recorded version.
    Sometime,
    /// The value as of the given event variable's time.
    AtEvent(VarId),
}

/// A constraint on an attribute value, keyed by expected value type.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ValueFilter {
    /// Integer value within an optional `[min, max]` range.
    Integer {
        /// Inclusive lower bound, if any.
        min: Option<i64>,
        /// Inclusive upper bound, if any.
        max: Option<i64>,
    },
    /// Float value within an optional `[min, max]` range.
    Float {
        /// Inclusive lower bound, if any.
        min: Option<f64>,
        /// Inclusive upper bound, if any.
        max: Option<f64>,
    },
    /// Boolean value equal to `is`.
    Boolean {
        /// The required boolean value.
        is: bool,
    },
    /// String value contained in `is_in`.
    String {
        /// The allowed set of string values.
        is_in: Vec<String>,
    },
    /// Timestamp within an optional `[from, to]` range.
    Time {
        /// Inclusive lower bound, if any.
        from: Option<DateTime<FixedOffset>>,
        /// Inclusive upper bound, if any.
        to: Option<DateTime<FixedOffset>>,
    },
}

/// A correlated child box, folded to one scalar per entry of `aggs`. Several folds share one
/// child so that folding the same relationship two ways does not re-declare its variables.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChildBox {
    /// The child binding box.
    pub box_: Box,
    /// How the child box's bindings are folded, each entry addressed by
    /// [`Expr::ChildAgg`]/[`Filter::AggRange`].
    pub aggs: Vec<Agg>,
}

/// A value derived from a binding.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Expr {
    /// The identity (id) of a bound variable.
    Id(VarId),
    /// The OCEL type of a bound variable.
    Type(VarId),
    /// The timestamp of a bound event variable.
    Time(VarId),
    /// An attribute value of a bound variable.
    Attr {
        /// The variable the attribute is read from.
        var: VarId,
        /// The attribute name.
        name: String,
        /// Which version of the attribute value to read.
        at: OutAt,
    },
    /// One scalar of a correlated child box's fold: which child, and which of its `aggs`.
    ChildAgg(ChildRef, usize),
    /// Whether a [`Filter`] holds for this binding, as a boolean value: annotates rather than
    /// prunes, unlike an entry of [`Box::filters`].
    Satisfies(std::boxed::Box<Filter>),
}

/// Which version of a (possibly time-varying) attribute value to output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OutAt {
    /// The most recently recorded value.
    Latest,
    /// The first recorded value.
    First,
    /// The value as of the given event variable's time.
    AtEvent(VarId),
}

/// How a query's root box bindings are turned into output rows.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Output {
    /// Per-binding rows.
    Rows(RowsSpec),
    /// Grouped/aggregated rows.
    Aggregate(AggSpec),
}

/// Specification of a per-binding row output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RowsSpec {
    /// Expressions projected into each output row.
    pub project: Vec<Expr>,
    /// Sort keys, applied in order.
    pub order_by: Vec<(Expr, Dir)>,
    /// Maximum number of rows to return, if any.
    pub limit: Option<usize>,
}

/// Specification of a grouped/aggregated output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AggSpec {
    /// Expressions to group by.
    pub group_by: Vec<Expr>,
    /// Aggregates computed per group.
    pub aggregates: Vec<Agg>,
    /// Post-grouping range checks on `aggregates`, by index. Note this indexes `aggregates` only,
    /// unlike `order_by`.
    pub having: Vec<(usize, Option<f64>, Option<f64>)>,
    /// Sort keys, indexing into `group_by` then `aggregates`, applied in order.
    pub order_by: Vec<(usize, Dir)>,
    /// Maximum number of rows to return, if any.
    pub limit: Option<usize>,
}

/// An aggregate function over an expression (or, for `Count`, over rows directly).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Agg {
    /// `COUNT(*)`; takes no argument.
    Count,
    /// Count of distinct values of the expression.
    CountDistinct(Expr),
    /// Minimum value of the expression.
    Min(Expr),
    /// Maximum value of the expression.
    Max(Expr),
    /// Sum of the expression's values.
    Sum(Expr),
    /// Average of the expression's values.
    Avg(Expr),
    /// The values of `of` within the group, ordered `by`, collected into a
    /// [`Value::List`](super::eval::Value::List). An object's activity trace, for example, is
    /// `Sequence { of: Type(event), by: [(Time(event), Asc)] }`.
    Sequence {
        /// The per-row value to collect.
        of: Expr,
        /// Intra-group ordering of the collected values.
        by: Vec<(Expr, Dir)>,
    },
}

/// Sort direction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Dir {
    /// Ascending.
    Asc,
    /// Descending.
    Desc,
}

/// Info about a single variable, as enumerated by [`Query::collect_vars`].
#[derive(Debug, Clone, PartialEq)]
pub struct VarInfo {
    /// The variable's global id.
    pub id: VarId,
    /// The variable's kind (event or object).
    pub kind: VarKind,
    /// Nesting depth of the box that declares this variable (root = 0).
    pub box_depth: usize,
}

impl Expr {
    /// The highest child index this expression reads, or `None` if it reads no child fold.
    /// Decides whether evaluating it has to wait until the owning box's children are folded.
    #[must_use]
    pub fn max_child_ref(&self) -> Option<ChildRef> {
        match self {
            Expr::Id(_) | Expr::Type(_) | Expr::Time(_) | Expr::Attr { .. } => None,
            Expr::ChildAgg(c, _) => Some(*c),
            Expr::Satisfies(f) => f.max_child_ref(),
        }
    }
}

impl Filter {
    /// The highest child index this filter reads, or `None` if it reads no child fold. See
    /// [`Expr::max_child_ref`].
    #[must_use]
    pub fn max_child_ref(&self) -> Option<ChildRef> {
        match self {
            Filter::E2O { .. }
            | Filter::O2O { .. }
            | Filter::TimeBetweenEvents { .. }
            | Filter::EventAttr { .. }
            | Filter::ObjectAttr { .. } => None,
            Filter::AggRange { child, .. } => Some(*child),
            Filter::Compare { left, right, .. } => {
                max_opt(left.max_child_ref(), right.max_child_ref())
            }
            Filter::Not(inner) => inner.max_child_ref(),
            Filter::Or(fs) => fs.iter().fold(None, |acc, f| max_opt(acc, f.max_child_ref())),
        }
    }

    /// This filter as a top-level relational join, or `None`. Only ever matches the outermost
    /// variant, since lifting an `E2O` out of a [`Filter::Or`] or [`Filter::Not`] into an
    /// enumeration step would drop bindings the composite keeps.
    #[must_use]
    pub fn as_relation(&self) -> Option<&Filter> {
        match self {
            Filter::E2O { .. } | Filter::O2O { .. } => Some(self),
            _ => None,
        }
    }
}

fn max_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl Query {
    /// Enumerate all variables across the box tree in pre-order (root's own vars first, then
    /// depth-first into `children[0].box_`, etc.), assigning each its global [`VarId`].
    pub fn collect_vars(&self) -> Vec<VarInfo> {
        let mut out = Vec::new();
        collect_box_vars(&self.root, 0, &mut out);
        out
    }

    /// Check that every variable/child reference is in range, in scope and of the expected kind,
    /// and that no `List`-valued fold is used where only a scalar works. Returns the first
    /// violation found.
    pub fn validate(&self) -> Result<(), String> {
        let kinds: Vec<VarKind> = self.collect_vars().into_iter().map(|v| v.kind).collect();
        let mut next_id: VarId = 0;
        validate_box(&self.root, &[], &kinds, &mut next_id, None)?;
        let root_scope: Vec<VarId> = (0..self.root.new_vars.len()).collect();
        validate_output(&self.output, &root_scope, &kinds, &self.root)?;
        self.validate_emits(&kinds)
    }

    /// Every [`NodeEmit`] names a real node and projects only what is in that node's scope.
    fn validate_emits(&self, kinds: &[VarKind]) -> Result<(), String> {
        let mut seen: BTreeSet<&[ChildRef]> = BTreeSet::new();
        for emit in &self.emits {
            if !seen.insert(&emit.path) {
                return Err(format!("Query.emits: duplicate path {:?}", emit.path));
            }
            if emit.project.is_empty() {
                return Err(format!("Query.emits: path {:?} projects nothing", emit.path));
            }
            let mut box_ = &self.root;
            let mut scope: Vec<VarId> = (0..box_.new_vars.len()).collect();
            let mut start = box_.new_vars.len();
            for (depth, c) in emit.path.iter().enumerate() {
                let child = box_.children.get(*c).ok_or_else(|| {
                    format!(
                        "Query.emits: path {:?} has no child {c} at depth {depth}",
                        emit.path
                    )
                })?;
                start += box_.children[..*c]
                    .iter()
                    .map(|x| subtree_var_count(&x.box_))
                    .sum::<usize>();
                box_ = &child.box_;
                scope.extend(start..start + box_.new_vars.len());
                start += box_.new_vars.len();
            }
            for e in &emit.project {
                validate_expr(e, &scope, kinds, box_)?;
            }
        }
        Ok(())
    }

    /// Which [`Self::emits`] entry addresses `path`, if any.
    #[must_use]
    pub fn emit_index_of(&self, path: &[ChildRef]) -> Option<usize> {
        self.emits.iter().position(|e| e.path == path)
    }
}

fn collect_box_vars(box_: &Box, depth: usize, out: &mut Vec<VarInfo>) {
    for v in &box_.new_vars {
        out.push(VarInfo {
            id: out.len(),
            kind: v.kind.clone(),
            box_depth: depth,
        });
    }
    for child in &box_.children {
        collect_box_vars(&child.box_, depth + 1, out);
    }
}

fn validate_box(
    box_: &Box,
    ancestor_scope: &[VarId],
    kinds: &[VarKind],
    next_id: &mut VarId,
    own_aggs: Option<&[Agg]>,
) -> Result<(), String> {
    let mut scope = ancestor_scope.to_vec();
    for _ in &box_.new_vars {
        scope.push(*next_id);
        *next_id += 1;
    }

    for f in &box_.filters {
        validate_filter(f, &scope, kinds, box_)?;
    }
    if let Some(aggs) = own_aggs {
        for agg in aggs {
            validate_agg(agg, &scope, kinds, box_)?;
        }
    }
    for child in &box_.children {
        validate_box(&child.box_, &scope, kinds, next_id, Some(&child.aggs))?;
    }
    Ok(())
}

fn check_in_scope(id: VarId, scope: &[VarId], kinds: &[VarKind], ctx: &str) -> Result<(), String> {
    if id >= kinds.len() {
        return Err(format!("{ctx}: var {id} out of range"));
    }
    if !scope.contains(&id) {
        return Err(format!("{ctx}: var {id} not in scope"));
    }
    Ok(())
}

fn check_kind(
    id: VarId,
    want: &VarKind,
    scope: &[VarId],
    kinds: &[VarKind],
    ctx: &str,
) -> Result<(), String> {
    check_in_scope(id, scope, kinds, ctx)?;
    if &kinds[id] != want {
        return Err(format!(
            "{ctx}: var {id} expected kind {want:?}, found {:?}",
            kinds[id]
        ));
    }
    Ok(())
}

/// Resolve `ChildAgg(child, agg_idx)` against `owner`'s children, or report why it does not.
fn resolve_child_agg<'a>(
    owner: &'a Box,
    child: ChildRef,
    agg_idx: usize,
    ctx: &str,
) -> Result<&'a Agg, String> {
    let cb = owner.children.get(child).ok_or_else(|| {
        format!(
            "{ctx}: child ref {child} out of range (num_children={})",
            owner.children.len()
        )
    })?;
    cb.aggs.get(agg_idx).ok_or_else(|| {
        format!(
            "{ctx}: agg index {agg_idx} out of range (child {child} has {} aggs)",
            cb.aggs.len()
        )
    })
}

/// Whether `e` yields a [`Value::List`](super::eval::Value::List).
fn is_list_valued(e: &Expr, owner: &Box) -> bool {
    match e {
        Expr::ChildAgg(c, i) => owner
            .children
            .get(*c)
            .and_then(|cb| cb.aggs.get(*i))
            .is_some_and(|a| matches!(a, Agg::Sequence { .. })),
        _ => false,
    }
}

/// A `List` is only ever an output column: it has no group-key representation, and is neither a
/// range bound nor a comparison or aggregate operand.
fn reject_list(e: &Expr, owner: &Box, ctx: &str) -> Result<(), String> {
    if is_list_valued(e, owner) {
        return Err(format!(
            "{ctx}: an Agg::Sequence fold is List-valued and can only be projected"
        ));
    }
    Ok(())
}

fn validate_filter(
    f: &Filter,
    scope: &[VarId],
    kinds: &[VarKind],
    owner: &Box,
) -> Result<(), String> {
    match f {
        Filter::E2O { event, object, .. } => {
            check_kind(*event, &VarKind::Event, scope, kinds, "E2O.event")?;
            check_kind(*object, &VarKind::Object, scope, kinds, "E2O.object")
        }
        Filter::O2O { from, to, .. } => {
            check_kind(*from, &VarKind::Object, scope, kinds, "O2O.from")?;
            check_kind(*to, &VarKind::Object, scope, kinds, "O2O.to")
        }
        Filter::TimeBetweenEvents { from, to, .. } => {
            check_kind(
                *from,
                &VarKind::Event,
                scope,
                kinds,
                "TimeBetweenEvents.from",
            )?;
            check_kind(*to, &VarKind::Event, scope, kinds, "TimeBetweenEvents.to")
        }
        Filter::EventAttr { event, .. } => {
            check_kind(*event, &VarKind::Event, scope, kinds, "EventAttr.event")
        }
        Filter::ObjectAttr { object, at, .. } => {
            check_kind(*object, &VarKind::Object, scope, kinds, "ObjectAttr.object")?;
            if let FilterAt::AtEvent(ev) = at {
                check_kind(*ev, &VarKind::Event, scope, kinds, "ObjectAttr.at.AtEvent")?;
            }
            Ok(())
        }
        Filter::AggRange {
            child, agg_idx, ..
        } => {
            let agg = resolve_child_agg(owner, *child, *agg_idx, "AggRange")?;
            if matches!(agg, Agg::Sequence { .. }) {
                return Err(
                    "AggRange: an Agg::Sequence fold is List-valued and has no numeric range"
                        .to_string(),
                );
            }
            Ok(())
        }
        Filter::Compare { left, right, .. } => {
            validate_expr(left, scope, kinds, owner)?;
            validate_expr(right, scope, kinds, owner)?;
            reject_list(left, owner, "Compare.left")?;
            reject_list(right, owner, "Compare.right")
        }
        Filter::Not(inner) => validate_filter(inner, scope, kinds, owner),
        Filter::Or(fs) => fs
            .iter()
            .try_for_each(|f| validate_filter(f, scope, kinds, owner)),
    }
}

fn validate_agg(agg: &Agg, scope: &[VarId], kinds: &[VarKind], owner: &Box) -> Result<(), String> {
    match agg {
        Agg::Count => Ok(()),
        Agg::CountDistinct(e) | Agg::Min(e) | Agg::Max(e) | Agg::Sum(e) | Agg::Avg(e) => {
            validate_expr(e, scope, kinds, owner)?;
            reject_list(e, owner, "Agg operand")
        }
        Agg::Sequence { of, by } => {
            validate_expr(of, scope, kinds, owner)?;
            reject_list(of, owner, "Agg::Sequence.of")?;
            for (e, _) in by {
                validate_expr(e, scope, kinds, owner)?;
                reject_list(e, owner, "Agg::Sequence.by")?;
            }
            Ok(())
        }
    }
}

fn validate_expr(e: &Expr, scope: &[VarId], kinds: &[VarKind], owner: &Box) -> Result<(), String> {
    match e {
        Expr::Id(v) => check_in_scope(*v, scope, kinds, "Expr::Id"),
        Expr::Type(v) => check_in_scope(*v, scope, kinds, "Expr::Type"),
        Expr::Time(v) => check_kind(*v, &VarKind::Event, scope, kinds, "Expr::Time"),
        Expr::Attr { var, at, .. } => {
            check_in_scope(*var, scope, kinds, "Expr::Attr.var")?;
            if let OutAt::AtEvent(ev) = at {
                check_kind(*ev, &VarKind::Event, scope, kinds, "Expr::Attr.at.AtEvent")?;
            }
            Ok(())
        }
        Expr::ChildAgg(c, i) => resolve_child_agg(owner, *c, *i, "Expr::ChildAgg").map(|_| ()),
        Expr::Satisfies(f) => validate_filter(f, scope, kinds, owner),
    }
}

/// Every ancestor variable (id below `own_start`) that `box_`'s subtree reads.
fn ancestor_vars_read(box_: &Box, own_start: VarId, out: &mut BTreeSet<VarId>) {
    fn walk_expr(e: &Expr, own_start: VarId, out: &mut BTreeSet<VarId>) {
        match e {
            Expr::Id(v) | Expr::Type(v) | Expr::Time(v) => {
                if *v < own_start {
                    out.insert(*v);
                }
            }
            Expr::Attr { var, at, .. } => {
                if *var < own_start {
                    out.insert(*var);
                }
                if let OutAt::AtEvent(ev) = at {
                    if *ev < own_start {
                        out.insert(*ev);
                    }
                }
            }
            Expr::ChildAgg(..) => {}
            Expr::Satisfies(f) => walk_filter(f, own_start, out),
        }
    }
    fn walk_filter(f: &Filter, own_start: VarId, out: &mut BTreeSet<VarId>) {
        let mut add = |v: VarId| {
            if v < own_start {
                out.insert(v);
            }
        };
        match f {
            Filter::E2O { event, object, .. } => {
                add(*event);
                add(*object);
            }
            Filter::O2O { from, to, .. } | Filter::TimeBetweenEvents { from, to, .. } => {
                add(*from);
                add(*to);
            }
            Filter::EventAttr { event, .. } => add(*event),
            Filter::ObjectAttr { object, at, .. } => {
                add(*object);
                if let FilterAt::AtEvent(ev) = at {
                    add(*ev);
                }
            }
            Filter::AggRange { .. } => {}
            Filter::Compare { left, right, .. } => {
                walk_expr(left, own_start, out);
                walk_expr(right, own_start, out);
            }
            Filter::Not(inner) => walk_filter(inner, own_start, out),
            Filter::Or(fs) => fs.iter().for_each(|f| walk_filter(f, own_start, out)),
        }
    }
    fn walk_agg(a: &Agg, own_start: VarId, out: &mut BTreeSet<VarId>) {
        match a {
            Agg::Count => {}
            Agg::CountDistinct(e) | Agg::Min(e) | Agg::Max(e) | Agg::Sum(e) | Agg::Avg(e) => {
                walk_expr(e, own_start, out);
            }
            Agg::Sequence { of, by } => {
                walk_expr(of, own_start, out);
                by.iter().for_each(|(e, _)| walk_expr(e, own_start, out));
            }
        }
    }
    for f in &box_.filters {
        walk_filter(f, own_start, out);
    }
    for child in &box_.children {
        for a in &child.aggs {
            walk_agg(a, own_start, out);
        }
        ancestor_vars_read(&child.box_, own_start, out);
    }
}

/// The global [`VarId`] of `children[idx]`'s first own variable.
fn child_start_id(root: &Box, idx: usize) -> VarId {
    root.new_vars.len()
        + root.children[..idx]
            .iter()
            .map(|c| subtree_var_count(&c.box_))
            .sum::<usize>()
}

fn subtree_var_count(box_: &Box) -> usize {
    box_.new_vars.len()
        + box_
            .children
            .iter()
            .map(|c| subtree_var_count(&c.box_))
            .sum::<usize>()
}

/// A child fold may be a group key only when every root variable it reads is itself keyed by
/// `Expr::Id`. Otherwise the fold varies within a group and grouping by it splits what should be
/// one row.
fn check_group_key_determinacy(spec: &AggSpec, root: &Box) -> Result<(), String> {
    let keyed_ids: BTreeSet<VarId> = spec
        .group_by
        .iter()
        .filter_map(|e| match e {
            Expr::Id(v) => Some(*v),
            _ => None,
        })
        .collect();
    for e in &spec.group_by {
        let Expr::ChildAgg(c, _) = e else { continue };
        let mut read = BTreeSet::new();
        ancestor_vars_read(&root.children[*c].box_, child_start_id(root, *c), &mut read);
        if let Some(v) = read.difference(&keyed_ids).next() {
            return Err(format!(
                "AggSpec.group_by: child {c}'s fold depends on var {v}, which is not a group key, \
                 so the fold is not determined by the grouping"
            ));
        }
    }
    Ok(())
}

fn validate_output(
    o: &Output,
    scope: &[VarId],
    kinds: &[VarKind],
    root: &Box,
) -> Result<(), String> {
    match o {
        Output::Rows(r) => {
            for e in &r.project {
                validate_expr(e, scope, kinds, root)?;
            }
            for (e, _) in &r.order_by {
                validate_expr(e, scope, kinds, root)?;
            }
            Ok(())
        }
        Output::Aggregate(a) => {
            for e in &a.group_by {
                validate_expr(e, scope, kinds, root)?;
                reject_list(e, root, "AggSpec.group_by")?;
            }
            for agg in &a.aggregates {
                validate_agg(agg, scope, kinds, root)?;
            }
            for (idx, _, _) in &a.having {
                let agg =
                    a.aggregates
                        .get(*idx)
                        .ok_or_else(|| {
                            format!(
                                "AggSpec.having: index {idx} out of range (aggregates={})",
                                a.aggregates.len()
                            )
                        })?;
                if matches!(agg, Agg::Sequence { .. }) {
                    return Err(format!(
                        "AggSpec.having: aggregate {idx} is an Agg::Sequence, which has no numeric \
                         range"
                    ));
                }
            }
            let n = a.group_by.len() + a.aggregates.len();
            for (idx, _) in &a.order_by {
                if *idx >= n {
                    return Err(format!(
                        "AggSpec.order_by: index {idx} out of range (n={n})"
                    ));
                }
            }
            check_group_key_determinacy(a, root)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) "orders and their directly-related events"
    fn query_bindings() -> Query {
        Query {
            root: Box {
                new_vars: vec![
                    VarDecl {
                        kind: VarKind::Object,
                        types: TypeConstraint::OneOf(vec!["order".to_string()]),
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
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(1), Expr::Id(0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    /// (b) `type_counts`
    fn query_type_counts() -> Query {
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

    fn items_child(aggs: Vec<Agg>) -> ChildBox {
        ChildBox {
            box_: Box {
                new_vars: vec![VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec!["item".to_string()]),
                }],
                filters: vec![Filter::O2O {
                    from: 0,
                    to: 1,
                    qualifier: None,
                }],
                children: vec![],
            },
            aggs,
        }
    }

    /// (d) >=3 items
    fn query_at_least_3_items() -> Query {
        Query {
            root: Box {
                new_vars: vec![VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec!["order".to_string()]),
                }],
                filters: vec![Filter::AggRange {
                    child: 0,
                    agg_idx: 0,
                    min: Some(3.0),
                    max: None,
                }],
                children: vec![items_child(vec![Agg::Count])],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    /// (e) orders + item count, most first
    fn query_child_scalar_output() -> Query {
        Query {
            root: Box {
                new_vars: vec![VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec!["order".to_string()]),
                }],
                filters: vec![],
                children: vec![items_child(vec![Agg::Count])],
            },
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                order_by: vec![(Expr::ChildAgg(0, 0), Dir::Desc)],
                limit: None,
            }),
            emits: Vec::new(),
        }
    }

    #[test]
    fn examples_validate_ok() {
        assert_eq!(query_bindings().validate(), Ok(()));
        assert_eq!(query_type_counts().validate(), Ok(()));
        assert_eq!(query_at_least_3_items().validate(), Ok(()));
        assert_eq!(query_child_scalar_output().validate(), Ok(()));
    }

    #[test]
    fn malformed_query_wrong_kind_is_rejected() {
        // v0 is an Object var, v1 is an Event var; E2O.event/object are swapped.
        let mut q = query_bindings();
        q.root.filters = vec![Filter::E2O {
            event: 0,
            object: 1,
            qualifier: None,
        }];
        assert!(q.validate().is_err());
    }

    #[test]
    fn malformed_query_out_of_range_var_is_rejected() {
        let mut q = query_bindings();
        q.root.filters = vec![Filter::E2O {
            event: 5,
            object: 0,
            qualifier: None,
        }];
        assert!(q.validate().is_err());
    }

    #[test]
    fn out_of_range_agg_index_is_rejected() {
        let mut q = query_child_scalar_output();
        q.output = Output::Rows(RowsSpec {
            project: vec![Expr::ChildAgg(0, 1)],
            order_by: vec![],
            limit: None,
        });
        assert!(q.validate().is_err());
    }

    /// Without this, `eval` reaches an unreachable arm on a query that validated.
    #[test]
    fn a_sequence_fold_is_rejected_as_a_group_key() {
        let q = Query {
            root: Box {
                new_vars: vec![VarDecl {
                    kind: VarKind::Object,
                    types: TypeConstraint::OneOf(vec!["order".to_string()]),
                }],
                filters: vec![],
                children: vec![items_child(vec![Agg::Sequence {
                    of: Expr::Id(1),
                    by: vec![],
                }])],
            },
            output: Output::Aggregate(AggSpec {
                group_by: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                aggregates: vec![Agg::Count],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        assert!(q.validate().is_err());
    }

    #[test]
    fn a_sequence_fold_is_rejected_as_an_agg_operand_and_a_range() {
        let child = items_child(vec![Agg::Sequence {
            of: Expr::Id(1),
            by: vec![],
        }]);
        let mut q = query_child_scalar_output();
        q.root.children = vec![child];
        q.output = Output::Aggregate(AggSpec {
            group_by: vec![Expr::Id(0)],
            aggregates: vec![Agg::Sum(Expr::ChildAgg(0, 0))],
            having: vec![],
            order_by: vec![],
            limit: None,
        });
        assert!(q.validate().is_err());

        let mut q2 = query_at_least_3_items();
        q2.root.children[0].aggs = vec![Agg::Sequence {
            of: Expr::Id(1),
            by: vec![],
        }];
        assert!(q2.validate().is_err());
    }

    /// One row per (customer, item-count) pair instead of one per customer: the fold reads `o2`,
    /// which the grouping throws away.
    #[test]
    fn a_child_fold_not_determined_by_the_group_keys_is_rejected() {
        let q = Query {
            root: Box {
                new_vars: vec![
                    VarDecl {
                        kind: VarKind::Object,
                        types: TypeConstraint::OneOf(vec!["customer".to_string()]),
                    },
                    VarDecl {
                        kind: VarKind::Object,
                        types: TypeConstraint::OneOf(vec!["order".to_string()]),
                    },
                ],
                filters: vec![Filter::O2O {
                    from: 0,
                    to: 1,
                    qualifier: None,
                }],
                children: vec![ChildBox {
                    box_: Box {
                        new_vars: vec![VarDecl {
                            kind: VarKind::Object,
                            types: TypeConstraint::OneOf(vec!["item".to_string()]),
                        }],
                        // Reads o2 (var 1), which is not a group key below.
                        filters: vec![Filter::O2O {
                            from: 1,
                            to: 2,
                            qualifier: None,
                        }],
                        children: vec![],
                    },
                    aggs: vec![Agg::Count],
                }],
            },
            output: Output::Aggregate(AggSpec {
                group_by: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                aggregates: vec![Agg::Count],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        assert!(q.validate().is_err());
    }

    #[test]
    fn having_indexes_aggregates_only() {
        let mut q = query_type_counts();
        let Output::Aggregate(spec) = &mut q.output else {
            unreachable!()
        };
        spec.having = vec![(0, Some(50.0), None)];
        assert_eq!(q.validate(), Ok(()));

        let Output::Aggregate(spec) = &mut q.output else {
            unreachable!()
        };
        spec.having = vec![(1, Some(50.0), None)];
        assert!(q.validate().is_err());
    }

    #[test]
    fn collect_vars_matches_expected_ids_and_depths() {
        let q = query_at_least_3_items();
        let vars = q.collect_vars();
        assert_eq!(vars.len(), 2);
        assert_eq!(vars[0].id, 0);
        assert_eq!(vars[0].kind, VarKind::Object);
        assert_eq!(vars[0].box_depth, 0);
        assert_eq!(vars[1].id, 1);
        assert_eq!(vars[1].box_depth, 1);
    }

    #[test]
    fn serde_round_trip_type_counts() {
        let q = query_type_counts();
        let json = serde_json::to_string(&q).expect("serialize");
        let back: Query = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(q, back);
    }

    #[test]
    fn a_nested_relational_filter_is_not_liftable_to_an_enumeration_step() {
        let or = Filter::Or(vec![
            Filter::E2O {
                event: 1,
                object: 0,
                qualifier: None,
            },
            Filter::E2O {
                event: 1,
                object: 0,
                qualifier: Some("q".to_string()),
            },
        ]);
        assert!(or.as_relation().is_none());
        assert!(Filter::Not(std::boxed::Box::new(or)).as_relation().is_none());
    }
}
