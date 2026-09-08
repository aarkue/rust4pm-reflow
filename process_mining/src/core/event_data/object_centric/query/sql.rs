//! Pure translation of a [`Query`] to one SQL statement over the EAV schema (see
//! `ocel_sql::duckdb::schema::tables`).
//!
//! Feature-independent: no `duckdb` dependency. Execution lives on `DuckDbLinkedOCEL`
//! (`ocel_sql::duckdb::schema::db_linked::run_query`), which is feature-gated.
//!
//! Correlated `ChildBox`es lower to scalar subqueries. [`VarId`] aliases (`v{id}`) are global
//! across the whole box tree, so a subquery's `WHERE` references an outer `v{id}` directly
//! without re-tabling it.
//!
//! Only a top-level `Filter::E2O`/`O2O` becomes a `FROM`-list join. Under a `Filter::Not` or a
//! `Filter::Or` there is no join to push, so those render as `EXISTS` semi-joins instead, and
//! every leaf that could be `NULL` is wrapped in `COALESCE(.., FALSE)` so SQL's three-valued logic
//! matches the evaluator's two-valued one.
//!
//! A binding is a variable assignment, not a relationship-junction row: several junction rows can
//! share one assignment (multiple qualifiers linking the same pair). The root box's binding join
//! is therefore wrapped in a `SELECT DISTINCT <own var id cols>` subquery (see
//! [`box_from_where_distinct`]) before projecting/grouping. A `ChildBox`'s own join is not wrapped
//! that way, its `Agg::Count` uses `COUNT(DISTINCT <child's own var id cols>)` instead.

use std::collections::HashMap;

use super::{
    Agg, Box, ChildBox, ChildRef, CmpOp, Expr, Filter, FilterAt, Output, Query, TypeConstraint,
    ValueFilter, VarId, VarKind,
};
use crate::core::event_data::object_centric::ocel_struct::OCELAttributeType;

/// Declared value type per object-attribute name, so the EAV `value` column (always `VARCHAR`)
/// can be cast back to a typed SQL value.
///
/// Without it `Min`/`Max` and `ORDER BY` compare lexicographically here but numerically on the
/// in-memory backends. A name absent from the map stays text.
pub type ObjectAttrTypes = HashMap<String, OCELAttributeType>;

/// The `value` selector for an object attribute: cast to its declared type when known.
fn obj_attr_value_sql(attr_types: &ObjectAttrTypes, name: &str) -> &'static str {
    match attr_types.get(name) {
        Some(OCELAttributeType::Integer) => "TRY_CAST(value AS BIGINT)",
        Some(OCELAttributeType::Float) => "TRY_CAST(value AS DOUBLE)",
        Some(OCELAttributeType::Boolean) => "TRY_CAST(value AS BOOLEAN)",
        Some(OCELAttributeType::Time) => "TRY_CAST(value AS TIMESTAMPTZ)",
        _ => "value",
    }
}

/// One SQL statement plus its `?`-positional parameters, in emission (left-to-right,
/// textual) order.
#[derive(Debug, Clone, PartialEq)]
pub struct SqlQuery {
    /// The SQL text, with `?` placeholders.
    pub sql: String,
    /// Bind values, one per `?`, in the order they appear in `sql`.
    pub params: Vec<SqlParam>,
}

/// A single bind value for a [`SqlQuery`] placeholder.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlParam {
    /// A string (also used for booleans and RFC3339-formatted timestamps).
    Str(String),
    /// A signed integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// A boolean.
    Bool(bool),
}

/// Translate `query` to a single `SELECT` (or, for `Output::Aggregate`, a grouped
/// `SELECT ... GROUP BY`), correlated `ChildBox`es lowered to scalar subqueries.
pub fn to_sql(query: &Query, attr_types: &ObjectAttrTypes) -> Result<SqlQuery, String> {
    query.validate()?;
    let kinds: Vec<VarKind> = query.collect_vars().into_iter().map(|v| v.kind).collect();
    let children_base = query.root.new_vars.len();

    let (from, where_conds, mut where_params) =
        box_from_where_distinct(&query.root, 0, &kinds, attr_types)?;

    match &query.output {
        Output::Rows(spec) => {
            let mut select_params = Vec::new();
            let select_cols: Vec<String> = spec
                .project
                .iter()
                .map(|e| {
                    expr_sql(
                        e,
                        &kinds,
                        attr_types,
                        &query.root.children,
                        children_base,
                        &mut select_params,
                    )
                })
                .collect::<Result<_, String>>()?;

            let mut order_params = Vec::new();
            let order_parts: Vec<String> = spec
                .order_by
                .iter()
                .map(|(e, dir)| {
                    let e_sql = expr_sql(
                        e,
                        &kinds,
                        attr_types,
                        &query.root.children,
                        children_base,
                        &mut order_params,
                    )?;
                    Ok(format!("{e_sql} {}", dir_sql(dir)))
                })
                .collect::<Result<_, String>>()?;

            let mut limit_params = Vec::new();
            let limit_sql = spec.limit.map(|n| {
                limit_params.push(SqlParam::Int(n as i64));
                " LIMIT ?".to_string()
            });

            let mut sql = format!("SELECT {} FROM {}", select_cols.join(", "), from.join(", "));
            if !where_conds.is_empty() {
                sql.push_str(" WHERE ");
                sql.push_str(&where_conds.join(" AND "));
            }
            if !order_parts.is_empty() {
                sql.push_str(" ORDER BY ");
                sql.push_str(&order_parts.join(", "));
            }
            if let Some(l) = &limit_sql {
                sql.push_str(l);
            }

            let mut params = select_params;
            params.append(&mut where_params);
            params.append(&mut order_params);
            params.append(&mut limit_params);

            Ok(SqlQuery { sql, params })
        }
        Output::Aggregate(spec) => {
            let mut group_params = Vec::new();
            let group_cols: Vec<String> = spec
                .group_by
                .iter()
                .map(|e| {
                    expr_sql(
                        e,
                        &kinds,
                        attr_types,
                        &query.root.children,
                        children_base,
                        &mut group_params,
                    )
                })
                .collect::<Result<_, String>>()?;

            let mut agg_params = Vec::new();
            let agg_cols: Vec<String> = spec
                .aggregates
                .iter()
                .map(|a| {
                    // `box_from_where_distinct` already yields distinct bindings, so `Count`
                    // needs no `DISTINCT` of its own.
                    agg_expr_sql(
                        a,
                        &kinds,
                        attr_types,
                        &query.root.children,
                        children_base,
                        &[],
                        &mut agg_params,
                    )
                })
                .collect::<Result<_, String>>()?;

            let alias_aggs = !spec.having.is_empty();
            let select_list = group_cols
                .iter()
                .cloned()
                .chain(agg_cols.iter().enumerate().map(|(i, c)| {
                    if alias_aggs {
                        format!("{c} AS a{i}")
                    } else {
                        c.clone()
                    }
                }))
                .collect::<Vec<_>>()
                .join(", ");
            let mut sql = format!("SELECT {select_list} FROM {}", from.join(", "));
            if !where_conds.is_empty() {
                sql.push_str(" WHERE ");
                sql.push_str(&where_conds.join(" AND "));
            }
            // Positional GROUP BY/ORDER BY avoids re-emitting (and re-parameterizing) the
            // group/agg expressions.
            if !group_cols.is_empty() {
                let positions: Vec<String> =
                    (1..=group_cols.len()).map(|i| i.to_string()).collect();
                sql.push_str(" GROUP BY ");
                sql.push_str(&positions.join(", "));
            }
            let mut having_params = Vec::new();
            if alias_aggs {
                let mut conds = Vec::new();
                for (idx, min, max) in &spec.having {
                    for (bound, op) in [(min, ">="), (max, "<=")] {
                        let Some(m) = bound else { continue };
                        conds.push(format!("a{idx} {op} ?"));
                        having_params.push(SqlParam::Float(*m));
                    }
                }
                if !conds.is_empty() {
                    sql.push_str(" HAVING ");
                    sql.push_str(&conds.join(" AND "));
                }
            }
            if !spec.order_by.is_empty() {
                let order_parts: Vec<String> = spec
                    .order_by
                    .iter()
                    .map(|(idx, dir)| format!("{} {}", idx + 1, dir_sql(dir)))
                    .collect();
                sql.push_str(" ORDER BY ");
                sql.push_str(&order_parts.join(", "));
            }
            let mut limit_params = Vec::new();
            if let Some(n) = spec.limit {
                sql.push_str(" LIMIT ?");
                limit_params.push(SqlParam::Int(n as i64));
            }

            let mut params = group_params;
            params.append(&mut agg_params);
            params.append(&mut where_params);
            params.append(&mut having_params);
            params.append(&mut limit_params);

            Ok(SqlQuery { sql, params })
        }
    }
}

fn dir_sql(dir: &super::Dir) -> &'static str {
    match dir {
        super::Dir::Asc => "ASC",
        super::Dir::Desc => "DESC",
    }
}

/// SQL-quote an identifier by doubling embedded double-quotes.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// `(from_tables, where_conditions, params)` for a box's translated `FROM`/`WHERE`.
type FromWhere = (Vec<String>, Vec<String>, Vec<SqlParam>);

/// Total variable count of `box_`'s subtree, its own `new_vars` plus every descendant box's.
fn subtree_var_count(box_: &Box) -> usize {
    box_.new_vars.len()
        + box_
            .children
            .iter()
            .map(|c| subtree_var_count(&c.box_))
            .sum::<usize>()
}

/// Global `VarId` at which `own_children[idx]`'s own vars start, given `children_base`
/// (the id right after the owning box's own `new_vars`).
fn child_start_id(children_base: VarId, own_children: &[ChildBox], idx: usize) -> VarId {
    children_base
        + own_children[..idx]
            .iter()
            .map(|c| subtree_var_count(&c.box_))
            .sum::<VarId>()
}

/// Build the `FROM` table list and `WHERE` conditions (with their params, in textual order) for
/// `box_`'s own vars and filters. `start_id` is the global `VarId` of `box_.new_vars[0]`.
fn box_from_where(
    box_: &Box,
    start_id: VarId,
    kinds: &[VarKind],
    attr_types: &ObjectAttrTypes,
) -> Result<FromWhere, String> {
    let mut from = Vec::new();
    let mut where_conds = Vec::new();
    let mut params = Vec::new();

    for (i, decl) in box_.new_vars.iter().enumerate() {
        let id = start_id + i;
        let table = match decl.kind {
            VarKind::Event => "events",
            VarKind::Object => "objects",
        };
        from.push(format!("{table} v{id}"));
        if let TypeConstraint::OneOf(tys) = &decl.types {
            let placeholders = std::iter::repeat_n("?", tys.len())
                .collect::<Vec<_>>()
                .join(",");
            where_conds.push(format!("v{id}.ocel_type IN ({placeholders})"));
            params.extend(tys.iter().map(|t| SqlParam::Str(t.clone())));
        }
    }

    let children_base = start_id + box_.new_vars.len();
    for (k, f) in box_.filters.iter().enumerate() {
        // Only a top-level relational filter can join its junction table into `FROM`. Anything
        // else, a relational filter nested inside `Not`/`Or` included, renders as a boolean.
        match f {
            Filter::E2O {
                event,
                object,
                qualifier,
            } => {
                let r = format!("r{k}");
                where_conds.push(format!(
                    "{r}.event_id = v{event}.id AND {r}.object_id = v{object}.id"
                ));
                if let Some(q) = qualifier {
                    where_conds.push(format!("{r}.qualifier = ?"));
                    params.push(SqlParam::Str(q.clone()));
                }
                from.push(format!("e2o {r}"));
            }
            Filter::O2O {
                from: f_var,
                to,
                qualifier,
            } => {
                let r = format!("r{k}");
                where_conds.push(format!(
                    "{r}.source_id = v{f_var}.id AND {r}.target_id = v{to}.id"
                ));
                if let Some(q) = qualifier {
                    where_conds.push(format!("{r}.qualifier = ?"));
                    params.push(SqlParam::Str(q.clone()));
                }
                from.push(format!("o2o {r}"));
            }
            Filter::TimeBetweenEvents {
                from: f_var,
                to,
                min_seconds,
                max_seconds,
            } => {
                // Two independent conditions at the top level, where they are already conjoined.
                let expr = time_between_expr(*f_var, *to);
                if let Some(min) = min_seconds {
                    where_conds.push(format!("{expr} >= ?"));
                    params.push(SqlParam::Float(*min));
                }
                if let Some(max) = max_seconds {
                    where_conds.push(format!("{expr} <= ?"));
                    params.push(SqlParam::Float(*max));
                }
            }
            other => {
                let mut ctx = FilterCtx {
                    kinds,
                    attr_types,
                    own_children: &box_.children,
                    children_base,
                    next_alias: &mut 0,
                };
                where_conds.push(filter_cond_sql(other, &mut ctx, &mut params)?);
            }
        }
    }

    Ok((from, where_conds, params))
}

/// The elapsed-seconds expression a [`Filter::TimeBetweenEvents`] compares.
fn time_between_expr(from: VarId, to: VarId) -> String {
    format!(r#"(date_diff('millisecond', v{from}."time", v{to}."time") / 1000.0)"#)
}

/// Force a possibly-`NULL` boolean to `FALSE`, so `NOT`/`OR` compose over the same two-valued
/// logic the in-memory evaluator uses.
fn total_bool(cond: &str) -> String {
    format!("COALESCE({cond}, FALSE)")
}

/// What [`filter_cond_sql`] needs to resolve a filter's operands.
struct FilterCtx<'a> {
    kinds: &'a [VarKind],
    attr_types: &'a ObjectAttrTypes,
    own_children: &'a [ChildBox],
    children_base: VarId,
    /// Distinguishes the junction aliases of nested `EXISTS` subqueries.
    next_alias: &'a mut usize,
}

/// Any [`Filter`] as one boolean SQL expression that is never `NULL`. The relational variants
/// render as `EXISTS` semi-joins here rather than as `FROM`-list joins.
fn filter_cond_sql(
    f: &Filter,
    ctx: &mut FilterCtx<'_>,
    params: &mut Vec<SqlParam>,
) -> Result<String, String> {
    match f {
        Filter::E2O {
            event,
            object,
            qualifier,
        } => {
            *ctx.next_alias += 1;
            let r = format!("x{}", ctx.next_alias);
            let mut conds = vec![format!(
                "{r}.event_id = v{event}.id AND {r}.object_id = v{object}.id"
            )];
            if let Some(q) = qualifier {
                conds.push(format!("{r}.qualifier = ?"));
                params.push(SqlParam::Str(q.clone()));
            }
            Ok(format!(
                "EXISTS (SELECT 1 FROM e2o {r} WHERE {})",
                conds.join(" AND ")
            ))
        }
        Filter::O2O {
            from,
            to,
            qualifier,
        } => {
            *ctx.next_alias += 1;
            let r = format!("x{}", ctx.next_alias);
            let mut conds = vec![format!(
                "{r}.source_id = v{from}.id AND {r}.target_id = v{to}.id"
            )];
            if let Some(q) = qualifier {
                conds.push(format!("{r}.qualifier = ?"));
                params.push(SqlParam::Str(q.clone()));
            }
            Ok(format!(
                "EXISTS (SELECT 1 FROM o2o {r} WHERE {})",
                conds.join(" AND ")
            ))
        }
        Filter::TimeBetweenEvents {
            from,
            to,
            min_seconds,
            max_seconds,
        } => {
            let expr = time_between_expr(*from, *to);
            let mut conds = Vec::new();
            if let Some(min) = min_seconds {
                conds.push(format!("{expr} >= ?"));
                params.push(SqlParam::Float(*min));
            }
            if let Some(max) = max_seconds {
                conds.push(format!("{expr} <= ?"));
                params.push(SqlParam::Float(*max));
            }
            Ok(if conds.is_empty() {
                "TRUE".to_string()
            } else {
                total_bool(&format!("({})", conds.join(" AND ")))
            })
        }
        Filter::EventAttr { .. } | Filter::ObjectAttr { .. } => {
            Ok(attr_filter_sql(f, params).expect("matched an attribute filter"))
        }
        Filter::AggRange {
            child,
            agg_idx,
            min,
            max,
        } => {
            let mut conds = Vec::new();
            for (bound, op) in [(min, ">="), (max, "<=")] {
                let Some(m) = bound else { continue };
                let sub = child_agg_sql(*child, *agg_idx, ctx, params)?;
                conds.push(format!("{sub} {op} ?"));
                params.push(SqlParam::Float(*m));
            }
            Ok(if conds.is_empty() {
                "TRUE".to_string()
            } else {
                total_bool(&format!("({})", conds.join(" AND ")))
            })
        }
        Filter::Compare { left, op, right } => {
            let l = expr_sql(
                left,
                ctx.kinds,
                ctx.attr_types,
                ctx.own_children,
                ctx.children_base,
                params,
            )?;
            let r = expr_sql(
                right,
                ctx.kinds,
                ctx.attr_types,
                ctx.own_children,
                ctx.children_base,
                params,
            )?;
            Ok(total_bool(&format!("({l} {} {r})", cmp_sql(*op))))
        }
        // `COALESCE` first, so absence negates to true rather than staying `NULL`.
        Filter::Not(inner) => {
            let c = filter_cond_sql(inner, ctx, params)?;
            Ok(format!("NOT {}", total_bool(&c)))
        }
        Filter::Or(fs) => {
            if fs.is_empty() {
                return Ok("FALSE".to_string());
            }
            let parts = fs
                .iter()
                .map(|f| filter_cond_sql(f, ctx, params))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(format!("({})", parts.join(" OR ")))
        }
    }
}

fn cmp_sql(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "=",
        CmpOp::Ne => "<>",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

/// The scalar subquery for `ChildAgg(child, agg_idx)` against the box `ctx` describes.
fn child_agg_sql(
    child: ChildRef,
    agg_idx: usize,
    ctx: &mut FilterCtx<'_>,
    params: &mut Vec<SqlParam>,
) -> Result<String, String> {
    let cb = ctx
        .own_children
        .get(child)
        .ok_or_else(|| format!("child ref {child} out of range"))?;
    let start = child_start_id(ctx.children_base, ctx.own_children, child);
    child_subquery_sql(cb, agg_idx, start, ctx.kinds, ctx.attr_types, params)
}

/// Wrap `box_`'s own binding join in a `SELECT DISTINCT <own var id cols>` subquery, then
/// re-table each var by joining back to its base table on id. One row per distinct binding,
/// keeping the same `v{id}` aliases. Root box only, a `ChildBox` uses `COUNT(DISTINCT ...)`.
fn box_from_where_distinct(
    box_: &Box,
    start_id: VarId,
    kinds: &[VarKind],
    attr_types: &ObjectAttrTypes,
) -> Result<FromWhere, String> {
    let (inner_from, inner_where, params) = box_from_where(box_, start_id, kinds, attr_types)?;
    let var_ids: Vec<VarId> = (start_id..start_id + box_.new_vars.len()).collect();
    let select_list = var_ids
        .iter()
        .map(|v| format!("v{v}.id AS b{v}"))
        .collect::<Vec<_>>()
        .join(", ");

    let mut inner_sql = format!(
        "SELECT DISTINCT {select_list} FROM {}",
        inner_from.join(", ")
    );
    if !inner_where.is_empty() {
        inner_sql.push_str(" WHERE ");
        inner_sql.push_str(&inner_where.join(" AND "));
    }

    let mut from = vec![format!("({inner_sql}) b")];
    for (i, decl) in box_.new_vars.iter().enumerate() {
        let id = start_id + i;
        let table = match decl.kind {
            VarKind::Event => "events",
            VarKind::Object => "objects",
        };
        from.push(format!("{table} v{id}"));
    }
    let where_conds: Vec<String> = var_ids
        .iter()
        .map(|v| format!("v{v}.id = b.b{v}"))
        .collect();

    Ok((from, where_conds, params))
}

/// Lower one fold of a correlated `ChildBox` to a scalar subquery: `(SELECT <agg> FROM <child's
/// own vars>, <child's own junctions> WHERE <child's conditions>)`. A filter referencing a parent
/// var uses that var's already-tabled outer alias `v{id}` directly.
fn child_subquery_sql(
    child: &ChildBox,
    agg_idx: usize,
    start_id: VarId,
    kinds: &[VarKind],
    attr_types: &ObjectAttrTypes,
    params: &mut Vec<SqlParam>,
) -> Result<String, String> {
    let agg = child
        .aggs
        .get(agg_idx)
        .ok_or_else(|| format!("agg index {agg_idx} out of range"))?;
    let own_var_ids: Vec<VarId> = (start_id..start_id + child.box_.new_vars.len()).collect();
    let own_children_base = start_id + child.box_.new_vars.len();
    let mut agg_params = Vec::new();
    let agg_sql = agg_expr_sql(
        agg,
        kinds,
        attr_types,
        &child.box_.children,
        own_children_base,
        &own_var_ids,
        &mut agg_params,
    )?;

    let (from, where_conds, mut fw_params) =
        box_from_where(&child.box_, start_id, kinds, attr_types)?;

    let mut sql = format!("SELECT {agg_sql} FROM {}", from.join(", "));
    if !where_conds.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&where_conds.join(" AND "));
    }

    // Textual order: the SELECT clause precedes from/where.
    params.append(&mut agg_params);
    params.append(&mut fw_params);

    Ok(format!("({sql})"))
}

/// SQL for an [`Agg`], appending any params its operand `Expr` needs. `Sum`/`Avg` `TRY_CAST`
/// their operand to `DOUBLE`, so non-numeric operands become `NULL` and are skipped, mirroring
/// `eval`'s `value_as_f64` filtering. `Min`/`Max` use the operand as-is, which is lexicographic
/// for an `Attr` whose stored value is numeric-as-text.
///
/// `distinct_scope` lists the own var id(s) whose distinctness defines "one binding" for
/// `Agg::Count`. Empty for the root box, whose `FROM` is already distinct-wrapped.
fn agg_expr_sql(
    agg: &Agg,
    kinds: &[VarKind],
    attr_types: &ObjectAttrTypes,
    own_children: &[ChildBox],
    children_base: VarId,
    distinct_scope: &[VarId],
    params: &mut Vec<SqlParam>,
) -> Result<String, String> {
    match agg {
        Agg::Count => Ok(match distinct_scope {
            [] => "COUNT(*)".to_string(),
            [v] => format!("COUNT(DISTINCT v{v}.id)"),
            vs => {
                let cols = vs
                    .iter()
                    .map(|v| format!("v{v}.id"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("COUNT(DISTINCT ({cols}))")
            }
        }),
        Agg::CountDistinct(e) => {
            let e_sql = expr_sql(e, kinds, attr_types, own_children, children_base, params)?;
            Ok(format!("COUNT(DISTINCT {e_sql})"))
        }
        Agg::Min(e) => {
            let e_sql = expr_sql(e, kinds, attr_types, own_children, children_base, params)?;
            Ok(format!("MIN({e_sql})"))
        }
        Agg::Max(e) => {
            let e_sql = expr_sql(e, kinds, attr_types, own_children, children_base, params)?;
            Ok(format!("MAX({e_sql})"))
        }
        Agg::Sum(e) => {
            let e_sql = expr_sql(e, kinds, attr_types, own_children, children_base, params)?;
            Ok(format!("SUM(TRY_CAST({e_sql} AS DOUBLE))"))
        }
        Agg::Avg(e) => {
            let e_sql = expr_sql(e, kinds, attr_types, own_children, children_base, params)?;
            Ok(format!("AVG(TRY_CAST({e_sql} AS DOUBLE))"))
        }
        Agg::Sequence { of, by } => {
            let of_sql = expr_sql(of, kinds, attr_types, own_children, children_base, params)?;
            if by.is_empty() {
                return Ok(format!("ARRAY_AGG({of_sql})"));
            }
            let order = by
                .iter()
                .map(|(e, dir)| {
                    Ok(format!(
                        "{} {}",
                        expr_sql(e, kinds, attr_types, own_children, children_base, params)?,
                        dir_sql(dir)
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?
                .join(", ");
            Ok(format!("ARRAY_AGG({of_sql} ORDER BY {order})"))
        }
    }
}

/// SQL for one projected/ordered/grouped [`Expr`], appending any params it needs (in the order
/// its own `?` placeholders appear). `own_children`/`children_base` resolve `Expr::ChildAgg`
/// against the box that owns this expression.
fn expr_sql(
    e: &Expr,
    kinds: &[VarKind],
    attr_types: &ObjectAttrTypes,
    own_children: &[ChildBox],
    children_base: VarId,
    params: &mut Vec<SqlParam>,
) -> Result<String, String> {
    match e {
        Expr::Id(v) => Ok(format!("v{v}.id")),
        Expr::Type(v) => Ok(format!("v{v}.ocel_type")),
        Expr::Time(v) => Ok(format!(r#"v{v}."time""#)),
        Expr::Attr { var, name, at } => {
            match kinds[*var] {
                // Event attributes are typed wide columns, so the name is a column identifier
                // rather than a bind param, and `at` does not apply (matches eval.rs).
                VarKind::Event => Ok(format!(r#"v{var}.{}"#, quote_ident(name))),
                // Object attributes are EAV text, cast back to the declared type so comparisons
                // match the in-memory backends. See [`ObjectAttrTypes`].
                VarKind::Object => {
                    params.push(SqlParam::Str(name.clone()));
                    let val = obj_attr_value_sql(attr_types, name);
                    Ok(match at {
                        super::OutAt::Latest => format!(
                            r#"(SELECT {val} FROM object_attribute_changes WHERE id = v{var}.id AND name = ? ORDER BY "time" DESC LIMIT 1)"#
                        ),
                        super::OutAt::First => format!(
                            r#"(SELECT {val} FROM object_attribute_changes WHERE id = v{var}.id AND name = ? ORDER BY "time" ASC LIMIT 1)"#
                        ),
                        super::OutAt::AtEvent(ev) => format!(
                            r#"(SELECT {val} FROM object_attribute_changes WHERE id = v{var}.id AND name = ? AND "time" <= v{ev}."time" ORDER BY "time" DESC LIMIT 1)"#
                        ),
                    })
                }
            }
        }
        Expr::ChildAgg(cref, agg_idx) => {
            let child = own_children
                .get(*cref)
                .ok_or_else(|| format!("Expr::ChildAgg: child ref {cref} out of range"))?;
            let start = child_start_id(children_base, own_children, *cref);
            child_subquery_sql(child, *agg_idx, start, kinds, attr_types, params)
        }
        Expr::Satisfies(f) => {
            let mut ctx = FilterCtx {
                kinds,
                attr_types,
                own_children,
                children_base,
                next_alias: &mut 0,
            };
            filter_cond_sql(f, &mut ctx, params)
        }
    }
}

/// SQL for an attribute filter, or `None` for a variant that is not one.
///
/// `EventAttr` needs [`total_bool`] because an absent attribute is a `NULL` wide column, which
/// reads as `NULL` rather than `false` under a `NOT`. The `ObjectAttr` renderings are
/// `EXISTS`/`NOT EXISTS` and already total.
fn attr_filter_sql(f: &Filter, params: &mut Vec<SqlParam>) -> Option<String> {
    Some(match f {
        Filter::EventAttr { event, name, vf } => {
            let col = format!("v{event}.{}", quote_ident(name));
            total_bool(&format!(
                "({})",
                event_attr_value_filter_cond(&col, vf, params)
            ))
        }
        Filter::ObjectAttr {
            object,
            name,
            at,
            vf,
        } => match at {
            FilterAt::Sometime => {
                params.push(SqlParam::Str(name.clone()));
                let vf_cond = value_filter_cond("oac", vf, params);
                format!(
                    "EXISTS (SELECT 1 FROM object_attribute_changes oac WHERE oac.id = v{object}.id AND oac.name = ? AND {vf_cond})"
                )
            }
            FilterAt::Always => {
                // Vacuously true with zero recorded values, matching eval.rs's `Iterator::all`,
                // so deliberately no existence check.
                params.push(SqlParam::Str(name.clone()));
                let vf_cond = value_filter_cond("oac", vf, params);
                format!(
                    "NOT EXISTS (SELECT 1 FROM object_attribute_changes oac WHERE oac.id = v{object}.id AND oac.name = ? AND NOT ({vf_cond}))"
                )
            }
            FilterAt::AtEvent(ev) => {
                params.push(SqlParam::Str(name.clone()));
                params.push(SqlParam::Str(name.clone()));
                let vf_cond = value_filter_cond("oac", vf, params);
                format!(
                    r#"EXISTS (SELECT 1 FROM object_attribute_changes oac WHERE oac.id = v{object}.id AND oac.name = ? AND oac."time" <= v{ev}."time" AND oac."time" = (SELECT MAX("time") FROM object_attribute_changes WHERE id = v{object}.id AND name = ? AND "time" <= v{ev}."time") AND {vf_cond})"#
                )
            }
        },
        _ => return None,
    })
}

/// SQL condition testing `<alias>.value`/`<alias>.value_type` against a [`ValueFilter`].
///
/// `TRY_CAST` sidesteps `DuckDB`'s lack of a guaranteed left-to-right `AND` short-circuit. The
/// `value_type` equality check matches `eval::value_filter_matches`, where the type must match
/// rather than merely happen to parse.
fn value_filter_cond(alias: &str, vf: &ValueFilter, params: &mut Vec<SqlParam>) -> String {
    match vf {
        ValueFilter::Integer { min, max } => {
            let mut conds = vec![format!("{alias}.value_type = 'integer'")];
            if let Some(m) = min {
                conds.push(format!("TRY_CAST({alias}.value AS BIGINT) >= ?"));
                params.push(SqlParam::Int(*m));
            }
            if let Some(m) = max {
                conds.push(format!("TRY_CAST({alias}.value AS BIGINT) <= ?"));
                params.push(SqlParam::Int(*m));
            }
            conds.join(" AND ")
        }
        ValueFilter::Float { min, max } => {
            let mut conds = vec![format!("{alias}.value_type = 'float'")];
            if let Some(m) = min {
                conds.push(format!("TRY_CAST({alias}.value AS DOUBLE) >= ?"));
                params.push(SqlParam::Float(*m));
            }
            if let Some(m) = max {
                conds.push(format!("TRY_CAST({alias}.value AS DOUBLE) <= ?"));
                params.push(SqlParam::Float(*m));
            }
            conds.join(" AND ")
        }
        ValueFilter::Boolean { is } => {
            params.push(SqlParam::Str(is.to_string()));
            format!("{alias}.value_type = 'boolean' AND {alias}.value = ?")
        }
        ValueFilter::String { is_in } => {
            if is_in.is_empty() {
                return "FALSE".to_string();
            }
            let placeholders = std::iter::repeat_n("?", is_in.len())
                .collect::<Vec<_>>()
                .join(",");
            params.extend(is_in.iter().map(|s| SqlParam::Str(s.clone())));
            format!("{alias}.value_type = 'string' AND {alias}.value IN ({placeholders})")
        }
        ValueFilter::Time { from, to } => {
            let mut conds = vec![format!("{alias}.value_type = 'time'")];
            if let Some(f) = from {
                conds.push(format!(
                    "TRY_CAST({alias}.value AS TIMESTAMPTZ) >= TRY_CAST(? AS TIMESTAMPTZ)"
                ));
                params.push(SqlParam::Str(f.to_rfc3339()));
            }
            if let Some(t) = to {
                conds.push(format!(
                    "TRY_CAST({alias}.value AS TIMESTAMPTZ) <= TRY_CAST(? AS TIMESTAMPTZ)"
                ));
                params.push(SqlParam::Str(t.to_rfc3339()));
            }
            conds.join(" AND ")
        }
    }
}

/// SQL condition testing a typed wide event-attribute column `col` against a [`ValueFilter`].
/// Unlike the EAV [`value_filter_cond`] there is no `value_type` column, so `TRY_CAST` on the
/// column is used directly, yielding `NULL` on mismatch or absence.
fn event_attr_value_filter_cond(col: &str, vf: &ValueFilter, params: &mut Vec<SqlParam>) -> String {
    match vf {
        ValueFilter::Integer { min, max } => {
            let mut conds = Vec::new();
            if let Some(m) = min {
                conds.push(format!("TRY_CAST({col} AS BIGINT) >= ?"));
                params.push(SqlParam::Int(*m));
            }
            if let Some(m) = max {
                conds.push(format!("TRY_CAST({col} AS BIGINT) <= ?"));
                params.push(SqlParam::Int(*m));
            }
            if conds.is_empty() {
                conds.push(format!("{col} IS NOT NULL"));
            }
            conds.join(" AND ")
        }
        ValueFilter::Float { min, max } => {
            let mut conds = Vec::new();
            if let Some(m) = min {
                conds.push(format!("TRY_CAST({col} AS DOUBLE) >= ?"));
                params.push(SqlParam::Float(*m));
            }
            if let Some(m) = max {
                conds.push(format!("TRY_CAST({col} AS DOUBLE) <= ?"));
                params.push(SqlParam::Float(*m));
            }
            if conds.is_empty() {
                conds.push(format!("{col} IS NOT NULL"));
            }
            conds.join(" AND ")
        }
        ValueFilter::Boolean { is } => {
            params.push(SqlParam::Bool(*is));
            format!("TRY_CAST({col} AS BOOLEAN) = ?")
        }
        ValueFilter::String { is_in } => {
            if is_in.is_empty() {
                return "FALSE".to_string();
            }
            let placeholders = std::iter::repeat_n("?", is_in.len())
                .collect::<Vec<_>>()
                .join(",");
            params.extend(is_in.iter().map(|s| SqlParam::Str(s.clone())));
            format!("CAST({col} AS VARCHAR) IN ({placeholders})")
        }
        ValueFilter::Time { from, to } => {
            let mut conds = Vec::new();
            if let Some(f) = from {
                conds.push(format!(
                    "TRY_CAST({col} AS TIMESTAMPTZ) >= TRY_CAST(? AS TIMESTAMPTZ)"
                ));
                params.push(SqlParam::Str(f.to_rfc3339()));
            }
            if let Some(t) = to {
                conds.push(format!(
                    "TRY_CAST({col} AS TIMESTAMPTZ) <= TRY_CAST(? AS TIMESTAMPTZ)"
                ));
                params.push(SqlParam::Str(t.to_rfc3339()));
            }
            if conds.is_empty() {
                conds.push(format!("{col} IS NOT NULL"));
            }
            conds.join(" AND ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    /// Orders and their directly-related events. Mirrors `model.rs`'s `query_bindings`.
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

    /// Mirrors `eval.rs`'s `projection_order_and_limit` test query.
    fn query_projection_order_limit() -> Query {
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
                project: vec![Expr::Id(1), Expr::Type(0), Expr::Time(0)],
                order_by: vec![(Expr::Time(0), Dir::Asc)],
                limit: Some(5),
            }),
            emits: Vec::new(),
        }
    }

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

    /// Mirrors `eval.rs`'s `order_items_child`: orders (0) -O2O-> items (1) child box.
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

    #[test]
    fn bindings_query_translates_and_binds_no_params() {
        let q = query_bindings();
        let sql = to_sql(&q, &ObjectAttrTypes::new()).unwrap();
        assert!(sql.sql.contains("FROM objects v0, events v1, e2o r0"));
        assert!(sql
            .sql
            .contains("r0.event_id = v1.id AND r0.object_id = v0.id"));
        assert!(sql.sql.contains("v0.ocel_type IN (?)"));
        assert_eq!(sql.params, vec![SqlParam::Str("order".to_string())]);
    }

    #[test]
    fn root_binding_is_distinct_wrapped_for_rows_and_aggregate() {
        let rows_sql = to_sql(&query_bindings(), &ObjectAttrTypes::new())
            .unwrap()
            .sql;
        assert!(rows_sql.contains("SELECT DISTINCT v0.id AS b0, v1.id AS b1 FROM"));
        assert!(rows_sql.contains(") b, objects v0, events v1"));
        assert!(rows_sql.contains("v0.id = b.b0 AND v1.id = b.b1"));

        let agg_sql = to_sql(&query_type_counts(), &ObjectAttrTypes::new())
            .unwrap()
            .sql;
        assert!(agg_sql.contains("SELECT DISTINCT v0.id AS b0, v1.id AS b1 FROM"));
        assert!(agg_sql.contains("v0.id = b.b0 AND v1.id = b.b1"));
    }

    #[test]
    fn projection_order_limit_query_places_params_in_textual_order() {
        let q = query_projection_order_limit();
        let sql = to_sql(&q, &ObjectAttrTypes::new()).unwrap();
        assert!(sql.sql.contains("ORDER BY"));
        assert!(sql.sql.ends_with("LIMIT ?"));
        assert_eq!(sql.params.last(), Some(&SqlParam::Int(5)));
    }

    #[test]
    fn aggregate_output_translates_with_positional_group_by() {
        let q = query_type_counts();
        let sql = to_sql(&q, &ObjectAttrTypes::new()).unwrap();
        assert!(sql
            .sql
            .starts_with("SELECT v0.ocel_type, v1.ocel_type, COUNT(*) FROM"));
        assert!(sql.sql.contains("GROUP BY 1, 2"));
        assert!(sql.params.is_empty());
    }

    #[test]
    fn children_filter_lowers_to_correlated_subquery() {
        let q = Query {
            root: order_items_child(Some((Some(3.0), None))),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let sql = to_sql(&q, &ObjectAttrTypes::new()).unwrap();
        assert!(sql.sql.contains(
            "(SELECT COUNT(DISTINCT v1.id) FROM objects v1, o2o r0 WHERE v1.ocel_type IN (?) AND r0.source_id = v0.id AND r0.target_id = v1.id) >= ?"
        ));
        assert_eq!(
            sql.params,
            vec![
                SqlParam::Str("orders".to_string()),
                SqlParam::Str("items".to_string()),
                SqlParam::Float(3.0)
            ]
        );
    }

    #[test]
    fn child_scalar_output_lowers_correlated_count_subquery() {
        let q = Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };
        let sql = to_sql(&q, &ObjectAttrTypes::new()).unwrap();
        assert!(sql.sql.contains("SELECT v0.id, (SELECT COUNT(DISTINCT v1.id) FROM objects v1, o2o r0 WHERE v1.ocel_type IN (?) AND r0.source_id = v0.id AND r0.target_id = v1.id)"));
        assert_eq!(
            sql.params,
            vec![
                SqlParam::Str("items".to_string()),
                SqlParam::Str("orders".to_string())
            ]
        );
    }

    #[test]
    fn value_filters_lower_to_gated_try_casts() {
        let mut params = Vec::new();
        let cond = value_filter_cond(
            "ea",
            &ValueFilter::Integer {
                min: Some(1),
                max: Some(2),
            },
            &mut params,
        );
        assert!(cond.contains("ea.value_type = 'integer'"));
        assert!(cond.contains("TRY_CAST(ea.value AS BIGINT)"));
        assert_eq!(params, vec![SqlParam::Int(1), SqlParam::Int(2)]);
    }
}
