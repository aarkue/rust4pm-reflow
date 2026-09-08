//! Live, out-of-core `QueryableOCEL` backed by a `DuckDB` connection. Events and objects are
//! plain `String` ids, so no id->type map is kept in memory. Scalar accessors are point queries
//! and `get_all_*` uses keyset pagination over `&self.con`.
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::path::Path;

use chrono::{DateTime, FixedOffset};
use duckdb::{Connection, OptionalExt};

use crate::core::event_data::object_centric::linked_ocel::QueryableOCEL;
use crate::core::event_data::object_centric::ocel_struct::OCELAttributeType;
use crate::core::event_data::object_centric::query::eval::{
    agg_column_name, column_name, QueryResult, Value,
};
use crate::core::event_data::object_centric::query::sql::{to_sql, ObjectAttrTypes, SqlParam};
use crate::core::event_data::object_centric::query::{Agg, Expr, Output, Query};
use crate::core::event_data::object_centric::OCELAttributeValue;
use crate::core::event_data::timestamp_utils::parse_timestamp;

use super::schema_access::{DefaultSchema, DuckDbOcelSchema};
use super::value::{duck_timestamp_to_datetime, duck_value_to_ocel, from_sql_value};

const PAGE_SIZE: usize = 1024;

/// A live, out-of-core [`QueryableOCEL`] backed by a `DuckDB` database file in the EAV schema
/// (see [`stream_ocel_file_to_duckdb`](super::stream_ocel_file_to_duckdb)).
///
/// Events and objects are represented by their `String` id, so no id->type map is held in memory
/// and datasets larger than RAM are supported. Scalar accessors run one point query per call, and
/// [`QueryableOCEL::get_all_evs`]/[`QueryableOCEL::get_all_obs`] page with keyset pagination.
pub struct DuckDbLinkedOCEL<S = DefaultSchema> {
    con: Connection,
    schema: S,
    ev_types: Vec<String>,
    ob_types: Vec<String>,
    /// Wide event-attribute columns on `events`, so an unknown attribute name answers `None`
    /// instead of raising a missing-column SQL error.
    ev_attr_cols: std::collections::HashSet<String>,
    /// Declared type per object-attribute name; see [`ObjectAttrTypes`].
    ob_attr_types: ObjectAttrTypes,
}

impl<S> std::fmt::Debug for DuckDbLinkedOCEL<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDbLinkedOCEL")
            .field("con", &self.con)
            .field("ev_types", &self.ev_types)
            .field("ob_types", &self.ob_types)
            .finish_non_exhaustive()
    }
}

impl DuckDbLinkedOCEL<DefaultSchema> {
    /// Open a `DuckDB` database file in the schema (as written by
    /// [`stream_ocel_file_to_duckdb`](super::stream_ocel_file_to_duckdb)) as a live
    /// [`QueryableOCEL`].
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, duckdb::Error> {
        let con = Connection::open(path)?;
        // Covers `DuckDbOcelSchema`'s ~15 fixed statements plus a handful of batch sizes, each
        // distinct arity being its own cache entry.
        con.set_prepared_statement_cache_capacity(32);
        let schema = DefaultSchema;
        let ev_types = {
            let mut stmt = con.prepare(schema.distinct_ev_types_sql())?;
            stmt.query_map([], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        let ob_types = {
            let mut stmt = con.prepare(schema.distinct_ob_types_sql())?;
            stmt.query_map([], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        let ev_attr_cols = {
            let mut stmt = con.prepare(
                "SELECT column_name FROM information_schema.columns \
                 WHERE table_name = 'events' AND column_name NOT IN ('id', 'ocel_type', 'time')",
            )?;
            stmt.query_map([], |r| r.get::<_, String>(0))?
                .collect::<Result<std::collections::HashSet<_>, _>>()?
        };
        // Type per object-attribute name, so the EAV `value` (VARCHAR) can be cast back on the way
        // out. A name observed with several types is widened via `OCELAttributeType::coalesce`, so
        // integer+float becomes float rather than collapsing to text, staying correctly ordered.
        //
        // Backends that do not reconcile values against their declared type (`IndexLinkedOCEL`,
        // plain `OCEL`) therefore hold `Int` where this reports `Float`.
        let ob_attr_types = {
            let mut stmt =
                con.prepare("SELECT DISTINCT name, value_type FROM object_attribute_changes")?;
            let rows =
                stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
            let mut map = ObjectAttrTypes::new();
            for row in rows {
                let (name, vt) = row?;
                let ty = OCELAttributeType::from_type_str(&vt);
                map.entry(name)
                    .and_modify(|existing| *existing = existing.coalesce(ty))
                    .or_insert(ty);
            }
            map
        };
        Ok(Self {
            con,
            schema,
            ev_types,
            ob_types,
            ev_attr_cols,
            ob_attr_types,
        })
    }
}

impl<S> DuckDbLinkedOCEL<S> {
    /// Distinct event type names present in the database (loaded once at [`open`](Self::open) time).
    pub fn ev_type_names(&self) -> &[String] {
        &self.ev_types
    }

    /// Distinct object type names present in the database (loaded once at [`open`](Self::open) time).
    pub fn ob_type_names(&self) -> &[String] {
        &self.ob_types
    }

    /// Tune the `DuckDB` engine: `memory_limit` (e.g. `"4GB"`) caps engine working memory and
    /// `threads` bounds parallelism. `None` leaves that setting at its `DuckDB` default.
    ///
    /// Exceeding `memory_limit` is an `Out of Memory Error`, not a slower disk-backed query:
    /// spilling needs a `temp_directory`, which this crate never sets. Set one via
    /// `execute_batch` if you need spill-to-disk.
    pub fn configure_engine(
        &self,
        memory_limit: Option<&str>,
        threads: Option<usize>,
    ) -> Result<(), duckdb::Error> {
        if let Some(m) = memory_limit {
            self.con
                .execute_batch(&format!("SET memory_limit='{m}';"))?;
        }
        if let Some(t) = threads {
            self.con.execute_batch(&format!("SET threads={t};"))?;
        }
        Ok(())
    }
}

/// Keyset-paging iterator over a `(id, ...)` table. Each `refill` fully drains a page into an
/// owned `Vec<String>`, so nothing borrowed from `duckdb` is stored on the struct.
struct KeysetPager<'a> {
    con: &'a Connection,
    page_sql: &'static str,
    buf: VecDeque<String>,
    last_id: String,
    page_size: usize,
    done: bool,
}

impl<'a> KeysetPager<'a> {
    fn new(con: &'a Connection, page_sql: &'static str, page_size: usize) -> Self {
        Self {
            con,
            page_sql,
            buf: VecDeque::new(),
            last_id: String::new(),
            page_size,
            done: false,
        }
    }

    fn refill(&mut self) {
        let mut stmt = self
            .con
            .prepare_cached(self.page_sql)
            .expect("prepare keyset page query");
        let page: Vec<String> = stmt
            .query_map(duckdb::params![self.last_id, self.page_size as i64], |r| {
                r.get::<_, String>(0)
            })
            .expect("query keyset page")
            .collect::<Result<Vec<_>, _>>()
            .expect("read keyset page rows");
        if page.len() < self.page_size {
            self.done = true;
        }
        match page.last() {
            Some(last) => self.last_id = last.clone(),
            None => self.done = true,
        }
        self.buf.extend(page);
    }
}

impl Iterator for KeysetPager<'_> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() && !self.done {
            self.refill();
        }
        self.buf.pop_front()
    }
}

impl QueryableOCEL for DuckDbLinkedOCEL<DefaultSchema> {
    type EventRepr = String;
    type ObjectRepr = String;
    // A position into the `open`-time `ev_types`/`ob_types` lists: `Copy`, and no extra index to
    // maintain. `run_query`/`run_query_fold` push down to SQL, so these accessors are off the hot
    // path.
    type EvTypeId = usize;
    type ObTypeId = usize;

    fn get_ev_type_id(&self, ev: &String) -> Self::EvTypeId {
        let name = self.get_ev_type_of(ev);
        self.ev_types
            .iter()
            .position(|t| t == name.as_ref())
            .expect("event type returned by the database is one of the open-time ev_types")
    }

    fn get_ob_type_id(&self, ob: &String) -> Self::ObTypeId {
        let name = self.get_ob_type_of(ob);
        self.ob_types
            .iter()
            .position(|t| t == name.as_ref())
            .expect("object type returned by the database is one of the open-time ob_types")
    }

    fn resolve_ev_type(&self, id: Self::EvTypeId) -> Cow<'_, str> {
        Cow::Borrowed(&self.ev_types[id])
    }

    fn resolve_ob_type(&self, id: Self::ObTypeId) -> Cow<'_, str> {
        Cow::Borrowed(&self.ob_types[id])
    }

    fn get_all_evs(&self) -> impl Iterator<Item = String> + '_ {
        KeysetPager::new(&self.con, self.schema.all_events_page_sql(), PAGE_SIZE)
    }

    fn get_all_obs(&self) -> impl Iterator<Item = String> + '_ {
        KeysetPager::new(&self.con, self.schema.all_objects_page_sql(), PAGE_SIZE)
    }

    fn get_ev_id(&self, ev: &String) -> Cow<'_, str> {
        // The trait's elided `Cow<'_, str>` borrows from `&self`, but `ev` is a standalone id
        // living outside `self`, so it has to be cloned.
        Cow::Owned(ev.clone())
    }

    fn get_ob_id(&self, ob: &String) -> Cow<'_, str> {
        Cow::Owned(ob.clone())
    }

    fn get_ev_type_of(&self, ev: &String) -> Cow<'_, str> {
        let t: String = self
            .con
            .prepare_cached(self.schema.ev_type_sql())
            .expect("prepare ev_type query")
            .query_row(duckdb::params![ev], |r| r.get::<_, String>(0))
            .expect("event id should exist in the database");
        Cow::Owned(t)
    }

    fn get_ob_type_of(&self, ob: &String) -> Cow<'_, str> {
        let t: String = self
            .con
            .prepare_cached(self.schema.ob_type_sql())
            .expect("prepare ob_type query")
            .query_row(duckdb::params![ob], |r| r.get::<_, String>(0))
            .expect("object id should exist in the database");
        Cow::Owned(t)
    }

    fn get_ev_time(&self, ev: &String) -> DateTime<FixedOffset> {
        let s: String = self
            .con
            .prepare_cached(self.schema.ev_time_sql())
            .expect("prepare ev_time query")
            .query_row(duckdb::params![ev], |r| r.get::<_, String>(0))
            .expect("event id should exist in the database");
        parse_timestamp(&s, None, false).expect("valid timestamp stored in the database")
    }

    fn get_e2o(&self, ev: &String) -> impl Iterator<Item = (Cow<'_, str>, String)> + '_ {
        let mut stmt = self
            .con
            .prepare_cached(self.schema.e2o_sql())
            .expect("prepare e2o query");
        let rows: Vec<(String, String)> = stmt
            .query_map(duckdb::params![ev], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .expect("query e2o")
            .collect::<Result<Vec<_>, _>>()
            .expect("read e2o rows");
        rows.into_iter()
            .map(|(qualifier, object_id)| (Cow::Owned(qualifier), object_id))
    }

    fn get_e2o_rev(&self, ob: &String) -> impl Iterator<Item = (Cow<'_, str>, String)> + '_ {
        let mut stmt = self
            .con
            .prepare_cached(self.schema.e2o_rev_sql())
            .expect("prepare e2o_rev query");
        let rows: Vec<(String, String)> = stmt
            .query_map(duckdb::params![ob], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .expect("query e2o_rev")
            .collect::<Result<Vec<_>, _>>()
            .expect("read e2o_rev rows");
        rows.into_iter()
            .map(|(qualifier, event_id)| (Cow::Owned(qualifier), event_id))
    }

    fn get_o2o(&self, ob: &String) -> impl Iterator<Item = (Cow<'_, str>, String)> + '_ {
        let mut stmt = self
            .con
            .prepare_cached(self.schema.o2o_sql())
            .expect("prepare o2o query");
        let rows: Vec<(String, String)> = stmt
            .query_map(duckdb::params![ob], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .expect("query o2o")
            .collect::<Result<Vec<_>, _>>()
            .expect("read o2o rows");
        rows.into_iter()
            .map(|(qualifier, target_id)| (Cow::Owned(qualifier), target_id))
    }

    fn get_o2o_rev(&self, ob: &String) -> impl Iterator<Item = (Cow<'_, str>, String)> + '_ {
        let mut stmt = self
            .con
            .prepare_cached(self.schema.o2o_rev_sql())
            .expect("prepare o2o_rev query");
        let rows: Vec<(String, String)> = stmt
            .query_map(duckdb::params![ob], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .expect("query o2o_rev")
            .collect::<Result<Vec<_>, _>>()
            .expect("read o2o_rev rows");
        rows.into_iter()
            .map(|(qualifier, source_id)| (Cow::Owned(qualifier), source_id))
    }

    fn get_obs_of_type(&self, ty: &str) -> impl Iterator<Item = String> + '_ {
        // TODO: keyset-page large type scans
        let mut stmt = self
            .con
            .prepare_cached(self.schema.obs_of_type_sql())
            .expect("prepare obs_of_type query");
        let rows: Vec<String> = stmt
            .query_map(duckdb::params![ty], |r| r.get::<_, String>(0))
            .expect("query obs_of_type")
            .collect::<Result<Vec<_>, _>>()
            .expect("read obs_of_type rows");
        rows.into_iter()
    }

    fn get_evs_of_type(&self, ty: &str) -> impl Iterator<Item = String> + '_ {
        // TODO: keyset-page large type scans
        let mut stmt = self
            .con
            .prepare_cached(self.schema.evs_of_type_sql())
            .expect("prepare evs_of_type query");
        let rows: Vec<String> = stmt
            .query_map(duckdb::params![ty], |r| r.get::<_, String>(0))
            .expect("query evs_of_type")
            .collect::<Result<Vec<_>, _>>()
            .expect("read evs_of_type rows");
        rows.into_iter()
    }

    fn get_ev_types(&self) -> impl Iterator<Item = Cow<'_, str>> + '_ {
        self.ev_types.iter().map(|s| Cow::Borrowed(s.as_str()))
    }

    fn get_ob_types(&self) -> impl Iterator<Item = Cow<'_, str>> + '_ {
        self.ob_types.iter().map(|s| Cow::Borrowed(s.as_str()))
    }

    fn get_ev_attr_val(&self, ev: &String, name: &str) -> Option<OCELAttributeValue> {
        // An attribute named like one of the table's own columns is stored under a suffixed one.
        let column = super::tables::event_attr_column(name);
        // Unknown attribute name -> no such wide column -> None (avoids a SQL error).
        if !self.ev_attr_cols.contains(column.as_ref()) {
            return None;
        }
        let value: Option<duckdb::types::Value> = self
            .con
            .prepare_cached(&self.schema.ev_attr_val_sql(&column))
            .expect("prepare ev_attr_val query")
            .query_row(duckdb::params![ev], |r| r.get::<_, duckdb::types::Value>(0))
            .optional()
            .expect("query ev_attr_val");
        match value.map(duck_value_to_ocel) {
            // A NULL wide cell (attribute not set for this event) reads back as None.
            None | Some(OCELAttributeValue::Null) => None,
            some => some,
        }
    }

    fn get_ob_attr_vals(
        &self,
        ob: &String,
        name: &str,
    ) -> impl Iterator<Item = (DateTime<FixedOffset>, OCELAttributeValue)> + '_ {
        let mut stmt = self
            .con
            .prepare_cached(self.schema.ob_attr_vals_sql())
            .expect("prepare ob_attr_vals query");
        let rows: Vec<(String, String, String)> = stmt
            .query_map(duckdb::params![ob, name], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .expect("query ob_attr_vals")
            .collect::<Result<Vec<_>, _>>()
            .expect("read ob_attr_vals rows");
        rows.into_iter()
            .map(|(time_str, value_str, value_type_str)| {
                let time = parse_timestamp(&time_str, None, false)
                    .expect("valid timestamp stored in the database");
                let value = from_sql_value(&value_str, &value_type_str);
                (time, value)
            })
    }

    fn get_ev_types_of_batch(&self, evs: &[String]) -> Vec<Cow<'_, str>> {
        if evs.is_empty() {
            return Vec::new();
        }
        let sql = self.schema.ev_types_batch_sql(evs.len());
        let mut stmt = self
            .con
            .prepare_cached(&sql)
            .expect("prepare batch ev types query");
        let by_id: HashMap<String, String> = stmt
            .query_map(duckdb::params_from_iter(evs.iter()), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .expect("query batch ev types")
            .collect::<Result<HashMap<_, _>, _>>()
            .expect("read batch ev types rows");
        evs.iter()
            .map(|id| {
                Cow::Owned(
                    by_id
                        .get(id)
                        .expect("event id should exist in the database")
                        .clone(),
                )
            })
            .collect()
    }

    /// Translate `query` to one SQL statement over the schema (correlated
    /// `ChildBox`es lowered to scalar subqueries; see [`to_sql`]) and run it.
    fn run_query(&self, query: &Query) -> Result<QueryResult, String> {
        let sql_query = to_sql(query, &self.ob_attr_types)?;
        let duck_params: Vec<duckdb::types::Value> =
            sql_query.params.iter().map(sql_param_to_duckdb).collect();

        let mut stmt = self
            .con
            .prepare_cached(&sql_query.sql)
            .map_err(|e| e.to_string())?;

        match &query.output {
            Output::Rows(spec) => {
                let rows: Vec<Vec<Value>> = stmt
                    .query_map(duckdb::params_from_iter(duck_params.iter()), |r| {
                        spec.project
                            .iter()
                            .enumerate()
                            .map(|(i, e)| decode_col(r, i, e))
                            .collect()
                    })
                    .map_err(|e| e.to_string())?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())?;

                let columns = spec.project.iter().map(column_name).collect();
                Ok(QueryResult { columns, rows })
            }
            Output::Aggregate(spec) => {
                let n_group = spec.group_by.len();
                let rows: Vec<Vec<Value>> = stmt
                    .query_map(duckdb::params_from_iter(duck_params.iter()), |r| {
                        spec.group_by
                            .iter()
                            .enumerate()
                            .map(|(i, e)| decode_col(r, i, e))
                            .chain(
                                spec.aggregates
                                    .iter()
                                    .enumerate()
                                    .map(|(i, a)| decode_agg_col(r, n_group + i, a)),
                            )
                            .collect()
                    })
                    .map_err(|e| e.to_string())?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())?;

                let columns = spec
                    .group_by
                    .iter()
                    .map(column_name)
                    .chain(spec.aggregates.iter().map(agg_column_name))
                    .collect();
                Ok(QueryResult { columns, rows })
            }
        }
    }

    /// Streaming variant of [`Self::run_query`]: rows are handed to `f` one at a time, so the
    /// *Rust-side* heap stays bounded regardless of result size.
    ///
    /// Total memory is not bounded. `duckdb`'s `query()` uses the non-streaming Arrow path, so the
    /// engine still materializes the full result set before the first row is yielded.
    fn run_query_fold(&self, query: &Query, mut f: impl FnMut(&[Value])) -> Result<(), String> {
        let sql_query = to_sql(query, &self.ob_attr_types)?;
        let duck_params: Vec<duckdb::types::Value> =
            sql_query.params.iter().map(sql_param_to_duckdb).collect();

        let mut stmt = self
            .con
            .prepare_cached(&sql_query.sql)
            .map_err(|e| e.to_string())?;
        let mut rows = stmt
            .query(duckdb::params_from_iter(duck_params.iter()))
            .map_err(|e| e.to_string())?;

        match &query.output {
            Output::Rows(spec) => {
                while let Some(r) = rows.next().map_err(|e| e.to_string())? {
                    let decoded: Vec<Value> = spec
                        .project
                        .iter()
                        .enumerate()
                        .map(|(i, e)| decode_col(r, i, e))
                        .collect::<Result<_, _>>()
                        .map_err(|e| e.to_string())?;
                    f(&decoded);
                }
            }
            Output::Aggregate(spec) => {
                let n_group = spec.group_by.len();
                while let Some(r) = rows.next().map_err(|e| e.to_string())? {
                    let decoded: Vec<Value> = spec
                        .group_by
                        .iter()
                        .enumerate()
                        .map(|(i, e)| decode_col(r, i, e))
                        .chain(
                            spec.aggregates
                                .iter()
                                .enumerate()
                                .map(|(i, a)| decode_agg_col(r, n_group + i, a)),
                        )
                        .collect::<Result<_, _>>()
                        .map_err(|e| e.to_string())?;
                    f(&decoded);
                }
            }
        }
        Ok(())
    }
}

fn sql_param_to_duckdb(p: &SqlParam) -> duckdb::types::Value {
    match p {
        SqlParam::Str(s) => duckdb::types::Value::Text(s.clone()),
        SqlParam::Int(i) => duckdb::types::Value::BigInt(*i),
        SqlParam::Float(f) => duckdb::types::Value::Double(*f),
        SqlParam::Bool(b) => duckdb::types::Value::Boolean(*b),
    }
}

/// Map a raw `duckdb::types::Value` to the crate's typed [`Value`]. Handles both `Attr` column
/// shapes: natively-typed wide columns for events, the EAV `value` string for objects.
fn duck_value_to_eval(v: duckdb::types::Value) -> Value {
    use duckdb::types::Value as DV;
    match v {
        DV::Null => Value::Null,
        DV::Boolean(b) => Value::Bool(b),
        DV::TinyInt(i) => Value::Int(i as i64),
        DV::SmallInt(i) => Value::Int(i as i64),
        DV::Int(i) => Value::Int(i as i64),
        DV::BigInt(i) => Value::Int(i),
        DV::HugeInt(i) => Value::Int(i as i64),
        DV::UTinyInt(i) => Value::Int(i as i64),
        DV::USmallInt(i) => Value::Int(i as i64),
        DV::UInt(i) => Value::Int(i as i64),
        DV::UBigInt(i) => Value::Int(i as i64),
        DV::Float(f) => Value::Float(f as f64),
        DV::Double(f) => Value::Float(f),
        DV::Text(s) => Value::Str(s),
        DV::Timestamp(tu, t) => duck_timestamp_to_datetime(tu, t).map_or(Value::Null, Value::Time),
        _ => Value::Null,
    }
}

fn decode_col(row: &duckdb::Row<'_>, i: usize, e: &Expr) -> duckdb::Result<Value> {
    match e {
        Expr::Id(_) | Expr::Type(_) => Ok(Value::Str(row.get::<_, String>(i)?)),
        Expr::Time(_) => {
            let s: String = row.get(i)?;
            let t =
                parse_timestamp(&s, None, false).expect("valid timestamp stored in the database");
            Ok(Value::Time(t))
        }
        Expr::Attr { .. } => Ok(duck_value_to_eval(row.get::<_, duckdb::types::Value>(i)?)),
        // Best-effort: the referenced fold is usually a `Count`, so decode as `Int`. Threading
        // the referenced `Agg`'s kind through would need the owning box at this call site.
        Expr::ChildAgg(..) => {
            let v: i64 = row.get(i)?;
            Ok(Value::Int(v))
        }
        Expr::Satisfies(_) => Ok(Value::Bool(row.get(i)?)),
    }
}

/// Decode one `AggSpec::aggregates` result column by `Agg` kind. Unlike `eval::compute_agg`, an
/// all-integral `Sum` is reported as `Float` rather than `Int`.
fn decode_agg_col(row: &duckdb::Row<'_>, i: usize, agg: &Agg) -> duckdb::Result<Value> {
    match agg {
        Agg::Count | Agg::CountDistinct(_) => {
            let v: i64 = row.get(i)?;
            Ok(Value::Int(v))
        }
        Agg::Sum(_) | Agg::Avg(_) => {
            let v: Option<f64> = row.get(i)?;
            Ok(v.map(Value::Float).unwrap_or(Value::Null))
        }
        Agg::Min(e) | Agg::Max(e) => decode_col(row, i, e),
        // `Sequence.of` is a string-valued expr (Type/Id) in the supported queries, so every
        // element of the `ARRAY_AGG` list decodes as `Value::Str`.
        Agg::Sequence { .. } => {
            let v: duckdb::types::Value = row.get(i)?;
            let items = match v {
                duckdb::types::Value::List(xs) | duckdb::types::Value::Array(xs) => xs
                    .into_iter()
                    .map(|x| match x {
                        duckdb::types::Value::Text(s) => Value::Str(s),
                        _ => Value::Null,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            Ok(Value::List(items))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::super::stream::stream_ocel_file_to_duckdb;
    use super::super::tables::{build_events_table, create_schema};
    use super::*;
    use crate::core::event_data::object_centric::linked_ocel::IndexLinkedOCEL;
    use crate::core::event_data::object_centric::ocel_json::import_ocel_json_path;
    use crate::test_utils::get_test_data_path;

    fn order_management_path() -> std::path::PathBuf {
        get_test_data_path()
            .join("ocel")
            .join("order-management.json")
    }

    #[test]
    fn duckdb_queryable_matches_index() {
        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());

        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-parity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let db_ev_count = QueryableOCEL::get_all_evs(&db).count();
        let ref_ev_count = QueryableOCEL::get_all_evs(&reference).count();
        assert_eq!(db_ev_count, ref_ev_count);

        let db_ob_count = QueryableOCEL::get_all_obs(&db).count();
        let ref_ob_count = QueryableOCEL::get_all_obs(&reference).count();
        assert_eq!(db_ob_count, ref_ob_count);

        let ref_ev_type_by_id: HashMap<String, String> = QueryableOCEL::get_all_evs(&reference)
            .map(|ev| {
                (
                    QueryableOCEL::get_ev_id(&reference, &ev).into_owned(),
                    QueryableOCEL::get_ev_type_of(&reference, &ev).into_owned(),
                )
            })
            .collect();

        let mut db_ev_ids: Vec<String> = QueryableOCEL::get_all_evs(&db).collect();
        db_ev_ids.sort();

        // Every event, since order-management is small.
        for id in &db_ev_ids {
            let db_type = QueryableOCEL::get_ev_type_of(&db, id);
            let ref_type = ref_ev_type_by_id
                .get(id)
                .unwrap_or_else(|| panic!("reference should contain event {id}"));
            assert_eq!(
                db_type.as_ref(),
                ref_type.as_str(),
                "event type mismatch for {id}"
            );
        }

        // E2O parity for a sample of events.
        let ref_ev_by_id: HashMap<String, <IndexLinkedOCEL as QueryableOCEL>::EventRepr> =
            QueryableOCEL::get_all_evs(&reference)
                .map(|ev| (QueryableOCEL::get_ev_id(&reference, &ev).into_owned(), ev))
                .collect();

        for id in db_ev_ids.iter().take(500) {
            let db_e2o: HashSet<(String, String)> = QueryableOCEL::get_e2o(&db, id)
                .map(|(q, o)| (q.into_owned(), o))
                .collect();
            let ref_ev = ref_ev_by_id.get(id).expect("reference event by id");
            let ref_e2o: HashSet<(String, String)> = QueryableOCEL::get_e2o(&reference, ref_ev)
                .map(|(q, o)| {
                    (
                        q.into_owned(),
                        QueryableOCEL::get_ob_id(&reference, &o).into_owned(),
                    )
                })
                .collect();
            assert_eq!(db_e2o, ref_e2o, "e2o mismatch for event {id}");
        }

        // O2O / reverse-E2O parity for a sample of objects.
        let mut db_ob_ids: Vec<String> = QueryableOCEL::get_all_obs(&db).collect();
        db_ob_ids.sort();
        let ref_ob_by_id: HashMap<String, <IndexLinkedOCEL as QueryableOCEL>::ObjectRepr> =
            QueryableOCEL::get_all_obs(&reference)
                .map(|ob| (QueryableOCEL::get_ob_id(&reference, &ob).into_owned(), ob))
                .collect();

        for id in db_ob_ids.iter().take(500) {
            let db_o2o: HashSet<(String, String)> = QueryableOCEL::get_o2o(&db, id)
                .map(|(q, o)| (q.into_owned(), o))
                .collect();
            let ref_ob = ref_ob_by_id.get(id).expect("reference object by id");
            let ref_o2o: HashSet<(String, String)> = QueryableOCEL::get_o2o(&reference, ref_ob)
                .map(|(q, o)| {
                    (
                        q.into_owned(),
                        QueryableOCEL::get_ob_id(&reference, &o).into_owned(),
                    )
                })
                .collect();
            assert_eq!(db_o2o, ref_o2o, "o2o mismatch for object {id}");

            let db_e2o_rev: HashSet<(String, String)> = QueryableOCEL::get_e2o_rev(&db, id)
                .map(|(q, e)| (q.into_owned(), e))
                .collect();
            let ref_e2o_rev: HashSet<(String, String)> =
                QueryableOCEL::get_e2o_rev(&reference, ref_ob)
                    .map(|(q, e)| {
                        (
                            q.into_owned(),
                            QueryableOCEL::get_ev_id(&reference, &e).into_owned(),
                        )
                    })
                    .collect();
            assert_eq!(db_e2o_rev, ref_e2o_rev, "e2o_rev mismatch for object {id}");
        }

        let db_ev_types: HashSet<String> = QueryableOCEL::get_ev_types(&db)
            .map(|c| c.into_owned())
            .collect();
        let ref_ev_types: HashSet<String> = QueryableOCEL::get_ev_types(&reference)
            .map(|c| c.into_owned())
            .collect();
        assert_eq!(db_ev_types, ref_ev_types);

        let db_ob_types: HashSet<String> = QueryableOCEL::get_ob_types(&db)
            .map(|c| c.into_owned())
            .collect();
        let ref_ob_types: HashSet<String> = QueryableOCEL::get_ob_types(&reference)
            .map(|c| c.into_owned())
            .collect();
        assert_eq!(db_ob_types, ref_ob_types);

        for ty in &db_ev_types {
            let db_count = QueryableOCEL::get_evs_of_type(&db, ty).count();
            let ref_count = QueryableOCEL::get_evs_of_type(&reference, ty).count();
            assert_eq!(
                db_count, ref_count,
                "get_evs_of_type count mismatch for {ty}"
            );
        }
        for ty in &db_ob_types {
            let db_count = QueryableOCEL::get_obs_of_type(&db, ty).count();
            let ref_count = QueryableOCEL::get_obs_of_type(&reference, ty).count();
            assert_eq!(
                db_count, ref_count,
                "get_obs_of_type count mismatch for {ty}"
            );
        }

        // Object attribute parity: order-management objects carry attributes, its events do not.
        let ref_ob_with_attr = QueryableOCEL::get_all_obs(&reference)
            .find_map(|ob| {
                let name = reference
                    .get_ocel_ref()
                    .objects
                    .iter()
                    .find(|o| o.id == QueryableOCEL::get_ob_id(&reference, &ob).as_ref())?
                    .attributes
                    .first()?
                    .name
                    .clone();
                let vals: Vec<_> =
                    QueryableOCEL::get_ob_attr_vals(&reference, &ob, &name).collect();
                if vals.is_empty() {
                    None
                } else {
                    Some((
                        QueryableOCEL::get_ob_id(&reference, &ob).into_owned(),
                        name,
                        vals,
                    ))
                }
            })
            .expect("order-management should have at least one object with an attribute");
        let (ob_id, attr_name, ref_vals) = ref_ob_with_attr;

        let db_vals: Vec<_> = QueryableOCEL::get_ob_attr_vals(&db, &ob_id, &attr_name).collect();
        assert_eq!(db_vals.len(), ref_vals.len());
        // The DB query is `ORDER BY "time"` while the reference yields raw insertion order, so
        // sort both first. `OCELAttributeValue` is `PartialEq` but not `Hash`/`Ord` (it carries
        // an `f64`), leaving its `Display` string as the tiebreaker.
        let sort_key = |(t, v): &(DateTime<FixedOffset>, OCELAttributeValue)| (*t, v.to_string());
        let mut db_sorted = db_vals;
        db_sorted.sort_by_key(sort_key);
        let mut ref_sorted = ref_vals;
        ref_sorted.sort_by_key(sort_key);
        for ((db_t, db_v), (ref_t, ref_v)) in db_sorted.iter().zip(ref_sorted.iter()) {
            assert_eq!(db_t, ref_t, "attr time mismatch for {ob_id}/{attr_name}");
            assert_eq!(db_v, ref_v, "attr value mismatch for {ob_id}/{attr_name}");
        }
    }

    #[test]
    fn duckdb_ev_attr_val_matches_index() {
        // Uses ocel2-p2p because order-management events have no attributes.
        let src = get_test_data_path().join("ocel").join("ocel2-p2p.json");
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());

        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-ev-attr.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let (ev_id, attr_name, ref_val) = QueryableOCEL::get_all_evs(&reference)
            .find_map(|ev| {
                let ev_id = QueryableOCEL::get_ev_id(&reference, &ev).into_owned();
                let name = reference
                    .get_ocel_ref()
                    .events
                    .iter()
                    .find(|e| e.id == ev_id)?
                    .attributes
                    .first()?
                    .name
                    .clone();
                let val = QueryableOCEL::get_ev_attr_val(&reference, &ev, &name)?;
                Some((ev_id, name, val))
            })
            .expect("ocel2-p2p should have at least one event with an attribute");

        let db_val = QueryableOCEL::get_ev_attr_val(&db, &ev_id, &attr_name)
            .unwrap_or_else(|| panic!("db should have attribute {attr_name} for event {ev_id}"));
        assert_eq!(
            db_val, ref_val,
            "attr value mismatch for {ev_id}/{attr_name}"
        );

        assert_eq!(
            QueryableOCEL::get_ev_attr_val(&db, &ev_id, "definitely-not-an-attribute"),
            None
        );
    }

    // `events` is created dynamically by the sink, so raw-INSERT tests build it themselves.
    fn create_events_table(con: &Connection) {
        con.execute_batch(&build_events_table(&[])).unwrap();
    }

    fn insert_events(con: &Connection, ids: &[&str]) {
        for id in ids {
            con.execute(
                // The column is UTC-anchored, so a bare TIMESTAMP would be read in the session
                // timezone and make this test machine-dependent.
                "INSERT INTO events VALUES (?, 'et', TIMESTAMPTZ '2024-01-01 00:00:00+00')",
                duckdb::params![id],
            )
            .unwrap();
        }
    }

    fn collect_paged(con: &Connection, page_size: usize) -> Vec<String> {
        let pager = KeysetPager::new(con, DefaultSchema.all_events_page_sql(), page_size);
        pager.collect()
    }

    #[test]
    fn keyset_pager_page_size_does_not_divide_row_count() {
        let con = Connection::open_in_memory().unwrap();
        create_schema(&con).unwrap();
        create_events_table(&con);
        let raw_ids: Vec<String> = (0..7).map(|i| format!("id-{i:02}")).collect();
        let ids: Vec<&str> = raw_ids.iter().map(|s| s.as_str()).collect();
        insert_events(&con, &ids);

        let paged = collect_paged(&con, 3);
        let mut expected = raw_ids.clone();
        expected.sort();
        assert_eq!(paged.len(), 7);
        assert_eq!(paged, expected);
    }

    // Without an explicit `order_by`, neither the evaluator (binding-enumeration order) nor
    // DuckDB (query-plan order) pins row order, so sort before comparing.
    fn normalize(mut r: QueryResult) -> QueryResult {
        r.rows.sort();
        r
    }

    #[test]
    fn run_query_matches_evaluator_bindings_query() {
        use crate::core::event_data::object_centric::query::eval::evaluate;
        use crate::core::event_data::object_centric::query::*;

        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());
        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-query-parity-bindings.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        // "orders and their directly-related events" (mirrors eval.rs's base_box()).
        let root = Box {
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
        };
        let query = Query {
            root,
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::Id(1)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let eval_result = normalize(evaluate(&query, &reference).unwrap());
        let db_result = normalize(db.run_query(&query).unwrap());
        assert!(
            !eval_result.rows.is_empty(),
            "sanity: query should bind rows"
        );
        assert_eq!(eval_result, db_result);
    }

    #[test]
    fn run_query_matches_evaluator_projection_order_limit_query() {
        use crate::core::event_data::object_centric::query::eval::evaluate;
        use crate::core::event_data::object_centric::query::*;

        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());
        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-query-parity-proj.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        // Mirrors eval.rs's projection_order_and_limit. Ties on the primary order_by key (event
        // time) are broken by Id(0) so both engines pick the same top-N rows.
        let root = Box {
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
        };
        let query = Query {
            root,
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(1), Expr::Type(0), Expr::Time(0)],
                order_by: vec![(Expr::Time(0), Dir::Asc), (Expr::Id(0), Dir::Asc)],
                limit: Some(5),
            }),
            emits: Vec::new(),
        };

        let eval_result = evaluate(&query, &reference).unwrap();
        let db_result = db.run_query(&query).unwrap();
        assert_eq!(eval_result.rows.len(), 5);
        assert_eq!(eval_result, db_result);
    }

    /// orders (0) -O2O-> items (1) child box, mirroring `eval.rs`'s `order_items_child`.
    fn order_items_child(
        range: Option<(Option<f64>, Option<f64>)>,
    ) -> crate::core::event_data::object_centric::query::Box {
        use crate::core::event_data::object_centric::query::*;
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
    fn run_query_matches_evaluator_aggregate_type_counts() {
        use crate::core::event_data::object_centric::query::eval::evaluate;
        use crate::core::event_data::object_centric::query::*;

        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());
        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-query-parity-type-counts.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let root = Box {
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

        let eval_result = normalize(evaluate(&query, &reference).unwrap());
        let db_result = normalize(db.run_query(&query).unwrap());
        assert!(
            !eval_result.rows.is_empty(),
            "sanity: type_counts should produce groups"
        );
        assert_eq!(eval_result, db_result);
    }

    #[test]
    fn run_query_matches_evaluator_children_filter() {
        use crate::core::event_data::object_centric::query::eval::evaluate;
        use crate::core::event_data::object_centric::query::*;

        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());
        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-query-parity-children-filter.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

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

        let eval_result = normalize(evaluate(&query, &reference).unwrap());
        let db_result = normalize(db.run_query(&query).unwrap());
        assert!(
            !eval_result.rows.is_empty(),
            "sanity: some orders should have >= {N} items"
        );
        assert_eq!(eval_result, db_result);
    }

    #[test]
    fn run_query_matches_evaluator_child_scalar_output() {
        use crate::core::event_data::object_centric::query::eval::evaluate;
        use crate::core::event_data::object_centric::query::*;

        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());
        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-query-parity-child-scalar.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        let query = Query {
            root: order_items_child(None),
            output: Output::Rows(RowsSpec {
                project: vec![Expr::Id(0), Expr::ChildAgg(0, 0)],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let eval_result = normalize(evaluate(&query, &reference).unwrap());
        let db_result = normalize(db.run_query(&query).unwrap());
        assert!(
            !eval_result.rows.is_empty(),
            "sanity: query should bind rows"
        );
        assert_eq!(eval_result, db_result);
    }

    #[test]
    fn run_query_matches_evaluator_global_aggregate() {
        use crate::core::event_data::object_centric::query::eval::evaluate;
        use crate::core::event_data::object_centric::query::*;

        let src = order_management_path();
        let reference = IndexLinkedOCEL::from_ocel(import_ocel_json_path(&src).unwrap());
        let out = get_test_data_path()
            .join("export")
            .join("stream-db-linked-query-parity-global-agg.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let db = DuckDbLinkedOCEL::open(&out).unwrap();

        // group_by: [] -> one row, a single Count over all events.
        let root = Box {
            new_vars: vec![VarDecl {
                kind: VarKind::Event,
                types: TypeConstraint::Any,
            }],
            filters: vec![],
            children: vec![],
        };
        let query = Query {
            root,
            output: Output::Aggregate(AggSpec {
                group_by: vec![],
                aggregates: vec![Agg::Count],
                having: vec![],
                order_by: vec![],
                limit: None,
            }),
            emits: Vec::new(),
        };

        let eval_result = evaluate(&query, &reference).unwrap();
        let db_result = db.run_query(&query).unwrap();
        assert_eq!(eval_result.rows.len(), 1);
        assert_eq!(eval_result, db_result);
    }
}
