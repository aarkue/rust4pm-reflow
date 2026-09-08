//! A [`dbcon`]-backed [`RowProvider`]: `PostgreSQL`, `SQLite`, CSV and Parquet sources
//! (whatever `dbcon` itself supports), plus catalog discovery so a caller has a schema to
//! compile or validate against before it opens one of these.
//!
//! # Threading
//!
//! `dbcon`'s row-reading API is synchronous, and so is [`RowProvider`]: [`RowProvider::scan`]
//! is a direct call into [`DataSource::scan`], on the calling thread and blocking it.
//!
//! Connecting and discovering a schema are async, and happen once per source.
//! [`DbconRowProvider::connect`] and [`discover_catalog`] each drive that on a short-lived
//! thread with its own runtime, so both are safe to call from any thread, including one that
//! is already a Tokio worker.
//!
//! `PostgreSQL` is async underneath, so `dbcon` drives its own runtime once per scan and, from
//! inside another runtime, returns an error naming `spawn_blocking` rather than panicking.
//! **An async caller with a `PostgreSQL` source must wrap its scans in
//! `tokio::task::spawn_blocking`.**

use std::fmt;
use std::ops::ControlFlow;
use std::sync::mpsc;

use dbcon::{DataSource, NormalizedType, NormalizedValue};

use super::catalog::{
    ColumnSchema, ExtractionCatalog, TablePreview, TableSchema, UNTYPED_COL_TYPE,
};
use super::provider::{preview_rows, ProviderError, RowProvider};
use super::value::{Value, ValueKind};

/// Why connecting through [`DbconRowProvider`] or one of this module's free functions failed.
///
/// Distinct from [`ProviderError`], which covers a scan against an already-open connection.
/// `dbcon` reports connection failures as an unstructured `anyhow::Error`, so this carries a
/// message rather than inventing categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbconProviderError {
    /// Connecting to the source, or discovering its schema, failed. Carries `dbcon`'s message.
    Connect(String),
    /// A query against an already-open connection failed outside of [`RowProvider::scan`]
    /// (currently only [`DbconRowProvider::distinct_values`]). Carries `dbcon`'s message.
    Query(String),
    /// The short-lived thread that drives `dbcon`'s async connect/discover ended (it
    /// panicked, or could not be started) before it could reply.
    WorkerGone,
}

impl fmt::Display for DbconProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DbconProviderError::Connect(message) => write!(f, "connecting failed: {message}"),
            DbconProviderError::Query(message) => write!(f, "query failed: {message}"),
            DbconProviderError::WorkerGone => {
                write!(f, "the DbconRowProvider background thread is gone")
            }
        }
    }
}

impl std::error::Error for DbconProviderError {}

/// Run `job` on a short-lived thread with its own current-thread Tokio runtime, and block
/// until it returns.
///
/// A thread rather than a bare `Runtime::block_on` because `block_on` panics when the calling
/// thread is already inside any runtime's context, and this has to be callable from anywhere.
#[cfg(not(target_arch = "wasm32"))]
fn on_a_runtime<T, F>(thread_name: &str, job: F) -> Result<T, DbconProviderError>
where
    T: Send + 'static,
    F: FnOnce(&tokio::runtime::Runtime) -> T + Send + 'static,
{
    let (reply_tx, reply_rx) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(thread_name.to_string())
        .spawn(move || {
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => {
                    let _ = reply_tx.send(Ok(job(&rt)));
                }
                Err(e) => {
                    let _ = reply_tx.send(Err(e.to_string()));
                }
            }
        })
        .map_err(|e| DbconProviderError::Connect(e.to_string()))?;

    let result = reply_rx
        .recv()
        .map_err(|_| DbconProviderError::WorkerGone)?
        .map_err(DbconProviderError::Connect)?;
    let _ = handle.join();
    Ok(result)
}

/// [`on_a_runtime`] where there are no threads to hand off to.
///
/// `wasm32-unknown-unknown` has no `std::thread`, so the runtime is driven on the caller's own
/// stack. Sound only because SQLite, CSV and Parquet are all read from bytes already in memory,
/// so the futures never await real I/O. PostgreSQL would park forever, which is one more reason
/// it is behind `extraction-dbcon-postgres`.
#[cfg(target_arch = "wasm32")]
fn on_a_runtime<T, F>(_thread_name: &str, job: F) -> Result<T, DbconProviderError>
where
    T: Send + 'static,
    F: FnOnce(&tokio::runtime::Runtime) -> T + Send + 'static,
{
    // No `enable_all`: the I/O and time drivers it installs do not exist on this target.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| DbconProviderError::Connect(e.to_string()))?;
    Ok(job(&rt))
}

/// A [`RowProvider`] backed by [`dbcon`](https://github.com/aarkue/dbcon), giving [`scan`] a
/// connection to `PostgreSQL`, `SQLite`, a CSV file or a Parquet file -- whichever `dbcon`'s
/// own connection string dispatch recognises.
///
/// [`scan`]: RowProvider::scan
///
/// [`DataSource::scan`] abandons the query on [`ControlFlow::Break`], so breaking early also
/// saves the query's own time and transfer, not only the per-row work above this trait.
pub struct DbconRowProvider {
    ds: DataSource,
}

impl fmt::Debug for DbconRowProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DbconRowProvider").finish_non_exhaustive()
    }
}

impl DbconRowProvider {
    /// Connect to `connection_string` (a `postgres://`, `postgresql://`, `sqlite:`, `csv://`
    /// or `parquet://` URL, or a bare path ending in `.csv` or `.parquet` -- see
    /// `dbcon::DataSource::new_any`'s dispatch) without discovering its schema, matching the
    /// fast, query-only connection [`RowProvider::scan`] needs. Use [`discover_catalog`]
    /// separately to get a schema to compile or validate against before opening this.
    ///
    /// `name` is only used to label the connection in `dbcon`'s own bookkeeping. It has no
    /// effect on what `scan` returns.
    ///
    /// Blocks the calling thread until the connection is established or fails. Safe from any
    /// thread, including a Tokio worker.
    ///
    /// # Errors
    /// Returns [`DbconProviderError::Connect`] if `dbcon` cannot open the connection, or
    /// [`DbconProviderError::WorkerGone`] if the connecting thread ended before replying.
    pub fn connect(name: &str, connection_string: &str) -> Result<Self, DbconProviderError> {
        let name = name.to_string();
        let connection_string = connection_string.to_string();
        let ds = on_a_runtime("dbcon-connect", move |rt| {
            rt.block_on(DataSource::new_any_without_discovery(
                name,
                connection_string,
            ))
            .map_err(|e| e.to_string())
        })?
        .map_err(DbconProviderError::Connect)?;
        Ok(Self { ds })
    }

    /// Open a source from bytes already in memory, rather than from a path.
    ///
    /// `format` is a file extension: `csv`, `tsv`, or `parquet`. This is the route for contents
    /// with no name to open -- a browser upload, a `wasm32` build with no filesystem, or bytes
    /// fetched over a network.
    ///
    /// # Errors
    /// Returns [`DbconProviderError::Connect`] if the bytes cannot be read as `format`, or
    /// [`DbconProviderError::WorkerGone`] if the opening thread ended before replying.
    pub fn from_bytes(
        name: &str,
        format: &str,
        bytes: std::sync::Arc<[u8]>,
    ) -> Result<Self, DbconProviderError> {
        let name = name.to_string();
        let format = format.to_ascii_lowercase();
        let ds = on_a_runtime("dbcon-from-bytes", move |rt| {
            rt.block_on(async move {
                match format.as_str() {
                    "csv" | "tsv" => DataSource::new_csv_bytes(name, bytes)
                        .await
                        .map_err(|e| e.to_string()),
                    "parquet" => DataSource::new_parquet_bytes(name, bytes)
                        .await
                        .map_err(|e| e.to_string()),
                    "xlsx" => DataSource::new_xlsx_bytes(name, bytes)
                        .await
                        .map_err(|e| e.to_string()),
                    other => Err(format!(
                        "cannot read '{other}' from memory; expected csv, tsv, parquet or xlsx"
                    )),
                }
            })
        })?
        .map_err(DbconProviderError::Connect)?;
        Ok(Self { ds })
    }

    /// This source's schema as an [`ExtractionCatalog`] under `source_id`.
    ///
    /// Reads the tables `dbcon` discovered when the source was opened, rather than reconnecting
    /// as the free [`discover_catalog`] does -- which is the whole point for a byte-backed source,
    /// since there is no connection string to reopen from.
    #[must_use]
    pub fn discover_catalog(&self, source_id: &str) -> ExtractionCatalog {
        let mut catalog = ExtractionCatalog::new();
        for (table_name, info) in &self.ds.tables {
            let columns = info
                .columns
                .values()
                .map(|c| (c.name.clone(), col_type_of(&c.col_type), c.is_nullable));
            catalog = catalog.with_table(source_id, TableSchema::new(table_name, columns));
        }
        catalog
    }

    /// Run `SELECT DISTINCT` on `table.column` over this provider's already-open connection
    /// and return the distinct values as text, for populating
    /// [`ExtractionCatalog::with_domain`].
    ///
    /// Reuses the connection [`DbconRowProvider::connect`] already opened, unlike the free
    /// function [`discover_catalog`].
    ///
    /// # Errors
    /// Returns [`DbconProviderError::Query`] if the query fails.
    pub fn distinct_values(
        &self,
        table: &str,
        column: &str,
    ) -> Result<Vec<String>, DbconProviderError> {
        self.ds
            .get_distinct_values(table, column)
            .map_err(|e| DbconProviderError::Query(e.to_string()))
    }

    /// Read at most `limit` rows of `table`, for showing a person what the data looks like.
    ///
    /// [`preview_rows`] over this provider's own connection. `dbcon` exposes no `LIMIT`, so unlike
    /// [`distinct_values`](Self::distinct_values) this cannot better the generic function.
    ///
    /// # Errors
    /// Returns whatever [`RowProvider::scan`] reports for an unknown table or column.
    pub fn table_preview(
        &self,
        table: &str,
        columns: &[&str],
        limit: usize,
    ) -> Result<TablePreview, ProviderError> {
        preview_rows(self, table, columns, limit)
    }
}

impl RowProvider for DbconRowProvider {
    fn scan(
        &self,
        table: &str,
        columns: &[&str],
        f: &mut dyn FnMut(&[Value]) -> ControlFlow<()>,
    ) -> Result<(), ProviderError> {
        // One buffer for the whole scan, so a full-table scan allocates per scan, not per row.
        let mut row: Vec<Value> = Vec::with_capacity(columns.len());
        self.ds
            .scan(table, columns, None, &mut |values| {
                row.clear();
                row.extend(values.iter().map(convert));
                f(&row)
            })
            .map_err(|e| map_err(table, &e.to_string()))
    }
}

/// Convert one `dbcon` cell into this crate's [`Value`], mapping every variant explicitly.
///
/// - [`NormalizedValue::Null`] -- [`Value::Null`].
/// - [`NormalizedValue::Text`] -- [`Value::Text`].
/// - [`NormalizedValue::Integer`] -- [`Value::Integer`].
/// - [`NormalizedValue::Float`] -- [`Value::Float`].
/// - [`NormalizedValue::Boolean`] -- [`Value::Boolean`].
/// - [`NormalizedValue::Timestamp`] -- [`Value::Timestamp`], not text: every predicate that reads
///   a timestamp column checks `Value::kind()` first, and a `Text` cell falls back to the slower,
///   lossier multi-format parse cascade `dbcon` has already done the work of avoiding.
/// - [`NormalizedValue::Json`] and [`NormalizedValue::Unknown`] -- [`Value::Text`], see below.
///
/// # `Json` and `Unknown` become `Text`, not `Null`
///
/// Both carry real data with no dedicated `Value` case, and `Null` would leave a column a
/// blueprint genuinely reads silently and permanently empty. `Json` becomes its compact
/// serialisation, `Unknown` the string `dbcon` already produced, and a `Matches` predicate can
/// search either. Extracting one JSON field belongs in a source-side view.
///
/// Semantics a SQL compiler must reproduce: a compiled view's JSON column has to render through an
/// explicit textual cast, or `Matches` would see different text on the two paths.
fn convert(v: &NormalizedValue) -> Value {
    match v {
        NormalizedValue::Null => Value::Null,
        NormalizedValue::Text(s) | NormalizedValue::Unknown(s) => Value::Text(s.clone()),
        NormalizedValue::Integer(i) => Value::Integer(*i),
        NormalizedValue::Float(f) => Value::Float(*f),
        NormalizedValue::Boolean(b) => Value::Boolean(*b),
        NormalizedValue::Timestamp(t) => Value::Timestamp(*t),
        NormalizedValue::Json(j) => Value::Text(j.to_string()),
    }
}

/// [`ColumnSchema::declared_kind`] applied to a bare `col_type` string.
fn declared_kind_of(col_type: &str) -> Option<ValueKind> {
    ColumnSchema {
        name: String::new(),
        col_type: col_type.to_string(),
        nullable: true,
    }
    .declared_kind()
}

/// Map a [`NormalizedType`] to the `col_type` spelling [`ColumnSchema::declared_kind`] already
/// recognises, so a column [`discover_catalog`] finds still gets literal coercion.
///
/// **The `col_type` this produces must never make `declared_kind` claim a [`ValueKind`] that
/// `dbcon` itself declined to claim.** The six classified variants map to spellings
/// `declared_kind` reads back as the matching kind. [`NormalizedType::Json`] and
/// [`NormalizedType::Unknown`] must both read back as `None`.
///
/// `Unknown` covers every binary, array and range type and every `SQLite` NUMERIC-affinity column,
/// all of which decode per value. Its `declared` spelling is forwarded only when `declared_kind`
/// says `None` for it, since an array's element type is a word of its own and `int4[]` would
/// otherwise read back as an integer. Everything else becomes
/// [`UNTYPED_COL_TYPE`](super::catalog::UNTYPED_COL_TYPE), the original spellings still being
/// available from `dbcon::DataSource::unknown_column_types`.
fn col_type_of(t: &NormalizedType) -> String {
    match t {
        NormalizedType::Text => "TEXT".to_string(),
        NormalizedType::Integer => "INTEGER".to_string(),
        NormalizedType::Float => "DOUBLE PRECISION".to_string(),
        NormalizedType::Boolean => "BOOLEAN".to_string(),
        NormalizedType::Timestamp => "TIMESTAMP".to_string(),
        NormalizedType::Json => "JSON".to_string(),
        NormalizedType::Unknown(declared) => {
            if declared_kind_of(declared).is_none() {
                declared.clone()
            } else {
                UNTYPED_COL_TYPE.to_string()
            }
        }
    }
}

/// Connect to `connection_string`, discover its schema, and record every table `dbcon` finds
/// under `source_id` in a fresh [`ExtractionCatalog`] -- the schema a caller needs before it
/// can [`compile`](super::compile::compile) or [`validate`](super::validate::validate) a
/// blueprint against this source, and before it opens the faster, non-discovering
/// [`DbconRowProvider`] to actually run one.
///
/// A one-off connection, separate from any [`DbconRowProvider`]: discovery walks every table
/// `dbcon` can see, which [`DbconRowProvider::connect`] skips. Run it once ahead of execution, or
/// cache it.
///
/// Blocks the calling thread. See the module docs for the `spawn_blocking` obligation on an async
/// caller.
///
/// # Errors
/// Returns [`DbconProviderError::Connect`] if `dbcon` cannot connect or discover the schema.
pub fn discover_catalog(
    source_id: &str,
    connection_string: &str,
) -> Result<ExtractionCatalog, DbconProviderError> {
    let name = source_id.to_string();
    let connection_string = connection_string.to_string();
    let tables = on_a_runtime("dbcon-discover-catalog", move |rt| {
        rt.block_on(DataSource::new_any(name, connection_string))
            .map(|ds| ds.tables)
            .map_err(|e| e.to_string())
    })?
    .map_err(DbconProviderError::Connect)?;

    let mut catalog = ExtractionCatalog::new();
    for (table_name, info) in tables {
        let columns = info
            .columns
            .into_values()
            .map(|c| (c.name, col_type_of(&c.col_type), c.is_nullable));
        catalog = catalog.with_table(source_id, TableSchema::new(&table_name, columns));
    }
    Ok(catalog)
}

/// Translate one of `dbcon`'s `anyhow::Error` messages into a [`ProviderError`], recognising
/// the patterns its backends (`SQLite` via `rusqlite`, `PostgreSQL` via `sqlx`, and CSV) are
/// each known to emit for an unknown table or column, and falling back to
/// [`ProviderError::Backend`] for everything else.
///
/// Best-effort: `dbcon` reports failures as a stringly-typed `anyhow::Error`. An unrecognised
/// message still reaches the caller, as [`ProviderError::Backend`].
fn map_err(table: &str, message: &str) -> ProviderError {
    let lower = message.to_ascii_lowercase();
    // SQLite: "no such table: foo" / "no such column: bar".
    if let Some(e) = super::provider::sqlite_message_error(table, message) {
        return e;
    }
    // PostgreSQL: `relation "foo" does not exist` / `column "bar" does not exist`.
    if lower.contains("relation") && lower.contains("does not exist") {
        return ProviderError::UnknownTable {
            table: table.to_string(),
        };
    }
    if lower.contains("column") && lower.contains("does not exist") {
        if let Some(column) = extract_quoted(message, '"') {
            return ProviderError::UnknownColumn {
                table: table.to_string(),
                column,
            };
        }
    }
    // CSV: `Column 'foo' not found in CSV`.
    if lower.contains("not found in csv") {
        if let Some(column) = extract_quoted(message, '\'') {
            return ProviderError::UnknownColumn {
                table: table.to_string(),
                column,
            };
        }
    }
    ProviderError::Backend {
        table: table.to_string(),
        message: message.to_string(),
    }
}

/// The first substring of `message` delimited by a *pair* of `quote` characters.
///
/// `None` when nothing closes the quote, rather than the rest of the message -- which would name
/// the driver's own error text as the missing column.
fn extract_quoted(message: &str, quote: char) -> Option<String> {
    let mut parts = message.splitn(3, quote);
    parts.next()?;
    let quoted = parts.next()?;
    parts.next()?;
    Some(quoted.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::catalog::Catalog;
    use super::*;
    use std::io::Write;

    // -- NormalizedValue -> Value conversion: exhaustive, no database needed. --------------

    #[test]
    fn null_converts_to_null() {
        assert_eq!(convert(&NormalizedValue::Null), Value::Null);
    }

    #[test]
    fn text_converts_to_text() {
        assert_eq!(
            convert(&NormalizedValue::Text("hi".to_string())),
            Value::Text("hi".to_string())
        );
    }

    #[test]
    fn integer_converts_to_integer() {
        assert_eq!(convert(&NormalizedValue::Integer(42)), Value::Integer(42));
    }

    #[test]
    fn float_converts_to_float() {
        assert_eq!(convert(&NormalizedValue::Float(1.5)), Value::Float(1.5));
    }

    #[test]
    fn boolean_converts_to_boolean() {
        assert_eq!(
            convert(&NormalizedValue::Boolean(true)),
            Value::Boolean(true)
        );
    }

    #[test]
    fn timestamp_converts_to_timestamp_not_text() {
        let ts = chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z").unwrap();
        let converted = convert(&NormalizedValue::Timestamp(ts));
        assert_eq!(converted, Value::Timestamp(ts));
        // The whole point: it must carry `ValueKind::Timestamp`, not `Text`, so the fast path
        // in `TimestampSource::parse` and `Compare`'s literal coercion both see it.
        assert_eq!(converted.kind(), Some(ValueKind::Timestamp));
    }

    #[test]
    fn json_converts_to_its_compact_text_form() {
        let json = serde_json::json!({"a": 1, "b": [true, null]});
        let expected = json.to_string();
        assert_eq!(convert(&NormalizedValue::Json(json)), Value::Text(expected));
    }

    #[test]
    fn unknown_converts_to_its_carried_text() {
        assert_eq!(
            convert(&NormalizedValue::Unknown("raw-driver-text".to_string())),
            Value::Text("raw-driver-text".to_string())
        );
    }

    // -- col_type mapping, round-tripped through `ColumnSchema::declared_kind`. -------------

    #[test]
    fn col_type_mapping_round_trips_through_declared_kind_for_real_engine_spellings() {
        let cases: &[(&str, ValueKind)] = &[
            ("int4", ValueKind::Integer),
            ("INTEGER", ValueKind::Integer),
            ("BIGINT", ValueKind::Integer),
            ("VARCHAR", ValueKind::Text),
            ("TEXT", ValueKind::Text),
            ("timestamp", ValueKind::Timestamp),
            ("TIMESTAMPTZ", ValueKind::Timestamp),
            ("DOUBLE PRECISION", ValueKind::Float),
            ("REAL", ValueKind::Float),
            ("NUMERIC", ValueKind::Float),
            ("BOOLEAN", ValueKind::Boolean),
            ("bool", ValueKind::Boolean),
        ];
        for &(spelling, expected) in cases {
            let normalized = NormalizedType::from_raw(spelling);
            let col_type = col_type_of(&normalized);
            let schema = ColumnSchema {
                name: "c".to_string(),
                col_type,
                nullable: true,
            };
            assert_eq!(
                schema.declared_kind(),
                Some(expected),
                "spelling {spelling:?} (normalized to {normalized:?}) did not declare {expected:?}"
            );
        }
    }

    #[test]
    fn json_col_type_declares_no_kind_same_as_the_value_side_has_no_json_kind() {
        let schema = ColumnSchema {
            name: "c".to_string(),
            col_type: col_type_of(&NormalizedType::Json),
            nullable: true,
        };
        assert_eq!(schema.declared_kind(), None);
    }

    // -- `NormalizedType::Unknown`: dbcon's "do not assume" signal must not become a claim. ---

    /// Declared spellings `dbcon` reports as `Unknown`, taken from its own documented list:
    /// binary types, array types, range types, `SQLite` BLOB/NUMERIC-affinity columns, and a
    /// column declared with no type at all.
    const UNKNOWN_SPELLINGS: &[&str] = &[
        "blob",
        "bytea",
        "varbinary",
        "image",
        "raw",
        "int4[]",
        "_int4",
        "text[]",
        "int4range",
        "daterange",
        "numrange",
        "tsvector",
        "point",
        "bit varying",
        "blah",
        "",
    ];

    #[test]
    fn unknown_col_type_never_declares_a_kind_dbcon_declined_to_declare() {
        for spelling in UNKNOWN_SPELLINGS {
            let unknown = NormalizedType::Unknown((*spelling).to_string());
            let col_type = col_type_of(&unknown);
            assert_eq!(
                declared_kind_of(&col_type),
                None,
                "Unknown({spelling:?}) produced col_type {col_type:?}, which declares a kind; \
                 dbcon said it could not classify this column, so nothing downstream may coerce \
                 literals against it"
            );
        }
    }

    #[test]
    fn unknown_keeps_an_inert_declared_spelling_but_replaces_a_misleading_one() {
        // Inert: `declared_kind` already says `None`, so the spelling survives as a diagnostic.
        for inert in ["blob", "bytea", "point", "daterange", "int4range"] {
            assert_eq!(
                col_type_of(&NormalizedType::Unknown(inert.to_string())),
                inert,
                "`declared_kind` reads a col_type word by word, so a range type names no kind \
                 and its spelling is worth keeping"
            );
        }
        // Still misleading: an array's element type is a word of its own, so `int4[]` and
        // `_int4` do name a kind.
        for misleading in ["int4[]", "_int4", "text[]"] {
            assert_eq!(
                col_type_of(&NormalizedType::Unknown(misleading.to_string())),
                UNTYPED_COL_TYPE
            );
        }
    }

    #[test]
    fn a_binary_column_is_unknown_not_text_and_so_gets_no_coercion() {
        // The behaviour change with teeth: dbcon used to guess `Text` here (its old
        // `ends_with("char")|ends_with("text")` heuristic), which made a BLOB column look safe
        // to compare as a string. It is `Unknown` now, and must stay uncoerced.
        let blob = NormalizedType::from_sqlite_declared("BLOB");
        assert_eq!(blob, NormalizedType::Unknown("blob".to_string()));
        assert_eq!(declared_kind_of(&col_type_of(&blob)), None);
    }

    #[test]
    fn every_classified_normalized_type_round_trips_to_its_own_kind() {
        let cases: &[(NormalizedType, Option<ValueKind>)] = &[
            (NormalizedType::Text, Some(ValueKind::Text)),
            (NormalizedType::Integer, Some(ValueKind::Integer)),
            (NormalizedType::Float, Some(ValueKind::Float)),
            (NormalizedType::Boolean, Some(ValueKind::Boolean)),
            (NormalizedType::Timestamp, Some(ValueKind::Timestamp)),
            // `ValueKind` has no JSON case; `convert` lands JSON as `Value::Text`, but claiming
            // `Text` here would let a `Compare` literal coerce against the raw serialisation.
            (NormalizedType::Json, None),
        ];
        for (normalized, expected) in cases {
            assert_eq!(
                declared_kind_of(&col_type_of(normalized)),
                *expected,
                "{normalized:?} did not round-trip"
            );
        }
    }

    // -- End-to-end `scan` over a temporary CSV file: no server required. -------------------

    fn write_temp_csv(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn scan_streams_rows_from_a_csv_file() {
        let file = write_temp_csv("id,name,active\n1,alice,true\n2,bob,false\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let mut rows: Vec<Vec<Value>> = Vec::new();
        provider
            .scan("main", &["id", "name"], &mut |row| {
                rows.push(row.to_vec());
                ControlFlow::Continue(())
            })
            .unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            vec![
                Value::Text("1".to_string()),
                Value::Text("alice".to_string())
            ]
        );
        assert_eq!(
            rows[1],
            vec![Value::Text("2".to_string()), Value::Text("bob".to_string())]
        );
    }

    #[test]
    fn table_preview_stops_at_the_limit_and_keeps_column_order() {
        let file = write_temp_csv("id,name\n1,alice\n2,bob\n3,carol\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let preview = provider.table_preview("main", &["id", "name"], 2).unwrap();

        assert_eq!(preview.columns, vec!["id".to_string(), "name".to_string()]);
        assert_eq!(preview.rows.len(), 2, "the limit bounds the scan");
        assert_eq!(
            preview.rows[0],
            vec![Some("1".to_string()), Some("alice".to_string())]
        );
    }

    #[test]
    fn table_preview_with_a_zero_limit_reads_nothing() {
        let file = write_temp_csv("id\n1\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let preview = provider.table_preview("main", &["id"], 0).unwrap();

        // Still reports its shape: this is the path a "previews off" setting takes.
        assert_eq!(preview.columns, vec!["id".to_string()]);
        assert!(preview.rows.is_empty());
    }

    #[test]
    fn table_preview_projects_per_column_examples_deduplicated() {
        let file = write_temp_csv("status\ndraft\ndraft\ndone\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let preview = provider.table_preview("main", &["status"], 5).unwrap();

        assert_eq!(preview.column_values("status", 5), vec!["draft", "done"]);
        assert!(preview.column_values("nonexistent", 5).is_empty());
    }

    #[test]
    fn scan_breaks_early_without_calling_f_again() {
        let file = write_temp_csv("id\n1\n2\n3\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let mut seen = 0;
        provider
            .scan("main", &["id"], &mut |_row| {
                seen += 1;
                ControlFlow::Break(())
            })
            .unwrap();

        assert_eq!(seen, 1, "f must not be called again after it breaks");
    }

    #[test]
    fn scan_with_empty_columns_calls_f_once_per_row_with_an_empty_slice() {
        let file = write_temp_csv("id,name\n1,alice\n2,bob\n3,carol\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let mut row_count = 0;
        provider
            .scan("main", &[], &mut |row| {
                assert!(
                    row.is_empty(),
                    "empty columns must yield an empty row, not every column"
                );
                row_count += 1;
                ControlFlow::Continue(())
            })
            .unwrap();

        assert_eq!(row_count, 3);
    }

    #[test]
    fn scan_of_an_unknown_column_is_a_provider_error() {
        let file = write_temp_csv("id\n1\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let err = provider
            .scan("main", &["nope"], &mut |_| ControlFlow::Continue(()))
            .unwrap_err();
        assert!(
            matches!(
                err,
                ProviderError::UnknownColumn { .. } | ProviderError::Backend { .. }
            ),
            "unexpected error variant: {err:?}"
        );
    }

    // -- Catalog discovery and the distinct-values helper, both over the same CSV file. -----

    #[test]
    fn discover_catalog_finds_the_csv_s_single_table_as_all_text_columns() {
        let file = write_temp_csv("id,name\n1,alice\n2,bob\n");
        let catalog = discover_catalog("erp", file.path().to_str().unwrap()).unwrap();

        let table = catalog
            .table("erp", "main")
            .expect("csv table 'main' present");
        assert_eq!(table.columns.len(), 2);
        for col in table.columns.values() {
            // CSV carries no type information; dbcon reports every column as Text.
            assert_eq!(col.declared_kind(), Some(ValueKind::Text));
            assert!(col.nullable);
        }
    }

    #[test]
    fn distinct_values_reuses_the_open_connection() {
        let file = write_temp_csv("status\ndraft\nsale\ndraft\n");
        let provider =
            DbconRowProvider::connect("csv-test", file.path().to_str().unwrap()).unwrap();

        let mut values = provider.distinct_values("main", "status").unwrap();
        values.sort();
        assert_eq!(values, vec!["draft".to_string(), "sale".to_string()]);
    }
}
