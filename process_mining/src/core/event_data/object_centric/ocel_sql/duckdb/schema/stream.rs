//! Public entry points: own the connection, create schema, import inside one transaction, finalize.
use std::fs::File;
use std::path::Path;

use duckdb::Connection;

use crate::core::event_data::object_centric::appendable::{AppendableOCEL, StreamImportOCEL};
use crate::core::event_data::object_centric::io::OCELIOError;
use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;
use crate::core::event_data::object_centric::ocel_struct::OCEL;
use crate::core::event_data::object_centric::ocel_xml::OCELImportOptions;
use crate::core::event_data::object_centric::readable::ReadableOCEL;
use crate::core::io::infer_format_from_path;
use macros_process_mining::register_binding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::sink::DuckDbOcelSink;
use super::tables::{create_indexes, create_schema};

fn open_fresh(db_path: &Path) -> Result<Connection, OCELIOError> {
    if db_path.exists() {
        let _ = std::fs::remove_file(db_path);
    }
    Ok(Connection::open(db_path)?)
}

/// Options controlling how an OCEL is streamed into a `DuckDB` database.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DuckDbImportOptions {
    /// Whether `DuckDB` compresses columns. `true` (default) = smaller, faster-to-query
    /// file; `false` = ~20% faster import, larger file.
    pub compression: bool,
    /// Whether to rewrite tables clustered by key after import. `true` (default) reorders
    /// rows (events by type+time, relations by key) so similar values group -- better
    /// compression + range scans; drops then rebuilds indexes/PKs. `false` skips it.
    pub optimize_filesize: bool,
}

impl Default for DuckDbImportOptions {
    fn default() -> Self {
        Self {
            compression: true,
            optimize_filesize: true,
        }
    }
}

/// Run `import` (which appends into the sink) inside one transaction, then finalize.
pub(super) fn run_import<F>(
    db_path: &Path,
    options: &DuckDbImportOptions,
    import: F,
) -> Result<(), OCELIOError>
where
    F: FnOnce(&mut DuckDbOcelSink<'_>) -> Result<(), OCELIOError>,
{
    let con = open_fresh(db_path)?;
    if !options.compression {
        con.execute_batch("PRAGMA force_compression='uncompressed';")?;
    }
    create_schema(&con)?;
    con.execute_batch("BEGIN TRANSACTION")?;
    {
        let mut sink = DuckDbOcelSink::new(&con)?;
        import(&mut sink)?;
        sink.finalize()?;
    }
    if options.optimize_filesize {
        // Rewrite tables clustered by key for better compression + scan locality. The
        // `CREATE TABLE AS SELECT` rewrite drops indexes and primary keys; both are
        // rebuilt below (PKs here, secondary indexes in `create_indexes`).
        con.execute_batch("
            -- 1. Events: Group by type, then time
            CREATE TABLE events_new AS SELECT * FROM events ORDER BY ocel_type, time;
            DROP TABLE events;
            ALTER TABLE events_new RENAME TO events;
            ALTER TABLE events ADD PRIMARY KEY (id);

            -- 2. Event-to-Object: Group by event_id, then qualifier
            CREATE TABLE e2o_new AS SELECT * FROM e2o ORDER BY event_id, qualifier;
            DROP TABLE e2o;
            ALTER TABLE e2o_new RENAME TO e2o;

            -- 3. Objects: Group by object type
            CREATE TABLE objects_new AS SELECT * FROM objects ORDER BY ocel_type;
            DROP TABLE objects;
            ALTER TABLE objects_new RENAME TO objects;
            ALTER TABLE objects ADD PRIMARY KEY (id);

            -- 4. Object Attributes (EAV): Group by attribute name, then type
            CREATE TABLE oac_new AS SELECT * FROM object_attribute_changes ORDER BY name, value_type;
            DROP TABLE object_attribute_changes;
            ALTER TABLE oac_new RENAME TO object_attribute_changes;

            -- 5. Object-to-Object: Group by qualifier
            CREATE TABLE o2o_new AS SELECT * FROM o2o ORDER BY qualifier;
            DROP TABLE o2o;
            ALTER TABLE o2o_new RENAME TO o2o;
        ")?;
    }
    // Build secondary indexes last, so the optimize rewrite above cannot drop them.
    create_indexes(&con)?;
    con.execute_batch("COMMIT")?;
    con.execute_batch("CHECKPOINT")?;
    con.execute_batch("VACUUM")?;
    Ok(())
}

/// Write an in-memory OCEL into a fresh `DuckDB` database in the consolidated schema.
///
/// For writing the OCEL 2.0 standard per-type layout instead, see [`export_ocel_duckdb_to_path`](crate::core::event_data::object_centric::ocel_sql::export_ocel_duckdb_to_path).
pub fn write_ocel_to_duckdb_with<O: ReadableOCEL + ?Sized>(
    ocel: &O,
    db_path: impl AsRef<Path>,
    options: &DuckDbImportOptions,
) -> Result<(), OCELIOError> {
    run_import(db_path.as_ref(), options, |sink| {
        for et in ocel.event_types() {
            sink.declare_event_type(et.clone())?;
        }
        for ot in ocel.object_types() {
            sink.declare_object_type(ot.clone())?;
        }
        for e in ocel.iter_events() {
            let e = e.into_owned();
            sink.append_event(e.id, &e.event_type, e.time, e.attributes, e.relationships)?;
        }
        for o in ocel.iter_objects() {
            let o = o.into_owned();
            sink.append_object(o.id, &o.object_type, o.attributes, o.relationships)?;
        }
        Ok(())
    })
}

/// Write an in-memory OCEL into a fresh `DuckDB` database in the consolidated schema.
///
/// Like [`write_ocel_to_duckdb_with`] with default [`DuckDbImportOptions`].
pub fn write_ocel_to_duckdb<O: ReadableOCEL + ?Sized>(
    ocel: &O,
    db_path: impl AsRef<Path>,
) -> Result<(), OCELIOError> {
    write_ocel_to_duckdb_with(ocel, db_path, &DuckDbImportOptions::default())
}

/// Write an in-memory OCEL into a fresh `DuckDB` database in the consolidated schema.
#[register_binding(name = "write_ocel_to_consolidated_duckdb", stringify_error)]
fn write_ocel_to_duckdb_binding(
    ocel: &OCEL,
    db_path: impl AsRef<Path>,
    #[bind(default = Default::default())] options: &DuckDbImportOptions,
) -> Result<(), OCELIOError> {
    write_ocel_to_duckdb_with(ocel, db_path, options)
}

/// Write an in-memory Slim OCEL into a fresh `DuckDB` database in the consolidated schema.
#[register_binding(name = "write_slim_ocel_to_consolidated_duckdb", stringify_error)]
fn write_slim_ocel_to_duckdb_binding(
    ocel: &SlimLinkedOCEL,
    db_path: impl AsRef<Path>,
    #[bind(default = Default::default())] options: &DuckDbImportOptions,
) -> Result<(), OCELIOError> {
    write_ocel_to_duckdb_with(ocel, db_path, options)
}

/// Stream an OCEL file into a fresh `DuckDB` database, dispatching by extension: `.json`,
/// `.xml`, `.sqlite`/`.db`/`.sqlite3` (and `.gz` variants).
///
/// Read the result back with
/// [`read_ocel_from_duckdb`](super::reader::read_ocel_from_duckdb) or
/// [`SlimLinkedOCEL::from_duckdb`](crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL::from_duckdb).
///
/// # Timestamps
///
/// Stored as `TIMESTAMPTZ` holding the UTC instant. The source's UTC offset is **not**
/// preserved: `2023-01-01T10:00:00+02:00` reads back as `2023-01-01T08:00:00+00:00`.
///
/// # Event-attribute columns
///
/// Columns on the wide `events` table come from the event-type declarations, and are typed
/// accordingly. An attribute that is never declared still gets a column, added on the fly
/// as `VARCHAR`; no attribute is dropped.
pub fn stream_ocel_file_to_duckdb<P: AsRef<Path>, Q: AsRef<Path>>(
    src_path: P,
    db_path: Q,
) -> Result<(), OCELIOError> {
    stream_ocel_file_to_duckdb_with(src_path, db_path, &DuckDbImportOptions::default())
}

/// Like [`stream_ocel_file_to_duckdb`] with explicit [`DuckDbImportOptions`].
#[register_binding(stringify_error, name = "stream_ocel_to_duckdb")]
pub fn stream_ocel_file_to_duckdb_with(
    src_path: impl AsRef<Path>,
    db_path: impl AsRef<Path>,
    #[bind(default = Default::default())] options: &DuckDbImportOptions,
) -> Result<(), OCELIOError> {
    let src = src_path.as_ref();
    let db_path = db_path.as_ref();
    let format = infer_format_from_path(src).ok_or_else(|| {
        OCELIOError::UnsupportedFormat(format!("cannot infer OCEL format from {src:?}"))
    })?;
    match format.as_str() {
        // SQLite is materialized to a full OCEL then fed to the sink (see `sqlite_source`),
        // not truly streamed -- v1 limitation.
        "sqlite" | "db" | "sqlite3" => {
            #[cfg(feature = "ocel-sqlite")]
            {
                super::sqlite_source::stream_ocel_sqlite_to_duckdb(src, db_path, options)
            }
            #[cfg(not(feature = "ocel-sqlite"))]
            {
                Err(OCELIOError::Other(
                    "ocel-sqlite feature required for .sqlite source".into(),
                ))
            }
        }
        // Everything else (`json`/`xml` and their `.gz` variants) streams into the sink;
        // `stream_ocel_from_reader` validates the format and rejects unsupported ones.
        _ => run_import(db_path, options, |sink| {
            sink.stream_ocel_from_reader(File::open(src)?, &format, OCELImportOptions::default())
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_data::object_centric::ocel_json::import_ocel_json_path;
    use crate::test_utils::get_test_data_path;

    #[test]
    fn json_stream_roundtrip_counts() {
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let reference = import_ocel_json_path(&src).unwrap();

        let out = get_test_data_path()
            .join("export")
            .join("stream-order-mgmt.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();

        let con = Connection::open(&out).unwrap();
        let n_ev: i64 = con
            .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
            .unwrap();
        let n_ob: i64 = con
            .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
            .unwrap();
        let n_types: i64 = con
            .query_row("SELECT count(DISTINCT ocel_type) FROM events", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(n_ev as usize, reference.events.len());
        assert_eq!(n_ob as usize, reference.objects.len());
        assert_eq!(n_types as usize, reference.event_types.len());
    }

    #[test]
    fn uncompressed_option_roundtrips() {
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let reference = import_ocel_json_path(&src).unwrap();
        let out = get_test_data_path()
            .join("export")
            .join("stream-uncompressed.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb_with(
            &src,
            &out,
            &DuckDbImportOptions {
                compression: false,
                ..Default::default()
            },
        )
        .unwrap();
        let con = Connection::open(&out).unwrap();
        let n_ev: i64 = con
            .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_ev as usize, reference.events.len());
    }

    #[test]
    fn gz_stream_roundtrip_counts() {
        use std::io::Write;
        // Proves the `.gz` path: gzip a JSON fixture, then stream the `.json.gz` in. Format
        // (incl. the `.gz` layer) is resolved centrally via `StreamImportOCEL`.
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let reference = import_ocel_json_path(&src).unwrap();
        let raw = std::fs::read(&src).unwrap();

        let gz_path = get_test_data_path()
            .join("export")
            .join("order-management.json.gz");
        {
            let f = std::fs::File::create(&gz_path).unwrap();
            let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            enc.write_all(&raw).unwrap();
            enc.finish().unwrap();
        }

        let out = get_test_data_path().join("export").join("stream-gz.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&gz_path, &out).unwrap();

        let con = Connection::open(&out).unwrap();
        let n_ev: i64 = con
            .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
            .unwrap();
        let n_ob: i64 = con
            .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_ev as usize, reference.events.len());
        assert_eq!(n_ob as usize, reference.objects.len());
    }

    #[test]
    fn default_import_has_indexes_and_pks() {
        // The default import runs the optimize_filesize rewrite, which drops indexes and
        // PKs; run_import must rebuild all of them. Guards the D1 regression.
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.json");
        let out = get_test_data_path()
            .join("export")
            .join("stream-index-check.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let con = Connection::open(&out).unwrap();

        let mut idx: Vec<String> = con
            .prepare("SELECT index_name FROM duckdb_indexes()")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        idx.sort();
        for expected in [
            "e2o_event",
            "e2o_object",
            "o2o_source",
            "o2o_target",
            "object_attribute_changes_id",
        ] {
            assert!(
                idx.iter().any(|i| i == expected),
                "missing index {expected}; have {idx:?}"
            );
        }

        let n_pk: i64 = con
            .query_row(
                "SELECT count(*) FROM duckdb_constraints() \
                 WHERE constraint_type = 'PRIMARY KEY' AND table_name IN ('events','objects')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n_pk, 2, "expected PKs on events + objects");
    }

    #[test]
    fn xml_stream_roundtrip_counts() {
        use crate::core::event_data::object_centric::ocel_xml::xml_ocel_import::import_ocel_xml_path;

        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.xml");
        let reference = import_ocel_xml_path(&src).unwrap();

        let out = get_test_data_path()
            .join("export")
            .join("stream-order-mgmt-xml.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();

        let con = Connection::open(&out).unwrap();
        let n_ev: i64 = con
            .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
            .unwrap();
        let n_ob: i64 = con
            .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_ev as usize, reference.events.len());
        assert_eq!(n_ob as usize, reference.objects.len());
    }

    #[test]
    fn xml_stream_roundtrip_counts_ck() {
        use crate::core::event_data::object_centric::ocel_xml::xml_ocel_import::import_ocel_xml_path;

        let src = get_test_data_path()
            .join("ocel")
            .join("ContainerLogistics.xml");
        let reference = import_ocel_xml_path(&src).unwrap();

        let out = get_test_data_path()
            .join("export")
            .join("stream-container-logistics-xml.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();

        let con = Connection::open(&out).unwrap();
        let n_ev: i64 = con
            .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
            .unwrap();
        let n_ob: i64 = con
            .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_ev as usize, reference.events.len());
        assert_eq!(n_ob as usize, reference.objects.len());
    }

    #[test]
    fn json_stream_attribute_and_relationship_fidelity() {
        use super::super::tables::quote_ident;
        use super::super::value::duck_value_to_ocel;

        // order-management.json has zero events with any attributes (verified against
        // the fixture); ocel2-p2p.json has events with both attributes and
        // relationships, so it exercises the round-trip this test targets.
        let src = get_test_data_path().join("ocel").join("ocel2-p2p.json");
        let reference = import_ocel_json_path(&src).unwrap();
        let out = get_test_data_path()
            .join("export")
            .join("stream-fidelity.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let con = Connection::open(&out).unwrap();

        // Pick one event that has at least one attribute and one relationship.
        let ev = reference
            .events
            .iter()
            .find(|e| !e.attributes.is_empty() && !e.relationships.is_empty())
            .expect("an event with attrs + rels");

        // Attribute value round-trips via the typed wide column.
        let a = &ev.attributes[0];
        let value: duckdb::types::Value = con
            .query_row(
                &format!("SELECT {} FROM events WHERE id = ?", quote_ident(&a.name)),
                duckdb::params![ev.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(duck_value_to_ocel(value), a.value);

        // Relationship count matches.
        let n_rel: i64 = con
            .query_row(
                "SELECT count(*) FROM e2o WHERE event_id = ?",
                duckdb::params![ev.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n_rel as usize, ev.relationships.len());
    }
}
