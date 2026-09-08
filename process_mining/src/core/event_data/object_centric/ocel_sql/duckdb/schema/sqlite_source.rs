//! `.sqlite` source -> `DuckDB`: materialize a full OCEL, then feed it to the streaming sink.
#![cfg(all(feature = "ocel-duckdb", feature = "ocel-sqlite"))]

use std::path::Path;

use crate::core::event_data::object_centric::io::OCELIOError;
use crate::core::event_data::object_centric::ocel_sql::import_ocel_sqlite_from_path;

use super::stream::{write_ocel_to_duckdb_with, DuckDbImportOptions};

/// Load a `.sqlite` OCEL into `DuckDB`, reached via
/// [`stream_ocel_file_to_duckdb`](super::stream::stream_ocel_file_to_duckdb).
///
/// TODO: Currently reuses `import_ocel_sqlite_from_path` (whole-file read).
/// Streaming from `SQLite` to `DuckDB` is future work.
pub(super) fn stream_ocel_sqlite_to_duckdb<P: AsRef<Path>, Q: AsRef<Path>>(
    sqlite_path: P,
    db_path: Q,
    options: &DuckDbImportOptions,
) -> Result<(), OCELIOError> {
    let ocel = import_ocel_sqlite_from_path(sqlite_path)?;
    write_ocel_to_duckdb_with(&ocel, db_path, options)
}

#[cfg(test)]
mod tests {
    use crate::core::event_data::object_centric::ocel_sql::{
        import_ocel_sqlite_from_path, stream_ocel_file_to_duckdb,
    };
    use crate::test_utils::get_test_data_path;
    use duckdb::Connection;

    #[test]
    fn sqlite_stream_roundtrip_counts() {
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.sqlite");
        let reference = import_ocel_sqlite_from_path(&src).unwrap();
        let out = get_test_data_path()
            .join("export")
            .join("stream-from-sqlite.duckdb");
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
}
