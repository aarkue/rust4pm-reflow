//! Fixtures shared by the OCEL benchmarks.
//!
//! A directory module rather than `benches/common.rs`, because cargo would take the latter for a
//! bench target of its own.

// Each bench pulls in the whole module but uses only the part it needs.
#![allow(dead_code)]

use std::path::PathBuf;

use process_mining::test_utils::get_test_data_path;

/// The order-management log in the given serialization (`json`, `xml`, ...).
pub fn order_management(ext: &str) -> PathBuf {
    get_test_data_path()
        .join("ocel")
        .join(format!("order-management.{ext}"))
}

/// Stream the JSON fixture into a fresh `DuckDB` database named `db_name`.
///
/// The returned directory owns the database file and has to be kept alive alongside the
/// connection.
#[cfg(feature = "ocel-duckdb")]
pub fn duckdb_from_json(
    db_name: &str,
) -> (
    tempfile::TempDir,
    process_mining::core::event_data::object_centric::ocel_sql::DuckDbLinkedOCEL,
) {
    use process_mining::core::event_data::object_centric::ocel_sql::{
        stream_ocel_file_to_duckdb, DuckDbLinkedOCEL,
    };
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let db_path = tmp_dir.path().join(db_name);
    stream_ocel_file_to_duckdb(&order_management("json"), &db_path).expect("stream OCEL to DuckDB");
    let db = DuckDbLinkedOCEL::open(&db_path).expect("open DuckDB");
    (tmp_dir, db)
}
