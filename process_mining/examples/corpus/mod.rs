//! The nine-log corpus of the ReFlow paper and where its results go.
//!
//! `REFLOW_LOGS` names the directory holding the logs (default `artifacts/logs`), and
//! `REFLOW_STATS` the directory the examples write their JSON to (default
//! `artifacts/results/stats`). See `artifacts/README.md` for the download list.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const LOGS: &[&str] = &[
    "01_o2c.xml",
    "02_p2p.xml",
    "03_hiring.xml",
    "04_hospital.xml",
    "ocel2-p2p-updated.xml",
    "ContainerLogistics.xml",
    "socel2_hinge.xml",
    "order-management.xml",
    "bpic2017-no-W.xml.gz",
];

fn artifacts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../artifacts")
}

pub fn logs_dir() -> PathBuf {
    std::env::var("REFLOW_LOGS").map(PathBuf::from).unwrap_or_else(|_| artifacts().join("logs"))
}

pub fn stats_dir() -> PathBuf {
    std::env::var("REFLOW_STATS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| artifacts().join("results/stats"))
}

/// The corpus, or the log paths given on the command line.
pub fn logs_or_args() -> Vec<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        let dir = logs_dir();
        LOGS.iter().map(|l| dir.join(l).to_string_lossy().into_owned()).collect()
    } else {
        args
    }
}

/// The path as it is recorded in result files: `logs/<file name>`, so a result does not
/// carry the directory layout of the machine that produced it.
pub fn label(path: &str) -> String {
    match std::path::Path::new(path).file_name() {
        Some(f) => format!("logs/{}", f.to_string_lossy()),
        None => path.to_string(),
    }
}
