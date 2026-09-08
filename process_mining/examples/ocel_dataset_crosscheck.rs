//! Example: Cross-check the CSV and bundled OCEL importers against baseline logs.
//!
//! Reads the same log three ways -- from a `.ocel.zip` bundle, from a flat `.ocel.csv`, and from
//! a baseline `.json`/`.xml` -- and reports where the three disagree. Exits non-zero when any
//! comparison fails.
//!
//! # Layout it expects
//!
//! ```text
//! <datasets>/ocel2-bundle/<name>.ocel.zip
//! <datasets>/ocel2-csv/<name>.ocel.csv
//! <baselines>/<baseline file named in DATASETS>
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features ocel-bundle-parquet --example ocel_dataset_crosscheck -- \
//!   --datasets /path/to/drive-download-... --baselines /path/to/ocel
//!
//! # one dataset, no size ceiling, more example diffs
//! cargo run --release --features ocel-bundle-parquet --example ocel_dataset_crosscheck -- \
//!   --only p2p --max-mb 0 --max-diffs 25
//! ```
//!
//! # What is compared
//!
//! Each log is first checked on its own, for the defects a pairwise diff structurally cannot
//! see: duplicate ids, relations pointing at an object that does not exist, events or objects of
//! an undeclared type, and attributes their type never declares. Each pair of logs then runs
//! through events, objects, type declarations, E2O and O2O relations, and every attribute
//! observation.
//!
//! Attribute differences are split into three classes:
//!
//! - `VALUE` -- the values really differ.
//! - `TYPE` -- the values render the same but sit in different [`OCELAttributeValue`] variants.
//!   `Integer(5)` against `Float(5.0)` against `String("5")` lands here.
//! - `TIME` -- an object attribute carries the same value at a different point in time.
//!
//! A [`Policy`] decides which of those classes make a check fail. The bundled format is
//! lossless, so it is held to all three. The flat CSV carries no attribute type information and
//! dates an undated attribute row to the UNIX epoch, so `TYPE` and `TIME` differences are
//! reported for it but tolerated.

#[cfg(not(feature = "ocel-bundle-parquet"))]
fn main() {
    eprintln!(
        "This example needs the `ocel-bundle-parquet` feature:\n  \
         cargo run --release --features ocel-bundle-parquet --example ocel_dataset_crosscheck"
    );
    std::process::exit(2);
}

#[cfg(feature = "ocel-bundle-parquet")]
fn main() -> std::process::ExitCode {
    imp::run()
}

#[cfg(feature = "ocel-bundle-parquet")]
mod imp {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::path::{Path, PathBuf};
    use std::process::ExitCode;
    use std::time::Instant;

    use process_mining::core::event_data::object_centric::ocel_bundle::import_ocel_bundle;
    use process_mining::core::event_data::object_centric::ocel_csv::import_ocel_csv_from_path;
    use process_mining::core::event_data::object_centric::ocel_json::import_ocel_json_path;
    use process_mining::core::event_data::object_centric::ocel_xml::xml_ocel_import::import_ocel_xml_path;
    use process_mining::core::event_data::object_centric::{OCELAttributeValue, OCELType, OCEL};
    use process_mining::core::event_data::timestamp_utils::parse_timestamp;

    // ---------------------------------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------------------------------

    /// Directory holding the `ocel2-bundle/` and `ocel2-csv/` subdirectories.
    const DEFAULT_DATASET_DIR: &str = "ocel-datasets";

    /// Directory holding the baseline `.json`/`.xml` logs named in [`DATASETS`].
    const DEFAULT_BASELINE_DIR: &str = "ocel-baselines";

    /// Datasets whose largest input exceeds this are skipped unless `--max-mb` says otherwise.
    /// `0` means no ceiling.
    const DEFAULT_MAX_INPUT_MB: u64 = 100;

    /// How many example differences to print per failing check.
    const DEFAULT_MAX_DIFF_SAMPLES: usize = 5;

    /// A dataset and the baseline it is compared against.
    struct Dataset {
        /// Stem shared by `<name>.ocel.zip` and `<name>.ocel.csv`.
        name: &'static str,
        /// File inside the baseline directory, or `None` when no baseline is available. Without
        /// one the CSV is still cross-checked against the bundle.
        baseline: Option<&'static str>,
    }

    /// The datasets to check. Baselines were matched to their dataset by event-type and
    /// object-type counts.
    const DATASETS: &[Dataset] = &[
        Dataset {
            name: "logistics",
            baseline: Some("ContainerLogistics.json"),
        },
        Dataset {
            name: "order-management",
            baseline: Some("order-management.json"),
        },
        Dataset {
            name: "p2p",
            baseline: Some("ocel2-p2p.json"),
        },
        Dataset {
            name: "hinge-production",
            baseline: Some("socel2_hinge.xml"),
        },
        Dataset {
            name: "scientific-publications-wil-van-der-aalst",
            baseline: Some("Wil van der Aalst_clean.xml"),
        },
        Dataset {
            name: "aoe2-10-match-filter",
            baseline: Some("age_of_empires_ocel2_10_match_filter.json"),
        },
        Dataset {
            name: "angular-github-commits",
            baseline: Some("angular_github_commits_ocel-updated.xml"),
        },
        Dataset {
            name: "bundestag-period20-sub-steps",
            baseline: None,
        },
        // Beyond the default size ceiling: reachable with `--max-mb 0`.
        Dataset {
            name: "bundestag-period20-procedure-steps",
            baseline: Some(
                "pm4Bundestag_20250406_periode_20_including_promulgation_proclamation.json",
            ),
        },
        Dataset {
            name: "bundestag-period20-procedure-steps-filtered",
            baseline: None,
        },
        Dataset {
            name: "ethereum-blockchain",
            baseline: None,
        },
        Dataset {
            name: "aoe2-full",
            baseline: Some("age_of_empires_ocel2.json"),
        },
    ];

    /// Which classes of attribute difference make a check fail.
    #[derive(Clone, Copy)]
    struct Policy {
        /// Compare the declared `type` of each attribute in the event/object type definitions.
        declared_types: bool,
        /// Fail on values that render alike but sit in different variants.
        value_types: bool,
        /// Fail on object attribute values recorded at a different time.
        value_times: bool,
    }

    /// The bundled format is lossless, so nothing is tolerated.
    const STRICT: Policy = Policy {
        declared_types: true,
        value_types: true,
        value_times: true,
    };

    /// The flat CSV declares no attribute types and infers them from the text, and dates an
    /// attribute row with an empty `timestamp` to the UNIX epoch. Both are reported, neither
    /// fails the run.
    const CSV_TOLERANT: Policy = Policy {
        declared_types: false,
        value_types: false,
        value_times: false,
    };

    // ---------------------------------------------------------------------------------------
    // Command line
    // ---------------------------------------------------------------------------------------

    struct Args {
        datasets: PathBuf,
        baselines: PathBuf,
        only: Option<Vec<String>>,
        max_bytes: Option<u64>,
        max_samples: usize,
    }

    const HELP: &str = "\
Cross-check the CSV and bundled OCEL importers against baseline logs.

  --datasets <dir>   directory with ocel2-bundle/ and ocel2-csv/
  --baselines <dir>  directory with the baseline .json/.xml logs
  --only <a,b,...>   restrict the run to these dataset names
  --max-mb <n>       skip datasets whose largest input exceeds n MB (0 = no limit)
  --max-diffs <n>    example differences to print per failing check
  --list             print the dataset table and exit
  -h, --help         print this
";

    fn parse_args() -> Result<Args, String> {
        let mut args = Args {
            datasets: PathBuf::from(DEFAULT_DATASET_DIR),
            baselines: PathBuf::from(DEFAULT_BASELINE_DIR),
            only: None,
            max_bytes: (DEFAULT_MAX_INPUT_MB > 0).then_some(DEFAULT_MAX_INPUT_MB * 1024 * 1024),
            max_samples: DEFAULT_MAX_DIFF_SAMPLES,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut value = || {
                it.next()
                    .ok_or_else(|| format!("`{arg}` needs a value"))
                    .map_err(|e| e.to_string())
            };
            match arg.as_str() {
                "-h" | "--help" => {
                    print!("{HELP}");
                    std::process::exit(0);
                }
                "--list" => {
                    for d in DATASETS {
                        println!("{:<48} {}", d.name, d.baseline.unwrap_or("(no baseline)"));
                    }
                    std::process::exit(0);
                }
                "--datasets" => args.datasets = PathBuf::from(value()?),
                "--baselines" => args.baselines = PathBuf::from(value()?),
                "--only" => {
                    args.only = Some(
                        value()?
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                }
                "--max-mb" => {
                    let mb: u64 = value()?.parse().map_err(|_| "--max-mb needs a number")?;
                    args.max_bytes = (mb > 0).then(|| mb * 1024 * 1024);
                }
                "--max-diffs" => {
                    args.max_samples = value()?.parse().map_err(|_| "--max-diffs needs a number")?;
                }
                other => return Err(format!("unknown argument `{other}`\n\n{HELP}")),
            }
        }
        Ok(args)
    }

    // ---------------------------------------------------------------------------------------
    // Views of a log
    // ---------------------------------------------------------------------------------------

    /// `tagged` prefixes the variant, so `Integer(5)`, `Float(5.0)` and `String("5")` render as
    /// `i:5`, `f:5` and `s:5`. `plain` drops both the tag and a trailing `.0`, so those three
    /// collapse onto one string and only a real difference in the number survives.
    fn render(value: &OCELAttributeValue, tagged: bool) -> String {
        let (tag, body) = match value {
            OCELAttributeValue::Integer(i) => ('i', i.to_string()),
            OCELAttributeValue::Float(f) => ('f', number(*f)),
            OCELAttributeValue::Boolean(b) => ('b', b.to_string()),
            // The same instant is written with different offsets by different formats, which is
            // not a difference in the log.
            OCELAttributeValue::Time(t) => ('t', t.to_utc().to_rfc3339()),
            OCELAttributeValue::String(s) => (
                's',
                // Untagged, a string that *is* a number renders as one, so text `"5.0"` and a
                // float `5.0` agree on the value and differ only in the variant.
                if tagged {
                    s.clone()
                } else {
                    numeric_text(s)
                        .or_else(|| boolean_text(s))
                        .or_else(|| timestamp_text(s))
                        .unwrap_or_else(|| s.clone())
                },
            ),
            OCELAttributeValue::Null => ('n', String::new()),
        };
        if tagged {
            format!("{tag}:{body}")
        } else {
            body
        }
    }

    /// A float without a trailing `.0`, so `5.0` and `5` render alike.
    fn number(f: f64) -> String {
        if f.is_finite() && f.fract() == 0.0 {
            format!("{f:.0}")
        } else {
            f.to_string()
        }
    }

    fn numeric_text(s: &str) -> Option<String> {
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        t.parse::<i64>()
            .ok()
            .map(|i| i.to_string())
            .or_else(|| t.parse::<f64>().ok().filter(|f| f.is_finite()).map(number))
    }

    /// Case-folded, so a log holding Python's `False` as text and a log holding a real boolean
    /// agree on the value.
    fn boolean_text(s: &str) -> Option<String> {
        let t = s.trim();
        (t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("false"))
            .then(|| t.to_ascii_lowercase())
    }

    /// The UTC instant a string denotes, if it denotes one, so text kept verbatim by one format
    /// and parsed into an [`OCELAttributeValue::Time`] by another agree on the value.
    ///
    /// The shape test in front of the parse is not optional: the parser is tried against every
    /// string attribute in the log, and a log of this size has millions.
    fn timestamp_text(s: &str) -> Option<String> {
        let t = s.trim();
        if !(8..=40).contains(&t.len())
            || !t.starts_with(|c: char| c.is_ascii_digit())
            || !t.contains('-')
        {
            return None;
        }
        parse_timestamp(t, None, false)
            .ok()
            .map(|d| d.to_utc().to_rfc3339())
    }

    /// `id -> type@time`, so a renamed type, a shifted timestamp and a missing event are all one
    /// comparison.
    fn events(ocel: &OCEL) -> BTreeMap<String, String> {
        ocel.events
            .iter()
            .map(|e| {
                (
                    e.id.clone(),
                    format!("{}@{}", e.event_type, e.time.to_utc().to_rfc3339()),
                )
            })
            .collect()
    }

    /// `id -> type`.
    fn objects(ocel: &OCEL) -> BTreeMap<String, String> {
        ocel.objects
            .iter()
            .map(|o| (o.id.clone(), o.object_type.clone()))
            .collect()
    }

    /// `type name -> sorted "attribute:type" declarations`.
    fn declared(types: &[OCELType]) -> BTreeMap<String, Vec<String>> {
        types
            .iter()
            .map(|t| {
                let mut attrs: Vec<String> = t
                    .attributes
                    .iter()
                    .map(|a| format!("{}:{}", a.name, a.value_type))
                    .collect();
                attrs.sort();
                attrs.dedup();
                (t.name.clone(), attrs)
            })
            .collect()
    }

    /// `type name -> instance count`.
    fn per_type<'a>(items: impl Iterator<Item = &'a str>) -> BTreeMap<String, String> {
        let mut out: BTreeMap<String, u64> = BTreeMap::new();
        for t in items {
            *out.entry(t.to_string()).or_default() += 1;
        }
        out.into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect()
    }

    fn e2o(ocel: &OCEL) -> Multiset {
        let mut m = Multiset::default();
        for e in &ocel.events {
            for r in &e.relationships {
                m.add(format!("{} -> {} [{}]", e.id, r.object_id, r.qualifier));
            }
        }
        m
    }

    fn o2o(ocel: &OCEL) -> Multiset {
        let mut m = Multiset::default();
        for o in &ocel.objects {
            for r in &o.relationships {
                m.add(format!("{} -> {} [{}]", o.id, r.object_id, r.qualifier));
            }
        }
        m
    }

    /// One attribute observation seen three ways, so a difference can be attributed to the value,
    /// to the variant it is stored in, or to the time it was recorded at.
    ///
    /// `values` is a multiset because the same observation can legitimately repeat. A variant or
    /// timestamp difference is only meaningful where both logs agree on what came before it, so
    /// the other two are keyed maps and render as one paired line rather than two surplus ones.
    #[derive(Default)]
    struct AttrViews {
        /// `id/name = value`, ignoring variant and time.
        values: Multiset,
        /// `id/name = value` -> the variants it is stored in.
        variants: BTreeMap<String, Joined>,
        /// `id/name = tagged value` -> the times it was recorded at.
        times: BTreeMap<String, Joined>,
    }

    impl AttrViews {
        fn add(&mut self, key: &str, value: &OCELAttributeValue, time: Option<String>) {
            let (plain, tagged) = (render(value, false), render(value, true));
            self.values.add(format!("{key} = {plain}"));
            self.variants
                .entry(format!("{key} = {plain}"))
                .or_default()
                .0
                .push(tagged[..1].to_string());
            if let Some(time) = time {
                self.times
                    .entry(format!("{key} = {tagged}"))
                    .or_default()
                    .0
                    .push(time);
            }
        }

        /// Deduplicated, so that a log recording the same observation twice shows up only in the
        /// `values` multiset and does not also colour the variant and time checks.
        fn sort(mut self) -> Self {
            for v in self.variants.values_mut().chain(self.times.values_mut()) {
                v.0.sort();
                v.0.dedup();
            }
            self
        }
    }

    fn event_attrs(ocel: &OCEL) -> AttrViews {
        let mut v = AttrViews::default();
        for e in &ocel.events {
            for a in &e.attributes {
                v.add(&format!("{}/{}", e.id, a.name), &a.value, None);
            }
        }
        v.sort()
    }

    fn object_attrs(ocel: &OCEL) -> AttrViews {
        let mut v = AttrViews::default();
        for o in &ocel.objects {
            for a in &o.attributes {
                v.add(
                    &format!("{}/{}", o.id, a.name),
                    &a.value,
                    Some(a.time.to_utc().to_rfc3339()),
                );
            }
        }
        v.sort()
    }

    // ---------------------------------------------------------------------------------------
    // Diffing
    // ---------------------------------------------------------------------------------------

    /// One comparison, carrying enough to read a count without guessing: how large each side is,
    /// and which way the difference runs.
    #[derive(Default)]
    struct Diff {
        /// Entries compared on each side.
        a_total: u64,
        b_total: u64,
        /// Entries A has that B does not, and the reverse.
        only_a: u64,
        only_b: u64,
        /// Keys both sides have under a different value. Always 0 for a multiset comparison,
        /// which has no notion of a key holding one value.
        changed: u64,
        samples: Vec<String>,
    }

    impl Diff {
        fn count(&self) -> u64 {
            self.only_a + self.only_b + self.changed
        }

        /// `12 of 45 in A / 44 in B: 3 only in A, 2 only in B, 7 changed`, with the empty parts
        /// left out.
        fn describe(&self) -> String {
            let mut parts = Vec::new();
            if self.only_a > 0 {
                parts.push(format!("{} only in A", self.only_a));
            }
            if self.only_b > 0 {
                parts.push(format!("{} only in B", self.only_b));
            }
            if self.changed > 0 {
                parts.push(format!("{} changed", self.changed));
            }
            format!(
                "{} of {} in A / {} in B: {}",
                self.count(),
                self.a_total,
                self.b_total,
                parts.join(", ")
            )
        }
    }

    #[derive(Default)]
    struct Multiset(BTreeMap<String, i64>);

    impl Multiset {
        fn add(&mut self, key: String) {
            *self.0.entry(key).or_default() += 1;
        }

        fn len(&self) -> u64 {
            self.0.values().map(|&n| n.unsigned_abs()).sum()
        }

        /// Surplus entries on either side, with up to `max` of them rendered.
        fn diff(&self, other: &Self, max: usize) -> Diff {
            let mut out = Diff {
                a_total: self.len(),
                b_total: other.len(),
                ..Diff::default()
            };
            let (mut left_only, mut right_only) = (Vec::new(), Vec::new());
            for (key, &left) in &self.0 {
                let delta = left - other.0.get(key).copied().unwrap_or(0);
                if delta != 0 {
                    let (bucket, side, tally) = if delta > 0 {
                        (&mut left_only, "A", &mut out.only_a)
                    } else {
                        (&mut right_only, "B", &mut out.only_b)
                    };
                    *tally += delta.unsigned_abs();
                    if bucket.len() < max {
                        bucket.push(format!("only in {side} (x{}): {key}", delta.abs()));
                    }
                }
            }
            for (key, &right) in &other.0 {
                if !self.0.contains_key(key) {
                    out.only_b += right.unsigned_abs();
                    if right_only.len() < max {
                        right_only.push(format!("only in B (x{right}): {key}"));
                    }
                }
            }
            out.samples = interleave(left_only, right_only, max);
            out
        }
    }

    /// Difference between two keyed maps, split into keys missing from B, keys only in B, and
    /// keys present in both with a different value.
    fn map_diff<V: PartialEq + std::fmt::Display>(
        a: &BTreeMap<String, V>,
        b: &BTreeMap<String, V>,
        max: usize,
    ) -> Diff {
        let mut out = Diff {
            a_total: a.len() as u64,
            b_total: b.len() as u64,
            ..Diff::default()
        };
        let (mut left_only, mut right_only, mut differing) = (Vec::new(), Vec::new(), Vec::new());
        for (key, left) in a {
            match b.get(key) {
                None => {
                    out.only_a += 1;
                    if left_only.len() < max {
                        left_only.push(format!("only in A: {key} = {left}"));
                    }
                }
                Some(right) if right != left => {
                    out.changed += 1;
                    if differing.len() < max {
                        differing.push(format!("differs: {key}: A = {left} | B = {right}"));
                    }
                }
                Some(_) => {}
            }
        }
        for (key, right) in b {
            if !a.contains_key(key) {
                out.only_b += 1;
                if right_only.len() < max {
                    right_only.push(format!("only in B: {key} = {right}"));
                }
            }
        }
        differing.truncate(max);
        let rest = max.saturating_sub(differing.len());
        differing.extend(interleave(left_only, right_only, rest));
        out.samples = differing;
        out
    }

    /// Takes from both sides in turn, so a capped sample never shows only one direction of a
    /// difference and hides the counterpart that explains it.
    fn interleave(a: Vec<String>, b: Vec<String>, max: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(max.min(a.len() + b.len()));
        let (mut a, mut b) = (a.into_iter(), b.into_iter());
        loop {
            let before = out.len();
            for next in [a.next(), b.next()].into_iter().flatten() {
                if out.len() < max {
                    out.push(next);
                }
            }
            if out.len() == before || out.len() >= max {
                return out;
            }
        }
    }

    /// Keys the two logs share but disagree on. Keys only one side has are left out: they are
    /// already counted by the coarser check that this one refines.
    fn shared_diff<V: PartialEq + std::fmt::Display>(
        a: &BTreeMap<String, V>,
        b: &BTreeMap<String, V>,
        max: usize,
    ) -> Diff {
        let mut out = Diff {
            a_total: a.len() as u64,
            b_total: b.len() as u64,
            ..Diff::default()
        };
        for (key, left) in a {
            if let Some(right) = b.get(key) {
                if right != left {
                    out.changed += 1;
                    if out.samples.len() < max {
                        out.samples.push(format!("{key}: A = {left} | B = {right}"));
                    }
                }
            }
        }
        out
    }

    /// `Vec<String>` needs a `Display` to go through [`map_diff`]; this wraps it.
    #[derive(Default)]
    struct Joined(Vec<String>);

    impl PartialEq for Joined {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl std::fmt::Display for Joined {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0.join(", "))
        }
    }

    // ---------------------------------------------------------------------------------------
    // Reporting
    // ---------------------------------------------------------------------------------------

    enum Status {
        Ok,
        /// Differences that fail the comparison.
        Failed,
        /// Differences the policy expects for this pair of formats.
        Tolerated,
    }

    struct Check {
        name: &'static str,
        status: Status,
        detail: String,
        samples: Vec<String>,
    }

    impl Check {
        fn ok(name: &'static str) -> Self {
            Check {
                name,
                status: Status::Ok,
                detail: "match".into(),
                samples: Vec::new(),
            }
        }

        /// A check that counts offenders in one log rather than differences between two.
        fn from_count(
            name: &'static str,
            count: u64,
            total: u64,
            samples: Vec<String>,
            fails: bool,
        ) -> Self {
            if count == 0 {
                return Check::ok(name);
            }
            Check {
                name,
                status: if fails {
                    Status::Failed
                } else {
                    Status::Tolerated
                },
                detail: format!("{count} of {total}"),
                samples,
            }
        }

        fn from_diff(name: &'static str, diff: Diff, fails: bool) -> Self {
            if diff.count() == 0 {
                return Check::ok(name);
            }
            Check {
                name,
                status: if fails {
                    Status::Failed
                } else {
                    Status::Tolerated
                },
                detail: diff.describe(),
                samples: diff.samples,
            }
        }
    }

    /// Checks that need only one log.
    ///
    /// A pairwise diff cannot see any of these: two logs can be equally broken and agree
    /// perfectly, and a duplicated id collapses into a single map entry before the comparison
    /// ever runs.
    fn integrity(ocel: &OCEL, max: usize) -> Vec<Check> {
        let mut checks = Vec::new();

        let (n, s) = duplicates(ocel.events.iter().map(|e| e.id.as_str()), max);
        checks.push(Check::from_count(
            "duplicate event ids",
            n,
            ocel.events.len() as u64,
            s,
            true,
        ));

        let (n, s) = duplicates(ocel.objects.iter().map(|o| o.id.as_str()), max);
        checks.push(Check::from_count(
            "duplicate object ids",
            n,
            ocel.objects.len() as u64,
            s,
            true,
        ));

        let object_ids: HashSet<&str> = ocel.objects.iter().map(|o| o.id.as_str()).collect();

        let mut dangling = 0u64;
        let mut total = 0u64;
        let mut samples = Vec::new();
        for e in &ocel.events {
            for r in &e.relationships {
                total += 1;
                if !object_ids.contains(r.object_id.as_str()) {
                    dangling += 1;
                    if samples.len() < max {
                        samples.push(format!("{} -> {} (no such object)", e.id, r.object_id));
                    }
                }
            }
        }
        checks.push(Check::from_count(
            "E2O relations pointing at no object",
            dangling,
            total,
            samples,
            true,
        ));

        let mut dangling = 0u64;
        let mut total = 0u64;
        let mut samples = Vec::new();
        for o in &ocel.objects {
            for r in &o.relationships {
                total += 1;
                if !object_ids.contains(r.object_id.as_str()) {
                    dangling += 1;
                    if samples.len() < max {
                        samples.push(format!("{} -> {} (no such object)", o.id, r.object_id));
                    }
                }
            }
        }
        checks.push(Check::from_count(
            "O2O relations pointing at no object",
            dangling,
            total,
            samples,
            true,
        ));

        let declared_event: HashSet<&str> =
            ocel.event_types.iter().map(|t| t.name.as_str()).collect();
        let (n, s) = undeclared(
            ocel.events.iter().map(|e| (e.id.as_str(), e.event_type.as_str())),
            &declared_event,
            max,
        );
        checks.push(Check::from_count(
            "events of an undeclared type",
            n,
            ocel.events.len() as u64,
            s,
            true,
        ));

        let declared_object: HashSet<&str> =
            ocel.object_types.iter().map(|t| t.name.as_str()).collect();
        let (n, s) = undeclared(
            ocel.objects.iter().map(|o| (o.id.as_str(), o.object_type.as_str())),
            &declared_object,
            max,
        );
        checks.push(Check::from_count(
            "objects of an undeclared type",
            n,
            ocel.objects.len() as u64,
            s,
            true,
        ));

        let declared = attribute_names(&ocel.event_types);
        let mut missing = 0u64;
        let mut total = 0u64;
        let mut samples = Vec::new();
        for e in &ocel.events {
            for a in &e.attributes {
                total += 1;
                if !declared.contains(&(e.event_type.as_str(), a.name.as_str())) {
                    missing += 1;
                    if samples.len() < max {
                        samples.push(format!(
                            "{}/{} not declared for event type {}",
                            e.id, a.name, e.event_type
                        ));
                    }
                }
            }
        }
        checks.push(Check::from_count(
            "event attributes not declared by their type",
            missing,
            total,
            samples,
            true,
        ));

        let declared = attribute_names(&ocel.object_types);
        let mut missing = 0u64;
        let mut total = 0u64;
        let mut samples = Vec::new();
        for o in &ocel.objects {
            for a in &o.attributes {
                total += 1;
                if !declared.contains(&(o.object_type.as_str(), a.name.as_str())) {
                    missing += 1;
                    if samples.len() < max {
                        samples.push(format!(
                            "{}/{} not declared for object type {}",
                            o.id, a.name, o.object_type
                        ));
                    }
                }
            }
        }
        checks.push(Check::from_count(
            "object attributes not declared by their type",
            missing,
            total,
            samples,
            true,
        ));

        checks
    }

    /// Ids that occur more than once, and how many surplus occurrences there are in total.
    fn duplicates<'a>(ids: impl Iterator<Item = &'a str>, max: usize) -> (u64, Vec<String>) {
        let mut seen: HashMap<&str, u64> = HashMap::new();
        for id in ids {
            *seen.entry(id).or_default() += 1;
        }
        let mut extra = 0u64;
        let mut samples = Vec::new();
        let mut repeated: Vec<_> = seen.into_iter().filter(|(_, n)| *n > 1).collect();
        repeated.sort();
        for (id, n) in repeated {
            extra += n - 1;
            if samples.len() < max {
                samples.push(format!("{id} occurs {n} times"));
            }
        }
        (extra, samples)
    }

    fn undeclared<'a>(
        items: impl Iterator<Item = (&'a str, &'a str)>,
        declared: &HashSet<&str>,
        max: usize,
    ) -> (u64, Vec<String>) {
        let mut count = 0u64;
        let mut samples = Vec::new();
        for (id, type_name) in items {
            if !declared.contains(type_name) {
                count += 1;
                if samples.len() < max {
                    samples.push(format!("{id} has undeclared type {type_name}"));
                }
            }
        }
        (count, samples)
    }

    fn attribute_names(types: &[OCELType]) -> HashSet<(&str, &str)> {
        types
            .iter()
            .flat_map(|t| {
                t.attributes
                    .iter()
                    .map(move |a| (t.name.as_str(), a.name.as_str()))
            })
            .collect()
    }

    /// Every check for one pair of logs, in the order they are printed.
    fn compare(a: &OCEL, b: &OCEL, policy: Policy, max: usize) -> Vec<Check> {
        let mut checks = Vec::new();

        let d = map_diff(&events(a), &events(b), max);
        checks.push(Check::from_diff("events (id, type, time)", d, true));

        let d = map_diff(&objects(a), &objects(b), max);
        checks.push(Check::from_diff("objects (id, type)", d, true));

        let d = map_diff(
            &per_type(a.events.iter().map(|e| e.event_type.as_str())),
            &per_type(b.events.iter().map(|e| e.event_type.as_str())),
            max,
        );
        checks.push(Check::from_diff("events per event type", d, true));

        let d = map_diff(
            &per_type(a.objects.iter().map(|o| o.object_type.as_str())),
            &per_type(b.objects.iter().map(|o| o.object_type.as_str())),
            max,
        );
        checks.push(Check::from_diff("objects per object type", d, true));

        // Names always, declarations only where the format carries them.
        let (decl_a, decl_b) = (declared(&a.event_types), declared(&b.event_types));
        let d = map_diff(
            &decl_a
                .keys()
                .map(|k| (k.clone(), String::new()))
                .collect::<BTreeMap<_, _>>(),
            &decl_b
                .keys()
                .map(|k| (k.clone(), String::new()))
                .collect::<BTreeMap<_, _>>(),
            max,
        );
        checks.push(Check::from_diff("event type names", d, true));

        let d = map_diff(
            &decl_a.into_iter().map(|(k, v)| (k, Joined(v))).collect(),
            &decl_b.into_iter().map(|(k, v)| (k, Joined(v))).collect(),
            max,
        );
        checks.push(Check::from_diff(
            "event type attribute declarations",
            d,
            policy.declared_types,
        ));

        let (decl_a, decl_b) = (declared(&a.object_types), declared(&b.object_types));
        let d = map_diff(
            &decl_a
                .keys()
                .map(|k| (k.clone(), String::new()))
                .collect::<BTreeMap<_, _>>(),
            &decl_b
                .keys()
                .map(|k| (k.clone(), String::new()))
                .collect::<BTreeMap<_, _>>(),
            max,
        );
        checks.push(Check::from_diff("object type names", d, true));

        let d = map_diff(
            &decl_a.into_iter().map(|(k, v)| (k, Joined(v))).collect(),
            &decl_b.into_iter().map(|(k, v)| (k, Joined(v))).collect(),
            max,
        );
        checks.push(Check::from_diff(
            "object type attribute declarations",
            d,
            policy.declared_types,
        ));

        let d = e2o(a).diff(&e2o(b), max);
        checks.push(Check::from_diff("E2O relations", d, true));

        let d = o2o(a).diff(&o2o(b), max);
        checks.push(Check::from_diff("O2O relations", d, true));

        checks.extend(attribute_checks(
            "event attribute",
            &event_attrs(a),
            &event_attrs(b),
            policy,
            max,
            false,
        ));
        checks.extend(attribute_checks(
            "object attribute",
            &object_attrs(a),
            &object_attrs(b),
            policy,
            max,
            true,
        ));

        checks
    }

    /// Splits one attribute difference into the value, variant and time classes.
    ///
    /// The three views nest -- a value difference also shows up in the typed and full views -- so
    /// each class is the growth from the coarser view below it.
    fn attribute_checks(
        what: &'static str,
        a: &AttrViews,
        b: &AttrViews,
        policy: Policy,
        max: usize,
        timed: bool,
    ) -> Vec<Check> {
        let d = a.values.diff(&b.values, max);
        let mut out = vec![Check::from_diff(leak(format!("{what} values")), d, true)];

        let d = shared_diff(&a.variants, &b.variants, max);
        out.push(Check::from_diff(
            leak(format!("{what} value types (i/f/b/t/s/n)")),
            d,
            policy.value_types,
        ));

        if timed {
            let d = shared_diff(&a.times, &b.times, max);
            out.push(Check::from_diff(
                leak(format!("{what} value timestamps")),
            d,
                policy.value_times,
            ));
        }

        out
    }

    /// Check names are `&'static str` because most of them are literals; the handful built per
    /// attribute kind are leaked once each.
    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    // ---------------------------------------------------------------------------------------
    // Running
    // ---------------------------------------------------------------------------------------

    fn import(path: &Path) -> Result<OCEL, String> {
        let name = path.to_string_lossy();
        if name.ends_with(".ocel.zip") || name.ends_with(".ocel") {
            import_ocel_bundle(path).map_err(|e| e.to_string())
        } else if name.ends_with(".csv") {
            import_ocel_csv_from_path(path).map_err(|e| e.to_string())
        } else if name.ends_with(".xml") || name.ends_with(".xmlocel") {
            import_ocel_xml_path(path).map_err(|e| e.to_string())
        } else if name.ends_with(".json") || name.ends_with(".jsonocel") {
            import_ocel_json_path(path).map_err(|e| e.to_string())
        } else {
            Err(format!("no importer for `{name}`"))
        }
    }

    /// Imports `path`, printing what it read and how long it took.
    fn load(label: &str, path: &Path) -> Option<OCEL> {
        let started = Instant::now();
        match import(path) {
            Ok(ocel) => {
                println!(
                    "  {label:<10} {:>9} events {:>9} objects  {:>7.1}s  {}",
                    ocel.events.len(),
                    ocel.objects.len(),
                    started.elapsed().as_secs_f64(),
                    path.display()
                );
                Some(ocel)
            }
            Err(e) => {
                println!("  {label:<10} IMPORT FAILED: {e}");
                println!("             {}", path.display());
                None
            }
        }
    }

    fn report(title: &str, a: &OCEL, b: &OCEL, policy: Policy, max: usize) -> bool {
        print_checks(title, compare(a, b, policy, max))
    }

    fn report_integrity(label: &str, ocel: &OCEL, max: usize) -> bool {
        print_checks(&format!("integrity of {label}"), integrity(ocel, max))
    }

    fn print_checks(title: &str, checks: Vec<Check>) -> bool {
        println!("  {title}");
        let mut failed = false;
        for check in &checks {
            let (mark, note) = match check.status {
                Status::Ok => ("ok  ", ""),
                Status::Failed => ("FAIL", ""),
                Status::Tolerated => ("note", " (expected for this format)"),
            };
            if matches!(check.status, Status::Failed) {
                failed = true;
            }
            if matches!(check.status, Status::Ok) {
                continue;
            }
            println!(
                "    [{mark}] {:<42} {}{note}",
                check.name, check.detail
            );
            for sample in &check.samples {
                println!("           {sample}");
            }
        }
        let ok = checks
            .iter()
            .filter(|c| matches!(c.status, Status::Ok))
            .count();
        println!("    {ok}/{} checks clean", checks.len());
        failed
    }

    fn size_of(path: &Path) -> u64 {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }

    pub fn run() -> ExitCode {
        let args = match parse_args() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        };

        let mut failed: Vec<String> = Vec::new();
        let mut skipped: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for dataset in DATASETS {
            if let Some(only) = &args.only {
                if !only.iter().any(|n| n == dataset.name) {
                    continue;
                }
            }

            let bundle_path = args
                .datasets
                .join("ocel2-bundle")
                .join(format!("{}.ocel.zip", dataset.name));
            let csv_path = args
                .datasets
                .join("ocel2-csv")
                .join(format!("{}.ocel.csv", dataset.name));
            let baseline_path = dataset.baseline.map(|b| args.baselines.join(b));

            println!("\n=== {} ===", dataset.name);

            let mut missing = Vec::new();
            for p in [Some(&bundle_path), Some(&csv_path), baseline_path.as_ref()]
                .into_iter()
                .flatten()
            {
                if !p.exists() {
                    missing.push(p.display().to_string());
                }
            }
            if !missing.is_empty() {
                for m in &missing {
                    println!("  missing: {m}");
                }
                // A missing baseline still leaves the CSV/bundle cross-check worth running.
                if !bundle_path.exists() || !csv_path.exists() {
                    skipped.push(format!("{} (missing inputs)", dataset.name));
                    continue;
                }
            }
            let baseline_path = baseline_path.filter(|p| p.exists());

            let largest = [Some(&bundle_path), Some(&csv_path), baseline_path.as_ref()]
                .into_iter()
                .flatten()
                .map(|p| size_of(p))
                .max()
                .unwrap_or(0);
            if let Some(limit) = args.max_bytes {
                if largest > limit {
                    println!(
                        "  skipped: largest input is {} MB, above the {} MB ceiling (--max-mb 0 to lift)",
                        largest / (1024 * 1024),
                        limit / (1024 * 1024)
                    );
                    skipped.push(format!("{} (too large)", dataset.name));
                    continue;
                }
            }

            let Some(bundle) = load("bundle", &bundle_path) else {
                failed.push(format!("{} (bundle import)", dataset.name));
                continue;
            };
            let Some(csv) = load("csv", &csv_path) else {
                failed.push(format!("{} (csv import)", dataset.name));
                continue;
            };
            let baseline = match &baseline_path {
                Some(p) => match load("baseline", p) {
                    Some(o) => Some(o),
                    None => {
                        failed.push(format!("{} (baseline import)", dataset.name));
                        continue;
                    }
                },
                None => None,
            };

            checked += 1;
            let mut any = false;
            any |= report_integrity("bundle", &bundle, args.max_samples);
            any |= report_integrity("csv", &csv, args.max_samples);
            if let Some(baseline) = &baseline {
                any |= report_integrity("baseline", baseline, args.max_samples);
            }
            if let Some(baseline) = &baseline {
                any |= report(
                    "A = bundle, B = baseline",
                    &bundle,
                    baseline,
                    STRICT,
                    args.max_samples,
                );
                any |= report(
                    "A = csv, B = baseline",
                    &csv,
                    baseline,
                    CSV_TOLERANT,
                    args.max_samples,
                );
            }
            any |= report(
                "A = csv, B = bundle",
                &csv,
                &bundle,
                CSV_TOLERANT,
                args.max_samples,
            );
            if any {
                failed.push(dataset.name.to_string());
            }
        }

        println!("\n=== summary ===");
        println!("  {checked} dataset(s) checked");
        if !skipped.is_empty() {
            println!("  skipped:");
            for s in &skipped {
                println!("    {s}");
            }
        }
        if failed.is_empty() {
            println!("  no mismatches");
            ExitCode::SUCCESS
        } else {
            println!("  mismatches in:");
            for f in &failed {
                println!("    {f}");
            }
            ExitCode::FAILURE
        }
    }
}
