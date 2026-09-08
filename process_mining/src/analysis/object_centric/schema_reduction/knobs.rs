//! Every constant the reduction reads, in one place.
//!
//! [`knobs`] is the whole parameter set, published as `artifacts/results/params.json` rather
//! than repeated in each result file.

use serde_json::{json, Value};

use super::{
    cells::DETERMINATION_THETA,
    closure::{MAX_COMPOSE_DEPTH, MAX_WITNESSES_PER_PAIR},
    declare_abstraction::{
        DECLARE_ARC_TYPES, DECLARE_COUNTS_FOR_FILTER, DECLARE_COUNTS_FOR_GENERATION,
        DECLARE_NOISE_THRESHOLD,
    },
    expansion::{DEFAULT_THETA, EXPANSION_QUALIFIER, EXPANSION_WORK_BUDGET},
    facts::DEFAULT_NOISE_THRESHOLD,
    model_abstraction::MODEL_NOISE_THRESHOLD,
    o2o_reduction::O2O_REDUCTION_BUDGET,
    schema::MIN_TARGET_OBJECTS,
    search::SEARCH_CLOSURE_BUDGET,
    sigil::{NOT_FLOWING, WRITTEN},
};

/// Every constant the reduction reads, grouped by the stage that reads it.
pub fn knobs() -> Value {
    json!({
        "schema": {
            "min_target_objects": MIN_TARGET_OBJECTS,
            "max_compose_depth": MAX_COMPOSE_DEPTH,
            "max_witnesses_per_pair": MAX_WITNESSES_PER_PAIR,
        },
        "search": {
            "closure_budget_word_ops": SEARCH_CLOSURE_BUDGET,
        },
        "expansion": {
            "default_theta": DEFAULT_THETA,
            "work_budget": EXPANSION_WORK_BUDGET,
            "qualifier": EXPANSION_QUALIFIER,
            "o2o_reduction_budget": O2O_REDUCTION_BUDGET,
        },
        "log_abstraction": {
            "noise_threshold": DEFAULT_NOISE_THRESHOLD,
        },
        "model_abstraction": {
            "imf_noise_threshold": MODEL_NOISE_THRESHOLD,
        },
        "declare_abstraction": {
            "noise_threshold": DECLARE_NOISE_THRESHOLD,
            "counts_for_generation": count_range(DECLARE_COUNTS_FOR_GENERATION),
            "counts_for_filter": count_range(DECLARE_COUNTS_FOR_FILTER),
            "arc_types": DECLARE_ARC_TYPES.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
            "o2o_mode": "None",
            "reduction": "None",
            "refinement": false,
        },
        "tag": {
            "nonflow_prefix": NOT_FLOWING,
            "written_prefix": WRITTEN,
            "determination_theta": DETERMINATION_THETA,
        },
    })
}

fn count_range(r: (Option<usize>, Option<usize>)) -> Value {
    json!({ "min": r.0, "max": r.1 })
}
