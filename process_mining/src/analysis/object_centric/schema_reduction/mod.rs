//! Structural schema discovery for object-centric event logs (ReFlow).
//!
//! The *structural schema* of a log is the set of total maps between object types that
//! the log witnesses, each with its residual. A map is witnessed in one of two ways,
//! and neither subsumes the other:
//!
//! - **recorded**: a finite union of qualifier-restricted object-to-object relations is
//!   functional and total on the source type;
//! - **derived**: co-participation forces it, the running intersection of the target
//!   objects an event names collapsing to a singleton for every source object.
//!
//! A candidate map can fail in three distinct ways:
//!
//! | outcome | meaning |
//! |---|---|
//! | conflict | the running intersection emptied after having been non-empty, so no function exists. Rejected, never absorbed into the residual |
//! | ambiguous | the intersection never collapsed to a singleton. Co-participation is consistent with several images and forces none |
//! | untotal | the source object never co-occurred with a target object. Residual |
//!
//! The schema runs both ways over the same cell grid. **Reduction** tags a participation the
//! schema already determines as non-flow; **expansion** ([`expansion_candidates`]) writes one
//! it determines and the extraction never recorded. Both decide per (activity, object type).
//! Expansion has one condition reduction does not: a tuple whose event precedes its own
//! object's first recorded event is rejected, since determinacy says which object would be
//! named, not that it existed yet.
//!
//! An event that names the source but no object of the target type gives no constraint
//! and is not a residual. Object-level totality is what a map needs; event-level
//! co-presence is a separate condition, tested per cell when a reconstruction is checked.
//!
//! The output is a *tagged OCEL* ([`tagged`]): the same events, objects, attributes,
//! object-to-object relation and event-to-object tuples as the input, each tuple tagged
//! `flow` or `nonflow`. The tag is a prefix on the tuple's event-to-object qualifier --
//! [`sigil::NOT_FLOWING`] (`!`) means `nonflow`, no prefix means `flow`, and
//! [`sigil::WRITTEN`] (`+`) marks a participation an expansion wrote -- so a tagged log is
//! an ordinary OCEL 2.0 log. [`flow_projection`] drops the `nonflow` tuples and is what
//! discovery reads; [`full_projection`] drops the tags and is what every other analysis
//! reads, and it returns the input log. [`annotation`] recomputes the implied maps and the
//! involved marks from the tagged log alone.
//!
//! The two cases are the two directions above: **reduction**, whose candidate cells are the
//! recorded ones and which adds no tuple, and **expansion**, whose candidate cells also hold
//! the unrecorded cells a relation admits and whose participations go into `tag`'s `added`
//! argument.
//!
//! ```text
//! let flow  = reflow_layer(&SearchInput { .. });          // the flow layer K
//! let l_tag = tag(&locel, &schema, &acts, &flow.cells, &added);   // the tagged OCEL
//! let disco = flow_projection(&l_tag);                    // what discovery reads
//! let rest  = full_projection(&l_tag);                    // == locel, what everything else reads
//! ```
//!
//! On the paper's Order Management log, 14 of the 40 recorded cells flow, 3 are involved and
//! 23 implied, and the OCPN goes from 163 arcs to 40. Reproduce the cell counts with
//! `cargo run --release --features "ocel-sqlite,bindings" --example schema_census -- <log> --json`
//! and the arc counts with `process_mining/tools/ocpn-quality`; see `artifacts/README.md`.
//!
//! This is the implementation accompanying the ICPM 2027 paper on structure-based
//! reduction.

mod abstraction;
mod annotate;
mod arcs;
mod assignment;
mod attributes;
mod bounds;
mod canonical;
mod cells;
mod density;
mod expansion;
mod facts;
mod guarantees;
mod lifetime;
mod closure;
mod contribution;
mod coverage;
mod declare_abstraction;
mod model_abstraction;
mod reconstruct;
mod novelty;
mod knobs;
mod residual;
mod route_delivery;
mod o2o_reduction;
mod saturation;
mod schema;
mod search;
mod sigil;
mod simultaneity;
mod tagged;

pub use assignment::{assign, participations, Assignment, CellState};
pub use attributes::{demotable_types, event_attribute_cells, Demotion, EventAttribute};
pub use expansion::{
    agreed_routes, eventually_follows, expansion_candidates, expansion_work, free_types,
    routes, tuples_for, DerivationRoute, Disagreement, ExpansionCandidate, ExpansionDirection,
    Route, DEFAULT_THETA, EXPANSION_QUALIFIER, EXPANSION_WORK_BUDGET,
};
pub use lifetime::Lifetimes;
pub use arcs::{
    trace_variants_with,
    arc_set, arcs_and_components, df_fidelity, incidence_components, repair_connectivity,
    repair_incidence, trace_variants, ActivityIndexing, Arc, DfFidelity, TraceVariant,
    TraceVariants,
};
pub use bounds::{Bounds, ObjectBounds};
pub use canonical::{Difference, Fingerprint};
pub use annotate::{
    annotate, AnnotationRecord, CellAnnotation, Determination, Realisation, Relation,
};
pub use density::{
    state_tally, type_densities, TypeDensity, TypeState,
};
pub use cells::{
    ActivityCells, ActivityIndex, Cell, CellGrid, CutDecision, FoldDirection, KeepSet,
    DETERMINATION_THETA,
    ReconRoute,
};
pub use closure::{
    compose_closure, recorded_relations, ObjectFibre, ObjectFn, SchemaClosure,
    MAX_COMPOSE_DEPTH, MAX_WITNESSES_PER_PAIR,
};
pub use facts::{
    asserted_by_type, asserted_of_type, delivered_facts, fact_repair, facts, facts_from, Facts,
    DEFAULT_NOISE_THRESHOLD,
};
pub use contribution::{drawing_types, marginal_contribution, Contribution};
pub use coverage::{coverage, drawn, Chained, Coverage};
pub use declare_abstraction::{
    declare_abstraction, declare_coverage, declare_drawn, DeclareAbstraction, DeclareTypeModel,
    DECLARE_ARC_TYPES, DECLARE_COUNTS_FOR_FILTER, DECLARE_COUNTS_FOR_GENERATION,
    DECLARE_NOISE_THRESHOLD,
};
pub use model_abstraction::{model_abstraction, model_abstraction_with, Decision, TreeAbstraction, TypeModel, MODEL_NOISE_THRESHOLD};
pub use reconstruct::{
    carriers_of, involvement_clusters, reconstruct_ocpn, repair_ocpn, AbsenceRender, Carriers,
    InvolvementClusters, InvolvementRender, PlaceRole,
};
pub use route_delivery::{asserted_of_route, route_delivered};
pub use novelty::{
    novel_cells, novelty, novelty_by_type, ordered_pairs, Novelty, Pair,
};
pub use knobs::knobs;
pub use residual::{residual_report, CellResidual, CoveredBy, ResidualReport};
pub use o2o_reduction::{
    apply_o2o_reduction, expand_o2o, reduce_o2o, Implied, O2OReduction, RelationKey,
    O2O_REDUCTION_BUDGET,
};
pub use saturation::Saturation;
pub use abstraction::{
    Abstraction, Assertion, AssertionKind, ChainedCover, Cover, LogAbstraction, PairAbstraction,
};
pub use search::{
    exact_cells_per_type, reflow_layer, reflow_layer_with,
    activities_without_flow, best_of_two, best_of_two_with, best_of_two_with_objective, exact,
    exact_with_objective, greedy, greedy_with, greedy_with_objective, handoff, handoff_with,
    handoff_with_objective, FlowLayer, Objective, SearchInput, Strategy,
    EXACT_NODE_BUDGET, SEARCH_CLOSURE_BUDGET,
};
pub use guarantees::Guarantees;
pub use sigil::{
    decode, encode, flows, non_flow_cells, partially_marked_cells, set_flow, written,
    written_cells, Marks, NOT_FLOWING, WRITTEN,
};
pub use tagged::{
    annotation, annotation_of, flow_projection, full_projection, is_tagged, tag, tags,
    AnnotatedCell, Annotation,
};
pub use simultaneity::precedes;
pub use schema::{
    coparticipation_maps, generators, recorded_maps, Map, MapOrigin, ObjectTypeIndex,
    Rejection, StructuralSchema, MIN_TARGET_OBJECTS,
};
