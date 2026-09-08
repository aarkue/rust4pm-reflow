//! Binding wrappers for ReFlow over a [`SlimLinkedOCEL`].
//!
//! See [`crate::analysis::object_centric::schema_reduction`] for the algorithm.
//!
//! The entry point returns one computed assignment, the better of the two search
//! constructions ranked on arcs with participations as the tie-break, and the client edits
//! it cell by cell from there.
//!
//! The client sets the state of one cell at a time: `flow` says which cells contribute arcs,
//! `hold` reports a determined cell as involved instead of implied, and `remove`
//! empties a cell whether or not a map recomputes it. Only `remove` can lose data, and the
//! answer carries `removalRecoverable` per cell so a caller can tell the two cases apart
//! before choosing.
//!
//! Everything crosses the boundary by name, never by type or activity index. The indices are
//! assigned per run from a sorted list, so a client that stored one would read a different
//! cell after any change to the log.
use std::collections::{HashMap, HashSet};

use macros_process_mining::register_binding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analysis::object_centric::schema_reduction::{
    activities_without_flow, agreed_routes, annotate, assign, best_of_two,
    best_of_two_with_objective, carriers_of, coverage, delivered_facts, demotable_types,
    exact_with_objective, fact_repair, facts_from, flow_projection, flows, greedy_with_objective,
    handoff_with_objective, novel_cells, novelty, participations,
    reconstruct_ocpn, residual_report, tag, trace_variants, type_densities, AbsenceRender,
    ActivityIndexing, AnnotationRecord, Assignment, Bounds, Cell, CellGrid, Coverage, CoveredBy,
    ExpansionDirection, FlowLayer, InvolvementRender, LogAbstraction, MapOrigin, Novelty,
    Objective, Pair, PlaceRole, ReconRoute, Saturation, SchemaClosure, SearchInput, Strategy,
    StructuralSchema, TraceVariants, TypeState, DEFAULT_NOISE_THRESHOLD, DEFAULT_THETA,
    SEARCH_CLOSURE_BUDGET,
};
use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};
use crate::core::process_models::object_centric::ocpn::json::ObjectCentricPetriNetJson;
use crate::core::process_models::object_centric::ocpn::ObjectCentricPetriNet;
use crate::discovery::case_centric::inductive_miner::InductiveMinerOptions;
use crate::discovery::object_centric::ocpn::{discover_ocpn, ObjectCentricDiscoveryOptions};

/// Recorded participations above which the trace variants are not shipped.
///
/// They exist so a client can evaluate an edit without a round trip; past this size the
/// payload costs more than the round trip saves, so the client is told to ask instead.
const VARIANT_PARTICIPATION_BUDGET: usize = 400_000;

/// Facts listed back to the caller are capped: on a large log the difference can run to
/// thousands and the list is for reading, not for machine consumption.
const FACT_LIST_CAP: usize = 200;

/// One fact a flow layer no longer draws, named.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FactRef {
    /// The object type whose relation states it.
    pub object_type: String,
    /// Ordering: the earlier activity. Role: one of the two activities.
    pub from: String,
    /// Ordering: the later activity. Role: the other activity.
    pub to: String,
    /// The model still asserts it, it just no longer draws it as its own edge: two other
    /// orderings imply it. Adding participations can do this, so a fact leaving the drawn
    /// set is not the same as a fact leaving the model.
    pub implied: bool,
}

/// One cell of the grid, named.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CellRef {
    /// Activity name.
    pub activity: String,
    /// Object type name.
    pub object_type: String,
}

/// Where one cell ends up.
///
/// Three states of one fact, not three facts. The extraction picks one of them once, before
/// anyone knows what the log is for, and ReFlow moves cells between them in both
/// directions. **The state is a default, not a determination**: measurement gives a good
/// default and the reasoning behind it, and the decision stays the analyst's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CellStateKind {
    /// Tagged flow: the tuples contribute arcs.
    Flow,
    /// Tagged non-flow and nothing at the activity determines it: a badge in the activity
    /// node carrying counts.
    Involvement,
    /// Tagged non-flow and a route from a flow cell at the activity recomputes it, so the
    /// annotation draws it as a map. The tuples stay in the log.
    Implied,
}

/// Why a cell stopped drawing arcs. The two reasons are not interchangeable: only the first
/// permits the cell to leave the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CellReasonKind {
    /// **Reason (a).** A route determines it: another type kept at this activity, plus a
    /// relation from the schema, gives this type's objects at every event of the activity.
    /// Reaches `implied`.
    Determined,
    /// **Reason (b).** Its type's arcs carry no ordering another drawn type does not
    /// already carry -- marginal contribution zero. Reaches `involvement` only, because
    /// nothing recomputes the participations.
    Restated,
    /// Neither. A cell out of flow for neither reason is one the objective simply did not
    /// buy, and it stays in the log.
    Neither,
}

/// Which witness established a map. Complementary, and neither subsumes the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MapOriginKind {
    /// Read off the object-to-object relation, under one qualifier.
    Recorded,
    /// Forced by co-participation in the same events.
    Derived,
}

/// Which artifact to write: the tagged log, or one of its projections.
///
/// Qualifiers are free strings in OCEL 2.0, so both are valid OCEL 2.0 logs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum DepositMode {
    /// The tagged OCEL, which is what Reflow produces: nothing leaves, and every non-flow
    /// participation carries the tag.
    ///
    /// A tool that does not know the convention reads a non-flow tuple as an ordinary one
    /// and draws its arcs, which is the safe failure -- it loses no objects -- but it is
    /// also the reason this artifact reduces nothing for such a tool.
    #[default]
    Tagged,
    /// The flow projection of that log: only the tuples that draw arcs. This is what
    /// discovery reads, and the artifact that reduces anything for a tool that ignores the
    /// tag.
    ///
    /// This is the one artifact that loses data. An implied cell comes back through its
    /// route; an involved cell does not, and [`FlowProjectionCost`] is how big that is
    /// before the file is written.
    FlowProjection,
}

/// What the flow projection costs against the tagged log.
///
/// The projection drops every non-flow tuple, and the split is the whole content here: an
/// implied cell is recomputed by its route, and an involved cell -- a reason (b) cell,
/// or a cell no route reaches at all -- is gone for good. Both halves are counted in cells
/// and in participations, because a cell is one entry of a grid and carries anywhere from
/// seven tuples to a million.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FlowProjectionCost {
    /// Non-flow cells a route at their activity recomputes, i.e. the implied ones.
    pub recoverable_cells: usize,
    /// Participations those cells carry.
    pub recoverable_participations: usize,
    /// Non-flow cells nothing determines, i.e. the involved ones. **This is the one-way
    /// part**: the annotation carries no determination for them, so nothing puts their
    /// participations back once the projection has dropped them.
    pub unrecoverable_cells: usize,
    /// Participations those cells carry, and lose.
    pub unrecoverable_participations: usize,
}

impl FlowProjectionCost {
    /// Cells the projection empties that the tagged log keeps.
    pub fn cells(&self) -> usize {
        self.recoverable_cells + self.unrecoverable_cells
    }

    /// Participations it drops that the tagged log keeps.
    pub fn participations(&self) -> usize {
        self.recoverable_participations + self.unrecoverable_participations
    }
}

/// How a rule puts a cut cell back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CellRuleKind {
    /// Apply a map: `obj^T(e) = f[obj^S(e)]`.
    Function,
    /// Expand a fibre: `obj^T(e) = f^-1[obj^S(e)]`.
    Fibre,
    /// Union the images under a subset of the recorded qualifiers.
    QualifiedUnion,
}

/// A discovered map, as the type-by-type matrix shows it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SchemaMapInfo {
    /// Short name, `R1`, `R2`, ..., in the order of this list.
    ///
    /// The maps are what every cut rule and every expansion route is stated in terms of, so
    /// they need names. Position in this list, not a hash of the type pair: two maps between
    /// the same pair under different qualifiers are different maps and a reader choosing
    /// between them has to be able to say which one.
    pub id: String,
    /// Whether the map is in the transitive reduction of the map graph: an annotation that
    /// ships the generators can compose the rest.
    pub generator: bool,
    /// Source object type.
    pub source: String,
    /// Target object type.
    pub target: String,
    /// Which witness established it.
    pub origin: MapOriginKind,
    /// Qualifier of a recorded map; a leading `~` marks the reverse of the direction the
    /// log records.
    pub qualifier: Option<String>,
    /// Share of source objects the map is defined on.
    pub coverage: f64,
    /// Distinct objects the map reaches.
    pub image: usize,
    /// Source objects with no image.
    pub residual: usize,
    /// Source objects whose image is not forced, and which are therefore not guessed.
    pub ambiguous: usize,
}

/// A rule that puts one cell back, offered as a reason it can leave the log.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CellRule {
    /// The object type at the same activity that the rule reads from.
    pub witness_type: String,
    /// How it reconstructs the cell.
    pub kind: CellRuleKind,
    /// For a qualified union, the qualifiers it uses. Empty otherwise.
    pub qualifiers: Vec<String>,
}

/// One recorded cell and everything about it that does not depend on an assignment.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CellInfo {
    /// Activity name.
    pub activity: String,
    /// Object type name.
    pub object_type: String,
    /// Participations the cell holds.
    pub participations: usize,
    /// Rules that reconstruct it, one per usable witness type. A cell with none is
    /// irreducible at this activity and can never leave the log, whatever it says.
    ///
    /// Shipped so a client can recompute reason (a) itself: a cell is determined under an
    /// assignment exactly when some rule reads from a type at flow here, transitively.
    pub rules: Vec<CellRule>,
}

/// One cell under one assignment: which state it is in, why, and what recomputes it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CellAssignment {
    /// Activity name.
    pub activity: String,
    /// Object type name.
    pub object_type: String,
    /// Whether the extraction recorded the cell. `false` is an expansion: ReFlow
    /// wrote it, and the write is a modelling commitment the schema licensed.
    pub recorded: bool,
    /// The state it lands in.
    pub state: CellStateKind,
    /// Why it is not drawing arcs. `neither` on a flow cell.
    pub reason: CellReasonKind,
    /// The flow type at this activity whose map recomputes it. Reason (a), named.
    pub determined_by: Option<String>,
    /// A drawn type that already orders every activity pair this cell's type orders.
    /// Reason (b), named: "`Case_R` says nothing `Application` does not".
    pub restated_by: Option<String>,
    /// Participations it carries in the log as recorded.
    pub participations: usize,
    /// Whether the flow projection still recomputes this cell: whether some route at this
    /// activity puts its participations back out of what flows.
    ///
    /// Exactly `determinedBy != null`, restated as the question the client asks at the
    /// moment of choosing rather than as the reason a search gave. The tagged log keeps the
    /// tuples either way; this is what a reader of the flow projection alone loses. `false`
    /// on an expansion, which holds no recorded participations.
    pub removal_recoverable: bool,
    /// The reconstruction that recovers the cell. `null` unless the cell is `implied`.
    pub reconstruction: Option<Reconstruction>,
}

/// What recomputes one implied cell, read off the annotation (Def. Annotation).
///
/// The evidence grade is the whole content here: a **recorded** witness is an
/// object-to-object edge the extraction already wrote down, and a **derived** one is forced
/// by co-participation alone. Neither is stored in the tagged log -- both are recomputed
/// from it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Reconstruction {
    /// Participations the cell carries. They stay in the log, tagged non-flow.
    pub participations: usize,
    /// The type at the same activity the route reads from.
    pub source_type: String,
    /// The chain of types the reconstruction passes through, ending at `sourceType`. Its
    /// **depth** is what makes a route readable.
    pub route: Vec<String>,
    /// Which witness carries the relation.
    pub witness: MapOriginKind,
    /// Whether the relation is derived from co-participation rather than read off a
    /// recorded object-to-object qualifier. True exactly when `witness` is `derived`.
    pub derived: bool,
    /// The relation is read as a preimage rather than as a function.
    pub backward: bool,
    /// Distinct event-to-object qualifiers the cell's participations carry, tag dropped.
    pub qualifiers: Vec<String>,
    /// Objects of the cell's type the route does not put back. Empty whenever the per-event
    /// determinacy test held at every event, which is the normal case.
    pub residuals: usize,
}

/// One derivation of a cell the log does not record: which type it reads from, and which
/// maps it applies on the way.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExpansionRouteInfo {
    /// The type at the same activity the route reads from.
    pub source: String,
    /// Whether the route is read backwards, as a preimage rather than as a function.
    pub backward: bool,
    /// The maps it applies, in order, by [`SchemaMapInfo::id`]. One id is a generator,
    /// several are a composition.
    pub via: Vec<String>,
    /// Participations this route alone accounts for. Routes overlap, so these do not sum to
    /// the cell's total.
    pub tuples: usize,
}

/// An object type the schema licenses to be an attribute of another.
///
/// A **realisation of `implied`**, not a fourth state: both recover the type through the
/// same relation. The attribute case is strictly stronger, because the value travels on the
/// determining object itself and the flow projection then recovers it from the log alone,
/// with no schema attached.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DemotionInfo {
    /// The type that would become an attribute.
    pub object_type: String,
    /// The type that would carry it: every one of its objects has exactly one value.
    pub carrier: String,
    /// Distinct values the attribute would take.
    pub distinct_values: usize,
    /// Objects that stop being objects.
    pub objects_removed: usize,
    /// Participations that leave with them.
    pub participations_removed: usize,
    /// Ordering facts the type asserts in the full model.
    ///
    /// Licence is not advice: a type asserting none is a classification wearing a type's
    /// clothes, and one asserting some is an entity that merely happens to be determined.
    pub ordering_facts: usize,
    /// Types this one determines that its carrier cannot reach, which the move would put
    /// out of reach.
    pub blocked: Vec<String>,
}

/// A cell expansion could write: the log does not record it and the schema determines it.
///
/// Two filters, and they divide cleanly. `theta` is **correctness** -- do not write objects
/// that were not there -- and it is `admissible`. Novelty is **value** -- only write a cell
/// if the model learns an ordering it did not have -- and it is `novel`. Fan-out is a
/// diagnostic and decides nothing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExpansionCellInfo {
    /// Activity name.
    pub activity: String,
    /// Object type name.
    pub object_type: String,
    /// Participations it would write.
    pub tuples: usize,
    /// Events that gain at least one.
    pub events: usize,
    /// Distinct objects it would name. `null` where the cell was rejected by `theta` and so
    /// never scored: only an admitted cell is asked what it teaches.
    pub objects: Option<usize>,
    /// Participations whose event falls inside the written object's own recorded lifetime,
    /// `first(t) <= time(e) <= last(t)`. Counted per **tuple**: per event the test passes
    /// trivially whenever a cell writes a set.
    pub alive: usize,
    /// `alive / tuples`, which is what `theta` is compared against.
    pub admission_rate: f64,
    /// Tuples an admitted cell writes outside their own object's lifetime. An admitted cell
    /// is written **whole** and these are marked rather than dropped: leaving them out is
    /// what makes a cell ragged.
    pub extrapolated: usize,
    /// Whether the cell reaches `theta` of its tuples alive and may therefore be written.
    pub admissible: bool,
    /// Activity pairs its type would order that **no** type orders on the recorded log.
    /// Zero means the write teaches the model nothing and the search will not buy it;
    /// `null` means the cell was rejected by `theta` and never scored.
    pub novel: Option<usize>,
    /// Pairs its type would order that some type already orders. Restatement.
    pub echo: Option<usize>,
    /// Participations per distinct object: how far one object is smeared. A diagnostic.
    pub fanout: Option<f64>,
    /// Every derivation of the cell, largest share first.
    pub routes: Vec<ExpansionRouteInfo>,
}

/// One distinct activity sequence, and how many objects of the type walk it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TraceVariantInfo {
    /// Object type name.
    pub object_type: String,
    /// The activities in order, as indices into this overview's own `activities` list.
    ///
    /// Ordered by (timestamp, activity, event id), so a client that chains adjacent entries
    /// gets exactly the arcs the server would give it.
    ///
    /// Indices rather than names only here, and only within one response: a trace is as
    /// long as the object's life, and the resource types of a mid-sized log run to tens of
    /// thousands of entries.
    pub activity_indices: Vec<usize>,
    /// Objects of the type with exactly this sequence.
    pub objects: usize,
}

/// Something that was not computed, and why.
///
/// A client says what is missing instead of silently offering less on a big log than on a
/// small one.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkippedWork {
    /// What was skipped.
    pub name: String,
    /// Why it was skipped.
    pub reason: String,
}

/// What one object type's own behaviour says, before any map is consulted.
///
/// The classifier is `unique` and it is **threshold-free**: a type keeps drawing arcs while
/// some activity pair it orders no other surviving type orders. Ordering density decided
/// this until it was falsified at scale and is not reported.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TypeInfo {
    /// Object type name.
    pub object_type: String,
    /// Objects of the type.
    pub objects: usize,
    /// Activities it is recorded at.
    pub activities: usize,
    /// Activity pairs it orders: the strict eventually-follows closure over the recorded
    /// log.
    pub orders: usize,
    /// Of those, the pairs no **other** surviving type orders. **Zero is reason (b).**
    pub unique: usize,
    /// Whether it survives as a type that draws arcs.
    pub drawn: bool,
    /// A surviving type that orders one of its pairs, when it does not survive.
    pub covered_by: Option<String>,
    /// Activity pairs no object of the type ever joins: role facts, which hold a type at
    /// involvement rather than letting it leave.
    pub role: usize,
    /// Activity pairs the log cannot order, because the type's objects take part in both at
    /// one instant. **Not** "no ordering": the log declining to answer is not evidence for
    /// deleting participations, so a type whose pairs are predominantly tied holds at
    /// involvement.
    pub tied: usize,
    /// Some map lands in this type, so applying it recomputes the type's participations.
    pub determined: bool,
    /// Some map is defined on this type, so expanding its fibre recovers it.
    pub determines: bool,
    /// What its own behaviour asks for, before the map check.
    pub wants: CellStateKind,
    /// Where it lands after it. A type no map relates in either direction cannot leave the
    /// log however little it says, so it stops at involvement.
    pub state: CellStateKind,
}

/// What a flow layer delivers of the coverage target.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CoverageInfo {
    /// Target pairs a flow type orders with both endpoints at flow.
    pub drawn: usize,
    /// Target pairs delivered drawn **or** by a chain through a third activity, possibly
    /// using a different type per hop. Chaining is legitimate because the reduced model is
    /// exact for eventually-follows, and drawn-only coverage forces a type spanning the
    /// whole process.
    pub chained: usize,
    /// Every activity pair ordered in `max`, the theta-admissible saturation the constraint
    /// is stated against.
    pub target: usize,
}

/// How many cells and participations land in each state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StateTally {
    /// Recorded cells that draw arcs.
    pub flow_cells: usize,
    /// Recorded cells that stay in the log and draw none.
    pub involvement_cells: usize,
    /// Recorded cells whose participations a route recomputes.
    pub implied_cells: usize,
    /// Flow cells the extraction did not record, which expansion writes.
    pub expanded_cells: usize,
    /// Participations in each state, over the log as recorded.
    pub flow_participations: usize,
    /// Participations held at involvement.
    pub involvement_participations: usize,
    /// Participations the implied cells carry.
    pub implied_participations: usize,
    /// Participations the expansions write.
    pub written_participations: usize,
}

/// The annotation of a tagged log, in sizes.
///
/// Not required to *read* the result -- the log carries the tags and the annotation is
/// recomputed from them -- but it is what says how much of the log a reader of the flow
/// projection alone would have to rebuild.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AnnotationSummary {
    /// Non-flow cells, i.e. one entry per annotated cell.
    pub cells: usize,
    /// Of those, the involved ones: nothing at their activity determines them.
    pub involved: usize,
    /// Participations the non-flow cells carry. They stay in the log.
    pub nonflow_participations: usize,
    /// Implied cells whose relation has a recorded object-to-object witness.
    pub recorded_witnesses: usize,
    /// Implied cells whose relation is witnessed by co-participation only.
    pub coparticipation_witnesses: usize,
    /// The longest reconstruction chain any implied cell needs.
    pub max_route_depth: usize,
    /// Implied cells whose route does not return the recorded participations exactly.
    pub inexact: usize,
}

/// One search construction and what it produced.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StrategyInfo {
    /// `handoff`, `greedy` or `exact`.
    pub strategy: String,
    /// What it was minimising: `arcs`, `cells` or `participations`.
    pub objective: String,
    /// Whether it ran at all. A search over the chained-closure budget stands down and says
    /// so rather than approximating.
    pub ran: bool,
    /// Flow cells it bought.
    pub cells: usize,
    /// Arcs of the model it induces. **The objective.**
    pub arcs: usize,
    /// Participations it carries. The tie-break: arcs are coarse -- 225 against 147,385 on
    /// full Order Management -- so they rank flow layers but do not separate them.
    pub participations: usize,
    /// Cells the post-pass removed because no delivered pair needed them.
    pub dropped: usize,
    /// What it delivers of the target.
    pub coverage: CoverageInfo,
    /// Components of the activity-type incidence graph. Reported rather than enforced:
    /// checking connectivity first pins BPIC2017's `Case_R` to the flow layer for good.
    pub incidence_components: usize,
    /// Whether this is the computed assignment.
    pub winner: bool,
}

/// The oracle's own answer: one flow layer, and the assignment it induces.
///
/// **An optimum, not a rule and not a menu.** Minimise the flow layer subject to coverage,
/// connectivity and recoverability; whatever still says something but is not in the flow
/// layer is involved, and the rest is implied. The three states fall out of one
/// objective rather than being assigned by a separate rule.
///
/// Two constructions are run and the better is taken, because neither dominates: handoff
/// wins Order Management decisively and Container Logistics marginally, greedy wins Hinge
/// and BPIC2017. `strategy` says which won on this log.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RecommendationInfo {
    /// The construction that won.
    pub strategy: String,
    /// Whether any search ran. When false the flow layer is empty and every cell is
    /// reported unassigned rather than silently sent to involvement.
    pub ran: bool,
    /// Both constructions, so the margin is visible. At 3-5% the heuristic gap is not
    /// distinguishable from noise.
    pub strategies: Vec<StrategyInfo>,
    /// The cells that draw arcs, including any the extraction did not record.
    pub flow: Vec<CellRef>,
    /// Determined cells reported as involved anyway. Empty in the recommendation, which
    /// reports every cell the schema determines as implied; the client's `hold` fills this.
    pub held: Vec<CellRef>,
    /// Every recorded cell placed, plus the expansions.
    pub assignment: Vec<CellAssignment>,
    /// Cells and participations per state.
    pub tally: StateTally,
    /// What it delivers of the target.
    pub coverage: CoverageInfo,
    /// Arcs of the model it induces.
    pub arcs: usize,
    /// Components of the directly-follows graph.
    pub components: usize,
    /// Components of the activity-type incidence graph.
    pub incidence_components: usize,
    /// Activities left with no flow cell at all. Not fatal -- involvement still carries the
    /// objects -- but it is what the badge has to render.
    pub activities_without_flow: Vec<String>,
    /// The annotation of the tagged log it produces.
    pub annotation: AnnotationSummary,
    /// What the flow projection would cost against the tagged log.
    pub flow_projection: FlowProjectionCost,
}

/// Everything the cell editor needs about one log, computed once.
///
/// The schema is discovered from the **full** log and held fixed while an assignment is
/// edited, which is what makes derived maps safe to offer: re-deriving from the flow
/// projection could erase the co-occurrence that witnessed the map in the first place.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SchemaReductionOverview {
    /// Object type names, sorted.
    pub object_types: Vec<String>,
    /// Activity names, sorted.
    pub activities: Vec<String>,
    /// Every discovered map.
    pub maps: Vec<SchemaMapInfo>,
    /// The transitive reduction of the map graph, as (source, target) name pairs.
    pub generators: Vec<(String, String)>,
    /// Type pairs no map witnesses directly and a composition of maps reaches. They are
    /// usable as cut rules exactly like witnessed ones, and a reader following a badge
    /// through one takes an extra hop.
    pub composed_pairs: Vec<(String, String)>,
    /// Longest generator chain a derivation needs.
    pub derivation_depth: usize,
    /// Every recorded cell, with its rules.
    pub cells: Vec<CellInfo>,
    /// Per object type, what its own behaviour says. This is where reason (b) is measured.
    pub types: Vec<TypeInfo>,
    /// The oracle's own answer, and the default assignment the editor opens with.
    pub recommendation: RecommendationInfo,
    /// Every cell the log does not record and the schema determines, with what writing it
    /// would cost and what it would teach. Empty on a log past the expansion budget, where
    /// the reason appears in `skipped`.
    pub addable: Vec<ExpansionCellInfo>,
    /// Object types the schema licenses to be an attribute of another type: a realisation
    /// of `implied` that recovers from the log alone.
    pub demotable: Vec<DemotionInfo>,
    /// Recorded participations over the whole log.
    pub participations_total: usize,
    /// Coloured arcs of the directly-follows graph the log draws with every cell at flow:
    /// the model before any of this runs.
    ///
    /// Shipped so a client can put an arc count beside what it was without asking for a
    /// second evaluation of an assignment nobody chose.
    pub arcs_recorded: usize,
    /// Ordering facts the full model asserts: covering pairs of each type's
    /// eventually-follows relation.
    pub ordering_facts: usize,
    /// Role facts the full model asserts: activity pairs no object of the type ever joins.
    pub role_facts: usize,
    /// Activity pairs the log cannot order because both events share a timestamp. Reported
    /// beside the other two because a tie is *undetermined*, not "no ordering".
    pub tied_facts: usize,
    /// The noise threshold the facts were computed at.
    pub noise_threshold: f64,
    /// The share of a candidate cell's tuples that had to fall inside their own object's
    /// lifetime for the cell to be admitted.
    ///
    /// **Its published justification is void** and it is a parameter rather than a tuned
    /// constant; a client showing expansion should show this number beside it.
    pub theta: f64,
    /// Work that was not done on this log, with the reason.
    pub skipped: Vec<SkippedWork>,
    /// The distinct unprojected activity sequences per object type.
    ///
    /// Arcs, components and the directly-follows cost of *any* flow layer are functions of
    /// these, so a client holding them evaluates an edit itself instead of asking for
    /// [`schema_reduction_evaluate`] on every click.
    ///
    /// **Empty on a large log** (see `VARIANT_PARTICIPATION_BUDGET`): the client then has
    /// no way to compute arcs locally and must call [`schema_reduction_evaluate`] for them.
    pub trace_variants: Vec<TraceVariantInfo>,
}

/// What one assignment costs and what it still delivers.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AssignmentEvaluation {
    /// Every recorded cell placed, plus the expansions the flow layer writes.
    pub cells: Vec<CellAssignment>,
    /// Cells and participations per state.
    pub tally: StateTally,
    /// What the flow layer delivers of the target.
    pub coverage: CoverageInfo,
    /// Coloured arcs of the directly-follows graph the flow layer induces.
    pub arcs: usize,
    /// Components of that graph.
    pub components: usize,
    /// Arcs the reduced model asserts that no unprojected trace supports: what the
    /// directly-follows over-approximation costs. An arc jumps across a non-flow activity
    /// rather than breaking, so nothing is dropped and a little is over-approximated.
    pub arcs_unsupported: usize,
    /// Arcs between flow cells the reduction lost. Zero by construction; a non-zero value
    /// is a bug rather than a finding.
    pub arcs_missing: usize,
    /// Components of the activity-type incidence graph.
    pub incidence_components: usize,
    /// Activities left with no flow cell at all.
    pub activities_without_flow: Vec<String>,
    /// Ordering facts this model draws, of the full model's.
    pub ordering_facts_drawn: usize,
    /// Role facts this model draws, of the full model's.
    pub role_facts_drawn: usize,
    /// Ordering facts the full model draws and this one does not, capped for reading.
    pub ordering_facts_lost: Vec<FactRef>,
    /// Role facts the full model draws and this one does not, capped for reading.
    pub role_facts_lost: Vec<FactRef>,
    /// The annotation of the tagged log this assignment produces.
    pub annotation: AnnotationSummary,
    /// What the flow projection costs against the tagged log, split by whether a route puts
    /// the cell back. Stated here rather than at the moment of writing, because it is the
    /// one thing about that artifact that cannot be undone once it is chosen.
    pub flow_projection: FlowProjectionCost,
    /// Cells the caller named that neither the log records nor `theta` admits, so they were
    /// ignored. A client sending one is working from a stale grid.
    pub unknown_cells: Vec<CellRef>,
}

/// Everything both the overview and an evaluation read, built once per call.
struct Prepared {
    schema: StructuralSchema,
    closure: SchemaClosure,
    grid: CellGrid,
    acts: ActivityIndexing,
    bounds: Bounds,
    max: Saturation,
    /// Trace variants over `max`, so a flow layer holding a written cell reads its arcs as
    /// present rather than as missing.
    variants: TraceVariants,
    target: Vec<Pair>,
    novelty: Vec<Novelty>,
    skipped: Vec<SkippedWork>,
}

impl Prepared {
    fn n_activities(&self) -> usize {
        self.grid.activities.len()
    }

    fn cell_ref(&self, c: Cell) -> CellRef {
        CellRef {
            activity: self.grid.activities[c.0].clone(),
            object_type: self.schema.types[c.1].clone(),
        }
    }

    /// The grid a model that writes cells has to be measured against: a type written at
    /// every activity must not read as split.
    fn grid_with(&self, extra: &HashSet<Cell>) -> CellGrid {
        let mut g = self.grid.clone();
        g.cells.extend(extra.iter().copied());
        g
    }

    /// Participations one cell carries in the log as recorded.
    fn participations_of(&self, c: Cell) -> usize {
        self.grid
            .per_activity
            .get(c.0)
            .and_then(|here| here.slot(c.1).map(|j| here.counts[j]))
            .unwrap_or(0)
    }
}

fn prepare(ocel: &SlimLinkedOCEL, theta: f64) -> Prepared {
    let schema = StructuralSchema::discover(ocel);
    let closure = SchemaClosure::build(ocel, &schema);
    let grid = CellGrid::build(ocel, &schema, &closure);
    let acts = ActivityIndexing::build(ocel, &grid);
    let bounds = Bounds::build(ocel, &schema, &acts);

    // `max` is not a normal form and is not proposed as one. It exists because the coverage
    // constraint has to be stated against something, and stating it against the recorded log
    // makes the objective a function of how generously the extraction happened to record
    // participations rather than of the process.
    let (routes, _clashes) = agreed_routes(&schema);
    let max = Saturation::build(
        ocel,
        &schema,
        &grid,
        &acts,
        &routes,
        &bounds,
        theta,
        ExpansionDirection::default(),
    );

    let mut skipped = Vec::new();
    if !max.within_budget {
        skipped.push(SkippedWork {
            name: "expansion".to_string(),
            reason: format!(
                "{} route-object checks over the budget; `max` is the recorded log, so the \
                 coverage target and every expansion field are recorded-log numbers",
                max.work
            ),
        });
    }

    let variants = TraceVariants::build_with(ocel, &schema, &acts, &max.written);
    let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
    target.sort_unstable();
    let novelty = novelty(ocel, &acts, &bounds, &grid.cells, &max);

    Prepared {
        schema,
        closure,
        grid,
        acts,
        bounds,
        max,
        variants,
        target,
        novelty,
        skipped,
    }
}

/// A flow layer standing for "no search has been asked for".
///
/// Not an empty recommendation: `ran` is false, so the client shows every cell where the
/// extraction left it rather than reading an empty flow layer as a decision to draw nothing.
fn unrun_layer(p: &Prepared) -> FlowLayer {
    FlowLayer {
        strategy: Strategy::Handoff,
        objective: Objective::Arcs,
        cells: HashSet::new(),
        coverage: Coverage {
            drawn: 0,
            chained: 0,
            target: p.target.len(),
        },
        arcs: 0,
        participations: 0,
        dropped: 0,
        incidence_components: 0,
        ran: false,
    }
}

/// Run both constructions and take the better.
///
/// Novelty gates which expansions a search may buy: `theta` says the objects were there,
/// novelty says the model learns something by saying so.
fn recommend(p: &Prepared) -> (FlowLayer, Vec<FlowLayer>) {
    let objects = Saturation::objects_per_type(&p.schema);
    let novel = novel_cells(&p.novelty);
    let mut allowed = p.grid.cells.clone();
    allowed.extend(novel.keys().copied());
    let input = SearchInput {
        max: &p.max,
        variants: &p.variants,
        allowed: &allowed,
        recorded: &p.grid.cells,
        target: &p.target,
        objects_per_type: &objects,
        rep: &p.closure.rep,
        types: &p.schema.types,
        n_activities: p.n_activities(),
    };
    best_of_two(&input)
}

fn strategy_info(f: &FlowLayer, winner: bool) -> StrategyInfo {
    StrategyInfo {
        strategy: f.strategy.label().to_string(),
        objective: f.objective.label().to_string(),
        ran: f.ran,
        cells: f.cells.len(),
        arcs: f.arcs,
        participations: f.participations,
        dropped: f.dropped,
        coverage: CoverageInfo {
            drawn: f.coverage.drawn,
            chained: f.coverage.chained,
            target: f.coverage.target,
        },
        incidence_components: f.incidence_components,
        winner,
    }
}

fn type_state_kind(s: TypeState) -> CellStateKind {
    match s {
        TypeState::Flow => CellStateKind::Flow,
        TypeState::Involvement => CellStateKind::Involvement,
        TypeState::Implied => CellStateKind::Implied,
    }
}

fn annotation_summary(rec: &AnnotationRecord) -> AnnotationSummary {
    let (recorded, coparticipation) = rec.witness_split();
    let (involved, _implied) = rec.split();
    AnnotationSummary {
        cells: rec.cells.len(),
        involved,
        nonflow_participations: rec.participations(),
        recorded_witnesses: recorded,
        coparticipation_witnesses: coparticipation,
        max_route_depth: rec.max_route_depth(),
        inexact: rec.inexact(),
    }
}

/// What the flow projection costs, read off the rows the client is already shown.
///
/// The projection empties every non-flow cell, so the question is only which of them a route
/// puts back. That is `removalRecoverable`, summed. Rows are read after the `hold` fixup, so
/// a determined cell the analyst chose to show as involved still counts as recoverable.
fn flow_projection_cost(rows: &[CellAssignment]) -> FlowProjectionCost {
    let mut out = FlowProjectionCost::default();
    for r in rows
        .iter()
        .filter(|r| r.recorded && r.state != CellStateKind::Flow)
    {
        if r.removal_recoverable {
            out.recoverable_cells += 1;
            out.recoverable_participations += r.participations;
        } else {
            out.unrecoverable_cells += 1;
            out.unrecoverable_participations += r.participations;
        }
    }
    out
}

/// The tally of one assignment.
fn tally_of(p: &Prepared, a: &Assignment, written: usize) -> StateTally {
    let recorded_flow: Vec<Cell> = a
        .flow
        .iter()
        .filter(|c| p.grid.cells.contains(*c))
        .copied()
        .collect();
    StateTally {
        flow_cells: recorded_flow.len(),
        involvement_cells: a.involvement.len(),
        implied_cells: a.implied.len(),
        expanded_cells: a.expanded.len(),
        flow_participations: participations(&p.grid, &recorded_flow),
        involvement_participations: participations(&p.grid, &a.involvement),
        implied_participations: participations(&p.grid, &a.implied),
        written_participations: written,
    }
}

/// Place every cell, name the reason, and hang the reconstruction off it.
///
/// The two reasons are independent facts about a cell and both are reported. Reason (a) is
/// per cell and depends on the flow layer; reason (b) is per type and does not. Only (a)
/// makes a cell `implied`, because only a route can put its participations back.
fn assignment_rows(
    p: &Prepared,
    a: &Assignment,
    rec: &AnnotationRecord,
    restated_by: &HashMap<usize, String>,
) -> Vec<CellAssignment> {
    let determined = determining_types(p, &a.flow.iter().copied().collect());
    let record: HashMap<(String, String), usize> = rec
        .cells
        .iter()
        .enumerate()
        .map(|(i, m)| ((m.activity.clone(), m.object_type.clone()), i))
        .collect();

    let mut rows: Vec<CellAssignment> = [
        (&a.flow, CellStateKind::Flow),
        (&a.involvement, CellStateKind::Involvement),
        (&a.implied, CellStateKind::Implied),
    ]
    .into_iter()
    .flat_map(|(cells, state)| cells.iter().map(move |c| (*c, state)))
    .map(|(c, state)| row_for(p, rec, &record, &determined, restated_by, c, state))
    .collect();
    rows.sort_by(|x, y| (&x.activity, &x.object_type).cmp(&(&y.activity, &y.object_type)));
    rows
}

/// Reason (a) per cell, named: which flow type at this activity puts the cell back.
///
/// Read off the grid rather than off the annotation, because the annotation reports what the
/// analyst *chose* and this reports what the schema *permits*: a determined cell the analyst
/// shows as involved is still determined, and the client has to be able to see the cost.
fn determining_types(p: &Prepared, flow: &HashSet<Cell>) -> HashMap<Cell, String> {
    let mut out = HashMap::new();
    for (a, here) in p.grid.per_activity.iter().enumerate() {
        let kept: Vec<usize> = here
            .present
            .iter()
            .filter(|t| flow.contains(&(a, **t)))
            .copied()
            .collect();
        for (t, route) in here.present.iter().zip(here.determining_routes(&kept)) {
            if let Some((i, _)) = route {
                out.insert((a, *t), p.schema.types[here.present[i]].clone());
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn row_for(
    p: &Prepared,
    rec: &AnnotationRecord,
    record: &HashMap<(String, String), usize>,
    determined: &HashMap<Cell, String>,
    restated_by: &HashMap<usize, String>,
    cell: Cell,
    state: CellStateKind,
) -> CellAssignment {
    let activity = p.grid.activities[cell.0].clone();
    let object_type = p.schema.types[cell.1].clone();
    let mv = record
        .get(&(activity.clone(), object_type.clone()))
        .map(|i| &rec.cells[*i]);
    let determination = mv.and_then(|m| m.determined_by.as_ref());
    let determined_by = determined.get(&cell).cloned();
    let reconstruction = match (mv, determination) {
        (Some(mv), Some(d)) if state == CellStateKind::Implied => Some(Reconstruction {
            participations: mv.participations,
            source_type: d.source_type.clone(),
            route: d.route.clone(),
            witness: if d.realisation.derived {
                MapOriginKind::Derived
            } else {
                MapOriginKind::Recorded
            },
            derived: d.realisation.derived,
            backward: d.realisation.backward,
            qualifiers: mv.qualifiers.clone(),
            residuals: d.residuals.len(),
        }),
        _ => None,
    };
    let restated = restated_by.get(&cell.1).cloned();
    let reason = match state {
        CellStateKind::Flow => CellReasonKind::Neither,
        _ if determined_by.is_some() => CellReasonKind::Determined,
        _ if restated.is_some() => CellReasonKind::Restated,
        _ => CellReasonKind::Neither,
    };
    CellAssignment {
        activity,
        object_type,
        recorded: p.grid.cells.contains(&cell),
        state,
        reason,
        removal_recoverable: determined_by.is_some(),
        determined_by,
        restated_by: restated,
        participations: p.participations_of(cell),
        reconstruction,
    }
}

/// Short name of a map, by its position in the overview's own map list.
///
/// `R` for relation, so a cut rule, an expansion route and a guard on a transition all cite
/// the same symbol: `orders = R3(items)` names the same `R3` the schema list does.
fn map_id(i: usize) -> String {
    format!("R{}", i + 1)
}

/// Resolve named cells against the grid and the admitted expansions.
///
/// A cell the log neither records nor `theta` admits is returned rather than silently
/// dropped, since a client sending one is working from a stale grid.
fn resolve_cells(p: &Prepared, refs: Vec<CellRef>) -> (HashSet<Cell>, Vec<CellRef>) {
    let admitted = p.max.expanded_cells();
    let mut ok = HashSet::new();
    let mut rejected = Vec::new();
    for r in refs {
        let a = p.grid.activities.iter().position(|x| *x == r.activity);
        let t = p.schema.types.iter().position(|x| *x == r.object_type);
        match (a, t) {
            (Some(a), Some(t))
                if p.grid.cells.contains(&(a, t)) || admitted.contains(&(a, t)) =>
            {
                ok.insert((a, t));
            }
            _ => rejected.push(r),
        }
    }
    (ok, rejected)
}

/// Recorded cells only, for `hold`: an expansion cannot be held, since it is not in the log
/// to begin with.
fn resolve_recorded(p: &Prepared, refs: Vec<CellRef>) -> (HashSet<Cell>, Vec<CellRef>) {
    let mut ok = HashSet::new();
    let mut rejected = Vec::new();
    for r in refs {
        let a = p.grid.activities.iter().position(|x| *x == r.activity);
        let t = p.schema.types.iter().position(|x| *x == r.object_type);
        match (a, t) {
            (Some(a), Some(t)) if p.grid.cells.contains(&(a, t)) => {
                ok.insert((a, t));
            }
            _ => rejected.push(r),
        }
    }
    (ok, rejected)
}

/// The types marginal contribution takes out of the flow layer, each named with a survivor
/// that already says what it says. Reason (b), per type.
fn restated_by(p: &Prepared, rows: &[TypeInfo]) -> HashMap<usize, String> {
    let index: HashMap<&str, usize> = p
        .schema
        .types
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    rows.iter()
        .filter(|r| !r.drawn && r.orders > 0)
        .filter_map(|r| {
            let t = *index.get(r.object_type.as_str())?;
            Some((t, r.covered_by.clone()?))
        })
        .collect()
}

/// The per-type table, which is where reason (b) is measured.
fn type_rows(p: &Prepared, tau: f64) -> Vec<TypeInfo> {
    let recorded_facts = facts_from(&p.bounds, &p.grid.cells, tau);
    let max_facts = facts_from(&p.max.bounds, &p.max.cells, tau);
    let objects = Saturation::objects_per_type(&p.schema);
    type_densities(
        p.schema.types.len(),
        &p.grid.cells,
        &recorded_facts,
        &p.max.cells,
        &max_facts,
        &p.schema.pairs(),
        &p.closure.rep,
        &p.schema.types,
    )
    .into_iter()
    .map(|r| TypeInfo {
        object_type: p.schema.types[r.object_type].clone(),
        objects: objects.get(r.object_type).copied().unwrap_or(0),
        activities: r.activities,
        orders: r.asserted,
        unique: r.unique,
        drawn: r.unique > 0,
        covered_by: r.covered_by.map(|s| p.schema.types[s].clone()),
        role: r.role,
        tied: r.tied,
        determined: r.determined,
        determines: r.determines,
        wants: type_state_kind(r.wants),
        state: type_state_kind(r.state),
    })
    .collect()
}

/// The schema, the cell grid and the computed assignment for one log.
#[register_binding]
pub fn schema_reduction_overview(
    ocel: &SlimLinkedOCEL,
    noise_threshold: Option<f64>,
    theta: Option<f64>,
    search: Option<bool>,
) -> SchemaReductionOverview {
    let tau = noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let p = prepare(ocel, theta);
    let types = &p.schema.types;

    let (gens, depth) = p.schema.generators();
    let gen_pairs: HashSet<(usize, usize)> = gens.iter().copied().collect();

    let maps: Vec<SchemaMapInfo> = p
        .schema
        .maps()
        .enumerate()
        .map(|(i, m)| SchemaMapInfo {
            id: map_id(i),
            generator: gen_pairs.contains(&(m.source, m.target)),
            source: types[m.source].clone(),
            target: types[m.target].clone(),
            origin: match m.origin {
                MapOrigin::Recorded => MapOriginKind::Recorded,
                MapOrigin::Coparticipation => MapOriginKind::Derived,
            },
            qualifier: m.qualifier.clone(),
            coverage: m.coverage(),
            image: m.image,
            residual: m.residual,
            ambiguous: m.ambiguous,
        })
        .collect();

    let generators = gens
        .into_iter()
        .map(|(s, t)| (types[s].clone(), types[t].clone()))
        .collect();

    let witnessed = p.schema.pairs();
    let mut composed_pairs: Vec<(String, String)> = p
        .closure
        .maps
        .keys()
        .filter(|pair| !witnessed.contains(pair))
        .map(|(s, t)| (types[*s].clone(), types[*t].clone()))
        .collect();
    composed_pairs.sort();

    let mut cells = Vec::with_capacity(p.grid.len());
    for (aix, cellset) in p.grid.per_activity.iter().enumerate() {
        for (j, t) in cellset.present.iter().enumerate() {
            let rules = cellset
                .recon
                .iter()
                .enumerate()
                .filter_map(|(i, row)| {
                    row[j].as_ref().map(|route| CellRule {
                        witness_type: types[cellset.present[i]].clone(),
                        kind: match route {
                            ReconRoute::Function { .. } => CellRuleKind::Function,
                            ReconRoute::Fibre { .. } => CellRuleKind::Fibre,
                            ReconRoute::QualifiedUnion { .. } => CellRuleKind::QualifiedUnion,
                        },
                        qualifiers: match route {
                            ReconRoute::QualifiedUnion { qualifiers } => qualifiers.clone(),
                            _ => Vec::new(),
                        },
                    })
                })
                .collect();
            cells.push(CellInfo {
                activity: p.grid.activities[aix].clone(),
                object_type: types[*t].clone(),
                participations: cellset.counts[j],
                rules,
            });
        }
    }
    cells.sort_by(|x, y| (&x.activity, &x.object_type).cmp(&(&y.activity, &y.object_type)));

    let type_table = type_rows(&p, tau);
    let restated = restated_by(&p, &type_table);

    // Expansion, the other direction of the same oracle, over the same grid: cells the log
    // does not record and the schema determines. Scored on both filters, because they answer
    // different questions and a client has to see both.
    let novelty_of: HashMap<Cell, &Novelty> =
        p.novelty.iter().map(|n| (n.cell, n)).collect();
    let addable: Vec<ExpansionCellInfo> = p
        .max
        .candidates
        .iter()
        .map(|c| {
            let n = novelty_of.get(&c.cell);
            ExpansionCellInfo {
                activity: p.grid.activities[c.cell.0].clone(),
                object_type: types[c.cell.1].clone(),
                tuples: c.tuples.len(),
                events: c.events,
                objects: n.map(|n| n.objects),
                alive: c.alive,
                admission_rate: c.admission_rate(),
                extrapolated: c.extrapolated(),
                admissible: c.admissible,
                novel: n.map(|n| n.novel()),
                echo: n.map(|n| n.echo),
                fanout: n.map(|n| n.fanout),
                routes: c
                    .routes
                    .iter()
                    .map(|r| ExpansionRouteInfo {
                        source: types[r.source].clone(),
                        backward: r.backward,
                        via: r.via.iter().map(|i| map_id(*i)).collect(),
                        tuples: r.tuples.len(),
                    })
                    .collect(),
            }
        })
        .collect();

    let recorded_facts = facts_from(&p.bounds, &p.grid.cells, tau);
    let demotable = demotable_types(&p.schema, &p.closure, &p.grid, &recorded_facts)
        .into_iter()
        .map(|d| DemotionInfo {
            object_type: types[d.object_type].clone(),
            carrier: types[d.carrier].clone(),
            distinct_values: d.distinct_values,
            objects_removed: d.objects_removed,
            participations_removed: d.participations_removed,
            ordering_facts: d.ordering_facts,
            blocked: d.blocked.iter().map(|u| types[*u].clone()).collect(),
        })
        .collect();

    let variants = if p.grid.e2o_total <= VARIANT_PARTICIPATION_BUDGET {
        trace_variants(ocel, &p.schema, &p.acts)
    } else {
        Vec::new()
    };
    let variants = variants
        .into_iter()
        .map(|v| TraceVariantInfo {
            object_type: types[v.object_type].clone(),
            activity_indices: v.activities,
            objects: v.objects,
        })
        .collect();

    // The searches are the expensive half -- minutes on a large log, where everything above is
    // seconds -- and nothing on the screen needs them to open. Off unless asked for, so the
    // editor appears at once with every cell where the extraction left it, and the analyst
    // asks for a construction when they want one.
    let (best, both) = if search.unwrap_or(false) {
        recommend(&p)
    } else {
        (unrun_layer(&p), Vec::new())
    };
    let mut skipped = p.skipped.clone();
    for f in both.iter().filter(|f| !f.ran) {
        skipped.push(SkippedWork {
            name: format!("{} search", f.strategy.label()),
            reason: format!(
                "over the {SEARCH_CLOSURE_BUDGET} chained-closure budget; reported as not run \
                 rather than approximated"
            ),
        });
    }
    let recommendation = recommendation_of(ocel, &p, &best, &both, &restated);

    SchemaReductionOverview {
        object_types: types.clone(),
        activities: p.grid.activities.clone(),
        maps,
        generators,
        composed_pairs,
        derivation_depth: depth,
        cells,
        types: type_table,
        recommendation,
        addable,
        demotable,
        participations_total: p.grid.e2o_total,
        arcs_recorded: p.variants.arcs_and_components(&p.grid.cells).0,
        ordering_facts: recorded_facts.ordering.len(),
        role_facts: recorded_facts.role.len(),
        tied_facts: recorded_facts.tied.len(),
        noise_threshold: tau,
        theta,
        skipped,
        trace_variants: variants,
    }
}

fn recommendation_of(
    ocel: &SlimLinkedOCEL,
    p: &Prepared,
    best: &FlowLayer,
    both: &[FlowLayer],
    restated: &HashMap<usize, String>,
) -> RecommendationInfo {
    // The search covers pairs, not cells, so its own answer can still leave an ordering fact
    // drawn by a type the license does not cover at both endpoints; repairing before anything
    // is shown keeps the preview the same assignment `apply_assignment` would deposit.
    let full_facts_for_repair = facts_from(&p.bounds, &p.grid.cells, DEFAULT_NOISE_THRESHOLD);
    let mut cells = best.cells.clone();
    fact_repair(&p.grid, &full_facts_for_repair, &mut cells);

    let a = assign(&p.grid, &cells, &HashSet::new());
    let rec = annotate(ocel, &p.schema, &p.closure, &p.grid, &cells, &HashSet::new());
    let written = written_participations(p, &a);
    let (arcs, components) = p.variants.arcs_and_components(&cells);
    let rows = assignment_rows(p, &a, &rec, restated);
    RecommendationInfo {
        strategy: best.strategy.label().to_string(),
        ran: best.ran,
        strategies: both
            .iter()
            .map(|f| strategy_info(f, best.ran && f.strategy == best.strategy))
            .collect(),
        flow: cells.iter().map(|c| p.cell_ref(*c)).collect(),
        held: Vec::new(),
        flow_projection: flow_projection_cost(&rows),
        assignment: rows,
        tally: tally_of(p, &a, written),
        coverage: CoverageInfo {
            drawn: best.coverage.drawn,
            chained: best.coverage.chained,
            target: best.coverage.target,
        },
        arcs,
        components,
        incidence_components: best.incidence_components,
        activities_without_flow: activities_without_flow(&cells, p.n_activities())
            .into_iter()
            .map(|a| p.grid.activities[a].clone())
            .collect(),
        annotation: annotation_summary(&rec),
    }
}

/// Participations the expansions in a flow layer would write.
fn written_participations(p: &Prepared, a: &Assignment) -> usize {
    let expanded: HashSet<Cell> = a.expanded.iter().copied().collect();
    p.max
        .admitted_cells()
        .filter(|c| expanded.contains(&c.cell))
        .map(|c| c.tuples.len())
        .sum()
}

/// Evaluate one assignment: where every cell lands, what the model still delivers, and what
/// applying it would cost.
///
/// One override set per state, all three per cell, so the client sets a state rather than
/// learning a verb. `flow` names the cells that draw arcs, and it is both levers of the
/// oracle at once: a cell the log records is kept drawing, a cell it does not is written.
/// `hold` names determined cells to report as involved rather than implied -- a rendering
/// choice, not a measurement. `remove` names cells to take out of the flow layer whether or
/// not anything determines them.
///
/// Nothing leaves the tagged log under any of them; what a `remove` costs is what the flow
/// projection then drops, which `removalRecoverable` states per cell.
///
/// A cell named on neither side of the grid is reported in `unknownCells` rather than
/// silently dropped, and a cell named in both `hold` and `remove` is held.
#[register_binding]
pub fn schema_reduction_evaluate(
    ocel: &SlimLinkedOCEL,
    flow: Vec<CellRef>,
    hold: Vec<CellRef>,
    remove: Vec<CellRef>,
    noise_threshold: Option<f64>,
    theta: Option<f64>,
) -> AssignmentEvaluation {
    let tau = noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let p = prepare(ocel, theta);

    let (mut flow_cells, mut unknown_cells) = resolve_cells(&p, flow);
    let (held, unknown_held) = resolve_recorded(&p, hold);
    let (dropped, unknown_dropped) = resolve_recorded(&p, remove);
    unknown_cells.extend(unknown_held);
    unknown_cells.extend(unknown_dropped);

    // Same repair `apply_assignment` runs, so a cell another type's own arcs need is shown at
    // flow here too -- the analyst sees the cost of the edit that is actually going to be
    // deposited, not the one they typed.
    flow_cells.retain(|c| !dropped.contains(c));
    let full_facts_for_repair = facts_from(&p.bounds, &p.grid.cells, DEFAULT_NOISE_THRESHOLD);
    fact_repair(&p.grid, &full_facts_for_repair, &mut flow_cells);

    let a = assign(&p.grid, &flow_cells, &HashSet::new());
    let rec = annotate(ocel, &p.schema, &p.closure, &p.grid, &flow_cells, &held);
    let type_table = type_rows(&p, tau);
    let restated = restated_by(&p, &type_table);
    let mut rows = assignment_rows(&p, &a, &rec, &restated);
    // A held cell is at involvement whatever the map says, and the reason stays `determined`
    // so the client can see what the hold is costing in file size.
    let held_names: HashSet<(&str, &str)> = held
        .iter()
        .map(|(x, y)| {
            (
                p.grid.activities[*x].as_str(),
                p.schema.types[*y].as_str(),
            )
        })
        .collect();
    for r in rows.iter_mut() {
        if held_names.contains(&(r.activity.as_str(), r.object_type.as_str())) {
            r.state = CellStateKind::Involvement;
            r.reconstruction = None;
        }
    }

    let expanded: HashSet<Cell> = a.expanded.iter().copied().collect();
    let grid_ex = p.grid_with(&expanded);
    let (arcs, components) = p.variants.arcs_and_components(&flow_cells);
    let fid = p.variants.df_fidelity(&grid_ex, &flow_cells);
    let cov = coverage(&p.max.bounds, &flow_cells, &p.target, p.n_activities());

    let full_facts = facts_from(&p.bounds, &p.grid.cells, tau);
    let drawn_facts = facts_from(&p.max.bounds, &flow_cells, tau);
    // Delivered, not drawn: a fact a flow type carries through a map at both endpoints is
    // still in the model, one application away.
    let kept_facts = delivered_facts(
        &grid_ex,
        &full_facts,
        &drawn_facts,
        &flow_cells,
        p.schema.types.len(),
    );

    let name_fact = |f: &(usize, usize, usize)| FactRef {
        object_type: p.schema.types[f.0].clone(),
        from: p.grid.activities[f.1].clone(),
        to: p.grid.activities[f.2].clone(),
        implied: drawn_facts.asserted.contains(f),
    };
    let sort_facts = |mut v: Vec<FactRef>| {
        v.sort_by(|a, b| {
            (&a.object_type, &a.from, &a.to).cmp(&(&b.object_type, &b.from, &b.to))
        });
        v.truncate(FACT_LIST_CAP);
        v
    };
    let ordering_lost = sort_facts(
        full_facts
            .ordering
            .difference(&kept_facts.ordering)
            .map(name_fact)
            .collect(),
    );
    let role_lost = sort_facts(
        full_facts
            .role
            .difference(&kept_facts.role)
            .map(name_fact)
            .collect(),
    );

    let mut tally = tally_of(&p, &a, written_participations(&p, &a));
    // Holding a cell moves its participations from the map column to the badge column;
    // nothing else about the model changes, which is exactly why it is a rendering choice.
    let held_participations: usize = held
        .iter()
        .filter(|c| a.implied.contains(c))
        .map(|c| p.participations_of(*c))
        .sum();
    let held_cells = held.iter().filter(|c| a.implied.contains(c)).count();
    tally.implied_cells -= held_cells;
    tally.involvement_cells += held_cells;
    tally.implied_participations -= held_participations;
    tally.involvement_participations += held_participations;

    AssignmentEvaluation {
        flow_projection: flow_projection_cost(&rows),
        cells: rows,
        tally,
        coverage: CoverageInfo {
            drawn: cov.drawn,
            chained: cov.chained,
            target: cov.target,
        },
        arcs,
        components,
        arcs_unsupported: fid.spurious,
        arcs_missing: fid.missing,
        incidence_components:
            crate::analysis::object_centric::schema_reduction::incidence_components(&flow_cells),
        activities_without_flow: activities_without_flow(&flow_cells, p.n_activities())
            .into_iter()
            .map(|a| p.grid.activities[a].clone())
            .collect(),
        ordering_facts_drawn: kept_facts.ordering.intersection(&full_facts.ordering).count(),
        role_facts_drawn: kept_facts.role.intersection(&full_facts.role).count(),
        ordering_facts_lost: ordering_lost,
        role_facts_lost: role_lost,
        annotation: annotation_summary(&rec),
        unknown_cells,
    }
}

/// One ordering assertion `(from, to)`, named.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OrderingRef {
    /// The earlier activity.
    pub from: String,
    /// The later activity.
    pub to: String,
}

/// How a covered assertion reaches the reader, named for the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CoveredByKind {
    /// A kept type orders it directly, both endpoints kept.
    Drawn,
    /// Reached by composing drawn pairs through a third activity.
    Chained,
    /// Delivered by an object-to-object route, not by activity composition.
    Routed,
}

/// One assertion still shown, and how.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CoveredOrdering {
    /// The activity pair this assertion orders.
    pub pair: OrderingRef,
    /// How the keep-set still shows it.
    pub covered_by: CoveredByKind,
}

/// The assertions of one non-flow cell, split by whether the keep-set still shows them.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CellResidualInfo {
    /// The cell these assertions are named against.
    pub cell: CellRef,
    /// Assertions still shown, and how.
    pub covered: Vec<CoveredOrdering>,
    /// Assertions the keep-set would leave unshown.
    pub residual: Vec<OrderingRef>,
}

/// The coverage report of a candidate flow layer: every ordering assertion of the log,
/// how many the keep-set delivers and by which mechanism, and the per-cell breakdown for
/// cells outside flow.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResidualReportInfo {
    /// Ordering assertions of the full log.
    pub total: usize,
    /// Assertions a kept type orders directly, both endpoints kept.
    pub drawn: usize,
    /// Assertions delivered only by composing drawn pairs through a third activity.
    pub chained: usize,
    /// Assertions delivered only by an object-to-object route.
    pub routed: usize,
    /// Assertions covered by none of the three mechanisms.
    pub residual: usize,
    /// Cells the keep-set left outside flow, each with its own assertions split
    /// covered/uncovered.
    pub cells: Vec<CellResidualInfo>,
    /// Cells the caller named that neither the log records nor `theta` admits.
    pub unknown_cells: Vec<CellRef>,
}

/// The coverage report for a candidate `flow` layer: which ordering assertions of the log
/// it still shows -- directly, chained, or by a delivered route -- and which non-flow cell
/// an unshown one is named against.
///
/// Additive to [`schema_reduction_evaluate`]: that endpoint's `orderingFactsLost` reports
/// what the determinacy repair (`delivered_facts`) fails to carry forward. This reports
/// the narrower, different guarantee the search itself enforces when it builds `flow` --
/// `coverage::drawn` and the chained closure, widened here to also credit a delivered
/// route -- so a client can check a hand-edited `flow` against the same test the
/// oracle's own recommendation already passes.
#[register_binding]
pub fn schema_reduction_residual(
    ocel: &SlimLinkedOCEL,
    flow: Vec<CellRef>,
    noise_threshold: Option<f64>,
    theta: Option<f64>,
) -> ResidualReportInfo {
    let tau = noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let p = prepare(ocel, theta);
    let (flow_cells, unknown_cells) = resolve_cells(&p, flow);

    // Routes are schema-only and cheap to recompute; `Prepared` does not carry them.
    let (routes, _clashes) = agreed_routes(&p.schema);
    let full = facts_from(&p.max.bounds, &p.max.cells, tau);
    let report = residual_report(&p.max.bounds, &full, &flow_cells, &routes, p.n_activities());

    let name_pair = |(x, y): Pair| OrderingRef {
        from: p.grid.activities[x].clone(),
        to: p.grid.activities[y].clone(),
    };
    let name_covered_by = |c: CoveredBy| match c {
        CoveredBy::Drawn => CoveredByKind::Drawn,
        CoveredBy::Chained => CoveredByKind::Chained,
        CoveredBy::Routed => CoveredByKind::Routed,
    };

    ResidualReportInfo {
        total: report.total,
        drawn: report.drawn,
        chained: report.chained,
        routed: report.routed,
        residual: report.residual,
        cells: report
            .cells
            .iter()
            .map(|c| CellResidualInfo {
                cell: p.cell_ref(c.cell),
                covered: c
                    .covered
                    .iter()
                    .map(|(pair, by)| CoveredOrdering {
                        pair: name_pair(*pair),
                        covered_by: name_covered_by(*by),
                    })
                    .collect(),
                residual: c.residual.iter().copied().map(name_pair).collect(),
            })
            .collect(),
        unknown_cells,
    }
}

/// Apply an assignment and hand back the tagged log, or its flow projection.
///
/// One step: tag every recorded participation `flow` or `nonflow` by the cell it belongs to.
/// Expansions are written last, carrying the `+` mark, so schema rediscovery does not read
/// them back as evidence for the relation that licensed them.
///
/// The override sets are [`schema_reduction_evaluate`]'s, and they have to be, or the
/// deposit is not the assignment the analyst was shown the cost of.
///
/// `mode` chooses between the two artifacts, and the choice is about who reads the result
/// rather than about the assignment.
///
/// - `tagged`: the tagged OCEL. Nothing is removed and every non-flow participation carries
///   the tag, so dropping the tags returns the input log exactly.
/// - `flowProjection`: only the participations that draw arcs stay. This is what discovery
///   reads, and the artifact for a tool that does not read the tag, which would otherwise
///   see a non-flow participation as an ordinary one and draw its arcs back.
///
/// Qualifiers are free strings in OCEL 2.0, so both are valid OCEL 2.0 logs.
///
/// `flowProjection` is the one mode that can lose data. Its cost is not in the log it hands
/// back but in `schema_reduction_evaluate`'s `flowProjection`, which splits the non-flow
/// cells into the ones a route recomputes and the ones nothing does, in cells and in
/// participations, so the size of the one-way part is on the screen before the file is
/// written.
#[register_binding]
pub fn schema_reduction_apply(
    ocel: &SlimLinkedOCEL,
    flow: Vec<CellRef>,
    hold: Vec<CellRef>,
    remove: Vec<CellRef>,
    mode: Option<DepositMode>,
    theta: Option<f64>,
) -> SlimLinkedOCEL {
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let p = prepare(ocel, theta);
    let (flow_cells, _) = resolve_cells(&p, flow);
    let (held, _) = resolve_recorded(&p, hold);
    let (dropped, _) = resolve_recorded(&p, remove);
    apply_assignment(
        ocel,
        &p,
        &flow_cells,
        &held,
        &dropped,
        mode.unwrap_or_default(),
    )
}

/// Which construction to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum SearchStrategy {
    /// Run both constructions and take the better under the chosen objective.
    Best,
    /// One flow type per activity, a second only where coverage forces it.
    Handoff,
    /// Cheapest-first cover.
    Greedy,
    /// The minimum of the objective over every preserving layer, by enumeration over per-type
    /// rows. Affordable on a log the size of the corpus and not on a large one: where it is
    /// not, the layer comes back with `ran: false` rather than an approximation wearing the
    /// name `exact`.
    Exact,
}

/// What a search minimises.
///
/// The feasible region is the same for all of them -- it is what the schema and the
/// abstraction carve out -- and this picks the point in it. Every one is a function of the
/// assignment alone, so no model is discovered inside the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum SearchObjective {
    /// Coloured directly-follows arcs. The default.
    Arcs,
    /// Cells left in the flow layer. Counts the grid and never the model.
    Cells,
    /// Participations the flow layer carries.
    Participations,
}

impl From<SearchObjective> for Objective {
    fn from(o: SearchObjective) -> Self {
        match o {
            SearchObjective::Arcs => Objective::Arcs,
            SearchObjective::Cells => Objective::Cells,
            SearchObjective::Participations => Objective::Participations,
        }
    }
}

/// Run one construction and return the flow layer it builds.
///
/// Separate from [`schema_reduction_overview`] because it is the expensive half: the schema,
/// the grid and the saturation are seconds on the largest log in the corpus and the searches
/// are minutes. Opening an editor should not cost minutes, and an analyst who wants to place
/// the cells by hand should not have to wait for an answer they are about to overwrite.
///
/// The strategies are offered separately, not only as `best`, because neither dominates and
/// the margin between them is often within noise: handoff wins Order Management and BPIC2017,
/// greedy wins Container Logistics and Age of Empires, Hinge is a tie. Which one an analyst
/// prefers is a question about the picture they want, and `best` answers only the arc count.
#[register_binding]
pub fn schema_reduction_search(
    ocel: &SlimLinkedOCEL,
    strategy: Option<SearchStrategy>,
    objective: Option<SearchObjective>,
    noise_threshold: Option<f64>,
    theta: Option<f64>,
) -> RecommendationInfo {
    let tau = noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let p = prepare(ocel, theta);
    let objects = Saturation::objects_per_type(&p.schema);
    let novel = novel_cells(&p.novelty);
    let mut allowed = p.grid.cells.clone();
    allowed.extend(novel.keys().copied());
    let input = SearchInput {
        max: &p.max,
        variants: &p.variants,
        allowed: &allowed,
        recorded: &p.grid.cells,
        target: &p.target,
        objects_per_type: &objects,
        rep: &p.closure.rep,
        types: &p.schema.types,
        n_activities: p.n_activities(),
    };
    let obj: Objective = objective.unwrap_or(SearchObjective::Arcs).into();
    let lens = LogAbstraction::over(&p.max.bounds, &p.max.cells);
    let (best, both) = match strategy.unwrap_or(SearchStrategy::Best) {
        SearchStrategy::Best => best_of_two_with_objective(&input, &lens, obj),
        SearchStrategy::Handoff => {
            let f = handoff_with_objective(&input, &lens, obj);
            (f.clone(), vec![f])
        }
        SearchStrategy::Greedy => {
            let f = greedy_with_objective(&input, &lens, obj);
            (f.clone(), vec![f])
        }
        SearchStrategy::Exact => {
            let f = exact_with_objective(&input, &lens, obj);
            (f.clone(), vec![f])
        }
    };
    let type_table = type_rows(&p, tau);
    let restated = restated_by(&p, &type_table);
    recommendation_of(ocel, &p, &best, &both, &restated)
}

/// Apply the computed assignment, without an editing round trip.
#[register_binding]
pub fn schema_reduction_reduce(
    ocel: &SlimLinkedOCEL,
    mode: Option<DepositMode>,
    theta: Option<f64>,
) -> SlimLinkedOCEL {
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let p = prepare(ocel, theta);
    let (best, _) = recommend(&p);
    apply_assignment(
        ocel,
        &p,
        &best.cells,
        &HashSet::new(),
        &HashSet::new(),
        mode.unwrap_or_default(),
    )
}

fn apply_assignment(
    ocel: &SlimLinkedOCEL,
    p: &Prepared,
    flow: &HashSet<Cell>,
    _held: &HashSet<Cell>,
    dropped: &HashSet<Cell>,
    mode: DepositMode,
) -> SlimLinkedOCEL {
    // A hand edit is free to drop a cell another type's own arcs still need: a search never
    // proposes that. Repairing here, once, before the assignment is read anywhere downstream,
    // is what keeps every deposit preserving instead of only the ones the search produced.
    let mut flow: HashSet<Cell> = flow.difference(dropped).copied().collect();
    let full_facts = facts_from(&p.bounds, &p.grid.cells, DEFAULT_NOISE_THRESHOLD);
    fact_repair(&p.grid, &full_facts, &mut flow);

    let a = assign(&p.grid, &flow, &HashSet::new());
    let expanded: HashSet<Cell> = a.expanded.iter().copied().collect();
    let added: Vec<_> = p
        .max
        .admitted_cells()
        .filter(|c| expanded.contains(&c.cell))
        .flat_map(|c| c.tuples.iter().copied())
        .collect();

    let tagged = tag(ocel, &p.schema, &p.acts, &flow, &added);
    match mode {
        DepositMode::Tagged => tagged,
        DepositMode::FlowProjection => flow_projection(&tagged).into_owned(),
    }
}

/// How a client wants an involved cell drawn. Mirrors
/// [`InvolvementRender`](crate::analysis::object_centric::schema_reduction::InvolvementRender)
/// across the binding boundary, which stays a plain Rust enum with no serde dependency of its
/// own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub enum InvolvementRenderArg {
    /// The type's own component, bracketed with the flow and absence branches into one
    /// shared token. Richer, and what the paper's own measurements use.
    #[default]
    Connected,
    /// One place per activity, read and written back by that activity's transition alone.
    SelfLoop,
    /// No place. The client reads involvement off the cell data it already has and draws its
    /// own marker.
    Hidden,
}

impl From<InvolvementRenderArg> for InvolvementRender {
    fn from(v: InvolvementRenderArg) -> Self {
        match v {
            InvolvementRenderArg::Connected => InvolvementRender::Connected,
            InvolvementRenderArg::SelfLoop => InvolvementRender::SelfLoop,
            InvolvementRenderArg::Hidden => InvolvementRender::Hidden,
        }
    }
}

/// How a client wants an implied cell drawn. Mirrors
/// [`AbsenceRender`](crate::analysis::object_centric::schema_reduction::AbsenceRender).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub enum AbsenceRenderArg {
    /// A mapped place: the carrier's own component, copied and mirrored into a place of the
    /// non-flow type. The only structural form Sec. 6 gives an implied cell.
    #[default]
    Mapped,
    /// No place. The client reads the determining map off the cell data it already has and
    /// draws a "flows with" rider on the carrier's own arc instead.
    Hidden,
}

impl From<AbsenceRenderArg> for AbsenceRender {
    fn from(v: AbsenceRenderArg) -> Self {
        match v {
            AbsenceRenderArg::Mapped => AbsenceRender::Mapped,
            AbsenceRenderArg::Hidden => AbsenceRender::Hidden,
        }
    }
}

/// Why a place in [`AnnotatedOcpn`] exists beyond the mined flow structure. Mirrors
/// [`PlaceRole`](crate::analysis::object_centric::schema_reduction::PlaceRole) across the
/// binding boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PlaceRoleArg {
    /// A self-loop place added for an involved cell.
    Involved,
    /// A place mirroring `carrier`'s own structure, added for an implied cell.
    Mapped { carrier: String },
}

impl From<PlaceRole> for PlaceRoleArg {
    fn from(v: PlaceRole) -> Self {
        match v {
            PlaceRole::Involved => PlaceRoleArg::Involved,
            PlaceRole::Mapped { carrier } => PlaceRoleArg::Mapped { carrier },
        }
    }
}

/// One place the reconstruction added, and why.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PlaceRoleRef {
    /// The place's name in [`AnnotatedOcpn::ocpn`].
    pub place: String,
    pub role: PlaceRoleArg,
}

/// The annotated net, plus why each place beyond ordinary flow structure is there.
///
/// A place not named in `place_roles` is one the flow-only discovery itself produced. The
/// net's own shape does not say the rest: a mapped place and a flow place read alike after
/// silent simplification collapses the mapped copy down, which is exactly why the paper warns
/// a self-loop can look like flow -- a mapped place can too, and a drawing rule that wants to
/// mark either has to be handed this rather than guess from topology.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AnnotatedOcpn {
    pub ocpn: ObjectCentricPetriNetJson,
    pub place_roles: Vec<PlaceRoleRef>,
}

/// Simplify every component and carry `roles` through the place merges that does -- series-tau
/// fusion is the one simplification rule that folds a place into another rather than dropping
/// it, and the survivor is not necessarily the one a caller's own annotation was keyed on.
fn simplify_and_convert(
    mut net: ObjectCentricPetriNet,
    mut roles: HashMap<uuid::Uuid, PlaceRole>,
) -> AnnotatedOcpn {
    for component in net.nets.values_mut() {
        for (gone, kept) in component.simplify_silent_tracked() {
            if let Some(role) = roles.remove(&gone) {
                roles.insert(kept, role);
            }
        }
    }
    let (ocpn, named) = net.to_json_form_named();
    let place_roles = named
        .into_iter()
        .filter_map(|(id, place)| {
            roles.remove(&id).map(|role| PlaceRoleRef { place, role: role.into() })
        })
        .collect();
    AnnotatedOcpn { ocpn, place_roles }
}

/// Mine the flow layer and draw the annotated net Sec. 6 specifies: involvement as a place
/// (in the form `involvement_render` chooses) or as no place at all, absence as the
/// determining type's own structure mirrored into a place of the non-flow type. This is not
/// the flow-only discovery alone -- that net has no component for a type nothing flows, and
/// comparing it to the recorded model per type would read ReFlow's claim (the
/// behaviour is still in the picture, carried by another type) as a total loss instead.
///
/// Inputs are exhaustive and match [`schema_reduction_evaluate`]/[`schema_reduction_apply`]:
/// the flow-only sublog this mines on, the assignment, and the schema. Nothing beyond that is
/// read, and no population-cluster or precedence-place enrichment is added regardless of
/// `involvement_render` -- the minimal construction is the default the paper reports, and a
/// browser client has no business depending on an env var of the process it happens to be
/// talking to.
#[register_binding]
pub fn schema_reduction_annotated_ocpn(
    ocel: &SlimLinkedOCEL,
    flow: Vec<CellRef>,
    involvement_render: Option<InvolvementRenderArg>,
    absence_render: Option<AbsenceRenderArg>,
    noise_threshold: Option<f64>,
    theta: Option<f64>,
) -> AnnotatedOcpn {
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let noise = noise_threshold.unwrap_or(0.2);
    let involvement_render: InvolvementRender = involvement_render.unwrap_or_default().into();
    let absence_render: AbsenceRender = absence_render.unwrap_or_default().into();
    let p = prepare(ocel, theta);

    let (mut flow_cells, _) = resolve_cells(&p, flow);
    let full_facts = facts_from(&p.bounds, &p.grid.cells, DEFAULT_NOISE_THRESHOLD);
    fact_repair(&p.grid, &full_facts, &mut flow_cells);

    // The assignment the reconstruction reads: flow as given, implied where the kept cells
    // determine it, involvement otherwise -- the two-reason license `assign` already applies,
    // with no cell forced by a caller-side drop.
    let a = assign(&p.grid, &flow_cells, &HashSet::new());

    // The mining input: the flow projection of the tagged log, materialised here rather than
    // round-tripped through the dataset store.
    let tagged = tag(ocel, &p.schema, &p.acts, &flow_cells, &[]);
    let flow_only = flow_projection(&tagged);

    let reduced = discover_ocpn(
        flow_only.as_ref(),
        ObjectCentricDiscoveryOptions::new(InductiveMinerOptions::imf(noise)),
    );

    let carriers = carriers_of(&p.grid, &p.max.bounds, &p.max.cells, &flow_cells);
    let (annotated, roles) = reconstruct_ocpn(
        &reduced,
        &a,
        &carriers,
        &HashMap::new(),
        &p.bounds,
        &p.schema.types,
        &p.grid.activities,
        involvement_render,
        absence_render,
    );
    // Dead `p -> tau -> q` chains, single-branch choice skeletons, and emptied husks are
    // construction verbosity, not size: strip them before a client ever sees them, the same
    // pass `ocpn_quality`'s own numbers are measured after.
    simplify_and_convert(annotated, roles)
}

/// Draw the annotated net over a net already mined, without mining again.
///
/// [`schema_reduction_annotated_ocpn`] mines and draws in one call, which is right for a
/// one-shot request but wrong for a client flipping between detail levels: the flow-only
/// sublog and its discovery do not depend on `involvement_render`/`absence_render` at all, so
/// re-running IM(f) on every flip re-pays a cost the edit did not ask for. Here `reduced` is
/// the net a prior discovery already returned (e.g. from `app_bindings::discover_oc_inductive_miner`
/// on the same flow-only deposit `flow` describes), and only the repair -- `fact_repair`,
/// the assignment, the carriers, `reconstruct_ocpn` -- runs again, all of it milliseconds
/// work at editor scale.
#[register_binding]
pub fn schema_reduction_repair_ocpn(
    ocel: &SlimLinkedOCEL,
    flow: Vec<CellRef>,
    reduced: ObjectCentricPetriNetJson,
    involvement_render: Option<InvolvementRenderArg>,
    absence_render: Option<AbsenceRenderArg>,
    theta: Option<f64>,
) -> AnnotatedOcpn {
    let theta = theta.unwrap_or(DEFAULT_THETA);
    let involvement_render: InvolvementRender = involvement_render.unwrap_or_default().into();
    let absence_render: AbsenceRender = absence_render.unwrap_or_default().into();
    let p = prepare(ocel, theta);

    let (mut flow_cells, _) = resolve_cells(&p, flow);
    let full_facts = facts_from(&p.bounds, &p.grid.cells, DEFAULT_NOISE_THRESHOLD);
    fact_repair(&p.grid, &full_facts, &mut flow_cells);

    let a = assign(&p.grid, &flow_cells, &HashSet::new());
    let reduced = ObjectCentricPetriNet::from_json_form(&reduced);
    let carriers = carriers_of(&p.grid, &p.max.bounds, &p.max.cells, &flow_cells);
    let (annotated, roles) = reconstruct_ocpn(
        &reduced,
        &a,
        &carriers,
        &HashMap::new(),
        &p.bounds,
        &p.schema.types,
        &p.grid.activities,
        involvement_render,
        absence_render,
    );
    // Dead `p -> tau -> q` chains, single-branch choice skeletons, and emptied husks are
    // construction verbosity, not size: strip them before a client ever sees them, the same
    // pass `ocpn_quality`'s own numbers are measured after.
    simplify_and_convert(annotated, roles)
}

/// Objects of one type per event of one activity, over the events that carry any.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub struct MarkedCellCounts {
    pub min: usize,
    pub mean: f64,
    pub max: usize,
}

/// Where a cell of a tagged log sits, read off the tags alone: see the binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MarkedCellState {
    Flow,
    Involvement,
}

/// One cell of a tagged log, as a model viewer takes it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MarkedCell {
    pub activity: String,
    pub object_type: String,
    /// Never `implied`: telling implied from involved needs the object schema.
    pub state: MarkedCellState,
    /// What an involvement badge shows. The count is what separates a supervisor, one per
    /// event, from a batch.
    pub counts: MarkedCellCounts,
}

/// The cell states a log carries in its own qualifiers.
///
/// A model viewer needs the states, and this reads them off the tags rather than out of an
/// assignment held in the session. A tagged log imported from disk carries [`NOT_FLOWING`]
/// on every participation that does not flow, so whoever opens that file sees the picture the
/// analyst who wrote it saw, with no session state and no re-running of the reduction, which
/// would be minutes on a large log and is not a thing a viewer should do in order to draw.
///
/// Deliberately independent of [`StructuralSchema`]: schema discovery is the expensive half
/// and none of it is needed to read a prefix. One pass over the events, everything keyed by
/// name.
///
/// **`implied` cells are not emitted.** Telling an implied cell from an involved one needs
/// the object schema, which is what [`annotation`](crate::analysis::object_centric::schema_reduction::annotation)
/// is for; a non-flow cell is reported as `involvement` here, which is the safe rendering.
///
/// A cell counts as flowing when *any* participation it carries is untagged, matching
/// [`non_flow_cells`](crate::analysis::object_centric::schema_reduction::non_flow_cells): a
/// half-marked cell draws arcs, so calling it involvement would understate what the model
/// says.
#[register_binding]
pub fn schema_reduction_marked_cells(ocel: &SlimLinkedOCEL) -> Vec<MarkedCell> {
    struct Tally {
        min: usize,
        max: usize,
        sum: usize,
        events: usize,
        flows: bool,
    }
    // Keyed by borrowed names and allocated once at the end. Keying by `String` cost two
    // allocations per (event, object type), which on a 1.2M-event log is the whole runtime of
    // a call a viewer makes every time it opens.
    let mut tally: HashMap<(&str, &str), Tally> = HashMap::new();
    let activities: Vec<String> = ocel.get_ev_types().map(ToString::to_string).collect();
    // Reused across events rather than rebuilt: one allocation, not one per event.
    let mut here: HashMap<&str, (HashSet<&ObjectIndex>, bool)> = HashMap::new();
    for act in &activities {
        for ev in ocel.get_evs_of_type(act) {
            // Per event: the distinct objects of each type, and whether any of that type's
            // participations here is unmarked. A pair can appear under several qualifiers,
            // so objects are counted distinctly and marks are read per relationship.
            here.clear();
            for (q, o) in ev.get_e2o_q(ocel) {
                let entry = here.entry(o.get_ob_type(ocel).as_str()).or_default();
                entry.0.insert(o);
                entry.1 |= flows(q);
            }
            for (ot, (objs, flowing)) in &here {
                let t = tally.entry((act.as_str(), ot)).or_insert(Tally {
                    min: usize::MAX,
                    max: 0,
                    sum: 0,
                    events: 0,
                    flows: false,
                });
                t.min = t.min.min(objs.len());
                t.max = t.max.max(objs.len());
                t.sum += objs.len();
                t.events += 1;
                t.flows |= *flowing;
            }
        }
    }
    let mut out: Vec<MarkedCell> = tally
        .into_iter()
        .map(|((activity, object_type), t)| MarkedCell {
            activity: activity.to_string(),
            object_type: object_type.to_string(),
            state: if t.flows {
                MarkedCellState::Flow
            } else {
                MarkedCellState::Involvement
            },
            counts: MarkedCellCounts {
                min: t.min,
                mean: t.sum as f64 / t.events as f64,
                max: t.max,
            },
        })
        .collect();
    // Sorted, so a client that renders the list in order gets the same picture every run.
    out.sort_by(|a, b| {
        (&a.activity, &a.object_type).cmp(&(&b.activity, &b.object_type))
    });
    out
}

#[cfg(test)]
mod marked_cell_tests {
    use super::*;
    use crate::{Importable, OCEL};

    fn om() -> SlimLinkedOCEL {
        SlimLinkedOCEL::from_ocel(
            OCEL::import_from_path("test_data/ocel/order-management.xml").expect("import"),
        )
    }

    #[test]
    fn an_unmarked_log_is_every_cell_at_flow() {
        let cells = schema_reduction_marked_cells(&om());
        assert_eq!(cells.len(), 40, "Order Management records 40 cells");
        assert!(cells.iter().all(|c| c.state == MarkedCellState::Flow));
        assert!(cells.iter().all(|c| c.counts.min >= 1 && c.counts.max >= c.counts.min));
    }

    #[test]
    fn marking_a_cell_moves_it_to_involvement_and_leaves_its_counts() {
        let ocel = om();
        let before = schema_reduction_marked_cells(&ocel);
        let reduced = schema_reduction_reduce(&ocel, Some(DepositMode::Tagged), None);
        let after = schema_reduction_marked_cells(&reduced);

        let involved: Vec<&MarkedCell> = after.iter().filter(|c| c.state == MarkedCellState::Involvement).collect();
        assert!(!involved.is_empty(), "the recommendation holds cells at involvement");
        // Marking changes a qualifier and no participation, so the badge a viewer draws is
        // the same count it would have drawn before the reduction.
        for c in &involved {
            let was = before
                .iter()
                .find(|b| b.activity == c.activity && b.object_type == c.object_type)
                .expect("an involvement cell was recorded");
            assert_eq!((was.counts.min, was.counts.max), (c.counts.min, c.counts.max));
        }
    }
}

#[cfg(test)]
mod annotated_ocpn_tests {
    use super::*;
    use crate::{Importable, OCEL};

    fn om() -> SlimLinkedOCEL {
        SlimLinkedOCEL::from_ocel(
            OCEL::import_from_path("test_data/ocel/order-management.xml").expect("import"),
        )
    }

    /// Exactly the call chain propel's cell editor makes for the "Repaired" preset: the
    /// recommended search's own flow cells, involvement as a self-loop place, absence as a
    /// mapped place. Every type Order Management records is either flowing or has a
    /// determined-absence carrier at every activity it is dropped from, so every one of the
    /// six should end up with a nonempty component -- customers and products included, the
    /// two the paper's table reports at 4 and 22 repaired arcs.
    #[test]
    fn every_type_gets_a_repaired_component() {
        let ocel = om();
        let recommendation = schema_reduction_search(&ocel, Some(SearchStrategy::Best), None, None, None);
        let flow = recommendation.flow.clone();
        let ocpn = schema_reduction_annotated_ocpn(
            &ocel,
            flow,
            Some(InvolvementRenderArg::SelfLoop),
            Some(AbsenceRenderArg::Mapped),
            None,
            None,
        );
        let types_with_places: std::collections::HashSet<&str> =
            ocpn.ocpn.places.iter().map(|p| p.object_type.as_str()).collect();
        for ot in ["customers", "employees", "items", "orders", "packages", "products"] {
            assert!(
                types_with_places.contains(ot),
                "{ot} has no place in the repaired net; recommendation flow = {:?}",
                recommendation.flow
            );
        }
    }
}
