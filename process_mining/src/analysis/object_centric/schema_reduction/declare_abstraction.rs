//! The OC-DECLARE behavioural instantiation: whether a type's discovered constraints order an
//! activity pair, read off the crate's object-centric DECLARE miner
//! ([`discovery::object_centric::oc_declare`](crate::discovery::object_centric::oc_declare)).
//!
//! Discovery runs once per type over that type's own participations, restricted through
//! [`OCDeclareDiscoveryOptions::types_to_use`] and [`OCDeclareDiscoveryOptions::acts_to_use`].
//! It is not re-run per keep-set.
//!
//! Ordering comes from [`OCDeclareArcType::EF`] (response) and [`OCDeclareArcType::EP`]
//! (precedence); both readings of one pair collapse onto the same ordered pair `(a, b)`.
//! Coexistence comes from [`OCDeclareArcType::AS`] (association), the second assertion kind
//! this instantiation supplies. The chain variants `DF`/`DP` are excluded through
//! [`OCDeclareDiscoveryOptions::considered_arrow_types`]: a chain arc is reported only where its
//! base arc also holds, so dropping them loses no ordered pair.
//!
//! Thresholds are [`DECLARE_NOISE_THRESHOLD`], [`DECLARE_COUNTS_FOR_GENERATION`] and
//! [`DECLARE_COUNTS_FOR_FILTER`]. Object-to-object involvement is off ([`O2OMode::None`]), since
//! each run sees one type.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;
use crate::core::process_models::oc_declare::OCDeclareArcType;
use crate::discovery::object_centric::oc_declare::{
    discover_behavior_constraints, O2OMode, OCDeclareDiscoveryOptions, OCDeclareReductionMode,
};

use super::{
    abstraction::{requirement, Abstraction, Assertion, Cover},
    arcs::ActivityIndexing,
    cells::{ActivityIndex, Cell},
    coverage::{Chained, Coverage},
    novelty::Pair,
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// OC-DECLARE noise threshold this instantiation mines at: the crate's own
/// `OCDeclareDiscoveryOptions::default().noise_threshold`.
pub const DECLARE_NOISE_THRESHOLD: f64 = 0.2;
/// Witnessing counts per binding at the candidate-generation step, the crate's own default.
pub const DECLARE_COUNTS_FOR_GENERATION: (Option<usize>, Option<usize>) = (Some(1), None);
/// Witnessing counts per binding at the final filtering step. The max caps object
/// multiplicity per binding.
pub const DECLARE_COUNTS_FOR_FILTER: (Option<usize>, Option<usize>) = (Some(1), Some(20));

/// The arc types this instantiation mines.
///
/// `EF`/`EP` are the orderings, `AS` is a coexistence. The chain variants `DF`/`DP` are left
/// out: a chain arc is only reported where its base arc also holds, so dropping them changes
/// no verdict.
pub const DECLARE_ARC_TYPES: [OCDeclareArcType; 3] =
    [OCDeclareArcType::AS, OCDeclareArcType::EF, OCDeclareArcType::EP];

/// The OC-DECLARE analysis of one object type's per-type sub-log.
#[derive(Debug, Clone, Default)]
pub struct DeclareTypeModel {
    /// The type this sub-log was mined for.
    pub object_type: ObjectTypeIndex,
    /// Activities the type's participations record, i.e. its cells.
    pub activities: BTreeSet<ActivityIndex>,
    /// `(a, b)`: a response-family arc (`EF` or `DF`, `a -> b`) orders `a` before `b`.
    pub response: HashSet<Pair>,
    /// `(a, b)`: a precedence-family arc (`EP` or `DP`, `b -> a`) orders `a` before `b`.
    pub precedence: HashSet<Pair>,
    /// `response` union `precedence`: every pair this type's discovery orders.
    pub ordered: HashSet<Pair>,
    /// `(a, b)` with `a < b`: an `AS` arc co-occurs the two activities for this type without
    /// ordering them. Carried as [`Assertion::Together`](super::Assertion::Together).
    pub coexistence: HashSet<Pair>,
    /// The discovered arc type(s) evidencing each ordered pair, sorted and deduplicated, e.g.
    /// `[EF]` or `[EP, DP]`.
    pub templates: HashMap<Pair, Vec<OCDeclareArcType>>,
    /// Wall-clock time to build the per-type sub-log and run discovery on it.
    pub discovery_seconds: f64,
}

impl DeclareTypeModel {
    /// Activities the type's participations record but no discovered constraint involves.
    ///
    /// A coexistence counts: an activity only an `AS` arc touches is not neutral.
    pub fn neutral(&self) -> BTreeSet<ActivityIndex> {
        let constrained: BTreeSet<ActivityIndex> = self
            .ordered
            .iter()
            .chain(&self.coexistence)
            .flat_map(|&(a, b)| [a, b])
            .collect();
        self.activities.difference(&constrained).copied().collect()
    }

    /// Every assertion this type makes over `over`, both kinds, endpoint-filtered.
    pub fn assertions_over(&self, over: &HashSet<Cell>) -> Vec<Assertion> {
        let kept = |a: ActivityIndex| over.contains(&(a, self.object_type));
        let orders = self
            .ordered
            .iter()
            .filter(|&&(x, y)| kept(x) && kept(y))
            .map(|&(x, y)| Assertion::Order(x, y));
        let together = self
            .coexistence
            .iter()
            .filter(|&&(x, y)| kept(x) && kept(y))
            .map(|&(x, y)| Assertion::Together(x, y));
        orders.chain(together).collect()
    }

    /// The arc type(s) evidencing one ordered pair, as label strings, or `None` if the pair is
    /// not ordered.
    pub fn pair_templates(&self, x: ActivityIndex, y: ActivityIndex) -> Option<Vec<&'static str>> {
        self.templates
            .get(&(x, y))
            .map(|ts| ts.iter().map(|t| t.get_name()).collect())
    }
}

/// Mines every object type's per-type sub-log with the crate's OC-DECLARE discovery and reads its
/// binary positive ordering off the surviving `EF`/`EP`/`DF`/`DP` arcs.
///
/// `recorded` gives each type's activities, i.e. the cells it is analysed over. Types with no
/// recorded cell are skipped, and discovery runs once per type, not once per keep-set.
pub fn declare_abstraction(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    recorded: &HashSet<Cell>,
) -> Vec<DeclareTypeModel> {
    let mut out = Vec::new();
    for t in 0..schema.types.len() {
        let activities: BTreeSet<ActivityIndex> = recorded
            .iter()
            .filter(|(_, tt)| *tt == t)
            .map(|(a, _)| *a)
            .collect();
        if activities.is_empty() {
            continue;
        }
        out.push(declare_type_model(locel, schema, acts, t, activities));
    }
    out
}

/// One type's [`DeclareTypeModel`], timed over sub-log construction and discovery.
fn declare_type_model(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    t: ObjectTypeIndex,
    activities: BTreeSet<ActivityIndex>,
) -> DeclareTypeModel {
    let start = Instant::now();
    let arcs = if activities.is_empty() {
        // A recorded cell without a witnessing event should not occur, but an empty restriction
        // would divide by zero inside the crate's own violation-fraction computation.
        Vec::new()
    } else {
        let options = OCDeclareDiscoveryOptions {
            noise_threshold: DECLARE_NOISE_THRESHOLD,
            o2o_mode: O2OMode::None,
            acts_to_use: Some(
                activities
                    .iter()
                    .map(|&a| acts.activities[a].clone())
                    .collect(),
            ),
            types_to_use: Some([schema.types[t].clone()].into_iter().collect()),
            counts_for_generation: DECLARE_COUNTS_FOR_GENERATION,
            counts_for_filter: DECLARE_COUNTS_FOR_FILTER,
            reduction: OCDeclareReductionMode::None,
            refinement: false,
            considered_arrow_types: DECLARE_ARC_TYPES.into_iter().collect(),
        };
        discover_behavior_constraints(locel, options)
    };
    let discovery_seconds = start.elapsed().as_secs_f64();

    let name_index: HashMap<&str, ActivityIndex> = activities
        .iter()
        .map(|&a| (acts.activities[a].as_str(), a))
        .collect();

    let mut response = HashSet::new();
    let mut precedence = HashSet::new();
    let mut coexistence: HashSet<Pair> = HashSet::new();
    let mut templates: HashMap<Pair, Vec<OCDeclareArcType>> = HashMap::new();
    for arc in &arcs {
        let from = name_index[arc.from.as_str()];
        let to = name_index[arc.to.as_str()];
        if from == to {
            // Binary ordering needs two distinct activities; a self-loop asserts nothing here.
            continue;
        }
        let (pair, is_response) = match arc.arc_type {
            OCDeclareArcType::EF | OCDeclareArcType::DF => ((from, to), true),
            OCDeclareArcType::EP | OCDeclareArcType::DP => ((to, from), false),
            OCDeclareArcType::AS => {
                coexistence.insert(if from < to { (from, to) } else { (to, from) });
                continue;
            }
        };
        if is_response {
            response.insert(pair);
        } else {
            precedence.insert(pair);
        }
        templates.entry(pair).or_default().push(arc.arc_type);
    }
    for v in templates.values_mut() {
        v.sort();
        v.dedup();
    }
    let ordered: HashSet<Pair> = response.union(&precedence).copied().collect();

    DeclareTypeModel {
        object_type: t,
        activities,
        response,
        precedence,
        ordered,
        coexistence,
        templates,
        discovery_seconds,
    }
}

/// Ordered pairs a keep-set's OC-DECLARE ordering draws, split by family: response-sourced
/// (`EF`/`DF`) and precedence-sourced (`EP`/`DP`).
///
/// Mirrors [`coverage::drawn`](super::coverage::drawn): a type draws a pair only when it is kept
/// at both endpoints. The filter is applied after the keep-set-independent discovery in
/// [`DeclareTypeModel`], since OC-DECLARE discovery is not re-run per keep-set.
pub fn declare_drawn(
    models: &[DeclareTypeModel],
    kept: &HashSet<Cell>,
) -> (HashSet<Pair>, HashSet<Pair>) {
    let mut response = HashSet::new();
    let mut precedence = HashSet::new();
    for m in models {
        let endpoint_kept = |a: ActivityIndex| kept.contains(&(a, m.object_type));
        response.extend(
            m.response
                .iter()
                .copied()
                .filter(|&(x, y)| endpoint_kept(x) && endpoint_kept(y)),
        );
        precedence.extend(
            m.precedence
                .iter()
                .copied()
                .filter(|&(x, y)| endpoint_kept(x) && endpoint_kept(y)),
        );
    }
    (response, precedence)
}

/// Both coverage measures of one keep-set's OC-DECLARE ordering, in the shape of
/// [`coverage::Coverage`](super::coverage::Coverage).
///
/// Response and precedence closures are composed separately. A target pair counts as covered
/// once either family's closure reaches it: chain already implies base at discovery time, and
/// succession is both families agreeing on the same pair.
///
/// The searches use the same rule through [`DeclareAbstraction`].
pub fn declare_coverage(
    models: &[DeclareTypeModel],
    kept: &HashSet<Cell>,
    target: &[Pair],
    n_activities: usize,
) -> Coverage {
    let (response, precedence) = declare_drawn(models, kept);
    let drawn_pairs: HashSet<Pair> = response.union(&precedence).copied().collect();
    let response_chain = Chained::of(&response, n_activities);
    let precedence_chain = Chained::of(&precedence, n_activities);
    Coverage {
        drawn: target.iter().filter(|p| drawn_pairs.contains(p)).count(),
        chained: target
            .iter()
            .filter(|&&(x, y)| response_chain.holds(x, y) || precedence_chain.holds(x, y))
            .count(),
        target: target.len(),
    }
}

/// OC-DECLARE as a behavioural abstraction: its own assertions, and its own covering rule.
///
/// It asserts a second kind, [`Assertion::Together`] from the `AS` arcs, and its orderings
/// close per family (response with response, precedence with precedence), the rule of
/// [`declare_coverage`].
///
/// Discovery is not re-run per keep-set, so the endpoint filter is applied afterwards, on the
/// models this borrows.
pub struct DeclareAbstraction<'a> {
    models: &'a [DeclareTypeModel],
    by_type: HashMap<ObjectTypeIndex, &'a DeclareTypeModel>,
    required: Vec<Assertion>,
    reported: Vec<Assertion>,
}

impl<'a> DeclareAbstraction<'a> {
    /// Orderings block a cell from leaving the flow layer, coexistence is reported as
    /// residual. The default;
    /// [`strict`](Self::strict) blocks on coexistence too.
    pub fn over(models: &'a [DeclareTypeModel], over: &HashSet<Cell>) -> Self {
        Self::split(models, over, false)
    }

    /// Every assertion blocks, coexistence included.
    pub fn strict(models: &'a [DeclareTypeModel], over: &HashSet<Cell>) -> Self {
        Self::split(models, over, true)
    }

    fn split(models: &'a [DeclareTypeModel], over: &HashSet<Cell>, strict: bool) -> Self {
        let all: Vec<Assertion> = models.iter().flat_map(|m| m.assertions_over(over)).collect();
        let (orders, together): (Vec<_>, Vec<_>) = all
            .into_iter()
            .partition(|s| matches!(s, Assertion::Order(..)));
        let (required, reported) = if strict {
            (requirement(orders.into_iter().chain(together)), Vec::new())
        } else {
            (requirement(orders), requirement(together))
        };
        Self {
            models,
            by_type: models.iter().map(|m| (m.object_type, m)).collect(),
            required,
            reported,
        }
    }
}

impl Abstraction for DeclareAbstraction<'_> {
    fn asserts(&self, kept: &HashSet<Cell>, t: ObjectTypeIndex) -> HashSet<Assertion> {
        self.by_type
            .get(&t)
            .map(|m| m.assertions_over(kept).into_iter().collect())
            .unwrap_or_default()
    }

    fn required(&self) -> &[Assertion] {
        &self.required
    }

    fn reported(&self) -> &[Assertion] {
        &self.reported
    }

    fn cover(
        &self,
        kept: &HashSet<Cell>,
        asserted: &HashSet<Assertion>,
        n_activities: usize,
    ) -> Box<dyn Cover> {
        let (response, precedence) = declare_drawn(self.models, kept);
        Box::new(DeclareCover {
            response: Chained::of(&response, n_activities),
            precedence: Chained::of(&precedence, n_activities),
            together: asserted
                .iter()
                .filter(|s| matches!(s, Assertion::Together(..)))
                .copied()
                .collect(),
        })
    }
}

/// [`declare_coverage`]'s rule as a [`Cover`], plus the coexistence kind.
///
/// A coexistence is covered only where a kept type shows it. Chaining cannot deliver one, and
/// the map rule that would carry it from a fine type to a coarse one needs the schema, which
/// this abstraction does not hold. Reporting it uncovered keeps a cell flowing instead of
/// dropping an assertion no model would then show.
struct DeclareCover {
    response: Chained,
    precedence: Chained,
    together: HashSet<Assertion>,
}

impl Cover for DeclareCover {
    fn holds(&self, s: Assertion) -> bool {
        match s {
            Assertion::Order(x, y) => self.response.holds(x, y) || self.precedence.holds(x, y),
            Assertion::Together(..) => self.together.contains(&s),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::{CellGrid, SchemaClosure};
    use crate::core::event_data::object_centric::appendable::AppendableOCEL;
    use crate::core::event_data::object_centric::{OCELRelationship as Rel, OCELType};
    use chrono::DateTime;

    fn empty_type(name: &str) -> OCELType {
        OCELType {
            name: name.into(),
            attributes: Vec::new(),
        }
    }

    /// One object type, `order`, whose two objects always `place` strictly before they `ship`:
    /// a clean response/chain-response pattern for `declare_abstraction` to find end to end.
    fn ordered_locel() -> SlimLinkedOCEL {
        let mut s = SlimLinkedOCEL::new();
        for t in ["place", "ship"] {
            s.declare_event_type(empty_type(t)).unwrap();
        }
        s.declare_object_type(empty_type("order")).unwrap();
        for id in ["o1", "o2"] {
            s.append_object(id.into(), "order", Vec::new(), Vec::new())
                .unwrap();
        }
        let evs = [
            ("e0", "place", "o1", 0),
            ("e1", "place", "o2", 1),
            ("e2", "ship", "o1", 2),
            ("e3", "ship", "o2", 3),
        ];
        for (id, et, ob, day) in evs {
            let time =
                DateTime::parse_from_rfc3339(&format!("2024-01-0{}T00:00:00Z", day + 1)).unwrap();
            s.append_event(
                id.into(),
                et,
                time,
                Vec::new(),
                vec![Rel::new(ob, "q")],
            )
            .unwrap();
        }
        s.finalize().unwrap();
        s
    }

    #[test]
    fn discovers_a_clean_response_and_marks_no_activity_neutral() {
        let locel = ordered_locel();
        let schema = StructuralSchema::discover(&locel);
        let closure = SchemaClosure::build(&locel, &schema);
        let grid = CellGrid::build(&locel, &schema, &closure);
        let acts = ActivityIndexing::build(&locel, &grid);

        let models = declare_abstraction(&locel, &schema, &acts, &grid.cells);
        assert_eq!(models.len(), 1, "exactly one recorded type: order");
        let model = &models[0];

        let place = acts.activities.iter().position(|a| a == "place").unwrap();
        let ship = acts.activities.iter().position(|a| a == "ship").unwrap();

        assert!(model.ordered.contains(&(place, ship)));
        assert!(!model.ordered.contains(&(ship, place)));
        assert!(model.neutral().is_empty());
        let labels = model.pair_templates(place, ship).unwrap();
        assert!(labels.contains(&"EF") || labels.contains(&"DF"));
    }

    fn model(object_type: ObjectTypeIndex, response: &[Pair], precedence: &[Pair]) -> DeclareTypeModel {
        let response: HashSet<Pair> = response.iter().copied().collect();
        let precedence: HashSet<Pair> = precedence.iter().copied().collect();
        let ordered = response.union(&precedence).copied().collect();
        DeclareTypeModel {
            object_type,
            activities: response
                .iter()
                .chain(&precedence)
                .flat_map(|&(a, b)| [a, b])
                .collect(),
            response,
            precedence,
            ordered,
            coexistence: HashSet::new(),
            templates: HashMap::new(),
            discovery_seconds: 0.0,
        }
    }

    #[test]
    fn response_chains_compose_but_do_not_cross_into_precedence() {
        // Type 0 draws response 0<1, type 1 draws response 1<2: chained response covers 0<2.
        let models = vec![model(0, &[(0, 1)], &[]), model(1, &[(1, 2)], &[])];
        let kept: HashSet<Cell> = [(0, 0), (1, 0), (1, 1), (2, 1)].into_iter().collect();
        let cov = declare_coverage(&models, &kept, &[(0, 1), (1, 2), (0, 2)], 3);
        assert_eq!(cov.drawn, 2, "0<2 is not drawn directly by either type");
        assert_eq!(cov.chained, 3, "the response chain composes 0<1 and 1<2 into 0<2");
    }

    #[test]
    fn a_response_link_and_a_precedence_link_do_not_compose_across_families() {
        // Type 0 draws response 0<1 only, type 1 draws precedence 1<2 only (no response
        // counterpart): 0<2 must not be claimed covered by mixing the two closures.
        let models = vec![model(0, &[(0, 1)], &[]), model(1, &[], &[(1, 2)])];
        let kept: HashSet<Cell> = [(0, 0), (1, 0), (1, 1), (2, 1)].into_iter().collect();
        let cov = declare_coverage(&models, &kept, &[(0, 2)], 3);
        assert_eq!(cov.drawn, 0);
        assert_eq!(cov.chained, 0, "response and precedence chains stay separate");
    }

    #[test]
    fn endpoints_must_both_be_kept_for_a_pair_to_be_drawn() {
        let models = vec![model(0, &[(0, 1)], &[])];
        let kept: HashSet<Cell> = [(0, 0)].into_iter().collect(); // activity 1 not kept for type 0
        let (response, precedence) = declare_drawn(&models, &kept);
        assert!(response.is_empty());
        assert!(precedence.is_empty());
    }
}
