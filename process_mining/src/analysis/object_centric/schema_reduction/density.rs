use std::collections::HashSet;

use super::{
    cells::{ActivityIndex, Cell},
    contribution::marginal_contribution,
    facts::Facts,
    novelty::Pair,
    schema::ObjectTypeIndex,
};

/// The state a type's own behaviour asks for, before any map is consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeState {
    /// Its arcs carry ordering, so it keeps drawing them.
    Flow,
    /// It orders nothing but says who never meets whom, so it stops drawing arcs and its
    /// cells are involved: nothing recomputes the participations.
    Involvement,
    /// It asserts neither kind of fact, and a route recomputes it, so its cells are implied
    /// and a reader recovers them from the relation.
    ///
    /// A type whose activity pairs are all ties is not "asserts neither": its clock cannot
    /// answer the question, and it must not be taken out of the flow layer on that basis.
    Implied,
}

impl TypeState {
    /// The name used in the census and in the paper.
    pub fn label(&self) -> &'static str {
        match self {
            TypeState::Flow => "flow",
            TypeState::Involvement => "involvement",
            TypeState::Implied => "implied",
        }
    }
}

/// The two ordering densities of one object type, and the state the rule gives it.
///
/// `density_recorded` and `density_max` are different numbers and disagree in general.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypeDensity {
    /// The type.
    pub object_type: ObjectTypeIndex,
    /// Activities the type is recorded at.
    pub activities: usize,
    /// Covering ordering pairs over the recorded log.
    pub ordering: usize,
    /// Never-together pairs over the recorded log.
    pub role: usize,
    /// Activity pairs the log cannot order: the type's objects take part in both, at one
    /// instant. Neither an ordering nor a role fact.
    pub tied: usize,
    /// `ordering / C(activities, 2)`. Diagnostic only.
    pub density_recorded: f64,
    /// Strict-ordering closure pairs over the recorded log: what the type orders.
    pub asserted: usize,
    /// Of those, the pairs no other surviving type orders. Zero means the type only
    /// restates, so it stops drawing arcs.
    pub unique: usize,
    /// A surviving type that orders one of its pairs, when this type does not survive.
    pub covered_by: Option<ObjectTypeIndex>,
    /// Activities the type would be at once every determined cell is written.
    pub activities_max: usize,
    /// Strict-ordering closure pairs over that saturated log. Closure, not covering, because
    /// covering is not monotone under adding tuples.
    pub asserted_max: usize,
    /// `asserted_max / C(activities_max, 2)`. Diagnostic only.
    pub density_max: f64,
    /// What the type's own behaviour asks for, from marginal contribution, `role` and
    /// `tied`.
    pub wants: TypeState,
    /// Some map lands in this type: applying it recomputes the type's participations.
    pub determined: bool,
    /// Some map is defined on this type: expanding its fibre recovers it.
    pub determines: bool,
    /// The state after the map check.
    pub state: TypeState,
}

/// Ordering facts as a fraction of the activity pairs that could carry one.
fn density(facts: usize, activities: usize) -> f64 {
    let pairs = activities * activities.saturating_sub(1) / 2;
    if pairs == 0 {
        0.0
    } else {
        facts as f64 / pairs as f64
    }
}

/// The three-way state each object type lands in, with both densities beside it.
///
/// `recorded` and `max_facts` are both taken at `tau = 0`: the strict test, where a single
/// reversed object makes a pair parallel.
///
/// The rule is marginal contribution and it is threshold-free. A type keeps drawing arcs
/// while some activity pair it orders no other surviving type orders; otherwise a type with
/// role facts or with a tied pair is involvement; otherwise implied. Both densities are
/// computed for the table, but nothing reads them.
///
/// A tie holds the type at involvement. Implied claims a route recomputes the cell, and a
/// tie is the log declining to answer, which is not that.
///
/// Implied means a route puts the participations back, so a type no route
/// relates in either direction stays recorded as involvement. This is a necessary condition
/// only: determinacy is still tested per cell.
///
/// `map_pairs` is `(source, target)` per discovered map, which is
/// [`StructuralSchema::pairs`](super::StructuralSchema::pairs).
#[allow(clippy::too_many_arguments)]
pub fn type_densities(
    n_types: usize,
    recorded_cells: &HashSet<Cell>,
    recorded: &Facts,
    max_cells: &HashSet<Cell>,
    max_facts: &Facts,
    map_pairs: &HashSet<(ObjectTypeIndex, ObjectTypeIndex)>,
    rep: &[ObjectTypeIndex],
    types: &[String],
) -> Vec<TypeDensity> {
    let acts_of = |cells: &HashSet<Cell>, t: ObjectTypeIndex| -> usize {
        cells
            .iter()
            .filter(|(_, ty)| *ty == t)
            .map(|(a, _)| *a)
            .collect::<HashSet<ActivityIndex>>()
            .len()
    };
    let count = |set: &HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)>,
                 t: ObjectTypeIndex| set.iter().filter(|(ty, _, _)| *ty == t).count();

    // Over the recorded log: the classifier needs no saturation.
    let mut per_type: Vec<HashSet<Pair>> = vec![HashSet::new(); n_types];
    for (t, x, y) in &recorded.asserted {
        if *t < n_types {
            per_type[*t].insert((*x, *y));
        }
    }
    let contrib = marginal_contribution(&per_type, rep, types);

    (0..n_types)
        .map(|t| {
            let activities = acts_of(recorded_cells, t);
            let ordering = count(&recorded.ordering, t);
            let role = count(&recorded.role, t);
            let tied = count(&recorded.tied, t);
            let density_recorded = density(ordering, activities);
            let activities_max = acts_of(max_cells, t);
            let asserted_max = count(&max_facts.asserted, t);
            let wants = if contrib[t].drawn {
                TypeState::Flow
            } else if role > 0 || tied > 0 {
                TypeState::Involvement
            } else {
                TypeState::Implied
            };
            let determined = map_pairs.iter().any(|(_, target)| *target == t);
            let determines = map_pairs.iter().any(|(source, _)| *source == t);
            let state = if wants == TypeState::Implied && !determined && !determines {
                TypeState::Involvement
            } else {
                wants
            };
            TypeDensity {
                object_type: t,
                activities,
                ordering,
                role,
                tied,
                density_recorded,
                asserted: contrib[t].orders,
                unique: contrib[t].unique,
                covered_by: contrib[t].covered_by,
                activities_max,
                asserted_max,
                density_max: density(asserted_max, activities_max),
                wants,
                determined,
                determines,
                state,
            }
        })
        .collect()
}

/// How many types land in each state.
pub fn state_tally(rows: &[TypeDensity]) -> (usize, usize, usize) {
    let n = |s: TypeState| rows.iter().filter(|r| r.state == s).count();
    (
        n(TypeState::Flow),
        n(TypeState::Involvement),
        n(TypeState::Implied),
    )
}
