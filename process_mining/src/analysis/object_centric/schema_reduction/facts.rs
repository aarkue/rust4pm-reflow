use std::collections::{HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;

use super::{
    arcs::ActivityIndexing,
    bounds::Bounds,
    cells::{ActivityIndex, Cell, CellGrid},
    schema::{ObjectTypeIndex, StructuralSchema},
    simultaneity::precedes,
};

/// Default noise threshold: none.
///
/// At zero, a single object ordered the other way makes the pair parallel. The ordering
/// density read by the state rule is compared against a fixed 0.05 on both sides, so both
/// have to be measured the same way or the threshold means two different things.
pub const DEFAULT_NOISE_THRESHOLD: f64 = 0.0;

/// What a model asserts, as opposed to how big it is.
///
/// Control flow lives in the object types, role structure in the resource types.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    /// `(type, from, to)`: the type's eventually-follows relation, reduced to its covering
    /// pairs, so an ordering implied by two others is not counted twice.
    pub ordering: HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)>,
    /// `(type, a, b)`: no object of the type ever takes part in both activities. This is
    /// what a resource type carries.
    pub role: HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)>,
    /// Every strict ordering the model asserts, before the covering reduction.
    ///
    /// Only this set is monotone in the participations. `ordering` can shrink when
    /// participations are added, because one new edge can make old ones implied.
    pub asserted: HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)>,
    /// `(type, a, b)` with `a < b`: objects of the type take part in both, and every one of
    /// them does so at a single instant, so the log resolves no order either way.
    ///
    /// A tie is in neither `ordering` nor `role`, so without this set it is
    /// indistinguishable from concurrency.
    pub tied: HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)>,
}

impl Facts {
    /// Total number of facts. A tie is not counted.
    pub fn len(&self) -> usize {
        self.ordering.len() + self.role.len()
    }

    /// Whether the model says nothing at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The cells a fact needs drawn: its type at both endpoints.
    pub fn required_cells(&self) -> HashSet<Cell> {
        self.ordering
            .iter()
            .chain(self.role.iter())
            .flat_map(|(t, x, y)| [(*x, *t), (*y, *t)])
            .collect()
    }
}

/// The facts a keep-set's model asserts, at noise threshold `tau`.
///
/// An object's events at one activity are summarised by their first and last timestamps,
/// and `x` precedes `y` for that object when its earliest `x` is strictly before its latest
/// `y` (see [`precedes`]).
///
/// A pair is an *ordering* when the objects disagreeing with it are at most a `tau` share
/// of those that witness it either way.
pub fn facts(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    kept: &HashSet<Cell>,
    tau: f64,
) -> Facts {
    facts_from(&Bounds::build(locel, schema, acts), kept, tau)
}

/// The same, over bounds already read off the log.
///
/// The bounds do not depend on the keep-set, so a caller evaluating more than one keep-set
/// builds them once and calls this. [`facts`] is the one-shot form.
pub fn facts_from(bounds: &Bounds, kept: &HashSet<Cell>, tau: f64) -> Facts {
    let mut fwd: HashMap<(ObjectTypeIndex, ActivityIndex, ActivityIndex), usize> = HashMap::new();
    let mut co_occurs: HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)> = HashSet::new();
    let mut present: HashMap<ObjectTypeIndex, HashSet<ActivityIndex>> = HashMap::new();

    let mut live: Vec<(ActivityIndex, i64, i64)> = Vec::new();
    for (t, objs) in bounds.per_type.iter().enumerate() {
        if objs.is_empty() {
            continue;
        }
        let here = present.entry(t).or_default();
        for ob in objs {
            live.clear();
            live.extend(ob.at.iter().filter(|(a, _, _)| kept.contains(&(*a, t))));
            for (a, _, _) in &live {
                here.insert(*a);
            }
            for (x, xmin, _) in &live {
                for (y, _, ymax) in &live {
                    if x == y {
                        continue;
                    }
                    co_occurs.insert((t, *x, *y));
                    if precedes(*xmin, *ymax) {
                        *fwd.entry((t, *x, *y)).or_default() += 1;
                    }
                }
            }
        }
    }

    // Ordered when the reverse direction is within the noise share.
    let ordered: HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)> = fwd
        .iter()
        .filter(|((t, x, y), n)| {
            let rev = fwd.get(&(*t, *y, *x)).copied().unwrap_or(0);
            let total = **n + rev;
            total > 0 && (rev as f64) / (total as f64) <= tau
        })
        .map(|(k, _)| *k)
        .collect();

    // Covering pairs: drop an ordering that two other orderings already imply.
    let ordering: HashSet<_> = ordered
        .iter()
        .filter(|(t, x, y)| {
            let acts_of_type = present.get(t).map(|s| s.iter()).into_iter().flatten();
            !acts_of_type
                .clone()
                .any(|m| m != x && m != y && ordered.contains(&(*t, *x, *m)) && ordered.contains(&(*t, *m, *y)))
        })
        .copied()
        .collect();

    // No shared object is a role fact; shared objects with nothing strictly before
    // anything is a tie.
    let mut role = HashSet::new();
    let mut tied = HashSet::new();
    for (t, acts_of_type) in &present {
        let mut sorted: Vec<ActivityIndex> = acts_of_type.iter().copied().collect();
        sorted.sort_unstable();
        for (i, x) in sorted.iter().enumerate() {
            for y in sorted.iter().skip(i + 1) {
                if !co_occurs.contains(&(*t, *x, *y)) && !co_occurs.contains(&(*t, *y, *x)) {
                    role.insert((*t, *x, *y));
                } else if !fwd.contains_key(&(*t, *x, *y)) && !fwd.contains_key(&(*t, *y, *x)) {
                    tied.insert((*t, *x, *y));
                }
            }
        }
    }

    Facts {
        ordering,
        role,
        asserted: ordered,
        tied,
    }
}

/// The strict ordering pairs one object type asserts under a keep-set.
///
/// [`Facts::asserted`] restricted to a single type, without the covering reduction, the
/// role scan or the tie scan. The searches call this once per candidate cell per round.
pub fn asserted_of_type(
    bounds: &Bounds,
    kept: &HashSet<Cell>,
    t: ObjectTypeIndex,
) -> HashSet<(ActivityIndex, ActivityIndex)> {
    let Some(objs) = bounds.per_type.get(t) else {
        return HashSet::new();
    };
    let mut fwd: HashSet<(ActivityIndex, ActivityIndex)> = HashSet::new();
    let mut live: Vec<(ActivityIndex, i64, i64)> = Vec::new();
    for ob in objs {
        live.clear();
        live.extend(ob.at.iter().filter(|(a, _, _)| kept.contains(&(*a, t))));
        for (x, xmin, _) in &live {
            for (y, _, ymax) in &live {
                if x != y && precedes(*xmin, *ymax) {
                    fwd.insert((*x, *y));
                }
            }
        }
    }
    // Strict: a pair witnessed both ways orders nothing (the `tau = 0` reading).
    fwd.iter()
        .filter(|(x, y)| !fwd.contains(&(*y, *x)))
        .copied()
        .collect()
}

/// The same, for every type at once.
pub fn asserted_by_type(
    bounds: &Bounds,
    kept: &HashSet<Cell>,
) -> Vec<HashSet<(ActivityIndex, ActivityIndex)>> {
    (0..bounds.per_type.len())
        .map(|t| asserted_of_type(bounds, kept, t))
        .collect()
}

/// The facts of `full` that a keep-set still delivers, drawn or pushed forward.
///
/// A fact `(T, x, y)` survives when some kept type `S` orders `x < y` in the reduced model
/// and determines `T` at both endpoints, so `S`'s ordering carries `T`'s.
pub fn delivered_facts(
    grid: &CellGrid,
    full: &Facts,
    reduced: &Facts,
    kept: &HashSet<Cell>,
    n_types: usize,
) -> Facts {
    let carries = |(t, x, y): &(ObjectTypeIndex, ActivityIndex, ActivityIndex),
                   reduced_set: &HashSet<(ObjectTypeIndex, ActivityIndex, ActivityIndex)>| {
        (0..n_types).any(|s| {
            reduced_set.contains(&(s, *x, *y))
                && kept.contains(&(*x, s))
                && kept.contains(&(*y, s))
                && readable_from(grid, *x, s, *t)
                && readable_from(grid, *y, s, *t)
        })
    };
    Facts {
        ordering: full
            .ordering
            .iter()
            .filter(|f| carries(f, &reduced.ordering))
            .copied()
            .collect(),
        role: full
            .role
            .iter()
            .filter(|f| carries(f, &reduced.role))
            .copied()
            .collect(),
        asserted: reduced.asserted.clone(),
        tied: reduced.tied.clone(),
    }
}

/// Can the cell `(a, t)` be read off a cell of type `s` kept at the same activity?
///
/// A fact about `t` needs no cell of its own where a kept type determines `t`.
fn readable_from(grid: &CellGrid, a: ActivityIndex, s: ObjectTypeIndex, t: ObjectTypeIndex) -> bool {
    if s == t {
        return true;
    }
    // Determinacy at an activity is a closure, not one step: a fibre out to a third type
    // followed by a function still determines.
    let cells = &grid.per_activity[a];
    let (Some(from), Some(target)) = (cells.slot(s), cells.slot(t)) else {
        return false;
    };
    let k = cells.present.len();
    let mut reached = vec![false; k];
    reached[from] = true;
    loop {
        let mut grew = false;
        for i in 0..k {
            if !reached[i] {
                continue;
            }
            for j in 0..k {
                if !reached[j] && cells.recon[i][j].is_some() {
                    reached[j] = true;
                    grew = true;
                }
            }
        }
        if !grew {
            return reached[target];
        }
    }
}

/// Promote cells until every ordering fact is drawn by its own type or excused by
/// determination, endpoint by endpoint.
///
/// A fact `(T, x, y)` is fine at an endpoint when `(x, T)` is at flow or when the flow
/// types determine `T` at `x`. An endpoint that is neither promotes that cell to flow.
/// Cross-type coverage without a map is not an excuse: another type ordering the same pair
/// says nothing about `T`.
///
/// Only ordering facts are enforced. Separations and multiplicities live in the
/// involvement badge, which keeps the participations.
///
/// Promotion is monotone (added flow cells only grow every determined set), so the loop
/// reaches a fixpoint.
pub fn fact_repair(grid: &CellGrid, facts: &Facts, flow: &mut HashSet<Cell>) -> Vec<Cell> {
    let mut added = Vec::new();
    loop {
        let endpoint_ok = |a: ActivityIndex, t: ObjectTypeIndex, flow: &HashSet<Cell>| {
            if flow.contains(&(a, t)) {
                return true;
            }
            let Some(here) = grid.per_activity.get(a) else {
                return true;
            };
            let kept: Vec<ObjectTypeIndex> = here
                .present
                .iter()
                .filter(|s| flow.contains(&(a, **s)))
                .copied()
                .collect();
            let reached = here.determined_by(&kept);
            here.slot(t).map(|j| reached[j]).unwrap_or(true)
        };
        let mut to_add: Vec<Cell> = Vec::new();
        for (t, x, y) in &facts.ordering {
            for a in [*x, *y] {
                if !endpoint_ok(a, *t, flow) && grid.cells.contains(&(a, *t)) {
                    to_add.push((a, *t));
                }
            }
        }
        to_add.sort_unstable();
        to_add.dedup();
        to_add.retain(|c| !flow.contains(c));
        if to_add.is_empty() {
            break;
        }
        for c in to_add {
            flow.insert(c);
            added.push(c);
        }
    }
    added.sort_unstable();
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::cells::{ActivityCells, ReconRoute};

    /// Two activities, "a" then "b"; three types at each: `items` (0), `orders` (1),
    /// `employees` (2). `items` determines `orders` at both activities; nothing determines
    /// `employees` anywhere.
    fn grid() -> CellGrid {
        let cells_at = || ActivityCells {
            present: vec![0, 1, 2],
            counts: vec![1, 1, 1],
            recon: vec![
                vec![None, Some(ReconRoute::Fibre { witness: 0 }), None],
                vec![None, None, None],
                vec![None, None, None],
            ],
        };
        CellGrid {
            activities: vec!["a".into(), "b".into()],
            per_activity: vec![cells_at(), cells_at()],
            cells: HashSet::from([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
            e2o_total: 6,
        }
    }

    /// One ordering fact per type, activity 0 before activity 1.
    fn facts() -> Facts {
        Facts {
            ordering: HashSet::from([(0, 0, 1), (1, 0, 1), (2, 0, 1)]),
            ..Default::default()
        }
    }

    #[test]
    fn the_three_cases_are_told_apart() {
        let grid = grid();
        let facts = facts();
        let mut flow: HashSet<Cell> = HashSet::from([(0, 0), (1, 0)]);

        let added = fact_repair(&grid, &facts, &mut flow);

        // `items`: already flow at both endpoints, left alone.
        assert!(flow.contains(&(0, 0)));
        assert!(flow.contains(&(1, 0)));

        // `orders`: determined by `items` at both activities, so no promotion.
        assert!(!flow.contains(&(0, 1)));
        assert!(!flow.contains(&(1, 1)));

        // `employees`: neither flow nor determined at either endpoint, so both promoted.
        assert_eq!(added, vec![(0, 2), (1, 2)]);
        assert!(flow.contains(&(0, 2)));
        assert!(flow.contains(&(1, 2)));
    }

    #[test]
    fn repair_is_idempotent() {
        let grid = grid();
        let facts = facts();
        let mut flow: HashSet<Cell> = HashSet::from([(0, 0), (1, 0)]);
        fact_repair(&grid, &facts, &mut flow);

        let added_again = fact_repair(&grid, &facts, &mut flow);
        assert!(added_again.is_empty());
    }

    #[test]
    fn promotion_never_touches_a_cell_outside_the_grid() {
        let mut grid = grid();
        grid.cells.remove(&(1, 2));
        let facts = facts();
        let mut flow: HashSet<Cell> = HashSet::from([(0, 0), (1, 0)]);

        let added = fact_repair(&grid, &facts, &mut flow);

        assert_eq!(added, vec![(0, 2)]);
        assert!(!flow.contains(&(1, 2)));
    }
}
