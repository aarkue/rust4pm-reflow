use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::{EventIndex, ObjectIndex},
    LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    arcs::ActivityIndexing,
    bounds::Bounds,
    cells::{ActivityIndex, Cell, CellGrid},
    closure::{ObjectFn, MAX_COMPOSE_DEPTH, MAX_WITNESSES_PER_PAIR},
    lifetime::Lifetimes,
    schema::{ObjectTypeIndex, StructuralSchema},
    simultaneity::precedes,
};

/// Route-object checks above which candidate enumeration is not attempted.
///
/// Compared against [`expansion_work`], which counts what the enumeration loop does.
pub const EXPANSION_WORK_BUDGET: u64 = 200_000_000;

/// Qualifier written on a participation that expansion adds, so added tuples stay
/// separable from recorded ones.
pub const EXPANSION_QUALIFIER: &str = "derived";

/// Share of a candidate cell's determined tuples that must fall inside their own object's
/// recorded lifetime for the cell to be written.
///
/// A lifetime guard, separate from how a map is admitted. At `1` a single-event object has
/// a degenerate lifetime, so any participation written to it at another instant falls
/// outside by construction; at `0` there is no guard.
pub const DEFAULT_THETA: f64 = 0.95;

/// Which way a candidate cell is derived.
///
/// Forward applies the map: at an event naming a source object, write its image. Backward
/// applies the preimage: at an event naming a target object, write every source object that
/// maps to it. Forward writes at most one object per source, backward the whole fibre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpansionDirection {
    /// Apply the map only.
    #[default]
    Forward,
    /// Expand the preimage only.
    Backward,
    /// Both, merged per cell.
    Both,
}

impl ExpansionDirection {
    fn forward(self) -> bool {
        self != ExpansionDirection::Backward
    }

    fn backward(self) -> bool {
        self != ExpansionDirection::Forward
    }
}

/// A derivation route: a chain of maps applied in order, with the function it computes.
///
/// The chain is the route's name. [`routes`] enumerates one per chain; expansion applies
/// [`agreed_routes`], one per (source, target), because a cell two chains disagree about
/// is not determined.
#[derive(Debug, Clone)]
pub struct Route {
    /// Type the route is defined on.
    pub source: ObjectTypeIndex,
    /// Type it lands in.
    pub target: ObjectTypeIndex,
    /// Indices into the schema's map list ([`StructuralSchema::maps`]), in application
    /// order. A single index is a generator, several are a composition.
    pub via: Vec<usize>,
    /// The composed function.
    pub f: ObjectFn,
}

/// Every derivation route the schema offers, closed under composition.
///
/// A composite is the composition of two partial functions and is kept whenever it is
/// defined anywhere. Nothing is filtered on how total it is: whether a route can carry a
/// cell is decided per cell, at [`DETERMINATION_THETA`](super::DETERMINATION_THETA).
///
/// A function discovered twice (recorded and by co-participation) is one route.
pub fn routes(schema: &StructuralSchema) -> Vec<Route> {
    let mut base: Vec<Route> = Vec::new();
    for (i, m) in schema.maps().enumerate() {
        if base
            .iter()
            .any(|r| (r.source, r.target) == (m.source, m.target) && r.f == m.f)
        {
            continue;
        }
        base.push(Route {
            source: m.source,
            target: m.target,
            via: vec![i],
            f: m.f.clone(),
        });
    }

    let mut out = base.clone();
    let mut frontier = base.clone();
    for _ in 1..MAX_COMPOSE_DEPTH {
        let mut grown: Vec<Route> = Vec::new();
        for r in &frontier {
            for b in &base {
                if b.source != r.target || b.target == r.source {
                    continue;
                }
                let f: ObjectFn = r
                    .f
                    .iter()
                    .filter_map(|(x, y)| b.f.get(y).map(|z| (*x, *z)))
                    .collect();
                if f.is_empty() {
                    continue;
                }
                let pair = (r.source, b.target);
                let (mut seen, mut duplicate) = (0usize, false);
                for x in out.iter().chain(grown.iter()) {
                    if (x.source, x.target) != pair {
                        continue;
                    }
                    seen += 1;
                    duplicate |= x.f == f;
                }
                if seen >= MAX_WITNESSES_PER_PAIR || duplicate {
                    continue;
                }
                let mut via = r.via.clone();
                via.extend_from_slice(&b.via);
                grown.push(Route {
                    source: r.source,
                    target: b.target,
                    via,
                    f,
                });
            }
        }
        if grown.is_empty() {
            break;
        }
        out.extend(grown.iter().cloned());
        frontier = grown;
    }
    out
}

/// A type pair whose derivation routes contradict each other, and on how many objects.
///
/// Not an error: two chains from `items` to `employees`, one through `orders` and one
/// through `packages`, need not agree. The pair then determines nothing for those objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Disagreement {
    /// Type the routes are defined on.
    pub source: ObjectTypeIndex,
    /// Type they land in.
    pub target: ObjectTypeIndex,
    /// Source objects two routes name different targets for. These carry no candidate.
    pub objects: usize,
    /// Source objects at least one route is defined on.
    pub domain: usize,
}

/// The routes expansion applies: one per (source, target), on the objects its chains agree
/// about.
///
/// A map determines one object. Unioning what every chain of a pair writes would put two
/// determined images for one object at one event into `max`, the coverage target. So the
/// chains are merged, and a source object two of them disagree about is dropped from the
/// merged function and counted.
///
/// `via` on the merged route is its shortest chain, as a name for the pair; the merged
/// function can be defined on objects that chain is not.
pub fn agreed_routes(schema: &StructuralSchema) -> (Vec<Route>, Vec<Disagreement>) {
    let mut by_pair: BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), Vec<Route>> = BTreeMap::new();
    for r in routes(schema) {
        by_pair.entry((r.source, r.target)).or_default().push(r);
    }

    let mut out: Vec<Route> = Vec::new();
    let mut clashes: Vec<Disagreement> = Vec::new();
    for ((source, target), mut rs) in by_pair {
        rs.sort_by(|x, y| (x.via.len(), &x.via).cmp(&(y.via.len(), &y.via)));
        let mut f: ObjectFn = HashMap::new();
        let mut split: HashSet<ObjectIndex> = HashSet::new();
        for r in &rs {
            for (x, y) in &r.f {
                match f.get(x) {
                    Some(z) if z == y => {}
                    Some(_) => {
                        split.insert(*x);
                    }
                    None => {
                        f.insert(*x, *y);
                    }
                }
            }
        }
        for x in &split {
            f.remove(x);
        }
        if !split.is_empty() {
            clashes.push(Disagreement {
                source,
                target,
                objects: split.len(),
                domain: f.len() + split.len(),
            });
        }
        if f.is_empty() {
            continue;
        }
        out.push(Route {
            source,
            target,
            via: rs[0].via.clone(),
            f,
        });
    }
    (out, clashes)
}

/// Route-object checks that enumerating candidates would take on this log.
///
/// A bound on the loop [`expansion_candidates`] runs, which skips an activity where the
/// route's target type is already recorded and visits only the objects of the route's
/// source type:
///
/// ```text
/// sum over routes (s, t) of  sum over activities a where (a, t) is not recorded  of  |E2O(a, s)|
/// ```
///
/// Backward is the same sum with the roles swapped, multiplied by the mean fibre size,
/// because one target object expands to its whole preimage. Needs only the grid's
/// per-activity counts.
pub fn expansion_work(routes: &[Route], grid: &CellGrid, dir: ExpansionDirection) -> u64 {
    let mut work = 0u64;
    for r in routes {
        if dir.forward() {
            for (a, here) in grid.per_activity.iter().enumerate() {
                if grid.cells.contains(&(a, r.target)) {
                    continue;
                }
                work += here.slot(r.source).map_or(0, |j| here.counts[j]) as u64;
            }
        }
        if dir.backward() {
            let image: HashSet<ObjectIndex> = r.f.values().copied().collect();
            if image.is_empty() {
                continue;
            }
            let fibre = r.f.len().div_ceil(image.len()) as u64;
            for (a, here) in grid.per_activity.iter().enumerate() {
                if grid.cells.contains(&(a, r.source)) {
                    continue;
                }
                work += fibre * here.slot(r.target).map_or(0, |j| here.counts[j]) as u64;
            }
        }
    }
    work
}

/// One way to write a cell the log does not record, and what it alone would write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationRoute {
    /// The type at the same activity the route reads from.
    pub source: ObjectTypeIndex,
    /// Whether the route was read backwards, as a preimage rather than as a function.
    pub backward: bool,
    /// The maps it applies, as indices into the schema's map list.
    pub via: Vec<usize>,
    /// The admissible tuples this route alone writes.
    ///
    /// Held per route because a cell reachable from two source types at the same activity
    /// is reachable by two different functions, and a caller picking one must get that one.
    /// Chains from the same source type are merged before this point, see [`agreed_routes`].
    pub tuples: Vec<(EventIndex, ObjectIndex)>,
}

/// A cell the schema determines and the log does not record.
///
/// The tuples are carried so a level can be scored on what it writes and then written
/// without a second pass.
#[derive(Debug, Clone)]
pub struct ExpansionCandidate {
    /// The cell it would create.
    pub cell: Cell,
    /// The participations it would write, deduplicated across routes and sorted.
    pub tuples: Vec<(EventIndex, ObjectIndex)>,
    /// Events that gain at least one participation.
    pub events: usize,
    /// Tuples whose event falls inside the written object's own recorded lifetime.
    ///
    /// Counted per tuple, not per event: per event, one alive object among five would make
    /// the event pass while four fifths of what it writes is anachronistic.
    pub alive: usize,
    /// Whether [`alive`](Self::alive) reaches `theta` of the tuples.
    pub admissible: bool,
    /// Every route that derives it, with the share it accounts for.
    pub routes: Vec<DerivationRoute>,
}

impl ExpansionCandidate {
    /// Share of the written tuples inside their own object's lifetime, which is what
    /// `theta` is compared against.
    pub fn admission_rate(&self) -> f64 {
        if self.tuples.is_empty() {
            0.0
        } else {
            self.alive as f64 / self.tuples.len() as f64
        }
    }

    /// Tuples written outside their own object's lifetime.
    ///
    /// An admitted cell is written whole, so these are not rejected. They are expansion's
    /// residuals. Leaving them out would make the cell ragged, present at some events of an
    /// activity and missing at others.
    pub fn extrapolated(&self) -> usize {
        self.tuples.len() - self.alive
    }
}

/// Participations, as the pairs a log would carry.
type Tuples = HashSet<(EventIndex, ObjectIndex)>;

#[derive(Default)]
struct Accumulator {
    tuples: Tuples,
    per_route: BTreeMap<(ObjectTypeIndex, bool, Vec<usize>), Tuples>,
}

/// The cells expansion could write, with the tuples each would add and whether `theta`
/// admits it.
///
/// A tuple is alive when its event falls inside the written object's own recorded
/// lifetime, `first(t) <= time(e) <= last(t)`. Determinacy says which object would be named
/// if one were named, not that the object existed yet, or still. Two-sided, because the
/// one-sided version admits an object's whole tail after its last recorded event.
///
/// The decision is per cell, at `theta` of the cell's determined tuples, and an admitted
/// cell is written whole. A per-event decision would leave the cell ragged.
pub fn expansion_candidates(
    locel: &SlimLinkedOCEL,
    routes: &[Route],
    schema: &StructuralSchema,
    grid: &CellGrid,
    acts: &ActivityIndexing,
    theta: f64,
    dir: ExpansionDirection,
) -> Vec<ExpansionCandidate> {
    if routes.is_empty() {
        return Vec::new();
    }
    // A type recorded at every activity can be the target of no candidate cell, so its
    // routes are dropped before the pass instead of once per event inside it. Backward
    // filters on the source, since the cell a preimage creates is `(a, source)`.
    let n_acts = grid.activities.len();
    let fibres: Vec<Option<HashMap<ObjectIndex, Vec<ObjectIndex>>>> = routes
        .iter()
        .map(|r| {
            if !dir.backward() || (0..n_acts).all(|a| grid.cells.contains(&(a, r.source))) {
                return None;
            }
            let mut fib: HashMap<ObjectIndex, Vec<ObjectIndex>> = HashMap::new();
            for (x, y) in &r.f {
                fib.entry(*y).or_default().push(*x);
            }
            for v in fib.values_mut() {
                v.sort_unstable();
            }
            Some(fib)
        })
        .collect();
    let at = |a: ActivityIndex, want: fn(&Route) -> ObjectTypeIndex, on: bool| -> Vec<usize> {
        if !on {
            return Vec::new();
        }
        (0..routes.len())
            .filter(|i| !grid.cells.contains(&(a, want(&routes[*i]))))
            .collect()
    };
    let fwd_at: Vec<Vec<usize>> = (0..n_acts)
        .map(|a| at(a, |r| r.target, dir.forward()))
        .collect();
    let bwd_at: Vec<Vec<usize>> = (0..n_acts)
        .map(|a| {
            at(a, |r| r.source, dir.backward())
                .into_iter()
                .filter(|i| fibres[*i].is_some())
                .collect()
        })
        .collect();
    if fwd_at.iter().chain(bwd_at.iter()).all(Vec::is_empty) {
        return Vec::new();
    }

    let lifetimes = Lifetimes::build(locel);

    let mut per_cell: BTreeMap<Cell, Accumulator> = BTreeMap::new();
    let mut by_type: HashMap<ObjectTypeIndex, Vec<ObjectIndex>> = HashMap::new();
    for e in locel.get_all_evs() {
        let a = acts.act_of[e.get_ev(locel).event_type];
        if fwd_at[a].is_empty() && bwd_at[a].is_empty() {
            continue;
        }
        by_type.clear();
        let here: HashSet<ObjectIndex> = e.get_e2o(locel).copied().collect();
        for o in &here {
            by_type.entry(schema.type_of[o]).or_default().push(*o);
        }
        for i in &fwd_at[a] {
            let r = &routes[*i];
            let Some(sources) = by_type.get(&r.source) else {
                continue;
            };
            for o in sources {
                let Some(img) = r.f.get(o) else { continue };
                if here.contains(img) {
                    continue;
                }
                let acc = per_cell.entry((a, r.target)).or_default();
                acc.tuples.insert((e, *img));
                acc.per_route
                    .entry((r.source, false, r.via.clone()))
                    .or_default()
                    .insert((e, *img));
            }
        }
        for i in &bwd_at[a] {
            let r = &routes[*i];
            let Some(targets) = by_type.get(&r.target) else {
                continue;
            };
            let Some(fib) = fibres[*i].as_ref() else {
                continue;
            };
            for y in targets {
                for x in fib.get(y).into_iter().flatten() {
                    if here.contains(x) {
                        continue;
                    }
                    let acc = per_cell.entry((a, r.source)).or_default();
                    acc.tuples.insert((e, *x));
                    acc.per_route
                        .entry((r.target, true, r.via.clone()))
                        .or_default()
                        .insert((e, *x));
                }
            }
        }
    }

    per_cell
        .into_iter()
        .filter(|(_, acc)| !acc.tuples.is_empty())
        .map(|(cell, acc)| {
            let mut tuples: Vec<(EventIndex, ObjectIndex)> = acc.tuples.into_iter().collect();
            tuples.sort();
            let events = tuples
                .iter()
                .map(|(e, _)| *e)
                .collect::<HashSet<EventIndex>>()
                .len();
            let alive = tuples
                .iter()
                .filter(|(e, o)| lifetimes.contains(*o, e.get_time(locel).timestamp_millis()))
                .count();
            let mut route_list: Vec<DerivationRoute> = acc
                .per_route
                .into_iter()
                .map(|((source, backward, via), written)| {
                    let mut tuples: Vec<(EventIndex, ObjectIndex)> = written.into_iter().collect();
                    tuples.sort();
                    DerivationRoute {
                        source,
                        backward,
                        via,
                        tuples,
                    }
                })
                .collect();
            route_list.sort_by(|x, y| {
                (std::cmp::Reverse(x.tuples.len()), x.via.len(), x.source, x.backward).cmp(&(
                    std::cmp::Reverse(y.tuples.len()),
                    y.via.len(),
                    y.source,
                    y.backward,
                ))
            });
            let admissible = alive as f64 / tuples.len() as f64 >= theta;
            ExpansionCandidate {
                cell,
                tuples,
                events,
                alive,
                admissible,
                routes: route_list,
            }
        })
        .collect()
}

/// Eventually-follows per object type, read off bounds already taken from the log.
///
/// Whether `x` precedes `y` for one object depends only on that object's own events, so
/// this does not change as cells are added elsewhere. Compute it once over everything
/// expansion could write and restrict it to a cell set afterwards.
///
/// Fold candidate tuples into the bounds with [`Bounds::plus`](super::Bounds::plus) first.
pub fn eventually_follows(
    bounds: &Bounds,
) -> HashMap<ObjectTypeIndex, HashSet<(ActivityIndex, ActivityIndex)>> {
    let mut ef: HashMap<ObjectTypeIndex, HashSet<(ActivityIndex, ActivityIndex)>> = HashMap::new();
    for (t, objs) in bounds.per_type.iter().enumerate() {
        if objs.is_empty() {
            continue;
        }
        let set = ef.entry(t).or_default();
        for ob in objs {
            for (x, xmin, _) in &ob.at {
                for (y, _, ymax) in &ob.at {
                    if x != y && precedes(*xmin, *ymax) {
                        set.insert((*x, *y));
                    }
                }
            }
        }
    }
    ef
}

/// Types that occur at every activity.
///
/// They connect the whole incidence graph and so say nothing to a connectivity objective.
pub fn free_types(grid: &CellGrid) -> HashSet<ObjectTypeIndex> {
    let n = grid.activities.len();
    let mut at: HashMap<ObjectTypeIndex, usize> = HashMap::new();
    for (_, t) in &grid.cells {
        *at.entry(*t).or_default() += 1;
    }
    at.into_iter().filter(|(_, k)| *k == n).map(|(t, _)| t).collect()
}

/// The tuples a chosen set of expansion cells writes.
///
/// A cell mapped to `None` is written from every route it has, which is what the levels ask
/// for; a cell mapped to a route's map chain is written from that route alone.
pub fn tuples_for(
    candidates: &[ExpansionCandidate],
    chosen: &HashMap<Cell, Option<Vec<usize>>>,
) -> Vec<(EventIndex, ObjectIndex)> {
    let mut out: Vec<(EventIndex, ObjectIndex)> = Vec::new();
    for c in candidates {
        let Some(via) = chosen.get(&c.cell) else {
            continue;
        };
        match via {
            None => out.extend(c.tuples.iter().copied()),
            Some(via) => out.extend(
                c.routes
                    .iter()
                    .filter(|r| r.via == *via)
                    .flat_map(|r| r.tuples.iter().copied()),
            ),
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::SchemaClosure;
    use crate::core::chrono::DateTime;

    /// Two items of one order. `place` names both, `pick` names only the item, and `ship`
    /// only the order: `orders` at `pick` is determined by the item, and `items` at `ship`
    /// by the fibre of the same map.
    fn toy() -> SlimLinkedOCEL {
        let mut ocel = SlimLinkedOCEL::new();
        ocel.add_object_type("items", Vec::new());
        ocel.add_object_type("orders", Vec::new());
        for a in ["place", "pick", "ship"] {
            ocel.add_event_type(a, Vec::new());
        }
        let o1 = ocel
            .add_object("orders", Some("o1".into()), Vec::new(), Vec::new())
            .unwrap();
        let o2 = ocel
            .add_object("orders", Some("o2".into()), Vec::new(), Vec::new())
            .unwrap();
        let i1 = ocel
            .add_object("items", Some("i1".into()), Vec::new(), vec![("of".into(), o1)])
            .unwrap();
        let i2 = ocel
            .add_object("items", Some("i2".into()), Vec::new(), vec![("of".into(), o2)])
            .unwrap();

        let t = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        let mut clock = 1_600_000_000_000i64;
        for (item, order) in [(i1, o1), (i2, o2)] {
            for (act, objs) in [
                ("place", vec![("item".into(), item), ("order".into(), order)]),
                ("pick", vec![("item".into(), item)]),
                ("ship", vec![("order".into(), order)]),
            ] {
                clock += 1000;
                ocel.add_event(act, t(clock), Some(format!("{act}-{clock}")), Vec::new(), objs);
            }
        }
        ocel
    }

    fn prepared(ocel: &SlimLinkedOCEL) -> (StructuralSchema, CellGrid, ActivityIndexing) {
        let schema = StructuralSchema::discover(ocel);
        let closure = SchemaClosure::build(ocel, &schema);
        let grid = CellGrid::build(ocel, &schema, &closure);
        let acts = ActivityIndexing::build(ocel, &grid);
        (schema, grid, acts)
    }

    #[test]
    fn candidates_are_the_cells_the_schema_determines_and_the_log_omits() {
        let ocel = toy();
        let (schema, grid, acts) = prepared(&ocel);
        let cand = expansion_candidates(
            &ocel,
            &agreed_routes(&schema).0,
            &schema,
            &grid,
            &acts,
            DEFAULT_THETA,
            ExpansionDirection::Forward,
        );

        let named: Vec<(String, String, usize)> = cand
            .iter()
            .map(|c| {
                (
                    grid.activities[c.cell.0].clone(),
                    schema.types[c.cell.1].clone(),
                    c.tuples.len(),
                )
            })
            .collect();
        // `pick` gains the order of its item, `ship` gains the items of its order.
        assert_eq!(
            named,
            vec![
                ("pick".to_string(), "orders".to_string(), 2),
                ("ship".to_string(), "items".to_string(), 2),
            ]
        );
        // Every candidate names the route it came from.
        assert!(cand.iter().all(|c| !c.routes.is_empty()));
    }

    /// A cell every one of whose tuples antedates its own object is admitted by nothing,
    /// whatever `theta` is: the rate is 0.
    #[test]
    fn a_cell_that_antedates_its_own_objects_is_not_admissible() {
        let mut ocel = toy();
        // A package created after the order ships: determinacy would name it at `place`,
        // where it did not yet exist.
        ocel.add_object_type("packages", Vec::new());
        ocel.add_event_type("pack", Vec::new());
        let orders: Vec<_> = ocel
            .get_all_obs()
            .filter(|o| o.get_ob_type(&ocel).as_str() == "orders")
            .collect();
        let t = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        for (n, o) in orders.iter().enumerate() {
            let p = ocel
                .add_object(
                    "packages",
                    Some(format!("p{n}")),
                    Vec::new(),
                    vec![("for".into(), *o)],
                )
                .unwrap();
            ocel.add_event(
                "pack",
                t(1_700_000_000_000 + n as i64 * 1000),
                Some(format!("pack-{n}")),
                Vec::new(),
                vec![("package".into(), p), ("order".into(), *o)],
            );
        }
        let (schema, grid, acts) = prepared(&ocel);
        let cand = expansion_candidates(
            &ocel,
            &agreed_routes(&schema).0,
            &schema,
            &grid,
            &acts,
            DEFAULT_THETA,
            ExpansionDirection::Forward,
        );
        let at_place = cand
            .iter()
            .find(|c| grid.activities[c.cell.0] == "place" && schema.types[c.cell.1] == "packages")
            .expect("the cell is a candidate; what it is not is admissible");
        assert_eq!((at_place.alive, at_place.tuples.len()), (0, 2));
        assert!(
            !at_place.admissible,
            "packages at `place` antedate their own creation and must not be written"
        );
    }

    /// Backward writes the fibre: at an event naming one order, every item of that order.
    #[test]
    fn the_preimage_is_the_other_direction_of_the_same_route() {
        let ocel = toy();
        let (schema, grid, acts) = prepared(&ocel);
        let cand = expansion_candidates(
            &ocel,
            &agreed_routes(&schema).0,
            &schema,
            &grid,
            &acts,
            DEFAULT_THETA,
            ExpansionDirection::Backward,
        );
        let named: Vec<(String, String, usize, bool)> = cand
            .iter()
            .map(|c| {
                (
                    grid.activities[c.cell.0].clone(),
                    schema.types[c.cell.1].clone(),
                    c.tuples.len(),
                    c.routes.iter().all(|r| r.backward),
                )
            })
            .collect();
        // `ship` gains its items through the fibre of `items -> orders`; `pick` gains its
        // order through the fibre of `orders -> items`, which co-participation discovers
        // because each toy order holds one item.
        assert_eq!(
            named,
            vec![
                ("pick".to_string(), "orders".to_string(), 2, true),
                ("ship".to_string(), "items".to_string(), 2, true),
            ]
        );
    }

    /// `items -> employees` two ways, through the order and through the package, naming two
    /// different people. Unioning them writes both at one event and calls the cell
    /// determined; it is not.
    #[test]
    fn routes_that_contradict_each_other_determine_nothing_and_are_counted() {
        let mut ocel = SlimLinkedOCEL::new();
        for t in ["items", "orders", "packages", "employees"] {
            ocel.add_object_type(t, Vec::new());
        }
        ocel.add_event_type("place", Vec::new());
        let seller = ocel
            .add_object("employees", Some("sells".into()), Vec::new(), Vec::new())
            .unwrap();
        let packer = ocel
            .add_object("employees", Some("packs".into()), Vec::new(), Vec::new())
            .unwrap();

        let t = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        for n in 0..3i64 {
            let o = ocel
                .add_object(
                    "orders",
                    Some(format!("o{n}")),
                    Vec::new(),
                    vec![("sold by".into(), seller)],
                )
                .unwrap();
            let p = ocel
                .add_object(
                    "packages",
                    Some(format!("p{n}")),
                    Vec::new(),
                    vec![("packed by".into(), packer)],
                )
                .unwrap();
            let i = ocel
                .add_object(
                    "items",
                    Some(format!("i{n}")),
                    Vec::new(),
                    vec![("of".into(), o), ("in".into(), p)],
                )
                .unwrap();
            ocel.add_event(
                "place",
                t(1_600_000_000_000 + n * 1000),
                Some(format!("place-{n}")),
                Vec::new(),
                vec![("item".into(), i)],
            );
        }

        let schema = StructuralSchema::discover(&ocel);
        let ix = |name: &str| schema.types.iter().position(|x| x == name).unwrap();
        let (items, employees) = (ix("items"), ix("employees"));

        // Both chains exist, so the union would have written two employees per item.
        let chains = routes(&schema)
            .into_iter()
            .filter(|r| (r.source, r.target) == (items, employees))
            .count();
        assert_eq!(chains, 2);

        let (agreed, clashes) = agreed_routes(&schema);
        assert!(
            !agreed
                .iter()
                .any(|r| (r.source, r.target) == (items, employees)),
            "a pair its routes contradict each other on determines nothing"
        );
        let d = clashes
            .iter()
            .find(|d| (d.source, d.target) == (items, employees))
            .expect("the contradiction is reported rather than resolved");
        assert_eq!((d.objects, d.domain), (3, 3));
    }
}
