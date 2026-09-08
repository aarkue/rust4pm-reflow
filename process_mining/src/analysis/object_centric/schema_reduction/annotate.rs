//! Def. Annotation: which non-flow cells are implied, by what, and which are involved.
//!
//! Nothing here changes a log. A non-flow cell is implied when a route from a flow cell at
//! the same activity reconstructs it, and the route is what the annotation draws as a map;
//! it is involved otherwise. The reconstruction is evaluated against the object schema and
//! the object-to-object relation the log already records -- no edge is written, and the
//! tagged log itself carries nothing but the tags.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::{EventIndex, ObjectIndex},
    SlimLinkedOCEL,
};

use super::{
    assignment::CellState,
    cells::{Cell, CellGrid, ReconRoute},
    closure::SchemaClosure,
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// Objects of each type carried by one event.
type PerType = HashMap<ObjectTypeIndex, HashSet<ObjectIndex>>;

/// The relation a determination evaluates: which objects of the target type an object of the
/// source type puts at an event.
pub type Relation = HashMap<ObjectIndex, HashSet<ObjectIndex>>;

/// How the determining relation is witnessed, and therefore what a reader evaluates.
///
/// Every case is the same evaluation: collect the object-to-object edges of the named
/// qualifiers between the two types, orient them, and take the image of the source objects
/// the event carries. The cases differ only in where the edges came from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Realisation {
    /// The object-to-object qualifiers carrying the relation, empty for a derived one.
    pub qualifiers: Vec<String>,
    /// The edges run from the target type to the source type and are read as a preimage.
    pub backward: bool,
    /// The witness kind. `false` is a recorded object-to-object witness. `true` is
    /// co-participation, i.e. a derived relation of Def. Relation, which no qualifier of the
    /// log carries.
    pub derived: bool,
}

impl Realisation {
    /// The census label for the witness kind.
    pub fn witness(&self) -> &'static str {
        if self.derived {
            "co-participation"
        } else {
            "recorded"
        }
    }
}

/// What recomputes one implied cell's participations.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Determination {
    /// The type at the same activity the route reads from.
    pub source_type: String,
    /// How the relation is witnessed.
    pub realisation: Realisation,
    /// The chain of types the reconstruction passes through, from a flow cell down to this
    /// one, ending at [`Determination::source_type`].
    pub route: Vec<String>,
    /// Objects of the cell's type the route does not put back.
    ///
    /// Empty whenever the grid's determinacy test held at every event of the activity, not
    /// only at a share $\theta$ of them.
    pub residuals: Vec<String>,
}

/// One non-flow cell and what the annotation says about it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CellAnnotation {
    /// The activity.
    pub activity: String,
    /// The object type.
    pub object_type: String,
    /// [`CellState::Implied`] or [`CellState::Involvement`].
    pub state: CellState,
    /// The reconstruction that recovers an implied cell, and [`None`] for an involved one.
    pub determined_by: Option<Determination>,
    /// Participations the cell carries. They stay in the log, tagged non-flow.
    pub participations: usize,
    /// The event-to-object qualifiers they carry, distinct and with the tag dropped.
    pub qualifiers: Vec<String>,
}

/// The annotation of a flow layer: one entry per non-flow recorded cell.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AnnotationRecord {
    /// One entry per non-flow recorded cell, in emission order.
    pub cells: Vec<CellAnnotation>,
}

impl AnnotationRecord {
    /// Participations the non-flow cells carry.
    pub fn participations(&self) -> usize {
        self.cells.iter().map(|c| c.participations).sum()
    }

    /// Cells at involvement, and cells implied.
    pub fn split(&self) -> (usize, usize) {
        let implied = self.cells.iter().filter(|c| c.determined_by.is_some()).count();
        (self.cells.len() - implied, implied)
    }

    /// Implied cells by witness kind: recorded relations, then derived ones.
    pub fn witness_split(&self) -> (usize, usize) {
        let mut out = (0, 0);
        for d in self.cells.iter().filter_map(|c| c.determined_by.as_ref()) {
            if d.realisation.derived {
                out.1 += 1;
            } else {
                out.0 += 1;
            }
        }
        out
    }

    /// The longest reconstruction chain any implied cell needs.
    pub fn max_route_depth(&self) -> usize {
        self.cells
            .iter()
            .filter_map(|c| c.determined_by.as_ref())
            .map(|d| d.route.len())
            .max()
            .unwrap_or(0)
    }

    /// Implied cells whose route does not return the recorded participations exactly.
    pub fn inexact(&self) -> usize {
        self.cells
            .iter()
            .filter_map(|c| c.determined_by.as_ref())
            .filter(|d| !d.residuals.is_empty())
            .count()
    }
}

/// Annotate the non-flow cells of a flow layer (Def. Annotation).
///
/// `forced` names cells to report as involved even where a route determines them, which is
/// how an analyst's own decision to keep a cell readable in its own right is carried
/// through.
pub fn annotate(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    closure: &SchemaClosure,
    grid: &CellGrid,
    flow: &HashSet<Cell>,
    forced: &HashSet<Cell>,
) -> AnnotationRecord {
    let mut out = AnnotationRecord::default();
    let mut o2o = RecordedO2O::new();
    // The chain that reached a cell, so a cell determined through another non-flow cell can
    // name the whole route.
    let mut route_of: HashMap<Cell, Vec<String>> = HashMap::new();
    let mut non_flow: HashSet<Cell> = HashSet::new();

    for (a, here) in grid.per_activity.iter().enumerate() {
        let kept: Vec<ObjectTypeIndex> = here
            .present
            .iter()
            .filter(|t| flow.contains(&(a, **t)))
            .copied()
            .collect();
        if kept.len() == here.present.len() {
            continue;
        }
        let routes = here.determining_routes(&kept);
        let evs = events_of(locel, schema, grid, a);

        // Emission order, not slot order. `determining_routes` reaches cells in rounds, so a
        // cell at slot 0 can be reached from slot 3; walking `present` would emit a chained
        // cell before the one it reads from and lose the route.
        let k = here.present.len();
        let mut placed: Vec<bool> = (0..k).map(|j| flow.contains(&(a, here.present[j]))).collect();
        let mut order: Vec<usize> = Vec::new();
        loop {
            let mut grew = false;
            for j in 0..k {
                if placed[j] {
                    continue;
                }
                let ready = match &routes[j] {
                    None => true,
                    Some((i, _)) => placed[*i],
                };
                if ready {
                    order.push(j);
                    placed[j] = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }

        for j in order {
            let t = &here.present[j];
            let cell = (a, *t);
            non_flow.insert(cell);
            let determining = routes[j].clone().filter(|_| !forced.contains(&cell));
            let Some((i, route)) = determining else {
                let h = harvest(locel, *t, None, &evs, &Relation::new());
                out.cells.push(CellAnnotation {
                    activity: grid.activities[a].clone(),
                    object_type: schema.types[*t].clone(),
                    state: CellState::Involvement,
                    determined_by: None,
                    participations: h.participations,
                    qualifiers: h.qualifiers,
                });
                continue;
            };
            let s = here.present[i];
            let (realisation, relation) = realise(closure, &evs, s, *t, &route, &mut o2o);
            let h = harvest(locel, *t, Some(s), &evs, &relation);
            let mut chain = if non_flow.contains(&(a, s)) {
                route_of.get(&(a, s)).cloned().unwrap_or_default()
            } else {
                Vec::new()
            };
            chain.push(schema.types[s].clone());
            route_of.insert(cell, chain.clone());
            out.cells.push(CellAnnotation {
                activity: grid.activities[a].clone(),
                object_type: schema.types[*t].clone(),
                state: CellState::Implied,
                determined_by: Some(Determination {
                    source_type: schema.types[s].clone(),
                    realisation,
                    route: chain,
                    residuals: h.residuals,
                }),
                participations: h.participations,
                qualifiers: h.qualifiers,
            });
        }
    }
    out
}

/// The per-event object sets of one activity, which is what every determinacy question is
/// asked against.
fn events_of(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    grid: &CellGrid,
    a: usize,
) -> Vec<(EventIndex, PerType)> {
    locel
        .get_evs_of_type(&grid.activities[a])
        .map(|e| {
            let mut per: PerType = HashMap::new();
            for o in e.get_e2o(locel) {
                per.entry(schema.type_of[o]).or_default().insert(*o);
            }
            (*e, per)
        })
        .collect()
}

/// How the determining relation is witnessed, and the relation a reader evaluates.
///
/// A [`ReconRoute::Function`] or [`ReconRoute::Fibre`] already means no recorded qualifier
/// carries the relation forwards, so only the recorded relation read backwards, which the
/// grid never tests, is tried before falling back to the derived one.
fn realise(
    closure: &SchemaClosure,
    evs: &[(EventIndex, PerType)],
    s: ObjectTypeIndex,
    t: ObjectTypeIndex,
    route: &ReconRoute,
    o2o: &mut RecordedO2O,
) -> (Realisation, Relation) {
    if let ReconRoute::QualifiedUnion { qualifiers } = route {
        let rel = o2o.union(closure, s, t, qualifiers, false);
        return (
            Realisation {
                qualifiers: qualifiers.clone(),
                backward: false,
                derived: false,
            },
            rel,
        );
    }
    if let Some(quals) = o2o.reconstructs_backward(closure, evs, s, t) {
        let rel = o2o.union(closure, t, s, &quals, true);
        return (
            Realisation {
                qualifiers: quals,
                backward: true,
                derived: false,
            },
            rel,
        );
    }
    match route {
        ReconRoute::Function { witness } => {
            let f = &closure.maps[&(s, t)][*witness];
            let mut rel: Relation = HashMap::new();
            for (_, per) in evs {
                for x in per.get(&s).into_iter().flatten() {
                    if let Some(y) = f.get(x) {
                        rel.entry(*x).or_default().insert(*y);
                    }
                }
            }
            (
                Realisation {
                    qualifiers: Vec::new(),
                    backward: false,
                    derived: true,
                },
                rel,
            )
        }
        ReconRoute::Fibre { witness } => {
            let g = &closure.maps[&(t, s)][*witness];
            let mut rel: Relation = HashMap::new();
            for (_, per) in evs {
                for y in per.get(&t).into_iter().flatten() {
                    if let Some(x) = g.get(y) {
                        rel.entry(*x).or_default().insert(*y);
                    }
                }
            }
            (
                Realisation {
                    qualifiers: Vec::new(),
                    backward: true,
                    derived: true,
                },
                rel,
            )
        }
        ReconRoute::QualifiedUnion { .. } => unreachable!("handled above"),
    }
}

/// What a cell carries and which of its objects the route misses.
struct Harvest {
    participations: usize,
    qualifiers: Vec<String>,
    residuals: Vec<String>,
}

/// `s` is the type the route reads from, and `None` is an involved cell: nothing recomputes
/// it, so there is nothing to compare it against either.
fn harvest(
    locel: &SlimLinkedOCEL,
    t: ObjectTypeIndex,
    s: Option<ObjectTypeIndex>,
    evs: &[(EventIndex, PerType)],
    relation: &Relation,
) -> Harvest {
    let mut participations = 0usize;
    let mut quals: BTreeSet<String> = BTreeSet::new();
    let mut residuals: HashSet<ObjectIndex> = HashSet::new();
    for (e, per) in evs {
        let Some(here) = per.get(&t) else { continue };
        if let Some(s) = s {
            let rebuilt: HashSet<ObjectIndex> = per
                .get(&s)
                .into_iter()
                .flatten()
                .filter_map(|x| relation.get(x))
                .flat_map(|v| v.iter().copied())
                .collect();
            for o in here.difference(&rebuilt) {
                residuals.insert(*o);
            }
        }
        for (q, o) in e.get_e2o_q(locel) {
            if !here.contains(o) {
                continue;
            }
            participations += 1;
            let base = super::sigil::decode(q).1;
            if !quals.contains(base) {
                quals.insert(base.to_string());
            }
        }
    }
    let mut res: Vec<String> = residuals.iter().map(|o| o.get_ob(locel).id.clone()).collect();
    res.sort();
    Harvest {
        participations,
        qualifiers: quals.into_iter().collect(),
        residuals: res,
    }
}

/// The recorded object-to-object relation, indexed the two ways a determination reads it.
struct RecordedO2O {
    reverse: HashMap<(ObjectTypeIndex, ObjectTypeIndex, String), Relation>,
}

impl RecordedO2O {
    fn new() -> Self {
        Self {
            reverse: HashMap::new(),
        }
    }

    /// The union of the named qualified relations between two types, oriented so the key is
    /// always the source type's object.
    fn union(
        &mut self,
        closure: &SchemaClosure,
        from: ObjectTypeIndex,
        to: ObjectTypeIndex,
        qualifiers: &[String],
        reversed: bool,
    ) -> Relation {
        let mut out: Relation = HashMap::new();
        let Some(by_q) = closure.relations.get(&(from, to)) else {
            return out;
        };
        for q in qualifiers {
            let Some(r) = by_q.get(q) else { continue };
            for (x, ys) in r {
                for y in ys {
                    let (k, v) = if reversed { (*y, *x) } else { (*x, *y) };
                    out.entry(k).or_default().insert(v);
                }
            }
        }
        out
    }

    /// Does the recorded relation from `t` to `s`, read as a preimage, put the cell back?
    ///
    /// Same shape as the forward union test the grid runs: a qualifier is admissible when its
    /// preimage never exceeds the target set at any event, and if any subset of qualifiers
    /// works the union of the admissible ones does.
    fn reconstructs_backward(
        &mut self,
        closure: &SchemaClosure,
        evs: &[(EventIndex, PerType)],
        s: ObjectTypeIndex,
        t: ObjectTypeIndex,
    ) -> Option<Vec<String>> {
        let by_q = closure.relations.get(&(t, s))?;
        let mut admissible: Vec<String> = Vec::new();
        for q in by_q.keys() {
            let rel = self.reversed(closure, t, s, q);
            let ok = evs.iter().all(|(_, per)| match (per.get(&s), per.get(&t)) {
                (Some(os), Some(ot)) => os
                    .iter()
                    .filter_map(|x| rel.get(x))
                    .flat_map(|v| v.iter())
                    .all(|y| ot.contains(y)),
                (Some(os), None) => os.iter().all(|x| rel.get(x).is_none_or(HashSet::is_empty)),
                _ => true,
            });
            if ok {
                admissible.push(q.clone());
            }
        }
        if admissible.is_empty() {
            return None;
        }
        let rels: Vec<&Relation> = admissible
            .iter()
            .map(|q| self.reverse.get(&(t, s, q.clone())).unwrap())
            .collect();
        let exact = evs.iter().all(|(_, per)| {
            let (Some(os), Some(ot)) = (per.get(&s), per.get(&t)) else {
                return per.get(&s).is_none() && per.get(&t).is_none();
            };
            let image: HashSet<ObjectIndex> = rels
                .iter()
                .flat_map(|r| os.iter().filter_map(move |x| r.get(x)))
                .flat_map(|v| v.iter().copied())
                .collect();
            &image == ot
        });
        exact.then_some(admissible)
    }

    /// One qualified relation from `from` to `to`, indexed by the `to` object.
    fn reversed(
        &mut self,
        closure: &SchemaClosure,
        from: ObjectTypeIndex,
        to: ObjectTypeIndex,
        q: &str,
    ) -> &Relation {
        let key = (from, to, q.to_string());
        self.reverse.entry(key).or_insert_with(|| {
            let mut out: Relation = HashMap::new();
            if let Some(r) = closure.relations.get(&(from, to)).and_then(|m| m.get(q)) {
                for (x, ys) in r {
                    for y in ys {
                        out.entry(*y).or_default().insert(*x);
                    }
                }
            }
            out
        })
    }
}
