use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

/// A type with fewer objects than this makes every map into it trivially total while
/// saying nothing.
pub const MIN_TARGET_OBJECTS: usize = 2;

/// Index into the log's object types, sorted by name.
///
/// Sorted, and not left in import order: the importer assigns type indices in an order
/// that varies between runs, so anything tie-breaking on a type index silently changes
/// answer.
pub type ObjectTypeIndex = usize;

/// How a map came to be known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOrigin {
    /// Read off the object-to-object relation, under one qualifier.
    Recorded,
    /// Forced by co-participation of source and target in the same events.
    Coparticipation,
}

impl MapOrigin {
    /// The census label, kept stable because the differential check diffs the printed
    /// census against the Python oracle's.
    pub fn label(&self) -> &'static str {
        match self {
            MapOrigin::Recorded => "recorded",
            MapOrigin::Coparticipation => "coparticip",
        }
    }
}

/// A total map between two object types, with the objects the log refuses.
#[derive(Debug, Clone)]
pub struct Map {
    /// The type the map is defined on.
    pub source: ObjectTypeIndex,
    /// The type it lands in.
    pub target: ObjectTypeIndex,
    /// Which witness established it.
    pub origin: MapOrigin,
    /// Set for recorded maps; a leading `~` marks the reverse of the recorded direction.
    pub qualifier: Option<String>,
    /// The function itself. Reduction needs it; a census only prints its size.
    pub f: HashMap<ObjectIndex, ObjectIndex>,
    /// Number of distinct objects the function reaches.
    pub image: usize,
    /// Source objects for which no image exists.
    pub residual: usize,
    /// Source objects whose image is not forced. Counts against coverage; never guessed.
    pub ambiguous: usize,
}

impl Map {
    /// Share of source objects the map is defined on.
    pub fn coverage(&self) -> f64 {
        let n = self.f.len() + self.residual + self.ambiguous;
        if n == 0 {
            0.0
        } else {
            self.f.len() as f64 / n as f64
        }
    }

    /// One census line. The format is part of the differential check against the Python
    /// oracle, so changing it changes what that check compares.
    pub fn line(&self, types: &[String]) -> String {
        let q = match &self.qualifier {
            Some(q) => format!(" [{q}]"),
            None => String::new(),
        };
        format!(
            "{} -> {}{} ({}) cov={:.4} img={} res={} amb={}",
            types[self.source],
            types[self.target],
            q,
            self.origin.label(),
            self.coverage(),
            self.image,
            self.residual,
            self.ambiguous
        )
    }
}

/// Why a candidate pair carries no map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// The target type has too few objects for a map into it to say anything.
    DegenerateTarget {
        /// Objects the target type has.
        objects: usize,
    },
    /// Two events name disjoint target sets for one source object, so no function exists.
    NonFunctional {
        /// Source objects for which two events named disjoint target sets.
        conflicts: usize,
    },
    /// Too few source objects have an image.
    Untotal {
        /// Coverage reached, in thousandths, so the reason stays comparable.
        coverage_millis: u32,
    },
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::DegenerateTarget { objects } => write!(f, "degenerate target: {objects}"),
            Rejection::NonFunctional { conflicts } => write!(f, "non-functional: {conflicts} conflicts"),
            Rejection::Untotal { coverage_millis } => {
                write!(f, "untotal: coverage {:.3}", *coverage_millis as f64 / 1000.0)
            }
        }
    }
}

/// The structural schema of one log: its object types, the maps witnessed over them, and
/// what it cost to find them.
#[derive(Debug, Clone)]
pub struct StructuralSchema {
    /// Object type names, sorted. Every index in this struct refers to this vector.
    pub types: Vec<String>,
    /// Which type each object has.
    pub type_of: HashMap<ObjectIndex, ObjectTypeIndex>,
    /// Maps read off the object-to-object relation.
    pub recorded: Vec<Map>,
    /// Maps forced by co-participation.
    pub derived: Vec<Map>,
    /// Why each rejected pair was rejected. Only co-participation candidates appear.
    pub rejected: HashMap<(ObjectTypeIndex, ObjectTypeIndex), Rejection>,
    /// Candidate updates performed by the co-participation pass, so a caller can report
    /// the cost per event-to-object tuple rather than an asymptotic bound alone.
    pub candidate_updates: u64,
}

impl StructuralSchema {
    /// Discover the schema in one pass over the events, plus one over the
    /// object-to-object relation.
    pub fn discover(locel: &SlimLinkedOCEL) -> Self {
        let mut types: Vec<String> = locel.get_ob_types().map(str::to_string).collect();
        types.sort();
        let type_ix: HashMap<&str, ObjectTypeIndex> = types
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let type_of: HashMap<ObjectIndex, ObjectTypeIndex> = locel
            .get_all_obs()
            .map(|o| (o, type_ix[o.get_ob_type(locel).as_str()]))
            .collect();

        let recorded = recorded_maps(locel, &type_of, &types);
        let (derived, rejected, candidate_updates) = coparticipation_maps(locel, &type_of, &types);

        Self {
            types,
            type_of,
            recorded,
            derived,
            rejected,
            candidate_updates,
        }
    }

    /// Every map, whatever its witness.
    pub fn maps(&self) -> impl Iterator<Item = &Map> {
        self.recorded.iter().chain(self.derived.iter())
    }

    /// The ordered type pairs some map covers.
    pub fn pairs(&self) -> HashSet<(ObjectTypeIndex, ObjectTypeIndex)> {
        self.maps().map(|m| (m.source, m.target)).collect()
    }

    /// The transitive reduction of the map graph, and the derivation depth it needs.
    pub fn generators(&self) -> (Vec<(ObjectTypeIndex, ObjectTypeIndex)>, usize) {
        generators(&self.pairs())
    }
}

/// Total maps read off the object-to-object relation, one candidate per
/// (source type, target type, qualifier).
///
/// A source object with two distinct targets under the same qualifier makes the pair
/// non-functional and is rejected outright.
///
/// Both orientations of every edge are tested: an edge recorded as `A -[q]-> B` may be
/// functional in either direction while the log records only one of them.
pub fn recorded_maps(
    locel: &SlimLinkedOCEL,
    type_of: &HashMap<ObjectIndex, ObjectTypeIndex>,
    types: &[String],
) -> Vec<Map> {
    // BTreeMap: closing the map set under composition breaks ties between equally sized
    // maps by which it sees first, so hash order here would leak into the reduction.
    let mut by_pair: BTreeMap<
        (ObjectTypeIndex, ObjectTypeIndex, String),
        HashMap<ObjectIndex, HashSet<ObjectIndex>>,
    > = BTreeMap::new();
    let mut objects_of_type: Vec<usize> = vec![0; types.len()];
    for t in type_of.values() {
        objects_of_type[*t] += 1;
    }

    for src in locel.get_all_obs() {
        let st = type_of[&src];
        let ob = src.get_ob(locel);
        for (q, tgt) in &ob.relationships {
            let Some(tt) = type_of.get(tgt) else { continue };
            let qual = locel.qualifier_str(*q).to_string();
            by_pair
                .entry((st, *tt, qual.clone()))
                .or_default()
                .entry(src)
                .or_default()
                .insert(*tgt);
            by_pair
                .entry((*tt, st, format!("~{qual}")))
                .or_default()
                .entry(*tgt)
                .or_default()
                .insert(src);
        }
    }

    let mut out = Vec::new();
    for ((st, tt, qual), rel) in by_pair {
        if objects_of_type[tt] < MIN_TARGET_OBJECTS {
            continue;
        }
        if rel.values().any(|v| v.len() > 1) {
            continue; // not functional
        }
        let f: HashMap<ObjectIndex, ObjectIndex> = rel
            .iter()
            .map(|(s, v)| (*s, *v.iter().next().unwrap()))
            .collect();
        let image: HashSet<&ObjectIndex> = f.values().collect();
        let m = Map {
            source: st,
            target: tt,
            origin: MapOrigin::Recorded,
            qualifier: Some(qual),
            image: image.len(),
            residual: objects_of_type[st].saturating_sub(f.len()),
            ambiguous: 0,
            f,
        };
        out.push(m);
    }
    out
}

/// One image candidate for a (source object, target type) pair.
enum Cand {
    /// Exactly one image survives. Further updates are O(1) membership tests.
    One(ObjectIndex),
    /// Several images still possible.
    Many(HashSet<ObjectIndex>),
    /// Conflict: no function exists for this source object.
    Dead,
}

/// Co-participation maps in ONE pass over the events.
///
/// The per-pair formulation costs `O(|OT|^2)` scans and is hopeless on a log with 120
/// object types, so every (source object, target type) running intersection is maintained
/// together. Cost is `sum_e |obj(e)| * |types(e)|`, and the number of candidate updates is
/// returned so a caller can report operations per event-to-object tuple.
///
/// Returns the maps, the reason each rejected pair was rejected, and the update count.
pub fn coparticipation_maps(
    locel: &SlimLinkedOCEL,
    type_of: &HashMap<ObjectIndex, ObjectTypeIndex>,
    types: &[String],
) -> (
    Vec<Map>,
    HashMap<(ObjectTypeIndex, ObjectTypeIndex), Rejection>,
    u64,
) {
    let mut objects_of_type: Vec<Vec<ObjectIndex>> = vec![Vec::new(); types.len()];
    for o in locel.get_all_obs() {
        objects_of_type[type_of[&o]].push(o);
    }
    let eligible: Vec<bool> = objects_of_type
        .iter()
        .map(|v| v.len() >= MIN_TARGET_OBJECTS)
        .collect();

    let mut cand: HashMap<(ObjectIndex, ObjectTypeIndex), Cand> = HashMap::new();
    let mut updates: u64 = 0;

    // Reused per event so the inner loops allocate nothing.
    let mut per_type: HashMap<ObjectTypeIndex, Vec<ObjectIndex>> = HashMap::new();
    for ev in locel.get_all_evs() {
        per_type.clear();
        // Only recorded tuples witness a map. A `+`-marked tuple was written by expansion
        // from this schema, so letting it witness reads the schema's own output back as
        // evidence. See `Marks::counts_for_discovery` for the object-to-object side.
        let objs: Vec<ObjectIndex> = ev
            .get_e2o_q(locel)
            .filter(|(q, _)| !super::sigil::written(q))
            .map(|(_, o)| *o)
            .collect();
        for o in &objs {
            per_type.entry(type_of[o]).or_default().push(*o);
        }
        for (tt, ts) in &per_type {
            if !eligible[*tt] {
                continue;
            }
            let tset: HashSet<ObjectIndex> = ts.iter().copied().collect();
            for o in &objs {
                if type_of[o] == *tt {
                    continue;
                }
                updates += 1;
                match cand.get_mut(&(*o, *tt)) {
                    None => {
                        let c = if tset.len() == 1 {
                            Cand::One(*ts.first().unwrap())
                        } else {
                            Cand::Many(tset.clone())
                        };
                        cand.insert((*o, *tt), c);
                    }
                    Some(Cand::Dead) => {}
                    Some(Cand::One(x)) => {
                        if !tset.contains(x) {
                            cand.insert((*o, *tt), Cand::Dead);
                        }
                    }
                    Some(Cand::Many(cur)) => {
                        cur.retain(|x| tset.contains(x));
                        match cur.len() {
                            0 => {
                                cand.insert((*o, *tt), Cand::Dead);
                            }
                            1 => {
                                let x = *cur.iter().next().unwrap();
                                cand.insert((*o, *tt), Cand::One(x));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    let mut rejected: HashMap<(ObjectTypeIndex, ObjectTypeIndex), Rejection> = HashMap::new();
    let mut out = Vec::new();
    for st in 0..types.len() {
        for tt in 0..types.len() {
            if st == tt {
                continue;
            }
            if !eligible[tt] {
                rejected.insert(
                    (st, tt),
                    Rejection::DegenerateTarget {
                        objects: objects_of_type[tt].len(),
                    },
                );
                continue;
            }
            let mut image: HashSet<ObjectIndex> = HashSet::new();
            let mut f: HashMap<ObjectIndex, ObjectIndex> = HashMap::new();
            let (mut residual, mut ambiguous) = (0, 0);
            for s in &objects_of_type[st] {
                match cand.get(&(*s, tt)) {
                    None => residual += 1,
                    // Contradicted across events: not in the map's domain, but does not
                    // disqualify the map.
                    Some(Cand::Dead) => residual += 1,
                    Some(Cand::Many(_)) => ambiguous += 1,
                    Some(Cand::One(x)) => {
                        f.insert(*s, *x);
                        image.insert(*x);
                    }
                }
            }
            let m = Map {
                source: st,
                target: tt,
                origin: MapOrigin::Coparticipation,
                qualifier: None,
                image: image.len(),
                residual,
                ambiguous,
                f,
            };
            // No coverage threshold here: the only threshold sits on determination, per
            // cell.
            if m.image > 0 {
                out.push(m);
            } else {
                rejected.insert(
                    (st, tt),
                    Rejection::Untotal {
                        coverage_millis: (m.coverage() * 1000.0).round() as u32,
                    },
                );
            }
        }
    }
    (out, rejected, updates)
}

/// Transitive reduction of the map graph, plus the maximum derivation depth needed to
/// reach every non-generator edge.
pub fn generators(
    pairs: &HashSet<(ObjectTypeIndex, ObjectTypeIndex)>,
) -> (Vec<(ObjectTypeIndex, ObjectTypeIndex)>, usize) {
    let mut succ: HashMap<ObjectTypeIndex, Vec<ObjectTypeIndex>> = HashMap::new();
    for (s, t) in pairs {
        succ.entry(*s).or_default().push(*t);
    }
    let mut gens = Vec::new();
    let mut max_depth = 1;
    for (s, t) in pairs {
        let mut seen: HashMap<ObjectTypeIndex, usize> = HashMap::from([(*s, 0)]);
        let mut frontier = vec![*s];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for u in frontier {
                for v in succ.get(&u).into_iter().flatten() {
                    if (u, *v) == (*s, *t) || seen.contains_key(v) {
                        continue;
                    }
                    seen.insert(*v, seen[&u] + 1);
                    next.push(*v);
                }
            }
            frontier = next;
        }
        match seen.get(t) {
            Some(d) => max_depth = max_depth.max(*d),
            None => gens.push((*s, *t)),
        }
    }
    gens.sort();
    (gens, max_depth)
}
