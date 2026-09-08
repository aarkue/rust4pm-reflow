use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

use super::schema::{ObjectTypeIndex, StructuralSchema};

/// Maximum composition depth when closing the map set.
pub const MAX_COMPOSE_DEPTH: usize = 4;

/// Cap on distinct witnesses kept per type pair. Composition routes multiply, and a pair
/// with more than a handful of genuinely different functions is a sign the schema is not
/// a schema.
pub const MAX_WITNESSES_PER_PAIR: usize = 8;

/// A function between objects of two types.
pub type ObjectFn = HashMap<ObjectIndex, ObjectIndex>;
/// The inverse of a function: each target object to the sources that reach it.
pub type ObjectFibre = HashMap<ObjectIndex, HashSet<ObjectIndex>>;

/// The schema closed under composition, with everything a reconstruction check needs.
///
/// All witnesses for a type pair are kept. A pair can carry several different functions,
/// one per qualifier and one per composition route, and only some of them reconstruct at
/// a given activity. Keeping a single representative would lose reductions and make the
/// result depend on which route is discovered first.
#[derive(Debug, Clone)]
pub struct SchemaClosure {
    /// Object type names, sorted, shared with the schema this was built from.
    pub types: Vec<String>,
    /// Composed functions per ordered type pair.
    pub maps: BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), Vec<ObjectFn>>,
    /// The fibres of those functions, keyed by the same pair as the function.
    pub fibres: BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), Vec<ObjectFibre>>,
    /// Recorded object-to-object relations per type pair, split by qualifier, because a
    /// cell may need a union of qualified maps that no single function provides.
    pub relations:
        BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), BTreeMap<String, ObjectFibre>>,
    /// `reach[s][t]`: some composed map takes `s` to `t`. The refinement preorder.
    pub reach: Vec<Vec<bool>>,
    /// Representative of each refinement class: the lexicographically least name among
    /// mutually reachable types. Without this tie-break the maximum is not unique whenever
    /// two types are mutually total.
    pub rep: Vec<ObjectTypeIndex>,
}

impl SchemaClosure {
    /// Close a discovered schema under composition and index it for reconstruction
    /// checks.
    pub fn build(locel: &SlimLinkedOCEL, schema: &StructuralSchema) -> Self {
        let maps = compose_closure(schema);
        let fibres = maps
            .iter()
            .map(|(pair, fs)| {
                let inverted = fs
                    .iter()
                    .map(|f| {
                        let mut fib: ObjectFibre = HashMap::new();
                        for (x, y) in f {
                            fib.entry(*y).or_default().insert(*x);
                        }
                        fib
                    })
                    .collect();
                (*pair, inverted)
            })
            .collect();
        let relations = recorded_relations(locel, &schema.type_of);

        let n = schema.types.len();
        let mut reach = vec![vec![false; n]; n];
        for (s, t) in maps.keys() {
            reach[*s][*t] = true;
        }
        for k in 0..n {
            for i in 0..n {
                if reach[i][k] {
                    for j in 0..n {
                        if reach[k][j] {
                            reach[i][j] = true;
                        }
                    }
                }
            }
        }
        let types = schema.types.clone();
        let rep: Vec<ObjectTypeIndex> = (0..n)
            .map(|t| {
                (0..n)
                    .filter(|s| *s == t || (reach[t][*s] && reach[*s][t]))
                    .min_by_key(|s| &types[*s])
                    .unwrap()
            })
            .collect();

        Self {
            types,
            maps,
            fibres,
            relations,
            reach,
            rep,
        }
    }

    /// Is `s` strictly finer than `t`: reachable one way and not the other.
    pub fn strictly_finer(&self, s: ObjectTypeIndex, t: ObjectTypeIndex) -> bool {
        self.reach[s][t] && !self.reach[t][s]
    }
}

/// Close the discovered maps under composition, keyed by (source type, target type).
///
/// `BTreeMap`, not `HashMap`: which composition route reaches a type pair first decides
/// the witness stored for it, so iterating in hash order would make the reduction
/// nondeterministic across runs.
pub fn compose_closure(
    schema: &StructuralSchema,
) -> BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), Vec<ObjectFn>> {
    let mut by_pair: BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), Vec<ObjectFn>> = BTreeMap::new();
    for m in schema.maps() {
        by_pair.entry((m.source, m.target)).or_default().push(m.f.clone());
    }
    for _ in 0..MAX_COMPOSE_DEPTH {
        let mut added = Vec::new();
        for ((s, t), fs) in &by_pair {
            for ((t2, u), gs) in &by_pair {
                if t != t2 || s == u {
                    continue;
                }
                for f in fs {
                    for g in gs {
                        let composed: ObjectFn = f
                            .iter()
                            .filter_map(|(x, y)| g.get(y).map(|z| (*x, *z)))
                            .collect();
                        if !composed.is_empty() {
                            added.push(((*s, *u), composed));
                        }
                    }
                }
            }
        }
        let before: usize = by_pair.values().map(Vec::len).sum();
        for (k, v) in added {
            let e = by_pair.entry(k).or_default();
            // Composition routes often agree; without deduplication the witness lists
            // grow exponentially in the depth bound.
            if !e.iter().any(|x| *x == v) && e.len() < MAX_WITNESSES_PER_PAIR {
                e.push(v);
            }
        }
        if by_pair.values().map(Vec::len).sum::<usize>() == before {
            break;
        }
    }
    by_pair
}

/// Recorded object-to-object relations as a relation per type pair, split by qualifier.
///
/// Split by qualifier because a relation that is not functional unqualified can be a
/// union of total functions, one per qualifier, and a cell may need that union.
pub fn recorded_relations(
    locel: &SlimLinkedOCEL,
    type_of: &HashMap<ObjectIndex, ObjectTypeIndex>,
) -> BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), BTreeMap<String, ObjectFibre>> {
    let mut rel: BTreeMap<(ObjectTypeIndex, ObjectTypeIndex), BTreeMap<String, ObjectFibre>> =
        BTreeMap::new();
    for src in locel.get_all_obs() {
        let st = type_of[&src];
        for (q, tgt) in &src.get_ob(locel).relationships {
            if let Some(tt) = type_of.get(tgt) {
                rel.entry((st, *tt))
                    .or_default()
                    .entry(locel.qualifier_str(*q).to_string())
                    .or_default()
                    .entry(src)
                    .or_default()
                    .insert(*tgt);
            }
        }
    }
    rel
}
