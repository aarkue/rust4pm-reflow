use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    closure::ObjectFibre,
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// A qualified object-to-object relation, named by its two types and its qualifier.
pub type RelationKey = (ObjectTypeIndex, ObjectTypeIndex, String);

/// One relation dropped because the composition of two others reproduces it exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Implied {
    /// The relation removed.
    pub relation: RelationKey,
    /// The two relations it composes from, each with the direction it is read in.
    pub via: [(RelationKey, bool); 2],
    /// The type the composition passes through.
    pub through: ObjectTypeIndex,
    /// Object-to-object tuples the drop removes.
    pub edges: usize,
}

impl Implied {
    /// One census line.
    pub fn line(&self, types: &[String]) -> String {
        let name = |k: &RelationKey, rev: bool| {
            if rev {
                format!("{} <-[{}]- {}", types[k.1], k.2, types[k.0])
            } else {
                format!("{} -[{}]-> {}", types[k.0], k.2, types[k.1])
            }
        };
        format!(
            "{} = {} then {}   ({} edges, through {})",
            name(&self.relation, false),
            name(&self.via[0].0, self.via[0].1),
            name(&self.via[1].0, self.via[1].1),
            self.edges,
            types[self.through]
        )
    }
}

/// What reducing the object-to-object relation removes.
///
/// No event, object or participation changes, and the dropped edges are recomputed from
/// the ones that stay. This is unrelated to constraint-level reduction of declarative
/// models, which removes constraints implied by other constraints.
#[derive(Debug, Clone, Default)]
pub struct O2OReduction {
    /// The relations dropped, in the order they were dropped.
    pub implied: Vec<Implied>,
    /// Object-to-object tuples the log carries.
    pub edges_total: usize,
    /// Object-to-object tuples the reduction removes.
    pub edges_dropped: usize,
    /// Qualified relations the log carries.
    pub relations_total: usize,
    /// Composition checks the search ran.
    pub checks: u64,
}

/// Composition checks above which the search stops.
///
/// The search is cubic in the type count and quadratic in the qualifier count per pair.
pub const O2O_REDUCTION_BUDGET: u64 = 50_000_000;

/// Find the qualified relations that are exactly the composition of two others.
///
/// Exactly, not merely contained: a composition that covers the relation would restore more
/// edges than were dropped. Both orientations of every surviving relation are available as
/// composition steps, since an object-to-object edge is readable either way.
///
/// A dropped relation leaves the surviving set at once, so two relations that imply each
/// other cannot both go.
pub fn reduce_o2o(locel: &SlimLinkedOCEL, schema: &StructuralSchema) -> O2OReduction {
    let rels = qualified_relations(locel, &schema.type_of);
    let mut out = O2OReduction {
        relations_total: rels.len(),
        edges_total: rels.values().map(|r| r.values().map(HashSet::len).sum::<usize>()).sum(),
        ..Default::default()
    };
    let mut alive: HashSet<RelationKey> = rels.keys().cloned().collect();

    let keys: Vec<RelationKey> = rels.keys().cloned().collect();
    for k in &keys {
        if !alive.contains(k) {
            continue;
        }
        let (s, t, _) = k;
        let target = &rels[k];
        let mut found: Option<Implied> = None;
        'outer: for u in 0..schema.types.len() {
            if u == *s || u == *t {
                continue;
            }
            let first: Vec<(RelationKey, bool)> = oriented(&alive, *s, u, k);
            if first.is_empty() {
                continue;
            }
            let second: Vec<(RelationKey, bool)> = oriented(&alive, u, *t, k);
            for a in &first {
                for b in &second {
                    out.checks += 1;
                    if out.checks > O2O_REDUCTION_BUDGET {
                        return out;
                    }
                    let composed = compose(&rels, a, b);
                    if same(&composed, target) {
                        found = Some(Implied {
                            relation: k.clone(),
                            via: [a.clone(), b.clone()],
                            through: u,
                            edges: target.values().map(HashSet::len).sum(),
                        });
                        break 'outer;
                    }
                }
            }
        }
        if let Some(f) = found {
            alive.remove(k);
            out.edges_dropped += f.edges;
            out.implied.push(f);
        }
    }
    out
}

/// Apply a reduction: drop the implied relations' edges and nothing else.
pub fn apply_o2o_reduction(locel: &SlimLinkedOCEL, reduction: &O2OReduction) -> SlimLinkedOCEL {
    let mut out = locel.clone();
    let doomed: HashSet<(ObjectTypeIndex, ObjectTypeIndex, u32)> = reduction
        .implied
        .iter()
        .filter_map(|i| {
            out.qualifier_idx_of(&i.relation.2)
                .map(|q| (i.relation.0, i.relation.1, q.into_inner()))
        })
        .collect();
    if doomed.is_empty() {
        return out;
    }
    let ob_ix = type_indexing(&out);
    out.retain_o2o_by(|s, t, q| !doomed.contains(&(ob_ix[s], ob_ix[t], q.into_inner())));
    out
}

/// Put the dropped relations back by recomputing them.
///
/// Last dropped, first restored: a later drop may compose through an earlier one, so
/// restoring in drop order would recompute against a relation that is not back yet.
pub fn expand_o2o(
    reduced: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    reduction: &O2OReduction,
) -> SlimLinkedOCEL {
    let mut out = reduced.clone();
    // Read the surviving relations once and grow them as each is restored, instead of
    // re-reading the whole object-to-object relation per implied relation.
    let mut rels = qualified_relations(&out, &schema.type_of);
    for imp in reduction.implied.iter().rev() {
        let composed = compose(&rels, &imp.via[0], &imp.via[1]);
        for (x, ys) in &composed {
            for y in ys {
                out.add_o2o(*x, *y, imp.relation.2.clone());
            }
        }
        rels.insert(imp.relation.clone(), composed);
    }
    out
}

/// Native object type index to the sorted [`ObjectTypeIndex`].
fn type_indexing(locel: &SlimLinkedOCEL) -> Vec<ObjectTypeIndex> {
    let mut sorted: Vec<String> = locel.get_ob_types().map(str::to_string).collect();
    sorted.sort();
    let ix: HashMap<&str, usize> = sorted.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    locel.get_ob_types().map(|t| ix[t]).collect()
}

/// Every qualified object-to-object relation the log records.
fn qualified_relations(
    locel: &SlimLinkedOCEL,
    type_of: &HashMap<ObjectIndex, ObjectTypeIndex>,
) -> BTreeMap<RelationKey, ObjectFibre> {
    let mut out: BTreeMap<RelationKey, ObjectFibre> = BTreeMap::new();
    for src in locel.get_all_obs() {
        let Some(st) = type_of.get(&src) else { continue };
        for (q, tgt) in src.get_o2o_q(locel) {
            let Some(tt) = type_of.get(tgt) else { continue };
            out.entry((*st, *tt, q.to_string()))
                .or_default()
                .entry(src)
                .or_default()
                .insert(*tgt);
        }
    }
    out
}

/// Surviving relations that read from `from` to `to`, in either recorded direction, other
/// than `skip` itself.
fn oriented(
    alive: &HashSet<RelationKey>,
    from: ObjectTypeIndex,
    to: ObjectTypeIndex,
    skip: &RelationKey,
) -> Vec<(RelationKey, bool)> {
    alive
        .iter()
        .filter(|k| *k != skip)
        .filter_map(|k| {
            if (k.0, k.1) == (from, to) {
                Some((k.clone(), false))
            } else if (k.1, k.0) == (from, to) {
                Some((k.clone(), true))
            } else {
                None
            }
        })
        .collect()
}

/// One oriented relation as a map from its reading direction's source.
fn oriented_map(rels: &BTreeMap<RelationKey, ObjectFibre>, step: &(RelationKey, bool)) -> ObjectFibre {
    let r = &rels[&step.0];
    if !step.1 {
        return r.clone();
    }
    let mut out: ObjectFibre = HashMap::new();
    for (x, ys) in r {
        for y in ys {
            out.entry(*y).or_default().insert(*x);
        }
    }
    out
}

/// The composition of two oriented relations.
fn compose(
    rels: &BTreeMap<RelationKey, ObjectFibre>,
    a: &(RelationKey, bool),
    b: &(RelationKey, bool),
) -> ObjectFibre {
    let first = oriented_map(rels, a);
    let second = oriented_map(rels, b);
    let mut out: ObjectFibre = HashMap::new();
    for (x, us) in &first {
        for u in us {
            for y in second.get(u).into_iter().flatten() {
                out.entry(*x).or_default().insert(*y);
            }
        }
    }
    out
}

/// Whether two relations hold the same pairs.
fn same(a: &ObjectFibre, b: &ObjectFibre) -> bool {
    a.len() == b.len() && a.iter().all(|(x, ys)| b.get(x) == Some(ys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::{canonical::Fingerprint, StructuralSchema};

    /// Items belong to orders, orders to customers, and the log also records the composite.
    fn chain() -> SlimLinkedOCEL {
        let mut ocel = SlimLinkedOCEL::new();
        for t in ["items", "orders", "customers"] {
            ocel.add_object_type(t, Vec::new());
        }
        let c: Vec<_> = (0..2)
            .map(|i| {
                ocel.add_object("customers", Some(format!("c{i}")), Vec::new(), Vec::new())
                    .unwrap()
            })
            .collect();
        let o: Vec<_> = (0..2)
            .map(|i| {
                ocel.add_object(
                    "orders",
                    Some(format!("o{i}")),
                    Vec::new(),
                    vec![("of".into(), c[i])],
                )
                .unwrap()
            })
            .collect();
        for i in 0..4 {
            ocel.add_object(
                "items",
                Some(format!("i{i}")),
                Vec::new(),
                vec![("in".into(), o[i / 2]), ("for".into(), c[i / 2])],
            )
            .unwrap();
        }
        ocel
    }

    #[test]
    fn a_relation_two_others_compose_to_is_dropped_and_recomputed() {
        let ocel = chain();
        let before = Fingerprint::build(&ocel);
        let schema = StructuralSchema::discover(&ocel);
        let r = reduce_o2o(&ocel, &schema);
        assert_eq!(r.implied.len(), 1, "{:?}", r.implied);
        assert_eq!(r.implied[0].relation.2, "for");
        assert_eq!(r.edges_dropped, 4);

        let reduced = apply_o2o_reduction(&ocel, &r);
        let o2o = |l: &SlimLinkedOCEL| l.get_all_obs().map(|o| o.get_o2o(l).count()).sum::<usize>();
        assert_eq!(o2o(&reduced), o2o(&ocel) - 4);
        assert_eq!(reduced.get_all_evs().count(), ocel.get_all_evs().count());

        let back = expand_o2o(&reduced, &schema, &r);
        let d = Fingerprint::build(&back).diff(&before, 5);
        assert!(d.is_empty(), "{}", d.summary());
    }

    #[test]
    fn a_relation_the_composition_only_covers_is_kept() {
        let mut ocel = chain();
        // One more customer edge no composition through `orders` reproduces.
        let items: Vec<_> = ocel
            .get_all_obs()
            .filter(|o| o.get_ob_type(&ocel) == "items")
            .collect();
        let cust: Vec<_> = ocel
            .get_all_obs()
            .filter(|o| o.get_ob_type(&ocel) == "customers")
            .collect();
        ocel.add_o2o(items[0], cust[1], "for".to_string());
        let schema = StructuralSchema::discover(&ocel);
        assert!(reduce_o2o(&ocel, &schema).implied.is_empty());
    }
}
