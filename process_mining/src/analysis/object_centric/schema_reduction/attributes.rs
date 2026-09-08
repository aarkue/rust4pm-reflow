use std::collections::HashSet;

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    arcs::ActivityIndexing,
    cells::{ActivityIndex, CellGrid},
    closure::SchemaClosure,
    facts::Facts,
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// An object type that could be an attribute instead.
///
/// Where a type is determined by another everywhere it occurs, the log can carry it as an
/// attribute of its determiner. An implied cell is recovered given the schema; an attribute
/// is recovered from the flow projection alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Demotion {
    /// The type that would become an attribute.
    pub object_type: ObjectTypeIndex,
    /// The type that would carry it. Every object of this type has exactly one value.
    pub carrier: ObjectTypeIndex,
    /// Distinct values the attribute would take: the image of the map.
    pub distinct_values: usize,
    /// Objects that leave the log as objects.
    pub objects_removed: usize,
    /// Participations that leave with them.
    pub participations_removed: usize,
    /// Activities the type occurred at.
    pub activities: usize,
    /// Ordering facts this type asserts in the full model.
    ///
    /// An attribute has no lifecycle to draw, so the move loses exactly these facts.
    pub ordering_facts: usize,
    /// Types this one determines that its carrier cannot reach. The move would put them
    /// out of reach, so a non-empty list means it is not free.
    pub blocked: Vec<ObjectTypeIndex>,
}

/// Every type that could be an attribute of another, with what the move would lose.
///
/// Three conditions:
///
/// 1. a total map `carrier -> type` exists, so every carrier object has exactly one value;
/// 2. the type occurs only at activities where the carrier occurs, and
/// 3. at each of them the carrier's cell determines the type's cell.
///
/// Condition 3 is the reduction's own test: a type that can become an attribute of `S` is
/// exactly a type `S` cuts everywhere.
pub fn demotable_types(
    schema: &StructuralSchema,
    closure: &SchemaClosure,
    grid: &CellGrid,
    facts: &Facts,
) -> Vec<Demotion> {
    let n_types = schema.types.len();
    let mut objects_of_type = vec![0usize; n_types];
    for t in schema.type_of.values() {
        objects_of_type[*t] += 1;
    }

    let mut out = Vec::new();
    for t in 0..n_types {
        let cells: Vec<usize> = (0..grid.activities.len())
            .filter(|a| grid.cells.contains(&(*a, t)))
            .collect();
        if cells.is_empty() {
            continue;
        }
        // A total map into `t` is what makes the value single.
        let carriers: HashSet<ObjectTypeIndex> = schema
            .maps()
            .filter(|m| m.target == t && m.source != t)
            .map(|m| m.source)
            .collect();

        for s in carriers {
            let determines_everywhere = cells.iter().all(|a| {
                let here = &grid.per_activity[*a];
                match (here.slot(s), here.slot(t)) {
                    (Some(i), Some(j)) => here.recon[i][j].is_some(),
                    _ => false,
                }
            });
            if !determines_everywhere {
                continue;
            }
            let image = schema
                .maps()
                .filter(|m| m.source == s && m.target == t)
                .map(|m| m.image)
                .max()
                .unwrap_or(0);
            let participations: usize = cells
                .iter()
                .map(|a| {
                    let here = &grid.per_activity[*a];
                    here.slot(t).map_or(0, |j| here.counts[j])
                })
                .sum();
            let blocked: Vec<ObjectTypeIndex> = (0..n_types)
                .filter(|u| *u != t && *u != s && closure.reach[t][*u] && !closure.reach[s][*u])
                .collect();

            out.push(Demotion {
                ordering_facts: facts.ordering.iter().filter(|(ft, _, _)| *ft == t).count(),
                object_type: t,
                carrier: s,
                distinct_values: image,
                objects_removed: objects_of_type[t],
                participations_removed: participations,
                activities: cells.len(),
                blocked,
            });
        }
    }
    out.sort_by_key(|d| {
        (
            std::cmp::Reverse(d.participations_removed),
            d.object_type,
            d.carrier,
        )
    });
    out
}

/// A cell that could be an attribute of the event instead of a participation.
///
/// Where every event of an activity names at most one object of a type, that participation
/// is a single value. This needs no schema: the identifier is written into the event, so a
/// cell no relation can put back can still become an attribute. What is lost is the type's
/// flow, since
/// an attribute has no lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventAttribute {
    /// The activity whose events would carry the attribute.
    pub activity: ActivityIndex,
    /// The type that would stop being a participation here.
    pub object_type: ObjectTypeIndex,
    /// Events of the activity that name an object of the type.
    pub events: usize,
    /// Distinct objects named, i.e. distinct values the attribute would take.
    pub distinct_values: usize,
    /// Participations the move removes. Equal to `events`, since the cell is single-valued.
    pub participations_removed: usize,
    /// Whether any rule reconstructs this cell from another kept at the same activity. When
    /// nothing does, the cell is irreducible and this is the only move available for it.
    pub reducible: bool,
    /// Ordering facts of this type that this activity is an endpoint of. Taking the cell out
    /// of the flow layer takes the activity out of the type's flow and loses exactly these facts.
    pub ordering_facts: usize,
}

/// Cells that could be an event attribute: those naming at most one object per event.
///
/// One pass over the events. A cell where some event names two objects of the type is not
/// reported.
pub fn event_attribute_cells(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    grid: &CellGrid,
    acts: &ActivityIndexing,
    facts: &Facts,
) -> Vec<EventAttribute> {
    let n_types = schema.types.len();
    let mut naming = vec![vec![0usize; n_types]; grid.activities.len()];
    let mut multi = vec![vec![false; n_types]; grid.activities.len()];
    let mut values: std::collections::HashMap<(ActivityIndex, ObjectTypeIndex), HashSet<ObjectIndex>> =
        Default::default();

    for e in locel.get_all_evs() {
        let a = acts.act_of[e.get_ev(locel).event_type];
        let mut here = vec![0usize; n_types];
        for o in e.get_e2o(locel) {
            let t = schema.type_of[o];
            here[t] += 1;
            values.entry((a, t)).or_default().insert(*o);
        }
        for (t, n) in here.iter().enumerate() {
            if *n == 0 {
                continue;
            }
            naming[a][t] += 1;
            if *n > 1 {
                multi[a][t] = true;
            }
        }
    }

    let mut out = Vec::new();
    for (a, t) in &grid.cells {
        if multi[*a][*t] {
            continue;
        }
        let cells = &grid.per_activity[*a];
        let reducible = cells.slot(*t).is_some_and(|j| {
            (0..cells.present.len()).any(|i| i != j && cells.recon[i][j].is_some())
        });
        out.push(EventAttribute {
            ordering_facts: facts
                .ordering
                .iter()
                .filter(|(ft, x, y)| ft == t && (x == a || y == a))
                .count(),
            activity: *a,
            object_type: *t,
            events: naming[*a][*t],
            distinct_values: values.get(&(*a, *t)).map_or(0, HashSet::len),
            participations_removed: naming[*a][*t],
            reducible,
        });
    }
    out.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.participations_removed),
            c.activity,
            c.object_type,
        )
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::{CellGrid, SchemaClosure};
    use crate::core::chrono::DateTime;
    use crate::core::event_data::object_centric::linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL};

    /// Two items of one order, each of a product. `products` is named only where `items` is
    /// and is a function of it.
    fn toy() -> SlimLinkedOCEL {
        let mut ocel = SlimLinkedOCEL::new();
        for t in ["items", "orders", "products"] {
            ocel.add_object_type(t, Vec::new());
        }
        for a in ["place", "pick"] {
            ocel.add_event_type(a, Vec::new());
        }
        let o = ocel
            .add_object("orders", Some("o1".into()), Vec::new(), Vec::new())
            .unwrap();
        let p1 = ocel
            .add_object("products", Some("p1".into()), Vec::new(), Vec::new())
            .unwrap();
        let p2 = ocel
            .add_object("products", Some("p2".into()), Vec::new(), Vec::new())
            .unwrap();
        let t = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        let mut clock = 1_600_000_000_000i64;
        for (n, p) in [p1, p2].into_iter().enumerate() {
            let i = ocel
                .add_object(
                    "items",
                    Some(format!("i{n}")),
                    Vec::new(),
                    vec![("of".into(), o), ("is a".into(), p)],
                )
                .unwrap();
            for act in ["place", "pick"] {
                clock += 1000;
                ocel.add_event(
                    act,
                    t(clock),
                    Some(format!("{act}-{n}")),
                    Vec::new(),
                    vec![("item".into(), i), ("product".into(), p)],
                );
            }
        }
        ocel
    }

    #[test]
    fn a_type_determined_everywhere_is_an_attribute_of_its_determiner() {
        let ocel = toy();
        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let grid = CellGrid::build(&ocel, &schema, &closure);
        let found = demotable_types(&schema, &closure, &grid, &Facts::default());

        let named: Vec<(String, String)> = found
            .iter()
            .map(|d| {
                (
                    schema.types[d.object_type].clone(),
                    schema.types[d.carrier].clone(),
                )
            })
            .collect();
        assert!(
            named.contains(&("products".to_string(), "items".to_string())),
            "products is a function of items and named only where items is: {named:?}"
        );
        let d = found
            .iter()
            .find(|d| schema.types[d.object_type] == "products")
            .unwrap();
        assert_eq!(d.distinct_values, 2);
        assert_eq!(d.objects_removed, 2);
        assert_eq!(d.participations_removed, 4);
    }

    #[test]
    fn a_type_with_a_life_of_its_own_is_not_demotable() {
        let mut ocel = toy();
        // `products` at an activity `items` never touches.
        ocel.add_event_type("restock", Vec::new());
        let products: Vec<_> = ocel
            .get_all_obs()
            .filter(|o| o.get_ob_type(&ocel).as_str() == "products")
            .collect();
        let t = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        for (n, p) in products.iter().enumerate() {
            ocel.add_event(
                "restock",
                t(1_700_000_000_000 + n as i64 * 1000),
                Some(format!("restock-{n}")),
                Vec::new(),
                vec![("product".into(), *p)],
            );
        }
        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let grid = CellGrid::build(&ocel, &schema, &closure);
        let found = demotable_types(&schema, &closure, &grid, &Facts::default());
        assert!(
            !found.iter().any(|d| schema.types[d.object_type] == "products"),
            "products has behaviour of its own and must keep its type"
        );
    }
}
