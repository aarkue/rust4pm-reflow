//! Discovery of object-centric Petri nets.
//!
//! Implements the discovery procedure of van der Aalst and Berti, "Discovering Object-Centric
//! Petri Nets" (Fundamenta Informaticae 175, 2020): flatten the object-centric event log on each
//! object type, discover a Petri net per type, stitch those together on their shared activities,
//! and decide which arcs are variable.

use std::collections::{BTreeSet, HashMap, HashSet};

use rayon::prelude::*;

use crate::core::event_data::case_centric::EventLogClassifier;
use crate::core::event_data::object_centric::linked_ocel::LinkedOCELAccess;
use crate::core::event_data::object_centric::utils::flatten::flatten_ocel_on;
use crate::core::process_models::object_centric::ocpn::ObjectCentricPetriNet;
use crate::discovery::case_centric::inductive_miner::{inductive_miner, InductiveMinerOptions};

/// Which object types to discover a net for.
///
/// Each type's net comes from that type's flattening alone and its variable arcs are decided per
/// activity and type, so leaving a type out changes nothing about the others: it only leaves out
/// its places, its silent transitions, and the activities no remaining type sees.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ObjectTypeFilter {
    /// Every object type of the log.
    #[default]
    All,
    /// Only the named object types.
    Only(BTreeSet<String>),
    /// Every object type but the named ones.
    Except(BTreeSet<String>),
}

impl ObjectTypeFilter {
    /// Discovers only the given object types.
    pub fn only<S: Into<String>>(object_types: impl IntoIterator<Item = S>) -> Self {
        Self::Only(object_types.into_iter().map(Into::into).collect())
    }

    /// Discovers every object type but the given ones.
    pub fn except<S: Into<String>>(object_types: impl IntoIterator<Item = S>) -> Self {
        Self::Except(object_types.into_iter().map(Into::into).collect())
    }

    /// Whether a net is discovered for the given object type.
    pub fn includes(&self, object_type: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(types) => types.contains(object_type),
            Self::Except(types) => !types.contains(object_type),
        }
    }
}

/// Settings for object-centric Petri net discovery.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectCentricDiscoveryOptions {
    /// How the Petri net of each object type is discovered from its flattened log.
    pub inductive_miner: InductiveMinerOptions,
    /// Which object types to discover a net for.
    pub object_types: ObjectTypeFilter,
    /// Fraction of an activity's executions that may deviate from "exactly one object of this
    /// type" before the arc is still called normal.
    ///
    /// `0.0`, the default, is the strict reading. A small tolerance keeps data-quality noise from
    /// turning an arc variable on its own, such as the 10 of 1966 `Load to Vehicle` events relating
    /// to no vehicle in the container-logistics log. `PM4Py` applies `0.2` implicitly, as its
    /// `double_arc_threshold` of `0.8`.
    pub variable_arc_tolerance: f64,
}

impl ObjectCentricDiscoveryOptions {
    /// Discovery with the given Inductive Miner settings, every object type, and no tolerance for
    /// variable arcs.
    pub fn new(inductive_miner: InductiveMinerOptions) -> Self {
        Self {
            inductive_miner,
            ..Self::default()
        }
    }

    /// Sets which [object types](Self::object_types) to discover.
    pub fn with_object_types(self, object_types: ObjectTypeFilter) -> Self {
        Self {
            object_types,
            ..self
        }
    }

    /// Sets the [`variable_arc_tolerance`](Self::variable_arc_tolerance), clamped to `[0, 1]`.
    pub fn with_variable_arc_tolerance(self, tolerance: f64) -> Self {
        Self {
            variable_arc_tolerance: tolerance.clamp(0.0, 1.0),
            ..self
        }
    }
}

/// Discovers an [`ObjectCentricPetriNet`] from an object-centric event log.
///
/// Flattens the log on every object type, one case per object, mines a Petri net from each with
/// the Inductive Miner, and decides the [variable arcs](discover_variable_arcs). Stitching needs
/// no work: transitions carrying the same activity label are one transition of the object-centric
/// net, which is how [`ObjectCentricPetriNet`] is read.
///
/// The object types are independent, so they are mined in parallel and
/// [`options.object_types`](ObjectCentricDiscoveryOptions::object_types) can leave some out
/// without changing the rest.
pub fn discover_ocpn<'a, O>(
    ocel: &'a O,
    options: ObjectCentricDiscoveryOptions,
) -> ObjectCentricPetriNet
where
    O: LinkedOCELAccess<'a> + Sync,
    O::ObjectRepr: Eq + std::hash::Hash,
{
    let object_types: Vec<String> = ocel
        .get_ob_types()
        .filter(|object_type| options.object_types.includes(object_type))
        .map(str::to_string)
        .collect();

    let nets = object_types
        .par_iter()
        .map(|object_type| {
            let mut flattened = flatten_ocel_on(ocel, object_type);
            // Objects taking part in no event carry no behaviour. Keeping them would put an
            // empty trace in the log, which the miner reads as "this object type may be skipped"
            // and makes the whole net optional. The flattened log of the paper is derived from
            // the event-to-object relations, so those objects have no case.
            flattened.traces.retain(|trace| !trace.events.is_empty());

            let tree = inductive_miner(
                &flattened,
                &EventLogClassifier::default(),
                options.inductive_miner,
            );
            (object_type.clone(), tree.to_petri_net())
        })
        .collect();

    let mut variable_arcs = discover_variable_arcs(ocel, options.variable_arc_tolerance);
    variable_arcs.retain(|object_type, _| options.object_types.includes(object_type));

    ObjectCentricPetriNet {
        nets,
        variable_arcs,
    }
}

/// Determines, per object type, which activities have variable arcs.
///
/// A normal arc moves exactly one token, so an arc is variable as soon as one execution of the
/// activity involves other than exactly one object of the type: several at once, or none.
/// Flattening hides both, an event touching three items being indistinguishable from three
/// executions and one touching none not appearing at all, so this reads the object-centric log.
/// `tolerance` allows a fraction of the executions to deviate anyway.
///
/// Types without a variable arc are absent from the result, as are activities never relating to
/// the type, which have no arc to decide.
pub fn discover_variable_arcs<'a, O>(
    ocel: &'a O,
    tolerance: f64,
) -> HashMap<String, HashSet<String>>
where
    O: LinkedOCELAccess<'a>,
    O::ObjectRepr: Eq + std::hash::Hash,
{
    /// How the events of one activity relate to one object type.
    #[derive(Default)]
    struct Involvement {
        /// Events of the activity that relate to exactly one object of the type.
        with_exactly_one: u64,
        /// Events of the activity that relate to two or more objects of the type.
        with_several: u64,
    }

    let mut activity_occurrences: HashMap<&str, u64> = HashMap::new();
    let mut involvement: HashMap<(&str, &str), Involvement> = HashMap::new();
    let mut objects_per_type: HashMap<&str, HashSet<&O::ObjectRepr>> = HashMap::new();

    for event in ocel.get_all_evs() {
        objects_per_type.clear();
        for (_qualifier, object) in ocel.get_e2o(&event) {
            // Counted per distinct object: the same object may be related to an event under
            // several qualifiers, which is one object, not two.
            objects_per_type
                .entry(ocel.get_ob_type_of(object))
                .or_default()
                .insert(object);
        }

        // An event related to no object appears in no flattened log and takes part in no per-type
        // net, so it says nothing about an arc. The container-logistics log has 41 of them, some
        // because their object references point at objects that are not in the log, and counting
        // them would make arcs look variable because of a data-quality problem.
        if objects_per_type.is_empty() {
            continue;
        }

        let activity = ocel.get_ev_type_of(&event);
        *activity_occurrences.entry(activity).or_default() += 1;

        for (object_type, objects) in &objects_per_type {
            let entry = involvement.entry((activity, object_type)).or_default();
            if objects.len() == 1 {
                entry.with_exactly_one += 1;
            } else {
                entry.with_several += 1;
            }
        }
    }

    let mut variable: HashMap<String, HashSet<String>> = HashMap::new();
    for ((activity, object_type), counts) in involvement {
        let occurrences = activity_occurrences[activity];
        // Executions that did not involve exactly one object of the type: either several at once,
        // or none at all.
        let deviating = occurrences - counts.with_exactly_one;
        if deviating as f64 > tolerance * occurrences as f64 {
            variable
                .entry(object_type.to_string())
                .or_default()
                .insert(activity.to_string());
        }
    }

    variable
}

#[cfg(test)]
mod test_discover_ocpn {
    use super::*;
    use crate::core::event_data::object_centric::linked_ocel::IndexLinkedOCEL;
    use crate::core::event_data::object_centric::ocel_struct::{
        OCELEvent, OCELObject, OCELRelationship, OCELType,
    };
    use crate::OCEL;

    fn object(id: &str, object_type: &str) -> OCELObject {
        OCELObject {
            id: id.to_string(),
            object_type: object_type.to_string(),
            attributes: vec![],
            relationships: vec![],
        }
    }

    fn event(id: &str, event_type: &str, objects: &[&str]) -> OCELEvent {
        OCELEvent {
            id: id.to_string(),
            event_type: event_type.to_string(),
            time: chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap(),
            attributes: vec![],
            relationships: objects
                .iter()
                .map(|o| OCELRelationship {
                    object_id: (*o).to_string(),
                    qualifier: "rel".to_string(),
                })
                .collect(),
        }
    }

    fn types(names: &[&str]) -> Vec<OCELType> {
        names
            .iter()
            .map(|name| OCELType {
                name: name.to_string(),
                attributes: vec![],
            })
            .collect()
    }

    /// One order with two items, picked individually and then paid. "place order" touches both
    /// items at once, everything else one object at a time.
    fn example_ocel() -> OCEL {
        OCEL {
            event_types: types(&["place order", "pick", "pay"]),
            object_types: types(&["order", "item"]),
            objects: vec![
                object("o1", "order"),
                object("i1", "item"),
                object("i2", "item"),
            ],
            events: vec![
                event("e1", "place order", &["o1", "i1", "i2"]),
                event("e2", "pick", &["i1"]),
                event("e3", "pick", &["i2"]),
                event("e4", "pay", &["o1"]),
            ],
        }
    }

    #[test]
    fn test_variable_arcs_follow_the_number_of_objects_per_event() {
        let locel = IndexLinkedOCEL::from(example_ocel());
        let variable = discover_variable_arcs(&locel, 0.0);

        // "place order" relates to two items at once but to a single order.
        assert!(variable["item"].contains("place order"));
        assert!(!variable.contains_key("order"));
        assert!(!variable["item"].contains("pick"));
    }

    /// An event may relate to the same object under several qualifiers, which is one occurrence.
    /// Counting it twice would invent a repetition and make the arc look variable.
    #[test]
    fn test_repeated_relations_to_the_same_object_count_once() {
        let mut ocel = example_ocel();
        ocel.events[3].relationships.push(OCELRelationship {
            object_id: "o1".to_string(),
            qualifier: "also".to_string(),
        });

        let locel = IndexLinkedOCEL::from(ocel);
        assert!(!discover_variable_arcs(&locel, 0.0).contains_key("order"));
    }

    #[test]
    fn test_variable_arc_tolerance() {
        // A second "pick" that touches no item at all: one of two executions deviates.
        let mut ocel = example_ocel();
        ocel.events.push(event("e5", "pick", &["o1"]));
        let locel = IndexLinkedOCEL::from(ocel);

        assert!(discover_variable_arcs(&locel, 0.0)["item"].contains("pick"));
        assert!(!discover_variable_arcs(&locel, 0.6)["item"].contains("pick"));
    }

    /// Leaving a type out has to give the same nets for the rest, since nothing about one type's
    /// net depends on the others being discovered.
    #[test]
    fn test_selecting_object_types_leaves_the_rest_alone() {
        let locel = IndexLinkedOCEL::from(example_ocel());
        let options = ObjectCentricDiscoveryOptions::default();
        let all = discover_ocpn(&locel, options.clone());

        for filter in [
            ObjectTypeFilter::only(["item"]),
            ObjectTypeFilter::except(["order"]),
        ] {
            let some = discover_ocpn(&locel, options.clone().with_object_types(filter));
            assert_eq!(some.object_types(), vec!["item"]);
            assert_eq!(some.num_places(), all.nets["item"].places.len());
            assert_eq!(some.num_arcs(), all.nets["item"].arcs.len());
            assert_eq!(some.activities(), vec!["pick", "place order"]);
            assert_eq!(some.variable_arcs, {
                let mut expected = all.variable_arcs.clone();
                expected.retain(|object_type, _| object_type == "item");
                expected
            });
        }
    }

    #[test]
    fn test_discovers_a_net_per_object_type() {
        let locel = IndexLinkedOCEL::from(example_ocel());
        let ocpn = discover_ocpn(&locel, ObjectCentricDiscoveryOptions::default());

        assert_eq!(ocpn.object_types(), vec!["item", "order"]);
        assert_eq!(ocpn.activities(), vec!["pay", "pick", "place order"]);

        // "place order" is shared by both types, which is the stitching point.
        assert_eq!(ocpn.object_types_of("place order"), vec!["item", "order"]);
        assert_eq!(ocpn.object_types_of("pay"), vec!["order"]);
        assert!(ocpn.is_variable_arc("item", "place order"));
        assert!(!ocpn.is_variable_arc("order", "place order"));

        for (object_type, net) in &ocpn.nets {
            assert!(
                net.initial_marking.is_some(),
                "{object_type} has no initial marking"
            );
            assert!(
                net.final_markings.is_some(),
                "{object_type} has no final marking"
            );
        }
    }

    /// Objects taking part in no event must not contribute an empty trace, which the miner would
    /// read as "this object type may be skipped".
    #[test]
    fn test_objects_without_events_do_not_make_the_net_optional() {
        let mut ocel = example_ocel();
        ocel.objects.push(object("i3", "item"));
        let locel = IndexLinkedOCEL::from(ocel);

        let ocpn = discover_ocpn(&locel, ObjectCentricDiscoveryOptions::default());
        assert_eq!(ocpn.object_types_of("pick"), vec!["item"]);
        assert!(ocpn.nets["item"]
            .transitions
            .values()
            .any(|t| t.label.as_deref() == Some("pick")));
    }
}
