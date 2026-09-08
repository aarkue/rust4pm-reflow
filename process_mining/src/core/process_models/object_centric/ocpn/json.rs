//! A flat JSON form of an object-centric Petri net, for exchanging one with other tools.
//!
//! One list of places, transitions and arcs rather than one net per object type. A transition
//! appears once no matter how many types share it, and an arc takes its type from the place it
//! connects, so nothing is lost. Names come from the net rather than from its UUIDs, so exporting
//! the same net twice gives the same file.
//!
//! ```text
//! {
//!   "places":      [{"name": "orders p0", "object_type": "orders"}],
//!   "transitions": [{"name": "place order", "label": "place order"},
//!                   {"name": "orders tau0", "label": null}],
//!   "arcs":        [{"source": "orders p0", "target": "place order",
//!                    "type": "variable", "weight": 1}],
//!   "initial_marking": {"orders p0": 1},
//!   "final_marking":   {"orders p3": 1}
//! }
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::object_centric_petri_net_struct::ObjectCentricPetriNet;
use crate::core::process_models::case_centric::petri_net::petri_net_struct::{
    ArcType, Marking, PetriNet, PlaceID, TransitionID,
};

/// An object-centric Petri net as a flat list of places, transitions and arcs.
///
/// See the [module documentation](self).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObjectCentricPetriNetJson {
    /// The places, each belonging to one object type.
    pub places: Vec<JsonPlace>,
    /// The transitions, each appearing once even when several object types share it.
    pub transitions: Vec<JsonTransition>,
    /// The arcs, each between a place and a transition.
    pub arcs: Vec<JsonArc>,
    /// Tokens per place at the start, by place name.
    pub initial_marking: BTreeMap<String, u64>,
    /// Tokens per place at the end, by place name.
    pub final_marking: BTreeMap<String, u64>,
}

/// A place of one object type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct JsonPlace {
    /// Unique among all places and transitions.
    pub name: String,
    /// The object type whose tokens this place holds.
    pub object_type: String,
}

/// A transition, silent if it has no label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct JsonTransition {
    /// Unique among all places and transitions; the activity itself for a labelled transition.
    pub name: String,
    /// The activity, or `None` for a silent transition.
    pub label: Option<String>,
}

/// An arc from a place to a transition or the other way round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct JsonArc {
    /// Name of the place or transition the arc leaves.
    pub source: String,
    /// Name of the place or transition the arc enters.
    pub target: String,
    /// Whether the arc moves one object or a whole set of them.
    #[serde(rename = "type", default)]
    pub arc_type: JsonArcType,
    /// How many tokens the arc moves.
    #[serde(default = "one")]
    pub weight: u32,
}

fn one() -> u32 {
    1
}

/// Whether an arc moves a single object or all of them at once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum JsonArcType {
    /// Moves one object of the place's type.
    #[default]
    Normal,
    /// Moves a set of objects of the place's type at once.
    Variable,
}

impl ObjectCentricPetriNet {
    /// Converts to the [flat JSON form](ObjectCentricPetriNetJson).
    pub fn to_json_form(&self) -> ObjectCentricPetriNetJson {
        self.to_json_form_named().0
    }

    /// [`to_json_form`](Self::to_json_form), plus the JSON name assigned to each place and
    /// transition id -- what a caller needs to translate its own uuid-keyed annotations into the
    /// exported JSON names.
    pub fn to_json_form_named(&self) -> (ObjectCentricPetriNetJson, HashMap<uuid::Uuid, String>) {
        let activities: HashSet<&str> = self.activities().into_iter().collect();
        let mut json = ObjectCentricPetriNetJson::default();
        let mut named: HashMap<uuid::Uuid, String> = HashMap::new();

        for object_type in self.object_types() {
            let net = &self.nets[object_type];
            let mut name_of = |id: uuid::Uuid, json: &mut ObjectCentricPetriNetJson| -> String {
                if let Some(known) = named.get(&id) {
                    return known.clone();
                }
                let name = match net.transitions.get(&id).map(|t| t.label.clone()) {
                    // The activity names the transition, so the nets sharing it agree on the name
                    // and it is written out once.
                    Some(Some(activity)) => {
                        if !json.transitions.iter().any(|t| t.name == activity) {
                            json.transitions.push(JsonTransition {
                                name: activity.clone(),
                                label: Some(activity.clone()),
                            });
                        }
                        activity
                    }
                    Some(None) => {
                        let name = unique(format!("{object_type} tau"), &activities, &named);
                        json.transitions.push(JsonTransition {
                            name: name.clone(),
                            label: None,
                        });
                        name
                    }
                    None => {
                        let name = unique(format!("{object_type} p"), &activities, &named);
                        json.places.push(JsonPlace {
                            name: name.clone(),
                            object_type: object_type.to_string(),
                        });
                        name
                    }
                };
                named.insert(id, name.clone());
                name
            };

            // Walking the arcs, which are stored in a fixed order, keeps the names reproducible.
            for arc in &net.arcs {
                let (from, to, place, transition) = match arc.from_to {
                    ArcType::PlaceTransition(place, transition) => {
                        (place, transition, place, transition)
                    }
                    ArcType::TransitionPlace(transition, place) => {
                        (transition, place, place, transition)
                    }
                };
                let _ = place;
                let variable = net
                    .transitions
                    .get(&transition)
                    .and_then(|t| t.label.as_deref())
                    .is_some_and(|activity| {
                        self.variable_arcs
                            .get(object_type)
                            .is_some_and(|activities| activities.contains(activity))
                    });

                let source = name_of(from, &mut json);
                let target = name_of(to, &mut json);
                json.arcs.push(JsonArc {
                    source,
                    target,
                    arc_type: match variable {
                        true => JsonArcType::Variable,
                        false => JsonArcType::Normal,
                    },
                    weight: arc.weight,
                });
            }

            for (place, tokens) in net.initial_marking.iter().flatten() {
                if let Some(name) = named.get(&place.0) {
                    json.initial_marking.insert(name.clone(), *tokens);
                }
            }
            // A net discovered from a process tree has exactly one final marking.
            for (place, tokens) in net.final_markings.iter().flatten().flatten() {
                if let Some(name) = named.get(&place.0) {
                    json.final_marking.insert(name.clone(), *tokens);
                }
            }
        }

        (json, named)
    }

    /// Rebuilds an object-centric Petri net from the [flat JSON form](ObjectCentricPetriNetJson).
    ///
    /// Places are grouped into one net per object type, and a transition joins every net that has
    /// an arc to it. Identifiers are new, so a net does not survive a round trip unchanged, but its
    /// structure does.
    pub fn from_json_form(json: &ObjectCentricPetriNetJson) -> Self {
        let type_of: HashMap<&str, &str> = json
            .places
            .iter()
            .map(|place| (place.name.as_str(), place.object_type.as_str()))
            .collect();
        let label_of: HashMap<&str, Option<&str>> = json
            .transitions
            .iter()
            .map(|transition| (transition.name.as_str(), transition.label.as_deref()))
            .collect();

        let mut ocpn = Self::new();
        let mut ids: HashMap<(&str, &str), uuid::Uuid> = HashMap::new();

        for arc in &json.arcs {
            let (place, transition) = match type_of.contains_key(arc.source.as_str()) {
                true => (arc.source.as_str(), arc.target.as_str()),
                false => (arc.target.as_str(), arc.source.as_str()),
            };
            let Some(&object_type) = type_of.get(place) else {
                continue;
            };
            let net = ocpn.nets.entry(object_type.to_string()).or_default();

            let place_id = *ids
                .entry((object_type, place))
                .or_insert_with(|| net.add_place(None).0);
            let label = label_of.get(transition).copied().flatten();
            let transition_id = *ids
                .entry((object_type, transition))
                .or_insert_with(|| net.add_transition(label.map(str::to_string), None).0);

            net.add_arc(
                match type_of.contains_key(arc.source.as_str()) {
                    true => {
                        ArcType::place_to_transition(PlaceID(place_id), TransitionID(transition_id))
                    }
                    false => {
                        ArcType::transition_to_place(TransitionID(transition_id), PlaceID(place_id))
                    }
                },
                Some(arc.weight),
            );

            if arc.arc_type == JsonArcType::Variable {
                if let Some(activity) = label {
                    ocpn.variable_arcs
                        .entry(object_type.to_string())
                        .or_default()
                        .insert(activity.to_string());
                }
            }
        }

        for (marking, into_final) in [(&json.initial_marking, false), (&json.final_marking, true)] {
            let mut per_type: HashMap<&str, Marking> = HashMap::new();
            for (place, tokens) in marking {
                let Some(&object_type) = type_of.get(place.as_str()) else {
                    continue;
                };
                if let Some(&id) = ids.get(&(object_type, place.as_str())) {
                    per_type
                        .entry(object_type)
                        .or_default()
                        .insert(PlaceID(id), *tokens);
                }
            }
            for (object_type, marking) in per_type {
                let net: &mut PetriNet = ocpn.nets.entry(object_type.to_string()).or_default();
                match into_final {
                    true => net.final_markings = Some(vec![marking]),
                    false => net.initial_marking = Some(marking),
                }
            }
        }

        ocpn
    }
}

/// A name of the form `prefix0`, `prefix1`, … that no activity and no other node already uses.
fn unique(
    prefix: String,
    activities: &HashSet<&str>,
    named: &HashMap<uuid::Uuid, String>,
) -> String {
    let taken: HashSet<&str> = named.values().map(String::as_str).collect();
    (0..)
        .map(|n| format!("{prefix}{n}"))
        .find(|name| !activities.contains(name.as_str()) && !taken.contains(name.as_str()))
        .expect("the counter is unbounded")
}

#[cfg(test)]
mod test_json_form {
    use super::*;
    use crate::core::process_models::process_tree::{Node, OperatorType, ProcessTree};

    fn net_of(activities: &[&str]) -> PetriNet {
        let mut sequence = Node::new_operator(OperatorType::Sequence);
        for activity in activities {
            sequence.add_child(Node::new_leaf(Some(activity.to_string())));
        }
        ProcessTree::new(sequence).to_petri_net()
    }

    fn example() -> ObjectCentricPetriNet {
        let mut ocpn = ObjectCentricPetriNet::new();
        ocpn.nets
            .insert("orders".to_string(), net_of(&["place order", "pay order"]));
        ocpn.nets
            .insert("items".to_string(), net_of(&["place order", "pick item"]));
        ocpn.variable_arcs.insert(
            "items".to_string(),
            HashSet::from(["place order".to_string()]),
        );
        ocpn
    }

    #[test]
    fn test_shared_transitions_appear_once() {
        let json = example().to_json_form();

        // "place order" is in both nets but is one transition of the object-centric net.
        assert_eq!(
            json.transitions
                .iter()
                .filter(|t| t.name == "place order")
                .count(),
            1
        );
        assert_eq!(json.transitions.len(), 3);
        assert_eq!(json.places.len(), 6);
        assert_eq!(json.arcs.len(), 8);

        // Every arc of the items type at "place order" is variable, and no other arc is.
        let variable: Vec<&JsonArc> = json
            .arcs
            .iter()
            .filter(|arc| arc.arc_type == JsonArcType::Variable)
            .collect();
        assert_eq!(variable.len(), 2);
        assert!(variable
            .iter()
            .all(|arc| arc.source == "place order" || arc.target == "place order"));
    }

    #[test]
    fn test_export_is_reproducible() {
        assert_eq!(example().to_json_form(), example().to_json_form());
    }

    #[test]
    fn test_round_trip_keeps_the_structure() {
        let json = example().to_json_form();
        let rebuilt = ObjectCentricPetriNet::from_json_form(&json);

        assert_eq!(rebuilt.object_types(), example().object_types());
        assert_eq!(rebuilt.activities(), example().activities());
        assert_eq!(rebuilt.num_places(), example().num_places());
        assert_eq!(rebuilt.num_transitions(), example().num_transitions());
        assert_eq!(rebuilt.num_arcs(), example().num_arcs());
        assert_eq!(rebuilt.variable_arcs, example().variable_arcs);
        assert_eq!(rebuilt.to_json_form(), json);
    }
}
