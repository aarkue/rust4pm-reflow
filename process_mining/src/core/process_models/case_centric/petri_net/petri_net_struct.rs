#[cfg(feature = "token-based-replay")]
use itertools::Itertools;
#[cfg(feature = "token-based-replay")]
use nalgebra::{DMatrix, Dyn, OMatrix};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

/// Fresh node ids are a process-wide counter in v4 clothing rather than random bytes.
///
/// Ids only have to be unique, but several consumers order nodes by id (the alignment
/// sync product, escaping-edges precision), and a random id hands them a different
/// permutation on every run. With several equally optimal alignments, which one comes
/// back then varies per run, and alignment-derived precision varies with it. A counter
/// makes id order creation order, which is deterministic for a deterministic miner.
static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn fresh_node_id() -> Uuid {
    let n = NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed);
    let mut b = [0u8; 16];
    b[9..16].copy_from_slice(&n.to_be_bytes()[1..]);
    b[6] = 0x40; // version 4
    b[8] = 0x80; // RFC variant
    Uuid::from_bytes(b)
}

use crate::core::process_models::case_centric::petri_net::pnml::{
    export_pnml,
    import_pnml::{self, PNMLParseError},
};
use crate::core::process_models::process_tree::ProcessTree;

#[derive(
    Debug, Clone, PartialEq, Deserialize, Serialize, Hash, Eq, PartialOrd, Ord, JsonSchema,
)]
/// Place in a Petri net
pub struct Place {
    id: Uuid,
}

#[derive(
    Debug, Clone, PartialEq, Deserialize, Serialize, Hash, Eq, PartialOrd, Ord, JsonSchema,
)]
/// Transition in a Petri net
pub struct Transition {
    /// Transition label (None if this transition is _invisible_)
    pub label: Option<String>,
    id: Uuid,
}

#[derive(Debug, Serialize, Deserialize)]
/// Nodes (Places or Transitions) in a Petri net
pub enum PetriNetNodes {
    /// None
    None,
    /// List of places
    Places(Vec<PlaceID>),
    /// List of transitions
    Transitions(Vec<TransitionID>),
}

#[derive(
    Debug, Deserialize, Serialize, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, JsonSchema,
)]
#[serde(tag = "type", content = "nodes")]
/// Arc type in a Petri net
pub enum ArcType {
    /// From Place to Transition
    PlaceTransition(Uuid, Uuid),
    /// From Transition to Place
    TransitionPlace(Uuid, Uuid),
}

impl ArcType {
    /// Create new from place to transition
    pub fn place_to_transition(from: PlaceID, to: TransitionID) -> ArcType {
        ArcType::PlaceTransition(from.0, to.0)
    }
    /// Create new from transition to place
    pub fn transition_to_place(from: TransitionID, to: PlaceID) -> ArcType {
        ArcType::TransitionPlace(from.0, to.0)
    }
    /// Checks if a given node ID is start or end of this arc
    pub fn contains(&self, id: &Uuid) -> bool {
        match self {
            ArcType::PlaceTransition(from, to) => from == id || to == id,
            ArcType::TransitionPlace(from, to) => from == id || to == id,
        }
    }
}

#[derive(
    Debug, Deserialize, Serialize, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, JsonSchema,
)]
/// Arc in a Petri net
///
/// Connecting a transition and a place (or the other way around)
pub struct Arc {
    /// Source and target of Arc
    pub from_to: ArcType,
    /// Weight (i.e., how many tokens this arc moves)
    pub weight: u32,
}

#[derive(
    Debug, PartialEq, Clone, Copy, Serialize, Deserialize, Hash, Eq, PartialOrd, Ord, JsonSchema,
)]
/// Place ID
pub struct PlaceID(pub Uuid);
impl PlaceID {
    /// Get UUID
    pub fn get_uuid(self) -> Uuid {
        self.0
    }
}
impl From<&Place> for PlaceID {
    fn from(value: &Place) -> Self {
        PlaceID(value.id)
    }
}

#[derive(
    Debug, PartialEq, Clone, Copy, Serialize, Deserialize, Hash, Eq, PartialOrd, Ord, JsonSchema,
)]
/// Transition ID
pub struct TransitionID(pub Uuid);

impl From<&Transition> for TransitionID {
    fn from(value: &Transition) -> Self {
        TransitionID(value.id)
    }
}
impl TransitionID {
    /// Get  UUID
    pub fn get_uuid(self) -> Uuid {
        self.0
    }
}

/// Marking of a Petri net: Assigning [`PlaceID`]s to a number of tokens
pub type Marking = HashMap<PlaceID, u64>;

#[derive(Debug, Deserialize, Serialize, Clone, JsonSchema)]
///
/// A Petri net of [`Place`]s and [`Transition`]s
///
/// Bipartite graph of [`Place`]s and [`Transition`]s with [`Arc`]s connecting them, as well as initial and final [`Marking`]s
pub struct PetriNet {
    /// Places
    pub places: HashMap<Uuid, Place>,
    /// Transitions
    pub transitions: HashMap<Uuid, Transition>,
    /// Arcs
    pub arcs: Vec<Arc>,
    /// Initial marking
    pub initial_marking: Option<Marking>,
    /// Final markings (any of them are accepted as a final marking)
    pub final_markings: Option<Vec<Marking>>,
}

impl Default for PetriNet {
    fn default() -> Self {
        Self::new()
    }
}
impl PetriNet {
    /// Create new [`PetriNet`] with no places or transitions
    pub fn new() -> Self {
        Self {
            places: HashMap::new(),
            transitions: HashMap::new(),
            arcs: Vec::new(),
            initial_marking: None,
            final_markings: None,
        }
    }
    /// Serialize to JSON string
    pub fn to_json(self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    /// Add a place (with an optional passed UUID)
    ///
    /// If no ID is passed, a new UUID will be generated
    pub fn add_place(&mut self, place_id: Option<Uuid>) -> PlaceID {
        let place_id = place_id.unwrap_or_else(fresh_node_id);
        let place = Place { id: place_id };
        self.places.insert(place_id, place);
        PlaceID(place_id)
    }

    /// Add a transition with an label (and with an optional passed UUID)
    ///
    /// If no ID is passed, a new UUID will be generated
    pub fn add_transition(
        &mut self,
        label: Option<String>,
        transition_id: Option<Uuid>,
    ) -> TransitionID {
        let transition_id = transition_id.unwrap_or_else(fresh_node_id);
        let transition = Transition {
            id: transition_id,
            label,
        };
        self.transitions.insert(transition_id, transition);
        TransitionID(transition_id)
    }
    /// Add an arc
    pub fn add_arc(&mut self, from_to: ArcType, weight: Option<u32>) {
        self.arcs.push(Arc {
            from_to,
            weight: weight.unwrap_or(1),
        });
    }

    /// Remove any node (Transition/Place) from the Petri net
    pub fn remove_node(&mut self, id: &Uuid) {
        if let Some(p) = self.places.remove(id) {
            if let Some(im) = &mut self.initial_marking {
                im.remove(&(&p).into());
            }
            if let Some(fm) = &mut self.final_markings {
                for m in fm {
                    m.remove(&(&p).into());
                }
            }
        }
        self.transitions.remove(id);
        self.arcs.retain(|arc| !arc.from_to.contains(id));
    }

    /// Remove a Place from the Petri net
    pub fn remove_place(&mut self, place_id: &Uuid) {
        if self.places.contains_key(place_id) {
            self.remove_node(place_id);
        }
    }

    /// Remove a Transition from the Petri net
    pub fn remove_transition(&mut self, transition_id: &Uuid) {
        if self.transitions.contains_key(transition_id) {
            self.remove_node(transition_id);
        }
    }

    /// Get the preset of a [`PetriNet`] node referred to by passed id
    pub fn preset_of(&self, id: Uuid) -> PetriNetNodes {
        if self.places.contains_key(&id) {
            let p = self.places.get(&id).unwrap();
            PetriNetNodes::Transitions(self.preset_of_place(p.into()))
        } else if self.transitions.contains_key(&id) {
            let t = self.transitions.get(&id).unwrap();
            PetriNetNodes::Places(self.preset_of_transition(t.into()))
        } else {
            PetriNetNodes::None
        }
    }

    /// Get the preset of a [`PetriNet`] place
    pub fn preset_of_place(&self, p: PlaceID) -> Vec<TransitionID> {
        self.arcs
            .iter()
            .filter_map(|x: &Arc| match x.from_to {
                ArcType::TransitionPlace(from, to) if to == p.0 => Some(TransitionID(from)),
                _ => None,
            })
            .collect()
    }

    /// Get the preset of [`PetriNet`] transition referred to by passed id
    pub fn preset_of_transition(&self, t: TransitionID) -> Vec<PlaceID> {
        self.arcs
            .iter()
            .filter_map(|x: &Arc| match x.from_to {
                ArcType::PlaceTransition(from, to) if to == t.0 => Some(PlaceID(from)),
                _ => None,
            })
            .collect()
    }

    /// Get postset of [`PetriNet`] node referred to by passed id
    pub fn postset_of(&self, id: &Uuid) -> PetriNetNodes {
        if self.places.contains_key(id) {
            let p = self.places.get(id).unwrap();
            PetriNetNodes::Transitions(self.postset_of_place(p.into()))
        } else if self.transitions.contains_key(id) {
            let t = self.transitions.get(id).unwrap();
            PetriNetNodes::Places(self.postset_of_transition(t.into()))
        } else {
            PetriNetNodes::None
        }
    }

    /// Get postset of [`PetriNet`] place referred to by passed id
    pub fn postset_of_place(&self, p: PlaceID) -> Vec<TransitionID> {
        self.arcs
            .iter()
            .filter_map(|x: &Arc| match x.from_to {
                ArcType::PlaceTransition(from, to) if from == p.0 => Some(TransitionID(to)),
                _ => None,
            })
            .collect()
    }

    /// Get postset of [`PetriNet`] transition referred to by passed id
    pub fn postset_of_transition(&self, t: TransitionID) -> Vec<PlaceID> {
        self.arcs
            .iter()
            .filter_map(|x: &Arc| match x.from_to {
                ArcType::TransitionPlace(from, to) if from == t.0 => Some(PlaceID(to)),
                _ => None,
            })
            .collect()
    }

    /// Check if place is in initial marking
    pub fn is_in_initial_marking(&self, p: &PlaceID) -> bool {
        self.initial_marking.is_some() && self.initial_marking.as_ref().unwrap().contains_key(p)
    }

    /// Check if place is in _any_ final marking
    pub fn is_in_a_final_marking(&self, p: &PlaceID) -> bool {
        self.final_markings.is_some()
            && self
                .final_markings
                .as_ref()
                .unwrap()
                .iter()
                .any(|m| m.contains_key(p))
    }

    /// Checks if the Petri net contains duplicate or silent transitions
    pub fn contains_duplicate_or_silent_transitions(&self) -> bool {
        let mut activities = HashSet::new();

        for transition in self.transitions.values() {
            if let Some(label) = &transition.label {
                if activities.contains(label) {
                    return true;
                } else {
                    activities.insert(label.clone());
                }
            }
        }

        false
    }

    #[cfg(feature = "token-based-replay")]
    /// Creates a dictionary for the creation of matrices and vectors
    pub fn create_vector_dictionary(&self) -> HashMap<Uuid, usize> {
        let mut result: HashMap<Uuid, usize> = HashMap::new();

        self.places
            .keys()
            .sorted()
            .enumerate()
            .for_each(|(pos, id)| {
                result.insert(*id, pos);
            });

        self.transitions
            .keys()
            .sorted()
            .enumerate()
            .for_each(|(pos, id)| {
                result.insert(*id, pos);
            });

        result
    }

    #[cfg(feature = "token-based-replay")]
    /// Creates the pre-incidence matrix of the Petri net
    pub fn create_pre_incidence_matrix(
        &self,
        vector_dictionary: &HashMap<Uuid, usize>,
    ) -> DMatrix<u8> {
        let mut result: OMatrix<u8, Dyn, Dyn> =
            DMatrix::zeros(self.places.len(), self.transitions.len());

        self.arcs.iter().for_each(|arc| match arc.from_to {
            ArcType::PlaceTransition(place_id, transition_id) => {
                result[(
                    *vector_dictionary.get(&place_id).unwrap(),
                    *vector_dictionary.get(&transition_id).unwrap(),
                )] += 1;
            }
            ArcType::TransitionPlace(_, _) => {}
        });

        result
    }

    #[cfg(feature = "token-based-replay")]
    /// Creates the post-incidence matrix of the Petri net
    pub fn create_post_incidence_matrix(
        &self,
        vector_dictionary: &HashMap<Uuid, usize>,
    ) -> DMatrix<u8> {
        let mut result: OMatrix<u8, Dyn, Dyn> =
            DMatrix::zeros(self.places.len(), self.transitions.len());

        self.arcs.iter().for_each(|arc| match arc.from_to {
            ArcType::PlaceTransition(_, _) => {}
            ArcType::TransitionPlace(transition_id, place_id) => {
                result[(
                    *vector_dictionary.get(&place_id).unwrap(),
                    *vector_dictionary.get(&transition_id).unwrap(),
                )] += 1;
            }
        });

        result
    }

    #[cfg(feature = "token-based-replay")]
    /// Creates the incidence matrix of the Petri net
    pub fn create_incidence_matrix(&self, vector_dictionary: &HashMap<Uuid, usize>) -> DMatrix<i8> {
        self.create_post_incidence_matrix(vector_dictionary)
            .cast::<i8>()
            - self
                .create_pre_incidence_matrix(vector_dictionary)
                .cast::<i8>()
    }

    #[cfg(feature = "graphviz-export")]
    /// Export Petri net as a PNG image
    ///
    /// The PNG file is written to the specified filepath
    ///
    /// _Note_: This is an export method for __visualizing__ the Petri net.
    /// The resulting PNG file cannot be imported as a Petri net again (for that functionality, see [`PetriNet::export_pnml`]).
    ///
    /// Only available with the `graphviz-export` feature.
    pub fn export_png<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
        super::image_export::export_petri_net_image_png(self, path)
    }

    #[cfg(feature = "graphviz-export")]
    /// Export Petri net as a SVG image
    ///
    /// The SVG file is written to the specified filepath
    ///
    /// _Note_: This is an export method for __visualizing__ the Petri net.
    /// The resulting SVG file cannot be imported as a Petri net again (for that functionality, see [`PetriNet::export_pnml`]).
    ///
    /// Only available with the `graphviz-export` feature.
    pub fn export_svg<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
        super::image_export::export_petri_net_image_svg(self, path)
    }

    /// Export Petri net to a PNML file
    ///
    /// The PNML file is written to the specified filepath
    ///
    /// _Note_: This is an export method for __saving__ the Petri net data.
    /// The resulting file can also be imported as a Petri net again (see [`PetriNet::import_pnml`]).
    pub fn export_pnml<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), quick_xml::Error> {
        export_pnml::export_petri_net_to_pnml_path(self, path)
    }
    /// Import Petri net from a PNML file
    ///
    /// The PNML file is read from the specified filepath
    ///
    ///
    /// For the related export function, see [`PetriNet::export_pnml`])
    pub fn import_pnml<P: AsRef<std::path::Path>>(path: P) -> Result<PetriNet, PNMLParseError> {
        import_pnml::import_pnml_from_path(path)
    }
}

/// Creates a [`PetriNet`] from a [`ProcessTree`]
impl From<ProcessTree> for PetriNet {
    fn from(process_tree: ProcessTree) -> Self {
        process_tree.to_petri_net()
    }
}

#[cfg(test)]
mod tests {
    pub const SAMPLE_JSON_NET: &str = r#"
{
    "places": {
        "f20ded2a-d308-44d7-abb2-6d0acd30e43e": {
            "id": "f20ded2a-d308-44d7-abb2-6d0acd30e43e"
        },
        "25f9c84b-f220-4e7f-a86e-bb3f82676bb9": {
            "id": "25f9c84b-f220-4e7f-a86e-bb3f82676bb9"
        },
        "15810d3d-922c-43fc-bcd5-8d6e592ea537": {
            "id": "15810d3d-922c-43fc-bcd5-8d6e592ea537"
        },
        "a75faf03-731d-4c8c-9810-5a36c7e8c26b": {
            "id": "a75faf03-731d-4c8c-9810-5a36c7e8c26b"
        }
    },
    "transitions": {
        "0c768c77-6408-4f4f-88b8-13d9cc8fca20": {
            "id": "0c768c77-6408-4f4f-88b8-13d9cc8fca20",
            "label": "Inform User"
        },
        "54f78f93-523f-4e1e-a0f7-cd79e73dc473": {
            "id": "54f78f93-523f-4e1e-a0f7-cd79e73dc473",
            "label": "Register"
        },
        "f18e00b0-e90b-48f6-99b7-9ee526571213": {
            "id": "f18e00b0-e90b-48f6-99b7-9ee526571213",
            "label": "Archive Repair"
        },
        "97d666fc-a78b-481d-9a5a-0cad157682ca": {
            "id": "97d666fc-a78b-481d-9a5a-0cad157682ca",
            "label": "Analyze Defect"
        },
        "78266d34-8abf-43ab-99bc-69e5e93c24b1": {
            "id": "78266d34-8abf-43ab-99bc-69e5e93c24b1",
            "label": "Repair (Simple)"
        },
        "5e8f7aff-81d4-4822-a30f-875ecc0a06f0": {
            "id": "5e8f7aff-81d4-4822-a30f-875ecc0a06f0",
            "label": "Repair (Complex)"
        },
        "18915408-cc29-4a7c-ab93-8a33e78a277a": {
            "id": "18915408-cc29-4a7c-ab93-8a33e78a277a",
            "label": "Test Repair"
        },
        "2da04f6f-dacb-46ac-82fd-39d0dfe44c33": {
            "id": "2da04f6f-dacb-46ac-82fd-39d0dfe44c33",
            "label": "Restart Repair"
        }
    },
    "arcs": [
        {
            "from_to": {
                "type": "TransitionPlace",
                "nodes": [
                    "f18e00b0-e90b-48f6-99b7-9ee526571213",
                    "f20ded2a-d308-44d7-abb2-6d0acd30e43e"
                ]
            },
            "weight": 1
        },
        {
            "from_to": {
                "type": "TransitionPlace",
                "nodes": [
                    "0c768c77-6408-4f4f-88b8-13d9cc8fca20",
                    "a75faf03-731d-4c8c-9810-5a36c7e8c26b"
                ]
            },
            "weight": 1
        },
        {
            "from_to": {
                "type": "PlaceTransition",
                "nodes": [
                    "15810d3d-922c-43fc-bcd5-8d6e592ea537",
                    "54f78f93-523f-4e1e-a0f7-cd79e73dc473"
                ]
            },
            "weight": 1
        },
        {
            "from_to": {
                "type": "PlaceTransition",
                "nodes": [
                    "25f9c84b-f220-4e7f-a86e-bb3f82676bb9",
                    "f18e00b0-e90b-48f6-99b7-9ee526571213"
                ]
            },
            "weight": 1
        },
        {
            "from_to": {
                "type": "TransitionPlace",
                "nodes": [
                    "0c768c77-6408-4f4f-88b8-13d9cc8fca20",
                    "25f9c84b-f220-4e7f-a86e-bb3f82676bb9"
                ]
            },
            "weight": 1
        }
    ]
}
"#;
    use std::str::FromStr;

    use super::*;

    #[test]
    fn petri_nets() {
        let mut net = PetriNet::new();
        let p1 = net.add_place(None);
        let t1 = net.add_transition(Some("Have fun".into()), None);
        let t2 = net.add_transition(Some("Sleep".into()), None);
        net.add_arc(ArcType::place_to_transition(p1, t1), None);
        net.add_arc(ArcType::transition_to_place(t2, p1), None);

        assert!(net.postset_of_transition(t1).is_empty());
        assert!(net.preset_of_transition(t1) == vec![p1]);
        assert!(net.postset_of_place(p1) == vec![t1]);
        assert!(net.preset_of_place(p1) == vec![t2]);
        assert!(net.preset_of_transition(t2).is_empty());
    }

    #[test]
    fn deserialize_petri_net_test() {
        let pn: PetriNet = serde_json::from_str(SAMPLE_JSON_NET).unwrap();
        assert!(pn.places.len() == 4);
        assert!(
            pn.postset_of_transition(TransitionID(
                Uuid::parse_str("0c768c77-6408-4f4f-88b8-13d9cc8fca20").unwrap()
            ))
            .len()
                == 2
        );
    }

    #[test]
    fn remove_nodes_petri_net_test() {
        let mut pn: PetriNet = serde_json::from_str(SAMPLE_JSON_NET).unwrap();
        let p1_id = Uuid::from_str("f20ded2a-d308-44d7-abb2-6d0acd30e43e").unwrap();
        let t1_id = Uuid::from_str("f18e00b0-e90b-48f6-99b7-9ee526571213").unwrap();
        if let PetriNetNodes::Places(p) = pn.postset_of(&t1_id) {
            assert!(p.len() == 1);
        } else {
            unreachable!();
        }

        pn.remove_transition(&p1_id);
        assert!(pn.places.contains_key(&p1_id));
        pn.remove_place(&p1_id);
        assert!(!pn.places.contains_key(&p1_id));
        if let PetriNetNodes::Places(p) = pn.postset_of(&t1_id) {
            assert!(p.is_empty());
        } else {
            unreachable!();
        }
    }

    #[cfg(feature = "token-based-replay")]
    #[test]
    fn create_incidence_matrix_test() {
        let mut net = PetriNet::new();
        let p1 = net.add_place(Some(
            Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe000").unwrap(),
        ));
        let p2 = net.add_place(Some(
            Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe001").unwrap(),
        ));
        let p3 = net.add_place(Some(
            Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe002").unwrap(),
        ));
        let t1 = net.add_transition(
            Some("a".into()),
            Some(Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe003").unwrap()),
        );
        let t2 = net.add_transition(
            Some("b".into()),
            Some(Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe004").unwrap()),
        );
        let t3 = net.add_transition(
            Some("c".into()),
            Some(Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe005").unwrap()),
        );
        let t4 = net.add_transition(
            Some("d".into()),
            Some(Uuid::parse_str("67e55044-10b1-426f-9247-bb680e5fe006").unwrap()),
        );
        net.add_arc(ArcType::place_to_transition(p1, t1), None);
        net.add_arc(ArcType::place_to_transition(p1, t2), None);
        net.add_arc(ArcType::transition_to_place(t1, p2), None);
        net.add_arc(ArcType::transition_to_place(t2, p2), None);
        net.add_arc(ArcType::place_to_transition(p2, t3), None);
        net.add_arc(ArcType::transition_to_place(t3, p3), None);
        net.add_arc(ArcType::transition_to_place(t4, p2), None);
        net.add_arc(ArcType::place_to_transition(p2, t4), None);

        let vector_dictionary: HashMap<Uuid, usize> = net.create_vector_dictionary();
        let pre_matrix = net.create_pre_incidence_matrix(&vector_dictionary);
        let expected_pre_matrix =
            DMatrix::from_row_slice(3, 4, &[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);

        assert_eq!(pre_matrix, expected_pre_matrix);

        let post_matrix = net.create_post_incidence_matrix(&vector_dictionary);
        let expected_post_matrix =
            DMatrix::from_row_slice(3, 4, &[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]);

        assert_eq!(post_matrix, expected_post_matrix);

        let incidence_matrix = net.create_incidence_matrix(&vector_dictionary);
        let expected_incidence_matrix =
            DMatrix::from_row_slice(3, 4, &[-1, -1, 0, 0, 1, 1, -1, 0, 0, 0, 1, 0]);

        assert_eq!(incidence_matrix, expected_incidence_matrix);
    }
}

impl PetriNet {
    /// Remove silent structure that constrains nothing, preserving the net's language.
    ///
    /// Constructed nets (and occasionally mined ones) carry silent transitions that do no
    /// work: a `p -> tau -> q` chain where `p` has no other successor and `q` no other
    /// predecessor, a tau that reads and writes the same places, two taus with identical
    /// surroundings, or a fragment left with no labeled transition at all. Each rule below
    /// removes only structure whose firing options are unchanged with it gone, so fitness
    /// and precision of the simplified net are those of the original; what changes is the
    /// size a reader (and a size column) sees.
    pub fn simplify_silent(&mut self) {
        self.simplify_silent_tracked();
    }

    /// [`simplify_silent`](Self::simplify_silent), returning the place merges it performed as
    /// `(removed, kept)` pairs in the order they happened.
    ///
    /// A caller carrying its own per-place annotations (which cell a place answers for, why
    /// it exists) has no way to recover them after simplification otherwise: series-tau
    /// fusion is the one rule here that removes a place by folding it into another rather
    /// than dropping it outright, and which one survives is a coin only this function calls.
    /// Replaying the pairs in order (move `removed`'s annotation onto `kept`, if any) is
    /// enough even across a chain of fusions, since each pair names the id current at the
    /// moment it fired.
    pub fn simplify_silent_tracked(&mut self) -> Vec<(Uuid, Uuid)> {
        let mut merges: Vec<(Uuid, Uuid)> = Vec::new();
        loop {
            let mut changed = false;

            // Pre/post sets as (place, weight) lists per transition, rebuilt per round.
            let mut pre: HashMap<Uuid, Vec<(Uuid, u32)>> = HashMap::new();
            let mut post: HashMap<Uuid, Vec<(Uuid, u32)>> = HashMap::new();
            let mut place_out: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
            let mut place_in: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
            for arc in &self.arcs {
                match arc.from_to {
                    ArcType::PlaceTransition(p, t) => {
                        pre.entry(t).or_default().push((p, arc.weight));
                        place_out.entry(p).or_default().push(t);
                    }
                    ArcType::TransitionPlace(t, p) => {
                        post.entry(t).or_default().push((p, arc.weight));
                        place_in.entry(p).or_default().push(t);
                    }
                }
            }

            let tau_ids: Vec<Uuid> = self
                .transitions
                .iter()
                .filter(|(_, tr)| tr.label.is_none())
                .map(|(id, _)| *id)
                .collect();

            // Self-loop tau: reads exactly what it writes. Firing it changes nothing.
            for t in tau_ids.iter() {
                let mut a = pre.get(t).cloned().unwrap_or_default();
                let mut b = post.get(t).cloned().unwrap_or_default();
                a.sort_unstable();
                b.sort_unstable();
                if !a.is_empty() && a == b {
                    self.transitions.remove(t);
                    self.arcs.retain(|arc| !arc.from_to.contains(t));
                    changed = true;
                }
            }
            if changed {
                continue;
            }

            // Unconditional fork: a silent transition whose one input place is a pure source
            // -- nothing produces it, the transition is its only consumer -- is enabled the
            // instant the net is and has nothing else to wait on. Firing it "at time zero" and
            // crediting its outputs directly in the initial marking is the same behaviour with
            // the step gone: a token duplicated by a forced fork and a marking that starts
            // with one token in each branch already are the same fact stated two ways, and the
            // second needs no transition to say it.
            'fork: for t in tau_ids.iter() {
                if !self.transitions.contains_key(t) {
                    continue;
                }
                let Some(ins) = pre.get(t) else { continue };
                if ins.len() != 1 {
                    continue;
                }
                let (p, w) = ins[0];
                if w != 1 {
                    continue;
                }
                if place_in.get(&p).is_some_and(|v| !v.is_empty()) {
                    continue;
                }
                if place_out.get(&p).map(Vec::len) != Some(1) {
                    continue;
                }
                let Some(init) = &self.initial_marking else { continue };
                if init.get(&PlaceID(p)).copied().unwrap_or(0) != 1 {
                    continue;
                }
                if self
                    .final_markings
                    .iter()
                    .flatten()
                    .any(|m| m.get(&PlaceID(p)).copied().unwrap_or(0) > 0)
                {
                    continue;
                }
                let Some(outs) = post.get(t).cloned() else { continue };
                if outs.is_empty() {
                    continue;
                }
                self.transitions.remove(t);
                self.places.remove(&p);
                self.arcs.retain(|arc| !arc.from_to.contains(t));
                if let Some(m) = &mut self.initial_marking {
                    m.remove(&PlaceID(p));
                    for (q, wq) in &outs {
                        *m.entry(PlaceID(*q)).or_default() += u64::from(*wq);
                    }
                }
                changed = true;
                break 'fork;
            }
            if changed {
                continue;
            }

            // Unconditional join, the mirror of the fork above: a silent transition whose one
            // output place is a pure sink -- nothing consumes it, the transition is its only
            // producer, and it sits in every final marking with exactly one token -- fires as
            // soon as all its inputs are ready and gates nothing afterward. Crediting its
            // inputs directly to the final marking in its place says the same thing: the token
            // in the sink only ever meant "every branch got here", which the branches' own
            // places already say once the sink and its transition are gone.
            'join: for t in tau_ids.iter() {
                if !self.transitions.contains_key(t) {
                    continue;
                }
                let Some(outs) = post.get(t) else { continue };
                if outs.len() != 1 {
                    continue;
                }
                let (q, w) = outs[0];
                if w != 1 {
                    continue;
                }
                if place_out.get(&q).is_some_and(|v| !v.is_empty()) {
                    continue;
                }
                if place_in.get(&q).map(Vec::len) != Some(1) {
                    continue;
                }
                if self
                    .initial_marking
                    .as_ref()
                    .is_some_and(|m| m.contains_key(&PlaceID(q)))
                {
                    continue;
                }
                let Some(fms) = &self.final_markings else { continue };
                if fms.is_empty() || fms.iter().any(|m| m.get(&PlaceID(q)).copied().unwrap_or(0) != 1) {
                    continue;
                }
                let Some(ins) = pre.get(t).cloned() else { continue };
                if ins.is_empty() {
                    continue;
                }
                self.transitions.remove(t);
                self.places.remove(&q);
                self.arcs.retain(|arc| !arc.from_to.contains(t));
                if let Some(fms) = &mut self.final_markings {
                    for m in fms {
                        if m.remove(&PlaceID(q)).is_some() {
                            for (p, wp) in &ins {
                                *m.entry(PlaceID(*p)).or_default() += u64::from(*wp);
                            }
                        }
                    }
                }
                changed = true;
                break 'join;
            }
            if changed {
                continue;
            }

            // Series tau: p -> tau -> q, both arcs weight 1, p != q, where the tau is
            // either the only consumer of p or the only producer of q. In the first case
            // every token in p can only ever move, silently, to q, so p fuses into q; in
            // the second every token in q came, silently, from p being consumed, so q
            // fuses into p. Either way the firing options and the projected language are
            // unchanged; only the free move disappears.
            'series: for t in tau_ids.iter() {
                let (Some(a), Some(b)) = (pre.get(t), post.get(t)) else { continue };
                if a.len() != 1 || b.len() != 1 {
                    continue;
                }
                let (p, wp) = a[0];
                let (q, wq) = b[0];
                if p == q || wp != 1 || wq != 1 {
                    continue;
                }
                let in_final = |x: Uuid| {
                    self.final_markings
                        .iter()
                        .flatten()
                        .any(|m| m.get(&PlaceID(x)).copied().unwrap_or(0) > 0)
                };
                let in_initial = |x: Uuid| {
                    self.initial_marking
                        .as_ref()
                        .map(|m| m.get(&PlaceID(x)).copied().unwrap_or(0) > 0)
                        .unwrap_or(false)
                };
                // Forward fusion (p into q) pre-fires the tau, which is free only if no
                // accepting marking asks for the token still in p. Backward fusion (q into
                // p) un-fires it, which is free only if no initial token starts in q.
                let p_exclusive = place_out.get(&p).map(Vec::len) == Some(1) && !in_final(p);
                let q_exclusive = place_in.get(&q).map(Vec::len) == Some(1) && !in_initial(q);
                let (gone, kept) = if p_exclusive {
                    (p, q)
                } else if q_exclusive {
                    (q, p)
                } else {
                    continue;
                };
                self.transitions.remove(t);
                self.places.remove(&gone);
                self.arcs.retain(|arc| !arc.from_to.contains(t));
                for arc in &mut self.arcs {
                    match &mut arc.from_to {
                        ArcType::PlaceTransition(from, _) if *from == gone => *from = kept,
                        ArcType::TransitionPlace(_, to) if *to == gone => *to = kept,
                        _ => {}
                    }
                }
                merges.push((gone, kept));
                let move_tokens = |m: &mut Marking| {
                    if let Some(n) = m.remove(&PlaceID(gone)) {
                        *m.entry(PlaceID(kept)).or_default() += n;
                    }
                };
                if let Some(m) = &mut self.initial_marking {
                    move_tokens(m);
                }
                if let Some(fs) = &mut self.final_markings {
                    for m in fs {
                        move_tokens(m);
                    }
                }
                changed = true;
                break 'series;
            }
            if changed {
                continue;
            }

            // Dual series: tau1 -> p -> tau2 with p exclusive between two silent
            // transitions and unmarked. Firing tau1 makes tau2 the only consumer of p, so
            // the pair acts as one silent step; merging them removes the seat place. The
            // merged tau consumes tau1's inputs plus tau2's other inputs and produces
            // tau2's outputs plus tau1's other outputs.
            'dual: for t1 in &tau_ids {
                if !self.transitions.contains_key(t1) {
                    continue;
                }
                let Some(outs) = post.get(t1) else { continue };
                for (p_mid, w) in outs {
                    if *w != 1 {
                        continue;
                    }
                    let only_in = place_in.get(p_mid).map(Vec::len) == Some(1);
                    let only_out = place_out.get(p_mid).map(Vec::len) == Some(1);
                    if !only_in || !only_out {
                        continue;
                    }
                    let t2 = place_out[p_mid][0];
                    if t2 == *t1 || self.transitions.get(&t2).and_then(|tr| tr.label.clone()).is_some() {
                        continue;
                    }
                    // Merging is only a pure sequencing of silents when one side owns the
                    // seat exclusively: t2 consumes nothing but p (it could always fire
                    // eagerly after t1), or t1 produces nothing but p (it could always fire
                    // lazily right before t2). With extra inputs AND extra outputs, merging
                    // moves a synchronization point and changes the language.
                    let t2_only_p = pre.get(&t2).map(|a| a.as_slice()) == Some(&[(*p_mid, 1)][..]);
                    let t1_only_p = post.get(t1).map(|a| a.as_slice()) == Some(&[(*p_mid, 1)][..]);
                    if !t2_only_p && !t1_only_p {
                        continue;
                    }
                    if pre.get(&t2).map(|a| a.iter().filter(|(x, _)| x == p_mid).count()) != Some(1) {
                        continue;
                    }
                    let marked = self
                        .initial_marking
                        .iter()
                        .chain(self.final_markings.iter().flatten())
                        .any(|m| m.get(&PlaceID(*p_mid)).copied().unwrap_or(0) > 0);
                    if marked {
                        continue;
                    }
                    // Build the merged tau.
                    let merged = self.add_transition(None, None);
                    let mut add: HashMap<(bool, Uuid), u32> = HashMap::new();
                    for (pl, w) in pre.get(t1).into_iter().flatten() {
                        *add.entry((true, *pl)).or_default() += w;
                    }
                    for (pl, w) in pre.get(&t2).into_iter().flatten() {
                        if pl != p_mid {
                            *add.entry((true, *pl)).or_default() += w;
                        }
                    }
                    for (pl, w) in post.get(t1).into_iter().flatten() {
                        if pl != p_mid {
                            *add.entry((false, *pl)).or_default() += w;
                        }
                    }
                    for (pl, w) in post.get(&t2).into_iter().flatten() {
                        *add.entry((false, *pl)).or_default() += w;
                    }
                    for ((is_in, pl), w) in add {
                        let a = if is_in {
                            ArcType::PlaceTransition(pl, merged.0)
                        } else {
                            ArcType::TransitionPlace(merged.0, pl)
                        };
                        self.add_arc(a, Some(w));
                    }
                    let p_mid = *p_mid;
                    let t1 = *t1;
                    self.transitions.remove(&t1);
                    self.transitions.remove(&t2);
                    self.places.remove(&p_mid);
                    self.arcs.retain(|arc| {
                        !arc.from_to.contains(&t1)
                            && !arc.from_to.contains(&t2)
                            && !arc.from_to.contains(&p_mid)
                    });
                    changed = true;
                    break 'dual;
                }
            }
            if changed {
                continue;
            }

            // Duplicate tau: identical pre and post sets as another tau. A tau matching a
            // labeled transition is NOT a duplicate: it is a silent alternative, and
            // removing it forces the label into every trace that used the silent route.
            let mut seen: HashMap<(Vec<(Uuid, u32)>, Vec<(Uuid, u32)>), Uuid> = HashMap::new();
            let mut sorted_taus = tau_ids.clone();
            sorted_taus.sort_unstable();
            for id in sorted_taus.iter() {
                if !self.transitions.contains_key(id) {
                    continue;
                }
                let mut a = pre.get(id).cloned().unwrap_or_default();
                let mut b = post.get(id).cloned().unwrap_or_default();
                a.sort_unstable();
                b.sort_unstable();
                let key = (a, b);
                if seen.contains_key(&key) {
                    self.transitions.remove(id);
                    self.arcs.retain(|arc| !arc.from_to.contains(id));
                    changed = true;
                    break;
                } else {
                    seen.insert(key, *id);
                }
            }
            if changed {
                continue;
            }

            // Disconnected, unmarked places.
            let connected: std::collections::HashSet<Uuid> = self
                .arcs
                .iter()
                .flat_map(|arc| match arc.from_to {
                    ArcType::PlaceTransition(p, t) => [p, t],
                    ArcType::TransitionPlace(t, p) => [t, p],
                })
                .collect();
            let marked: std::collections::HashSet<Uuid> = self
                .initial_marking
                .iter()
                .flat_map(|m| m.keys())
                .chain(self.final_markings.iter().flatten().flat_map(|m| m.keys()))
                .map(|p| p.0)
                .collect();
            {
                let before = self.places.len();
                self.places
                    .retain(|id, _| connected.contains(id) || marked.contains(id));
                if self.places.len() != before {
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
        merges
    }
}

#[cfg(test)]
mod simplify_silent_tests {
    use super::*;

    #[test]
    fn series_tau_fuses_and_tracks_the_merge() {
        // producer -> p0 --tau--> p1 -> consumer. p0 exclusive to the tau and never
        // final-marked, so it fuses into p1; a caller carrying per-place annotations (a
        // `PlaceRole`, in this crate) needs the (removed, kept) pair to move them across.
        // Both places get an extra labeled neighbour so neither the unconditional-fork
        // nor unconditional-join shortcut fires first and preempts the series rule this
        // test means to exercise.
        let mut net = PetriNet::default();
        let p0 = net.add_place(None);
        let p1 = net.add_place(None);
        let producer = net.add_transition(Some("P".into()), None);
        let t = net.add_transition(None, None);
        let consumer = net.add_transition(Some("C".into()), None);
        net.add_arc(ArcType::transition_to_place(producer, p0), Some(1));
        net.add_arc(ArcType::place_to_transition(p0, t), Some(1));
        net.add_arc(ArcType::transition_to_place(t, p1), Some(1));
        net.add_arc(ArcType::place_to_transition(p1, consumer), Some(1));

        let merges = net.simplify_silent_tracked();

        assert_eq!(merges, vec![(p0.get_uuid(), p1.get_uuid())]);
        assert_eq!(net.places.len(), 1, "p0 fuses away, only p1 remains");
        assert!(net.places.contains_key(&p1.get_uuid()));
        assert!(!net.transitions.contains_key(&t.get_uuid()), "the tau itself is gone");
        assert!(net.transitions.contains_key(&producer.get_uuid()));
        assert!(net.transitions.contains_key(&consumer.get_uuid()));
        assert!(
            net.arcs.iter().any(|a| a.from_to == ArcType::transition_to_place(producer, p1)),
            "producer's arc is rewired onto the surviving place"
        );
    }

    #[test]
    fn a_self_loop_tau_is_removed_without_touching_its_place() {
        // A tau that reads and writes the same place changes nothing when it fires, so it
        // is dropped outright, the place itself (marked, so it survives the disconnected-
        // place sweep) is untouched.
        let mut net = PetriNet::default();
        let p = net.add_place(None);
        let t = net.add_transition(None, None);
        net.add_arc(ArcType::place_to_transition(p, t), Some(1));
        net.add_arc(ArcType::transition_to_place(t, p), Some(1));
        net.initial_marking = Some(Marking::from([(p, 1)]));

        let merges = net.simplify_silent_tracked();

        assert!(merges.is_empty(), "a self-loop drop is not a place merge");
        assert!(net.transitions.is_empty());
        assert!(net.arcs.is_empty());
        assert!(net.places.contains_key(&p.get_uuid()), "the marked place survives");
    }

    #[test]
    fn a_duplicate_tau_is_removed_but_a_labeled_twin_is_not() {
        // Two silent transitions with the same pre/post are one fact stated twice; a
        // labeled transition with the same pre/post is a silent alternative to a real
        // step, and removing it would force every trace onto the labeled route. p0/p1
        // each get a third neighbour so neither is exclusive to a tau, which keeps the
        // series-fusion rule from firing first and merging the taus' endpoints away
        // before duplicate detection gets a look at them.
        let mut net = PetriNet::default();
        let p0 = net.add_place(None);
        let p1 = net.add_place(None);
        let t1 = net.add_transition(None, None);
        let t2 = net.add_transition(None, None);
        let labeled = net.add_transition(Some("X".into()), None);
        for t in [t1, t2, labeled] {
            net.add_arc(ArcType::place_to_transition(p0, t), Some(1));
            net.add_arc(ArcType::transition_to_place(t, p1), Some(1));
        }

        net.simplify_silent();

        assert_eq!(net.transitions.len(), 2, "one duplicate tau drops, the labeled one never does");
        assert!(net.transitions.contains_key(&labeled.get_uuid()));
        let taus_left = [t1.get_uuid(), t2.get_uuid()]
            .into_iter()
            .filter(|id| net.transitions.contains_key(id))
            .count();
        assert_eq!(taus_left, 1);
    }
}
