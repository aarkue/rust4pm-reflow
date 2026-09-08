//! Binding semantics for object-centric Petri nets.
//!
//! A transition of an [`ObjectCentricPetriNet`] does not fire on its own: it fires on a
//! *binding*, which names one object per incident place type (or a set of them, where the
//! arc is variable). Firing consumes those objects from the input places and produces them
//! into the output places. That is the whole difference between an OCPN and a stack of
//! Petri nets, and it is what a per-type flattened analysis cannot observe: flattening on
//! `packages` sees that a package was sent, never that the items sent with it were *its*
//! items.
//!
//! Definitions follow Adams and van der Aalst, *Precision and Fitness in Object-Centric
//! Process Mining* (Defs. 4-6).

use std::collections::{HashMap, HashSet};

use crate::core::process_models::{
    case_centric::petri_net::{PlaceID, TransitionID},
    object_centric::ocpn::ObjectCentricPetriNet,
};

/// One token: an object sitting in a place. Places are typed, so the object's type and the
/// place's type always agree; the pair is what a binding consumes and produces.
pub type Token = (PlaceID, ObjectId);

/// An object, by identity. Kept as a string so a marking can be built straight from an
/// OCEL without an index side-table.
pub type ObjectId = String;

/// Which objects sit in which places. A multiset in the definition; a set here, because an
/// object is in a place or it is not -- an OCPN never holds two tokens of the same object
/// in one place.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OcMarking {
    tokens: HashSet<Token>,
}

impl OcMarking {
    /// The marking with one token per object in its type's source places.
    pub fn initial(net: &ObjectCentricPetriNet, objects: &[(ObjectId, String)]) -> Self {
        let mut tokens = HashSet::new();
        for (ot, sub) in net.nets.iter() {
            let Some(init) = sub.initial_marking.as_ref() else { continue };
            for (place, _) in init.iter() {
                for (o, o_type) in objects.iter().filter(|(_, t)| t == ot) {
                    tokens.insert((*place, o.clone()));
                }
            }
        }
        Self { tokens }
    }

    pub fn holds(&self, place: PlaceID, object: &str) -> bool {
        self.tokens.contains(&(place, object.to_string()))
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The marking seen by one context: only the tokens of its objects.
    ///
    /// Every other object's tokens have to go, or its source place makes its first activity
    /// look enabled at every event in the log.
    pub fn restricted_to(&self, objects: &HashSet<ObjectId>) -> Self {
        Self {
            tokens: self
                .tokens
                .iter()
                .filter(|(_, o)| objects.contains(o))
                .cloned()
                .collect(),
        }
    }

    /// Where one object currently sits, as a sorted key for cycle detection.
    pub fn tokens_of_object(&self, object: &str) -> Vec<PlaceID> {
        let mut out: Vec<PlaceID> = self
            .tokens
            .iter()
            .filter(|(_, o)| o == object)
            .map(|(p, _)| *p)
            .collect();
        out.sort_by_key(|p| p.0);
        out
    }

    /// The objects sitting in one place.
    pub fn tokens_of(&self, place: PlaceID) -> Vec<(ObjectId, PlaceID)> {
        self.tokens
            .iter()
            .filter(|(p, _)| *p == place)
            .map(|(p, o)| (o.clone(), *p))
            .collect()
    }

    /// Fire a binding: consume the named objects from the preset, produce them into the
    /// postset. `None` if the binding is not enabled here.
    ///
    /// One activity is one transition of the object-centric net, but each sub-net may carry
    /// the label on **several** transitions -- a mined net routinely does, where a
    /// hand-drawn one does not. The binding therefore *chooses* one transition per sub-net,
    /// and is enabled when some choice is. Unioning their presets instead demands the token
    /// be in every alternative's input place at once, which nothing satisfies.
    pub fn fire(&self, net: &OcNet, binding: &Binding) -> Option<Self> {
        let chosen = self.choose(net, binding)?;
        let mut out = self.clone();
        for t in &chosen {
            for (place, ot) in net.preset(*t) {
                for o in binding.objects_of(ot) {
                    out.tokens.remove(&(*place, o.clone()));
                }
            }
            for (place, ot) in net.postset(*t) {
                for o in binding.objects_of(ot) {
                    out.tokens.insert((*place, o.clone()));
                }
            }
        }
        Some(out)
    }

    /// One transition per sub-net whose preset this marking satisfies, or `None`.
    pub fn choose(&self, net: &OcNet, binding: &Binding) -> Option<Vec<TransitionID>> {
        match &binding.activity {
            BindingTarget::Silent(t) => {
                self.satisfies(net, binding, *t).then(|| vec![*t])
            }
            BindingTarget::Activity(a) => {
                let by_type = net.transitions_by_type(a);
                if by_type.is_empty() {
                    return None;
                }
                let mut out = Vec::new();
                for (_, candidates) in by_type {
                    let pick = candidates
                        .into_iter()
                        .find(|t| self.satisfies(net, binding, *t))?;
                    out.push(pick);
                }
                Some(out)
            }
        }
    }

    fn satisfies(&self, net: &OcNet, binding: &Binding, t: TransitionID) -> bool {
        let pre = net.preset(t);
        if pre.is_empty() {
            return false;
        }
        pre.iter().all(|(place, ot)| {
            let named = binding.objects_of(ot);
            if named.is_empty() {
                return false;
            }
            if !net.is_variable(&binding.activity, ot) && named.len() != 1 {
                return false;
            }
            named.iter().all(|o| self.holds(*place, o))
        })
    }

    /// Def. 6: enabled when every input place holds every object the binding names for its
    /// type, and the binding names exactly **one** object for each type whose arc is not
    /// variable (`|b(ot)| = 1` for `ot in tpl_nv(t)`).
    ///
    /// The variable check is not decoration. A variable arc is how an OCPN says "all the
    /// items of this order at once"; a plain arc says "one object". Ignoring the difference
    /// accepts an event naming three items at a single-object arc, which is behaviour the
    /// model forbids, and makes it look more permissive than it is.
    pub fn enables(&self, net: &OcNet, binding: &Binding) -> bool {
        self.choose(net, binding).is_some()
    }
}

/// An activity together with the objects it acts on, one set per object type.
///
/// Keyed on the **activity**, not on a transition: one activity is one transition of the
/// object-centric net, stored as several transitions across the per-type sub-nets. Binding
/// to a single `TransitionID` would consume from one object type and leave the others
/// untouched, which is exactly the synchronisation the measure exists to see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub activity: BindingTarget,
    pub objects: HashMap<String, Vec<ObjectId>>,
}

/// What a binding fires: a labelled activity across every sub-net, or one silent transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingTarget {
    Activity(String),
    Silent(TransitionID),
}

impl Binding {
    pub fn objects_of(&self, object_type: &str) -> &[ObjectId] {
        self.objects.get(object_type).map_or(&[], Vec::as_slice)
    }
}

/// The net, indexed the way binding execution needs it.
///
/// [`ObjectCentricPetriNet`] stores one flat net per type and stitches them on shared
/// transition labels, which is right for storage and wrong for firing: a binding needs the
/// places of *every* type incident to a label at once.
pub struct OcNet<'a> {
    pub net: &'a ObjectCentricPetriNet,
    /// Place -> the object type whose sub-net it belongs to.
    place_type: HashMap<PlaceID, String>,
    /// Activity label -> the transitions carrying it, across all sub-nets.
    by_label: HashMap<String, Vec<TransitionID>>,
    preset: HashMap<TransitionID, Vec<(PlaceID, String)>>,
    postset: HashMap<TransitionID, Vec<(PlaceID, String)>>,
}

impl<'a> OcNet<'a> {
    pub fn build(net: &'a ObjectCentricPetriNet) -> Self {
        let mut place_type = HashMap::new();
        let mut by_label: HashMap<String, Vec<TransitionID>> = HashMap::new();
        let mut preset: HashMap<TransitionID, Vec<(PlaceID, String)>> = HashMap::new();
        let mut postset: HashMap<TransitionID, Vec<(PlaceID, String)>> = HashMap::new();

        for (ot, sub) in net.nets.iter() {
            for p in sub.places.keys() {
                place_type.insert(PlaceID(*p), ot.clone());
            }
            for (id, tr) in sub.transitions.iter() {
                let t = TransitionID(*id);
                if let Some(label) = tr.label.as_ref() {
                    by_label.entry(label.clone()).or_default().push(t);
                }
                preset.insert(
                    t,
                    sub.preset_of_transition(t)
                        .into_iter()
                        .map(|p| (p, ot.clone()))
                        .collect(),
                );
                postset.insert(
                    t,
                    sub.postset_of_transition(t)
                        .into_iter()
                        .map(|p| (p, ot.clone()))
                        .collect(),
                );
            }
        }
        Self {
            net,
            place_type,
            by_label,
            preset,
            postset,
        }
    }

    /// The transitions carrying an activity, grouped by the sub-net they belong to.
    pub fn transitions_by_type(&self, activity: &str) -> Vec<(String, Vec<TransitionID>)> {
        let mut out: HashMap<String, Vec<TransitionID>> = HashMap::new();
        for t in self.transitions_of(activity) {
            if let Some((_, ot)) = self.preset(*t).first() {
                out.entry(ot.clone()).or_default().push(*t);
            }
        }
        out.into_iter().collect()
    }

    /// Whether the arc between this object type and this activity consumes a whole set.
    ///
    /// Silent transitions are per-object by construction, so they are never variable.
    pub fn is_variable(&self, target: &BindingTarget, object_type: &str) -> bool {
        match target {
            BindingTarget::Silent(_) => false,
            BindingTarget::Activity(a) => self
                .net
                .variable_arcs
                .get(object_type)
                .is_some_and(|acts| acts.contains(a)),
        }
    }

    pub fn type_of_place(&self, place: PlaceID) -> Option<&str> {
        self.place_type.get(&place).map(String::as_str)
    }

    pub fn preset(&self, t: TransitionID) -> &[(PlaceID, String)] {
        self.preset.get(&t).map_or(&[], Vec::as_slice)
    }

    pub fn postset(&self, t: TransitionID) -> &[(PlaceID, String)] {
        self.postset.get(&t).map_or(&[], Vec::as_slice)
    }

    /// Input places of a binding target: for an activity, the union over every sub-net that
    /// carries the label.
    pub fn preset_of(&self, target: &BindingTarget) -> Vec<(PlaceID, String)> {
        match target {
            BindingTarget::Silent(t) => self.preset(*t).to_vec(),
            BindingTarget::Activity(a) => self
                .transitions_of(a)
                .iter()
                .flat_map(|t| self.preset(*t).iter().cloned())
                .collect(),
        }
    }

    /// Output places of a binding target.
    pub fn postset_of(&self, target: &BindingTarget) -> Vec<(PlaceID, String)> {
        match target {
            BindingTarget::Silent(t) => self.postset(*t).to_vec(),
            BindingTarget::Activity(a) => self
                .transitions_of(a)
                .iter()
                .flat_map(|t| self.postset(*t).iter().cloned())
                .collect(),
        }
    }

    /// Every transition carrying this activity, in every sub-net. One activity is one
    /// transition of the object-centric net, split across the per-type storage.
    pub fn transitions_of(&self, activity: &str) -> &[TransitionID] {
        self.by_label.get(activity).map_or(&[], Vec::as_slice)
    }

    /// Places reachable from each place by firing **silent transitions only**, per sub-net.
    ///
    /// Computed once from the net, because it depends on nothing else. Asking it per object
    /// per event with a fresh search is what dominates the replay: 23,884 searches over
    /// 430,000 states for a hundred events. As a closure it is a set lookup.
    pub fn silent_reachability(&self) -> HashMap<PlaceID, HashSet<PlaceID>> {
        let mut step: HashMap<PlaceID, HashSet<PlaceID>> = HashMap::new();
        for (_, sub) in self.net.nets.iter() {
            for (id, tr) in sub.transitions.iter() {
                if tr.label.is_some() {
                    continue;
                }
                let t = TransitionID(*id);
                let post: Vec<PlaceID> = sub.postset_of_transition(t);
                for p in sub.preset_of_transition(t) {
                    step.entry(p).or_default().extend(post.iter().copied());
                }
            }
        }
        // Transitive closure. Nets here are small; a Floyd-Warshall style pass is simpler
        // than anything cleverer and runs once.
        let places: Vec<PlaceID> = self.place_type.keys().copied().collect();
        let mut reach: HashMap<PlaceID, HashSet<PlaceID>> = places
            .iter()
            .map(|p| (*p, step.get(p).cloned().unwrap_or_default()))
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for p in &places {
                let mut add: HashSet<PlaceID> = HashSet::new();
                if let Some(direct) = reach.get(p) {
                    for q in direct {
                        if let Some(further) = reach.get(q) {
                            for r in further {
                                if !direct.contains(r) {
                                    add.insert(*r);
                                }
                            }
                        }
                    }
                }
                if !add.is_empty() {
                    reach.entry(*p).or_default().extend(add);
                    changed = true;
                }
            }
        }
        for p in &places {
            reach.entry(*p).or_default().insert(*p);
        }
        reach
    }

    /// Silent transitions, which a replay may fire freely to reach a state.
    pub fn silent(&self) -> Vec<(TransitionID, &str)> {
        let mut out = Vec::new();
        for (ot, sub) in self.net.nets.iter() {
            for (id, tr) in sub.transitions.iter() {
                if tr.label.is_none() {
                    out.push((TransitionID(*id), ot.as_str()));
                }
            }
        }
        out
    }
}

/// The schema's object-to-object maps, as a binding constraint.
///
/// An object-centric Petri net cannot say that the items consumed by `pay order` are *that
/// order's* items: its arcs are typed, not related. The tokens are interchangeable, which is
/// why even a correct OCPN scores far below one -- the model permits an order to be paid
/// with any items at all.
///
/// A structural schema does say it. Carrying the maps alongside the net turns a binding from
/// "one token of each incident type" into "one token of each incident type, *related to each
/// other*", which forbids a large class of behaviour no OCPN can rule out. The difference in
/// precision between the two readings is what the schema is worth.
#[derive(Debug, Clone, Default)]
pub struct ObjectRelations {
    /// `(source type, target type) -> (source object -> its target object)`. Total maps, so
    /// every source object has exactly one image.
    forward: HashMap<(String, String), HashMap<ObjectId, ObjectId>>,
}

impl ObjectRelations {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a total map from `source` objects to `target` objects.
    pub fn add_map(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        f: HashMap<ObjectId, ObjectId>,
    ) {
        self.forward.insert((source.into(), target.into()), f);
    }

    /// Objects of `target` that `object` relates to, in either direction: its image under a
    /// map out of `source`, or its fibre under a map into it.
    pub fn related(&self, source: &str, object: &str, target: &str) -> Option<Vec<ObjectId>> {
        if let Some(f) = self.forward.get(&(source.to_string(), target.to_string())) {
            return f.get(object).map(|t| vec![t.clone()]);
        }
        if let Some(f) = self.forward.get(&(target.to_string(), source.to_string())) {
            let fibre: Vec<ObjectId> = f
                .iter()
                .filter(|(_, image)| image.as_str() == object)
                .map(|(pre, _)| pre.clone())
                .collect();
            return Some(fibre);
        }
        None
    }

    /// Is this binding schema-consistent? Every object of a related type must be the one the
    /// map names. Types the schema says nothing about are unconstrained, exactly as in a
    /// plain net.
    pub fn admits(&self, binding: &Binding) -> bool {
        for (s_type, s_objs) in &binding.objects {
            for (t_type, t_objs) in &binding.objects {
                if s_type == t_type {
                    continue;
                }
                let Some(f) = self.forward.get(&(s_type.clone(), t_type.clone())) else {
                    continue;
                };
                for s in s_objs {
                    match f.get(s) {
                        Some(image) if t_objs.contains(image) => {}
                        // The map is total, so an object with no image here is one the
                        // binding pairs with the wrong partner.
                        _ => return false,
                    }
                }
            }
        }
        true
    }
}
