//! Precision by perturbation: generate behaviour the process cannot produce, and see which
//! models reject it.
//!
//! A flattened conformance measure scores each object type's component on its own, so it
//! cannot see the constraint that makes an object-centric model more than a stack of Petri
//! nets: that `pay order` consumes *that order's* items. Neither can an object-centric Petri
//! net express it -- its arcs are typed, not related. A structural schema can, and this is
//! how to measure the difference.
//!
//! The method is artificial negative events. From the recorded log we take small sub-logs,
//! apply edits the process cannot produce, and ask each model to replay the result. A model
//! that accepts a perturbation permits behaviour the log never shows; the share it rejects
//! is the number reported.
//!
//! # Why fragments, and how they are bounded
//!
//! Closing over co-participation without a bound gives the whole log back: `employees` and
//! `customers` touch everything, so one order's closure reached 5,218 of Order Management's
//! events. Excluding resources fixes the size and ruins the experiment, because resources
//! are exactly what the reduction demotes -- an evaluation that drops them tests only the
//! types we barely touched.
//!
//! So every type closes, and expansion is bounded instead: each object contributes the
//! shortest prefix of its events that covers `k` occurrences of each of its activities
//! **and ends on an activity objects of its type are observed to end on**. Acyclic objects
//! come in whole, since `k` never binds on them. Repeating ones are cut at a legitimate
//! stopping point rather than mid-cycle, so a fragment is always a log the process could
//! have produced -- if a customer's cycle is `place order, confirm order`, the cut never
//! leaves it owing the confirmation.
//!
//! `k` is a parameter and the results are reported across several values: flat rejection
//! rates mean the bound is harmless, and moving ones mean it has to be justified.

use std::collections::{HashMap, HashSet};

use crate::core::process_models::object_centric::ocpn::ObjectCentricPetriNet;

use super::{
    binding_semantics::{Binding, BindingTarget, ObjectId, ObjectRelations},
    oc_precision::OcEvent,
};
use crate::conformance::case_centric::alignments::{align_trace, AlignmentOptions};

/// Activities objects of a type are observed to start and end on.
///
/// Read off the log, and the reason a truncated object still looks like a complete one.
/// They are also what an accepting net encodes in its initial and final markings, so a
/// discovered net that disagrees with them has a defect worth knowing about.
#[derive(Debug, Clone, Default)]
pub struct TypeBoundaries {
    pub start: HashMap<String, HashSet<String>>,
    pub end: HashMap<String, HashSet<String>>,
}

impl TypeBoundaries {
    pub fn of(log: &[OcEvent]) -> Self {
        let mut first: HashMap<ObjectId, (String, String)> = HashMap::new();
        let mut last: HashMap<ObjectId, (String, String)> = HashMap::new();
        for ev in log {
            for (o, ot) in &ev.objects {
                first
                    .entry(o.clone())
                    .or_insert_with(|| (ot.clone(), ev.activity.clone()));
                last.insert(o.clone(), (ot.clone(), ev.activity.clone()));
            }
        }
        let mut out = Self::default();
        for (ot, a) in first.into_values() {
            out.start.entry(ot).or_default().insert(a);
        }
        for (ot, a) in last.into_values() {
            out.end.entry(ot).or_default().insert(a);
        }
        out
    }

    fn ends(&self, object_type: &str, activity: &str) -> bool {
        self.end
            .get(object_type)
            .is_some_and(|s| s.contains(activity))
    }

    fn starts(&self, object_type: &str, activity: &str) -> bool {
        self.start
            .get(object_type)
            .is_some_and(|s| s.contains(activity))
    }
}

/// A sub-log: events in order, and the objects they involve.
#[derive(Debug, Clone)]
pub struct Fragment {
    pub events: Vec<OcEvent>,
    pub objects: HashSet<ObjectId>,
}

/// Take the bounded closure of one object.
///
/// `k` is a floor on how much of each object's behaviour is included, not a hard cut: the
/// prefix is extended until the object stops on an activity its type ends on. `max_objects`
/// is a safety valve for a log where even the bounded closure runs away; hitting it returns
/// what has been collected, which is a smaller fragment and not a wrong one.
///
/// # Why there are two bounds
///
/// `k` bounds how much of *one* object is taken, and says nothing about how many objects the
/// closure reaches. Co-participation through a shared object is transitive: an order reaches
/// its products, and a product reaches every other order that ever used it. Left unbounded,
/// one hop returns the log -- every seed gave back the same 107-event blob, a quarter of
/// Order Management rather than a fragment of it.
///
/// `radius` bounds that breadth in hops, and the two knobs then mean different things. A
/// product serving two independent orders is behaviour the process really produces, so it
/// belongs in a fragment: `k = 2` is what puts it there, and the radius decides how much of
/// those orders comes with it. Neither bound singles out a type, which matters because which
/// types are shared is a property of the log and not something an evaluation should assert.
pub fn extract(
    log: &[OcEvent],
    seed: &ObjectId,
    k: usize,
    bounds: &TypeBoundaries,
    radius: usize,
    max_objects: usize,
) -> Fragment {
    // Per object, the indices of the events touching it, in order.
    let mut touching: HashMap<&ObjectId, Vec<usize>> = HashMap::new();
    let mut type_of: HashMap<&ObjectId, &str> = HashMap::new();
    for (i, ev) in log.iter().enumerate() {
        for (o, ot) in &ev.objects {
            touching.entry(o).or_default().push(i);
            type_of.insert(o, ot);
        }
    }

    // The object's slice: a window of its own events, anchored where the closure reached it.
    //
    // Anchoring is what makes the slice the *seed's* neighbourhood. A customer's events run
    // over hundreds of orders, so its earliest `k` cycles are almost never the ones the seed
    // takes part in; a prefix from the object's global start would quietly relocate the
    // fragment to a different order and make every seed of one customer give one answer.
    // From the anchor, the slice runs back to an activity the type starts on and forward to
    // one it ends on, so it is a whole number of that object's cycles either way.
    let slice_of = |o: &ObjectId, anchor: usize| -> (usize, usize) {
        let Some(idx) = touching.get(o) else {
            return (0, 0);
        };
        let ot = type_of.get(o).copied().unwrap_or("");
        let at = idx.partition_point(|j| *j < anchor).min(idx.len() - 1);
        let mut lo = at;
        while lo > 0 && !bounds.starts(ot, &log[idx[lo]].activity) {
            lo -= 1;
        }

        let mut count: HashMap<&str, usize> = HashMap::new();
        let mut satisfied = false;
        let mut hi = lo;
        for (pos, i) in idx.iter().enumerate().skip(lo) {
            let a = log[*i].activity.as_str();
            *count.entry(a).or_default() += 1;
            if count.values().all(|c| *c >= k) {
                satisfied = true;
            }
            hi = pos + 1;
            // `k` is a floor, not a cut: stop at the first activity the type is observed to
            // end on at or after the coverage point, so a truncated object still looks like
            // a complete one rather than owing the rest of its cycle. At `k = 2` a customer
            // contributes two whole orders, and a product two independent ones.
            if satisfied && bounds.ends(ot, a) {
                break;
            }
        }
        (lo, hi)
    };

    let mut objects: HashSet<ObjectId> = [seed.clone()].into_iter().collect();
    let seed_anchor = touching.get(seed).and_then(|i| i.first().copied()).unwrap_or(0);
    let mut frontier: Vec<(ObjectId, usize, usize)> = vec![(seed.clone(), seed_anchor, 0)];
    let mut cuts: HashMap<ObjectId, (usize, usize)> = HashMap::new();

    // Objects strictly inside the radius are *interior*: the fragment carries their whole
    // slice and they constrain which events it may keep. Objects reached at the radius are
    // boundary, and are dropped from the events instead.
    //
    // The split is what stops the fixpoint below from cascading to nothing. A product's slice
    // spans orders the radius never reached, so demanding that slice be complete deletes the
    // events that demanded it, then their neighbours, until the fragment is empty -- which is
    // exactly what a single tier produced. A boundary object is context the fragment does not
    // claim to explain, so it makes no demand at all.
    while let Some((o, anchor, hops)) = frontier.pop() {
        if cuts.contains_key(&o) || hops >= radius {
            continue;
        }
        let (lo, hi) = slice_of(&o, anchor);
        cuts.insert(o.clone(), (lo, hi));
        let Some(idx) = touching.get(&o) else { continue };
        for i in &idx[lo..hi] {
            for (o2, _) in &log[*i].objects {
                if objects.len() >= max_objects && !objects.contains(o2) {
                    continue;
                }
                objects.insert(o2.clone());
                if !cuts.contains_key(o2) {
                    // Anchored at the event we came through, so the neighbour's slice is the
                    // part of its life that overlaps this fragment.
                    frontier.push((o2.clone(), *i, hops + 1));
                }
            }
        }
    }

    // What a fragment has to guarantee is that every interior object can be *replayed*: its
    // projection starts on an activity its type starts on, ends on one its type ends on, and
    // has no hole in the middle. That is the property, and enforcing it directly is what
    // keeps fragments alive.
    //
    // Requiring instead that each object's whole slice survive collapses the fragment to
    // nothing. Slices are anchored independently, so one object's slice routinely reaches
    // events another's does not cover; under a containment rule each such event is deleted,
    // which orphans its neighbours, and the cascade runs to empty -- at `k = 2` it did.
    //
    // Trimming is monotone -- events are only ever removed -- so the loop terminates, and
    // every removal is justified by a named object rather than by bookkeeping.
    let mut chosen: HashSet<usize> = HashSet::new();
    for (o, (lo, hi)) in &cuts {
        if let Some(idx) = touching.get(o) {
            chosen.extend(idx[*lo..*hi].iter().copied());
        }
    }
    loop {
        let mut drop: HashSet<usize> = HashSet::new();
        for (o, (lo, hi)) in &cuts {
            let Some(idx) = touching.get(o) else { continue };
            let ot = type_of.get(o).copied().unwrap_or("");
            let mut proj: Vec<usize> = idx[*lo..*hi]
                .iter()
                .copied()
                .filter(|i| chosen.contains(i) && !drop.contains(i))
                .collect();
            // A hole truncates the projection: everything after the first missing event of
            // this object goes, so what remains is contiguous in the object's own history.
            if let Some(gap) = idx[*lo..*hi]
                .iter()
                .position(|i| !chosen.contains(i) || drop.contains(i))
            {
                for i in idx[*lo..*hi].iter().skip(gap) {
                    if chosen.contains(i) {
                        drop.insert(*i);
                    }
                }
                proj.truncate(gap);
            }
            while proj
                .first()
                .is_some_and(|i| !bounds.starts(ot, &log[*i].activity))
            {
                drop.insert(proj.remove(0));
            }
            while proj
                .last()
                .is_some_and(|i| !bounds.ends(ot, &log[*i].activity))
            {
                drop.insert(proj.pop().unwrap());
            }
        }
        if drop.is_empty() {
            break;
        }
        for i in drop {
            chosen.remove(&i);
        }
    }

    let mut order: Vec<usize> = chosen.into_iter().collect();
    order.sort_unstable();
    objects.retain(|o| cuts.contains_key(o));
    let events: Vec<OcEvent> = order
        .into_iter()
        .map(|i| OcEvent {
            activity: log[i].activity.clone(),
            objects: log[i]
                .objects
                .iter()
                .filter(|(o, _)| cuts.contains_key(o))
                .cloned()
                .collect(),
        })
        .collect();
    Fragment { events, objects }
}

/// The families of edit, each isolating a different kind of over-permissiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edit {
    /// Replace one object of the fragment, throughout, by an object of the same type from
    /// elsewhere in the log. The one family a flattened measure and a plain object-centric
    /// net are both blind to.
    ///
    /// The substitution is total rather than at a single event, and that is what makes it a
    /// fair test. Replacing an object everywhere leaves every per-type projection character
    /// for character as it was, so no sub-net can distinguish the result from the original --
    /// not because the nets happen to be permissive, but because they are shown the identical
    /// sequence. An earlier version swapped at one event and rejected almost everything,
    /// which only measured that the substitute was at the wrong point in its own lifecycle.
    Reassign,
    /// Swap two consecutive events of one object. Tests ordering within a type.
    Reorder,
    /// Remove an event. Tests what the model insists upon.
    Drop,
    /// Repeat an event of the fragment at another position. Tests permissiveness.
    Insert,
}

/// Apply one edit deterministically, returning `None` where the fragment has no site for it.
///
/// Deterministic on `nth` rather than random, so a run is reproducible and the perturbed
/// fragments can be deposited alongside the numbers.
pub fn perturb(
    log: &[OcEvent],
    fragment: &Fragment,
    edit: Edit,
    nth: usize,
) -> Option<Fragment> {
    let mut events = fragment.events.clone();
    let mut objects = fragment.objects.clone();
    match edit {
        Edit::Reassign => {
            let mut present: Vec<(ObjectId, String)> = Vec::new();
            for ev in &fragment.events {
                for (o, ot) in &ev.objects {
                    if !present.iter().any(|(p, _)| p == o) {
                        present.push((o.clone(), ot.clone()));
                    }
                }
            }
            present.sort();
            let (victim, ot) = present.get(nth % present.len().max(1))?.clone();
            let foreign = log
                .iter()
                .flat_map(|e| e.objects.iter())
                .find(|(o, t)| *t == ot && !fragment.objects.contains(o))
                .map(|(o, _)| o.clone())?;
            for ev in &mut events {
                for (o, _) in &mut ev.objects {
                    if *o == victim {
                        *o = foreign.clone();
                    }
                }
            }
            objects.remove(&victim);
            objects.insert(foreign);
        }
        Edit::Reorder => {
            let mut sites: Vec<usize> = Vec::new();
            for i in 0..events.len().saturating_sub(1) {
                // Only where the two events share an object: swapping independent events is
                // not necessarily behaviour the process forbids.
                if events[i]
                    .objects
                    .iter()
                    .any(|(o, _)| events[i + 1].objects.iter().any(|(p, _)| p == o))
                {
                    sites.push(i);
                }
            }
            let i = *sites.get(nth % sites.len().max(1))?;
            events.swap(i, i + 1);
        }
        Edit::Drop => {
            if events.is_empty() {
                return None;
            }
            events.remove(nth % events.len());
        }
        Edit::Insert => {
            if events.is_empty() {
                return None;
            }
            let from = nth % events.len();
            let at = (nth * 7 + 1) % (events.len() + 1);
            let copy = events[from].clone();
            events.insert(at, copy);
        }
    }
    Some(Fragment {
        events,
        objects,
    })
}

/// Can the model replay this fragment?
///
/// # Why this is per object and not a joint search
///
/// The log fixes every binding, and that collapses the problem. Sub-nets of an
/// object-centric net interact only at shared transitions, and each object's tokens live in
/// its own places, so no object's silent moves can obstruct another's. With the event order
/// fixed by the fragment, the joint replay succeeds exactly when **every object can follow
/// its own projected sequence** in its type's component.
///
/// So this is per-object case-centric alignment on the flattened projection, at cost zero.
/// The alternative -- pushing tokens greedily through the joint marking -- cannot handle a
/// branching tau skeleton: after `place order` an item may silently reach either its final
/// place or the rest of the process, and a greedy walk commits to one with no way back. An
/// aligner searches that branch space properly, which is what it is for.
///
/// The schema constraint is checked separately, because relatedness is not a Petri-net
/// question: no arc can express that these items belong to that order.
pub fn replays(
    net: &ObjectCentricPetriNet,
    fragment: &Fragment,
    relations: Option<&ObjectRelations>,
) -> bool {
    cost(net, fragment, relations).is_some_and(|c| c == 0)
}

/// The model's alignment cost for this fragment, summed over objects, or `None` where the
/// schema forbids a binding outright.
///
/// Cost rather than a yes/no, because a fragment truncates objects mid-life by construction
/// -- a resource cannot be included whole, which is why expansion is bounded at all -- so no
/// fragment reaches every final marking and an absolute test would reject all of them. The
/// perturbation test is therefore **relative**: an edit is rejected when it raises the cost
/// above the unperturbed fragment's. Truncation affects both equally and cancels.
pub fn cost(
    net: &ObjectCentricPetriNet,
    fragment: &Fragment,
    relations: Option<&ObjectRelations>,
) -> Option<u64> {
    if let Some(rel) = relations {
        for ev in &fragment.events {
            let mut by_type: HashMap<String, Vec<ObjectId>> = HashMap::new();
            for (o, ot) in &ev.objects {
                by_type.entry(ot.clone()).or_default().push(o.clone());
            }
            let binding = Binding {
                activity: BindingTarget::Activity(ev.activity.clone()),
                objects: by_type,
            };
            if !rel.admits(&binding) {
                return None;
            }
        }
    }

    // Each object's projection: the activities of the fragment's events it takes part in.
    let mut projection: HashMap<&ObjectId, (&str, Vec<&str>)> = HashMap::new();
    for ev in &fragment.events {
        for (o, ot) in &ev.objects {
            projection
                .entry(o)
                .or_insert_with(|| (ot.as_str(), Vec::new()))
                .1
                .push(ev.activity.as_str());
        }
    }

    // The aligner's 100k default is not enough for a large component even on a four-event
    // trace -- `products` has 146 arcs and exhausts it. These traces are short, so a
    // generous ceiling costs nothing and turns a spurious failure into an answer.
    let options = AlignmentOptions {
        max_states: Some(4_000_000),
        ..AlignmentOptions::default()
    };
    let mut total = 0u64;
    for (_, (object_type, trace)) in projection {
        let Some(sub) = net.nets.get(object_type) else {
            if std::env::var("PB_WHY").is_ok() {
                eprintln!("  no component for type {object_type}");
            }
            // No component for this type at all: the model cannot account for the object,
            // which is the most expensive outcome there is.
            return None;
        };
        match align_trace(sub, &trace, &options) {
            Ok(result) => total += result.cost as u64,
            Err(e) => {
                if std::env::var("PB_WHY").is_ok() {
                    eprintln!("  [{object_type}] align error {e:?} for {:?}", trace);
                }
                return None;
            }
        }
    }
    Some(total)
}

/// Rejection rates, per edit family.
#[derive(Debug, Clone, Default)]
pub struct PerturbationResult {
    pub tried: HashMap<&'static str, usize>,
    pub rejected: HashMap<&'static str, usize>,
}

impl PerturbationResult {
    pub fn rate(&self, family: &str) -> Option<f64> {
        let t = *self.tried.get(family)? as f64;
        (t > 0.0).then(|| *self.rejected.get(family).unwrap_or(&0) as f64 / t)
    }
}

/// Run every family over a set of fragments and report what the model rejects.
///
/// A sanity condition worth checking before reading anything else: the model must accept the
/// **unperturbed** fragments. A model that rejects those is failing for its own reasons and
/// its rejection rates say nothing about the edits.
pub fn evaluate(
    net: &ObjectCentricPetriNet,
    log: &[OcEvent],
    fragments: &[Fragment],
    relations: Option<&ObjectRelations>,
    per_fragment: usize,
) -> (PerturbationResult, usize) {
    let mut out = PerturbationResult::default();
    let mut baseline_ok = 0usize;

    for fragment in fragments {
        let Some(base) = cost(net, fragment, relations) else { continue };
        baseline_ok += 1;
        for (name, edit) in [
            ("reassign", Edit::Reassign),
            ("reorder", Edit::Reorder),
            ("drop", Edit::Drop),
            ("insert", Edit::Insert),
        ] {
            for nth in 0..per_fragment {
                let Some(bad) = perturb(log, fragment, edit, nth) else { continue };
                // An edit that happens to reproduce the original is not a negative example.
                if bad.events.len() == fragment.events.len()
                    && bad
                        .events
                        .iter()
                        .zip(&fragment.events)
                        .all(|(a, b)| a.activity == b.activity && a.objects == b.objects)
                {
                    continue;
                }
                *out.tried.entry(name).or_default() += 1;
                // Rejected when the edit costs the model something it did not already pay
                // for the unperturbed fragment.
                match cost(net, &bad, relations) {
                    Some(c) if c <= base => {}
                    _ => *out.rejected.entry(name).or_default() += 1,
                }
            }
        }
    }
    (out, baseline_ok)
}
