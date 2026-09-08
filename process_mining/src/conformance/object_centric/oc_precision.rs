//! Object-centric fitness and precision, after Adams and van der Aalst, *Precision and
//! Fitness in Object-Centric Process Mining*.
//!
//! Traditional precision asks, at each prefix of a trace, which activities the model allows
//! against which the log shows. Object-centrically there is no single prefix: an event
//! depends on several objects, each with its own history. The paper's answer is the
//! **context** of an event (Def. 8) -- everything that had to happen for it to occur, as a
//! multiset of activity sequences per object type, read off the event-object graph.
//!
//! Two sets are then compared for each event:
//!
//! - `en_L(e)` (Def. 12): activities the log executes at **any** event sharing that context.
//! - `en_OCPN(e)` (Def. 11): activities enabled in **any** marking the model reaches by a
//!   binding sequence producing that context.
//!
//! and
//!
//! ```text
//! fitness   = 1/|E|   * sum_e   |en_L(e) & en_OCPN(e)| / |en_L(e)|
//! precision = 1/|E_f| * sum_e_f |en_L(e) & en_OCPN(e)| / |en_OCPN(e)|
//! ```
//!
//! over the replayable events `E_f = {e | en_OCPN(e) != {}}`. Events the model cannot
//! replay are **skipped** for precision and counted against fitness, and their share is
//! reported rather than hidden: the paper's own restricted model skips 54% of events.
//!
//! # Why this and not a flattened measure
//!
//! Flattening scores each object type's component on its own, so it never sees that
//! `send package` needs *this* package's items rather than some other package's. Every
//! constraint that makes an object-centric net more than a stack of Petri nets is invisible
//! to it. This measure is defined on bindings, so it sees exactly that.
//!
//! # Calibration
//!
//! An object-centric net puts no constraint on the order of activities *between* object
//! types, so even a model that is correct by construction scores well below one. In the
//! paper's own evaluation the appropriate model reaches precision 0.57 against a flower
//! model's 0.25. Read these numbers against that scale, not against a flattened precision.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// Where the time goes. Counters rather than guesses: the replay has three plausible
/// bottlenecks -- the quadratic preset walk, how many contexts are actually replayed, and
/// the per-object silent search -- and each calls for a different fix.
pub static STAT_PRESET_EDGES: AtomicU64 = AtomicU64::new(0);
pub static STAT_CONTEXTS: AtomicU64 = AtomicU64::new(0);
pub static STAT_REPLAY_FIRES: AtomicU64 = AtomicU64::new(0);
pub static STAT_ADVANCE_CALLS: AtomicU64 = AtomicU64::new(0);
pub static STAT_ADVANCE_STATES: AtomicU64 = AtomicU64::new(0);
pub static STAT_ENABLED_CALLS: AtomicU64 = AtomicU64::new(0);
/// Total preset size and total context-object count, to see whether contexts stay small.
pub static STAT_PRESET_SIZE: AtomicU64 = AtomicU64::new(0);
pub static STAT_CTX_OBJECTS: AtomicU64 = AtomicU64::new(0);
pub static STAT_MAX_PRESET: AtomicU64 = AtomicU64::new(0);

/// Read the counters and reset them.
pub fn take_stats() -> [u64; 9] {
    [
        &STAT_PRESET_EDGES,
        &STAT_CONTEXTS,
        &STAT_REPLAY_FIRES,
        &STAT_ADVANCE_CALLS,
        &STAT_ADVANCE_STATES,
        &STAT_ENABLED_CALLS,
        &STAT_PRESET_SIZE,
        &STAT_CTX_OBJECTS,
        &STAT_MAX_PRESET,
    ]
    .map(|c| c.swap(0, Ordering::Relaxed))
}


use crate::core::process_models::object_centric::ocpn::ObjectCentricPetriNet;

use super::binding_semantics::{
    Binding, BindingTarget, ObjectId, ObjectRelations, OcMarking, OcNet,
};

/// One event, reduced to what the measure needs.
#[derive(Debug, Clone)]
pub struct OcEvent {
    pub activity: String,
    /// The objects the event names, with their types.
    pub objects: Vec<(ObjectId, String)>,
}

/// What a run of the measure found, including what it could not do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcConformance {
    /// Def. 13.
    pub fitness: f64,
    /// Def. 14, over replayable events only.
    pub precision: f64,
    /// Events whose context the model could not replay: no enabled activities. Reported
    /// because it is the qualifier on `precision`, not a diagnostic.
    pub skipped: usize,
    /// Events considered in total.
    pub events: usize,
}

impl OcConformance {
    /// Share of events the model could not replay.
    pub fn skipped_share(&self) -> f64 {
        if self.events == 0 {
            0.0
        } else {
            self.skipped as f64 / self.events as f64
        }
    }
}

/// How hard the replay may work before it gives up on an event.
///
/// The paper is explicit that silent transitions and the number of objects per event can
/// make the reachable-state search blow up exponentially, and singles out inductive-miner
/// nets as the case where it bites. A bound turns that into a skipped event, which the
/// result already reports, rather than a run that does not finish.
#[derive(Debug, Clone)]
pub struct OcConformanceOptions {
    /// The schema's maps, if the model carries them.
    ///
    /// Absent, an activity is enabled when tokens of the right types are available, which
    /// is all a plain object-centric net can say. Present, a binding must also relate its
    /// objects -- an order with *its* items -- which forbids behaviour no OCPN can. Running
    /// the same net and log both ways measures what the schema is worth.
    pub relations: Option<ObjectRelations>,
    /// Markings the silent-transition closure may visit per context.
    pub max_states: usize,
    /// Events to measure; `None` for all of them. Sampling is a stated approximation, and
    /// the context of an event needs the whole prefix regardless, so this bounds the
    /// *measured* events and not the replay.
    pub max_events: Option<usize>,
}

impl Default for OcConformanceOptions {
    fn default() -> Self {
        Self {
            relations: None,
            max_states: 2_000,
            max_events: None,
        }
    }
}

impl OcConformanceOptions {
    /// Bindings constrained by the schema: an activity counts as enabled only where the
    /// objects it would consume are related to one another.
    pub fn with_relations(relations: ObjectRelations) -> Self {
        Self {
            relations: Some(relations),
            ..Self::default()
        }
    }
}

/// The context of an event (Def. 8): per object type, the multiset of activity sequences of
/// the objects it depends on.
///
/// Two events with the same context are the same "state" of the log, which is what lets the
/// measure aggregate over them the way a prefix automaton aggregates over traces.
type Context = Vec<(String, Vec<Vec<String>>)>;

/// Fitness and precision of `net` with respect to `log`.
///
/// `log` is the events in time order; each carries its activity and the objects it names.
pub fn oc_conformance(
    net: &ObjectCentricPetriNet,
    log: &[OcEvent],
    options: &OcConformanceOptions,
) -> OcConformance {
    let oc = OcNet::build(net);
    let reach = oc.silent_reachability();

    // Per object, the activity sequence it has been through so far, and the events that
    // touched it. Walking the log once in order gives both, and the event preset of an
    // event is the union over its objects of what they have already seen -- which is the
    // event-object graph of Def. 8 without materialising it.
    let mut history: HashMap<ObjectId, Vec<String>> = HashMap::new();
    let mut object_type: HashMap<ObjectId, String> = HashMap::new();
    // Which earlier events touched each object, for the backward walk below.
    let mut touched: HashMap<ObjectId, Vec<usize>> = HashMap::new();

    let mut contexts: Vec<Context> = Vec::with_capacity(log.len());
    let mut presets: Vec<Vec<usize>> = Vec::with_capacity(log.len());

    for (i, ev) in log.iter().enumerate() {
        // Def. 8's event preset: every event with a *directed path* to this one in the
        // event-object graph, not merely one sharing an object. `Lift off` depends on the
        // baggage check-ins, which it shares no object with -- they reach it through
        // `Load cargo`.
        // Walk backwards over the event-object graph. An edge runs (e', e'') only when
        // e' < e'' and they share an object, so a predecessor is sought among events
        // *earlier than the one being expanded* -- not merely earlier than `e`. Dropping
        // that makes `Pick up @ dest` a predecessor of the `Clean` that follows it, because
        // both touch a bag the plane also touched, and the two contexts then collapse.
        let mut seen: HashSet<usize> = HashSet::new();
        let mut objs: HashSet<ObjectId> = ev.objects.iter().map(|(o, _)| o.clone()).collect();
        let mut frontier: VecDeque<(usize, Vec<ObjectId>)> =
            VecDeque::from([(i, ev.objects.iter().map(|(o, _)| o.clone()).collect())]);
        while let Some((at, at_objects)) = frontier.pop_front() {
            for o in at_objects {
                for j in touched.get(&o).cloned().unwrap_or_default() {
                    STAT_PRESET_EDGES.fetch_add(1, Ordering::Relaxed);
                    if j >= at || !seen.insert(j) {
                        continue;
                    }
                    let their: Vec<ObjectId> =
                        log[j].objects.iter().map(|(x, _)| x.clone()).collect();
                    objs.extend(their.iter().cloned());
                    frontier.push_back((j, their));
                }
            }
        }
        let mut preset: Vec<usize> = seen.into_iter().collect();
        preset.sort_unstable();

        // The context is over the objects of the preset *and* the event -- not the event's
        // own objects. `Pick up @ dest` for a bag and `Clean` for the plane it flew on
        // share every object and every history, so they are one state of the log and their
        // activities pool. Keying on the event's own objects splits them and makes each
        // look imprecise.
        // Each object's sequence is projected over the **preset**, not over everything that
        // happened earlier: Def. 8 reads the activities of `{e' in preset | o in omap(e')}`.
        // `Clean` is not in the preset of the `Pick up @ dest` that follows it, so the
        // plane's sequence there must not contain it -- and that is precisely what makes
        // the two events one context, which is what pools their activities.
        let mut seq: HashMap<&ObjectId, Vec<String>> = HashMap::new();
        for j in &preset {
            for (o, _) in &log[*j].objects {
                if objs.contains(o) {
                    seq.entry(o).or_default().push(log[*j].activity.clone());
                }
            }
        }
        let mut ctx: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for o in &objs {
            let Some(ot) = object_type
                .get(o)
                .cloned()
                .or_else(|| ev.objects.iter().find(|(x, _)| x == o).map(|(_, t)| t.clone()))
            else {
                continue;
            };
            ctx.entry(ot)
                .or_default()
                .push(seq.get(o).cloned().unwrap_or_default());
        }
        let mut ctx: Context = ctx.into_iter().collect();
        ctx.sort();
        for (_, seqs) in ctx.iter_mut() {
            seqs.sort();
        }
        STAT_PRESET_SIZE.fetch_add(preset.len() as u64, Ordering::Relaxed);
        STAT_CTX_OBJECTS.fetch_add(objs.len() as u64, Ordering::Relaxed);
        STAT_MAX_PRESET.fetch_max(preset.len() as u64, Ordering::Relaxed);
        contexts.push(ctx);
        presets.push(preset);

        for (o, ot) in &ev.objects {
            object_type.insert(o.clone(), ot.clone());
            history.entry(o.clone()).or_default().push(ev.activity.clone());
            touched.entry(o.clone()).or_default().push(i);
        }
    }

    // Def. 12: what the log does at every event sharing a context.
    let mut en_log: HashMap<&Context, HashSet<&str>> = HashMap::new();
    for (i, ev) in log.iter().enumerate() {
        en_log
            .entry(&contexts[i])
            .or_default()
            .insert(ev.activity.as_str());
    }


    // Def. 11, via Algorithm 1: replay the event's binding sequence, then read off what is
    // enabled. Cached per context, since that is the unit the definitions aggregate over.
    let mut en_model: HashMap<Context, HashSet<String>> = HashMap::new();

    let n = options.max_events.unwrap_or(log.len()).min(log.len());
    let mut fit_sum = 0.0;
    let mut prec_sum = 0.0;
    let mut replayable = 0usize;
    let mut skipped = 0usize;

    // One forward pass, one marking. The state after replaying an event's preset is the
    // global state after every earlier event, restricted to that event's objects -- any
    // event touching one of them is in the preset by definition, and no other event can
    // have moved their tokens. So the per-context replay that made this quadratic is
    // redundant: walk the log once and project.
    let all: Vec<(ObjectId, String)> = object_type.iter().map(|(o, t)| (o.clone(), t.clone())).collect();
    let mut marking = OcMarking::initial(net, &all);

    for i in 0..n {
        let ctx = &contexts[i];
        let ctx_objects: HashSet<ObjectId> = presets[i]
            .iter()
            .copied()
            .chain(std::iter::once(i))
            .flat_map(|j| log[j].objects.iter().map(|(o, _)| o.clone()))
            .collect();

        let model = if let Some(cached) = en_model.get(ctx) {
            cached.clone()
        } else {
            STAT_CONTEXTS.fetch_add(1, Ordering::Relaxed);
            let projected = marking.restricted_to(&ctx_objects);
            let found = enabled_activities(&oc, &projected, &reach, options);
            en_model.insert(ctx.clone(), found.clone());
            found
        };
        let logged = en_log.get(ctx).cloned().unwrap_or_default();

        let shared = logged.iter().filter(|a| model.contains(**a)).count() as f64;

        if std::env::var("OC_DEBUG").is_ok() {
            let mut l: Vec<&str> = logged.iter().copied().collect();
            let mut m: Vec<&str> = model.iter().map(String::as_str).collect();
            l.sort();
            m.sort();
            eprintln!("e{:<3} {:<16} en_L={:<34?} en_M={:?}", i + 1, log[i].activity, l, m);
        }
        if !logged.is_empty() {
            fit_sum += shared / logged.len() as f64;
        }
        if model.is_empty() {
            skipped += 1;
        } else {
            replayable += 1;
            prec_sum += shared / model.len() as f64;
        }

        // Advance the global marking by this event, so the next one sees the state it
        // leaves behind.
        if let Some(binding) = binding_for(&oc, &log[i]) {
            STAT_REPLAY_FIRES.fetch_add(1, Ordering::Relaxed);
            if let Some(next) = marking.fire(&oc, &binding) {
                marking = next;
            } else if let Some(next) = fire_through_silent(&oc, &marking, &binding, options) {
                marking = next;
            }
        }
    }

    OcConformance {
        fitness: if n == 0 { 1.0 } else { fit_sum / n as f64 },
        precision: if replayable == 0 {
            1.0
        } else {
            prec_sum / replayable as f64
        },
        skipped,
        events: n,
    }
}

/// The binding an event describes: its activity, and the objects it names per type.
///
/// `None` when the net has no transition for the activity at all, which is a context the
/// model cannot replay rather than a binding that happens not to be enabled.
fn binding_for(oc: &OcNet, ev: &OcEvent) -> Option<Binding> {
    let mut objects: HashMap<String, Vec<ObjectId>> = HashMap::new();
    for (o, ot) in &ev.objects {
        objects.entry(ot.clone()).or_default().push(o.clone());
    }
    if oc.transitions_of(&ev.activity).is_empty() {
        return None;
    }
    Some(Binding {
        activity: BindingTarget::Activity(ev.activity.clone()),
        objects,
    })
}

/// Move each of the binding's objects to the input place it needs, using silent
/// transitions only.
///
/// A silent transition binds **one** object and touches only its own type's places, so
/// positioning a token is reachability inside one flat sub-net -- not a search over joint
/// markings. Searching jointly is what made this intractable: with four object types
/// holding several objects each, the branching exhausts any state budget before all four
/// are in place, and every event is then reported as unreplayable. Per object it is a BFS
/// over the places of one sub-net.
fn fire_through_silent(
    oc: &OcNet,
    from: &OcMarking,
    binding: &Binding,
    options: &OcConformanceOptions,
) -> Option<OcMarking> {
    let mut marking = from.clone();
    for (place, ot) in oc.preset_of(&binding.activity) {
        for o in binding.objects_of(&ot) {
            if marking.holds(place, o) {
                continue;
            }
            marking = advance(oc, &marking, o, &ot, place, options)?;
        }
    }
    marking.fire(oc, binding)
}

/// Walk one object to `target` through silent transitions of its own type.
fn advance(
    oc: &OcNet,
    from: &OcMarking,
    object: &ObjectId,
    object_type: &str,
    target: crate::core::process_models::case_centric::petri_net::PlaceID,
    options: &OcConformanceOptions,
) -> Option<OcMarking> {
    let silent: Vec<_> = oc
        .silent()
        .into_iter()
        .filter(|(_, ot)| *ot == object_type)
        .map(|(t, _)| t)
        .collect();

    STAT_ADVANCE_CALLS.fetch_add(1, Ordering::Relaxed);
    STAT_ADVANCE_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([from.clone()]);
    let mut visited = 0usize;
    while let Some(m) = queue.pop_front() {
        if m.holds(target, object) {
            return Some(m);
        }
        if visited >= options.max_states {
            return None;
        }
        visited += 1;
        STAT_ADVANCE_STATES.fetch_add(1, Ordering::Relaxed);
        for t in &silent {
            let mut objects = HashMap::new();
            objects.insert(object_type.to_string(), vec![object.clone()]);
            let step = Binding {
                activity: BindingTarget::Silent(*t),
                objects,
            };
            if let Some(next) = m.fire(oc, &step) {
                let key: Vec<_> = next.tokens_of_object(object);
                if seen.insert(key) {
                    queue.push_back(next);
                }
            }
        }
    }
    None
}

/// Activities with at least one enabled binding in this marking, silent transitions
/// resolved through. An activity counts once however many bindings enable it, which keeps
/// the denominator the size of the alphabet rather than of the binding space.
fn enabled_activities(
    oc: &OcNet,
    marking: &OcMarking,
    reach: &HashMap<crate::core::process_models::case_centric::petri_net::PlaceID, HashSet<crate::core::process_models::case_centric::petri_net::PlaceID>>,
    options: &OcConformanceOptions,
) -> HashSet<String> {
    STAT_ENABLED_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut out = HashSet::new();
    for (label, _) in labelled(oc) {
        if out.contains(&label) {
            continue;
        }
        let target = BindingTarget::Activity(label.clone());
        // Directly enabled, or enabled once each object walks to its place through silent
        // transitions. Asked per activity and per object rather than by exploring the
        // joint state space.
        if any_binding_enabled(oc, marking, &target, options)
            || reachable_binding(oc, marking, &target, reach)
        {
            out.insert(label);
        }
    }
    out
}

/// Could some binding of this activity become enabled using silent transitions only?
fn reachable_binding(
    oc: &OcNet,
    marking: &OcMarking,
    target: &BindingTarget,
    reach: &HashMap<crate::core::process_models::case_centric::petri_net::PlaceID, HashSet<crate::core::process_models::case_centric::petri_net::PlaceID>>,
) -> bool {
    let pre = oc.preset_of(target);
    if pre.is_empty() {
        return false;
    }
    // A set lookup where this used to run a search per object: can any token of the right
    // type get to this place through silent transitions alone?
    pre.iter().all(|(place, ot)| {
        objects_in(marking, oc, ot).iter().any(|o| {
            marking
                .tokens_of_object(o)
                .iter()
                .any(|at| reach.get(at).is_some_and(|r| r.contains(place)))
        })
    })
}

fn labelled(oc: &OcNet) -> Vec<(String, Vec<crate::core::process_models::case_centric::petri_net::TransitionID>)> {
    let mut out: HashMap<String, Vec<_>> = HashMap::new();
    for (_, sub) in oc.net.nets.iter() {
        for (id, tr) in sub.transitions.iter() {
            if let Some(l) = tr.label.as_ref() {
                out.entry(l.clone())
                    .or_default()
                    .push(crate::core::process_models::case_centric::petri_net::TransitionID(*id));
            }
        }
    }
    out.into_iter().collect()
}

/// Is any binding of this transition enabled? One object per incident type suffices, so it
/// is enough to ask whether each input place holds something.
/// Is any binding of this activity enabled here?
///
/// Without a schema, tokens are interchangeable: it is enough that each input place holds
/// something of its type. With one, a binding has to be *built* -- take an object, follow
/// the maps to its partners in the other types, and ask whether those exact tokens are
/// available. An order whose items are still unpicked no longer enables `pay order` just
/// because some other order's items happen to be ready, and that is the behaviour a plain
/// object-centric net cannot forbid.
fn any_binding_enabled(
    oc: &OcNet,
    m: &OcMarking,
    target: &BindingTarget,
    options: &OcConformanceOptions,
) -> bool {
    let pre = oc.preset_of(target);
    if pre.is_empty() {
        return false;
    }
    let Some(relations) = options.relations.as_ref() else {
        return pre
            .iter()
            .all(|(p, ot)| objects_in(m, oc, ot).iter().any(|o| m.holds(*p, o)));
    };

    let types: HashSet<&str> = pre.iter().map(|(_, ot)| ot.as_str()).collect();
    // Try each available object as the anchor, deriving the rest through the maps.
    for (_, root_type) in pre.iter() {
        for root in objects_in(m, oc, root_type) {
            let mut objects: HashMap<String, Vec<ObjectId>> = HashMap::new();
            objects.insert(root_type.clone(), vec![root.clone()]);
            let mut ok = true;
            for ot in types.iter().filter(|t| *t != root_type) {
                match relations.related(root_type, &root, ot) {
                    Some(partners) if !partners.is_empty() => {
                        objects.insert(ot.to_string(), partners);
                    }
                    // Unrelated types stay free, as in a plain net.
                    None => {
                        let free = objects_in(m, oc, ot);
                        if free.is_empty() {
                            ok = false;
                            break;
                        }
                        objects.insert(ot.to_string(), vec![free[0].clone()]);
                    }
                    Some(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let candidate = Binding {
                activity: target.clone(),
                objects,
            };
            if relations.admits(&candidate) && m.enables(oc, &candidate) {
                return true;
            }
        }
    }
    false
}

/// Objects of a type present anywhere in the marking.
fn objects_in(m: &OcMarking, oc: &OcNet, object_type: &str) -> Vec<ObjectId> {
    let mut out = HashSet::new();
    for (_, sub) in oc.net.nets.iter() {
        for p in sub.places.keys() {
            let place = crate::core::process_models::case_centric::petri_net::PlaceID(*p);
            if oc.type_of_place(place) != Some(object_type) {
                continue;
            }
            for (o, _) in m.tokens_of(place) {
                out.insert(o);
            }
        }
    }
    out.into_iter().collect()
}
