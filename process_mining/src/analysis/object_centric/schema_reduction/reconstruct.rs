//! Give every object type a component again, so the reduced model can be compared to the
//! recorded one on the same terms.
//!
//! An [`ObjectCentricPetriNet`] discovered from the flow projection has a component only for
//! the types that still flow. This puts the two non-flow states back into the model in the
//! form each of them asserts (Def. Annotation).
//!
//! An involved cell is a self-loop place: the transition reads an object of the type and
//! writes it back, so firing needs one to be present and imposes no order (Sect. 6).
//!
//! An implied cell is a mapped place. Where a flow type `S` determines `T` at an activity,
//! `T` moves wherever `S` moves, so the component gets `S`'s local structure around that
//! transition mirrored as places of `T`.
//!
//! Where `S -> T` is many-to-one the mirrored structure fires once per `S` object where
//! `T`'s trace has the event once. Those arcs are marked variable. The reconstruction is
//! exact only at fibre 1.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::core::process_models::{
    case_centric::petri_net::petri_net_struct::{ArcType, Marking, PetriNet, PlaceID, TransitionID},
    object_centric::ocpn::ObjectCentricPetriNet,
};

use super::{
    assignment::Assignment,
    bounds::Bounds,
    cells::{ActivityIndex, Cell, CellGrid},
    schema::ObjectTypeIndex,
    simultaneity::precedes,
};

/// Which flow type puts each implied cell back, as the grid's determinacy walk found it.
pub type Carriers = HashMap<Cell, ObjectTypeIndex>;

/// Per type, the distinct activity sets its objects participate in.
///
/// A resource type is often several populations doing disjoint work, and merging them
/// into one place says every activity is possible for every object. Each set here becomes
/// its own branch. A missing entry means one branch over everything.
pub type InvolvementClusters = HashMap<ObjectTypeIndex, Vec<Vec<ActivityIndex>>>;

/// How an involved cell is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InvolvementRender {
    /// The type's own component: every involvement and carrier-less-absence branch bracketed
    /// together into one shared token, sequenced by separation where `ENRICH_REP` allows it.
    #[default]
    Connected,
    /// One place per activity, read and written back by that activity's transition alone,
    /// as Sect. 6 draws it. No shared token, no sequencing, no population clusters.
    SelfLoop,
    /// No place at all. A consumer reads involvement off the assignment or cell data and
    /// draws its own marker (e.g. a badge on the transition) instead of net structure.
    Hidden,
}

/// How an implied cell is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AbsenceRender {
    /// A mapped place: the determining carrier's own component, copied and mirrored so the
    /// non-flow type's participation is recoverable from the drawn net alone (Sect. 6).
    #[default]
    Mapped,
    /// No place. A consumer reads the determining map off the assignment or cell data and
    /// draws its own annotation (e.g., a "flows with" rider on the carrier's arc).
    Hidden,
}

/// Why a place exists beyond what mining the flow-only sublog would have produced. A place
/// missing from the returned map is an ordinary flow place. Roles are not recoverable from
/// the net's topology alone, since `Connected`'s branches use the same kind of place a
/// real flow does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaceRole {
    /// A self-loop place added for an involved cell (or an implied cell with no carrier,
    /// which draws the same way).
    Involved,
    /// A place mirroring `carrier`'s own structure, added for an implied cell that
    /// `carrier` determines.
    Mapped { carrier: String },
}

/// Rebuild the components the reduction emptied or thinned.
///
/// `reduced` is the net discovered from the flow projection. `carriers` names, per implied
/// cell, the flow type that determines it; a cell with no entry is one nothing recovers, and
/// its activity is left out rather than guessed at.
///
/// The second half of the return is [`PlaceRole`] by place id, covering every place this
/// function itself adds. A place the mining input already had is not in it.
pub fn reconstruct_ocpn(
    reduced: &ObjectCentricPetriNet,
    assignment: &Assignment,
    carriers: &Carriers,
    clusters: &InvolvementClusters,
    bounds: &Bounds,
    types: &[String],
    activities: &[String],
    involvement_render: InvolvementRender,
    absence_render: AbsenceRender,
) -> (ObjectCentricPetriNet, HashMap<Uuid, PlaceRole>) {
    let mut out = reduced.clone();
    let mut roles: HashMap<Uuid, PlaceRole> = HashMap::new();

    let inv_by_type = group(&assignment.involvement);
    let abs_by_type = group(&assignment.implied);

    let touched: HashSet<ObjectTypeIndex> =
        inv_by_type.keys().chain(abs_by_type.keys()).copied().collect();

    for t in touched {
        let Some(type_name) = types.get(t) else { continue };
        let mut net = out.nets.get(type_name).cloned().unwrap_or_default();
        // A component with no labeled transition is what a type with no flow cell leaves in
        // the net mined from the flow projection (initial place, a tau, final place). It
        // constrains nothing, so drop it.
        if !net.transitions.values().any(|tr| tr.label.is_some()) {
            net = PetriNet::default();
        }

        // Each repair is a branch of one component. Collect the endpoints and bracket them
        // all at the end, so the type keeps a single token.
        let mut branches: Vec<(Vec<PlaceID>, Vec<PlaceID>, Vec<ActivityIndex>)> = Vec::new();
        let kept_start: Vec<PlaceID> = net
            .initial_marking
            .iter()
            .flat_map(|m| m.keys().copied())
            .collect();
        let kept_end: Vec<PlaceID> = net
            .final_markings
            .as_ref()
            .and_then(|f| f.first())
            .into_iter()
            .flat_map(|m| m.keys().copied())
            .collect();
        if !kept_start.is_empty() || !kept_end.is_empty() {
            let kept_acts: Vec<ActivityIndex> = assignment
                .flow
                .iter()
                .filter(|(_, tt)| *tt == t)
                .map(|(a, _)| *a)
                .collect();
            branches.push((kept_start, kept_end, kept_acts));
        }
        net.initial_marking = None;
        net.final_markings = None;

        // Absence: the carrier's component is copied whole and the activities the type does
        // not participate in are silenced. A spliced fragment would leave tokens in places no
        // transition consumes from, so the component could never reach a final marking.
        let mut by_carrier: HashMap<ObjectTypeIndex, Vec<ActivityIndex>> = HashMap::new();
        // An implied cell with no carrier is still a participation, so it renders like involvement.
        let mut merged: Vec<ActivityIndex> = Vec::new();
        for a in abs_by_type.get(&t).into_iter().flatten() {
            match carriers.get(&(*a, t)) {
                Some(&s) => by_carrier.entry(s).or_default().push(*a),
                None => merged.push(*a),
            }
        }
        let mut carrier_keys: Vec<_> = by_carrier.keys().copied().collect();
        carrier_keys.sort_unstable();
        // Under `Hidden` no branch is built. Folding absence into involvement would assert a
        // weaker fact (present, unrecovered) than a determined absence does.
        for s in if matches!(absence_render, AbsenceRender::Mapped) {
            carrier_keys
        } else {
            Vec::new()
        } {
            let Some(source_name) = types.get(s) else { continue };
            let Some(source) = reduced.nets.get(source_name) else { continue };
            let observable: HashSet<&str> = by_carrier[&s]
                .iter()
                .filter_map(|a| activities.get(*a).map(String::as_str))
                .collect();
            // No skip is offered: a carried type is there whenever its carrier is, and an
            // optional activity would assert something the determinacy denies.
            // A return arc asserts that an object repeats one of these activities, which is
            // independent of the fibre size.
            let repeatable = bounds.per_type.get(t).into_iter().flatten().any(|ob| {
                by_carrier[&s].iter().any(|a| ob.count(*a) > 1)
            });
            let before: HashSet<Uuid> = net.places.keys().copied().collect();
            let (b_start, b_end) = copy_projected(&mut net, source, &observable, repeatable);
            for id in net.places.keys() {
                if !before.contains(id) {
                    roles.insert(*id, PlaceRole::Mapped { carrier: source_name.clone() });
                }
            }
            branches.push((b_start, b_end, by_carrier[&s].clone()));
            for a in &by_carrier[&s] {
                // The fibre is not known here, so the arc is declared variable. A one-to-one
                // map is the special case of the set-valued reading.
                if let Some(label) = activities.get(*a) {
                    out.variable_arcs
                        .entry(type_name.clone())
                        .or_default()
                        .insert(label.clone());
                }
            }
        }

        // Involvement, plus every implied cell whose type has no carrier. One start place,
        // a silent choice into one place per population, self-loops for that population's
        // activities, a silent merge, one final place. The choice asserts that an object
        // belongs to exactly one population.
        let mut inv: Vec<ActivityIndex> = inv_by_type
            .get(&t)
            .into_iter()
            .flatten()
            .copied()
            .chain(merged)
            .collect();
        inv.sort_unstable();
        inv.dedup();
        if !inv.is_empty() && matches!(involvement_render, InvolvementRender::Connected) {
            let inv_set: HashSet<ActivityIndex> = inv.iter().copied().collect();
            let mut groups: Vec<Vec<ActivityIndex>> = clusters
                .get(&t)
                .map(|cs| {
                    cs.iter()
                        .map(|c| {
                            let mut g: Vec<ActivityIndex> =
                                c.iter().filter(|a| inv_set.contains(a)).copied().collect();
                            g.sort_unstable();
                            g.dedup();
                            g
                        })
                        .filter(|g| !g.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            groups.sort();
            groups.dedup();
            if groups.is_empty() {
                groups.push(inv.clone());
            }

            // How often each involved activity occurs, and for how many objects. A self-loop
            // asserts "any number of times" where the log often says "exactly once".
            let population = bounds.per_type.get(t).map_or(0, Vec::len);
            let mut realised: HashMap<ActivityIndex, usize> = HashMap::new();
            let mut repeats: HashSet<ActivityIndex> = HashSet::new();
            for ob in bounds.per_type.get(t).into_iter().flatten() {
                for (a, n) in &ob.times {
                    *realised.entry(*a).or_default() += 1;
                    if *n > 1 {
                        repeats.insert(*a);
                    }
                }
            }

            // One block per activity that happens at most once, plus one shared seat for
            // those that repeat. Involvement asserts no order between them, so they are
            // composed in parallel; the fork and join come from `bracket`.
            let mut block = |net: &mut PetriNet, g: &[ActivityIndex]| -> Vec<(PlaceID, PlaceID, Vec<ActivityIndex>)> {
                let mut out = Vec::new();
                let looping: Vec<ActivityIndex> =
                    g.iter().copied().filter(|a| repeats.contains(a)).collect();
                if !looping.is_empty() {
                    let seat = net.add_place(None);
                    roles.insert(seat.get_uuid(), PlaceRole::Involved);
                    for a in &looping {
                        let Some(label) = activities.get(*a) else { continue };
                        // A fresh transition per population. A shared one for an activity two
                        // populations both perform would need a token in each seat, turning
                        // the exclusive choice into a conjunction neither can fire.
                        let tr = net.add_transition(Some(label.clone()), None);
                        net.add_arc(ArcType::place_to_transition(seat, tr), None);
                        net.add_arc(ArcType::transition_to_place(tr, seat), None);
                    }
                    out.push((seat, seat, looping.clone()));
                }
                for a in g.iter().filter(|a| !repeats.contains(a)) {
                    let Some(label) = activities.get(*a) else { continue };
                    let from = net.add_place(None);
                    let to = net.add_place(None);
                    roles.insert(from.get_uuid(), PlaceRole::Involved);
                    roles.insert(to.get_uuid(), PlaceRole::Involved);
                    let tr = net.add_transition(Some(label.clone()), None);
                    net.add_arc(ArcType::place_to_transition(from, tr), None);
                    net.add_arc(ArcType::transition_to_place(tr, to), None);
                    // A cell every object realises is mandatory; one only some realise gets a
                    // silent skip.
                    if realised.get(a).copied().unwrap_or(0) < population {
                        let skip = net.add_transition(None, None);
                        net.add_arc(ArcType::place_to_transition(from, skip), None);
                        net.add_arc(ArcType::transition_to_place(skip, to), None);
                    }
                    out.push((from, to, vec![*a]));
                }
                out
            };

            if groups.len() == 1 {
                // Nothing to choose between, so the blocks are branches of the type's own
                // component.
                for (from, to, acts) in block(&mut net, &groups[0]) {
                    branches.push((vec![from], vec![to], acts));
                }
            } else {
                let start = net.add_place(None);
                let end = net.add_place(None);
                for g in &groups {
                    let blocks = block(&mut net, g);
                    // Silent choice in, silent merge out: committing to a population is not
                    // an observable event.
                    let enter = net.add_transition(None, None);
                    let leave = net.add_transition(None, None);
                    net.add_arc(ArcType::place_to_transition(start, enter), None);
                    net.add_arc(ArcType::transition_to_place(leave, end), None);
                    for (from, to, _) in blocks {
                        net.add_arc(ArcType::transition_to_place(enter, from), None);
                        net.add_arc(ArcType::place_to_transition(to, leave), None);
                    }
                }
                branches.push((vec![start], vec![end], inv.clone()));
            }
        }

        // The type's own eventually-follows relation, used to sequence branches the data
        // orders. A block edge asserts every activity of one block before every activity of
        // the other, per object, so the relation is unanimous separation: for every object
        // carrying both activities, its last occurrence of the first strictly precedes its
        // first occurrence of the second. One straddling object vetoes the pair.
        let (ordered, cooccur) = {
            let mut sep: HashMap<(ActivityIndex, ActivityIndex), (usize, usize)> =
                HashMap::new();
            for ob in bounds.per_type.get(t).into_iter().flatten() {
                for (x, _, xmax) in &ob.at {
                    for (y, ymin, _) in &ob.at {
                        if x == y {
                            continue;
                        }
                        let e = sep.entry((*x, *y)).or_default();
                        e.0 += 1;
                        if precedes(*xmax, *ymin) {
                            e.1 += 1;
                        }
                    }
                }
            }
            let co: HashSet<_> = sep.keys().copied().collect();
            let unanimous: HashSet<_> = sep
                .iter()
                .filter(|(_, (n, k))| n == k)
                .map(|(pair, _)| *pair)
                .collect();
            (unanimous, co)
        };
        let branch_acts: Vec<Vec<ActivityIndex>> = branches.iter().map(|b| b.2.clone()).collect();
        // The ordering enrichment (block edges and precedence places) is a separate
        // discovery step, opt-in via ENRICH_REP, so it is never credited to the reduction.
        let plain = std::env::var("ENRICH_REP").is_err();
        let no_order: HashSet<(ActivityIndex, ActivityIndex)> = HashSet::new();
        bracket(
            &mut net,
            branches,
            if plain { &no_order } else { &ordered },
            &cooccur,
        );

        // `SelfLoop` draws involvement as Sect. 6 states it: one place per activity, read and
        // written back by that activity's transition alone. It runs after `bracket` because a
        // self-loop place is never scarce or sequenced, so it gains nothing from the fork/join.
        if matches!(involvement_render, InvolvementRender::SelfLoop) {
            for a in &inv {
                let Some(label) = activities.get(*a) else { continue };
                // A fresh transition per activity, never shared, the same rule `block`'s
                // non-repeating branch uses for `Connected`.
                let tr = net.add_transition(Some(label.clone()), None);
                let p = net.add_place(None);
                roles.insert(p.get_uuid(), PlaceRole::Involved);
                net.add_arc(ArcType::place_to_transition(p, tr), None);
                net.add_arc(ArcType::transition_to_place(tr, p), None);
                net.initial_marking.get_or_insert_with(Marking::default).insert(p, 1);
                match &mut net.final_markings {
                    Some(fms) => {
                        for m in fms {
                            m.insert(p, 1);
                        }
                    }
                    None => {
                        let mut m = Marking::default();
                        m.insert(p, 1);
                        net.final_markings = Some(vec![m]);
                    }
                }
            }
        }

        // Transition-level precedence for what the blocks cannot order. Where two blocks
        // interleave, a pair inside them can still be unanimously separated. Such a pair gets
        // a read-loop place: `x` produces it, `y` reads and returns it, so `y` waits for `x`
        // and repeats freely. Sound only when `x` occurs at most once per object and every
        // object with `y` has an `x`; otherwise the place would block a legal trace.
        {
            let objs = bounds.per_type.get(t).map(Vec::as_slice).unwrap_or(&[]);
            let n = branch_acts.len();
            let block_of = |a: ActivityIndex| branch_acts.iter().position(|acts| acts.contains(&a));
            // Same block-order relation the bracket used, transitively closed.
            let mut reach = vec![vec![false; n]; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let mut any = false;
                    let mut all = true;
                    for x in &branch_acts[i] {
                        for y in &branch_acts[j] {
                            if cooccur.contains(&(*x, *y)) {
                                if ordered.contains(&(*x, *y)) {
                                    any = true;
                                } else {
                                    all = false;
                                }
                            }
                        }
                    }
                    reach[i][j] = any && all;
                }
            }
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        if reach[i][k] && reach[k][j] {
                            reach[i][j] = true;
                        }
                    }
                }
            }
            let mut candidates: Vec<(ActivityIndex, ActivityIndex)> = Vec::new();
            for (x, y) in ordered.iter().filter(|_| !plain) {
                let (Some(bi), Some(bj)) = (block_of(*x), block_of(*y)) else { continue };
                if bi == bj || reach[bi][bj] || reach[bj][bi] {
                    continue;
                }
                // `x` must occur exactly once in every object, so the place holds one token
                // when the object completes and can be part of the final marking.
                if objs.iter().any(|ob| ob.count(*x) != 1) {
                    continue;
                }
                candidates.push((*x, *y));
            }
            candidates.sort_unstable();
            // Transitive reduction over the candidates: a chained pair is already
            // enforced, because every place source is mandatory exactly once.
            let cset: HashSet<_> = candidates.iter().copied().collect();
            for (x, y) in candidates {
                if cset.iter().any(|(a, z)| *a == x && cset.contains(&(*z, y))) {
                    continue;
                }
                let (Some(lx), Some(ly)) = (activities.get(x), activities.get(y)) else {
                    continue;
                };
                let (Some(tx), Some(ty)) = (
                    transition_by_label(&net, lx),
                    transition_by_label(&net, ly),
                ) else {
                    continue;
                };
                let q = net.add_place(None);
                net.add_arc(ArcType::transition_to_place(tx, q), None);
                net.add_arc(ArcType::place_to_transition(q, ty), None);
                net.add_arc(ArcType::transition_to_place(ty, q), None);
                if let Some(fs) = &mut net.final_markings {
                    for m in fs {
                        m.insert(q, 1);
                    }
                }
            }
        }
        out.nets.insert(type_name.clone(), net);
    }

    (out, roles)
}

/// Project `source` onto `observable`.
///
/// The type is at these activities exactly when its carrier is, so what it inherits is the
/// carrier's ordering between those activities. Every transition the type does not take
/// part in becomes silent, so a token crosses that stretch of the carrier's path without an
/// event.
fn copy_projected(
    net: &mut PetriNet,
    source: &PetriNet,
    observable: &HashSet<&str>,
    repeatable: bool,
) -> (Vec<PlaceID>, Vec<PlaceID>) {
    // Every place is kept and every unobserved transition becomes silent. Contracting them
    // instead would merge places across silent skips and loop redos and lose inherited
    // orderings. The silent copy asserts exactly the carrier's language projected onto the
    // observable labels.
    let mut place_map: HashMap<Uuid, PlaceID> = HashMap::new();
    let mut pids: Vec<Uuid> = source.places.keys().copied().collect();
    pids.sort_unstable();
    for p in pids {
        place_map.insert(p, net.add_place(None));
    }
    let mut trans_map: HashMap<Uuid, TransitionID> = HashMap::new();
    let mut tids: Vec<Uuid> = source.transitions.keys().copied().collect();
    tids.sort_unstable();
    for id in tids {
        let tr = &source.transitions[&id];
        let obs = tr.label.as_deref().is_some_and(|l| observable.contains(l));
        let t = if obs {
            // Reuse the transition the surviving component already has for this label. A
            // second one would let the flow chain fire without the mapped place.
            match tr.label.as_deref().and_then(|l| transition_by_label(net, l)) {
                Some(existing) => existing,
                None => net.add_transition(tr.label.clone(), None),
            }
        } else {
            net.add_transition(None, None)
        };
        trans_map.insert(id, t);
    }
    for arc in &source.arcs {
        match arc.from_to {
            ArcType::PlaceTransition(p, t) => {
                if let (Some(q), Some(u)) = (place_map.get(&p), trans_map.get(&t)) {
                    net.add_arc(ArcType::place_to_transition(*q, *u), Some(arc.weight));
                }
            }
            ArcType::TransitionPlace(t, p) => {
                if let (Some(u), Some(q)) = (trans_map.get(&t), place_map.get(&p)) {
                    net.add_arc(ArcType::transition_to_place(*u, *q), Some(arc.weight));
                }
            }
        }
    }

    // The branch's own endpoints, handed back for the caller to bracket.
    let starts: Vec<PlaceID> = source
        .initial_marking
        .iter()
        .flat_map(|m| m.keys())
        .filter_map(|p| place_map.get(&p.0).copied())
        .collect();
    let ends: Vec<PlaceID> = source
        .final_markings
        .iter()
        .flatten()
        .take(1)
        .flat_map(|m| m.keys())
        .filter_map(|p| place_map.get(&p.0).copied())
        .collect();

    // An object of the non-flow type may gather several carrier objects and run the carrier's
    // path once per carrier object. A silent return from end to start frees the number of
    // passes while keeping the order within a pass, only where the log shows a repeat.
    if repeatable && !starts.is_empty() && !ends.is_empty() {
        let repeat = net.add_transition(None, None);
        for q in &ends {
            net.add_arc(ArcType::place_to_transition(*q, repeat), None);
        }
        for q in &starts {
            net.add_arc(ArcType::transition_to_place(repeat, *q), None);
        }
    }
    (starts, ends)
}

fn transition_by_label(net: &PetriNet, label: &str) -> Option<TransitionID> {
    net.transitions
        .iter()
        .find(|(_, t)| t.label.as_deref() == Some(label))
        .map(|(id, _)| TransitionID(*id))
}

/// Join the type's branches into one component with a single source and a single sink.
///
/// A repaired type is a conjunction of constraints on one object, and a Petri net states a
/// conjunction on one token by synchronising, so the branches get a silent fork in and a
/// silent join out. Marking each branch separately would give the object one token per
/// branch, and independent tokens assert strictly less than any one branch does.
fn bracket(
    net: &mut PetriNet,
    branches: Vec<(Vec<PlaceID>, Vec<PlaceID>, Vec<ActivityIndex>)>,
    ordered: &HashSet<(ActivityIndex, ActivityIndex)>,
    cooccur: &HashSet<(ActivityIndex, ActivityIndex)>,
) {
    let mark = |net: &mut PetriNet, starts: Vec<PlaceID>, ends: Vec<PlaceID>| {
        let mut initial = Marking::default();
        for p in starts {
            initial.insert(p, 1);
        }
        net.initial_marking = Some(initial);
        let mut final_marking = Marking::default();
        for p in ends {
            final_marking.insert(p, 1);
        }
        net.final_markings = Some(vec![final_marking]);
    };

    let mut branches = branches;
    if branches.is_empty() {
        return;
    }
    if branches.len() == 1 {
        let (starts, ends, _) = branches.pop().expect("length checked");
        mark(net, starts, ends);
        return;
    }

    // Sequence branches the type itself orders; leave the rest concurrent.
    let n = branches.len();
    // Sequencing block i before block j asserts every activity of i before every activity
    // of j, per object. That holds only when every co-occurring cross pair is unanimously
    // ordered that way; a single mixed pair vetoes the edge.
    let before = |i: usize, j: usize| {
        let mut any = false;
        for x in &branches[i].2 {
            for y in &branches[j].2 {
                if cooccur.contains(&(*x, *y)) {
                    if !ordered.contains(&(*x, *y)) {
                        return false;
                    }
                    any = true;
                }
            }
        }
        any
    };
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && before(i, j) {
                adj[i][j] = true;
            }
        }
    }
    // Antisymmetry does not make `before` acyclic transitively; a cycle means no consistent
    // sequence, so everything stays concurrent.
    let mut reach = adj.clone();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if reach[i][k] && reach[k][j] {
                    reach[i][j] = true;
                }
            }
        }
    }
    if (0..n).any(|i| reach[i][i]) {
        for row in &mut adj {
            row.fill(false);
        }
    } else {
        // Transitive reduction: a covering edge is one no longer path implies.
        for i in 0..n {
            for j in 0..n {
                if adj[i][j] && (0..n).any(|k| k != i && k != j && reach[i][k] && reach[k][j]) {
                    adj[i][j] = false;
                }
            }
        }
    }

    // The partial order as a marked graph: one silent entry and exit per branch, one edge
    // place per covering pair. Two branches the data does not order share no place and stay
    // concurrent. A single source and sink keep the type on one token.
    let source = net.add_place(None);
    let sink = net.add_place(None);
    let fork = net.add_transition(None, None);
    let join = net.add_transition(None, None);
    net.add_arc(ArcType::place_to_transition(source, fork), None);
    net.add_arc(ArcType::transition_to_place(join, sink), None);
    let entries: Vec<TransitionID> = (0..n).map(|_| net.add_transition(None, None)).collect();
    let exits: Vec<TransitionID> = (0..n).map(|_| net.add_transition(None, None)).collect();
    for i in 0..n {
        for p in &branches[i].0 {
            net.add_arc(ArcType::transition_to_place(entries[i], *p), None);
        }
        for p in &branches[i].1 {
            net.add_arc(ArcType::place_to_transition(*p, exits[i]), None);
        }
        // Only minimal branches start from the fork and only maximal ones report to the
        // join; an interior branch is already forced by its edge places.
        if (0..n).all(|h| !adj[h][i]) {
            let seat_in = net.add_place(None);
            net.add_arc(ArcType::transition_to_place(fork, seat_in), None);
            net.add_arc(ArcType::place_to_transition(seat_in, entries[i]), None);
        }
        if (0..n).all(|j| !adj[i][j]) {
            let seat_out = net.add_place(None);
            net.add_arc(ArcType::transition_to_place(exits[i], seat_out), None);
            net.add_arc(ArcType::place_to_transition(seat_out, join), None);
        }
    }
    for i in 0..n {
        for j in 0..n {
            if adj[i][j] {
                let e = net.add_place(None);
                net.add_arc(ArcType::transition_to_place(exits[i], e), None);
                net.add_arc(ArcType::place_to_transition(e, entries[j]), None);
            }
        }
    }
    let initial = vec![source];
    let fin = vec![sink];
    mark(net, initial, fin);
}

#[cfg(test)]
mod bracket_tests {
    use super::*;

    fn branch(net: &mut PetriNet, acts: Vec<ActivityIndex>) -> (Vec<PlaceID>, Vec<PlaceID>, Vec<ActivityIndex>) {
        let start = net.add_place(None);
        let end = net.add_place(None);
        (vec![start], vec![end], acts)
    }

    #[test]
    fn two_unordered_branches_stay_concurrent() {
        let mut net = PetriNet::default();
        let b0 = branch(&mut net, vec![0]);
        let b1 = branch(&mut net, vec![1]);
        bracket(&mut net, vec![b0, b1], &HashSet::new(), &HashSet::new());

        // Both branches are minimal and maximal at once: a fork seat and a join seat each
        // (4 seats), plus each branch's own start/end (4) plus source/sink (2).
        assert_eq!(net.places.len(), 4 + 2 + 4);
        assert!(net.initial_marking.is_some());
        assert_eq!(net.final_markings.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn a_unanimously_ordered_pair_gets_sequenced_with_one_edge_place() {
        let mut net = PetriNet::default();
        let b0 = branch(&mut net, vec![0]);
        let b1 = branch(&mut net, vec![1]);
        let ordered = HashSet::from([(0, 1)]);
        let cooccur = HashSet::from([(0, 1)]);
        bracket(&mut net, vec![b0, b1], &ordered, &cooccur);

        // b0 is minimal only and b1 is maximal only: one seat each (2) plus one edge place
        // carrying b0's exit to b1's entry.
        assert_eq!(net.places.len(), 4 + 2 + 2 + 1);
    }

    #[test]
    fn a_contradictory_order_falls_back_to_fully_concurrent() {
        // A pair asserted both ways cannot be sequenced consistently; `bracket` must not
        // wire a cycle into the net.
        let mut net = PetriNet::default();
        let b0 = branch(&mut net, vec![0]);
        let b1 = branch(&mut net, vec![1]);
        let ordered = HashSet::from([(0, 1), (1, 0)]);
        let cooccur = HashSet::from([(0, 1), (1, 0)]);
        bracket(&mut net, vec![b0, b1], &ordered, &cooccur);

        // Both branches get both seats, same as the unordered case.
        assert_eq!(net.places.len(), 4 + 2 + 4);
    }
}

fn group(cells: &[Cell]) -> HashMap<ObjectTypeIndex, Vec<ActivityIndex>> {
    let mut out: HashMap<ObjectTypeIndex, Vec<ActivityIndex>> = HashMap::new();
    for (a, t) in cells {
        out.entry(*t).or_default().push(*a);
    }
    for v in out.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// The carrier of every implied cell that has an ordering for a carrier to carry.
///
/// The cell must be determined, read back off the grid's own determinacy walk, so the
/// reconstruction can only name a carrier the assignment already used to justify removing
/// the cell.
///
/// The non-flow type must also order something, tested by `asserted_of_type`. Determinacy
/// says the participation comes back and nothing about the type having a control flow. A
/// mapped place for a type that orders nothing would make the drawn component stricter
/// than the recorded one. A cell failing the test is rendered as involvement by
/// [`reconstruct_ocpn`].
pub fn carriers_of(
    grid: &CellGrid,
    bounds: &Bounds,
    saturated: &HashSet<Cell>,
    flow: &HashSet<Cell>,
) -> Carriers {
    let orders: Vec<bool> = (0..bounds.per_type.len())
        .map(|t| !super::facts::asserted_of_type(bounds, saturated, t).is_empty())
        .collect();

    let mut out = Carriers::new();
    for (a, cells) in grid.per_activity.iter().enumerate() {
        let kept: Vec<ObjectTypeIndex> = cells
            .present
            .iter()
            .copied()
            .filter(|t| flow.contains(&(a, *t)))
            .collect();
        // `determining_routes` walks transitively, so the cell it names can itself be a
        // non-flow one. A non-flow cell has no transition in the drawn model, so the chain is
        // followed to a cell that flows.
        let routes = cells.determining_routes(&kept);
        for j in 0..cells.present.len() {
            let t = cells.present[j];
            if !orders.get(t).copied().unwrap_or(false) {
                continue;
            }
            let mut at = j;
            let mut hops = 0;
            let carrier = loop {
                match routes.get(at).and_then(Option::as_ref) {
                    Some((i, _)) if hops < cells.present.len() => {
                        at = *i;
                        hops += 1;
                        if flow.contains(&(a, cells.present[at])) {
                            break Some(cells.present[at]);
                        }
                    }
                    // Nothing reaches it, or the chain closed without meeting a flow cell.
                    _ => break None,
                }
            };
            if let Some(c) = carrier {
                out.insert((a, t), c);
            }
        }
    }
    out
}

/// Populations worth drawing apart: the distinct activity sets a type's objects occur at,
/// kept only where the split says something.
///
/// A structural property of the participations alone. It is a rendering tweak and not
/// part of the reduction.
///
/// Two shapes are rejected: a single population (no partition to draw) and nested sets
/// (objects that had not finished when the log ended, which are a fact about the log
/// window and not different kinds of object). Population size is not a test. The sets
/// need not be disjoint.
pub fn involvement_clusters(bounds: &Bounds) -> InvolvementClusters {
    let mut out = InvolvementClusters::new();
    for (t, objs) in bounds.per_type.iter().enumerate() {
        let per_object: Vec<Vec<ActivityIndex>> = objs
            .iter()
            .map(|ob| {
                let mut acts: Vec<ActivityIndex> = ob.at.iter().map(|(a, _, _)| *a).collect();
                acts.sort_unstable();
                acts.dedup();
                acts
            })
            .collect();

        let mut sets = per_object.clone();
        sets.sort();
        sets.dedup();
        if sets.len() < 2 {
            continue;
        }
        let nested = sets.iter().any(|a| {
            sets.iter()
                .any(|b| a != b && a.iter().all(|x| b.contains(x)))
        });
        if nested {
            continue;
        }
        out.insert(t, sets);
    }
    out
}

/// Draw the assignment: the reduced model with involvement and absence rendered.
///
/// Everything it derives comes from the grid and the participations, so the repair is a
/// function of the log and the assignment, independent of the miner.
pub fn repair_ocpn(
    reduced: &ObjectCentricPetriNet,
    assignment: &Assignment,
    grid: &CellGrid,
    bounds: &Bounds,
    saturated: &HashSet<Cell>,
    flow: &HashSet<Cell>,
    types: &[String],
    activities: &[String],
    involvement_render: InvolvementRender,
    absence_render: AbsenceRender,
) -> (ObjectCentricPetriNet, HashMap<Uuid, PlaceRole>) {
    // Population clusters are the same opt-in enrichment `reconstruct_ocpn` gates its
    // ordering on, so they are never credited to the reduction.
    let clusters = if std::env::var("ENRICH_REP").is_err() {
        InvolvementClusters::default()
    } else {
        involvement_clusters(bounds)
    };
    reconstruct_ocpn(
        reduced,
        assignment,
        &carriers_of(grid, bounds, saturated, flow),
        &clusters,
        bounds,
        types,
        activities,
        involvement_render,
        absence_render,
    )
}
