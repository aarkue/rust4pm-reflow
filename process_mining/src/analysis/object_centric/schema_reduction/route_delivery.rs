//! Ordering delivered by composing flow with a schema map, object by object.
//!
//! [`Chained`](super::Chained) composes at the level of activities: `items` orders
//! `x < m`, `packages` orders `m < y`, so the closure delivers `x < y`. The two hops may
//! name different events of `m`, possibly in the opposite time order, so that step can
//! invent pairs the log does not support.
//!
//! This module composes at the level of objects instead. A route `f: S -> S'` in the
//! schema takes each source object to a target object, so `x < y` is asserted only when
//! every `S`-object at `x` has its own image at `y`, later. Every delivered pair is
//! witnessed by concrete objects with concrete timestamps that a reader of the reduced
//! log can re-check: `S` is kept at `x`, `S'` at `y`, and `f` is an object-to-object edge
//! the reduction writes out.
//!
//! Soundness follows from monotonicity in the keep-set. Whether `(x, y)` is asserted
//! through a route reads only the cells `(x, S)` and `(y, S')`, so `K subset K'` gives
//! `delivered(K) subset delivered(K')`, and no keep-set can assert what the log does not.
//! Completeness is not guaranteed.

use std::collections::{HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::slim_linked_ocel::ObjectIndex;

use super::{
    bounds::{Bounds, ObjectBounds},
    cells::Cell,
    expansion::Route,
    facts::asserted_by_type,
    novelty::Pair,
    simultaneity::precedes,
};

/// The pairs one route asserts under a keep-set.
///
/// `(x, y)` is asserted when some `S`-object is at `x` with its image at `y` strictly
/// after, and no `S`-object is at `y`-image with `x` strictly after. This is the same
/// forward-minus-backward reading [`asserted_of_type`](super::asserted_of_type) takes
/// within one type, with the map standing in for object identity.
///
/// `x == y` is skipped: an activity never orders itself.
pub fn asserted_of_route(bounds: &Bounds, kept: &HashSet<Cell>, route: &Route) -> HashSet<Pair> {
    let (Some(src), Some(tgt)) = (
        bounds.per_type.get(route.source),
        bounds.per_type.get(route.target),
    ) else {
        return HashSet::new();
    };
    let by_object: HashMap<ObjectIndex, &ObjectBounds> =
        tgt.iter().map(|ob| (ob.object, ob)).collect();

    let mut fwd: HashSet<Pair> = HashSet::new();
    let mut bwd: HashSet<Pair> = HashSet::new();
    let mut here: Vec<(usize, i64, i64)> = Vec::new();
    let mut there: Vec<(usize, i64, i64)> = Vec::new();
    for ob in src {
        let Some(image) = route.f.get(&ob.object).and_then(|o| by_object.get(o)) else {
            continue;
        };
        here.clear();
        here.extend(
            ob.at
                .iter()
                .filter(|(a, _, _)| kept.contains(&(*a, route.source))),
        );
        if here.is_empty() {
            continue;
        }
        there.clear();
        there.extend(
            image
                .at
                .iter()
                .filter(|(a, _, _)| kept.contains(&(*a, route.target))),
        );
        for (x, xmin, xmax) in &here {
            for (y, ymin, ymax) in &there {
                if x == y {
                    continue;
                }
                if precedes(*xmin, *ymax) {
                    fwd.insert((*x, *y));
                }
                if precedes(*ymin, *xmax) {
                    bwd.insert((*x, *y));
                }
            }
        }
    }
    // A map has a direction and an ordering does not, so the backward witnesses are read too.
    let mut out: HashSet<Pair> = fwd.difference(&bwd).copied().collect();
    out.extend(bwd.difference(&fwd).map(|(x, y)| (*y, *x)));
    out
}

/// Every ordering pair a keep-set delivers: drawn by one type, or carried by one route.
///
/// Routes are taken one at a time. Composing two routes is already a route
/// ([`routes`](super::routes) closes the map set under composition), and chaining route
/// assertions would reintroduce activity-level composition.
pub fn route_delivered(bounds: &Bounds, kept: &HashSet<Cell>, routes: &[Route]) -> HashSet<Pair> {
    let mut out: HashSet<Pair> = asserted_by_type(bounds, kept).into_iter().flatten().collect();
    for r in routes {
        out.extend(asserted_of_route(bounds, kept, r));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    fn oid(n: u32) -> ObjectIndex {
        ObjectIndex::from(n)
    }

    fn ob(object: u32, at: &[(usize, i64, i64)]) -> ObjectBounds {
        ObjectBounds {
            object: oid(object),
            at: at.to_vec(),
            // One occurrence per activity, i.e. not a loop; none of these tests read `times`.
            times: at.iter().map(|(a, _, _)| (*a, 1)).collect(),
        }
    }

    /// The handoff shape: an item at 0 and 1, its package at 1 and 2, no type at 0 and 2.
    fn handoff() -> (Bounds, Route) {
        let bounds = Bounds {
            per_type: vec![
                vec![ob(0, &[(0, 10, 10), (1, 20, 20)])],
                vec![ob(1, &[(1, 20, 20), (2, 30, 30)])],
            ],
        };
        let route = Route {
            source: 0,
            target: 1,
            via: vec![0],
            f: HashMap::from([(oid(0), oid(1))]),
        };
        (bounds, route)
    }

    #[test]
    fn route_spans_what_no_single_type_does() {
        let (bounds, route) = handoff();
        let kept: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (1, 1), (2, 1)]);

        let drawn: HashSet<Pair> = asserted_by_type(&bounds, &kept).into_iter().flatten().collect();
        assert!(!drawn.contains(&(0, 2)), "no type is at both 0 and 2");

        let delivered = route_delivered(&bounds, &kept, &[route]);
        assert!(delivered.contains(&(0, 2)), "the map carries it");
        assert!(!delivered.contains(&(2, 0)));
    }

    #[test]
    fn the_map_is_read_backwards_too() {
        // The package's activity 2 happens *before* the item's activity 0.
        let bounds = Bounds {
            per_type: vec![
                vec![ob(0, &[(0, 30, 30)])],
                vec![ob(1, &[(2, 10, 10)])],
            ],
        };
        let route = Route {
            source: 0,
            target: 1,
            via: vec![0],
            f: HashMap::from([(oid(0), oid(1))]),
        };
        let kept: HashSet<Cell> = HashSet::from([(0, 0), (2, 1)]);
        let delivered = route_delivered(&bounds, &kept, &[route]);
        assert!(delivered.contains(&(2, 0)), "target activity precedes source");
        assert!(!delivered.contains(&(0, 2)));
    }

    #[test]
    fn a_pair_witnessed_both_ways_asserts_nothing() {
        let (mut bounds, route) = handoff();
        // A second item whose package does activity 2 before the item does activity 0.
        bounds.per_type[0].push(ob(2, &[(0, 40, 40)]));
        bounds.per_type[1].push(ob(3, &[(2, 5, 5)]));
        let mut route = route;
        route.f.insert(oid(2), oid(3));

        let kept: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (1, 1), (2, 1)]);
        assert!(!route_delivered(&bounds, &kept, &[route]).contains(&(0, 2)));
    }

    #[test]
    fn an_object_without_an_image_carries_nothing() {
        let (mut bounds, route) = handoff();
        bounds.per_type[0].push(ob(9, &[(0, 100, 100)]));
        let kept: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (1, 1), (2, 1)]);
        // Object 9 is outside the map's domain, so it neither adds nor blocks.
        assert!(route_delivered(&bounds, &kept, &[route]).contains(&(0, 2)));
    }

    #[test]
    fn dropping_an_unrelated_cell_cannot_create_an_assertion() {
        // The item also visits activity 3, where its package runs earlier: a backward
        // witness for (3, 2) only. Dropping (3, 0) must leave (0, 2) alone.
        let bounds = Bounds {
            per_type: vec![
                vec![ob(0, &[(0, 10, 10), (1, 20, 20), (3, 90, 90)])],
                vec![ob(1, &[(1, 20, 20), (2, 30, 30)])],
            ],
        };
        let route = Route {
            source: 0,
            target: 1,
            via: vec![0],
            f: HashMap::from([(oid(0), oid(1))]),
        };
        let wide: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (3, 0), (1, 1), (2, 1)]);
        let narrow: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (1, 1), (2, 1)]);
        let big = route_delivered(&bounds, &wide, std::slice::from_ref(&route));
        let small = route_delivered(&bounds, &narrow, &[route]);
        assert!(small.is_subset(&big), "{small:?} must be inside {big:?}");
        assert!(big.contains(&(2, 3)), "the wider set sees the backward witness");
    }

    #[test]
    fn a_cell_left_out_of_the_keep_set_is_not_read() {
        let (bounds, route) = handoff();
        let kept: HashSet<Cell> = HashSet::from([(0, 0), (1, 0), (1, 1)]);
        assert!(!route_delivered(&bounds, &kept, &[route]).contains(&(0, 2)));
    }
}
