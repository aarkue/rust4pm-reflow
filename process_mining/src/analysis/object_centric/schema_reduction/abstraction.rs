//! What a behavioural abstraction supplies, and the log's instantiation of it.
//!
//! An abstraction answers three questions:
//!
//! - what does an object type **assert** over the activities where it is kept
//!   ([`Abstraction::asserts`]);
//! - which assertions a keep-set has to cover ([`Abstraction::required`]);
//! - when do the kept cells **cover** an assertion ([`Abstraction::cover`]).
//!
//! Every covering rule has to be monotone in the keep-set: adding a cell may add covered
//! assertions and never removes one. Each cutoff (composition depth, closure budget, expansion
//! work budget) then fails toward leaving a cell flowing instead of losing an assertion.

use std::collections::{BTreeSet, HashSet};

use super::{
    bounds::Bounds,
    cells::{ActivityIndex, Cell},
    coverage::Chained,
    facts::{asserted_by_type, asserted_of_type},
    novelty::Pair,
    schema::ObjectTypeIndex,
};

/// One statement a type's sublog supports over activities.
///
/// The kinds are named here so that a covering rule can be written per kind instead of per
/// instantiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Assertion {
    /// `a` before `b`.
    Order(ActivityIndex, ActivityIndex),
    /// No object of the type attends both activities.
    Never(ActivityIndex, ActivityIndex),
    /// An object attending either activity attends both.
    Together(ActivityIndex, ActivityIndex),
    /// An object attends the activity more than once.
    Repeats(ActivityIndex),
    /// An object alternates between the two activities: they sit on the body and redo sides
    /// of a loop cut, or a sequence cut inside a loop body separates them. Unordered, stored
    /// with the smaller index first.
    Looped(ActivityIndex, ActivityIndex),
}

/// Which kind an assertion is, for per-kind residual reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AssertionKind {
    Order,
    Never,
    Together,
    Repeats,
    Looped,
}

impl Assertion {
    /// The activities the assertion mentions, a repeat naming its own activity twice.
    ///
    /// Both entries have to be kept for the assertion to be shown, and the duplicate is
    /// what makes a one-activity assertion need one cell without a second code path.
    pub fn activities(self) -> [ActivityIndex; 2] {
        match self {
            Assertion::Order(x, y)
            | Assertion::Never(x, y)
            | Assertion::Together(x, y)
            | Assertion::Looped(x, y) => [x, y],
            Assertion::Repeats(a) => [a, a],
        }
    }

    pub fn mentions(self, a: ActivityIndex) -> bool {
        let [x, y] = self.activities();
        x == a || y == a
    }

    pub fn kind(self) -> AssertionKind {
        match self {
            Assertion::Order(..) => AssertionKind::Order,
            Assertion::Never(..) => AssertionKind::Never,
            Assertion::Together(..) => AssertionKind::Together,
            Assertion::Repeats(..) => AssertionKind::Repeats,
            Assertion::Looped(..) => AssertionKind::Looped,
        }
    }

    /// The ordered pair, for the ordering-only paths that still speak in pairs.
    pub fn as_pair(self) -> Option<Pair> {
        match self {
            Assertion::Order(x, y) => Some((x, y)),
            _ => None,
        }
    }
}

/// What a keep-set covers, under one instantiation's own rule.
pub trait Cover {
    fn holds(&self, s: Assertion) -> bool;

    /// How many of `required` the keep-set covers.
    fn count_in(&self, required: &[Assertion]) -> usize {
        required.iter().filter(|s| self.holds(**s)).count()
    }

    /// The assertions of `required` still uncovered.
    fn missing(&self, required: &[Assertion]) -> Vec<Assertion> {
        required.iter().filter(|s| !self.holds(**s)).copied().collect()
    }
}

/// A behavioural abstraction: what each type asserts, what must be covered, and what covers
/// it.
pub trait Abstraction {
    /// One type's assertions under a keep-set, restricted to assertions all of whose
    /// activities the keep-set holds for that type.
    fn asserts(&self, kept: &HashSet<Cell>, t: ObjectTypeIndex) -> HashSet<Assertion>;

    /// Every assertion a keep-set has to cover, stated by this abstraction over the cells it
    /// was built on. These block a cell from leaving the flow layer.
    fn required(&self) -> &[Assertion];

    /// Assertions this abstraction makes that are measured but do not block a cell from
    /// leaving the flow layer.
    ///
    /// The default is that only orderings block. An abstraction supplying a second kind
    /// reports it here, and a strict constructor moves the same assertions into
    /// [`required`](Self::required).
    fn reported(&self) -> &[Assertion] {
        &[]
    }

    /// How many of [`reported`](Self::reported) a keep-set happens to cover, out of the
    /// total. The residual is the difference: what a reduced model stops showing.
    fn residual(&self, kept: &HashSet<Cell>, n_activities: usize) -> (usize, usize) {
        let reported = self.reported();
        if reported.is_empty() {
            return (0, 0);
        }
        let asserted: HashSet<Assertion> = kept
            .iter()
            .map(|&(_, t)| t)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .flat_map(|t| self.asserts(kept, t))
            .collect();
        let cover = self.cover(kept, &asserted, n_activities);
        (cover.count_in(reported), reported.len())
    }

    /// The covering rule, given the keep-set and what it asserts.
    ///
    /// `asserted` is the union over kept types, which is all a rule closing over activities
    /// needs. `kept` is there for a rule that has to go back to its own model, as OC-DECLARE
    /// does to tell a response arc from a precedence one after the endpoint filter.
    fn cover(
        &self,
        kept: &HashSet<Cell>,
        asserted: &HashSet<Assertion>,
        n_activities: usize,
    ) -> Box<dyn Cover>;
}

/// Deduplicated and in a fixed order, so a requirement does not depend on hash order.
pub(super) fn requirement(assertions: impl IntoIterator<Item = Assertion>) -> Vec<Assertion> {
    assertions.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

/// The ordered pairs of `per_type` both of whose endpoints `over` holds for that type.
fn pairs_over(per_type: &[HashSet<Pair>], over: &HashSet<Cell>) -> Vec<Assertion> {
    requirement(per_type.iter().enumerate().flat_map(|(t, ps)| {
        ps.iter()
            .filter(move |&&(x, y)| over.contains(&(x, t)) && over.contains(&(y, t)))
            .map(|&(x, y)| Assertion::Order(x, y))
    }))
}

/// Orderings closed under composition.
///
/// A pair `x < z` is covered when kept types order `x < y` and `y < z`, whichever types
/// supply the two hops, because a reader composes them off the picture. Assertions of any
/// other kind are reported uncovered, since chaining says nothing about them.
pub struct ChainedCover(Chained);

impl ChainedCover {
    pub fn of(asserted: &HashSet<Assertion>, n_activities: usize) -> Self {
        let pairs: HashSet<Pair> = asserted.iter().filter_map(|s| s.as_pair()).collect();
        Self(Chained::of(&pairs, n_activities))
    }
}

impl Cover for ChainedCover {
    fn holds(&self, s: Assertion) -> bool {
        match s {
            Assertion::Order(x, y) => self.0.holds(x, y),
            _ => false,
        }
    }
}

/// The orderings the log asserts: [`asserted_of_type`] over the saturated bounds.
///
/// The default instantiation. Its requirement is the log's own ordering relation over the
/// cells it is built on, the same set `Saturation::target_pairs` computes.
pub struct LogAbstraction<'a> {
    bounds: &'a Bounds,
    required: Vec<Assertion>,
}

impl<'a> LogAbstraction<'a> {
    /// The orderings the log asserts over `over`, which is what a keep-set must cover.
    pub fn over(bounds: &'a Bounds, over: &HashSet<Cell>) -> Self {
        Self {
            bounds,
            required: requirement(
                asserted_by_type(bounds, over)
                    .into_iter()
                    .flatten()
                    .map(|(x, y)| Assertion::Order(x, y)),
            ),
        }
    }
}

impl Abstraction for LogAbstraction<'_> {
    fn asserts(&self, kept: &HashSet<Cell>, t: ObjectTypeIndex) -> HashSet<Assertion> {
        asserted_of_type(self.bounds, kept, t)
            .into_iter()
            .map(|(x, y)| Assertion::Order(x, y))
            .collect()
    }

    fn required(&self) -> &[Assertion] {
        &self.required
    }

    fn cover(
        &self,
        _kept: &HashSet<Cell>,
        asserted: &HashSet<Assertion>,
        n_activities: usize,
    ) -> Box<dyn Cover> {
        Box::new(ChainedCover::of(asserted, n_activities))
    }
}

/// A per-type ordered-pair set fixed before the search runs, e.g. the per-type `IMf` trees
/// of [`model_abstraction`](super::model_abstraction).
///
/// The endpoint filter is the one [`asserted_of_type`] applies: both activities kept for the
/// type. A type past the end of the slice asserts nothing. Its requirement is what the models
/// themselves order over the cells it is built on. Chaining is the covering rule, since the
/// pairs are eventually-follows and a reader composes two hops off the picture.
pub struct PairAbstraction<'a> {
    per_type: &'a [HashSet<Pair>],
    required: Vec<Assertion>,
}

impl<'a> PairAbstraction<'a> {
    /// The orderings these models assert over `over`.
    pub fn over(per_type: &'a [HashSet<Pair>], over: &HashSet<Cell>) -> Self {
        Self {
            per_type,
            required: pairs_over(per_type, over),
        }
    }
}

impl Abstraction for PairAbstraction<'_> {
    fn asserts(&self, kept: &HashSet<Cell>, t: ObjectTypeIndex) -> HashSet<Assertion> {
        self.per_type
            .get(t)
            .into_iter()
            .flatten()
            .filter(|&&(x, y)| kept.contains(&(x, t)) && kept.contains(&(y, t)))
            .map(|&(x, y)| Assertion::Order(x, y))
            .collect()
    }

    fn required(&self) -> &[Assertion] {
        &self.required
    }

    fn cover(
        &self,
        _kept: &HashSet<Cell>,
        asserted: &HashSet<Assertion>,
        n_activities: usize,
    ) -> Box<dyn Cover> {
        Box::new(ChainedCover::of(asserted, n_activities))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeat_names_its_own_activity_twice() {
        assert_eq!(Assertion::Repeats(3).activities(), [3, 3]);
        assert!(Assertion::Repeats(3).mentions(3));
        assert!(!Assertion::Repeats(3).mentions(4));
    }

    #[test]
    fn chaining_covers_orderings_and_refuses_the_other_kinds() {
        let asserted: HashSet<Assertion> =
            [Assertion::Order(0, 1), Assertion::Order(1, 2)].into_iter().collect();
        let cov = ChainedCover::of(&asserted, 3);
        assert!(cov.holds(Assertion::Order(0, 2)), "the handoff composes");
        assert!(!cov.holds(Assertion::Order(2, 0)));
        assert!(!cov.holds(Assertion::Never(0, 2)), "chaining says nothing about exclusion");
        assert!(!cov.holds(Assertion::Together(0, 1)));
    }

    #[test]
    fn a_requirement_is_deduplicated_and_ordered() {
        let r = requirement([Assertion::Order(1, 2), Assertion::Order(0, 1), Assertion::Order(1, 2)]);
        assert_eq!(r, vec![Assertion::Order(0, 1), Assertion::Order(1, 2)]);
    }

    /// The lens states its own requirement, and it is the endpoint filter that restricts it:
    /// a pair whose endpoints are not both kept for its type is not required.
    #[test]
    fn a_precomputed_lens_requires_only_what_its_cells_carry() {
        let per_type: Vec<HashSet<Pair>> =
            vec![[(0, 1), (1, 2)].into_iter().collect(), [(3, 4)].into_iter().collect()];
        let over: HashSet<Cell> = [(0, 0), (1, 0), (2, 0), (3, 1)].into_iter().collect();
        let lens = PairAbstraction::over(&per_type, &over);
        assert_eq!(
            lens.required(),
            &[Assertion::Order(0, 1), Assertion::Order(1, 2)],
            "type 1's pair needs activity 4, which no cell keeps"
        );
    }

    /// The default reading of a keep-set is [`asserted_of_type`].
    #[test]
    fn the_log_lens_reads_what_asserted_of_type_reads() {
        let bounds = Bounds::default();
        let lens = LogAbstraction::over(&bounds, &HashSet::new());
        let kept: HashSet<Cell> = HashSet::new();
        assert!(lens.asserts(&kept, 0).is_empty());
        assert!(lens.required().is_empty());
    }
}
