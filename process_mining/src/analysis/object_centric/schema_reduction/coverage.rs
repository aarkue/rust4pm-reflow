use std::collections::HashSet;

use super::{
    bounds::Bounds,
    cells::{ActivityIndex, Cell},
    facts::asserted_by_type,
    novelty::Pair,
};

/// The activity pairs a keep-set draws: some kept type orders them and is kept at both
/// endpoints.
///
/// [`asserted_of_type`] reads only the activities whose cell is kept, so this is the union
/// over types of what each one asserts under the keep-set.
///
/// [`asserted_of_type`]: super::asserted_of_type
pub fn drawn(bounds: &Bounds, kept: &HashSet<Cell>) -> HashSet<Pair> {
    asserted_by_type(bounds, kept).into_iter().flatten().collect()
}

/// The drawn relation closed under composition: the handoff chain.
///
/// Drawn-only coverage needs one type kept at both endpoints of a pair, which forces a
/// type spanning the whole process. Chaining through a third activity is sound because
/// the reduced model is exact for eventually-follows: if `items` orders `pick item <
/// create package` and `packages` orders `create package < package delivered`, a reader
/// composes them off the picture.
///
/// The closure is over activities and forgets which type supplied each hop: a different
/// type per hop is still a chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chained {
    n: usize,
    words: usize,
    bits: Vec<u64>,
}

impl Chained {
    /// Close a drawn relation over `n` activities.
    ///
    /// Bitset Warshall: the searches ask this once per candidate cell per round, so the
    /// closure dominates the cost of a search on logs with many activities.
    pub fn of(pairs: &HashSet<Pair>, n: usize) -> Self {
        let words = n.div_ceil(64);
        let mut bits = vec![0u64; n * words];
        for (x, y) in pairs {
            bits[x * words + y / 64] |= 1u64 << (y % 64);
        }
        for k in 0..n {
            let (head, tail) = bits.split_at_mut(k * words);
            let (row_k, rest) = tail.split_at_mut(words);
            let apply = |row: &mut [u64]| {
                if row[k / 64] >> (k % 64) & 1 == 1 {
                    for w in 0..words {
                        row[w] |= row_k[w];
                    }
                }
            };
            for row in head.chunks_mut(words) {
                apply(row);
            }
            for row in rest.chunks_mut(words) {
                apply(row);
            }
        }
        Self { n, words, bits }
    }

    /// Is `y` reachable from `x` through drawn pairs?
    pub fn holds(&self, x: ActivityIndex, y: ActivityIndex) -> bool {
        x < self.n && y < self.n && self.bits[x * self.words + y / 64] >> (y % 64) & 1 == 1
    }

    /// How many of `target` are delivered.
    pub fn count_in(&self, target: &[Pair]) -> usize {
        target.iter().filter(|(x, y)| self.holds(*x, *y)).count()
    }

    /// The pairs `target` still misses.
    pub fn missing<'a>(&'a self, target: &'a [Pair]) -> impl Iterator<Item = Pair> + 'a {
        target.iter().filter(|(x, y)| !self.holds(*x, *y)).copied()
    }
}

/// What a keep-set delivers of the target, drawn and chained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    /// Target pairs a kept type orders with both endpoints kept.
    pub drawn: usize,
    /// Target pairs delivered drawn or by a chain. Always at least `drawn`.
    pub chained: usize,
    /// Target pairs in total.
    pub target: usize,
}

impl Coverage {
    /// Whether every target pair is delivered.
    pub fn complete(&self) -> bool {
        self.chained >= self.target
    }
}

/// Both coverage measures of one keep-set.
pub fn coverage(
    bounds: &Bounds,
    kept: &HashSet<Cell>,
    target: &[Pair],
    n_activities: usize,
) -> Coverage {
    let d = drawn(bounds, kept);
    let in_target = target.iter().filter(|p| d.contains(*p)).count();
    Coverage {
        drawn: in_target,
        chained: Chained::of(&d, n_activities).count_in(target),
        target: target.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_delivers_what_no_single_type_draws() {
        // `items` orders 0 < 1, `packages` orders 1 < 2, nothing orders 0 < 2 directly.
        let d: HashSet<Pair> = [(0, 1), (1, 2)].into_iter().collect();
        let c = Chained::of(&d, 3);
        assert!(c.holds(0, 1) && c.holds(1, 2));
        assert!(c.holds(0, 2), "the handoff composes");
        assert!(!c.holds(2, 0), "and it does not run backwards");
    }

    #[test]
    fn the_closure_is_a_closure_and_not_one_hop() {
        let d: HashSet<Pair> = (0..8).map(|i| (i, i + 1)).collect();
        let c = Chained::of(&d, 9);
        assert!(c.holds(0, 8));
        assert_eq!(c.count_in(&[(0, 8), (3, 7), (8, 0)]), 2);
    }

    #[test]
    fn reachability_crosses_the_word_boundary() {
        let d: HashSet<Pair> = (0..130).map(|i| (i, i + 1)).collect();
        let c = Chained::of(&d, 131);
        assert!(c.holds(0, 130) && c.holds(63, 64) && c.holds(1, 129));
        assert!(!c.holds(130, 0));
    }
}
