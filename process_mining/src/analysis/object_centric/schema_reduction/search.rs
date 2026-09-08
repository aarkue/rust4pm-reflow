use std::collections::{BTreeSet, HashMap, HashSet};

use super::{
    abstraction::{Abstraction, Assertion, Cover, LogAbstraction},
    arcs::TraceVariants,
    cells::{ActivityIndex, Cell},
    coverage::Coverage,
    novelty::Pair,
    saturation::Saturation,
    schema::ObjectTypeIndex,
};

/// Word operations of chained-closure rebuilding above which a search gives up and
/// returns `ran: false`.
///
/// The closure is `O(A^3 / 64)` and a search rebuilds it once per candidate it scores.
/// The count is accumulated as the search runs, in word operations rather than seconds,
/// so whether a search ran depends on the log and not on the machine. At roughly 1.7e9
/// word operations per second this ceiling is a bit under half an hour of closure work.
pub const SEARCH_CLOSURE_BUDGET: u64 = 3_000_000_000_000;

/// Nodes an exact search may open before it gives up and returns `ran: false`.
///
/// The leaf check is one chained closure, so node count is also checked against
/// [`SEARCH_CLOSURE_BUDGET`] through [`SearchInput::affordable`]. This constant bounds the
/// shape of the enumeration separately, so a log with few activities and many types cannot
/// spend an hour inside closures that are individually cheap.
pub const EXACT_NODE_BUDGET: u64 = 50_000_000;

/// Which construction produced a flow layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Cheapest-first cover: walk the target pairs rarest-first and keep the cheapest type
    /// that orders each one still missing.
    Greedy,
    /// One flow type per activity, a second only where coverage forces it.
    Handoff,
    /// The minimum of the objective over every preserving layer, by branch and bound over
    /// the cells. See [`exact_with_objective`].
    Exact,
}

impl Strategy {
    /// The name used in the census.
    pub fn label(&self) -> &'static str {
        match self {
            Strategy::Greedy => "greedy",
            Strategy::Handoff => "handoff",
            Strategy::Exact => "exact",
        }
    }
}

/// What a search minimises.
///
/// The feasible region (what the schema and the abstraction allow) is the same for all
/// three. Each objective is a function of the assignment alone, so no discovery run is
/// needed inside the search loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Coloured directly-follows arcs. The default.
    Arcs,
    /// Cells left in the flow layer. Counts the grid, never the model.
    Cells,
    /// Participations the flow layer carries, i.e. `(event, object)` pairs.
    Participations,
}

impl Objective {
    /// The name used in the census.
    pub fn label(&self) -> &'static str {
        match self {
            Objective::Arcs => "arcs",
            Objective::Cells => "cells",
            Objective::Participations => "participations",
        }
    }

    /// Cost of one cell for a construction that adds cells one at a time.
    ///
    /// Arcs are not a per-cell quantity: a cell's arc contribution depends on the rest of
    /// its type's row, since projecting an activity out splices its neighbours together. So
    /// under [`Arcs`](Objective::Arcs) the constructions steer by participations and the
    /// finished layers are ranked by arcs. The other two objectives are per-cell.
    fn cell_cost(self, input: &SearchInput<'_>, c: Cell) -> usize {
        match self {
            Objective::Arcs | Objective::Participations => input.cost(c),
            Objective::Cells => 1,
        }
    }

    /// Rank finished layers: the objective, then the other two as tie-breaks.
    pub fn key(self, f: &FlowLayer) -> (usize, usize, usize) {
        match self {
            Objective::Arcs => (f.arcs, f.participations, f.cells.len()),
            Objective::Cells => (f.cells.len(), f.arcs, f.participations),
            Objective::Participations => (f.participations, f.arcs, f.cells.len()),
        }
    }

    /// The primary component, from the three quantities a row contributes.
    ///
    /// Additive over types for all three: a coloured arc names its own type, so distinct
    /// types contribute distinct arcs and the totals are sums, not unions. That is what
    /// lets [`exact_with_objective`] score a type's row without knowing the rest.
    fn primary(self, arcs: usize, cells: usize, participations: usize) -> usize {
        match self {
            Objective::Arcs => arcs,
            Objective::Cells => cells,
            Objective::Participations => participations,
        }
    }
}

/// A flow layer and its measurements.
#[derive(Debug, Clone)]
pub struct FlowLayer {
    /// Which search built it.
    pub strategy: Strategy,
    /// What that search was minimising.
    pub objective: Objective,
    /// The cells that draw arcs.
    pub cells: HashSet<Cell>,
    /// What it delivers of the target.
    pub coverage: Coverage,
    /// Arcs of the model it induces, over `max`. The objective under
    /// [`Objective::Arcs`], which is the default.
    pub arcs: usize,
    /// Participations it carries. The tie-break under [`Objective::Arcs`].
    pub participations: usize,
    /// Cells the post-pass removed because no delivered pair needed them.
    pub dropped: usize,
    /// Components of the activity-type incidence graph. Reported, not enforced: requiring
    /// connectivity would make some types permanently undemotable.
    pub incidence_components: usize,
    /// Whether the search ran at all.
    pub ran: bool,
}

/// What the kept cells assert, maintained incrementally.
///
/// Adding a cell of type `t` changes what `t` asserts and nothing else, so only that type's
/// set is rebuilt. The covering rule is rebuilt lazily, once per kept cell rather than once
/// per coverage query.
struct Drawn<'a> {
    lens: &'a dyn Abstraction,
    n_acts: usize,
    kept: HashSet<Cell>,
    per_type: Vec<HashSet<Assertion>>,
    cover: Option<Box<dyn Cover>>,
    rebuilds: u64,
}

impl<'a> Drawn<'a> {
    fn new(lens: &'a dyn Abstraction, n_types: usize, n_acts: usize) -> Self {
        Self {
            lens,
            n_acts,
            kept: HashSet::new(),
            per_type: vec![HashSet::new(); n_types],
            cover: None,
            rebuilds: 0,
        }
    }

    fn refresh(&mut self, t: ObjectTypeIndex) {
        if t < self.per_type.len() {
            self.per_type[t] = self.lens.asserts(&self.kept, t);
        }
        self.cover = None;
    }

    fn insert(&mut self, c: Cell) -> bool {
        if !self.kept.insert(c) {
            return false;
        }
        self.refresh(c.1);
        true
    }

    fn remove(&mut self, c: Cell) -> bool {
        if !self.kept.remove(&c) {
            return false;
        }
        self.refresh(c.1);
        true
    }

    /// What the kept cells assert directly, before any covering rule is applied.
    fn asserted(&self) -> HashSet<Assertion> {
        self.per_type.iter().flatten().copied().collect()
    }

    fn covered(&mut self) -> &dyn Cover {
        if self.cover.is_none() {
            self.rebuilds += 1;
            let asserted = self.asserted();
            self.cover = Some(self.lens.cover(&self.kept, &asserted, self.n_acts));
        }
        self.cover.as_deref().unwrap()
    }

    fn delivered(&mut self, required: &[Assertion]) -> usize {
        self.covered().count_in(required)
    }
}

/// Everything both searches read, so neither of them touches the log.
#[derive(Debug)]
pub struct SearchInput<'a> {
    /// The saturation the coverage constraint is stated against.
    pub max: &'a Saturation,
    /// Trace variants over `max`, for the arc objective.
    pub variants: &'a TraceVariants,
    /// The cells a search may keep. `max.cells` for the unrestricted run; `max.cells`
    /// minus the expansions novelty rejected for the novelty-filtered run.
    pub allowed: &'a HashSet<Cell>,
    /// Cells the extraction recorded, which is what "native" means in the handoff
    /// constraint.
    pub recorded: &'a HashSet<Cell>,
    /// Every activity pair ordered in `max`.
    ///
    /// No search reads this: each [`Abstraction`] states its own requirement, and
    /// [`LogAbstraction::over`] over `max.cells` recomputes exactly this set. Kept because
    /// [`coverage`](super::coverage) reports against it.
    pub target: &'a [Pair],
    /// Objects of each type: the handoff constraint's reading of "coarsest".
    pub objects_per_type: &'a [usize],
    /// Refinement-class representative per type, from the schema closure. Tie-break
    /// between types of one refinement class.
    pub rep: &'a [ObjectTypeIndex],
    /// Object type names, for a stable final tie-break.
    pub types: &'a [String],
    /// Activity count the chained closure spans.
    pub n_activities: usize,
}

impl SearchInput<'_> {
    fn n_types(&self) -> usize {
        self.types.len()
    }

    /// What each type asserts in `max`, under `lens`.
    fn asserts_in_max(&self, lens: &dyn Abstraction) -> Vec<HashSet<Assertion>> {
        (0..self.n_types())
            .map(|t| lens.asserts(&self.max.cells, t))
            .collect()
    }

    fn cost(&self, c: Cell) -> usize {
        self.max.cost.get(&c).copied().unwrap_or(0)
    }

    /// What one chained-closure rebuild costs, in word operations.
    fn rebuild_cost(&self) -> u64 {
        let a = self.n_activities as u64;
        (a * a * a / 64).max(1)
    }

    /// Whether a search that has rebuilt the closure this many times may continue.
    fn affordable(&self, rebuilds: u64) -> bool {
        rebuilds.saturating_mul(self.rebuild_cost()) <= SEARCH_CLOSURE_BUDGET
    }
}

/// Cheapest-first cover.
///
/// Target pairs are walked rarest-first (fewest types ordering them), because a pair only
/// one type can deliver forces that type, and keeping it early makes its other pairs free.
/// Ties are broken on the pair itself so the answer does not depend on hash order.
///
/// Among equally cheap types, the one that orders more target pairs wins, then the
/// refinement class representative, so the tie is closed by the schema and not by
/// enumeration order.
pub fn greedy(input: &SearchInput<'_>) -> FlowLayer {
    greedy_with(input, &LogAbstraction::over(&input.max.bounds, &input.max.cells))
}

/// [`greedy_with`] under any objective. [`greedy_with`] is this at [`Objective::Arcs`].
pub fn greedy_with_objective(
    input: &SearchInput<'_>,
    lens: &dyn Abstraction,
    obj: Objective,
) -> FlowLayer {
    let required = lens.required().to_vec();
    let mut st = Drawn::new(lens, input.n_types(), input.n_activities);
    if !cover(&mut st, input, &required, obj) {
        return unrun(Strategy::Greedy, obj, &required);
    }
    finish(Strategy::Greedy, obj, st, input, &required)
}

/// [`greedy`] under any behavioural abstraction.
///
/// [`greedy`] is this called with [`LogAbstraction`] over `max`'s own bounds and cells,
/// whose requirement is exactly
/// [`Saturation::target_pairs`](super::Saturation::target_pairs).
pub fn greedy_with(input: &SearchInput<'_>, lens: &dyn Abstraction) -> FlowLayer {
    greedy_with_objective(input, lens, Objective::Arcs)
}

/// Keep both endpoints of one type for every target pair `st` still misses.
///
/// Returns whether it stayed inside the closure budget.
///
/// Both endpoints at once is what makes this complete where a cell-at-a-time climb is not:
/// an ordering needs a type kept at `x` and at `y`, so neither half on its own gains
/// anything.
///
/// Complete relative to `allowed` only. `target` is stated against `max`, so a pair that
/// only a novelty-rejected expansion cell orders has no admissible candidate and is
/// skipped.
fn cover(
    st: &mut Drawn<'_>,
    input: &SearchInput<'_>,
    required: &[Assertion],
    obj: Objective,
) -> bool {
    let asserts: Vec<HashSet<Assertion>> = (0..input.n_types())
        .map(|t| st.lens.asserts(input.allowed, t))
        .collect();

    let wanted: HashSet<Assertion> = required.iter().copied().collect();
    let mut by_assertion: HashMap<Assertion, Vec<ObjectTypeIndex>> = HashMap::new();
    for (t, ss) in asserts.iter().enumerate() {
        for s in ss.iter().filter(|s| wanted.contains(*s)) {
            by_assertion.entry(*s).or_default().push(t);
        }
    }
    let mut order: Vec<Assertion> = required.to_vec();
    order.sort_by_key(|s| (by_assertion.get(s).map_or(0, Vec::len), *s));

    for s in &order {
        if !input.affordable(st.rebuilds) {
            return false;
        }
        if st.covered().holds(*s) {
            continue;
        }
        let Some(cands) = by_assertion.get(s) else { continue };
        let [x, y] = s.activities();
        let best = cands
            .iter()
            .filter(|t| input.allowed.contains(&(x, **t)) && input.allowed.contains(&(y, **t)))
            .min_by_key(|t| {
                // `y == x` for a one-activity assertion; count the cell once.
                let mut spend = 0;
                if !st.kept.contains(&(x, **t)) {
                    spend += obj.cell_cost(input, (x, **t));
                }
                if y != x && !st.kept.contains(&(y, **t)) {
                    spend += obj.cell_cost(input, (y, **t));
                }
                (
                    spend,
                    std::cmp::Reverse(asserts[**t].len()),
                    input.types[input.rep[**t]].clone(),
                    **t,
                )
            })
            .copied();
        if let Some(t) = best {
            st.insert((x, t));
            st.insert((y, t));
        }
    }
    true
}

/// One flow type per activity, a second only where coverage forces it.
///
/// A constraint on the shape, not a different objective. It reaches layers single-cell
/// swaps cannot, because a coarse type wins only when all of its activities switch
/// together.
///
/// "Coarsest" is operationalised as fewest objects. The seed type at an activity must
/// order something at that activity, not merely somewhere; otherwise a type present
/// everywhere and ordering almost nothing is seeded everywhere, and [`cover`] can repair
/// the coverage but not the seed. Everything the seed leaves uncovered is then passed to
/// [`cover`]; there is no hill climb, because single-cell additions cannot deliver an
/// ordering that needs both endpoints.
///
/// Deterministic across hash seeds. Not a normal form: the seed reads `recorded`, so the
/// answer changes with what the extraction happened to write down, and the ordering rule
/// is not monotone in participations (a pair witnessed both ways orders nothing), so
/// thinning a log can change the coverage target and with it the seed.
pub fn handoff(input: &SearchInput<'_>) -> FlowLayer {
    handoff_with(input, &LogAbstraction::over(&input.max.bounds, &input.max.cells))
}

/// [`handoff`] under any behavioural abstraction.
pub fn handoff_with(input: &SearchInput<'_>, lens: &dyn Abstraction) -> FlowLayer {
    handoff_with_objective(input, lens, Objective::Arcs)
}

/// [`handoff_with`] under any objective. [`handoff_with`] is this at [`Objective::Arcs`].
///
/// The seed is a constraint on the shape and does not read the objective: "coarsest type
/// that orders something here" is the same question whichever quantity is being minimised.
/// The objective enters through [`cover`]'s ranking and through [`prune`]'s order.
pub fn handoff_with_objective(
    input: &SearchInput<'_>,
    lens: &dyn Abstraction,
    obj: Objective,
) -> FlowLayer {
    let n_types = input.n_types();
    let required = lens.required().to_vec();
    let asserts = input.asserts_in_max(lens);

    let mut st = Drawn::new(lens, n_types, input.n_activities);
    for a in 0..input.n_activities {
        let native = (0..n_types)
            .filter(|t| {
                input.recorded.contains(&(a, *t))
                    && asserts[*t].iter().any(|s| s.mentions(a))
            })
            .min_by_key(|t| {
                (
                    input.objects_per_type.get(*t).copied().unwrap_or(usize::MAX),
                    input.types[input.rep[*t]].clone(),
                    *t,
                )
            });
        if let Some(t) = native {
            st.insert((a, t));
        }
    }

    // Every activity an uncovered assertion mentions is kept at once; no single cell can
    // deliver an ordering no kept type carries.
    if !cover(&mut st, input, &required, obj) {
        return unrun(Strategy::Handoff, obj, &required);
    }
    finish(Strategy::Handoff, obj, st, input, &required)
}

/// Drop every flow cell whose removal loses no covered assertion.
///
/// Both constructions keep more than needed: greedy keeps two cells for an ordering a
/// later addition chains anyway, and handoff seeds one type at every activity whether or
/// not it carries anything there. A cell is only redundant once its successors are kept.
///
/// Cells are considered most expensive first, so where two are jointly redundant the
/// cheaper one survives.
fn prune(
    st: &mut Drawn<'_>,
    input: &SearchInput<'_>,
    required: &[Assertion],
    obj: Objective,
) -> usize {
    let mut order: Vec<Cell> = st.kept.iter().copied().collect();
    order.sort_by_key(|c| (std::cmp::Reverse(obj.cell_cost(input, *c)), *c));
    let mut dropped = 0;
    for c in order {
        let before = st.delivered(required);
        st.remove(c);
        if st.delivered(required) < before {
            st.insert(c);
        } else {
            dropped += 1;
        }
    }
    dropped
}

fn finish(
    strategy: Strategy,
    obj: Objective,
    mut st: Drawn<'_>,
    input: &SearchInput<'_>,
    required: &[Assertion],
) -> FlowLayer {
    let dropped = prune(&mut st, input, required, obj);
    let cells = st.kept.clone();
    let shown = st.asserted();
    let coverage = Coverage {
        drawn: required.iter().filter(|s| shown.contains(*s)).count(),
        chained: st.delivered(required),
        target: required.len(),
    };
    score(strategy, obj, cells, coverage, dropped, input)
}

/// A finished layer's measurements, given its cells and what they cover.
///
/// Split out of [`finish`] because [`exact_with_objective`] must not run [`prune`]: under
/// [`Objective::Arcs`] dropping a cell can *raise* the arc count, since projecting an
/// activity out splices its neighbours into an arc no trace supported. The exact answer is
/// already minimal, so pruning it could only move it off the minimum.
fn score(
    strategy: Strategy,
    obj: Objective,
    cells: HashSet<Cell>,
    coverage: Coverage,
    dropped: usize,
    input: &SearchInput<'_>,
) -> FlowLayer {
    FlowLayer {
        strategy,
        objective: obj,
        arcs: input.variants.arc_set(&cells).len(),
        participations: input.max.participations(&cells),
        incidence_components: super::arcs::incidence_components(&cells),
        coverage,
        cells,
        dropped,
        ran: true,
    }
}

fn unrun(strategy: Strategy, obj: Objective, required: &[Assertion]) -> FlowLayer {
    FlowLayer {
        strategy,
        objective: obj,
        cells: HashSet::new(),
        coverage: Coverage {
            drawn: 0,
            chained: 0,
            target: required.len(),
        },
        arcs: 0,
        participations: 0,
        dropped: 0,
        incidence_components: 0,
        ran: false,
    }
}

/// Run both constructions, take the better, and say which won.
///
/// Neither dominates. Ranked on arcs, with participations as the tie-break, at equal
/// coverage.
pub fn best_of_two(input: &SearchInput<'_>) -> (FlowLayer, Vec<FlowLayer>) {
    best_of_two_with(input, &LogAbstraction::over(&input.max.bounds, &input.max.cells))
}

/// [`best_of_two`] under any behavioural abstraction.
pub fn best_of_two_with(
    input: &SearchInput<'_>,
    lens: &dyn Abstraction,
) -> (FlowLayer, Vec<FlowLayer>) {
    best_of_two_with_objective(input, lens, Objective::Arcs)
}

/// [`best_of_two_with`] under any objective, ranked by that objective's own key.
pub fn best_of_two_with_objective(
    input: &SearchInput<'_>,
    lens: &dyn Abstraction,
    obj: Objective,
) -> (FlowLayer, Vec<FlowLayer>) {
    let both = vec![
        handoff_with_objective(input, lens, obj),
        greedy_with_objective(input, lens, obj),
    ];
    let best = both
        .iter()
        .filter(|f| f.ran)
        .min_by_key(|f| (std::cmp::Reverse(f.coverage.chained), obj.key(f)))
        .cloned()
        .unwrap_or_else(|| unrun(Strategy::Greedy, obj, lens.required()));
    (best, both)
}

/// Everything the exact search needs about one object type, read from the log once.
///
/// None of it depends on the row being considered. [`Abstraction::asserts`] over a row is
/// the type's full assertion set restricted to the assertions whose activities the row
/// keeps (an assertion depends only on its two endpoints), so assertions are mined once per
/// type and a row is scored by intersection.
struct TypeRows {
    t: ObjectTypeIndex,
    /// Allowed activities of this type, sorted. A row is a subset of these.
    acts: Vec<ActivityIndex>,
    /// Every assertion the type can make, with the two activities it needs.
    all: Vec<(Assertion, [ActivityIndex; 2])>,
    /// Activity sequences of this type's objects, one per distinct variant.
    variants: Vec<Vec<ActivityIndex>>,
    /// Which variants each allowed activity occurs in, indexed like `acts`.
    in_variants: Vec<Vec<usize>>,
    /// Participations per allowed activity, indexed like `acts`.
    cost: Vec<usize>,
}

impl TypeRows {
    fn build(
        input: &SearchInput<'_>,
        lens: &dyn Abstraction,
        t: ObjectTypeIndex,
        acts: Vec<ActivityIndex>,
    ) -> Self {
        let full: HashSet<Cell> = acts.iter().map(|a| (*a, t)).collect();
        let all: Vec<(Assertion, [ActivityIndex; 2])> =
            lens.asserts(&full, t).into_iter().map(|s| (s, s.activities())).collect();
        let variants: Vec<Vec<ActivityIndex>> = input
            .variants
            .variants
            .iter()
            .filter(|v| v.object_type == t)
            .map(|v| v.activities.clone())
            .collect();
        let in_variants = acts
            .iter()
            .map(|a| {
                variants
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.contains(a))
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        let cost = acts.iter().map(|a| input.cost((*a, t))).collect();
        Self { t, acts, all, variants, in_variants, cost }
    }

    /// Coloured arcs this type draws under `row`, by projecting its variants onto it.
    fn arcs(&self, row: &[bool], n_acts: usize) -> usize {
        let keep: HashSet<ActivityIndex> =
            self.acts.iter().zip(row).filter(|(_, on)| **on).map(|(a, _)| *a).collect();
        if keep.len() < 2 {
            return 0;
        }
        let mut seen = vec![0u64; (n_acts * n_acts).div_ceil(64)];
        let mut n = 0;
        for v in &self.variants {
            let mut prev: Option<ActivityIndex> = None;
            for a in v.iter().filter(|a| keep.contains(a)) {
                if let Some(p) = prev {
                    let bit = p * n_acts + *a;
                    if seen[bit / 64] >> (bit % 64) & 1 == 0 {
                        seen[bit / 64] |= 1 << (bit % 64);
                        n += 1;
                    }
                }
                prev = Some(*a);
            }
        }
        n
    }
}

/// What the fixed rows have cost so far: arcs, cells, participations.
type Acc = (usize, usize, usize);

fn primary_of(obj: Objective, acc: Acc) -> usize {
    obj.primary(acc.0, acc.1, acc.2)
}

fn key_of(obj: Objective, acc: Acc) -> (usize, usize, usize) {
    match obj {
        Objective::Arcs => (acc.0, acc.2, acc.1),
        Objective::Cells => (acc.1, acc.0, acc.2),
        Objective::Participations => (acc.2, acc.0, acc.1),
    }
}

/// The depth-first walk, one cell at a time.
///
/// Cells are decided type by type, and a type's arcs join the running total only once its
/// whole row is decided, because taking an activity out splices its neighbours together and
/// a partial row therefore has no cost yet.
///
/// Three things keep the tree small. Each cell is tried **out before in**, so the cheapest
/// rows come first and the incumbent is strong early. Each node is cut against the incumbent
/// with a bound that stays valid even though arcs are not monotone in the row. And each type
/// boundary is cut on feasibility: coverage grows with the kept cells, so if everything still
/// undecided, taken at once, cannot cover the requirement, no completion can.
struct Descent<'a> {
    input: &'a SearchInput<'a>,
    lens: &'a dyn Abstraction,
    obj: Objective,
    required: Vec<Assertion>,
    types: Vec<TypeRows>,
    /// Union of what types `i..` can assert between them, for the feasibility cut.
    suffix: Vec<HashSet<Assertion>>,
    n_acts: usize,
    /// Row of the type being decided, indexed like its `acts`.
    row: Vec<bool>,
    /// How many activities of that row each of its variants holds, for the arc bound.
    overlap: Vec<usize>,
    /// Cells of the types already decided.
    fixed: Vec<Cell>,
    /// What those cells assert.
    asserted: HashSet<Assertion>,
    best: Option<(Vec<Cell>, (usize, usize, usize))>,
    nodes: u64,
    over_budget: bool,
}

impl Descent<'_> {
    fn best_primary(&self) -> usize {
        self.best.as_ref().map_or(usize::MAX, |(_, k)| k.0)
    }

    fn covers(&self, asserted: &HashSet<Assertion>) -> bool {
        let cover = self.lens.cover(&HashSet::new(), asserted, self.n_acts);
        cover.count_in(&self.required) >= self.required.len()
    }

    fn spend(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes > EXACT_NODE_BUDGET || !self.input.affordable(self.nodes) {
            self.over_budget = true;
            return false;
        }
        true
    }

    /// A lower bound on the objective of any completion of the current node.
    ///
    /// For cells and participations the partial row only adds, so it counts directly. For
    /// arcs it cannot: adding an activity can *remove* an arc, when the pair it splices was
    /// drawn only by variants that now route through the new activity. What does hold is
    /// that a variant holding `k` activities of the row draws at least `k - 1` arcs, and `k`
    /// only grows as the row does.
    fn bound(&self, ti: usize, acc: Acc) -> usize {
        match self.obj {
            Objective::Arcs => {
                acc.0 + self.overlap.iter().copied().max().unwrap_or(0).saturating_sub(1)
            }
            Objective::Cells => acc.1 + self.row.iter().filter(|on| **on).count(),
            Objective::Participations => {
                acc.2
                    + self
                        .row
                        .iter()
                        .enumerate()
                        .filter(|(_, on)| **on)
                        .map(|(i, _)| self.types[ti].cost[i])
                        .sum::<usize>()
            }
        }
    }

    /// Toggle cell `ci` of type `ti`, keeping the per-variant overlap in step.
    fn set(&mut self, ti: usize, ci: usize, on: bool) {
        self.row[ci] = on;
        let Descent { types, overlap, .. } = self;
        for v in &types[ti].in_variants[ci] {
            if on {
                overlap[*v] += 1;
            } else {
                overlap[*v] -= 1;
            }
        }
    }

    /// Begin type `ti`, or record the assignment when every type is decided.
    fn enter(&mut self, ti: usize, acc: Acc) {
        if self.over_budget || !self.spend() {
            return;
        }
        if primary_of(self.obj, acc) > self.best_primary() {
            return;
        }
        if ti == self.types.len() {
            if !self.covers(&self.asserted) {
                return;
            }
            let key = key_of(self.obj, acc);
            if self.best.as_ref().is_none_or(|(_, b)| key < *b) {
                let mut cells = self.fixed.clone();
                cells.sort_unstable();
                self.best = Some((cells, key));
            }
            return;
        }
        // Everything undecided, taken at once. Coverage is monotone in the kept cells, so if
        // this does not cover, nothing reachable from here does.
        let mut reach = self.asserted.clone();
        reach.extend(self.suffix[ti].iter().copied());
        if !self.covers(&reach) {
            return;
        }

        let saved_row = std::mem::replace(&mut self.row, vec![false; self.types[ti].acts.len()]);
        let saved_overlap =
            std::mem::replace(&mut self.overlap, vec![0; self.types[ti].variants.len()]);
        self.cells(ti, 0, acc);
        self.row = saved_row;
        self.overlap = saved_overlap;
    }

    /// Decide cell `ci` of type `ti`, out first so the cheap rows come first.
    fn cells(&mut self, ti: usize, ci: usize, acc: Acc) {
        if self.over_budget || !self.spend() {
            return;
        }
        if self.bound(ti, acc) > self.best_primary() {
            return;
        }
        if ci == self.types[ti].acts.len() {
            self.close(ti, acc);
            return;
        }
        self.cells(ti, ci + 1, acc);
        if self.over_budget {
            return;
        }
        self.set(ti, ci, true);
        self.cells(ti, ci + 1, acc);
        self.set(ti, ci, false);
    }

    /// The row of type `ti` is complete: score it, fix it, and move to the next type.
    fn close(&mut self, ti: usize, acc: Acc) {
        let arcs = self.types[ti].arcs(&self.row, self.n_acts);
        let t = self.types[ti].t;
        let kept: Vec<ActivityIndex> = self.types[ti]
            .acts
            .iter()
            .zip(&self.row)
            .filter(|(_, on)| **on)
            .map(|(a, _)| *a)
            .collect();
        let parts: usize = self.types[ti]
            .cost
            .iter()
            .zip(&self.row)
            .filter(|(_, on)| **on)
            .map(|(c, _)| *c)
            .sum();
        let next = (acc.0 + arcs, acc.1 + kept.len(), acc.2 + parts);
        if primary_of(self.obj, next) > self.best_primary() {
            return;
        }

        let live: HashSet<ActivityIndex> = kept.iter().copied().collect();
        let added: Vec<Assertion> = self.types[ti]
            .all
            .iter()
            .filter(|(_, [x, y])| live.contains(x) && live.contains(y))
            .map(|(s, _)| *s)
            .filter(|s| self.asserted.insert(*s))
            .collect();
        let n_fixed = self.fixed.len();
        self.fixed.extend(kept.into_iter().map(|a| (a, t)));

        self.enter(ti + 1, next);

        self.fixed.truncate(n_fixed);
        for s in added {
            self.asserted.remove(&s);
        }
    }
}

pub fn exact_with_objective(
    input: &SearchInput<'_>,
    lens: &dyn Abstraction,
    obj: Objective,
) -> FlowLayer {
    let required = lens.required().to_vec();
    let n_acts = input.n_activities;

    let mut by_type: HashMap<ObjectTypeIndex, Vec<ActivityIndex>> = HashMap::new();
    for (a, t) in input.allowed {
        by_type.entry(*t).or_default().push(*a);
    }
    let mut types: Vec<TypeRows> = by_type
        .into_iter()
        .map(|(t, mut acts)| {
            acts.sort_unstable();
            TypeRows::build(input, lens, t, acts)
        })
        .collect();
    // Types that can assert the most go first: they decide feasibility, so putting them
    // early makes the feasibility cut fire near the root rather than near the leaves.
    types.sort_by_key(|r| (std::cmp::Reverse(r.all.len()), r.t));

    let mut suffix: Vec<HashSet<Assertion>> = vec![HashSet::new(); types.len() + 1];
    for i in (0..types.len()).rev() {
        let mut s = suffix[i + 1].clone();
        s.extend(types[i].all.iter().map(|(a, _)| *a));
        suffix[i] = s;
    }

    // Seeded with the better construction, so the bound has something to cut against from
    // the first node. The answer is the optimum whether or not the seed was already it.
    let seed = best_of_two_with_objective(input, lens, obj).0;
    let best = seed.ran.then(|| {
        let mut cells: Vec<Cell> = seed.cells.iter().copied().collect();
        cells.sort_unstable();
        (cells, obj.key(&seed))
    });

    let mut walk = Descent {
        input,
        lens,
        obj,
        required,
        types,
        suffix,
        n_acts,
        row: Vec::new(),
        overlap: Vec::new(),
        fixed: Vec::new(),
        asserted: HashSet::new(),
        best,
        nodes: 0,
        over_budget: false,
    };
    walk.enter(0, (0, 0, 0));

    if walk.over_budget {
        return unrun(Strategy::Exact, obj, lens.required());
    }
    let Some((pick, _)) = walk.best else {
        return unrun(Strategy::Exact, obj, lens.required());
    };
    let cells: HashSet<Cell> = pick.into_iter().collect();
    let shown: HashSet<Assertion> = cells
        .iter()
        .map(|(_, t)| *t)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .flat_map(|t| lens.asserts(&cells, t))
        .collect();
    let cover = lens.cover(&cells, &shown, n_acts);
    let coverage = Coverage {
        drawn: walk.required.iter().filter(|s| shown.contains(*s)).count(),
        chained: cover.count_in(&walk.required),
        target: walk.required.len(),
    };
    score(Strategy::Exact, obj, cells, coverage, 0, input)
}

/// Allowed cells per object type, which is what the exact search branches over.
pub fn exact_cells_per_type(input: &SearchInput<'_>) -> Vec<usize> {
    let mut by_type: HashMap<ObjectTypeIndex, usize> = HashMap::new();
    for (_, t) in input.allowed {
        *by_type.entry(*t).or_default() += 1;
    }
    let mut counts: Vec<usize> = by_type.into_values().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    counts
}

/// [`exact_with_objective`] at [`Objective::Arcs`], under the log's own abstraction.
pub fn exact(input: &SearchInput<'_>) -> FlowLayer {
    exact_with_objective(
        input,
        &LogAbstraction::over(&input.max.bounds, &input.max.cells),
        Objective::Arcs,
    )
}

/// The flow layer ReFlow reports: both constructions are run and the better under
/// [`Objective::Arcs`] is taken. Neither construction dominates the other.
pub fn reflow_layer(input: &SearchInput<'_>) -> FlowLayer {
    reflow_layer_with(input, &LogAbstraction::over(&input.max.bounds, &input.max.cells))
}

/// [`reflow_layer`] under any behavioural abstraction.
pub fn reflow_layer_with(input: &SearchInput<'_>, lens: &dyn Abstraction) -> FlowLayer {
    best_of_two_with_objective(input, lens, Objective::Arcs).0
}

/// Activities the flow layer leaves with no flow cell at all. Involvement still carries
/// their objects.
pub fn activities_without_flow(cells: &HashSet<Cell>, n_activities: usize) -> Vec<ActivityIndex> {
    let live: HashSet<ActivityIndex> = cells.iter().map(|(a, _)| *a).collect();
    (0..n_activities).filter(|a| !live.contains(a)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::object_centric::schema_reduction::{
        agreed_routes, novel_cells, novelty, ActivityIndexing, Bounds, CellGrid,
        ExpansionDirection, SchemaClosure, StructuralSchema, TraceVariants, DEFAULT_THETA,
    };
    use crate::core::chrono::DateTime;
    use crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL;

    /// Three items, each in its own package, picked then packed then shipped.
    ///
    /// `items` order `pick item < create package`, `packages` order
    /// `create package < ship package`, and neither spans the outer pair, so covering it
    /// needs the chain. Small enough that every layer is enumerable by hand.
    fn fixture() -> SlimLinkedOCEL {
        let mut ocel = SlimLinkedOCEL::new();
        for t in ["items", "packages"] {
            ocel.add_object_type(t, Vec::new());
        }
        for a in ["pick item", "create package", "ship package"] {
            ocel.add_event_type(a, Vec::new());
        }
        let ts = |ms: i64| DateTime::from_timestamp_millis(ms).unwrap().fixed_offset();
        let mut clock = 1_700_000_000_000i64;
        for n in 0..3 {
            let item = ocel
                .add_object("items", Some(format!("i{n}")), Vec::new(), Vec::new())
                .unwrap();
            let package = ocel
                .add_object(
                    "packages",
                    Some(format!("p{n}")),
                    Vec::new(),
                    vec![("packed".into(), item)],
                )
                .unwrap();
            clock += 1000;
            ocel.add_event(
                "pick item",
                ts(clock),
                Some(format!("pick-{n}")),
                Vec::new(),
                vec![("item".into(), item)],
            );
            clock += 1000;
            ocel.add_event(
                "create package",
                ts(clock),
                Some(format!("create-{n}")),
                Vec::new(),
                vec![("item".into(), item), ("package".into(), package)],
            );
            clock += 1000;
            ocel.add_event(
                "ship package",
                ts(clock),
                Some(format!("ship-{n}")),
                Vec::new(),
                vec![("package".into(), package)],
            );
        }
        ocel
    }

    /// Run every strategy under every objective on the fixture.
    fn layers() -> Vec<(Objective, Vec<FlowLayer>)> {
        let ocel = fixture();
        let schema = StructuralSchema::discover(&ocel);
        let closure = SchemaClosure::build(&ocel, &schema);
        let grid = CellGrid::build(&ocel, &schema, &closure);
        let acts = ActivityIndexing::build(&ocel, &grid);
        let bounds = Bounds::build(&ocel, &schema, &acts);
        let (routes, _) = agreed_routes(&schema);
        let max = Saturation::build(
            &ocel,
            &schema,
            &grid,
            &acts,
            &routes,
            &bounds,
            DEFAULT_THETA,
            ExpansionDirection::default(),
        );
        let variants = TraceVariants::build_with(&ocel, &schema, &acts, &max.written);
        let objects = Saturation::objects_per_type(&schema);
        let novel = novel_cells(&novelty(&ocel, &acts, &bounds, &grid.cells, &max));
        let mut allowed = grid.cells.clone();
        allowed.extend(novel.keys().copied());
        let mut target: Vec<Pair> = max.target_pairs().into_iter().collect();
        target.sort_unstable();

        let input = SearchInput {
            max: &max,
            variants: &variants,
            allowed: &allowed,
            recorded: &grid.cells,
            target: &target,
            objects_per_type: &objects,
            rep: &closure.rep,
            types: &schema.types,
            n_activities: grid.activities.len(),
        };
        let lens = LogAbstraction::over(&max.bounds, &max.cells);
        [Objective::Arcs, Objective::Cells, Objective::Participations]
            .into_iter()
            .map(|obj| {
                (
                    obj,
                    vec![
                        greedy_with_objective(&input, &lens, obj),
                        handoff_with_objective(&input, &lens, obj),
                        exact_with_objective(&input, &lens, obj),
                    ],
                )
            })
            .collect()
    }

    /// Whatever is being minimised, no construction beats the enumeration, and the
    /// enumeration still covers everything.
    #[test]
    fn exact_is_never_worse_than_a_construction_and_still_preserves() {
        for (obj, ls) in layers() {
            let exact = ls.iter().find(|l| l.strategy == Strategy::Exact).unwrap();
            assert!(exact.ran, "{obj:?}: the fixture is small enough to enumerate");
            assert!(
                exact.coverage.complete(),
                "{obj:?}: exact must cover its own requirement, got {:?}",
                exact.coverage
            );
            for built in ls.iter().filter(|l| l.strategy != Strategy::Exact) {
                if !built.ran {
                    continue;
                }
                assert!(
                    obj.key(exact) <= obj.key(built),
                    "{obj:?}: {:?} beat the minimum, {:?} against {:?}",
                    built.strategy,
                    obj.key(built),
                    obj.key(exact),
                );
            }
        }
    }

    /// Every layer is preserving, so the three exact answers differ only in which quantity
    /// each minimises.
    #[test]
    fn each_objective_minimises_its_own_quantity() {
        let all = layers();
        let exact_of = |o: Objective| {
            all.iter()
                .find(|(x, _)| *x == o)
                .and_then(|(_, ls)| ls.iter().find(|l| l.strategy == Strategy::Exact))
                .cloned()
                .unwrap()
        };
        let arcs = exact_of(Objective::Arcs);
        let cells = exact_of(Objective::Cells);
        let parts = exact_of(Objective::Participations);
        assert!(arcs.arcs <= cells.arcs && arcs.arcs <= parts.arcs);
        assert!(cells.cells.len() <= arcs.cells.len() && cells.cells.len() <= parts.cells.len());
        assert!(parts.participations <= arcs.participations);
        assert!(parts.participations <= cells.participations);
    }

    /// The plain entry points minimise arcs.
    #[test]
    fn the_default_objective_is_arcs() {
        assert_eq!(
            Objective::Arcs.key(&unrun(Strategy::Greedy, Objective::Arcs, &[])),
            (0, 0, 0)
        );
        for (obj, ls) in layers() {
            for l in ls {
                assert_eq!(l.objective, obj, "a layer records what it was minimising");
            }
        }
    }
}
