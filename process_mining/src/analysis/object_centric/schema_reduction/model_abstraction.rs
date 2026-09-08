//! The model-based behavioural instantiation: what a type's mined process tree asserts over an
//! activity pair. Sequence cuts give orderings, exclusive choice cuts give [`Assertion::Never`],
//! and loop cuts, or sequence cuts inside a loop body, give [`Assertion::Looped`]. The latter
//! two are read only above every concurrent cut.
//!
//! Each type's sublog is mined with `IMf`. `(a, b)` is ordered when the lowest common ancestor
//! of their leaves is a `Sequence` with `a` in an earlier child and no `Loop` at or above it,
//! since a later iteration could put `b` first. Where a `Loop` sits at or above the separator,
//! the topmost such `Loop` decides the pair. [`TypeModel::decisions`] records the deciding node
//! per unordered pair.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::core::event_data::object_centric::linked_ocel::{LinkedOCELAccess, SlimLinkedOCEL};
use crate::core::process_models::process_tree::{LeafLabel, Node, OperatorType};
use crate::discovery::case_centric::inductive_miner::{
    inductive_miner_with_origin,
    log::ActivityLog,
    origin::{Construct, Origin},
    InductiveMinerOptions,
};

use super::{
    abstraction::{requirement, Abstraction, Assertion, ChainedCover, Cover},
    arcs::ActivityIndexing,
    cells::{ActivityIndex, Cell},
    novelty::Pair,
    schema::{ObjectTypeIndex, StructuralSchema},
};

/// `IMf` noise threshold the model-based instantiation mines at.
pub const MODEL_NOISE_THRESHOLD: f64 = 0.2;

/// The node that decided one activity pair's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// The construct that built the deciding node.
    pub construct: Construct,
    /// `Some((a, b))` when the deciding node is a sequence separation with no loop above it;
    /// `None` for every other case (concurrency, exclusive choice, or any separation under a
    /// loop, however it got there).
    pub ordered: Option<(ActivityIndex, ActivityIndex)>,
    /// The deciding node is an exclusive-choice cut with no loop above it: no trace of the
    /// type holds both activities.
    pub never: bool,
    /// The type alternates between the two activities: they sit on the body and redo sides
    /// of a loop cut, or a sequence cut inside a loop body separates them.
    pub looped: bool,
}

/// The model-based analysis of one object type's mined tree.
#[derive(Debug, Clone, Default)]
pub struct TypeModel {
    /// The type this tree was mined for.
    pub object_type: ObjectTypeIndex,
    /// Activities the type's participations record, i.e. its cells.
    pub activities: BTreeSet<ActivityIndex>,
    /// Activity pairs the tree orders.
    pub ordered: HashSet<(ActivityIndex, ActivityIndex)>,
    /// Unordered pairs (smaller index first) an exclusive-choice cut separates, no loop above.
    pub never: HashSet<Pair>,
    /// Unordered pairs (smaller index first) a loop cut separates into body and redo.
    pub looped: HashSet<Pair>,
    /// The mined tree, rendered, for reports.
    pub tree: String,
    /// Leaves the mined tree actually reached. A strict subset of `activities` when `IMf`'s
    /// second, filtered `DFG` pass drops every edge of an activity and cut detection never
    /// projects a sub-log containing it.
    pub tree_activities: BTreeSet<ActivityIndex>,
    /// Every activity pair the tree separated, keyed by the unordered pair, to the node(s) that
    /// decided it. More than one entry only when the same activity labels two leaves, which
    /// this miner's alphabet-partitioning cuts never produce.
    pub decisions: HashMap<(ActivityIndex, ActivityIndex), Vec<Decision>>,
}

impl TypeModel {
    /// Activities the type's participations record but the tree orders nothing about.
    pub fn silent(&self) -> BTreeSet<ActivityIndex> {
        let ordered_acts: BTreeSet<ActivityIndex> =
            self.ordered.iter().flat_map(|&(a, b)| [a, b]).collect();
        self.activities
            .difference(&ordered_acts)
            .copied()
            .collect()
    }

    /// Every assertion the tree makes: orderings, exclusive choices and loop alternations.
    pub fn assertions(&self) -> Vec<Assertion> {
        self.ordered
            .iter()
            .map(|&(x, y)| Assertion::Order(x, y))
            .chain(self.never.iter().map(|&(x, y)| Assertion::Never(x, y)))
            .chain(self.looped.iter().map(|&(x, y)| Assertion::Looped(x, y)))
            .collect()
    }

    /// Activities no assertion of [`assertions`](Self::assertions) touches.
    pub fn neutral(&self) -> BTreeSet<ActivityIndex> {
        let touched: BTreeSet<ActivityIndex> = self
            .assertions()
            .iter()
            .flat_map(|s| s.activities())
            .collect();
        self.activities.difference(&touched).copied().collect()
    }

    /// The deciding constructs recorded for one activity pair, sorted and deduplicated, or
    /// `None` if the tree never separated them.
    pub fn pair_constructs(&self, x: ActivityIndex, y: ActivityIndex) -> Option<Vec<Construct>> {
        let key = if x < y { (x, y) } else { (y, x) };
        self.decisions.get(&key).map(|decisions| {
            let mut constructs: Vec<Construct> = decisions.iter().map(|d| d.construct).collect();
            constructs.sort_by_key(|c| c.label());
            constructs.dedup();
            constructs
        })
    }

    /// [`pair_constructs`](Self::pair_constructs) as label strings. A pair the tree never
    /// separated reads `dropped_by_noise_filter` when an activity is missing from
    /// `tree_activities`, and `no_common_node` otherwise.
    pub fn pair_labels(&self, x: ActivityIndex, y: ActivityIndex) -> Vec<String> {
        match self.pair_constructs(x, y) {
            Some(constructs) => constructs.iter().map(|c| c.label().to_string()).collect(),
            None if !self.tree_activities.contains(&x) || !self.tree_activities.contains(&y) => {
                vec!["dropped_by_noise_filter".to_string()]
            }
            None => vec!["no_common_node".to_string()],
        }
    }
}

/// Mines every object type's sublog with `IMf` and reads its model-based ordering off the tree.
///
/// `recorded` gives each type's activities, i.e. the cells it is analysed over; types with no
/// recorded cell are skipped, since there is nothing to mine or to attribute.
pub fn model_abstraction(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    recorded: &HashSet<Cell>,
    noise_threshold: f64,
) -> Vec<TypeModel> {
    // The same miner and preset OCPN discovery runs, so the abstraction and the drawn model
    // read one tree.
    model_abstraction_with(locel, schema, acts, recorded, InductiveMinerOptions::imf(noise_threshold))
}

/// [`model_abstraction`] with the miner options given.
pub fn model_abstraction_with(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
    recorded: &HashSet<Cell>,
    options: InductiveMinerOptions,
) -> Vec<TypeModel> {
    let name_index: HashMap<&str, ActivityIndex> = acts
        .activities
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let traces = per_type_traces(locel, schema, acts);

    let mut out = Vec::new();
    for t in 0..schema.types.len() {
        let activities: BTreeSet<ActivityIndex> = recorded
            .iter()
            .filter(|(_, tt)| *tt == t)
            .map(|(a, _)| *a)
            .collect();
        if activities.is_empty() {
            continue;
        }
        let variants = traces.get(&t).cloned().unwrap_or_default();
        let activity_log = ActivityLog::new(acts.len(), variants);
        let (node, origin) = inductive_miner_with_origin(&acts.activities, &activity_log, options);

        let mut decisions: HashMap<(ActivityIndex, ActivityIndex), Vec<Decision>> = HashMap::new();
        let tree_activities = walk(&node, &origin, None, false, &name_index, &mut decisions);
        let ordered: HashSet<(ActivityIndex, ActivityIndex)> = decisions
            .values()
            .flatten()
            .filter_map(|d| d.ordered)
            .collect();
        let never: HashSet<Pair> = decisions
            .iter()
            .filter(|(_, ds)| ds.iter().any(|d| d.never))
            .map(|(&k, _)| k)
            .collect();
        let looped: HashSet<Pair> = decisions
            .iter()
            .filter(|(_, ds)| ds.iter().any(|d| d.looped))
            .map(|(&k, _)| k)
            .collect();

        out.push(TypeModel {
            object_type: t,
            activities,
            ordered,
            never,
            looped,
            tree: node.to_string(),
            tree_activities,
            decisions,
        });
    }
    out
}

/// One object's activities, sorted by `(timestamp, event id)` and mapped to their global
/// [`ActivityIndex`], grouped by object type and aggregated into distinct-sequence counts.
///
/// Ties on timestamp are broken by event id, so the order is a function of the log alone.
fn per_type_traces(
    locel: &SlimLinkedOCEL,
    schema: &StructuralSchema,
    acts: &ActivityIndexing,
) -> HashMap<ObjectTypeIndex, HashMap<Vec<ActivityIndex>, u64>> {
    let mut out: HashMap<ObjectTypeIndex, HashMap<Vec<ActivityIndex>, u64>> = HashMap::new();
    for o in locel.get_all_obs() {
        let t = schema.type_of[&o];
        let mut evs: Vec<(i64, &str, ActivityIndex)> = o
            .get_e2o_rev(locel)
            .map(|e| {
                (
                    e.get_time(locel).timestamp_millis(),
                    e.get_ev(locel).id.as_str(),
                    acts.act_of[e.get_ev(locel).event_type],
                )
            })
            .collect();
        evs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
        let trace: Vec<ActivityIndex> = evs.into_iter().map(|(_, _, a)| a).collect();
        *out.entry(t).or_default().entry(trace).or_default() += 1;
    }
    out
}

/// Walks a mined tree in lockstep with its [`Origin`], recording the deciding node of every
/// activity pair it separates and returning the leaves reached below `node`.
///
/// `top_loop` is the topmost `Loop` construct at or above `node`, if any. It governs the
/// recursion into children and, when set on entry, decides `node`'s own cross-child pairs.
fn walk(
    node: &Node,
    origin: &Origin,
    top_loop: Option<Construct>,
    under_parallel: bool,
    name_index: &HashMap<&str, ActivityIndex>,
    decisions: &mut HashMap<(ActivityIndex, ActivityIndex), Vec<Decision>>,
) -> BTreeSet<ActivityIndex> {
    match (node, origin) {
        (Node::Leaf(leaf), Origin::Leaf) => match &leaf.activity_label {
            LeafLabel::Activity(name) => BTreeSet::from([name_index[name.as_str()]]),
            LeafLabel::Tau => BTreeSet::new(),
        },
        (
            Node::Operator(op),
            Origin::Operator {
                construct,
                children: origin_children,
            },
        ) => {
            let is_loop = op.operator_type == OperatorType::Loop;
            let child_top_loop = match (is_loop, top_loop) {
                (true, None) => Some(*construct),
                _ => top_loop,
            };
            // Exclusions and alternations are read only above every concurrent cut: inside a
            // parallel branch they describe one branch only.
            let child_under_parallel = under_parallel
                || !matches!(
                    op.operator_type,
                    OperatorType::Sequence | OperatorType::ExclusiveChoice | OperatorType::Loop
                );
            let own_loop_cut =
                !under_parallel && is_loop && top_loop.is_none() && *construct == Construct::LoopCut;
            // A sequence cut inside a loop body is an alternation: every iteration puts the
            // earlier child before the later one, `(a b)+` for two leaves.
            let sequence_in_loop =
                !under_parallel && top_loop.is_some() && *construct == Construct::SequenceCut;
            let own_xor_cut =
                !under_parallel && top_loop.is_none() && *construct == Construct::ExclusiveChoiceCut;

            let kid_labels: Vec<BTreeSet<ActivityIndex>> = op
                .children
                .iter()
                .zip(origin_children)
                .map(|(child, child_origin)| {
                    walk(child, child_origin, child_top_loop, child_under_parallel, name_index, decisions)
                })
                .collect();

            for i in 0..kid_labels.len() {
                for j in (i + 1)..kid_labels.len() {
                    for &x in &kid_labels[i] {
                        for &y in &kid_labels[j] {
                            if x == y {
                                continue;
                            }
                            let decider = child_top_loop.unwrap_or(*construct);
                            let is_sequence_separation =
                                child_top_loop.is_none() && op.operator_type == OperatorType::Sequence;
                            let key = if x < y { (x, y) } else { (y, x) };
                            decisions.entry(key).or_default().push(Decision {
                                construct: decider,
                                ordered: is_sequence_separation.then_some((x, y)),
                                never: own_xor_cut,
                                looped: own_loop_cut || sequence_in_loop,
                            });
                        }
                    }
                }
            }

            kid_labels.into_iter().flatten().collect()
        }
        _ => unreachable!("a tree and its Origin always share shape"),
    }
}

/// The tree-construct instantiation: every cut of a type's mined tree is an assertion.
///
/// Sequence cuts give orderings, covered by chaining as in [`PairAbstraction`]. Exclusive
/// choice and loop cuts give [`Assertion::Never`] and [`Assertion::Looped`] pairs, covered
/// only where a kept type's own tree shows the same cut over the same pair.
///
/// [`PairAbstraction`]: super::abstraction::PairAbstraction
pub struct TreeAbstraction<'a> {
    by_type: HashMap<ObjectTypeIndex, &'a TypeModel>,
    required: Vec<Assertion>,
}

impl<'a> TreeAbstraction<'a> {
    /// Everything the trees assert over `over`, both cells of each pair kept for its type.
    pub fn over(models: &'a [TypeModel], over: &HashSet<Cell>) -> Self {
        let by_type: HashMap<ObjectTypeIndex, &'a TypeModel> =
            models.iter().map(|m| (m.object_type, m)).collect();
        let required = requirement(models.iter().flat_map(|m| {
            let t = m.object_type;
            m.assertions().into_iter().filter(move |s| {
                let [x, y] = s.activities();
                over.contains(&(x, t)) && over.contains(&(y, t))
            })
        }));
        Self { by_type, required }
    }
}

impl Abstraction for TreeAbstraction<'_> {
    fn asserts(&self, kept: &HashSet<Cell>, t: ObjectTypeIndex) -> HashSet<Assertion> {
        self.by_type
            .get(&t)
            .map(|m| {
                m.assertions()
                    .into_iter()
                    .filter(|s| {
                        let [x, y] = s.activities();
                        kept.contains(&(x, t)) && kept.contains(&(y, t))
                    })
                    .collect()
            })
            .unwrap_or_default()
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
        Box::new(TreeCover {
            chained: ChainedCover::of(asserted, n_activities),
            direct: asserted
                .iter()
                .filter(|s| matches!(s, Assertion::Never(..) | Assertion::Looped(..)))
                .copied()
                .collect(),
        })
    }
}

struct TreeCover {
    chained: ChainedCover,
    direct: HashSet<Assertion>,
}

impl Cover for TreeCover {
    fn holds(&self, s: Assertion) -> bool {
        match s {
            Assertion::Order(..) => self.chained.holds(s),
            Assertion::Never(..) | Assertion::Looped(..) => self.direct.contains(&s),
            _ => false,
        }
    }
}
