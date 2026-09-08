//! Which construct produced each node of a mined tree, tracked alongside mining rather than on
//! [`Node`](crate::core::process_models::process_tree::Node) itself.
//!
//! A tree built with [`origin tracking`](super::inductive_miner_with_origin) comes back paired
//! with an [`Origin`] of the same shape: one entry per operator node, naming the cut or fall
//! through that built it. Leaves need no entry, since only operator nodes decide anything a
//! reader of the tree would attribute.

/// The construct that produced one operator node of a mined tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Construct {
    /// [`exclusive_choice_cut`](super::cut_finder::exclusive_choice_cut).
    ExclusiveChoiceCut,
    /// [`sequence_cut`](super::cut_finder::sequence_cut) / [`strict_sequence_cut`](super::cut_finder::strict_sequence_cut).
    SequenceCut,
    /// [`concurrent_cut`](super::cut_finder::concurrent_cut).
    ConcurrentCut,
    /// [`concurrent_cut`](super::cut_finder::concurrent_cut) reported [`as_inclusive_choice`](super::cut_finder::as_inclusive_choice).
    InclusiveChoiceCut,
    /// [`loop_cut`](super::cut_finder::loop_cut).
    LoopCut,
    /// [`interleaved_cut`](super::cut_finder::interleaved_cut).
    InterleavedCut,
    /// [`empty_traces`](super::fallthrough::empty_traces), the `OptionalEmptyTraces` case.
    EmptyTraces,
    /// [`activity_once_per_trace`](super::fallthrough::activity_once_per_trace).
    ActivityOncePerTrace,
    /// [`activity_concurrent`](super::fallthrough::activity_concurrent).
    ActivityConcurrent,
    /// [`strict_tau_loop`](super::fallthrough::strict_tau_loop).
    StrictTauLoop,
    /// [`tau_loop`](super::fallthrough::tau_loop).
    TauLoop,
    /// [`two_activities_concurrent`](super::fallthrough::two_activities_concurrent).
    TwoActivitiesConcurrent,
    /// The flower model, the fall through of last resort.
    Flower,
}

impl Construct {
    /// A stable, lower-case name for the construct, for reports and JSON that read off it.
    pub fn label(self) -> &'static str {
        match self {
            Construct::ExclusiveChoiceCut => "xor_cut",
            Construct::SequenceCut => "sequence_cut",
            Construct::ConcurrentCut => "concurrent_cut",
            Construct::InclusiveChoiceCut => "inclusive_choice_cut",
            Construct::LoopCut => "loop_cut",
            Construct::InterleavedCut => "interleaved_cut",
            Construct::EmptyTraces => "empty_traces",
            Construct::ActivityOncePerTrace => "activity_once_per_trace",
            Construct::ActivityConcurrent => "activity_concurrent",
            Construct::StrictTauLoop => "strict_tau_loop",
            Construct::TauLoop => "tau_loop",
            Construct::TwoActivitiesConcurrent => "two_activities_concurrent",
            Construct::Flower => "flower",
        }
    }

    /// Whether this construct inserts a `Loop` node, so a separation at or below it is
    /// suppressed the way [`LoopCut`](Construct::LoopCut) suppresses one directly.
    pub fn is_loop_like(self) -> bool {
        matches!(
            self,
            Construct::LoopCut | Construct::StrictTauLoop | Construct::TauLoop | Construct::Flower
        )
    }
}

impl std::fmt::Display for Construct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A node of the origin tree, mirroring the shape of the [`Node`](crate::core::process_models::process_tree::Node)
/// it was mined alongside.
///
/// The pairing is positional: the `n`-th child of an [`Origin::Operator`] is the origin of the
/// `n`-th child of the matching `Node::Operator`. The tree this is paired with is left
/// [unfolded](crate::core::process_models::process_tree::ProcessTree::fold), so that
/// correspondence never has to survive a fold merging nodes an [`Origin`] cannot equally merge
/// without losing which construct decided what -- folding an associative operator's nested
/// occurrences never changes which activity pairs a sequence orders, so a reader after
/// activity pairs loses nothing by working from the unfolded pair instead.
#[derive(Debug, Clone, PartialEq)]
pub enum Origin {
    /// A leaf: a base case, needing no construct of its own.
    Leaf,
    /// An operator node, with the construct that built it and one origin per child.
    Operator {
        /// The cut or fall through that produced this node.
        construct: Construct,
        /// One entry per child of the paired `Node::Operator`, same order.
        children: Vec<Origin>,
    },
}
