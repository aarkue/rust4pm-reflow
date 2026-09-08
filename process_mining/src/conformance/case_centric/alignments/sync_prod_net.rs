//! Synchronous product net used for efficiently computing alignments

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    conformance::alignments::{
        cost::CostFunction,
        petri_net::{PlaceSet, TokenCount, TracePos, PLACE_SET_LIMIT},
        AlignmentMove,
    },
    core::process_models::petri_net::{ArcType, PlaceID, TransitionID},
    PetriNet,
};

/// The places a transition touches, as `(place_index, weight)` pairs
pub(crate) type Arcs<'a> = &'a [(usize, TokenCount)];

/// A transition in the synchronous product net
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SyncProdNetTransition {
    /// The move this transition represents (model-transition / trace-event indices)
    pub(crate) move_type: AlignmentMove,
    /// The pre-computed cost of firing this transition
    pub(crate) cost: u32,
    /// Incoming places (`place_index`, weight), i.e., which token to consume
    ///
    /// Only a model transition owns these; reach them through [`SyncProductNet::arcs`].
    pub(crate) inputs: Vec<(usize, TokenCount)>,
    /// Outgoing places (`place_index`, weight), i.e., which token to produce
    ///
    /// Only a model transition owns these; reach them through [`SyncProductNet::arcs`].
    pub(crate) outputs: Vec<(usize, TokenCount)>,
    /// Index of the model transition whose arcs this move fires, if it moves a token at all.
    ///
    /// A model move fires its own, a sync move those of the transition it synchronises with, and
    /// a log move none. Pointing at it spares a sync move copying them afresh for every trace.
    pub(crate) fires_model_transition: Option<u32>,
    /// The places this transition takes tokens from, to reject it without walking its inputs
    pub(crate) input_places: PlaceSet,
    /// The places it puts tokens into, used the same way by a backward search
    pub(crate) output_places: PlaceSet,
    /// Whether having the input places marked is on its own enough to fire this transition
    ///
    /// See [`marked_places_suffice`] for when that holds.
    pub(crate) marked_inputs_suffice: bool,
    /// The same for a backward search, which takes tokens from the output places
    pub(crate) marked_outputs_suffice: bool,
}

/// Whether knowing these places are marked settles on its own that they hold enough tokens.
///
/// It does when each gives up a single token, and when every place fits a [`PlaceSet`]: one left
/// out of the set would go unchecked entirely.
fn marked_places_suffice(places: &[(usize, TokenCount)]) -> bool {
    places
        .iter()
        .all(|(place, weight)| *weight == 1 && *place < PLACE_SET_LIMIT)
}

/// Which transitions a marking has to consider, grouped by the places it marks
///
/// A transition only fires when every place it consumes holds a token, so a marking need only
/// consider what its own marked places trigger, not every transition in the net.
#[derive(Debug, PartialEq)]
pub(crate) struct TriggerIndex {
    /// Where each place's run starts in [`triggered`], with one extra entry for the end
    ///
    /// [`triggered`]: TriggerIndex::triggered
    offsets: Vec<u32>,
    /// Transition indices, grouped by the place that triggers them
    triggered: Vec<u32>,
    /// Transitions that consume nothing, so every marking may fire them
    consuming_nothing: Vec<u32>,
}

impl TriggerIndex {
    /// Group `transitions` by the first place each consumes, which a backward search reads as
    /// its outputs rather than its inputs
    fn build(transitions: &[SyncProdNetTransition], num_places: usize, reverse: bool) -> Self {
        let trigger_of = |trans: &SyncProdNetTransition| {
            let consumed = if reverse {
                &trans.outputs
            } else {
                &trans.inputs
            };
            consumed.iter().map(|(place, _)| *place).min()
        };
        let mut offsets = vec![0u32; num_places + 1];
        let mut consuming_nothing = Vec::new();
        for (index, trans) in transitions.iter().enumerate() {
            match trigger_of(trans) {
                Some(place) => offsets[place + 1] += 1,
                None => consuming_nothing.push(index as u32),
            }
        }
        for place in 1..offsets.len() {
            offsets[place] += offsets[place - 1];
        }
        let mut cursor = offsets.clone();
        let mut triggered = vec![0u32; offsets[num_places] as usize];
        for (index, trans) in transitions.iter().enumerate() {
            if let Some(place) = trigger_of(trans) {
                triggered[cursor[place] as usize] = index as u32;
                cursor[place] += 1;
            }
        }
        Self {
            offsets,
            triggered,
            consuming_nothing,
        }
    }

    /// The transitions a token in `place` can trigger
    #[inline]
    pub(crate) fn triggered_by(&self, place: usize) -> &[u32] {
        &self.triggered[self.offsets[place] as usize..self.offsets[place + 1] as usize]
    }

    /// The transitions any marking may fire, as they consume nothing
    #[inline]
    pub(crate) fn consuming_nothing(&self) -> &[u32] {
        &self.consuming_nothing
    }
}

/// The places a transition touches
fn places_of(places: &[(usize, TokenCount)]) -> PlaceSet {
    let mut set = PlaceSet::default();
    for (place, _) in places {
        set.insert(*place);
    }
    set
}

impl SyncProdNetTransition {
    /// Whether this consumes a trace event without firing anything in the model
    #[inline]
    pub(crate) fn is_log_move(&self) -> bool {
        self.fires_model_transition.is_none()
    }
}

/// The trace-independent half of the synchronous product, built once per Petri net
#[derive(Debug, PartialEq)]
pub(crate) struct ModelNet {
    /// Number of model places
    pub(crate) num_places: usize,
    /// The model transitions (they keep these indices in the synchronous product)
    pub(crate) transitions: Vec<SyncProdNetTransition>,
    /// Model transitions per activity label, i.e. those a trace event can synchronize with
    by_label: HashMap<String, Vec<usize>>,
    /// Initial marking (tokens per place)
    pub(crate) initial_marking: Vec<TokenCount>,
    /// Final marking (tokens per place)
    pub(crate) final_marking: Vec<TokenCount>,
    /// Largest cost over the model transitions
    max_cost: u32,
    /// Whether any transition needs its token counts checked, so markings must be decoded
    pub(crate) needs_token_counts: bool,
    /// Transitions grouped by the place a forward search needs marked to fire them
    pub(crate) forward_triggers: TriggerIndex,
    /// The same for a backward search, which consumes a transition's outputs
    pub(crate) backward_triggers: TriggerIndex,
}

/// The synchronous product of a Petri net and a trace.
///
/// Only model places exist; the trace position is tracked separately in the search.
/// Transitions are indexed as `[model moves .., log and sync moves ..]`.
#[derive(Debug, PartialEq)]
pub(crate) struct SyncProductNet<'a> {
    /// The shared, trace-independent half
    pub(crate) model: &'a ModelNet,
    /// Length of the trace
    pub(crate) trace_length: TracePos,
    /// The log and sync moves
    trace_transitions: Vec<SyncProdNetTransition>,
    /// Where each trace position's run starts in [`trace_moves`], with one extra entry for the end
    ///
    /// [`trace_moves`]: SyncProductNet::trace_moves
    trace_move_offsets: Vec<u32>,
    /// Log and sync move indices, grouped by the trace position they belong to
    trace_moves: Vec<u32>,
    /// Largest cost over all transitions (precomputed for the search's bucket sizing)
    pub(crate) max_edge_cost: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
/// Error when constructing the sync product net
pub enum SyncProdNetConstructionError {
    /// A unknown place id was referenced in a marking
    InvalidPlaceInMarking(PlaceID),
    /// No final marking found
    NoFinalMarking,
    /// No initial marking found
    NoInitialMarking,
    /// An arc weight, or the sum over parallel arcs, exceeds [`TokenCount::MAX`]
    ///
    /// [`TokenCount::MAX`]: super::petri_net::TokenCount
    ArcWeightTooLarge(u32),
    /// A marking puts more than `TokenCount::MAX` tokens in a place
    MarkingTooLarge(PlaceID, u64),
}
impl std::fmt::Display for SyncProdNetConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SyncProdNetConstructionError {}

impl ModelNet {
    /// Build the trace-independent half of the synchronous product from a Petri net
    pub(crate) fn build(
        net: &PetriNet,
        cost_fn: &CostFunction,
    ) -> Result<Self, SyncProdNetConstructionError> {
        // Places and transitions are indexed in sorted id order rather than hash order.
        //
        // `HashMap` seeds its hasher per process, so iterating the net directly hands each
        // run a different index assignment. That permutes the successor order in the sync
        // product, and a trace with several equally optimal alignments then yields a
        // different one of them on every run. The alignment *cost* is unaffected, which is
        // why fitness stayed stable and hid this; escaping-edges precision depends on which
        // states the returned alignment visits, and swung by up to .25 on one Container
        // Logistics type across runs of identical input.
        let mut place_ids: Vec<&Uuid> = net.places.keys().collect();
        place_ids.sort_unstable();
        let place_index: HashMap<&Uuid, usize> = place_ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        let mut sorted_transitions: Vec<_> = net.transitions.iter().collect();
        sorted_transitions.sort_unstable_by_key(|(id, _)| *id);
        let trans_index: HashMap<&Uuid, usize> = sorted_transitions
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i))
            .collect();

        // Parallel arcs collapse into one entry per place, summed as `u32`
        let mut inputs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); net.transitions.len()];
        let mut outputs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); net.transitions.len()];
        fn add_weight(arcs: &mut Vec<(usize, u32)>, place: usize, weight: u32) {
            match arcs.iter_mut().find(|(p, _)| *p == place) {
                Some((_, total)) => *total = total.saturating_add(weight),
                None => arcs.push((place, weight)),
            }
        }
        for arc in &net.arcs {
            match arc.from_to {
                ArcType::PlaceTransition(place, trans) => {
                    if let (Some(place), Some(trans)) =
                        (place_index.get(&place), trans_index.get(&trans))
                    {
                        add_weight(&mut inputs[*trans], *place, arc.weight);
                    }
                }
                ArcType::TransitionPlace(trans, place) => {
                    if let (Some(place), Some(trans)) =
                        (place_index.get(&place), trans_index.get(&trans))
                    {
                        add_weight(&mut outputs[*trans], *place, arc.weight);
                    }
                }
            }
        }
        // A place holds at most `TokenCount::MAX` tokens, so reject (rather than truncate) larger ones
        fn to_token_count(
            arcs: Vec<(usize, u32)>,
        ) -> Result<Vec<(usize, TokenCount)>, SyncProdNetConstructionError> {
            arcs.into_iter()
                // An arc moving no token is the same as no arc, and keeping it would both make
                // the place look required and leave nothing for firing to change
                .filter(|(_, weight)| *weight != 0)
                .map(|(place, weight)| {
                    let weight = TokenCount::try_from(weight)
                        .map_err(|_| SyncProdNetConstructionError::ArcWeightTooLarge(weight))?;
                    Ok((place, weight))
                })
                .collect()
        }

        let mut by_label: HashMap<String, Vec<usize>> = HashMap::new();
        let mut transitions = Vec::with_capacity(net.transitions.len());
        // Same sorted order as `trans_index`, so the zipped input/output vectors line up.
        for (&(trans_id, trans), (inputs, outputs)) in sorted_transitions
            .iter()
            .zip(inputs.into_iter().zip(outputs))
        {
            let index = transitions.len();
            if let Some(label) = &trans.label {
                by_label.entry(label.clone()).or_default().push(index);
            }
            let (inputs, outputs) = (to_token_count(inputs)?, to_token_count(outputs)?);
            let fires_model_transition = Some(index as u32);
            transitions.push(SyncProdNetTransition {
                move_type: AlignmentMove::ModelMove {
                    transition: TransitionID(*trans_id),
                },
                cost: if trans.label.is_none() {
                    cost_fn.silent_move_cost
                } else {
                    cost_fn.model_move_cost
                },
                input_places: places_of(&inputs),
                output_places: places_of(&outputs),
                marked_inputs_suffice: marked_places_suffice(&inputs),
                marked_outputs_suffice: marked_places_suffice(&outputs),
                inputs,
                outputs,
                fires_model_transition,
            });
        }
        // Get marking as vec of token counts (based on place indices)
        let marking_of = |marking: &HashMap<PlaceID, u64>| {
            let mut tokens: Vec<TokenCount> = vec![0; net.places.len()];
            for (place_id, count) in marking {
                let index = place_index.get(&place_id.0).ok_or(
                    SyncProdNetConstructionError::InvalidPlaceInMarking(*place_id),
                )?;
                tokens[*index] = TokenCount::try_from(*count).map_err(|_| {
                    SyncProdNetConstructionError::MarkingTooLarge(*place_id, *count)
                })?;
            }
            Ok(tokens)
        };
        let initial_marking = marking_of(
            net.initial_marking
                .as_ref()
                .ok_or(SyncProdNetConstructionError::NoInitialMarking)?,
        )?;
        let final_marking = marking_of(
            net.final_markings
                .as_ref()
                .and_then(|f| f.first())
                .ok_or(SyncProdNetConstructionError::NoFinalMarking)?,
        )?;

        let max_cost = transitions.iter().map(|t| t.cost).max().unwrap_or(1);
        let needs_token_counts = transitions
            .iter()
            .any(|t| !t.marked_inputs_suffice || !t.marked_outputs_suffice);
        let forward_triggers = TriggerIndex::build(&transitions, net.places.len(), false);
        let backward_triggers = TriggerIndex::build(&transitions, net.places.len(), true);
        Ok(Self {
            num_places: net.places.len(),
            transitions,
            by_label,
            initial_marking,
            final_marking,
            max_cost,
            needs_token_counts,
            forward_triggers,
            backward_triggers,
        })
    }
}

impl<'a> SyncProductNet<'a> {
    /// Add the log and sync moves for `trace` to the shared model half
    pub(crate) fn construct(model: &'a ModelNet, trace: &[&str], cost_fn: &CostFunction) -> Self {
        let num_model_trans = model.transitions.len();
        let mut trace_transitions = Vec::with_capacity(trace.len() * 2);
        let mut trace_move_offsets = Vec::with_capacity(trace.len() + 1);
        let mut trace_moves = Vec::with_capacity(trace.len() * 2);
        // There are no log places: the search tracks the trace position, so a log move leaves
        // the marking alone and a sync move changes it like its model transition does.
        for activity in trace {
            trace_move_offsets.push(trace_moves.len() as u32);
            trace_moves.push((num_model_trans + trace_transitions.len()) as u32);
            trace_transitions.push(SyncProdNetTransition {
                move_type: AlignmentMove::LogMove {
                    trace_event_index: trace_move_offsets.len() - 1,
                },
                cost: cost_fn.log_move_cost,
                inputs: Vec::new(),
                outputs: Vec::new(),
                // A log move moves no token at all
                fires_model_transition: None,
                // A log move touches no place, so it is never rejected by the mask
                input_places: PlaceSet::default(),
                output_places: PlaceSet::default(),
                marked_inputs_suffice: true,
                marked_outputs_suffice: true,
            });
            for model_index in model.by_label.get(*activity).into_iter().flatten() {
                let model_trans = &model.transitions[*model_index];
                let AlignmentMove::ModelMove { transition } = model_trans.move_type else {
                    unreachable!("the model half holds model moves only")
                };
                trace_moves.push((num_model_trans + trace_transitions.len()) as u32);
                trace_transitions.push(SyncProdNetTransition {
                    move_type: AlignmentMove::SyncMove {
                        transition,
                        trace_event_index: trace_move_offsets.len() - 1,
                    },
                    cost: cost_fn.sync_move_cost,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    fires_model_transition: Some(*model_index as u32),
                    input_places: model_trans.input_places,
                    output_places: model_trans.output_places,
                    marked_inputs_suffice: model_trans.marked_inputs_suffice,
                    marked_outputs_suffice: model_trans.marked_outputs_suffice,
                });
            }
        }
        let max_edge_cost = trace_transitions
            .iter()
            .map(|t| t.cost)
            .max()
            .unwrap_or(0)
            .max(model.max_cost);
        trace_move_offsets.push(trace_moves.len() as u32);
        Self {
            model,
            trace_length: trace.len() as TracePos,
            trace_transitions,
            trace_move_offsets,
            trace_moves,
            max_edge_cost,
        }
    }

    /// The transition an index addresses, whether it belongs to the model or to the trace
    #[inline]
    pub(crate) fn transition(&self, index: usize) -> &SyncProdNetTransition {
        match index.checked_sub(self.num_model_trans()) {
            Some(trace_index) => &self.trace_transitions[trace_index],
            None => &self.model.transitions[index],
        }
    }

    /// The log and sync moves for the event at `position`
    #[inline]
    pub(crate) fn moves_at(&self, position: usize) -> &[u32] {
        match self.trace_move_offsets.get(position + 1) {
            Some(end) => {
                &self.trace_moves[self.trace_move_offsets[position] as usize..*end as usize]
            }
            None => &[],
        }
    }

    /// The places a transition takes tokens from and the ones it puts them into
    #[inline]
    pub(crate) fn arcs(&self, trans: &SyncProdNetTransition) -> (Arcs<'_>, Arcs<'_>) {
        match trans.fires_model_transition {
            Some(index) => {
                let fired = &self.model.transitions[index as usize];
                (&fired.inputs, &fired.outputs)
            }
            None => (&[], &[]),
        }
    }

    /// Number of model transitions; they occupy the indices `0..n`
    #[inline]
    pub(crate) fn num_model_trans(&self) -> usize {
        self.model.transitions.len()
    }

    /// Whether the transition at `index` is a model one, i.e. it leaves the trace position alone
    #[inline]
    pub(crate) fn is_model_transition(&self, index: usize) -> bool {
        index < self.num_model_trans()
    }
}

#[cfg(test)]
mod test {
    use super::marked_places_suffice;

    /// A [`PlaceSet`] holds only the first 64 places, so anything it leaves out has to fall back
    /// to the exact check rather than be taken as enabled
    ///
    /// [`PlaceSet`]: crate::conformance::alignments::petri_net::PlaceSet
    #[test]
    fn marked_places_only_suffice_for_what_the_word_covers() {
        assert!(marked_places_suffice(&[(0, 1), (63, 1)]));
        assert!(
            !marked_places_suffice(&[(64, 1)]),
            "past the word, so never tested"
        );
        assert!(
            !marked_places_suffice(&[(0, 1), (99, 1)]),
            "one place past the word is enough"
        );
        assert!(
            !marked_places_suffice(&[(0, 2)]),
            "two tokens need the count checked"
        );
    }
}
