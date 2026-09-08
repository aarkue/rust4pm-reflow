//! Implements optimal alignment search
use std::cell::RefCell;

use macros_process_mining::register_binding;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    conformance::alignments::{
        cost::CostFunction,
        petri_net::{AlignmentContext, AlignmentError},
        sync_prod_net::{ModelNet, SyncProductNet},
    },
    core::{
        event_data::case_centric::utils::activity_projection::EventLogActivityProjection,
        process_models::petri_net::TransitionID,
    },
    utils::dijkstra_search::{SearchError, SearchLimits},
    PetriNet,
};

pub mod cost;
pub mod petri_net;
pub(crate) mod reachability;
pub mod sync_prod_net;

/// A single alignment step
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum AlignmentMove {
    /// Synchronous move (model and log agree)
    SyncMove {
        /// The transition that was fired
        transition: TransitionID,
        /// Index of the event in the trace
        trace_event_index: usize,
    },
    /// Model move (only the model moves,)
    ModelMove {
        /// The transition that was fired
        transition: TransitionID,
    },
    /// Log move (only the log moves)
    LogMove {
        /// Index of the event in the trace
        trace_event_index: usize,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
/// Alignment Result
pub struct AlignmentResult {
    /// The sequence of alignment moves
    pub moves: Vec<AlignmentMove>,
    /// Total cost of the alignment
    pub cost: u32,
    /// Number of states visited during search
    pub states_visited: usize,
}
/// Alignment result for a single trace variant
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VariantAlignmentResult {
    /// The variant's activity sequence
    pub activities: Vec<String>,
    /// How many traces follow this variant
    pub frequency: u64,
    /// The alignment result or error for this variant
    pub result: Result<AlignmentResult, AlignmentError>,
}

/// Options for computing alignment
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AlignmentOptions {
    /// Cost function for alignment moves
    pub cost_fn: cost::CostFunction,
    /// Maximum number of states to visit before aborting (per trace).
    /// `None` means no limit.
    /// 
    /// Also see [`AlignmentOptions::max_states_queued`], which bounds the search based on both visited and pending states. 
    /// 
    /// This budget is shared for both ends of the bidirectional search.
    pub max_states: Option<usize>,
    /// Maximum number of states to hold at once before aborting (per trace).
    /// `None` means no limit.
    ///
    /// Also see [`AlignmentOptions::max_states`] which bounds only the visited states.
    /// 
    /// `max_states_queued` is better for bounding memory usage, as it counts both visited and pending states.
    /// Use [`states_in_memory`] to pick a value for a memory budget.
    ///
    pub max_states_queued: Option<usize>,
}

/// How many alignment states fit in `bytes`, for a net with `num_places` places.
///
/// Traces are aligned in parallel, so divide a whole-machine budget by the thread count first.
///
/// Pessimistic: it assumes every place is marked at once, while real markings mark a small
/// fraction of them, so treat the result as a floor rather than a prediction.
pub const fn states_in_memory(bytes: usize, num_places: usize) -> usize {
    bytes / petri_net::bytes_per_state(num_places)
}
impl Default for AlignmentOptions {
    fn default() -> Self {
        Self {
            cost_fn: CostFunction::standard(),
            max_states: None,
            max_states_queued: Some(10_000_000),
        }
    }
}
impl AlignmentOptions {
    fn limits(&self) -> SearchLimits {
        SearchLimits {
            max_states: self.max_states,
            max_states_queued: self.max_states_queued,
        }
    }
}
/// Compute alignments for all variants of an event log.
///
/// Permits at most 255 ([`petri_net::TokenCount::MAX`]) tokens in each place.
pub fn align_log<'a>(
    net: &PetriNet,
    log: impl Into<&'a EventLogActivityProjection>,
    options: &AlignmentOptions,
) -> Vec<VariantAlignmentResult> {
    let projection: &EventLogActivityProjection = log.into();
    align_variants(net, projection, options)
}

/// Compute alignments for all variants from a pre-computed activity projection.
///
/// Permits at most 255 ([`petri_net::TokenCount::MAX`]) tokens in each place.
#[register_binding]
pub fn align_variants(
    net: &PetriNet,
    projection: &EventLogActivityProjection,
    #[bind(default)] options: &AlignmentOptions,
) -> Vec<VariantAlignmentResult> {
    // Both the model half and its reachability are properties of the net, shared by every trace
    let model = build_model(net, &options.cost_fn);
    projection
        .traces
        .par_iter()
        .map(|(trace_indices, count)| {
            thread_local! {
                static CTX: RefCell<AlignmentContext> = RefCell::new(AlignmentContext::default());
            }
            let activities: Vec<String> = trace_indices
                .iter()
                .map(|&idx| projection.activities[idx].clone())
                .collect();
            let result = match &model {
                Err(e) => Err(e.clone()),
                Ok(model) => {
                    let trace: Vec<&str> = activities.iter().map(String::as_str).collect();
                    let sp = SyncProductNet::construct(model, &trace, &options.cost_fn);
                    CTX.with(|ctx| petri_net::align(&sp, &mut ctx.borrow_mut(), options.limits()))
                }
            };
            VariantAlignmentResult {
                activities,
                frequency: *count,
                result,
            }
        })
        .collect()
}

/// Build the trace-independent half, rejecting nets that cannot reach their final marking
fn build_model(net: &PetriNet, cost_fn: &CostFunction) -> Result<ModelNet, AlignmentError> {
    let model = ModelNet::build(net, cost_fn)?;
    if !reachability::final_marking_reachable(&model) {
        return Err(SearchError::Unreachable.into());
    }
    Ok(model)
}

/// Compute alignment for a single trace (given as activity sequence).
///
/// Permits at most 255 ([`petri_net::TokenCount::MAX`]) tokens in each place.
pub fn align_trace(
    net: &PetriNet,
    trace: &[&str],
    options: &AlignmentOptions,
) -> Result<AlignmentResult, AlignmentError> {
    let model = build_model(net, &options.cost_fn)?;
    let sp = SyncProductNet::construct(&model, trace, &options.cost_fn);
    petri_net::align(&sp, &mut AlignmentContext::default(), options.limits())
}

/// Compute alignment for a single trace (given as activity sequence).
///
/// Permits at most 255 ([`petri_net::TokenCount::MAX`]) tokens in each place.
#[allow(dead_code)]
#[register_binding(stringify_error, name = "align_trace")]
fn align_trace_binding(
    net: &PetriNet,
    trace: &[String],
    #[bind(default)] options: &AlignmentOptions,
) -> Result<AlignmentResult, AlignmentError> {
    let trace_as_str: Vec<_> = trace.iter().map(|s| s.as_str()).collect();
    align_trace(net, &trace_as_str, options)
}

/// Align the empty trace to the given model
/// with the specified options
///
/// Permits at most 255 ([`petri_net::TokenCount::MAX`]) tokens in each place.
#[register_binding(stringify_error)]
pub fn align_empty_trace(
    net: &PetriNet,
    #[bind(default)] options: &AlignmentOptions,
) -> Result<AlignmentResult, AlignmentError> {
    align_trace(net, &[], options)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
/// Alignment Fitness Result
pub struct FitnessResult {
    /// Log fitness, as the total computed fitness (summing up the costs for all traces)
    pub log_fitness: f64,
    /// Average trace fitness (across all traces)
    pub average_fitness: f64,
    /// Fraction of traces that perfectly fit (i.e., have an alignment cost of `0`)
    pub perfectly_fitting_frac: f64,
    /// The total cost, summed up from all traces
    pub total_costs: u64,
}

/// Compute fitness stats from alignment results
///
/// Also constructs the empty-trace alignment (shortest path through model)
#[register_binding(stringify_error)]
pub fn compute_fitness(
    align_res: &[VariantAlignmentResult],
    net: &PetriNet,
    #[bind(default)] options: &AlignmentOptions,
) -> Result<FitnessResult, AlignmentError> {
    let empty = align_empty_trace(net, options)?;
    let model_path_min = empty.cost;
    let mut num_perfectly_fitting = 0;
    let mut total_costs = 0;
    let mut fitness_sum_for_avg = 0f64;
    let mut num_traces = 0;
    let mut num_events = 0;
    for variant in align_res {
        let res = variant.result.as_ref().map_err(|e| e.clone())?;
        let costs = res.cost;
        if costs == 0 {
            num_perfectly_fitting += variant.frequency;
        }
        total_costs += variant.frequency * costs as u64;
        num_traces += variant.frequency;
        num_events += variant.frequency * variant.activities.len() as u64;
        let denom = variant.activities.len() as f64 * options.cost_fn.log_move_cost as f64
            + model_path_min as f64;
        // denom == 0 means an empty trace against a net with initial == final marking: Perfectly fitting
        let fitness = if denom == 0.0 {
            1f64
        } else {
            1f64 - (costs as f64 / denom)
        };
        fitness_sum_for_avg += variant.frequency as f64 * fitness;
    }
    let log_denom = num_events as f64 * options.cost_fn.log_move_cost as f64
        + num_traces as f64 * model_path_min as f64;
    let log_fitness = if log_denom == 0.0 {
        1f64
    } else {
        1f64 - (total_costs as f64 / log_denom)
    };
    Ok(FitnessResult {
        log_fitness,
        average_fitness: if num_traces == 0 {
            // Could be either way..
            0f64
        } else {
            fitness_sum_for_avg / num_traces as f64
        },
        perfectly_fitting_frac: if num_traces == 0 {
            0f64
        } else {
            num_perfectly_fitting as f64 / num_traces as f64
        },
        total_costs,
    })
}

#[cfg(test)]
mod test {
    use std::{collections::HashSet, time::Instant};

    use crate::{
        conformance::alignments::{
            align_empty_trace, align_log, align_trace, compute_fitness,
            cost::CostFunction,
            petri_net::AlignmentError,
            sync_prod_net::{ModelNet, SyncProdNetConstructionError},
            AlignmentOptions,
        },
        core::{
            event_data::case_centric::utils::activity_projection::log_to_activity_projection,
            process_models::petri_net::{Arc, ArcType, PlaceID},
        },
        test_utils::get_test_data_path,
        utils::dijkstra_search::SearchError,
        EventLog, Importable, PetriNet,
    };

    fn align_helper(
        log_name: &str,
        net_name: &str,
    ) -> (
        Vec<super::VariantAlignmentResult>,
        Result<super::FitnessResult, AlignmentError>,
    ) {
        let test_path = get_test_data_path();
        let log = EventLog::import_from_path(test_path.join("xes").join(log_name)).unwrap();
        let net = PetriNet::import_pnml(test_path.join("petri-net").join(net_name)).unwrap();
        let act_proj = log_to_activity_projection(&log);
        let options = AlignmentOptions::default();
        let now = Instant::now();
        let result = align_log(&net, &act_proj, &options);
        println!("Aligning traces took {:?}", now.elapsed());
        let fitness = compute_fitness(&result, &net, &options);
        println!("{fitness:?}");
        (result, fitness)
    }

    #[test]
    fn sepsis_total_cost() {
        let (_alignment, fitness) =
            align_helper("Sepsis Cases - Event Log.xes.gz", "sepsis-DISCovered.apnml");
        let fitness = fitness.unwrap();
        // Ground truth total alignment cost was computed and additionally verified with external source (PM4Py)
        assert_eq!(fitness.total_costs, 4118);
    }

    #[test]
    fn rtfm_total_cost() {
        let (_alignment, fitness) = align_helper(
            "Road_Traffic_Fine_Management_Process.xes.gz",
            "rtfm-imf-02.apnml",
        );
        let fitness = fitness.unwrap();
        // Ground truth total alignment cost was computed and additionally verified with external source (PM4Py)
        assert_eq!(fitness.total_costs, 17650);
    }

    #[test]
    fn no_initial_marking_err() {
        let test_path = get_test_data_path();
        let mut net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        net.initial_marking = None;
        let sn = ModelNet::build(&net, &CostFunction::standard());
        assert_eq!(sn, Err(SyncProdNetConstructionError::NoInitialMarking));
    }
    #[test]
    fn no_final_markings_err() {
        let test_path = get_test_data_path();
        let mut net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        net.final_markings = None;
        let sn = ModelNet::build(&net, &CostFunction::standard());
        assert_eq!(sn, Err(SyncProdNetConstructionError::NoFinalMarking));
    }
    #[test]
    fn unknown_place_in_initial_marking_err() {
        let test_path = get_test_data_path();
        let mut net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        let new_id = PlaceID(uuid::Uuid::new_v4());
        net.initial_marking
            .as_mut()
            .expect("exists in apnml")
            .insert(new_id, 1);
        let sn = ModelNet::build(&net, &CostFunction::standard());
        assert_eq!(
            sn,
            Err(SyncProdNetConstructionError::InvalidPlaceInMarking(new_id))
        );
    }
    #[test]
    fn unknown_place_in_final_marking_err() {
        let test_path = get_test_data_path();
        let mut net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        let new_id = PlaceID(uuid::Uuid::new_v4());
        net.final_markings
            .as_mut()
            .expect("exists in apnml")
            .first_mut()
            .expect("one final marking exists")
            .insert(new_id, 1);
        let sn = ModelNet::build(&net, &CostFunction::standard());
        assert_eq!(
            sn,
            Err(SyncProdNetConstructionError::InvalidPlaceInMarking(new_id))
        );
    }

    #[test]
    fn parallel_arcs_weights_add_up() {
        // Two arcs p0 -> t, so firing `t` needs two tokens, not one
        let mut net = PetriNet::new();
        let p0 = net.add_place(None);
        let p1 = net.add_place(None);
        let t = net.add_transition(Some("a".to_string()), None);
        net.add_arc(ArcType::PlaceTransition(p0.0, t.0), None);
        net.add_arc(ArcType::PlaceTransition(p0.0, t.0), None);
        net.add_arc(ArcType::TransitionPlace(t.0, p1.0), None);
        net.final_markings = Some(vec![[(p1, 1)].into_iter().collect()]);
        let options = AlignmentOptions::default();

        net.initial_marking = Some([(p0, 1)].into_iter().collect());
        assert_eq!(
            align_trace(&net, &["a"], &options),
            Err(AlignmentError::SearchError(SearchError::Unreachable))
        );
        net.initial_marking = Some([(p0, 2)].into_iter().collect());
        assert_eq!(align_trace(&net, &["a"], &options).unwrap().cost, 0);

        // The summed weight must still fit a TokenCount
        net.arcs.push(Arc {
            from_to: ArcType::PlaceTransition(p0.0, t.0),
            weight: 300,
        });
        assert_eq!(
            ModelNet::build(&net, &CostFunction::standard()),
            Err(SyncProdNetConstructionError::ArcWeightTooLarge(302))
        );
    }

    #[test]
    fn oversized_marking_err() {
        let test_path = get_test_data_path();
        let mut net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        let place = *net.initial_marking.as_ref().unwrap().keys().next().unwrap();
        net.initial_marking.as_mut().unwrap().insert(place, 400);
        assert_eq!(
            ModelNet::build(&net, &CostFunction::standard()),
            Err(SyncProdNetConstructionError::MarkingTooLarge(place, 400))
        );
    }

    /// The state equation is solvable, but nothing can ever fire, so the search must decide
    #[test]
    fn spurious_state_equation_solution() {
        let mut net = PetriNet::new();
        let p: Vec<_> = (0..4).map(|_| net.add_place(None)).collect();
        // t1: p1 + p3 -> p2   (needs a token in p3, which only t2 produces)
        let t1 = net.add_transition(Some("a".to_string()), None);
        net.add_arc(ArcType::PlaceTransition(p[0].0, t1.0), None);
        net.add_arc(ArcType::PlaceTransition(p[2].0, t1.0), None);
        net.add_arc(ArcType::TransitionPlace(t1.0, p[1].0), None);
        // t2: p2 -> p3 + p4
        let t2 = net.add_transition(Some("b".to_string()), None);
        net.add_arc(ArcType::PlaceTransition(p[1].0, t2.0), None);
        net.add_arc(ArcType::TransitionPlace(t2.0, p[2].0), None);
        net.add_arc(ArcType::TransitionPlace(t2.0, p[3].0), None);
        net.initial_marking = Some([(p[0], 1)].into_iter().collect());
        net.final_markings = Some(vec![[(p[3], 1)].into_iter().collect()]);

        let model = ModelNet::build(&net, &CostFunction::standard()).unwrap();
        // x = (1,1) solves initial + C x = final, so the cheap check cannot rule it out
        assert!(
            super::reachability::final_marking_reachable(&model),
            "expected the state equation to admit a spurious solution"
        );
        // ... but nothing is ever enabled, so the search must still report Unreachable
        let now = std::time::Instant::now();
        let res = align_trace(&net, &["a"], &AlignmentOptions::default());
        println!("spurious net -> {res:?} in {:?}", now.elapsed());
        assert_eq!(
            res,
            Err(AlignmentError::SearchError(SearchError::Unreachable))
        );
    }

    /// A transition consuming a place past the 64th is not covered by the place word, so it must
    /// still be checked against the marking rather than taken as enabled.
    ///
    /// Place indices come from a `HashMap`, so which place lands past the word varies per run;
    /// repeating makes it near-certain that at least one attempt puts it there.
    #[test]
    fn place_past_the_mask_word_still_blocks() {
        for _ in 0..20 {
            let mut net = PetriNet::new();
            let places: Vec<_> = (0..200).map(|_| net.add_place(None)).collect();
            // "a" carries the token from start to end
            let a = net.add_transition(Some("a".to_string()), None);
            net.add_arc(ArcType::PlaceTransition(places[0].0, a.0), None);
            net.add_arc(ArcType::TransitionPlace(a.0, places[1].0), None);
            // "b" needs a token in a place nothing ever marks, so it can never fire
            let b = net.add_transition(Some("b".to_string()), None);
            net.add_arc(ArcType::PlaceTransition(places[2].0, b.0), None);
            net.add_arc(ArcType::TransitionPlace(b.0, places[1].0), None);
            net.initial_marking = Some([(places[0], 1)].into_iter().collect());
            net.final_markings = Some(vec![[(places[1], 1)].into_iter().collect()]);

            let options = AlignmentOptions::default();
            assert_eq!(align_trace(&net, &["a"], &options).unwrap().cost, 0);
            assert_eq!(
                align_trace(&net, &["b"], &options).unwrap().cost,
                2,
                "\"b\" cannot fire, so only a log move plus a model move aligns"
            );
        }
    }

    /// A zero-weight arc moves no token, so it must neither block its transition nor fire
    #[test]
    fn zero_weight_arcs_are_ignored() {
        let mut net = PetriNet::new();
        let places: Vec<_> = (0..3).map(|_| net.add_place(None)).collect();
        let a = net.add_transition(Some("a".to_string()), None);
        net.add_arc(ArcType::PlaceTransition(places[0].0, a.0), None);
        // Takes nothing from a place that never holds a token, so it must not stop "a" firing
        net.add_arc(ArcType::PlaceTransition(places[2].0, a.0), Some(0));
        net.add_arc(ArcType::TransitionPlace(a.0, places[1].0), None);
        net.initial_marking = Some([(places[0], 1)].into_iter().collect());
        net.final_markings = Some(vec![[(places[1], 1)].into_iter().collect()]);
        assert_eq!(
            align_trace(&net, &["a"], &AlignmentOptions::default())
                .unwrap()
                .cost,
            0
        );
    }

    #[test]
    fn final_marking_unreachable_err() {
        let test_path = get_test_data_path();
        let log = EventLog::import_from_path(
            test_path
                .join("xes")
                .join("Sepsis Cases - Event Log.xes.gz"),
        )
        .unwrap();
        let mut net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        let places_in_final_marking: HashSet<_> = net
            .final_markings
            .as_mut()
            .expect("exists in file")
            .first_mut()
            .expect("not empty")
            .keys()
            .map(|id| id.0)
            .collect();
        net.arcs.retain(|arc| match arc.from_to {
            ArcType::PlaceTransition(_, _) => true,
            ArcType::TransitionPlace(_, place) => !places_in_final_marking.contains(&place),
        });
        let act_proj = log_to_activity_projection(&log);
        let options = AlignmentOptions {
            cost_fn: CostFunction::standard(),
            max_states: None,
            ..AlignmentOptions::default()
        };
        let empty_trace_align = align_empty_trace(&net, &options);
        assert_eq!(
            empty_trace_align,
            Err(AlignmentError::SearchError(SearchError::Unreachable))
        );
        let result = align_log(&net, &act_proj, &options);
        for variant in result {
            assert_eq!(
                variant.result,
                Err(AlignmentError::SearchError(SearchError::Unreachable))
            );
        }
    }
    #[test]
    fn max_states_reached_err() {
        let test_path = get_test_data_path();
        let log = EventLog::import_from_path(
            test_path
                .join("xes")
                .join("Sepsis Cases - Event Log.xes.gz"),
        )
        .unwrap();
        let net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        let act_proj = log_to_activity_projection(&log);
        let options = AlignmentOptions {
            cost_fn: CostFunction::standard(),
            max_states: Some(10),
            ..AlignmentOptions::default()
        };
        let empty_trace_align = align_empty_trace(&net, &options);
        assert_eq!(
            empty_trace_align,
            Err(AlignmentError::SearchError(SearchError::LimitReached))
        );
        let result = align_log(&net, &act_proj, &options);
        for variant in result {
            assert_eq!(
                variant.result,
                Err(AlignmentError::SearchError(SearchError::LimitReached))
            );
        }
    }
    #[test]
    fn not_easy_sound_unreachable_err() {
        let test_path = get_test_data_path();
        let log = EventLog::import_from_path(
            test_path
                .join("xes")
                .join("Sepsis Cases - Event Log.xes.gz"),
        )
        .unwrap();
        let net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-fodina.apnml")).unwrap();
        let act_proj = log_to_activity_projection(&log);
        let options = AlignmentOptions::default();
        // The state equation has no solution for this net, so the final marking is unreachable
        let empty_trace_align = align_empty_trace(&net, &options);
        assert_eq!(
            empty_trace_align,
            Err(AlignmentError::SearchError(SearchError::Unreachable))
        );
        let result = align_log(&net, &act_proj, &options);
        for variant in result {
            assert_eq!(
                variant.result,
                Err(AlignmentError::SearchError(SearchError::Unreachable))
            );
        }
    }
}
