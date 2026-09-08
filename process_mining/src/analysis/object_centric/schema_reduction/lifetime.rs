use std::collections::HashMap;

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::ObjectIndex, LinkedOCELAccess, SlimLinkedOCEL,
};

/// Each object's first and last recorded timestamp.
///
/// The input to expansion's admissibility rule. An object the extraction under-recorded
/// gets a shorter interval and therefore admits less expansion, so the rule fails
/// conservative.
///
/// Read off the recorded log only. Folding written tuples back in would let expansion widen
/// the interval that admits it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Lifetimes {
    /// `(first, last)` in milliseconds, for every object with at least one participation.
    pub per_object: HashMap<ObjectIndex, (i64, i64)>,
}

impl Lifetimes {
    /// Read the lifetimes off the log, once.
    pub fn build(locel: &SlimLinkedOCEL) -> Self {
        let mut per_object: HashMap<ObjectIndex, (i64, i64)> = HashMap::new();
        for e in locel.get_all_evs() {
            let ts = e.get_time(locel).timestamp_millis();
            for o in e.get_e2o(locel) {
                per_object
                    .entry(*o)
                    .and_modify(|(lo, hi)| {
                        *lo = (*lo).min(ts);
                        *hi = (*hi).max(ts);
                    })
                    .or_insert((ts, ts));
            }
        }
        Self { per_object }
    }

    /// Was the object recorded as alive at this instant?
    ///
    /// Both bounds are inclusive. This is the one check that does not go through
    /// [`precedes`](super::precedes): containment is a closed interval and accepts
    /// simultaneity, where an ordering fact refuses it.
    ///
    /// An object with no recorded participation has no interval and is inside nothing.
    pub fn contains(&self, object: ObjectIndex, ts: i64) -> bool {
        self.per_object
            .get(&object)
            .is_some_and(|(lo, hi)| *lo <= ts && ts <= *hi)
    }
}
