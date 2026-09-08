use std::collections::HashMap;

use crate::core::event_data::object_centric::linked_ocel::{
    slim_linked_ocel::{EventIndex, ObjectIndex},
    LinkedOCELAccess, SlimLinkedOCEL,
};

use super::{
    arcs::ActivityIndexing,
    cells::ActivityIndex,
    schema::StructuralSchema,
};

/// One object's earliest and latest timestamp at each activity it takes part in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectBounds {
    /// The object these bounds are of.
    pub object: ObjectIndex,
    /// `(activity, first, last)` in milliseconds, one entry per activity, sorted by
    /// activity.
    pub at: Vec<(ActivityIndex, i64, i64)>,
    /// How many events of each activity the object took part in, sorted by activity.
    ///
    /// The interval alone cannot tell one occurrence from several inside a tick, and the
    /// difference decides how a reconstruction draws the cell: exactly once is a transition,
    /// repeatedly is a loop.
    pub times: Vec<(ActivityIndex, u32)>,
}

impl ObjectBounds {
    /// Widen the object's interval at one activity, or open it.
    fn note(&mut self, a: ActivityIndex, ts: i64) {
        match self.at.binary_search_by_key(&a, |(x, _, _)| *x) {
            Ok(i) => {
                self.at[i].1 = self.at[i].1.min(ts);
                self.at[i].2 = self.at[i].2.max(ts);
            }
            Err(i) => self.at.insert(i, (a, ts, ts)),
        }
        match self.times.binary_search_by_key(&a, |(x, _)| *x) {
            Ok(i) => self.times[i].1 += 1,
            Err(i) => self.times.insert(i, (a, 1)),
        }
    }

    /// How often the object took part in an activity.
    pub fn count(&self, a: ActivityIndex) -> u32 {
        self.times
            .binary_search_by_key(&a, |(x, _)| *x)
            .map_or(0, |i| self.times[i].1)
    }
}

/// Per-object, per-activity earliest and latest timestamps, grouped by object type.
///
/// Every eventually-follows question factors through this: `x` precedes `y` for one object
/// exactly when its earliest `x` is strictly before its latest `y`
/// ([`precedes`](super::precedes)), so an object's whole trace at an activity reduces to
/// two numbers. Milliseconds, the resolution a tie is judged at.
///
/// None of it depends on the keep-set, so it is built once and shared by every search.
///
/// Directly-follows cannot use this: an object visiting `a, b, a` draws `a -> b` and
/// `b -> a`, and no pair of bounds recovers either. Use
/// [`TraceVariants`](super::TraceVariants) for directly-follows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bounds {
    /// Indexed by [`ObjectTypeIndex`](super::ObjectTypeIndex): the objects of that type, each with its bounds.
    pub per_type: Vec<Vec<ObjectBounds>>,
}

impl Bounds {
    /// Read the bounds off the log, once.
    pub fn build(
        locel: &SlimLinkedOCEL,
        schema: &StructuralSchema,
        acts: &ActivityIndexing,
    ) -> Self {
        let mut per_type: Vec<Vec<ObjectBounds>> = vec![Vec::new(); schema.types.len()];
        for o in locel.get_all_obs() {
            let t = schema.type_of[&o];
            let mut ob = ObjectBounds {
                object: o,
                at: Vec::new(),
                times: Vec::new(),
            };
            for e in o.get_e2o_rev(locel) {
                let a = acts.act_of[e.get_ev(locel).event_type];
                ob.note(a, e.get_time(locel).timestamp_millis());
            }
            per_type[t].push(ob);
        }
        Self { per_type }
    }

    /// The same bounds with a set of participations the log does not record folded in.
    pub fn plus(
        &self,
        locel: &SlimLinkedOCEL,
        acts: &ActivityIndexing,
        extra: &[(EventIndex, ObjectIndex)],
    ) -> Self {
        let mut out = self.clone();
        if extra.is_empty() {
            return out;
        }
        let mut slot: HashMap<ObjectIndex, (usize, usize)> = HashMap::new();
        for (t, objs) in out.per_type.iter().enumerate() {
            for (i, ob) in objs.iter().enumerate() {
                slot.insert(ob.object, (t, i));
            }
        }
        for (e, o) in extra {
            let Some((t, i)) = slot.get(o) else { continue };
            let a = acts.act_of[e.get_ev(locel).event_type];
            out.per_type[*t][*i].note(a, e.get_time(locel).timestamp_millis());
        }
        out
    }
}
