use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::core::event_data::object_centric::linked_ocel::{
    LinkedOCELAccess, SlimLinkedOCEL,
};

/// One log reduced to a comparable form: every event and every object, by id, with a hash of
/// everything about it the reduction could have changed.
///
/// The comparison cannot be index-wise: nothing guarantees two logs assign the same index
/// to the same object, and `relationships` is not sorted by qualifier within a duplicated
/// object. Both are normalised here, ids instead of indices and the relationship list
/// sorted by `(qualifier, target id)`.
///
/// Hashes instead of the content itself keep the memory of a round-trip check on a large
/// log bounded. The ids are kept so a mismatch names what differs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// `(event id, hash)`, sorted by id.
    pub events: Vec<(String, u64)>,
    /// `(object id, hash)`, sorted by id.
    pub objects: Vec<(String, u64)>,
}

/// What two fingerprints disagree about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Difference {
    /// Event ids present in one log and not the other.
    pub events_only_in_one: Vec<String>,
    /// Object ids present in one log and not the other.
    pub objects_only_in_one: Vec<String>,
    /// Event ids whose content differs.
    pub events_differing: Vec<String>,
    /// Object ids whose content differs.
    pub objects_differing: Vec<String>,
}

impl Difference {
    /// Whether the two logs are the same log.
    pub fn is_empty(&self) -> bool {
        self.events_only_in_one.is_empty()
            && self.objects_only_in_one.is_empty()
            && self.events_differing.is_empty()
            && self.objects_differing.is_empty()
    }

    /// The counts, one line.
    pub fn summary(&self) -> String {
        format!(
            "{} events differ, {} objects differ, {} events / {} objects on one side only",
            self.events_differing.len(),
            self.objects_differing.len(),
            self.events_only_in_one.len(),
            self.objects_only_in_one.len()
        )
    }
}

impl Fingerprint {
    /// Read the fingerprint off a log.
    pub fn build(locel: &SlimLinkedOCEL) -> Self {
        let mut events: Vec<(String, u64)> = locel
            .get_all_evs()
            .map(|e| {
                let ev = e.get_ev(locel);
                let mut rels: Vec<(&str, &str)> = e
                    .get_e2o_q(locel)
                    .map(|(q, o)| (q, o.get_ob(locel).id.as_str()))
                    .collect();
                rels.sort_unstable();
                let mut h = DefaultHasher::new();
                e.get_ev_type(locel).hash(&mut h);
                ev.time.timestamp_millis().hash(&mut h);
                format!("{:?}", ev.attributes).hash(&mut h);
                rels.hash(&mut h);
                (ev.id.clone(), h.finish())
            })
            .collect();
        let mut objects: Vec<(String, u64)> = locel
            .get_all_obs()
            .map(|o| {
                let ob = o.get_ob(locel);
                let mut rels: Vec<(&str, &str)> = o
                    .get_o2o_q(locel)
                    .map(|(q, t)| (q, t.get_ob(locel).id.as_str()))
                    .collect();
                rels.sort_unstable();
                let mut h = DefaultHasher::new();
                o.get_ob_type(locel).hash(&mut h);
                format!("{:?}", ob.attributes).hash(&mut h);
                rels.hash(&mut h);
                (ob.id.clone(), h.finish())
            })
            .collect();
        events.sort_unstable();
        objects.sort_unstable();
        Self { events, objects }
    }

    /// Where two logs disagree, capped so a total mismatch does not print a whole log.
    pub fn diff(&self, other: &Self, cap: usize) -> Difference {
        let mut d = Difference::default();
        compare(&self.events, &other.events, cap, &mut d.events_only_in_one, &mut d.events_differing);
        compare(
            &self.objects,
            &other.objects,
            cap,
            &mut d.objects_only_in_one,
            &mut d.objects_differing,
        );
        d
    }
}

/// Merge two id-sorted lists, splitting the disagreements into missing and differing.
fn compare(
    a: &[(String, u64)],
    b: &[(String, u64)],
    cap: usize,
    missing: &mut Vec<String>,
    differing: &mut Vec<String>,
) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                if a[i].1 != b[j].1 && differing.len() < cap {
                    differing.push(a[i].0.clone());
                } else if a[i].1 != b[j].1 {
                    differing.push(String::new());
                }
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                if missing.len() < cap {
                    missing.push(a[i].0.clone());
                } else {
                    missing.push(String::new());
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if missing.len() < cap {
                    missing.push(b[j].0.clone());
                } else {
                    missing.push(String::new());
                }
                j += 1;
            }
        }
    }
    for (id, _) in a.iter().skip(i).chain(b.iter().skip(j)) {
        missing.push(if missing.len() < cap { id.clone() } else { String::new() });
    }
}
