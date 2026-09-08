//! Reusable harness for comparing two extraction results, content for content rather than count
//! for count, shared by the sink-agreement tests and the SQL compiler's own differential tests.
//!
//! A count-only comparison (same number of events, same number of objects) passes against
//! several wrong implementations -- wrong ids, swapped attribute values, a relation attached to
//! the wrong event, a type declared with the wrong attribute list. [`snapshot`] builds a
//! canonical, [`PartialEq`]-comparable [`OcelSnapshot`] from anything implementing
//! [`ReadableOCEL`] -- [`SlimLinkedOCEL`](crate::core::event_data::object_centric::linked_ocel::SlimLinkedOCEL)
//! and [`OCEL`](crate::core::event_data::object_centric::ocel_struct::OCEL) both do -- so
//! `assert_eq!` on two snapshots checks every id, timestamp, attribute value and relation, and
//! prints a real diff when something differs instead of two numbers that happened to match.
//!
//! There are two ways to run one blueprint through two sinks, and they check different things.
//!
//! [`run_against_both`] fans one `extract` call out to both sinks. That is what a
//! [`Target::Event`](super::blueprint::Target::Event) with no `id` expression needs: `extract`
//! mints a fresh UUID per row, so two independent `extract` calls produce different, unrelated
//! event ids and are not comparable by id at all. Fanning out mints the id once, in shared code,
//! and hands it to both sinks identically. The price is that **the extractor sees one resolution
//! answer, not two**: [`FanOutSink`] relays the stronger of them, so with one eager sink in the
//! pair the extractor never takes a [`Resolution::Deferred`] branch, and the comparison is blind
//! to everything that branch decides. It compares write-side behaviour only.
//!
//! [`extract_separately`] is the other half, and the only one that can catch a resolution
//! divergence: it runs `extract` once per sink, so each sink answers `resolve_event`/
//! `resolve_object` its own way and the extractor takes whichever branch that answer selects.
//! Only a blueprint whose entity ids are all author-given can be compared this way, for the
//! reason above.
#![cfg(test)]

use std::collections::BTreeMap;

use chrono::{DateTime, FixedOffset};

// `snapshot`/`OcelSnapshot` are pure and are what the ordering tests compare with. Everything
// that runs a blueprint through two sinks needs a second sink to compare against, which is the
// `DuckDbSink`, so it and its imports are gated together.
#[cfg(feature = "ocel-duckdb")]
use super::{
    blueprint::Blueprint,
    catalog::Catalog,
    extract::extract,
    provider::RowProvider,
    report::{ExtractionError, ExtractionReport},
    sink::{EventRef, ExtractionSink, FinalizeReport, ObjectRef, Resolution, SinkError},
};
use crate::core::event_data::object_centric::readable::ReadableOCEL;
#[cfg(feature = "ocel-duckdb")]
use crate::core::event_data::object_centric::OCELTypeAttribute;
use crate::core::event_data::object_centric::{OCELAttributeValue, OCELType};
#[cfg(feature = "ocel-duckdb")]
use std::collections::HashMap;

/// `(attribute name, attribute type string)` for `t`, sorted by name -- two sinks are expected to
/// agree on the declared set, not on declaration order.
fn sorted_attrs(t: &OCELType) -> Vec<(String, String)> {
    let mut attrs: Vec<(String, String)> = t
        .attributes
        .iter()
        .map(|a| (a.name.clone(), a.value_type.clone()))
        .collect();
    attrs.sort();
    attrs
}

/// A canonical, comparable snapshot of an OCEL's declared types, events, objects and relations.
/// Build with [`snapshot`]; compare two with `assert_eq!` (or `PartialEq`) directly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OcelSnapshot {
    /// Declared event types: name -> `{(attribute name, attribute type string)}`.
    pub(crate) event_types: BTreeMap<String, Vec<(String, String)>>,
    /// Declared object types. See [`Self::event_types`].
    pub(crate) object_types: BTreeMap<String, Vec<(String, String)>>,
    /// Every event, keyed by id.
    pub(crate) events: BTreeMap<String, EventSnapshot>,
    /// Every object, keyed by id.
    pub(crate) objects: BTreeMap<String, ObjectSnapshot>,
}

/// One event's comparable content.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EventSnapshot {
    pub(crate) event_type: String,
    pub(crate) time: DateTime<FixedOffset>,
    /// `(name, value)`, sorted by name.
    pub(crate) attributes: Vec<(String, OCELAttributeValue)>,
    /// `(qualifier, object_id)`, sorted -- a multiset, not deduplicated: a relation repeated
    /// twice must appear twice on both sides.
    pub(crate) e2o: Vec<(String, String)>,
}

/// One object's comparable content.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ObjectSnapshot {
    pub(crate) object_type: String,
    /// `(name, time, value)`, sorted by `(name, time)`.
    pub(crate) attributes: Vec<(String, DateTime<FixedOffset>, OCELAttributeValue)>,
    /// `(qualifier, target_object_id)`, sorted. See [`EventSnapshot::e2o`].
    pub(crate) o2o: Vec<(String, String)>,
}

/// Build an [`OcelSnapshot`] from any [`ReadableOCEL`] implementor, for a content-for-content
/// `assert_eq!` between two extraction results, or between the reference executor and the SQL
/// compiler's output.
pub(crate) fn snapshot<O: ReadableOCEL + ?Sized>(ocel: &O) -> OcelSnapshot {
    let event_types = ocel
        .event_types()
        .iter()
        .map(|t| (t.name.clone(), sorted_attrs(t)))
        .collect();
    let object_types = ocel
        .object_types()
        .iter()
        .map(|t| (t.name.clone(), sorted_attrs(t)))
        .collect();

    let mut events = BTreeMap::new();
    for e in ocel.iter_events() {
        let mut attributes: Vec<(String, OCELAttributeValue)> = e
            .attributes
            .iter()
            .map(|a| (a.name.clone(), a.value.clone()))
            .collect();
        attributes.sort_by(|a, b| a.0.cmp(&b.0));
        let mut e2o: Vec<(String, String)> = e
            .relationships
            .iter()
            .map(|r| (r.qualifier.clone(), r.object_id.clone()))
            .collect();
        e2o.sort();
        events.insert(
            e.id.clone(),
            EventSnapshot {
                event_type: e.event_type.clone(),
                time: e.time,
                attributes,
                e2o,
            },
        );
    }

    let mut objects = BTreeMap::new();
    for o in ocel.iter_objects() {
        let mut attributes: Vec<(String, DateTime<FixedOffset>, OCELAttributeValue)> = o
            .attributes
            .iter()
            .map(|a| (a.name.clone(), a.time, a.value.clone()))
            .collect();
        attributes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut o2o: Vec<(String, String)> = o
            .relationships
            .iter()
            .map(|r| (r.qualifier.clone(), r.object_id.clone()))
            .collect();
        o2o.sort();
        objects.insert(
            o.id.clone(),
            ObjectSnapshot {
                object_type: o.object_type.clone(),
                attributes,
                o2o,
            },
        );
    }

    OcelSnapshot {
        event_types,
        object_types,
        events,
        objects,
    }
}

/// Run one `extract` per sink, so each sink answers endpoint resolution its own way and the
/// extractor takes the branch that answer selects -- the deferral path included, which
/// [`run_against_both`] structurally cannot reach. See the module docs for when to use which.
///
/// Only meaningful for a blueprint whose entity ids are all author-given: a minted event id
/// differs between the two runs by design.
///
/// # Errors
/// Returns whichever run's [`ExtractionError`] came first.
#[cfg(feature = "ocel-duckdb")]
pub(crate) fn extract_separately(
    blueprint: &Blueprint,
    catalog: &dyn Catalog,
    providers: &HashMap<String, &dyn RowProvider>,
    a: &mut dyn ExtractionSink,
    b: &mut dyn ExtractionSink,
) -> Result<(ExtractionReport, ExtractionReport), ExtractionError> {
    let ra = extract(blueprint, catalog, providers, a)?;
    let rb = extract(blueprint, catalog, providers, b)?;
    Ok((ra, rb))
}

/// Run `f` (an `extract` call, typically) against `a` and `b` simultaneously through a fan-out
/// [`ExtractionSink`], so both receive the exact same sequence of declarations, entities and
/// relations -- including any id `extract` mints itself, such as an omitted event id -- rather
/// than each generating its own from two separate `extract` calls. See the module docs.
#[cfg(feature = "ocel-duckdb")]
pub(crate) fn run_against_both<T>(
    a: &mut dyn ExtractionSink,
    b: &mut dyn ExtractionSink,
    f: impl FnOnce(&mut dyn ExtractionSink) -> T,
) -> T {
    let mut fan_out = FanOutSink { a, b };
    f(&mut fan_out)
}

/// Forwards every [`ExtractionSink`] call to both `a` and `b`, verbatim and in order. Always
/// hands back [`EventRef::Id`]/[`ObjectRef::Id`] itself (echoing the id it was given), regardless
/// of what either inner sink's own handle looks like, and translates an incoming ref back to each
/// inner sink's own handle (via that inner sink's own `find_event`/`find_object`) before
/// forwarding a relation call -- the two inner sinks never see each other's ref type.
#[cfg(feature = "ocel-duckdb")]
#[derive(Debug)]
struct FanOutSink<'a> {
    a: &'a mut dyn ExtractionSink,
    b: &'a mut dyn ExtractionSink,
}

#[cfg(feature = "ocel-duckdb")]
impl FanOutSink<'_> {
    fn event_id(r: &EventRef) -> Result<&str, SinkError> {
        match r {
            EventRef::Id(s) => Ok(s.as_str()),
            EventRef::Index(_) => Err(SinkError::InvalidRef),
        }
    }

    fn object_id(r: &ObjectRef) -> Result<&str, SinkError> {
        match r {
            ObjectRef::Id(s) => Ok(s.as_str()),
            ObjectRef::Index(_) => Err(SinkError::InvalidRef),
        }
    }

    /// One inner sink's own handle for `id`, however that sink represents handles.
    fn object_handle(sink: &mut dyn ExtractionSink, id: &str) -> Result<ObjectRef, SinkError> {
        sink.resolve_object(id, None)
            .into_ref()
            .ok_or(SinkError::InvalidRef)
    }

    /// One inner sink's own handle for an event id. See [`Self::object_handle`].
    fn event_handle(sink: &mut dyn ExtractionSink, id: &str) -> Result<EventRef, SinkError> {
        sink.resolve_event(id, None)
            .into_ref()
            .ok_or(SinkError::InvalidRef)
    }
}

/// The more informative of two resolutions: a definite `Exists`/`Missing` beats a `Deferred`,
/// which is a sink declining to answer and therefore compatible with either.
///
/// It has to be this way round -- an eager sink cannot be handed a relation against an entity it
/// does not have, and would answer [`SinkError::InvalidRef`] -- and it is exactly why a fan-out
/// run never exercises a deferral branch. [`extract_separately`] is the harness that does.
#[cfg(feature = "ocel-duckdb")]
fn stronger<R>(a: Resolution<R>, b: Resolution<R>) -> Resolution<R> {
    match (a, b) {
        (Resolution::Deferred(_), other) | (other, Resolution::Deferred(_)) => other,
        (definite, _) => definite,
    }
}

#[cfg(feature = "ocel-duckdb")]
impl<R> Resolution<R> {
    /// Replace the handle, keeping which of the three answers this is.
    fn map_ref<T>(self, f: impl FnOnce(R) -> T) -> Resolution<T> {
        match self {
            Resolution::Exists(r) => Resolution::Exists(f(r)),
            Resolution::Missing => Resolution::Missing,
            Resolution::Deferred(r) => Resolution::Deferred(f(r)),
        }
    }
}

#[cfg(feature = "ocel-duckdb")]
impl ExtractionSink for FanOutSink<'_> {
    fn declare_event_type(
        &mut self,
        name: &str,
        attrs: &[OCELTypeAttribute],
    ) -> Result<(), SinkError> {
        self.a.declare_event_type(name, attrs)?;
        self.b.declare_event_type(name, attrs)?;
        Ok(())
    }

    fn declare_object_type(
        &mut self,
        name: &str,
        attrs: &[OCELTypeAttribute],
    ) -> Result<(), SinkError> {
        self.a.declare_object_type(name, attrs)?;
        self.b.declare_object_type(name, attrs)?;
        Ok(())
    }

    fn add_event(
        &mut self,
        event_type: &str,
        time: DateTime<FixedOffset>,
        id: &str,
        attributes: &[(String, OCELAttributeValue)],
    ) -> Result<EventRef, SinkError> {
        self.a.add_event(event_type, time, id, attributes)?;
        self.b.add_event(event_type, time, id, attributes)?;
        Ok(EventRef::Id(id.to_string()))
    }

    fn add_object(
        &mut self,
        object_type: &str,
        id: &str,
        attributes: &[(String, DateTime<FixedOffset>, OCELAttributeValue)],
    ) -> Result<ObjectRef, SinkError> {
        self.a.add_object(object_type, id, attributes)?;
        self.b.add_object(object_type, id, attributes)?;
        Ok(ObjectRef::Id(id.to_string()))
    }

    fn add_object_attribute(
        &mut self,
        object: &ObjectRef,
        name: &str,
        time: DateTime<FixedOffset>,
        value: OCELAttributeValue,
    ) -> Result<(), SinkError> {
        let id = Self::object_id(object)?;
        let ra = Self::object_handle(self.a, id)?;
        let rb = Self::object_handle(self.b, id)?;
        self.a
            .add_object_attribute(&ra, name, time, value.clone())?;
        self.b.add_object_attribute(&rb, name, time, value)?;
        Ok(())
    }

    fn set_missing_endpoint_policy(
        &mut self,
        policy: crate::core::event_data::object_centric::extraction::MissingEndpointPolicy,
    ) -> Result<(), SinkError> {
        self.a.set_missing_endpoint_policy(policy)?;
        self.b.set_missing_endpoint_policy(policy)?;
        Ok(())
    }

    /// Both inner sinks are asked, so a deferring one records the endpoint it will have to
    /// settle at `finalize`; the answer relayed to the extractor is the *stronger* of the two,
    /// since a definite `Exists`/`Missing` is what the extractor can act on now and a `Deferred`
    /// sink accepts either outcome.
    fn resolve_event(&mut self, id: &str, event_type: Option<&str>) -> Resolution<EventRef> {
        let a = self.a.resolve_event(id, event_type);
        let b = self.b.resolve_event(id, event_type);
        stronger(a, b).map_ref(|_| EventRef::Id(id.to_string()))
    }

    /// See [`resolve_event`](ExtractionSink::resolve_event).
    fn resolve_object(&mut self, id: &str, object_type: Option<&str>) -> Resolution<ObjectRef> {
        let a = self.a.resolve_object(id, object_type);
        let b = self.b.resolve_object(id, object_type);
        stronger(a, b).map_ref(|_| ObjectRef::Id(id.to_string()))
    }

    fn finalize(&mut self) -> Result<FinalizeReport, SinkError> {
        self.a.finalize()?;
        self.b.finalize()
    }

    fn add_e2o(
        &mut self,
        event: &EventRef,
        object: &ObjectRef,
        qualifier: &str,
    ) -> Result<(), SinkError> {
        let eid = Self::event_id(event)?;
        let oid = Self::object_id(object)?;
        let ea = Self::event_handle(self.a, eid)?;
        let oa = Self::object_handle(self.a, oid)?;
        let eb = Self::event_handle(self.b, eid)?;
        let ob = Self::object_handle(self.b, oid)?;
        self.a.add_e2o(&ea, &oa, qualifier)?;
        self.b.add_e2o(&eb, &ob, qualifier)?;
        Ok(())
    }

    fn add_o2o(
        &mut self,
        source: &ObjectRef,
        target: &ObjectRef,
        qualifier: &str,
    ) -> Result<(), SinkError> {
        let sid = Self::object_id(source)?;
        let tid = Self::object_id(target)?;
        let sa = Self::object_handle(self.a, sid)?;
        let ta = Self::object_handle(self.a, tid)?;
        let sb = Self::object_handle(self.b, sid)?;
        let tb = Self::object_handle(self.b, tid)?;
        self.a.add_o2o(&sa, &ta, qualifier)?;
        self.b.add_o2o(&sb, &tb, qualifier)?;
        Ok(())
    }
}
