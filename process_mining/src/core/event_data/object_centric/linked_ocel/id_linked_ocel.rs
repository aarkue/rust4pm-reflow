use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
};

use chrono::{DateTime, FixedOffset};

use super::LinkedOCELAccess;
use crate::core::event_data::object_centric::ocel_struct::{OCELEvent, OCELObject, OCELType, OCEL};
use crate::core::event_data::object_centric::OCELAttributeValue;

impl<'a> LinkedOCELAccess<'a> for IDLinkedOCEL<'a> {
    // Represent objects and events by (String) ID
    type EventRepr = EventID<'a>;
    type ObjectRepr = ObjectID<'a>;

    fn get_evs_of_type(&'a self, ev_type: &'_ str) -> impl Iterator<Item = &'a Self::EventRepr> {
        self.events_per_type.get(ev_type).into_iter().flatten()
    }

    fn get_obs_of_type(&'a self, ob_type: &'_ str) -> impl Iterator<Item = &'a Self::ObjectRepr> {
        self.objects_per_type.get(ob_type).into_iter().flatten()
    }

    fn get_full_ev(&'a self, ev_id: impl Borrow<Self::EventRepr>) -> Cow<'a, OCELEvent> {
        Cow::Borrowed(*self.events.get(ev_id.borrow()).unwrap())
    }

    fn get_full_ob(&'a self, ob_id: impl Borrow<Self::ObjectRepr>) -> Cow<'a, OCELObject> {
        Cow::Borrowed(self.objects.get(ob_id.borrow()).unwrap())
    }

    fn get_e2o(
        &'a self,
        index: impl Borrow<Self::EventRepr>,
    ) -> impl Iterator<Item = (&'a str, &'a Self::ObjectRepr)> {
        self.e2o_rel
            .get(index.borrow())
            .into_iter()
            .flatten()
            .map(|(q, o)| (*q, o))
    }

    fn get_e2o_rev(
        &'a self,
        index: impl Borrow<Self::ObjectRepr>,
    ) -> impl Iterator<Item = (&'a str, &'a Self::EventRepr)> {
        self.e2o_rel_rev
            .get(index.borrow())
            .into_iter()
            .flatten()
            .map(|(q, e)| (*q, e))
    }

    fn get_o2o(
        &'a self,
        index: impl Borrow<Self::ObjectRepr>,
    ) -> impl Iterator<Item = (&'a str, &'a Self::ObjectRepr)> {
        self.o2o_rel
            .get(index.borrow())
            .into_iter()
            .flatten()
            .map(|(q, o)| (*q, o))
    }

    fn get_o2o_rev(
        &'a self,
        index: impl Borrow<Self::ObjectRepr>,
    ) -> impl Iterator<Item = (&'a str, &'a Self::ObjectRepr)> {
        self.o2o_rel_rev
            .get(index.borrow())
            .into_iter()
            .flatten()
            .map(|(q, o)| (*q, o))
    }

    fn get_ev_types(&'a self) -> impl Iterator<Item = &'a str> {
        self.events_per_type.keys().copied()
    }

    fn get_ob_types(&'a self) -> impl Iterator<Item = &'a str> {
        self.objects_per_type.keys().copied()
    }
    fn get_all_evs(&'a self) -> impl Iterator<Item = EventID<'a>> {
        self.events.iter().map(|e| *e.0)
    }

    fn get_all_obs(&'a self) -> impl Iterator<Item = ObjectID<'a>> {
        self.objects.iter().map(|o| *o.0)
    }

    fn get_ev_type(&'a self, ev_type: impl AsRef<str>) -> Option<&'a OCELType> {
        self.ocel
            .event_types
            .iter()
            .find(|et| et.name == ev_type.as_ref())
    }

    fn get_ob_type(&'a self, ob_type: impl AsRef<str>) -> Option<&'a OCELType> {
        self.ocel
            .object_types
            .iter()
            .find(|ot| ot.name == ob_type.as_ref())
    }

    fn get_ob_type_of(&'a self, object: impl Borrow<Self::ObjectRepr>) -> &'a str {
        &self.objects.get(object.borrow()).unwrap().object_type
    }

    fn get_ev_type_of(&'a self, event: impl Borrow<Self::EventRepr>) -> &'a str {
        &self.events.get(event.borrow()).unwrap().event_type
    }

    fn get_ev_attrs(&'a self, ev: impl Borrow<Self::EventRepr>) -> impl Iterator<Item = &'a str> {
        self.events
            .get(ev.borrow())
            .unwrap()
            .attributes
            .iter()
            .map(|a| a.name.as_str())
    }

    fn get_ev_attr_val(
        &'a self,
        ev: impl Borrow<Self::EventRepr>,
        attr_name: impl AsRef<str>,
    ) -> Option<&'a crate::core::event_data::object_centric::OCELAttributeValue> {
        let attr_name = attr_name.as_ref();
        self.events
            .get(ev.borrow())
            .unwrap()
            .attributes
            .iter()
            .filter(|a| a.name == attr_name)
            .map(|a| &a.value)
            .next()
    }

    fn get_ob_attrs(&'a self, ob: impl Borrow<Self::ObjectRepr>) -> impl Iterator<Item = &'a str> {
        self.objects
            .get(ob.borrow())
            .unwrap()
            .attributes
            .iter()
            .map(|a| a.name.as_str())
    }

    fn get_ob_attr_vals(
        &'a self,
        ob: impl Borrow<Self::ObjectRepr>,
        attr_name: impl AsRef<str>,
    ) -> impl Iterator<
        Item = (
            &'a chrono::DateTime<chrono::FixedOffset>,
            &'a crate::core::event_data::object_centric::OCELAttributeValue,
        ),
    > {
        let attr_name = attr_name.as_ref();
        self.objects
            .get(ob.borrow())
            .unwrap()
            .attributes
            .iter()
            .filter(|a| a.name == attr_name)
            .map(|a| (&a.time, &a.value))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn get_ob_id(&'a self, ob: impl Borrow<Self::ObjectRepr>) -> &'a str {
        &self.objects.get(ob.borrow()).unwrap().id
    }

    fn get_ev_id(&'a self, ev: impl Borrow<Self::EventRepr>) -> &'a str {
        &self.events.get(ev.borrow()).unwrap().id
    }

    fn get_ev_by_id(&'a self, ev_id: impl AsRef<str>) -> Option<Self::EventRepr> {
        let e = self.events.get(&EventID(ev_id.as_ref()))?;
        Some(EventID(e.id.as_str()))
    }

    fn get_ob_by_id(&'a self, ob_id: impl AsRef<str>) -> Option<Self::ObjectRepr> {
        let o = self.objects.get(&ObjectID(ob_id.as_ref()))?;
        Some(ObjectID(o.id.as_str()))
    }

    fn get_ev_time(
        &'a self,
        ev: impl Borrow<Self::EventRepr>,
    ) -> &'a chrono::DateTime<chrono::FixedOffset> {
        &self.events.get(ev.borrow()).unwrap().time
    }
}

// `LinkedOCELAccess<'a>` here is tied to `IDLinkedOCEL<'a>`'s own `'a` (unlike the
// Index/Slim backends, which implement it for every lifetime), so `QueryableOCEL`'s
// fresh-borrow `&'s self` can't be delegated to `LinkedOCELAccess` as `'s` and `'a`
// don't generally match. Access the fields directly instead of using
// `impl_queryable_from_linked!`.
impl<'a> super::QueryableOCEL for IDLinkedOCEL<'a> {
    type EventRepr = EventID<'a>;
    type ObjectRepr = ObjectID<'a>;
    // `&'a str` reprs are already zero-copy borrows out of `self.ocel` (unlike e.g. `String`),
    // so there's no native int index to reuse here the way `SlimLinkedOCEL`/`IndexLinkedOCEL`
    // do -- but this handle is exactly as cheap as the `Cow<str>` it replaces (same borrow, no
    // extra hashing cost either way), so there's no regression from using it as a group key.
    type EvTypeId = &'a str;
    type ObTypeId = &'a str;

    fn get_ev_type_id(&self, ev: &Self::EventRepr) -> Self::EvTypeId {
        self.events.get(ev).unwrap().event_type.as_str()
    }

    fn get_ob_type_id(&self, ob: &Self::ObjectRepr) -> Self::ObTypeId {
        self.objects.get(ob).unwrap().object_type.as_str()
    }

    fn resolve_ev_type(&self, id: Self::EvTypeId) -> Cow<'_, str> {
        Cow::Borrowed(id)
    }

    fn resolve_ob_type(&self, id: Self::ObTypeId) -> Cow<'_, str> {
        Cow::Borrowed(id)
    }

    fn get_all_evs(&self) -> impl Iterator<Item = Self::EventRepr> + '_ {
        self.events.keys().copied()
    }

    fn get_all_obs(&self) -> impl Iterator<Item = Self::ObjectRepr> + '_ {
        self.objects.keys().copied()
    }

    fn get_ev_id(&self, ev: &Self::EventRepr) -> Cow<'_, str> {
        Cow::Borrowed(self.events.get(ev).unwrap().id.as_str())
    }

    fn get_ob_id(&self, ob: &Self::ObjectRepr) -> Cow<'_, str> {
        Cow::Borrowed(self.objects.get(ob).unwrap().id.as_str())
    }

    fn get_ev_type_of(&self, ev: &Self::EventRepr) -> Cow<'_, str> {
        Cow::Borrowed(self.events.get(ev).unwrap().event_type.as_str())
    }

    fn get_ob_type_of(&self, ob: &Self::ObjectRepr) -> Cow<'_, str> {
        Cow::Borrowed(self.objects.get(ob).unwrap().object_type.as_str())
    }

    fn get_ev_time(&self, ev: &Self::EventRepr) -> DateTime<FixedOffset> {
        self.events.get(ev).unwrap().time
    }

    #[inline]
    fn get_e2o(
        &self,
        ev: &Self::EventRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::ObjectRepr)> + '_ {
        self.e2o_rel
            .get(ev)
            .into_iter()
            .flatten()
            .map(|(q, o)| (Cow::Borrowed(*q), *o))
    }

    fn get_ob_attr_vals(
        &self,
        ob: &Self::ObjectRepr,
        name: &str,
    ) -> impl Iterator<Item = (DateTime<FixedOffset>, OCELAttributeValue)> + '_ {
        self.objects
            .get(ob)
            .into_iter()
            .flat_map(|o| o.attributes.iter())
            .filter(|a| a.name == name)
            .map(|a| (a.time, a.value.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn get_e2o_rev(
        &self,
        ob: &Self::ObjectRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::EventRepr)> + '_ {
        self.e2o_rel_rev
            .get(ob)
            .into_iter()
            .flatten()
            .map(|(q, e)| (Cow::Borrowed(*q), *e))
    }

    fn get_o2o(
        &self,
        ob: &Self::ObjectRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::ObjectRepr)> + '_ {
        self.o2o_rel
            .get(ob)
            .into_iter()
            .flatten()
            .map(|(q, o)| (Cow::Borrowed(*q), *o))
    }

    fn get_o2o_rev(
        &self,
        ob: &Self::ObjectRepr,
    ) -> impl Iterator<Item = (Cow<'_, str>, Self::ObjectRepr)> + '_ {
        self.o2o_rel_rev
            .get(ob)
            .into_iter()
            .flatten()
            .map(|(q, o)| (Cow::Borrowed(*q), *o))
    }

    fn get_obs_of_type(&self, ty: &str) -> impl Iterator<Item = Self::ObjectRepr> + '_ {
        self.objects_per_type.get(ty).into_iter().flatten().copied()
    }

    fn get_evs_of_type(&self, ty: &str) -> impl Iterator<Item = Self::EventRepr> + '_ {
        self.events_per_type.get(ty).into_iter().flatten().copied()
    }

    fn get_ev_types(&self) -> impl Iterator<Item = Cow<'_, str>> + '_ {
        self.events_per_type.keys().copied().map(Cow::Borrowed)
    }

    fn get_ob_types(&self) -> impl Iterator<Item = Cow<'_, str>> + '_ {
        self.objects_per_type.keys().copied().map(Cow::Borrowed)
    }

    fn get_ev_attr_val(&self, ev: &Self::EventRepr, name: &str) -> Option<OCELAttributeValue> {
        self.events
            .get(ev)
            .into_iter()
            .flat_map(|e| e.attributes.iter())
            .filter(|a| a.name == name)
            .map(|a| a.value.clone())
            .next()
    }

    // In-memory, owned-ref backend (`EventID`/`ObjectID` are `Send + Sync` `&'a str` newtypes)
    // -> use the rayon-parallel evaluator instead of the trait's sequential default.
    fn run_query(
        &self,
        query: &crate::core::event_data::object_centric::query::Query,
    ) -> Result<crate::core::event_data::object_centric::query::eval::QueryResult, String> {
        crate::core::event_data::object_centric::query::eval::evaluate_par(query, self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
/// Object identifier in an [`OCEL`]
pub struct ObjectID<'a>(&'a str);

impl<'a> From<&'a OCELObject> for ObjectID<'a> {
    fn from(value: &'a OCELObject) -> Self {
        Self(value.id.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
/// Event identifier in an [`OCEL`]
pub struct EventID<'a>(&'a str);

impl<'a> From<&'a OCELEvent> for EventID<'a> {
    fn from(value: &'a OCELEvent) -> Self {
        Self(value.id.as_str())
    }
}

#[derive(Debug, Clone)]
/// A [`OCEL`] linked using event and object IDs (i.e., wrappers around [`String`]s)
pub struct IDLinkedOCEL<'a> {
    /// The reference to the inner [`OCEL`], which contains the actual event and object values
    pub ocel: &'a OCEL,
    events: HashMap<EventID<'a>, &'a OCELEvent>,
    objects: HashMap<ObjectID<'a>, &'a OCELObject>,
    events_per_type: HashMap<&'a str, Vec<EventID<'a>>>,
    objects_per_type: HashMap<&'a str, Vec<ObjectID<'a>>>,
    e2o_rel: HashMap<EventID<'a>, Vec<(&'a str, ObjectID<'a>)>>,
    o2o_rel: HashMap<ObjectID<'a>, Vec<(&'a str, ObjectID<'a>)>>,
    e2o_rel_rev: HashMap<ObjectID<'a>, Vec<(&'a str, EventID<'a>)>>,
    o2o_rel_rev: HashMap<ObjectID<'a>, Vec<(&'a str, ObjectID<'a>)>>,
}

impl<'a> IDLinkedOCEL<'a> {
    /// Create a ID-linked OCEL from a [`OCEL`] reference
    pub fn from_ocel(ocel: &'a OCEL) -> Self {
        Self::from(ocel)
    }
}

impl<'a> From<&'a OCEL> for IDLinkedOCEL<'a> {
    fn from(ocel: &'a OCEL) -> Self {
        let events: HashMap<_, _> = ocel.events.iter().map(|e| (EventID(&e.id), e)).collect();
        let objects: HashMap<_, _> = ocel.objects.iter().map(|o| (ObjectID(&o.id), o)).collect();
        let mut e2o_rel_rev: HashMap<ObjectID<'a>, Vec<(&str, EventID<'a>)>> = HashMap::new();
        let e2o_rel = events
            .values()
            .map(|e| {
                let e_id: EventID<'_> = EventID(&e.id);
                (
                    e_id,
                    e.relationships
                        .iter()
                        .flat_map(|rel| {
                            let obj_id: ObjectID<'_> = ObjectID(&rel.object_id);
                            let qualifier = rel.qualifier.as_str();
                            e2o_rel_rev
                                .entry(obj_id)
                                .or_default()
                                .push((qualifier, e_id));
                            let ob = ObjectID(&rel.object_id);
                            Some((qualifier, ob))
                        })
                        .collect(),
                )
            })
            .collect();

        let mut o2o_rel_rev: HashMap<ObjectID<'_>, Vec<(&'a str, ObjectID<'a>)>> = HashMap::new();

        let o2o_rel = objects
            .values()
            .map(|o| {
                (
                    ObjectID(&o.id),
                    o.relationships
                        .iter()
                        .flat_map(|rel| {
                            let qualifier = rel.qualifier.as_str();
                            let obj2_id: ObjectID<'_> = ObjectID(&rel.object_id);
                            o2o_rel_rev
                                .entry(obj2_id)
                                .or_default()
                                .push((qualifier, ObjectID(&o.id)));
                            Some((qualifier, obj2_id))
                        })
                        .collect(),
                )
            })
            .collect();
        let events_per_type = ocel
            .event_types
            .iter()
            .map(|et| {
                (
                    et.name.as_str(),
                    events
                        .iter()
                        .filter(|(_eid, e)| e.event_type == et.name)
                        .map(|(eid, _)| *eid)
                        .collect(),
                )
            })
            .collect();

        let objects_per_type = ocel
            .object_types
            .iter()
            .map(|ot| {
                (
                    ot.name.as_str(),
                    objects
                        .iter()
                        .filter(|(_oid, o)| o.object_type == ot.name)
                        .map(|(oid, _)| *oid)
                        .collect(),
                )
            })
            .collect();
        Self {
            ocel,
            events,
            objects,
            events_per_type,
            objects_per_type,
            e2o_rel,
            o2o_rel,
            e2o_rel_rev,
            o2o_rel_rev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queryable_matches_linked_for_id() {
        use crate::core::event_data::object_centric::linked_ocel::QueryableOCEL;
        let ocel = crate::core::event_data::object_centric::ocel_json::import_ocel_json_path(
            crate::test_utils::get_test_data_path()
                .join("ocel")
                .join("order-management.json"),
        )
        .unwrap();
        let id = IDLinkedOCEL::from_ocel(&ocel);

        // Same first-event type via both traits.
        let ev = QueryableOCEL::get_all_evs(&id).next().unwrap();
        let q_type = QueryableOCEL::get_ev_type_of(&id, &ev).into_owned();
        let l_type = LinkedOCELAccess::get_ev_type_of(&id, &ev).to_string();
        assert_eq!(q_type, l_type);
    }
}
