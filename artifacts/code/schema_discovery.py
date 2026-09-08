"""Structural schema discovery over an OCEL 2.0 SQLite log.

Two independent sources of candidate maps, reported separately because they
are complementary and neither subsumes the other:

  recorded    total maps read off the object-to-object relation, per qualifier
  coparticip  total maps forced by event co-participation

The co-participation pass is the one the pitch note flags as defective. Its
running intersection can empty for two different reasons and they mean
opposite things:

  CONFLICT  the source object co-occurred with T objects in two events that
            share no T object. There is no function; the pair is REJECTED.
  UNTOTAL   the source object appeared in an event carrying no T object at
            all. The map may still exist with this object as a residual.

Collapsing the two is what made the fast implementation report 14 maps on
Container Logistics where the reference reported 13: a conflict was absorbed
by the residual slack instead of killing the candidate.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Log:
    obj_type: dict[str, str]
    objects_of: dict[str, list[str]]
    event_objects: dict[str, list[str]]
    event_activity: dict[str, str]
    o2o: list[tuple[str, str, str]]

    @property
    def types(self) -> list[str]:
        return sorted(self.objects_of)


def load(path: str) -> Log:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    obj_type = dict(con.execute("select ocel_id, ocel_type from object"))
    objects_of: dict[str, list[str]] = defaultdict(list)
    for oid, ot in obj_type.items():
        objects_of[ot].append(oid)

    event_activity = dict(con.execute("select ocel_id, ocel_type from event"))
    event_objects: dict[str, list[str]] = defaultdict(list)
    for eid, oid in con.execute(
        "select ocel_event_id, ocel_object_id from event_object"
    ):
        if oid in obj_type:
            event_objects[eid].append(oid)

    o2o = [
        (s, t, q)
        for s, t, q in con.execute(
            "select ocel_source_id, ocel_target_id, ocel_qualifier from object_object"
        )
        if s in obj_type and t in obj_type
    ]
    con.close()
    return Log(obj_type, dict(objects_of), dict(event_objects), event_activity, o2o)


# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------

@dataclass
class Map:
    source: str
    target: str
    origin: str                     # "recorded" or "coparticip"
    qualifier: str | None           # None = unqualified / union over qualifiers
    f: dict[str, str]               # source object -> target object
    residual: set[str] = field(default_factory=set)   # untotal source objects
    ambiguous: int = 0              # sources whose candidate set stayed >1

    @property
    def coverage(self) -> float:
        n = len(self.f) + len(self.residual) + self.ambiguous
        return len(self.f) / n if n else 0.0

    @property
    def surjective_onto(self) -> int:
        return len(set(self.f.values()))

    def __str__(self) -> str:
        q = f" [{self.qualifier}]" if self.qualifier else ""
        return (
            f"{self.source} -> {self.target}{q} ({self.origin}) "
            f"cov={self.coverage:.4f} img={self.surjective_onto} "
            f"res={len(self.residual)} amb={self.ambiguous}"
        )


def recorded_maps(
    log: Log, min_coverage: float = 0.95, min_target_objects: int = 2
) -> list[Map]:
    """Total maps read off O2O, one candidate per (source type, target type,
    qualifier). A source object with two distinct targets under the same
    qualifier makes the pair non-functional and is rejected outright."""
    by_pair: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    # BOTH orientations. An O2O edge recorded as A -[q]-> B may be functional in
    # either direction and the log records only one of them; reading the forward
    # direction alone missed `Offer -> Application` on BPIC2017, a total function
    # over all 42,995 offers, and with it 16% of that log's expansion mass.
    for s, t, q in log.o2o:
        by_pair[(log.obj_type[s], log.obj_type[t], q)][s].add(t)
        by_pair[(log.obj_type[t], log.obj_type[s], f"~{q}")][t].add(s)

    out: list[Map] = []
    for (st, tt, q), rel in by_pair.items():
        if len(log.objects_of[tt]) < min_target_objects:
            continue
        if any(len(v) > 1 for v in rel.values()):
            continue                                  # not functional
        f = {s: next(iter(v)) for s, v in rel.items()}
        residual = set(log.objects_of[st]) - set(f)
        m = Map(st, tt, "recorded", q, f, residual)
        if m.coverage >= min_coverage:
            out.append(m)
    return out


def coparticipation_maps(
    log: Log,
    min_coverage: float = 0.95,
    strict_conflicts: bool = True,
    min_target_objects: int = 2,
) -> tuple[list[Map], dict[tuple[str, str], str]]:
    """Total maps forced by co-participation, via a running intersection per
    source object with a singleton fast path and early death.

    `strict_conflicts=False` reproduces the defect: a conflict is treated as
    an untotal source and absorbed by the residual slack.

    Returns the maps and, per rejected pair, the reason it was rejected.
    """
    types = log.types
    rejected: dict[tuple[str, str], str] = {}
    out: list[Map] = []

    objs_by_type_in_event: dict[str, dict[str, list[str]]] = {}
    for eid, oids in log.event_objects.items():
        per: dict[str, list[str]] = defaultdict(list)
        for oid in oids:
            per[log.obj_type[oid]].append(oid)
        objs_by_type_in_event[eid] = per

    events_of_object: dict[str, list[str]] = defaultdict(list)
    for eid, oids in log.event_objects.items():
        for oid in oids:
            events_of_object[oid].append(eid)

    for st in types:
        for tt in types:
            if st == tt:
                continue
            if len(log.objects_of[tt]) < min_target_objects:
                # A type with one object makes every map into it trivially
                # total and carries no information. Excluding these is not
                # cosmetic: Hinge has five such types.
                rejected[(st, tt)] = f"degenerate target: |{tt}|={len(log.objects_of[tt])}"
                continue
            cand: dict[str, set[str] | None] = {}
            untotal: set[str] = set()
            conflict: set[str] = set()
            updates = 0

            for s in log.objects_of[st]:
                cur: set[str] | None = None
                dead = False
                for eid in events_of_object[s]:
                    ts = objs_by_type_in_event[eid].get(tt)
                    if not ts:
                        untotal.add(s)          # event with no T object at all
                        continue
                    updates += 1
                    if cur is None:
                        cur = set(ts)
                    elif len(cur) == 1:         # singleton fast path
                        if next(iter(cur)) not in ts:
                            conflict.add(s)
                            dead = True
                            break
                    else:
                        cur &= set(ts)
                        if not cur:             # early death
                            conflict.add(s)
                            dead = True
                            break
                if dead:
                    cand[s] = None
                else:
                    cand[s] = cur

            if strict_conflicts and conflict:
                rejected[(st, tt)] = f"non-functional: {len(conflict)} conflicts"
                continue

            f: dict[str, str] = {}
            residual: set[str] = set()
            ambiguous = 0
            for s, c in cand.items():
                if not c:                       # None or empty
                    residual.add(s)
                    continue
                if len(c) > 1:
                    # Co-participation is CONSISTENT with several images and
                    # forces none of them. The map is undetermined here;
                    # choosing a representative would invent a function.
                    ambiguous += 1
                    continue
                f[s] = next(iter(c))

            m = Map(st, tt, "coparticip", None, f, residual, ambiguous)
            if m.coverage >= min_coverage:
                out.append(m)
            else:
                rejected[(st, tt)] = f"untotal: coverage {m.coverage:.3f}"
    return out, rejected


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------

def generators(pairs: set[tuple[str, str]]) -> tuple[set[tuple[str, str]], int]:
    """Transitive reduction of the map graph, plus the maximum derivation
    depth needed to reach every non-generator edge."""
    succ: dict[str, set[str]] = defaultdict(set)
    for s, t in pairs:
        succ[s].add(t)

    def reachable(start: str, banned: tuple[str, str]) -> dict[str, int]:
        seen = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for u in frontier:
                for v in succ[u]:
                    if (u, v) == banned or v in seen:
                        continue
                    seen[v] = seen[u] + 1
                    nxt.append(v)
            frontier = nxt
        return seen

    gens: set[tuple[str, str]] = set()
    max_depth = 1
    for s, t in pairs:
        depths = reachable(s, (s, t))
        if t in depths:
            max_depth = max(max_depth, depths[t])
        else:
            gens.add((s, t))
    return gens, max_depth


# --------------------------------------------------------------------------
# Single-pass co-participation discovery
# --------------------------------------------------------------------------

def coparticipation_maps_singlepass(
    log: Log,
    min_coverage: float = 0.95,
    strict_conflicts: bool = True,
    min_target_objects: int = 2,
) -> tuple[list[Map], dict[tuple[str, str], str], int]:
    """Same result as `coparticipation_maps`, in one pass over the events.

    The per-pair formulation costs O(|OT|^2) scans and is hopeless on a log
    with 120 object types. Here every (source object, target type) running
    intersection is maintained together, so the cost is
    sum_e |obj(e)| * |types(e)| rather than |OT|^2 * |O|.

    Returns the maps, the rejection reasons, and the number of candidate
    updates performed (the cost figure the paper reports per E2O tuple).
    """
    eligible = {t for t in log.types if len(log.objects_of[t]) >= min_target_objects}

    cand: dict[tuple[str, str], set[str] | None] = {}
    seen: dict[tuple[str, str], int] = defaultdict(int)   # events where T was present
    events_seen: dict[str, int] = defaultdict(int)        # events containing s
    updates = 0

    for oids in log.event_objects.values():
        per: dict[str, list[str]] = defaultdict(list)
        for oid in oids:
            per[log.obj_type[oid]].append(oid)
        for oid in oids:
            events_seen[oid] += 1
        for tt, ts in per.items():
            if tt not in eligible:
                continue
            tset = set(ts)
            for oid in oids:
                if log.obj_type[oid] == tt:
                    continue
                key = (oid, tt)
                seen[key] += 1
                updates += 1
                cur = cand.get(key, "unset")
                if cur == "unset":
                    cand[key] = set(tset)
                elif cur is None:
                    continue                       # already dead
                elif len(cur) == 1:                # singleton fast path
                    if next(iter(cur)) not in tset:
                        cand[key] = None           # conflict
                else:
                    cur &= tset
                    if not cur:
                        cand[key] = None           # early death

    rejected: dict[tuple[str, str], str] = {}
    out: list[Map] = []
    for st in log.types:
        for tt in log.types:
            if st == tt:
                continue
            if tt not in eligible:
                rejected[(st, tt)] = f"degenerate target: |{tt}|={len(log.objects_of[tt])}"
                continue
            f: dict[str, str] = {}
            residual: set[str] = set()
            ambiguous = 0
            conflicts = 0
            for s in log.objects_of[st]:
                c = cand.get((s, tt), "unset")
                if c is None:
                    conflicts += 1
                    residual.add(s)
                    continue
                if c == "unset":
                    residual.add(s)                # never co-occurred with T
                    continue
                # NOT a residual here: an event containing s but no T object
                # gives no constraint on f(s). Object-level totality is what a
                # map needs; event-level co-presence is a separate condition,
                # and it is tested per cell when reconstruction is checked.
                # Conflating the two rejects items -> orders, which holds.
                if len(c) > 1:
                    ambiguous += 1
                    continue
                f[s] = next(iter(c))
            if strict_conflicts and conflicts:
                rejected[(st, tt)] = f"non-functional: {conflicts} conflicts"
                continue
            m = Map(st, tt, "coparticip", None, f, residual, ambiguous)
            if m.coverage >= min_coverage:
                out.append(m)
            else:
                rejected[(st, tt)] = f"untotal: coverage {m.coverage:.3f}"
    return out, rejected, updates
