"""Independently verify the handoff-preserving reduced log from the session-2 handover.

Their implementation is gone, so this export is the only checkable artifact of that
line of work. Nothing here imports their code: the OCEL 2.0 XML is parsed directly and
the annotation's 24 reconstruction rules are re-executed from scratch against the full
log.

Three questions, in order:

  1. Is the reduced log exactly the full log restricted to the annotation's kept cells?
  2. Do the 24 rules, run in the given order, reproduce the full E2O relation exactly?
  3. What does the annotation actually have to ship for (2) to be executable from the
     reduced log alone? A rule naming a 'co' map is not executable unless that map
     travels with it, and the exported JSON is 4 KB.

Usage:
  python verify_handoff_export.py <full.xml> <reduced.xml> <annotation.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from xml.etree import ElementTree as ET


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

class Ocel:
    def __init__(self) -> None:
        self.obj_type: dict[str, str] = {}
        self.act: dict[str, str] = {}
        # ISO-8601 as written in the log, compared lexicographically. Good enough
        # to order events; not good enough to mix time zones, which OCEL 2.0 XML
        # does not do.
        self.time: dict[str, str] = {}
        self.e2o: set[tuple[str, str]] = set()
        self.e2o_q: list[tuple[str, str, str]] = []
        self.o2o: list[tuple[str, str, str]] = []

    @property
    def objects_of(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for oid, ot in self.obj_type.items():
            out[ot].add(oid)
        return out

    def cells(self) -> set[tuple[str, str]]:
        return {(self.act[e], self.obj_type[o]) for e, o in self.e2o}

    def cell_rows(self) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """(activity, object type) -> set of (event id, object id)."""
        out: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for e, o in self.e2o:
            out[(self.act[e], self.obj_type[o])].add((e, o))
        return out


def _rels(el) -> list:
    """OCEL 2.0 XML in the wild uses two tags for the same thing: <relationship> in the
    reference logs, <relobj> in the SAP-extracted ones."""
    return el.findall("./objects/relationship") + el.findall("./objects/relobj")


def parse(path: str) -> Ocel:
    log = Ocel()
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "object" and el.get("id") is not None:
            oid, ot = el.get("id"), el.get("type")
            log.obj_type[oid] = ot
            for rel in _rels(el):
                log.o2o.append((oid, rel.get("object-id"), rel.get("qualifier")))
            el.clear()
        elif el.tag == "event" and el.get("id") is not None:
            eid, a = el.get("id"), el.get("type")
            log.act[eid] = a
            log.time[eid] = el.get("time") or ""
            for rel in _rels(el):
                tgt = rel.get("object-id")
                log.e2o.add((eid, tgt))
                log.e2o_q.append((eid, tgt, rel.get("qualifier")))
            el.clear()

    # Container Logistics references objects in E2O that its objects table never
    # declares. pm4py crashes on it; silently ignoring them is worse, so they are
    # dropped and counted.
    dangling = {o for _, o in log.e2o if o not in log.obj_type}
    if dangling:
        print(f"  [{path}] dropping {len(dangling)} objects referenced in E2O but "
              f"never declared, over "
              f"{sum(1 for _, o in log.e2o if o in dangling)} tuples", file=sys.stderr)
        log.e2o = {(e, o) for e, o in log.e2o if o not in dangling}
        log.e2o_q = [(e, o, q) for e, o, q in log.e2o_q if o not in dangling]
    return log


# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------

def qualified_maps(log: Ocel) -> dict[tuple[str, str, str], dict[str, str]]:
    """Functional maps read off O2O, keyed by (source type, target type, qualifier).

    The edge is stored as recorded; a map source->target exists when every source
    object has at most one target under that qualifier. Both orientations are tried,
    because the export's rules use `fibre` for one direction and `union` for the other.
    """
    fwd: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    bwd: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for s, t, q in log.o2o:
        if s not in log.obj_type or t not in log.obj_type:
            continue
        st, tt = log.obj_type[s], log.obj_type[t]
        fwd[(st, tt, q)][s].add(t)
        bwd[(tt, st, q)][t].add(s)

    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for store in (fwd, bwd):
        for key, rel in store.items():
            if all(len(v) == 1 for v in rel.values()):
                out[key] = {s: next(iter(v)) for s, v in rel.items()}
    return out


def coparticipation_map(log: Ocel, st: str, tt: str) -> dict[str, str]:
    """Total map forced by event co-participation: running intersection per source
    object, singleton fast path, early death on conflict. Ambiguous and conflicting
    sources are dropped rather than resolved -- picking a representative would
    invent a function."""
    ev_objs: dict[str, list[str]] = defaultdict(list)
    for e, o in log.e2o:
        ev_objs[e].append(o)

    cand: dict[str, set[str] | None] = {}
    for objs in ev_objs.values():
        ts = {o for o in objs if log.obj_type[o] == tt}
        if not ts:
            continue
        for o in objs:
            if log.obj_type[o] != st:
                continue
            cur = cand.get(o, "unset")
            if cur == "unset":
                cand[o] = set(ts)
            elif cur is None:
                continue
            elif len(cur) == 1:
                if next(iter(cur)) not in ts:
                    cand[o] = None
            else:
                cur &= ts
                if not cur:
                    cand[o] = None
    return {s: next(iter(c)) for s, c in cand.items() if c and len(c) == 1}


# --------------------------------------------------------------------------
# Rule execution
# --------------------------------------------------------------------------

def parse_rule(rule: str) -> tuple[str, str, list[str]]:
    """'items = fibre[o2o](orders)'            -> ('fibre', 'orders', ['o2o'])
       "employees = union of ['a','b'](packages)" -> ('union', 'packages', ['a','b'])"""
    rhs = rule.split("=", 1)[1].strip()
    if rhs.startswith("fibre["):
        qual = rhs[len("fibre["):rhs.index("]")]
        src = rhs[rhs.index("(") + 1:rhs.rindex(")")]
        return "fibre", src, [qual]
    if rhs.startswith("union of ["):
        inner = rhs[len("union of ["):rhs.index("]")]
        quals = [q.strip().strip("'\"") for q in inner.split(",") if q.strip()]
        src = rhs[rhs.index("(") + 1:rhs.rindex(")")]
        return "union", src, quals
    raise ValueError(f"unparsed rule: {rule}")


def main() -> None:
    full_path, red_path, ann_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ann = json.load(open(ann_path))
    keep = {tuple(c) for c in ann["kept_cells"]}

    full = parse(full_path)
    red = parse(red_path)

    print(f"full    : {len(full.act)} events, {len(full.obj_type)} objects, "
          f"{len(full.e2o)} E2O pairs ({len(full.e2o_q)} qualified), "
          f"{len(full.o2o)} O2O, {len(full.cells())} cells")
    print(f"reduced : {len(red.act)} events, {len(red.obj_type)} objects, "
          f"{len(red.e2o)} E2O pairs ({len(red.e2o_q)} qualified), "
          f"{len(red.o2o)} O2O, {len(red.cells())} cells")

    # ---- Q1: is the reduced log the full log restricted to the kept cells? ----
    print("\n== Q1  reduced == full | keep-set")
    expect = {(e, o) for e, o in full.e2o if (full.act[e], full.obj_type[o]) in keep}
    print(f"  kept cells in annotation : {len(keep)}")
    print(f"  cells present in reduced : {len(red.cells())}")
    print(f"  extra cells in reduced   : {sorted(red.cells() - keep)}")
    print(f"  missing cells in reduced : {sorted(keep - red.cells())}")
    print(f"  expected E2O pairs       : {len(expect)}")
    print(f"  actual   E2O pairs       : {len(red.e2o)}")
    print(f"  symmetric difference     : {len(expect ^ red.e2o)}")
    print(f"  O2O preserved            : {len(red.o2o) == len(full.o2o)} "
          f"({len(red.o2o)} vs {len(full.o2o)})")
    removed = len(full.e2o_q) - len(red.e2o_q)
    print(f"  qualified E2O removed    : {removed} of {len(full.e2o_q)} "
          f"= {100 * removed / len(full.e2o_q):.1f}%")

    # ---- Q2: do the 24 rules reproduce the full E2O relation? ----
    print("\n== Q2  replay the reconstruction rules")
    qmaps = qualified_maps(full)
    co_cache: dict[tuple[str, str], dict[str, str]] = {}

    ev_objs: dict[str, set[str]] = defaultdict(set)
    for e, o in red.e2o:
        ev_objs[e].add(o)
    events_of_act: dict[str, list[str]] = defaultdict(list)
    for e, a in full.act.items():
        events_of_act[a].append(e)

    full_rows = full.cell_rows()
    needed_payload: dict[str, int] = {}
    ok = bad = 0

    fib_cache: dict[tuple[str, str, str], dict[str, set[str]]] = {}

    def fibre(st: str, tt: str, q: str) -> dict[str, set[str]]:
        """Preimage index of the map st -> tt. 'o2o' matches any qualifier."""
        key = (st, tt, q)
        if key not in fib_cache:
            qual = q[2:] if q.startswith("q:") else None
            inv: dict[str, set[str]] = defaultdict(set)
            for (a_, b_, q_), m in qmaps.items():
                if a_ != st or b_ != tt or (qual is not None and q_ != qual):
                    continue
                for x, y in m.items():
                    inv[y].add(x)
            fib_cache[key] = inv
        return fib_cache[key]

    for step in ann["reconstruction_order"]:
        a, tt, rule = step["activity"], step["object_type"], step["rule"]
        kind, src, quals = parse_rule(rule)

        got: set[tuple[str, str]] = set()
        for e in events_of_act[a]:
            srcs = [o for o in ev_objs[e] if full.obj_type[o] == src]
            for s in srcs:
                if kind == "fibre":
                    # preimage of s under (tt -> src); 'o2o' means any qualifier
                    for q in quals:
                        got.update((e, x) for x in fibre(tt, src, q).get(s, ()))
                else:
                    for q in quals:
                        if q == "co":
                            key = (src, tt)
                            if key not in co_cache:
                                co_cache[key] = coparticipation_map(full, src, tt)
                                needed_payload[f"co {src}->{tt}"] = len(co_cache[key])
                            t = co_cache[key].get(s)
                        else:
                            qual = q[2:] if q.startswith("q:") else None
                            if qual is None:
                                t = None
                                for (a_, b_, q_), m in qmaps.items():
                                    if a_ == src and b_ == tt and s in m:
                                        t = m[s]
                                        break
                            else:
                                m = qmaps.get((src, tt, qual))
                                t = m.get(s) if m else None
                        if t is not None:
                            got.add((e, t))
            # a reconstructed cell feeds later rules
        for e, o in got:
            ev_objs[e].add(o)

        want = full_rows.get((a, tt), set())
        miss, spur = want - got, got - want
        flag = "OK " if not miss and not spur else "FAIL"
        if flag == "OK ":
            ok += 1
        else:
            bad += 1
        print(f"  {flag} {a:18s} {tt:10s} want={len(want):6d} got={len(got):6d} "
              f"missing={len(miss):5d} spurious={len(spur):5d}  [{rule}]")

    print(f"\n  rules exact: {ok}/{ok + bad}")

    # ---- Q3: what must the annotation ship? ----
    print("\n== Q3  annotation payload")
    used_q: set[tuple[str, str, str]] = set()
    for step in ann["reconstruction_order"]:
        kind, src, quals = parse_rule(step["rule"])
        tt = step["object_type"]
        for q in quals:
            if q == "co":
                continue
            qual = q[2:] if q.startswith("q:") else None
            if kind == "fibre":
                used_q.add((tt, src, qual or "*"))
            else:
                used_q.add((src, tt, qual or "*"))
    print("  maps recoverable from the O2O the reduced log still carries:")
    for k in sorted(used_q):
        print(f"    {k[0]} -> {k[1]}  [{k[2]}]")
    print("  maps that must travel WITH the annotation (not in O2O):")
    if not needed_payload:
        print("    none")
    for k, v in sorted(needed_payload.items()):
        print(f"    {k}: {v} entries")

    rebuilt = {(e, o) for e, objs in ev_objs.items() for o in objs}
    print(f"\n== round trip: rebuilt {len(rebuilt)} of {len(full.e2o)} E2O pairs, "
          f"symmetric difference {len(rebuilt ^ full.e2o)}")


if __name__ == "__main__":
    main()
