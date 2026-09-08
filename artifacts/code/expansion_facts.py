#!/usr/bin/env python3
"""Does expansion give a reader facts the log did not already show?

Expansion's case in \\autoref{sec:evaluation} is that it adds a lot of tuples
and that it joins components. Neither says the resulting model tells anyone
anything new. This asks the question the reduction side already answers: how
many *ordering* facts and *co-occurrence* facts does the model carry before and
after, where a fact is a covering pair of a type's eventually-follows relation
or a pair of activities that never share an object.

BPIC2017 is the case to run it on. Its `A_`, `W_` and `O_` activity families
share no participating object type, so the full model has no cross-family
ordering fact at all; if expansion creates any, that is the first evidence the
operator improves a model rather than enlarging it.

Efficiency: a type-level eventually-follows relation only needs, per object and
activity, the earliest and latest timestamp. Activity `x` precedes `y` for that
object iff `min(x) < max(y)`, strictly, so simultaneous events stay unordered.
That turns a quadratic-in-trace-length computation into one that is quadratic in
the 26 activities, which is what makes 1.2M events tractable here.

Usage: expansion_facts.py <log.xml> [--max-depth=3]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_probe import compose, qualified_maps  # noqa: E402


def bounds(log, e2o):
    """object -> activity -> (earliest, latest) timestamp."""
    out: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for e, o in e2o:
        a, t = log.act[e], log.time.get(e, "")
        cur = out[o].get(a)
        out[o][a] = (t, t) if cur is None else (min(cur[0], t), max(cur[1], t))
    return out


def facts(log, e2o):
    """Per type: covering ordering pairs, and never-co-occur pairs."""
    bd = bounds(log, e2o)
    ef: dict[str, set[tuple[str, str]]] = defaultdict(set)
    acts: dict[str, set[str]] = defaultdict(set)
    for o, per in bd.items():
        t = log.obj_type[o]
        acts[t] |= set(per)
        for x, (xmin, _) in per.items():
            for y, (_, ymax) in per.items():
                if x != y and xmin < ymax:
                    ef[t].add((x, y))

    order, never = {}, {}
    for t in acts:
        aa = sorted(acts[t])
        so = {(x, y) for x, y in ef[t] if (y, x) not in ef[t]}
        order[t] = {(x, y) for (x, y) in so
                    if not any((x, m) in so and (m, y) in so for m in aa)}
        never[t] = {(x, y) for x, y in combinations(aa, 2)
                    if (x, y) not in ef[t] and (y, x) not in ef[t]}
    return order, never, acts


def family(a: str) -> str:
    """BPIC2017's three activity families, which is what expansion must join."""
    return a.split("_", 1)[0] if "_" in a[:2] else a[:1]


def main() -> None:
    depth = 3
    for a in sys.argv[1:]:
        if a.startswith("--max-depth="):
            depth = int(a.split("=", 1)[1])
    log = parse(sys.argv[1])

    by_type: dict[str, set[str]] = defaultdict(set)
    for oid, ot in log.obj_type.items():
        by_type[ot].add(oid)
    maps = compose(qualified_maps(log), by_type, depth)

    ev_objs: dict[str, set[str]] = defaultdict(set)
    for e, o in log.e2o:
        ev_objs[e].add(o)
    cells = log.cells()
    events_of: dict[str, list[str]] = defaultdict(list)
    for e, a in log.act.items():
        events_of[a].append(e)

    added: set[tuple[str, str]] = set()
    for (s, t, _q), m in maps.items():
        for a, evs in events_of.items():
            if (a, t) in cells:
                continue
            for e in evs:
                for o in ev_objs[e]:
                    if log.obj_type.get(o) != s:
                        continue
                    img = m.get(o)
                    if img is not None and img not in ev_objs[e]:
                        added.add((e, img))
    print(f"events {len(log.act)}  recorded E2O {len(log.e2o)}  "
          f"expansion adds {len(added)} ({100.0 * len(added) / len(log.e2o):.1f}%)")

    for label, rel in (("recorded", list(log.e2o)),
                       ("expanded", list(log.e2o) + sorted(added))):
        order, never, acts = facts(log, rel)
        to = sum(len(v) for v in order.values())
        tn = sum(len(v) for v in never.values())
        cross = sum(1 for t in order for x, y in order[t]
                    if family(x) != family(y))
        print(f"\n{label}: {to} ordering facts, {tn} co-occurrence facts, "
              f"{cross} of the orderings cross activity families")
        for t in sorted(order):
            xf = sum(1 for x, y in order[t] if family(x) != family(y))
            print(f"    {t:<14} {len(acts[t]):>3} activities  "
                  f"{len(order[t]):>4} ordering ({xf} cross-family)  "
                  f"{len(never[t]):>4} never")


if __name__ == "__main__":
    main()
