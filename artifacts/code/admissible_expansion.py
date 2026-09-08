#!/usr/bin/env python3
"""Does interval containment keep BPIC2017's expansion and kill the nonsense one?

Expansion cannot be checked against the data the way a demotion can: the cell
being empty is what makes it a candidate. So it needs an admissibility rule,
and saturation (write every determined participation) has none -- it puts
packages at `place order`, before the package exists.

Rule under test: write the determined object `t` at event `e` only when `e`
falls inside `t`'s own recorded lifetime, `first(t) <= time(e) <= last(t)`.
Uses only the target object's recorded timestamps, so an under-recorded target
yields a shorter interval and therefore *less* expansion, never bogus
expansion.

Reports, per log: candidates, how many survive, and what the surviving set does
to the ordering facts, against the unrestricted expansion.

Maps come from schema_maps.full_maps, i.e. the discovery oracle: recorded O2O
and co-participation both. The earlier compose(qualified_maps(.)) set was
recorded-O2O only, which happens to be exact for BPIC2017 (3 maps either way)
and an undercount everywhere else, so totals here are no longer comparable to
expansion_facts.py's published 405/72 except on BPIC2017.

Usage: admissible_expansion.py <log.xml>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from expansion_facts import facts, family, bounds  # noqa: E402
from schema_maps import full_maps  # noqa: E402


def closure(log, e2o) -> dict[str, set[tuple[str, str]]]:
    """Strict ordering pairs before reduction to covering pairs.

    Covering size is a readability measure and is NOT monotone under adding
    tuples: a subset can show more covering edges than its superset, because a
    new edge makes older ones implied. docs/SPINE.md records that as a metric
    bug that was hit once already. Closure size is the monotone one, so it is
    what any "asserts more / asserts less" comparison has to use.
    """
    bd = bounds(log, e2o)
    ef: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for o, per in bd.items():
        t = log.obj_type[o]
        for x, (xmin, _) in per.items():
            for y, (_, ymax) in per.items():
                if x != y and xmin < ymax:
                    ef[t].add((x, y))
    return {t: {(x, y) for x, y in v if (y, x) not in v} for t, v in ef.items()}


def lifetime(log) -> dict[str, tuple[str, str]]:
    """object -> (first, last) recorded timestamp."""
    out: dict[str, tuple[str, str]] = {}
    for e, o in log.e2o:
        t = log.time.get(e, "")
        if o not in out:
            out[o] = (t, t)
        else:
            lo, hi = out[o]
            out[o] = (min(lo, t), max(hi, t))
    return out


def run(path: str, depth: int = 3) -> None:
    log = parse(path)
    maps = full_maps(log, depth)
    life = lifetime(log)

    ev_objs: dict[str, set[str]] = defaultdict(set)
    for e, o in log.e2o:
        ev_objs[e].add(o)
    cells = log.cells()
    events_of: dict[str, list[str]] = defaultdict(list)
    for e, a in log.act.items():
        events_of[a].append(e)

    allc: set[tuple[str, str]] = set()
    adm: set[tuple[str, str]] = set()
    cell_all: set[tuple[str, str]] = set()
    cell_adm: set[tuple[str, str]] = set()
    for (s, t, _q), m in maps.items():
        for a, evs in events_of.items():
            if (a, t) in cells:
                continue
            for e in evs:
                ts = log.time.get(e, "")
                for o in ev_objs[e]:
                    if log.obj_type.get(o) != s:
                        continue
                    img = m.get(o)
                    if img is None or img in ev_objs[e]:
                        continue
                    allc.add((e, img))
                    cell_all.add((a, t))
                    lo, hi = life.get(img, ("", ""))
                    if lo <= ts <= hi:
                        adm.add((e, img))
                        cell_adm.add((a, t))

    print(f"\n=== {path.rsplit('/', 1)[-1]} ===")
    print(f"recorded E2O {len(log.e2o)}")
    print(f"  unrestricted expansion: {len(allc)} tuples over {len(cell_all)} cells")
    print(f"  interval-admissible   : {len(adm)} tuples over {len(cell_adm)} cells "
          f"({100.0 * len(adm) / max(1, len(allc)):.1f}% of tuples, "
          f"{100.0 * len(cell_adm) / max(1, len(cell_all)):.1f}% of cells)")
    blocked = sorted(cell_all - cell_adm)
    if blocked:
        print(f"  cells blocked entirely: {blocked}")

    for label, rel in (("recorded", list(log.e2o)),
                       ("admissible", list(log.e2o) + sorted(adm)),
                       ("unrestricted", list(log.e2o) + sorted(allc))):
        order, never, acts = facts(log, rel)
        clo = closure(log, rel)
        to = sum(len(v) for v in order.values())
        tc = sum(len(v) for v in clo.values())
        cross = sum(1 for t in clo for x, y in clo[t]
                    if family(x) != family(y))
        print(f"  {label:<13} closure {tc:>5}, {cross:>4} cross-family, "
              f"covering {to:>4}, role {sum(len(v) for v in never.values()):>3}")
        for t in sorted(order):
            xf = sum(1 for x, y in clo.get(t, ()) if family(x) != family(y))
            print(f"      {t:<14} {len(acts[t]):>3} acts  "
                  f"closure {len(clo.get(t, ())):>4} ({xf} cross)  "
                  f"covering {len(order[t]):>4}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
