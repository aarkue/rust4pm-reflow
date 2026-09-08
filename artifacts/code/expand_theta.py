#!/usr/bin/env python3
"""Per-cell interval-admissible expansion at theta, and what it adds.

Two passes. First, per candidate cell, the fraction of the activity's events at
which at least one determined object is inside its own recorded lifetime.
Second, for every cell at or above theta, write ALL determined objects, marking
those outside their lifetime as extrapolated.

Per cell rather than per event because a per-event rule leaves cells filled at
some events of an activity and empty at others, which is the ragged-cell defect
docs/SPINE.md rejected when it ruled firstlast out of the operator.

Usage: expand_theta.py [--theta=0.5] <log.xml> ...
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from expansion_facts import facts  # noqa: E402
from schema_maps import full_maps  # noqa: E402
from admissible_expansion import closure, lifetime  # noqa: E402


def run(path: str, theta: float = 0.5, depth: int = 3) -> None:
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

    # pass 1: candidates per cell, deduplicated, plus which events are covered
    cand: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    ev_alive: dict[tuple[str, str], set[str]] = defaultdict(set)
    ev_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
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
                    cand[(a, t)].add((e, img))
                    ev_seen[(a, t)].add(e)
                    lo, hi = life.get(img, ("", ""))
                    if lo <= ts <= hi:
                        ev_alive[(a, t)].add(e)

    # pass 2: admit whole cells, write every candidate in them
    added: set[tuple[str, str]] = set()
    extrap = 0
    kept: list[tuple[str, str]] = []
    for c, pairs in cand.items():
        cov = len(ev_alive[c]) / max(1, len(ev_seen[c]))
        if cov < theta:
            continue
        kept.append(c)
        for e, img in pairs:
            added.add((e, img))
            lo, hi = life.get(img, ("", ""))
            if not (lo <= log.time.get(e, "") <= hi):
                extrap += 1

    per_type: dict[str, int] = defaultdict(int)
    for a, t in kept:
        per_type[t] += 1

    print(f"\n=== {path.rsplit('/', 1)[-1]}  theta={theta} ===")
    print(f"recorded E2O {len(log.e2o)}")
    print(f"cells admitted {len(kept)} of {len(cand)}: "
          f"{dict(sorted(per_type.items()))}")
    print(f"tuples added {len(added)} "
          f"({100.0 * len(added) / len(log.e2o):.1f}% of recorded), "
          f"of which extrapolated {extrap} "
          f"({100.0 * extrap / max(1, len(added)):.1f}%)")
    for label, rel in (("recorded", list(log.e2o)),
                       ("expanded", list(log.e2o) + sorted(added))):
        order, never, acts = facts(log, rel)
        clo = closure(log, rel)
        print(f"  {label:<9} closure {sum(len(v) for v in clo.values()):>5}  "
              f"covering {sum(len(v) for v in order.values()):>4}  "
              f"role {sum(len(v) for v in never.values()):>3}")


if __name__ == "__main__":
    th = 0.5
    paths = []
    for a in sys.argv[1:]:
        if a.startswith("--theta="):
            th = float(a.split("=", 1)[1])
        else:
            paths.append(a)
    for p in paths:
        run(p, th)
