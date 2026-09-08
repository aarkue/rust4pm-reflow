#!/usr/bin/env python3
"""Rank expansion candidates by what the MODEL learns, not by the type's score.

Expansion candidates are exactly the cells reduction would remove: a cell is a
candidate because a recorded type determines it, which is the demotion
condition. So no cell-local rule separates a log that needs expansion from one
that does not. The separator has to be model-level.

Measured here, per candidate type:

  novel pairs  -- ordering facts on activity pairs that NO type currently
                  orders. This is what the model actually learns.
  echo pairs   -- ordering facts on pairs some type already orders. Restatement.
  fan-out      -- events of the activity per distinct target object, i.e. how
                  far one object gets smeared. Convergence.

Usage: novelty.py <log.xml> ...
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from schema_maps import full_maps  # noqa: E402
from admissible_expansion import closure, lifetime  # noqa: E402


def pairs_of(log, e2o) -> set[tuple[str, str]]:
    """Activity pairs ordered by SOME type."""
    out: set[tuple[str, str]] = set()
    for v in closure(log, e2o).values():
        out |= v
    return out


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

    admitted: dict[str, set[tuple[str, str]]] = defaultdict(set)
    fanout: dict[str, list[float]] = defaultdict(list)
    for c, prs in cand.items():
        if len(ev_alive[c]) / max(1, len(ev_seen[c])) < theta:
            continue
        admitted[c[1]] |= prs
        objs = {img for _e, img in prs}
        fanout[c[1]].append(len(prs) / max(1, len(objs)))

    before = pairs_of(log, list(log.e2o))
    print(f"\n=== {path.rsplit('/', 1)[-1]} ===")
    print(f"activity pairs ordered by some type, before: {len(before)}")
    print(f"{'type':<20}{'cells':>6}{'tuples':>10}{'novel':>7}{'echo':>7}"
          f"{'fan-out':>9}")
    for t in sorted(admitted):
        after = pairs_of(log, list(log.e2o) + sorted(admitted[t]))
        new = after - before
        gained = after - before
        # attribute: of this type's own orderings after expansion, how many are
        # on pairs nothing ordered before
        own = closure(log, list(log.e2o) + sorted(admitted[t])).get(t, set())
        novel = len({p for p in own if p not in before})
        echo = len(own) - novel
        fo = sum(fanout[t]) / max(1, len(fanout[t]))
        print(f"{t:<20}{len(fanout[t]):>6}{len(admitted[t]):>10}"
              f"{novel:>7}{echo:>7}{fo:>9.2f}")
        if new:
            ex = sorted(gained)[:3]
            print(f"    e.g. {ex}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
