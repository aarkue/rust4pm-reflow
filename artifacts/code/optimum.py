#!/usr/bin/env python3
"""Coverage-minimal cell sets, and how named keep-sets score against them.

Objective: choose flow cells minimising participations, subject to

  coverage      every activity pair ordered in `max` is ordered by some flow
                type that is kept at both endpoints
  connectivity  the flow graph is one component

`max` is the theta-admissible saturation, a function of the equivalence class
rather than of how generously E2O happened to be recorded, so the objective
itself is class-invariant.

Greedy, not exact: feasibility factorises per activity but the objective does
not, since a pair couples two activities. Reported as a heuristic.

Usage: optimum.py <log.xml> [keepsets.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from schema_maps import full_maps  # noqa: E402
from admissible_expansion import closure, lifetime  # noqa: E402

THETA = 0.5


def saturate(log, depth: int = 3):
    """theta-admissible maximum: every determined cell whose target is alive at
    at least THETA of the activity's events, written for all its objects."""
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
    alive: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
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
                    seen[(a, t)].add(e)
                    lo, hi = life.get(img, ("", ""))
                    if lo <= ts <= hi:
                        alive[(a, t)].add(e)
    # per-TUPLE coverage, not per-event: per-event passes trivially when a cell
    # writes a set, which is what both_directions.py established.
    add: set[tuple[str, str]] = set()
    for c, prs in cand.items():
        ok = sum(1 for e, img in prs
                 if life.get(img, ("", ""))[0] <= log.time.get(e, "")
                 <= life.get(img, ("", ""))[1])
        if ok / max(1, len(prs)) >= THETA:
            add |= prs
    return sorted(set(log.e2o) | add)


def score(log, e2o, keep: set[tuple[str, str]], target: set[tuple[str, str]]):
    """participations, covered pairs, components, for a flow-cell set."""
    clo = closure(log, [(e, o) for e, o in e2o
                        if (log.act[e], log.obj_type[o]) in keep])
    covered = set()
    for t, prs in clo.items():
        acts = {a for (a, tt) in keep if tt == t}
        covered |= {p for p in prs if p[0] in acts and p[1] in acts}
    parts = sum(1 for e, o in e2o if (log.act[e], log.obj_type[o]) in keep)
    # components over activities, joined when a kept type sits at both
    parent: dict[str, str] = {a for (a, _t) in keep} and {a: a for (a, _t) in keep}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bytype: dict[str, list[str]] = defaultdict(list)
    for a, t in keep:
        bytype[t].append(a)
    for t, aa in bytype.items():
        for b in aa[1:]:
            ra, rb = find(aa[0]), find(b)
            if ra != rb:
                parent[ra] = rb
    comps = len({find(a) for a in parent})
    return parts, len(covered & target), comps


def greedy(log, e2o, target: set[tuple[str, str]]):
    """Cheapest flow cells covering `target`, then joined up."""
    clo = closure(log, e2o)
    cost: dict[tuple[str, str], int] = defaultdict(int)
    for e, o in e2o:
        cost[(log.act[e], log.obj_type[o])] += 1
    orders_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t, prs in clo.items():
        for p in prs:
            if p in target:
                orders_pair[p].append(t)

    keep: set[tuple[str, str]] = set()
    for p in sorted(target, key=lambda p: len(orders_pair[p])):
        if any((p[0], t) in keep and (p[1], t) in keep for t in orders_pair[p]):
            continue
        best, bc = None, None
        for t in orders_pair[p]:
            c = sum(cost[(x, t)] for x in p if (x, t) not in keep)
            if bc is None or c < bc:
                best, bc = t, c
        if best is not None:
            keep |= {(p[0], best), (p[1], best)}
    return keep


def run(path: str, keepsets: str | None = None) -> None:
    log = parse(path)
    mx = saturate(log)
    target: set[tuple[str, str]] = set()
    for v in closure(log, mx).values():
        target |= v

    print(f"\n=== {path.rsplit('/', 1)[-1]} ===")
    print(f"recorded E2O {len(log.e2o)}, max E2O {len(mx)}, "
          f"target pairs {len(target)}")
    print(f"{'candidate':<22}{'cells':>6}{'E2O':>10}{'cover':>12}{'comp':>6}")

    cands: dict[str, tuple[set, list]] = {
        "recorded (all flow)": (log.cells(), list(log.e2o)),
        "max (all flow)": ({(log.act[e], log.obj_type[o]) for e, o in mx}, mx),
    }
    if keepsets:
        for n, cs in json.load(open(keepsets)).items():
            if "arcguided" in n or "amortised" in n:
                continue
            cands[n] = ({tuple(c) for c in cs}, list(log.e2o))
    cands["greedy on max"] = (greedy(log, mx, target), mx)
    cands["greedy on recorded"] = (
        greedy(log, list(log.e2o), target), list(log.e2o))

    for name, (keep, rel) in cands.items():
        parts, cov, comps = score(log, rel, keep, target)
        print(f"{name:<22}{len(keep):>6}{parts:>10}"
              f"{cov:>7}/{len(target):<4}{comps:>6}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
