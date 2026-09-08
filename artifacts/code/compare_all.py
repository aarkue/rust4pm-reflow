#!/usr/bin/env python3
"""All search strategies under identical settings.

Earlier runs varied the coverage measure (drawn vs chained) and the tuple basis
(recorded vs max) between experiments, so their numbers were not comparable.
Everything here uses: chained coverage, cost and closures on `max`, full
delivery required.

Strategies:
  greedy            cheapest-first cover of the target pairs
  greedy+post       then drop unneeded cells, then coarsen where free
  handoff           one flow type per activity, second only where forced
  handoff+post      then the same post-pass
"""
from __future__ import annotations
import sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_handoff_export import parse            # noqa: E402
from admissible_expansion import closure           # noqa: E402
from optimum import saturate                       # noqa: E402
from chained import transitive, greedy_chained     # noqa: E402
from fastdrawn import Bounds                       # noqa: E402


def hoisted_drawn(log, rel):
    """drawn(keep), with the keep-set-independent bounds computed once.

    Held per run rather than memoised globally: the previous cache was keyed on
    id(rel), and CPython reuses the id of a freed list, so a second log in the
    same process could be scored against the first log's bounds.
    """
    b = Bounds(log, rel)
    return b.drawn


def post(dr, keep, target, base):
    changed = True
    while changed:
        changed = False
        for c in sorted(keep):
            if len(transitive(dr(keep - {c})) & target) >= base:
                keep = keep - {c}; changed = True; break
    return keep


def handoff(dr, target, rec, flow_ok, nobj, cells_mx, cost):
    keep = set()
    for a in sorted({x for x, _ in rec}):
        native = [t for t in flow_ok if (a, t) in rec]
        if native:
            keep.add((a, min(native, key=lambda t: (nobj[t], t))))
    while True:
        cov = transitive(dr(keep)) & target
        if len(cov) >= len(target):
            break
        best, gain = None, 0
        for c in sorted(cells_mx - keep):
            if c[1] not in flow_ok:
                continue
            g = len(transitive(dr(keep | {c})) & target) - len(cov)
            if g > gain or (g == gain and g > 0 and best
                            and (cost[c], c) < (cost[best], best)):
                best, gain = c, g
        if not best:
            break
        keep.add(best)
    return keep


def run(path):
    log = parse(path); mx = sorted(saturate(log))
    dr = hoisted_drawn(log, mx)
    target = set()
    for v in closure(log, mx).values():
        target |= v
    rec = log.cells()
    nobj = defaultdict(int)
    for _o, t in log.obj_type.items():
        nobj[t] += 1
    clo = closure(log, mx)
    acts_of = defaultdict(set)
    for e, o in mx:
        acts_of[log.obj_type[o]].add(log.act[e])
    flow_ok = {t for t, a in acts_of.items()
               if len(clo.get(t, ())) / max(1, len(a) * (len(a) - 1) // 2) >= 0.05}
    cells_mx = {(log.act[e], log.obj_type[o]) for e, o in mx}
    cost = defaultdict(int)
    for e, o in mx:
        cost[(log.act[e], log.obj_type[o])] += 1

    n = len(target)
    g = greedy_chained(log, mx, target,
                       lambda t, p, k, c, n: sum(c[(x, t)] for x in p
                                                 if (x, t) not in k),
                       drawn_fn=lambda _log, _rel, keep: dr(keep))
    h = handoff(dr, target, rec, flow_ok, nobj, cells_mx, cost)
    out = {"greedy": g, "greedy+post": post(dr, g, target, n),
           "handoff": h, "handoff+post": post(dr, h, target, n)}

    print(f"\n=== {path.rsplit('/',1)[-1]} ===  {n} target pairs, "
          f"recorded {len(rec)} cells / {len(log.e2o)} E2O")
    print(f"{'strategy':<14}{'cells':>6}{'E2O':>10}{'delivered':>11}   types")
    for name, k in out.items():
        cov = len(transitive(dr(k)) & target)
        per = defaultdict(int)
        for a, t in k:
            per[t] += 1
        print(f"{name:<14}{len(k):>6}{sum(cost[c] for c in k):>10}"
              f"{cov:>7}/{n:<4}"
              f"   {dict(sorted(per.items(), key=lambda kv: -kv[1]))}")


for p in sys.argv[1:]:
    t = time.time(); run(p); print(f"   ({time.time()-t:.0f}s)")
