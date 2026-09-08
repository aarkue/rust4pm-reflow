#!/usr/bin/env python3
"""Coverage counting chained delivery, not only drawn.

A pair (x,y) is DRAWN when one kept type is at both endpoints and orders it.
It is CHAINED when kept types hand off: S orders (x,m), T orders (m,y), for
some activity m where both are kept. The reduced model is exact for
eventually-follows, so a chained pair is still readable off the picture -- the
draft already scores handoff as 10 drawn / 6 chained / 1 lost.

Drawn-only coverage forces a type that spans the whole process (items on Order
Management), which is exactly what rules handoff out. This checks whether
chained coverage lets the coarse types back in.

Usage: chained.py <log.xml> [keepsets.json]
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_handoff_export import parse       # noqa: E402
from admissible_expansion import closure      # noqa: E402
from optimum import saturate                  # noqa: E402


def drawn(log, rel, keep):
    clo = closure(log, [(e, o) for e, o in rel
                        if (log.act[e], log.obj_type[o]) in keep])
    out = set()
    for t, prs in clo.items():
        acts = {a for (a, tt) in keep if tt == t}
        out |= {p for p in prs if p[0] in acts and p[1] in acts}
    return out


def transitive(pairs):
    """Close the drawn relation under composition: the handoff chain."""
    out = set(pairs)
    succ = defaultdict(set)
    for x, y in out:
        succ[x].add(y)
    changed = True
    while changed:
        changed = False
        for x in list(succ):
            for m in list(succ[x]):
                for y in succ.get(m, ()):
                    if (x, y) not in out:
                        out.add((x, y))
                        succ[x].add(y)
                        changed = True
    return out


def greedy_chained(log, rel, target, prefer, drawn_fn=drawn):
    """Cheapest cells until every target pair is drawn OR chained.

    `drawn_fn` is injectable so a caller holding precomputed bounds
    (compare_all.py) can pass its own without shadowing this module's global.
    """
    clo = closure(log, rel)
    ordersp = defaultdict(list)
    for t, prs in clo.items():
        for p in prs:
            if p in target:
                ordersp[p].append(t)
    cost = defaultdict(int)
    for e, o in rel:
        cost[(log.act[e], log.obj_type[o])] += 1
    nobj = defaultdict(int)
    for _o, t in log.obj_type.items():
        nobj[t] += 1
    keep = set()
    # Total orders throughout. `ordersp[p]` comes from dict iteration and
    # `target` from a set, so without the type name as a final key the result
    # depends on PYTHONHASHSEED: measured, handoff swung 31,755 to 37,870 on
    # Order Management across seeds.
    for p in sorted(target, key=lambda p: (len(ordersp[p]), p)):
        if p in transitive(drawn_fn(log, rel, keep)):
            continue
        cands = sorted(ordersp[p], key=lambda t: (prefer(t, p, keep, cost, nobj), t))
        if cands:
            keep |= {(p[0], cands[0]), (p[1], cands[0])}
    return keep


def run(path, ks=None):
    log = parse(path)
    mx = saturate(log)
    target = set()
    for v in closure(log, mx).values():
        target |= v
    cost = defaultdict(int)
    for e, o in log.e2o:
        cost[(log.act[e], log.obj_type[o])] += 1

    cands = {"recorded": log.cells()}
    if ks:
        for n, cs in json.load(open(ks)).items():
            if "arcguided" in n or "amortised" in n:
                continue
            cands[n] = {tuple(c) for c in cs}
    print(f"\n=== {path.rsplit('/',1)[-1]} ===  target {len(target)} pairs")
    print(f"{'keep-set':<16}{'cells':>6}{'E2O':>9}{'drawn':>8}{'+chained':>10}")
    for n, keep in cands.items():
        d = drawn(log, list(log.e2o), keep)
        c = transitive(d)
        parts = sum(1 for e, o in log.e2o
                    if (log.act[e], log.obj_type[o]) in keep)
        print(f"{n:<16}{len(keep):>6}{parts:>9}"
              f"{len(d & target):>5}/{len(target):<3}"
              f"{len(c & target):>7}/{len(target):<3}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
