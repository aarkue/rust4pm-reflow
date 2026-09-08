#!/usr/bin/env python3
"""Silence against divergence, cell by cell, over the whole corpus.

Divergence at (a,T) means two events of a share an object of T while differing
in other objects: a statement about how objects attach to events. Silence at
(a,T) means T orders no activity pair involving a: a statement about the
orderings T's own traces support. The two are asked of different things, so
this counts how often they actually differ.

Usage: silence_vs_divergence.py <log.xml> ...
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402


def traces_of(log, t):
    per = defaultdict(list)
    for e, o in log.e2o:
        if log.obj_type.get(o) == t:
            per[o].append(e)
    out = []
    for evs in per.values():
        evs.sort(key=lambda e: (log.time.get(e, ""), e))
        out.append([log.act[e] for e in evs])
    return out


def ordered_pairs(traces):
    """Def. orders: some object votes a before b, none votes back."""
    votes = defaultdict(lambda: [0, 0])
    for tr in traces:
        first, last = {}, {}
        for i, a in enumerate(tr):
            first.setdefault(a, i)
            last[a] = i
        for a in first:
            for b in first:
                if a == b:
                    continue
                if first[a] < last[b]:
                    votes[(a, b)][0] += 1
                if first[b] < last[a]:
                    votes[(a, b)][1] += 1
    return {p for p, (f, b) in votes.items() if f and not b}


def diverging(log):
    """Cells (a,T) where two events of a share a T object but differ elsewhere."""
    ev_objs = defaultdict(set)
    for e, o in log.e2o:
        ev_objs[e].add(o)
    by_act = defaultdict(list)
    for e, a in log.act.items():
        by_act[a].append(e)
    out = set()
    for a, evs in by_act.items():
        owner = defaultdict(list)
        for e in evs:
            for o in ev_objs[e]:
                owner[o].append(e)
        for o, es in owner.items():
            if len(es) < 2:
                continue
            t = log.obj_type.get(o)
            if (a, t) in out:
                continue
            if any(ev_objs[es[i]] != ev_objs[es[j]]
                   for i in range(len(es)) for j in range(i + 1, len(es))):
                out.add((a, t))
    return out


def run(path):
    log = parse(path)
    div = diverging(log)
    silent, cells = set(), set()
    for t in {x for x in log.obj_type.values()}:
        trs = traces_of(log, t)
        if not trs:
            continue
        acts = {a for tr in trs for a in tr}
        spoken = {a for p in ordered_pairs(trs) for a in p}
        for a in acts:
            cells.add((a, t))
            if a not in spoken:
                silent.add((a, t))
    ss, dd = silent, div & cells
    both = len(ss & dd)
    s_only = len(ss - dd)
    d_only = len(dd - ss)
    neither = len(cells) - both - s_only - d_only
    name = path.rsplit("/", 1)[-1]
    print(f"{name:<24} cells {len(cells):>3}   both {both:>3}   "
          f"silent-not-div {s_only:>3}   div-not-silent {d_only:>3}   "
          f"neither {neither:>3}")
    return {"log": name, "cells": len(cells), "both": both,
            "silent_only": s_only, "divergent_only": d_only, "neither": neither,
            "silent_only_cells": sorted(list(ss - dd))[:12],
            "divergent_only_cells": sorted(list(dd - ss))[:12]}


if __name__ == "__main__":
    out = [run(p) for p in sys.argv[1:]]
    tot = {k: sum(r[k] for r in out)
           for k in ("cells", "both", "silent_only", "divergent_only", "neither")}
    print(f"{'TOTAL':<24} cells {tot['cells']:>3}   both {tot['both']:>3}   "
          f"silent-not-div {tot['silent_only']:>3}   "
          f"div-not-silent {tot['divergent_only']:>3}   neither {tot['neither']:>3}")
    with open(Path(__file__).resolve().parents[1] / "results" / "stats" / "silence_vs_divergence.json", "w") as fh:
        json.dump({"per_log": out, "total": tot}, fh, indent=1)
