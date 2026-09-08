#!/usr/bin/env python3
"""drawn() with the keep-set-independent work hoisted out.

The per-object, per-activity earliest/latest timestamps are a function of the
log alone, not of the keep-set, but drawn() recomputed them over every tuple on
every call -- 3.8M tuples, ~7s, several hundred times per search on BPIC2017.
Precompute once, then a call is a filter over the ~100k (object, activity)
entries.
"""
from __future__ import annotations
from collections import defaultdict


class Bounds:
    def __init__(self, log, e2o):
        b: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
        for e, o in e2o:
            a, t = log.act[e], log.time.get(e, "")
            cur = b[o].get(a)
            b[o][a] = (t, t) if cur is None else (min(cur[0], t), max(cur[1], t))
        self.per_type: dict[str, list[tuple[str, dict[str, tuple[str, str]]]]]
        self.per_type = defaultdict(list)
        for o, per in b.items():
            self.per_type[log.obj_type[o]].append((o, per))

    def drawn(self, keep):
        """Strict orderings each type draws, restricted to its kept cells."""
        acts_of: dict[str, set[str]] = defaultdict(set)
        for a, t in keep:
            acts_of[t].add(a)
        out: set[tuple[str, str]] = set()
        for t, aa in acts_of.items():
            ef: set[tuple[str, str]] = set()
            for _o, per in self.per_type.get(t, ()):
                items = [(a, v) for a, v in per.items() if a in aa]
                for x, (xmin, _) in items:
                    for y, (_, ymax) in items:
                        if x != y and xmin < ymax:
                            ef.add((x, y))
            out |= {(x, y) for x, y in ef if (y, x) not in ef}
        return out
