#!/usr/bin/env python3
"""Ordered pairs against object-level support, ours and DF2's.

Prop. carried licenses an ordering through a RELATED pair of objects: an O2O
edge in either direction, or co-participation at one event. A pair (x, y) is
*supported* when some related pair (o, t) has o's first x before t's last y.
That is the weakest reading of the proposition's premise, so a pair failing
it is asserted by no object pair anywhere in the log.

Ours: the recorded per-type orderings closed transitively (`chained.py`, the
closure Prop. carried licenses). DF2: strict reachability over the
divergence-free graph of van Detten et al. (ICPM 2024), one accumulated
graph over all types, where a chain may pass through a shared activity node
with no object-level licence. The asymmetry in the unsupported column is the
point: our closure chains through the O2O relation, so its pairs stay
recheckable in the reduced log.

Usage: df2_support.py <log.xml> ...
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from fastdrawn import Bounds  # noqa: E402
from chained import transitive  # noqa: E402
from df2_baseline import df2_edges  # noqa: E402
from df2_orderings import strict_reach  # noqa: E402


def spans(log):
    per = defaultdict(list)
    for e, o in log.e2o:
        per[o].append(e)
    first, last = {}, {}
    for o, evs in per.items():
        evs.sort(key=lambda e: (log.time.get(e, ""), e))
        f, l = {}, {}
        for e in evs:
            a = log.act[e]
            f.setdefault(a, log.time.get(e, ""))
            l[a] = log.time.get(e, "")
        first[o], last[o] = f, l
    return first, last


def related(log):
    rel = defaultdict(set)
    for a, b, *_ in log.o2o:
        rel[a].add(b)
        rel[b].add(a)
    ev = defaultdict(set)
    for e, o in log.e2o:
        ev[e].add(o)
    for objs in ev.values():
        for o in objs:
            rel[o] |= objs
    return rel


def supported_pairs(log):
    first, last = spans(log)
    rel = related(log)
    out = set()
    for o, f in first.items():
        for t in rel.get(o, ()):
            lt = last.get(t)
            if not lt:
                continue
            for x, fx in f.items():
                for y, ly in lt.items():
                    if x != y and fx < ly:
                        out.add((x, y))
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    tot_ours = tot_ours_un = tot_d2 = tot_d2_un = 0
    for p in sys.argv[1:]:
        log = parse(p)
        b = Bounds(log, [(e, o) for e, o in log.e2o])
        ours = transitive(b.drawn(log.cells()))
        d2 = strict_reach(df2_edges(log))
        sup = supported_pairs(log)
        d2_only = d2 - ours
        n = p.rsplit("/", 1)[-1]
        print(f"{n:<26} ours {len(ours):>4} unsupported {len(ours - sup):>3}"
              f"   DF2-only {len(d2_only):>3} unsupported {len(d2_only - sup):>3}")
        tot_ours += len(ours)
        tot_ours_un += len(ours - sup)
        tot_d2 += len(d2_only)
        tot_d2_un += len(d2_only - sup)
    print(f"{'TOTAL':<26} ours {tot_ours:>4} unsupported {tot_ours_un:>3}"
          f"   DF2-only {tot_d2:>3} unsupported {tot_d2_un:>3}")


if __name__ == "__main__":
    main()
