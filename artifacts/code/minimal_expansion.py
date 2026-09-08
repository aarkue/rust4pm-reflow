#!/usr/bin/env python3
"""Expansion does not have to be all-or-nothing: how few tuples buy the facts?

Full expansion materialises every participation the schema determines, which on
BPIC2017 is 1.4M tuples. Most of them are redundant for anything a model shows.
Eventually-follows between two activities depends only on the *earliest* and
*latest* time an object touches each of them, so materialising the first and
last occurrence per (object, activity) preserves the type-level EF relation
**exactly**, and every further tuple is invisible to it.

Variants, cheapest first:

  anchor     one tuple per materialised object, at its earliest event. The
             minimal repair: it places the object in the log and no more.
  first      earliest per (object, activity).
  firstlast  earliest and latest per (object, activity). Exact for EF.
  full       every determined participation.

Reported against the facts each buys, so the paper can say what a reader gets
per tuple written rather than only how many tuples exist.

Usage: minimal_expansion.py <log.xml> [--max-depth=3]
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_probe import compose, qualified_maps  # noqa: E402
from expansion_facts import facts, family  # noqa: E402


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

    # Bucket the additions so the variants are subsets of one another.
    per_oa: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e, o in added:
        per_oa[(o, log.act[e])].append(e)
    for k in per_oa:
        per_oa[k].sort(key=lambda e: log.time.get(e, ""))

    first = {(evs[0], o) for (o, _a), evs in per_oa.items()}
    firstlast = first | {(evs[-1], o) for (o, _a), evs in per_oa.items()}
    per_o: dict[str, list[str]] = defaultdict(list)
    for e, o in added:
        per_o[o].append(e)
    anchor = {(min(evs, key=lambda e: log.time.get(e, "")), o)
              for o, evs in per_o.items()}

    base_o, base_n, _ = facts(log, list(log.e2o))
    b_ord = sum(len(v) for v in base_o.values())
    b_nev = sum(len(v) for v in base_n.values())
    b_x = sum(1 for t in base_o for x, y in base_o[t] if family(x) != family(y))
    print(f"recorded {len(log.e2o)} tuples: {b_ord} ordering, {b_nev} "
          f"co-occurrence, {b_x} cross-family")
    print(f"\n{'variant':<11}{'added':>10}{'%rec':>7}{'ordering':>10}"
          f"{'cross':>7}{'co-occ':>8}{'facts/1k':>10}")
    for name, sub in (("anchor", anchor), ("first", first),
                      ("firstlast", firstlast), ("full", added)):
        o2, n2, _ = facts(log, list(log.e2o) + sorted(sub))
        no = sum(len(v) for v in o2.values())
        nn = sum(len(v) for v in n2.values())
        nx = sum(1 for t in o2 for x, y in o2[t] if family(x) != family(y))
        gain = (no - b_ord) + (nn - b_nev)
        print(f"{name:<11}{len(sub):>10}{100.0 * len(sub) / len(log.e2o):>6.1f}%"
              f"{no:>10}{nx:>7}{nn:>8}"
              f"{1000.0 * gain / max(1, len(sub)):>10.2f}")


if __name__ == "__main__":
    main()
