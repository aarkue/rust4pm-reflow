#!/usr/bin/env python3
"""Ground truth for BPIC2017 expansion: the original log's per-case order.

The conversion's recorded O2O map (Offer -> Application) IS the original
case notion, so joining each application's events with its offers' events
reproduces the original case-centric trace restricted to the A_/O_ events
the conversion keeps. Each added ordering (a pair the model orders only
after expansion) is checked against those joined traces under Def. orders
semantics: a case votes for (x, y) when its first x precedes its last y; a
pair is confirmed when at least one case votes for it and none votes for
the reverse.

Usage: expansion_ground_truth.py <log.xml> [theta]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from schema_maps import full_maps  # noqa: E402
from admissible_expansion import closure, lifetime  # noqa: E402


def admitted_cells(log, theta=0.5, depth=3):
    maps = full_maps(log, depth)
    life = lifetime(log)
    ev_objs = defaultdict(set)
    for e, o in log.e2o:
        ev_objs[e].add(o)
    cells = log.cells()
    events_of = defaultdict(list)
    for e, a in log.act.items():
        events_of[a].append(e)
    cand = defaultdict(set)
    alive = defaultdict(int)
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
                    if (e, img) in cand[(a, t)]:
                        continue
                    cand[(a, t)].add((e, img))
                    lo, hi = life.get(img, ("", ""))
                    if lo <= ts <= hi:
                        alive[(a, t)] += 1
    out = defaultdict(set)
    for c, prs in cand.items():
        if alive[c] / max(1, len(prs)) >= theta:
            out[c[1]] |= prs
    return out


def joined_case_votes(log):
    """Group each many-side object's events under its one-side partner, the
    original case (BPIC2017: each offer under its application)."""
    partners = defaultdict(set)
    for a, b, _q in log.o2o:
        if a in log.obj_type and b in log.obj_type:
            partners[a].add(b)
    app_of = {a: next(iter(bs)) for a, bs in partners.items() if len(bs) == 1}
    joined_types = {log.obj_type[a] for a in app_of} |                    {log.obj_type[b] for b in app_of.values()}
    case_events = defaultdict(set)
    for e, o in log.e2o:
        if log.obj_type.get(o) not in joined_types:
            continue
        case_events[app_of.get(o, o)].add(e)
    votes = set()
    for evs in case_events.values():
        first, last = {}, {}
        for e in sorted(evs, key=lambda e: (log.time.get(e, ""), e)):
            a = log.act[e]
            first.setdefault(a, log.time.get(e, ""))
            last[a] = log.time.get(e, "")
        for x, fx in first.items():
            for y, ly in last.items():
                if x != y and fx < ly:
                    votes.add((x, y))
    return votes


def main():
    path = sys.argv[1]
    theta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    log = parse(path)
    before = set()
    for v in closure(log, list(log.e2o)).values():
        before |= v
    admitted = admitted_cells(log, theta)
    added_tuples = [(e, o) for prs in admitted.values() for e, o in prs]
    after = set()
    for v in closure(log, list(log.e2o) + added_tuples).values():
        after |= v
    added = after - before
    votes = joined_case_votes(log)
    confirmed = {p for p in added if p in votes and (p[1], p[0]) not in votes}
    contradicted = {p for p in added if (p[1], p[0]) in votes and p not in votes}
    print(f"{path.rsplit('/', 1)[-1]}: ordered before {len(before)}, "
          f"after {len(after)}, added {len(added)}")
    print(f"confirmed by joined-case order: {len(confirmed)} of {len(added)}; "
          f"contradicted: {len(contradicted)}")
    for p in sorted(added - confirmed - contradicted):
        print(f"  unconfirmed (tied or unvoted): {p}")


if __name__ == "__main__":
    main()
