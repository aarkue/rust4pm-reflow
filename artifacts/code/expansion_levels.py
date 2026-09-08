#!/usr/bin/env python3
"""Expansion is a choice of cells, exactly as reduction is, so give it levels.

The paper argues that reduction must decide per (activity, object type) and
criticises hand deletion for deciding per type. Expansion has been all-or-
nothing, which is the same mistake on the other side: `full` adds every
determined cell, and `firstlast` is a compression of `full` rather than a
different level.

The design space is the set of cells the schema determines and the log does not
record. This scores each one on its own and then builds two levels out of them:

  bridge   the fewest cells that connect the activity families a
           participation-reading algorithm sees as separate. The counterpart of
           `handoff`: buy the connection and nothing else.
  guided   greedy by ordering facts gained per tuple written, until the next
           cell buys nothing.

Both are compared against `full` on facts and on tuples, so the cost of the
all-or-nothing default is visible.

Usage: expansion_levels.py <log.xml> [--max-depth=3]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_probe import compose, qualified_maps  # noqa: E402


def candidate_cells(log, depth):
    """(activity, type) -> the participations expansion would add there."""
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

    out: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
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
                        out[(a, t)].add((e, img))
    return out


def full_ef(log, e2o):
    """Every eventually-follows pair, per type, over recorded AND addable cells.

    Whether `x` precedes `y` for object `o` depends only on `o`'s own events at
    those two activities, so it does not change as other cells are added. Compute
    the relation once here and restrict it to a cell subset later: evaluating a
    candidate set then costs O(activities^2) instead of a pass over the log,
    which is the difference between seconds and hours on BPIC2017.
    """
    bd = bounds(log, e2o)
    ef: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for o, per in bd.items():
        t = log.obj_type[o]
        for x, (xmin, _) in per.items():
            for y, (_, ymax) in per.items():
                if x != y and xmin < ymax:
                    ef[t].add((x, y))
    return ef


def facts_for(ef, cellset):
    """Ordering content of a cell set, as (closure size, covering size, cross).

    Two numbers, because they measure different things and only one of them is
    monotone. The *closure* is the set of strict orderings the model asserts,
    and it can only grow as cells are added. The *covering* set is the Hasse
    diagram, what a reader would be drawn, and it can SHRINK when a cell is
    added: a new edge can make three old ones implied. Using the covering size
    as a greedy objective therefore rewards withholding a cell, which is how a
    33-cell expansion scored above the 34-cell one on BPIC2017.
    """
    fam = lambda a: a.split("_", 1)[0] if "_" in a[:2] else a
    keep: dict[str, set[str]] = defaultdict(set)
    for a, t in cellset:
        keep[t].add(a)
    closure = total = cross = 0
    for t, aa in keep.items():
        e = {(x, y) for (x, y) in ef.get(t, ()) if x in aa and y in aa}
        so = {(x, y) for x, y in e if (y, x) not in e}
        red = {(x, y) for (x, y) in so
               if not any((x, m) in so and (m, y) in so for m in aa)}
        closure += len(so)
        total += len(red)
        cross += sum(1 for x, y in red if fam(x) != fam(y))
    return closure, total, cross


def bounds(log, e2o):
    out: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for e, o in e2o:
        a, t = log.act[e], log.time.get(e, "")
        cur = out[o].get(a)
        out[o][a] = (t, t) if cur is None else (min(cur[0], t), max(cur[1], t))
    return out


def order_facts(log, e2o):
    """Covering ordering pairs per type, and how many cross activity families."""
    bd = bounds(log, e2o)
    ef: dict[str, set[tuple[str, str]]] = defaultdict(set)
    acts: dict[str, set[str]] = defaultdict(set)
    for o, per in bd.items():
        t = log.obj_type[o]
        acts[t] |= set(per)
        for x, (xmin, _) in per.items():
            for y, (_, ymax) in per.items():
                if x != y and xmin < ymax:
                    ef[t].add((x, y))
    total, cross = 0, 0
    fam = lambda a: a.split("_", 1)[0] if "_" in a[:2] else a
    for t in acts:
        aa = sorted(acts[t])
        so = {(x, y) for x, y in ef[t] if (y, x) not in ef[t]}
        red = {(x, y) for (x, y) in so
               if not any((x, m) in so and (m, y) in so for m in aa)}
        total += len(red)
        cross += sum(1 for x, y in red if fam(x) != fam(y))
    return total, cross


def components(log, e2o, skip):
    """Activity families a participation-reading algorithm can relate."""
    obj_acts: dict[str, set[str]] = defaultdict(set)
    for e, o in e2o:
        if log.obj_type.get(o) in skip:
            continue
        obj_acts[o].add(log.act[e])
    acts = {a for v in obj_acts.values() for a in v}
    par = {a: a for a in acts}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for v in obj_acts.values():
        it = iter(sorted(v))
        first = find(next(it))
        for a in it:
            r = find(a)
            if r != first:
                par[r] = first
    return len({find(a) for a in acts})


def guided_cells(log, depth=3):
    """The cell set a greedy on orderings-gained-per-tuple selects.

    Factored out of main so `write_expanded_ocel.py` can materialise exactly
    this level rather than re-deriving it from a pasted list.
    """
    cand = candidate_cells(log, depth)
    base = list(log.e2o)
    ef = full_ef(log, base + sorted({p for v in cand.values() for p in v}))
    rec = log.cells()
    cur, chosen = set(rec), []
    cur_clo, _t, _c = facts_for(ef, rec)
    while True:
        best = None
        for c, pairs in cand.items():
            if c in chosen:
                continue
            clo, _t2, _c2 = facts_for(ef, cur | {c})
            gain = clo - cur_clo
            if gain <= 0:
                continue
            rate = gain / max(1, len(pairs))
            if best is None or rate > best[0]:
                best = (rate, c, clo)
        if best is None:
            break
        _r, c, clo = best
        chosen.append(c)
        cur.add(c)
        cur_clo = clo
    return chosen, cand


def main() -> None:
    depth = 3
    for a in sys.argv[1:]:
        if a.startswith("--max-depth="):
            depth = int(a.split("=", 1)[1])
    log = parse(sys.argv[1])
    base = list(log.e2o)

    # A type present at every activity relates them all while relating none of
    # them usefully, and would make every bridge look unnecessary.
    counts: dict[str, set[str]] = defaultdict(set)
    for e, o in log.e2o:
        counts[log.obj_type[o]].add(log.act[e])
    n_acts = len({a for a in log.act.values()})
    free = {t for t, v in counts.items()
            if len(v) == n_acts and len({o for o in log.obj_type
                                         if log.obj_type[o] == t}) < 500}
    cand = candidate_cells(log, depth)
    allpairs = base + sorted({p for v in cand.values() for p in v})
    ef = full_ef(log, allpairs)
    rec_cells = log.cells()
    b_clo, b_tot, b_cross = facts_for(ef, rec_cells)
    b_comp = components(log, base, free)
    print(f"recorded: {len(log.e2o)} tuples, {b_clo} orderings / {b_tot} drawn "
          f"({b_cross} cross-family), {b_comp} components"
          f"{' (ignoring free type ' + ','.join(sorted(free)) + ')' if free else ''}")
    print(f"{len(cand)} cells expansion could add\n")

    print(f"{'cell':<44}{'tuples':>9}{'+ord':>8}{'+cross':>8}")
    scored = []
    for c, pairs in cand.items():
        clo, tot, cross = facts_for(ef, rec_cells | {c})
        scored.append((c, pairs, clo - b_clo, cross - b_cross))
    for c, pairs, d, dx in sorted(scored, key=lambda r: (-r[2], len(r[1])))[:12]:
        print(f"{c[0] + ' / ' + c[1]:<44}{len(pairs):>9}{d:>+8}{dx:>+8}")
    dead = sum(1 for _c, _p, d, _x in scored if d == 0)
    print(f"  ... {dead} of {len(cand)} cells add no ordering fact at all")

    chosen: list[tuple[str, str]] = []
    cur = set(rec_cells)
    cur_clo = b_clo
    while True:
        best = None
        for c, pairs in cand.items():
            if c in chosen:
                continue
            clo, _t, _cr = facts_for(ef, cur | {c})
            gain = clo - cur_clo
            if gain <= 0:
                continue
            rate = gain / max(1, len(pairs))
            if best is None or rate > best[0]:
                best = (rate, c, clo)
        if best is None:
            break
        _r, c, clo = best
        chosen.append(c)
        cur.add(c)
        cur_clo = clo

    print(f"\n{'level':<10}{'cells':>7}{'tuples':>10}{'%rec':>7}"
          f"{'orderings':>11}{'drawn':>7}{'cross':>7}{'comps':>7}")
    for name, cells_ in (("guided", chosen), ("full", list(cand))):
        pairs = {p for c in cells_ for p in cand[c]}
        clo, tot, cross = facts_for(ef, rec_cells | set(cells_))
        comp = components(log, base + sorted(pairs), free)
        print(f"{name:<10}{len(cells_):>7}{len(pairs):>10}"
              f"{100.0 * len(pairs) / len(log.e2o):>6.1f}%{clo:>11}{tot:>7}{cross:>7}{comp:>7}")
    print(f"\nguided picks: {chosen}")


if __name__ == "__main__":
    main()
