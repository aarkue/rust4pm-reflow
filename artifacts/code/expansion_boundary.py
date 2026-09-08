#!/usr/bin/env python3
"""First-and-last as a choice of CELLS, not of tuples.

The tuple-level truncation is incoherent: keeping an order's first and last
`create package` event leaves that activity holding a few orders and missing the
rest, so per-event involvement goes ragged and any technique reading
cardinalities is misled.

The fix is to apply the same first/last idea one level up. For each object, find
the activity where its expansion would *begin* and the one where it would *end*;
promote those whole cells, every participation in them. Involvement stays
uniform because a cell is all-or-nothing, and the temporal extremes that
eventually-follows depends on are exactly the ones retained.

Levels compared:

  boundary   cells that are first or last for some object of the type
  guided     greedy on orderings gained per tuple written
  full       every determined cell

Usage: expansion_boundary.py <log.xml> [--max-depth=3]
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_levels import candidate_cells, full_ef, facts_for, guided_cells  # noqa: E402


def boundary_cells(log, cand):
    """Cells at which some object's expanded participation starts or ends.

    Per object, over the candidate cells only: the activity of its earliest
    added participation and of its latest. Those two cells are what pin the
    object's expanded span, so keeping them whole preserves the endpoints that
    eventually-follows reads.
    """
    per_obj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (a, _t), pairs in cand.items():
        for e, o in pairs:
            per_obj[o].append((log.time.get(e, ""), a))
    keep: set[tuple[str, str]] = set()
    for o, evs in per_obj.items():
        evs.sort()
        keep.add((evs[0][1], log.obj_type[o]))
        keep.add((evs[-1][1], log.obj_type[o]))
    return {c for c in keep if c in cand}


def involvement(log, e2o):
    """min and max objects of a type per event, per activity."""
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e, o in e2o:
        per[e][log.obj_type[o]] += 1
    evs: dict[str, list[str]] = defaultdict(list)
    for e, a in log.act.items():
        evs[a].append(e)
    out = {}
    for a, es in evs.items():
        for t in {t for e in es for t in per[e]}:
            v = [per[e].get(t, 0) for e in es]
            out[(a, t)] = (min(v), max(v))
    return out


def main() -> None:
    depth = 3
    for a in sys.argv[1:]:
        if a.startswith("--max-depth="):
            depth = int(a.split("=", 1)[1])
    log = parse(sys.argv[1])
    base = list(log.e2o)
    cand = candidate_cells(log, depth)
    ef = full_ef(log, base + sorted({p for v in cand.values() for p in v}))
    rec = log.cells()
    b_clo, b_cov, _x = facts_for(ef, rec)

    bound = boundary_cells(log, cand)
    guided, _ = guided_cells(log, depth)

    print(f"recorded: {len(base)} tuples, {b_clo} orderings, {len(rec)} cells")
    print(f"{len(cand)} candidate cells\n")
    print(f"{'level':<10}{'cells':>7}{'tuples':>10}{'%rec':>7}"
          f"{'orderings':>11}{'ragged':>8}")
    for name, cells_ in (("boundary", bound), ("guided", set(guided)),
                         ("full", set(cand))):
        pairs = {p for c in cells_ for p in cand[c]}
        clo, _cov, _cr = facts_for(ef, rec | set(cells_))
        inv = involvement(log, base + sorted(pairs))
        ragged = sum(1 for v in inv.values() if v[0] == 0)
        print(f"{name:<10}{len(cells_):>7}{len(pairs):>10}"
              f"{100.0 * len(pairs) / len(base):>6.1f}%{clo:>11}{ragged:>8}")
    print(f"\nboundary cells: {sorted(bound)}")


if __name__ == "__main__":
    main()
