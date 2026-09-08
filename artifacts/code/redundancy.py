#!/usr/bin/env python3
"""How much of a model is one object type redrawing another's picture?

Structural determinacy says a participation can be *recovered*. It does not say
the model would have shown the reader anything without it. Those are different
claims and the operator currently only checks the first.

This measures the second. For a keep-set, take each surviving type's
eventually-follows edges over the activities where its cell is kept. An edge is
\\emph{shared} when some other surviving type draws the same pair of activities,
and a type is \\emph{behaviourally redundant} when every edge it draws is shared:
deleting it removes ink and no information a reader could have read off the
picture.

Eventually-follows and not directly-follows, deliberately. Projection preserves
EF exactly and DF only soundly (`docs/LOSSLESSNESS.md`), so a DF-based count
includes edges that projection invented, and those would be scored as this
type's own contribution when they are an artifact.

Usage: redundancy.py <log.xml> [keepsets.json] [variant ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402


def ef_by_type(log, keep) -> dict[str, set[tuple[str, str]]]:
    """Eventually-follows pairs each type draws, restricted to its kept cells.

    Two events of one object at the *same* timestamp are unordered, and this
    must be said explicitly: sorting by (time, event id) invents an order from
    the identifiers. On Order Management that put `create package` before
    `pick item` for 66 items whose two events are simultaneous, which was
    enough to turn a 7,593-to-0 ordering into a parallel pair and to lose a
    real behavioural fact.
    """
    evs_of: dict[str, list[str]] = defaultdict(list)
    for e, o in log.e2o:
        evs_of[o].append(e)
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for o, evs in evs_of.items():
        t = log.obj_type[o]
        kept = [e for e in evs if (log.act[e], t) in keep]
        kept.sort(key=lambda e: log.time.get(e, ""))
        for i, e in enumerate(kept):
            for f in kept[i + 1:]:
                if log.time.get(e, "") < log.time.get(f, ""):
                    out[t].add((log.act[e], log.act[f]))
    return out


def report(log, name, keep) -> None:
    """Per type: how much its coloured subgraph constrains, and whether any of
    that constraint is its own.

    Density is load-bearing and an earlier version of this script omitted it,
    with the result that \\otProduct{} scored as the only informative type on
    the full model. Its EF graph is *complete* -- all 121 ordered pairs of the
    11 activities -- so it covers every other type's edges while permitting
    every order and therefore saying nothing. Coverage is not information, and
    a measure that counts shared edges without asking who is more constrained
    gets exactly the wrong answer.
    """
    ef = ef_by_type(log, keep)
    acts_of = {t: {a for (a, tt) in keep if tt == t} for t in ef}
    dens = {t: len(ef[t]) / max(1, len(acts_of[t]) ** 2) for t in ef}

    # An edge is a type's own only against types that are at least as
    # constrained. A permissive graph subsuming a strict one is not evidence
    # that the strict one is redundant; it is the other way round.
    total = sum(len(v) for v in ef.values())
    print(f"\n{name}: {len(keep)} cells, {len(ef)} types, "
          f"{total} coloured EF edges over "
          f"{len({e for v in ef.values() for e in v})} distinct activity pairs")
    print(f"{'type':<14}{'acts':>5}{'edges':>7}{'dens':>7}{'own':>6}  verdict")
    own = {}
    for t in sorted(ef, key=lambda t: dens[t]):
        rivals = [s for s in ef if s != t and dens[s] <= dens[t]]
        own[t] = sum(1 for e in ef[t] if not any(e in ef[s] for s in rivals))
        if dens[t] >= 1.0:
            verdict = "SAYS NOTHING: permits every order of its activities"
        elif own[t] == 0:
            verdict = "REDUNDANT: every edge drawn by a type at least as strict"
        else:
            verdict = f"carries {own[t]} of its own"
        print(f"{t:<14}{len(acts_of[t]):>5}{len(ef[t]):>7}{dens[t]:>7.2f}"
              f"{own[t]:>6}  {verdict}")

    dead = [t for t in ef if ef[t] and (dens[t] >= 1.0 or own[t] == 0)]
    ink = sum(len(ef[t]) for t in dead)
    print(f"  contributing nothing: {dead or 'none'}")
    print(f"  their edges: {ink} of {total} "
          f"({100.0 * ink / max(1, total):.0f}% of the model's ink)")


def main() -> None:
    log = parse(sys.argv[1])
    cells = log.cells()
    report(log, "full", cells)
    if len(sys.argv) > 2:
        keepsets = json.load(open(sys.argv[2]))
        names = sys.argv[3:] or sorted(keepsets)
        for n in names:
            report(log, n, {tuple(c) for c in keepsets[n]})


if __name__ == "__main__":
    main()
