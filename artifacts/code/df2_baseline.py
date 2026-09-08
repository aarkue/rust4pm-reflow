#!/usr/bin/env python3
"""DF2 as a baseline: which activity pairs does its graph still order?

Implements the divergence-free directly-follows graph of van Detten et al.
(ICPM 2024), Sec. V-A, from the paper's own definitions:

  Divergence of (a, ot): two events of a share their ot-objects but differ in
  the objects of some other type.
      exists (a,Oi),(aj,Oj) in L : ai = aj = a and Oi|ot = Oj|ot
                                   and exists ot' : Oi|ot' != Oj|ot'

  DF2 edge a -> b: some object o directly-follows a by b in its own trace and
  ot(o) is not in div_a intersect div_b.
      exists o : L|o = <.. a, b ..> and w(o) not in div_a cap div_b

The comparison the paper needs is not arc counts. DF2 accumulates one graph
over all types where the reduction keeps a per-type model, so arcs are not
comparable. Ordered activity pairs are: both techniques answer "which
orderings does the model still assert", and that is the unit Sec. 6 already
reports for the expansion direction.

Usage: df2_baseline.py [log.xml ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402


def objects_by_event(log) -> dict[str, dict[str, frozenset[str]]]:
    """event -> object type -> its objects at that event."""
    per: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for e, o in log.e2o:
        per[e][log.obj_type[o]].add(o)
    return {e: {t: frozenset(os) for t, os in d.items()} for e, d in per.items()}


def divergent(log, per_event) -> dict[str, set[str]]:
    """div_a: the types that do not distinguish two events of a."""
    by_act: dict[str, list[str]] = defaultdict(list)
    for e in per_event:
        by_act[log.act[e]].append(e)

    div: dict[str, set[str]] = defaultdict(set)
    for a, events in by_act.items():
        # div_a is a subset of the types related to a (DF2 Sec. V-B).
        types = {t for e in events for t in per_event[e]}
        # Group events by their restriction to ot; ot is divergent at a when
        # two events in one group differ on some other type.
        for ot in types:
            groups: dict[frozenset[str], list[str]] = defaultdict(list)
            for e in events:
                groups[per_event[e].get(ot, frozenset())].append(e)
            for group in groups.values():
                if len(group) < 2:
                    continue
                other = {t for e in group for t in per_event[e]} - {ot}
                if any(
                    per_event[ei].get(t, frozenset())
                    != per_event[ej].get(t, frozenset())
                    for t in other
                    for ei, ej in ((group[0], g) for g in group[1:])
                ):
                    div[a].add(ot)
                    break
    return div


def traces(log) -> dict[str, list[str]]:
    """object -> its events, in recorded time order."""
    per: dict[str, list[str]] = defaultdict(list)
    for e, o in log.e2o:
        per[o].append(e)
    for o in per:
        per[o].sort(key=lambda e: (log.time.get(e, ""), e))
    return per


def df2_edges(log) -> set[tuple[str, str]]:
    per_event = objects_by_event(log)
    div = divergent(log, per_event)
    edges: set[tuple[str, str]] = set()
    for o, trace in traces(log).items():
        ot = log.obj_type[o]
        for ei, ej in zip(trace, trace[1:]):
            a, b = log.act[ei], log.act[ej]
            if a == b:
                continue
            if ot not in (div[a] & div[b]):
                edges.add((a, b))
    return edges


def recorded_edges(log) -> set[tuple[str, str]]:
    """Every directly-follows pair any type carries, i.e. DF2 with no filter."""
    edges: set[tuple[str, str]] = set()
    for trace in traces(log).values():
        for ei, ej in zip(trace, trace[1:]):
            a, b = log.act[ei], log.act[ej]
            if a != b:
                edges.add((a, b))
    return edges


def report(path: str) -> dict:
    log = parse(path)
    per_event = objects_by_event(log)
    div = divergent(log, per_event)
    rec, d2 = recorded_edges(log), df2_edges(log)
    return {
        "log": path.rsplit("/", 1)[-1],
        "activities": len({log.act[e] for e in log.act}),
        "types": len(set(log.obj_type.values())),
        "divergent_cells": sum(len(v) for v in div.values()),
        "cells": len(log.cells()),
        "df_pairs_recorded": len(rec),
        "df_pairs_df2": len(d2),
        "dropped": len(rec - d2),
        "dropped_pct": round(100 * len(rec - d2) / len(rec), 1) if rec else 0.0,
    }


def main() -> None:
    paths = sys.argv[1:] or [str(Path(__file__).resolve().parents[1] / "logs" / "order-management.xml")]
    out = [report(p) for p in paths]
    for r in out:
        print(
            f"{r['log']:<34} pairs {r['df_pairs_recorded']:>4} -> "
            f"{r['df_pairs_df2']:>4}  ({r['dropped_pct']:>5}% dropped)  "
            f"div cells {r['divergent_cells']:>3}/{r['cells']}"
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
