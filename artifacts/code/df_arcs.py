#!/usr/bin/env python3
"""Coloured directly-follows arcs of the OC-DFG a keep-set draws.

The paper's headline is model size, and the model is an OC-DFG, so the size is
ARCS. Everything else in this directory optimises participations (a log-size
measure) and reports arcs as an afterthought; this counts arcs directly.

Definition. For each object, take its events sorted by (timestamp, activity,
event id) and project onto the activities where (activity, the object's type)
is kept. Adjacent entries of the projection are directly-follows arcs. The arc
set of a type is the union over its objects; the model's size is the sum over
types, because the arcs are coloured -- the same activity pair drawn for two
types is two arcs.

SIMULTANEITY. The convention CHANGED here. This file used to treat an object's
trace as a sequence of timestamp GROUPS -- no arc inside a group, consecutive
groups joined pairwise -- on the grounds that simultaneous events are
unordered. That is right about ordering FACTS and wrong about arcs. A fact is a
claim about every object of a type, so a tie genuinely fails to establish one;
an arc is one linearisation of what the log recorded, and refusing to pick one
does not remove the pairs, it multiplies them. On BPIC2017 the cross-product
inflated the count from 363 to 604, because 21% of its (object, timestamp)
groups hold more than one event and the largest holds 53.

So: a plain sort, with the activity and then the event id breaking the tie.
Timestamp alone would leave the order to the parse and made the count vary
between runs (BPIC2017's resource type came out anywhere between 438 and 452
arcs); both tie-breakers are intrinsic to the log, so the graph is not. Same
key as the Rust implementation in schema_reduction::arcs.

Hoisting, as in fastdrawn.py: the traces are a function of the log alone, so
they are built once and a call is a projection over them. Objects with
identical traces are collapsed, since the arc set is a union and does not see
multiplicity.

Usage: df_arcs.py <log.xml> [keepsets.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402


class DfArcs:
    """Traces per object type, precomputed once per log."""

    def __init__(self, log) -> None:
        self.log = log
        per_obj: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for e, o in log.e2o:
            per_obj[o].append((log.time.get(e, ""), log.act[e], e))

        # type -> {trace: how many objects have it}
        self.traces: dict[str, dict[tuple[str, ...], int]] = defaultdict(dict)
        for o, evs in per_obj.items():
            t = log.obj_type[o]
            tr = tuple(a for _, a, _ in sorted(evs))
            d = self.traces[t]
            d[tr] = d.get(tr, 0) + 1

        self.parts: dict[tuple[str, str], int] = defaultdict(int)
        for e, o in log.e2o:
            self.parts[(log.act[e], log.obj_type[o])] += 1

    def arcs_per_type(self, keep, self_loops: bool = True
                      ) -> dict[str, set[tuple[str, str]]]:
        acts_of: dict[str, set[str]] = defaultdict(set)
        for a, t in keep:
            acts_of[t].add(a)
        out: dict[str, set[tuple[str, str]]] = {}
        for t, aa in acts_of.items():
            got: set[tuple[str, str]] = set()
            for tr in self.traces.get(t, ()):
                proj = [a for a in tr if a in aa]
                for x, y in zip(proj, proj[1:]):
                    if self_loops or x != y:
                        got.add((x, y))
            out[t] = got
        return out

    def arcs(self, keep, self_loops: bool = True) -> int:
        """Number of coloured DF arcs the keep-set draws."""
        return sum(len(v) for v in self.arcs_per_type(keep, self_loops).values())

    def merged_arcs(self, keep, self_loops: bool = True) -> int:
        """Arcs after forgetting colour: the uncoloured DF graph's size."""
        u: set[tuple[str, str]] = set()
        for v in self.arcs_per_type(keep, self_loops).values():
            u |= v
        return len(u)

    def participations(self, keep) -> int:
        return sum(n for c, n in self.parts.items() if c in keep)

    def frequency_per_type(self, keep) -> dict[str, dict[tuple[str, str], int]]:
        """Arc frequencies, for filter experiments: how many objects walk it."""
        acts_of: dict[str, set[str]] = defaultdict(set)
        for a, t in keep:
            acts_of[t].add(a)
        out: dict[str, dict[tuple[str, str], int]] = {}
        for t, aa in acts_of.items():
            freq: dict[tuple[str, str], int] = defaultdict(int)
            for tr, mult in self.traces.get(t, {}).items():
                proj = [a for a in tr if a in aa]
                seen = set(zip(proj, proj[1:]))
                for p in seen:
                    freq[p] += mult
            out[t] = dict(freq)
        return out


def candidates(log, keepsets: str | None):
    cands: dict[str, set[tuple[str, str]]] = {"full (recorded)": log.cells()}
    if keepsets:
        for n, cs in json.load(open(keepsets)).items():
            if "arcguided" in n or "amortised" in n:
                continue
            cands[n] = {tuple(c) for c in cs}
    return cands


def run(path: str, keepsets: str | None = None) -> None:
    log = parse(path)
    d = DfArcs(log)

    print(f"\n=== {path.rsplit('/', 1)[-1]} ===")
    print(f"{len(log.act)} events, {len(log.obj_type)} objects, "
          f"{len(log.e2o)} E2O, {len(log.cells())} cells, "
          f"{len(set(log.obj_type.values()))} types")
    print(f"{'keep-set':<18}{'cells':>6}{'parts':>9}{'arcs':>7}"
          f"{'no-loop':>9}{'merged':>8}  arcs per type")
    for name, keep in candidates(log, keepsets).items():
        per = d.arcs_per_type(keep)
        tot = sum(len(v) for v in per.values())
        detail = "  ".join(f"{t}:{len(v)}" for t, v in
                           sorted(per.items(), key=lambda kv: -len(kv[1])))
        print(f"{name:<18}{len(keep):>6}{d.participations(keep):>9}{tot:>7}"
              f"{d.arcs(keep, self_loops=False):>9}"
              f"{d.merged_arcs(keep):>8}  {detail}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
