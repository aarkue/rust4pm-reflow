#!/usr/bin/env python3
"""The model-based behavioral abstraction, read as orderings rather than as a floor.

A flower is not the only model that asserts nothing: a fully concurrent block
asserts nothing either, and on this corpus almost every demoted type's mined
model beats its flower while still ordering no activity pair. So the model-side
instantiation asks the same question the log-side one asks -- which activity
pairs does this artifact order -- and reads the answer off the mined process
tree instead of off the object's trace.

A tree orders (a, b) when their lowest common ancestor is a sequence with a in
an earlier child, and no loop sits at or above that ancestor: a loop lets a
later iteration put b first, which is the same both-ways vote Def. orders
discards as parallel.

Usage: model_silence.py <log.xml> ...
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402

import pm4py  # noqa: E402
from pm4py.objects.log.obj import Event, EventLog, Trace  # noqa: E402
from pm4py.objects.process_tree.obj import Operator  # noqa: E402

NOISE = 0.2


def traces_of(log, t: str) -> list[list[str]]:
    per = defaultdict(list)
    for e, o in log.e2o:
        if log.obj_type.get(o) == t:
            per[o].append(e)
    out = []
    for evs in per.values():
        evs.sort(key=lambda e: (log.time.get(e, ""), e))
        out.append([log.act[e] for e in evs])
    return out


def tree_orders(node, in_loop: bool = False):
    """(labels below node, ordered pairs the tree enforces)."""
    if not node.children:
        return ({node.label} if node.label else set()), set()
    op = node.operator
    loop = in_loop or op == Operator.LOOP
    kids = [tree_orders(c, loop) for c in node.children]
    labels = set().union(*(k[0] for k in kids))
    pairs = set().union(*(k[1] for k in kids))
    if op == Operator.SEQUENCE and not loop:
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                pairs |= {(a, b) for a in kids[i][0] for b in kids[j][0]}
    return labels, pairs


def log_orders(traces: list[list[str]]) -> set[tuple[str, str]]:
    """Def. orders on the log: first(a) before last(b), no object voting back."""
    acts = sorted({a for tr in traces for a in tr})
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
    return {p for p, (fwd, bwd) in votes.items() if fwd and not bwd}


def run(path: str) -> None:
    log = parse(path)
    types = sorted({t for t in log.obj_type.values()})
    stem = path.rsplit("/", 1)[-1]
    print(f"\n=== {stem} ===")
    print(f"{'type':<24}{'acts':>5}{'log ord':>9}{'tree ord':>10}"
          f"{'log mute':>10}{'tree mute':>11}  cells disagreeing")
    tot = dis = 0
    for t in types:
        trs = traces_of(log, t)
        if not trs:
            continue
        acts = {a for tr in trs for a in tr}
        el = EventLog()
        for tr in trs:
            trace = Trace()
            for a in tr:
                trace.append(Event({"concept:name": a}))
            el.append(trace)
        tree = pm4py.discover_process_tree_inductive(el, noise_threshold=NOISE)
        _, tpairs = tree_orders(tree)
        lpairs = log_orders(trs)
        lmute = acts - {a for p in lpairs for a in p}
        tmute = acts - {a for p in tpairs for a in p}
        d = lmute ^ tmute
        tot += len(acts)
        dis += len(d)
        print(f"{t:<24}{len(acts):>5}{len(lpairs):>9}{len(tpairs):>10}"
              f"{len(lmute):>10}{len(tmute):>11}  "
              f"{sorted(d) if d else ''}")
    print(f"  cells {tot}, silence verdicts disagreeing {dis}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
