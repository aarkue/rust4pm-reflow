#!/usr/bin/env python3
"""Which construct of the mined tree produced each model-side silence verdict.

Same discovery, same verdicts as model_silence.py -- this run only records, for
every operator node IMf builds, whether it came from a cut (xor, sequence,
concurrent, loop) or a fall-through (empty_traces, activity_once_per_trace,
activity_concurrent, strict_tau_loop, tau_loop, flower). A pair's verdict is
decided at the node that separates its two activities, except that a Loop node
at or above that separator decides instead: loop_cut, strict_tau_loop, tau_loop
and flower all insert a Loop and suppress orderings the same way.

Per cell, the report aggregates the deciding constructs of the pairs at that
activity, and for the log/model disagreements it lists each pair with its
construct and direction. The mined trees are asserted equal, pair for pair, to
what model_silence.py's tree_orders reads, so no verdict can drift.

Usage: model_silence_constructs.py <log.xml> ...
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_silence import NOISE, log_orders, traces_of, tree_orders  # noqa: E402
from verify_handoff_export import parse  # noqa: E402

import pm4py  # noqa: E402
from pm4py.objects.log.obj import Event, EventLog, Trace  # noqa: E402
from pm4py.objects.process_tree.obj import Operator  # noqa: E402

from pm4py.algo.discovery.inductive.cuts.factory import CutFactory  # noqa: E402
from pm4py.algo.discovery.inductive.fall_through.factory import FallThroughFactory  # noqa: E402
from pm4py.algo.discovery.inductive.fall_through.empty_traces import EmptyTracesUVCL  # noqa: E402

CONSTRUCT = {
    "ExclusiveChoiceCutUVCL": "xor_cut",
    "StrictSequenceCutUVCL": "sequence_cut",
    "ConcurrencyCutUVCL": "concurrent_cut",
    "LoopCutUVCL": "loop_cut",
    "EmptyTracesUVCL": "empty_traces",
    "ActivityOncePerTraceUVCL": "activity_once_per_trace",
    "ActivityConcurrentUVCL": "activity_concurrent",
    "StrictTauLoopUVCL": "strict_tau_loop",
    "TauLoopUVCL": "tau_loop",
    "FlowerModelUVCL": "flower",
}

LOOPY = {"loop_cut", "strict_tau_loop", "tau_loop", "flower"}


def _tagging_find_cut(cls, obj, inst, parameters=None):
    for c in CutFactory.get_cuts(obj, inst):
        r = c.apply(obj, parameters)
        if r is not None:
            r[0]._construct = CONSTRUCT.get(c.__name__, c.__name__)
            return r
    return None


def _tagging_fall_through(cls, obj, inst, pool, manager, parameters=None):
    for f in FallThroughFactory.get_fall_throughs(obj, inst):
        r = f.apply(obj, pool, manager, parameters)
        if r is not None:
            r[0]._construct = CONSTRUCT.get(f.__name__, f.__name__)
            return r
    return None


_ORIG_EMPTY_TRACES = EmptyTracesUVCL.apply


def _tagging_empty_traces(obj, pool=None, manager=None, parameters=None):
    r = _ORIG_EMPTY_TRACES(obj, pool, manager, parameters)
    if r is not None:
        r[0]._construct = "empty_traces"
    return r


CutFactory.find_cut = classmethod(_tagging_find_cut)
FallThroughFactory.fall_through = classmethod(_tagging_fall_through)
EmptyTracesUVCL.apply = staticmethod(_tagging_empty_traces)


def decisions_of(tree):
    """{(x, y) unordered pair as frozenset: [(construct, verdict)]}.

    verdict is ('ordered', (a, b)) for a sequence separation with no Loop at or
    above it, else ('unordered', construct). The deciding node of a separation
    under a Loop is the topmost such Loop.
    """
    out = defaultdict(list)

    def walk(node, top_loop):
        if node.operator == Operator.LOOP and top_loop is None:
            top_loop = node
        if not node.children:
            return {node.label} if node.label else set()
        kid_labels = [walk(c, top_loop) for c in node.children]
        for i in range(len(kid_labels)):
            for j in range(i + 1, len(kid_labels)):
                for x in kid_labels[i]:
                    for y in kid_labels[j]:
                        if x == y:
                            continue
                        decider = top_loop if top_loop is not None else node
                        construct = getattr(decider, "_construct", "untagged")
                        if (
                            top_loop is None
                            and node.operator == Operator.SEQUENCE
                        ):
                            out[frozenset((x, y))].append((construct, ("ordered", (x, y))))
                        else:
                            out[frozenset((x, y))].append((construct, ("unordered", None)))
        return set().union(*kid_labels) if kid_labels else set()

    walk(tree, None)
    return out


def ordered_from(decisions):
    pairs = set()
    for ds in decisions.values():
        for _, (kind, pair) in ds:
            if kind == "ordered":
                pairs.add(pair)
    return pairs


def pair_construct(decisions, labels, x, y):
    """The deciding constructs recorded for the activity pair {x, y}.

    A pair with no separating node means an endpoint never made it into the
    tree: the DFG noise filter of IMf's second iteration removed its every
    edge, so the cut projection dropped the activity.
    """
    ds = decisions.get(frozenset((x, y)))
    if ds is None:
        if x not in labels or y not in labels:
            return ["dropped_by_noise_filter"]
        return ["no_common_node"]
    return sorted({c for c, _ in ds})


def run(path: str):
    log = parse(path)
    types = sorted({t for t in log.obj_type.values()})
    stem = path.rsplit("/", 1)[-1]
    print(f"\n=== {stem} ===")
    report = {"log": stem, "cells": [], "disagreements": []}
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
        labels, tpairs = tree_orders(tree)
        decisions = decisions_of(tree)
        mine = ordered_from(decisions)
        assert mine == tpairs, f"{stem}/{t}: attribution disagrees with tree_orders"

        lpairs = log_orders(trs)
        tmute = acts - {a for p in tpairs for a in p}
        lmute = acts - {a for p in lpairs for a in p}

        for a in sorted(acts):
            verdict = "silent" if a in tmute else "asserting"
            if verdict == "asserting":
                cons = sorted(
                    {
                        c
                        for p in tpairs
                        if a in p
                        for c in pair_construct(decisions, labels, *p)
                    }
                )
            elif len(acts) == 1:
                cons = ["single_activity"]
            elif a not in labels:
                cons = ["dropped_by_noise_filter"]
            else:
                cons = sorted(
                    {
                        c
                        for x in acts
                        if x != a
                        for c in pair_construct(decisions, labels, a, x)
                    }
                )
            report["cells"].append(
                {"type": t, "activity": a, "verdict": verdict, "constructs": cons}
            )
            if (a in tmute) != (a in lmute):
                direction = (
                    "log orders, tree does not"
                    if a in tmute
                    else "tree orders, log does not"
                )
                if a in tmute:
                    pairs = [
                        {"pair": p, "constructs": pair_construct(decisions, labels, *p)}
                        for p in sorted(lpairs)
                        if a in p
                    ]
                else:
                    pairs = [
                        {"pair": p, "constructs": pair_construct(decisions, labels, *p)}
                        for p in sorted(tpairs - lpairs)
                        if a in p
                    ]
                report["disagreements"].append(
                    {
                        "type": t,
                        "activity": a,
                        "direction": direction,
                        "pairs": pairs,
                    }
                )
    n_sil = sum(1 for c in report["cells"] if c["verdict"] == "silent")
    print(
        f"  cells {len(report['cells'])}  silent {n_sil}  "
        f"disagreements {len(report['disagreements'])}"
    )
    for d in report["disagreements"]:
        print(f"  [{d['type']}] {d['activity']}: {d['direction']}")
        for p in d["pairs"]:
            print(f"      {p['pair'][0]} -> {p['pair'][1]}  via {', '.join(p['constructs'])}")
    return report


if __name__ == "__main__":
    reports = [run(p) for p in sys.argv[1:]]

    cells = [c for r in reports for c in r["cells"]]
    dis = [(r["log"], d) for r in reports for d in r["disagreements"]]
    print(f"\n=== corpus: {len(cells)} cells, {len(dis)} disagreements ===")

    def primary(c):
        cons = c["constructs"]
        loopy = [x for x in cons if x in LOOPY]
        return loopy[0] if loopy else (cons[0] if cons else "none")

    for verdict in ("asserting", "silent"):
        sub = Counter(primary(c) for c in cells if c["verdict"] == verdict)
        print(f"  {verdict}: " + ", ".join(f"{k} {v}" for k, v in sub.most_common()))

    mixed = [c for c in cells if len(c["constructs"]) > 1]
    print(f"  cells with more than one deciding construct: {len(mixed)}")

    by_family = Counter()
    for _, d in dis:
        cons = {c for p in d["pairs"] for c in p["constructs"]}
        if cons & {"strict_tau_loop", "tau_loop"}:
            fam = "tau_loop_fallthrough"
        elif "loop_cut" in cons:
            fam = "loop_cut"
        elif "flower" in cons:
            fam = "flower"
        elif "sequence_cut" in cons:
            fam = "sequence_cut"
        else:
            fam = "+".join(sorted(cons))
        by_family[(d["direction"], fam)] += 1
    print("  disagreements by direction and construct family:")
    for (direction, fam), n in sorted(by_family.items()):
        print(f"    {direction} | {fam}: {n}")

    out = Path(__file__).resolve().parents[1] / "results" / "stats" / "model_silence_constructs.json"
    with open(out, "w") as f:
        json.dump([{**r} for r in reports], f, indent=1, default=list)
    print(f"  written {out}")
