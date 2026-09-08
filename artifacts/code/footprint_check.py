#!/usr/bin/env python3
"""Check a CPN ground-truth footprint against the log's lenses and the reduced flow layer.

For every enforced pair of the footprint: does the owning type order it under Def. orders
(strict timestamps, ties assert nothing, matching facts.rs), does IMf's mined tree enforce
it, is it shown by the recorded flow layer, and is it still shown by the reduced one
(drawn by a kept type, or by the chained closure). Route delivery is not credited here; an
uncovered pair is reported, not hidden.

Usage: footprint_check.py <footprint.json> <log.xml> <keepsets.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from model_silence import NOISE, tree_orders  # noqa: E402

import pm4py  # noqa: E402
from pm4py.objects.log.obj import Event, EventLog, Trace  # noqa: E402


def strict_orders(traces):
    """Def. orders with strict timestamp votes: tied events assert nothing."""
    votes = defaultdict(lambda: [0, 0])
    for tr in traces:
        first, last = {}, {}
        for a, ts in tr:
            if a not in first or ts < first[a]:
                first[a] = ts
            if a not in last or ts > last[a]:
                last[a] = ts
        for a in first:
            for b in first:
                if a == b:
                    continue
                if first[a] < last[b]:
                    votes[(a, b)][0] += 1
                if first[b] < last[a]:
                    votes[(a, b)][1] += 1
    return {p for p, (f, w) in votes.items() if f and not w}


def chain(pairs):
    out = set(pairs)
    while True:
        add = {(a, d) for (a, b) in out for (c, d) in out if b == c and a != d} - out
        if not add:
            return out
        out |= add


def main(fp_path, log_path, keep_path):
    fp = json.load(open(fp_path))
    keep = json.load(open(keep_path))
    kept = {(a, t) for a, t in map(tuple, keep["flow"])}

    log = parse(log_path)
    per = defaultdict(list)
    for e, o in log.e2o:
        per[o].append(e)
    full_traces, kept_traces = defaultdict(list), defaultdict(list)
    for o, evs in per.items():
        t = log.obj_type[o]
        evs.sort(key=lambda e: (log.time.get(e, ""), e))
        tr = [(log.act[e], log.time.get(e, "")) for e in evs]
        full_traces[t].append(tr)
        kept_traces[t].append([(a, ts) for a, ts in tr if (a, t) in kept])

    log_lens, imf_lens, drawn_full, drawn_kept = {}, {}, set(), set()
    for t, trs in full_traces.items():
        log_lens[t] = strict_orders(trs)
        drawn_full |= log_lens[t]
        drawn_kept |= strict_orders(kept_traces[t])
        el = EventLog()
        for tr in trs:
            trace = Trace()
            for a, _ in tr:
                trace.append(Event({"concept:name": a}))
            el.append(trace)
        tree = pm4py.discover_process_tree_inductive(el, noise_threshold=NOISE)
        imf_lens[t] = tree_orders(tree)[1]

    rec_shown, red_shown = chain(drawn_full), chain(drawn_kept)

    print(f"{'type':<10} {'pair':<42} {'log':>4} {'IMf':>4} {'recorded':>9} {'reduced':>9}")
    bad = 0
    for t, spec in fp["types"].items():
        for a, b in spec.get("enforced", []):
            p = (a, b)
            rec = "drawn" if p in drawn_full else ("chain" if p in rec_shown else "LOST")
            red = "drawn" if p in drawn_kept else ("chain" if p in red_shown else "LOST")
            bad += red == "LOST"
            print(
                f"{t:<10} {a + ' < ' + b:<42} "
                f"{'y' if p in log_lens.get(t, set()) else '-':>4} "
                f"{'y' if p in imf_lens.get(t, set()) else '-':>4} "
                f"{rec:>9} {red:>9}"
            )
    print(f"\nenforced pairs not shown by the reduced flow layer (before route credit): {bad}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
