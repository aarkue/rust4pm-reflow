#!/usr/bin/env python3
"""Validate BPIC2017 expansion-added orderings against the original case-centric XES.

The added pairs are what the expanded keep-set asserts beyond the recorded one (65 -> 122
kept-layer pairs, 57 added, written Application cells at the eight offer activities). Each
added pair is checked against the original BPI Challenge 2017 log: per case, first(a)
strictly before last(b), a counter-witnessed pair counts as contradicted. Strict-time
votes, matching facts.rs.

Usage: expansion_vs_original_xes.py <expanded.ocel.xml> <recorded.ocel.xml> \
           <keepsets.json> <original.xes.gz>
"""

from __future__ import annotations

import gzip
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402


def asserted_kept(log, kept):
    per = defaultdict(list)
    for e, o in log.e2o:
        per[o].append(e)
    votes = defaultdict(lambda: [0, 0])
    for o, evs in per.items():
        t = log.obj_type[o]
        first, last = {}, {}
        for e in evs:
            a, ts = log.act[e], log.time.get(e, "")
            if (a, t) not in kept:
                continue
            if a not in first or ts < first[a]:
                first[a] = ts
            if a not in last or ts > last[a]:
                last[a] = ts
        for a in first:
            for b in first:
                if a == b:
                    continue
                if first[a] < last[b]:
                    votes[(t, a, b)][0] += 1
                if first[b] < last[a]:
                    votes[(t, a, b)][1] += 1
    return {(a, b) for (t, a, b), (f, w) in votes.items() if f and not w}


def main(expanded_path, recorded_path, keepsets_path, xes_path):
    keep = json.load(open(keepsets_path))
    kept = {(a, t) for a, t in map(tuple, keep["flow"])}

    exp = parse(expanded_path)
    written = {
        (exp.act[e], "Application")
        for e, o in exp.e2o
        if exp.obj_type.get(o) == "Application" and exp.act[e].startswith("O_")
    }
    exp_pairs = asserted_kept(exp, kept | written)
    rec_pairs = asserted_kept(parse(recorded_path), kept)
    added = sorted(exp_pairs - rec_pairs)
    print(f"kept-layer pairs recorded {len(rec_pairs)}, expanded {len(exp_pairs)}, added {len(added)}")

    acts_needed = {a for p in added for a in p}
    votes = {p: [0, 0] for p in added}
    n_cases = 0
    opener = gzip.open if xes_path.endswith(".gz") else open
    with opener(xes_path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag.split("}")[-1] != "trace":
                continue
            n_cases += 1
            first, last = {}, {}
            for ev in el:
                if ev.tag.split("}")[-1] != "event":
                    continue
                name = time = None
                for attr in ev:
                    k = attr.get("key")
                    if k == "concept:name":
                        name = attr.get("value")
                    elif k == "time:timestamp":
                        time = attr.get("value")
                if name in acts_needed and time:
                    if name not in first or time < first[name]:
                        first[name] = time
                    if name not in last or time > last[name]:
                        last[name] = time
            for a, b in added:
                if a in first and b in first:
                    if first[a] < last[b]:
                        votes[(a, b)][0] += 1
                    if first[b] < last[a]:
                        votes[(a, b)][1] += 1
            el.clear()

    confirmed = [p for p, (f, w) in votes.items() if f and not w]
    contradicted = [p for p, (f, w) in votes.items() if w]
    unwitnessed = [p for p, (f, w) in votes.items() if not f and not w]
    print(f"cases {n_cases}: confirmed {len(confirmed)}, contradicted {len(contradicted)}, unwitnessed {len(unwitnessed)}")
    for p in contradicted:
        print("  contradicted:", p, votes[p])
    for p in unwitnessed:
        print("  unwitnessed:", p)


if __name__ == "__main__":
    main(*sys.argv[1:5])
