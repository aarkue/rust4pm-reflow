#!/usr/bin/env python3
"""One JSON per log with every per-type quantity this repo can compute.

Written because the corpus table reported only object-weighted aggregates, so
the per-type minimum behind "fitness never falls" was not recoverable from the
deposited artifacts (see REVIEW-2026-08-24.md D).

Everything here comes from the log and, optionally, a keep-set. Fitness and
precision come from the Rust pipeline, so each type carries a `scores` slot
for that run to fill: merge with `--scores <file.json>`, whose shape is
{log: {type: {fitness, precision, floor, adjusted_precision}}}.

Usage:
  corpus_stats.py --out ../results/stats [--keepsets f.json --keepset handoff]
                  [--scores scores.json] log.xml [log.xml ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from df2_baseline import df2_edges, divergent, objects_by_event, recorded_edges, traces  # noqa: E402


def per_type(log, tr) -> dict:
    """Object counts, participations, activities and DF pairs, per type."""
    objs: dict[str, set[str]] = defaultdict(set)
    parts: dict[str, int] = defaultdict(int)
    acts: dict[str, set[str]] = defaultdict(set)
    for e, o in log.e2o:
        t = log.obj_type[o]
        objs[t].add(o)
        parts[t] += 1
        acts[t].add(log.act[e])

    pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for o, trace in tr.items():
        t = log.obj_type[o]
        for ei, ej in zip(trace, trace[1:]):
            a, b = log.act[ei], log.act[ej]
            if a != b:
                pairs[t].add((a, b))

    total_objs = sum(len(v) for v in objs.values())
    return {
        t: {
            "objects": len(objs[t]),
            "object_share": round(len(objs[t]) / total_objs, 6) if total_objs else 0.0,
            "participations": parts[t],
            "activities": len(acts[t]),
            "df_pairs": len(pairs[t]),
            "scores": None,
        }
        for t in sorted(objs)
    }


def multiplicity(log) -> dict:
    """Per cell, the max and mean objects of that type at one event."""
    ev: dict[tuple[str, str], int] = defaultdict(int)
    for e, o in log.e2o:
        ev[(e, log.obj_type[o])] += 1
    per: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (e, t), n in ev.items():
        per[(log.act[e], t)].append(n)
    return {
        f"{a}\t{t}": {"max": max(ns), "mean": round(sum(ns) / len(ns), 4), "events": len(ns)}
        for (a, t), ns in sorted(per.items())
    }


def projected_pairs(log, tr, keep: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Prop. order's projection: DF pairs an object still carries when its
    trace is restricted to the activities where its type flows."""
    out: set[tuple[str, str]] = set()
    for o, trace in tr.items():
        t = log.obj_type[o]
        kept = [e for e in trace if (log.act[e], t) in keep]
        for ei, ej in zip(kept, kept[1:]):
            a, b = log.act[ei], log.act[ej]
            if a != b:
                out.add((a, b))
    return out


def stats(path: str, keep: set[tuple[str, str]] | None, keep_name: str | None) -> dict:
    log = parse(path)
    tr = traces(log)
    pe = objects_by_event(log)
    div = divergent(log, pe)
    rec, d2 = recorded_edges(log), df2_edges(log)

    out = {
        "log": os.path.basename(path),
        "path": path,
        "events": len(log.act),
        "objects": len(log.obj_type),
        "participations": len(log.e2o),
        "activities": sorted({log.act[e] for e in log.act}),
        "cells": len(log.cells()),
        "types": per_type(log, tr),
        "multiplicity": multiplicity(log),
        "divergence": {a: sorted(ts) for a, ts in sorted(div.items()) if ts},
        "ordered_pairs": {
            "recorded": len(rec),
            "df2": len(d2),
            "df2_dropped": sorted(a + " -> " + b for a, b in rec - d2),
        },
    }
    if keep is not None:
        proj = projected_pairs(log, tr, keep)
        out["keepset"] = {
            "name": keep_name,
            "cells": len(keep),
            "ordered_pairs": len(proj),
            "lost_vs_recorded": sorted(a + " -> " + b for a, b in rec - proj),
            "kept_but_df2_drops": sorted(a + " -> " + b for a, b in proj - d2),
            "df2_keeps_but_lost": sorted(a + " -> " + b for a, b in d2 - proj),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--keepsets")
    ap.add_argument("--keepset", default="handoff")
    ap.add_argument("--scores", help="per-type fitness/precision from the Rust run")
    args = ap.parse_args()

    keepsets = json.load(open(args.keepsets)) if args.keepsets else {}
    scores = json.load(open(args.scores)) if args.scores else {}
    os.makedirs(args.out, exist_ok=True)

    for path in args.logs:
        keep = None
        if args.keepset in keepsets:
            cand = {(a, t) for a, t in keepsets[args.keepset]}
            # A keep-set belongs to one log; apply it only where it fits.
            if cand <= parse(path).cells():
                keep = cand
        s = stats(path, keep, args.keepset if keep else None)
        for t, v in scores.get(s["log"], {}).items():
            if t in s["types"]:
                s["types"][t]["scores"] = v
        if any(v["scores"] for v in s["types"].values()):
            scored = [v for v in s["types"].values() if v["scores"]]
            s["aggregates"] = {
                "min_fitness": min(v["scores"]["fitness"] for v in scored),
                "mean_precision_unweighted": round(
                    sum(v["scores"]["adjusted_precision"] for v in scored) / len(scored), 6
                ),
                "mean_precision_object_weighted": round(
                    sum(v["scores"]["adjusted_precision"] * v["objects"] for v in scored)
                    / sum(v["objects"] for v in scored),
                    6,
                ),
            }
        dest = os.path.join(args.out, s["log"].rsplit(".", 1)[0] + ".stats.json")
        json.dump(s, open(dest, "w"), indent=2)
        print(
            f"{s['log']:<34} types {len(s['types']):>2}  cells {s['cells']:>3}  "
            f"pairs rec {s['ordered_pairs']['recorded']:>3} "
            f"df2 {s['ordered_pairs']['df2']:>3}"
            + (f" ours {s['keepset']['ordered_pairs']:>3}" if keep else "")
            + f"  -> {dest}"
        )


if __name__ == "__main__":
    main()
