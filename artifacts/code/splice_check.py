#!/usr/bin/env python3
"""Independent check of the spliced directly-follows count, from the raw XML.

Demoting a cell drops that activity out of the type's trace, so two activities that
had something between them can become directly-follows and draw an arc the recorded
log never supported. `examples/eval_instruments.rs` reports zero such arcs on all
nine logs, which is the kind of clean number worth re-deriving from scratch. Nothing
here imports the Rust side: the log is parsed directly and the keep-set is read from
the exported `<log>.keepsets.json`.

Per object type, both traces are built from the same event order the reduction uses,
(timestamp, activity name, event id). The recorded arc set comes from the unprojected
traces, the reduced one from the traces projected onto the kept cells. An arc of the
reduced set whose two activities are both kept for that type and which the recorded
set lacks is spliced.

Usage: splice_check.py <log.xml|.gz> <keepsets.json> [...]
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402


def open_log(path):
    if not path.endswith(".gz"):
        return parse(path)
    with gzip.open(path, "rb") as src:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as dst:
            shutil.copyfileobj(src, dst)
            return parse(dst.name)


def df_arcs(traces, keep=None):
    """Directly-follows pairs per type, over traces projected onto `keep` if given."""
    arcs = set()
    for t, trs in traces.items():
        for tr in trs:
            seq = [a for a in tr if keep is None or (a, t) in keep]
            for x, y in zip(seq, seq[1:]):
                arcs.add((t, x, y))
    return arcs


def main(pairs):
    total_spliced = total_missing = 0
    print(f"{'log':<20} {'recorded':>9} {'reduced':>8} {'spliced':>8} {'missing':>8}")
    for log_path, keep_path in pairs:
        log = open_log(log_path)
        kept = {(a, t) for a, t in map(tuple, json.load(open(keep_path))["flow"])}

        per_object = defaultdict(list)
        for e, o in log.e2o:
            per_object[o].append(e)
        traces = defaultdict(list)
        for o, evs in per_object.items():
            evs.sort(key=lambda e: (log.time.get(e, ""), log.act[e], e))
            traces[log.obj_type[o]].append([log.act[e] for e in evs])

        recorded = df_arcs(traces)
        reduced = df_arcs(traces, kept)
        recorded_on_kept = {
            (t, x, y) for t, x, y in recorded if (x, t) in kept and (y, t) in kept
        }
        spliced = sorted(reduced - recorded_on_kept)
        missing = sorted(recorded_on_kept - reduced)
        total_spliced += len(spliced)
        total_missing += len(missing)

        name = log_path.rsplit("/", 1)[-1].replace(".xml", "").replace(".gz", "")
        print(
            f"{name:<20} {len(recorded):>9} {len(reduced):>8} "
            f"{len(spliced):>8} {len(missing):>8}"
        )
        for t, x, y in spliced:
            print(f"    spliced  {t}: {x} -> {y}")
        for t, x, y in missing:
            print(f"    missing  {t}: {x} -> {y}")

    print(f"\ncorpus: spliced {total_spliced}, missing {total_missing}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(list(zip(args[::2], args[1::2])))
