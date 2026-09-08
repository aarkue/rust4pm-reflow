#!/usr/bin/env python3
"""Write an expanded log as OCEL 2.0 XML, so other tools can discover from it.

The delta CSV is enough to check a measurement and useless for running a
discovery algorithm. This emits a real log: the source XML with the
materialised participations inserted, each carrying the qualifier
``expanded`` so they can be filtered back out or counted without re-deriving
them.

Three levels, and the difference matters for what you discover from them:

  full       every determined participation. Use this for anything that reads
             *directly*-follows -- OC-DFG arcs, OC-DECLARE's DF and DP arc
             types -- because adjacency depends on the events in between and
             is not preserved by any truncation.
  guided     only the cells that assert an ordering the model did not already
             have, greedy on orderings gained per tuple written. On Order
             Management that is 11 of 26 cells and 40% of the tuples for the
             same model; on BPIC2017 every cell qualifies, so it coincides with
             `full`.
  firstlast  first and last per (object, activity). Equivalent to `full` for
             *eventually*-follows only (\\autoref{sec:calculus}), so it is the
             right deposit for the fact measurements and the wrong input for a
             directly-follows model.

Usage: write_expanded_ocel.py <log.xml> <out.xml> [--full|--guided] [--max-depth=3]
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_probe import compose, qualified_maps  # noqa: E402


def added_pairs(log, depth, full, guided=False):
    if guided:
        from expansion_levels import guided_cells
        chosen, cand = guided_cells(log, depth)
        print(f"guided level: {len(chosen)} of {len(cand)} candidate cells")
        return {p for c in chosen for p in cand[c]}
    return _added_all(log, depth, full)


def _added_all(log, depth, full):
    by_type: dict[str, set[str]] = defaultdict(set)
    for oid, ot in log.obj_type.items():
        by_type[ot].add(oid)
    maps = compose(qualified_maps(log), by_type, depth)

    ev_objs: dict[str, set[str]] = defaultdict(set)
    for e, o in log.e2o:
        ev_objs[e].add(o)
    cells = log.cells()
    events_of: dict[str, list[str]] = defaultdict(list)
    for e, a in log.act.items():
        events_of[a].append(e)

    out: set[tuple[str, str]] = set()
    for (s, t, _q), m in maps.items():
        for a, evs in events_of.items():
            if (a, t) in cells:
                continue
            for e in evs:
                for o in ev_objs[e]:
                    if log.obj_type.get(o) != s:
                        continue
                    img = m.get(o)
                    if img is not None and img not in ev_objs[e]:
                        out.add((e, img))
    if full:
        return out
    per: dict[tuple[str, str], list[str]] = defaultdict(list)
    for e, o in out:
        per[(o, log.act[e])].append(e)
    keep = set()
    for (o, _a), evs in per.items():
        evs.sort(key=lambda e: log.time.get(e, ""))
        keep.add((evs[0], o))
        keep.add((evs[-1], o))
    return keep


def main() -> None:
    depth, full = 3, "--full" in sys.argv
    guided = "--guided" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--max-depth="):
            depth = int(a.split("=", 1)[1])
    src, dst = sys.argv[1], sys.argv[2]

    log = parse(src)
    add = added_pairs(log, depth, full, guided)
    per_event: dict[str, list[str]] = defaultdict(list)
    for e, o in add:
        per_event[e].append(o)
    print(f"{len(add)} participations to insert over {len(per_event)} events")

    text = open(src, encoding="utf-8").read()
    head, _, rest = text.partition("<events>")
    chunks = rest.split("<event ")
    written = 0
    with open(dst, "w", encoding="utf-8") as f:
        f.write(head + "<events>")
        f.write(chunks[0])
        for ch in chunks[1:]:
            eid = ch.split('id="', 1)[1].split('"', 1)[0]
            extra = per_event.get(eid)
            if extra:
                ins = "".join(f'<relationship object-id="{o}" '
                              f'qualifier="expanded"/>' for o in sorted(extra))
                # Every event closes its participation list exactly once.
                cut = ch.rindex("</objects>")
                ch = ch[:cut] + ins + ch[cut:]
                written += len(extra)
            f.write("<event " + ch)
    print(f"wrote {dst}: inserted {written} relationships "
          f"({'guided' if guided else 'full' if full else 'firstlast'} level)")
    if written != len(add):
        print(f"WARNING: {len(add) - written} not placed; event ids did not match")


if __name__ == "__main__":
    main()
