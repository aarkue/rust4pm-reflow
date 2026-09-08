#!/usr/bin/env python3
"""Export an expanded log and the OC-DFGs discovered before and after it.

The expansion measurements in \\autoref{sec:evaluation} were computed in memory
and left nothing on disk, so nothing could be checked or drawn. This writes:

  <tag>-expansion-firstlast.csv    the added participations, as a delta on the
                                   recorded log rather than a second copy of it
                                   -- 842k rows against a 288 MB source file
  <tag>-ocdfg-recorded.dot         coloured object-centric directly-follows
  <tag>-ocdfg-expanded.dot         graph, before and after

`firstlast` is the level \\autoref{sec:calculus} proves equivalent to full
expansion for eventually-follows, so the delta is the smallest thing that
reproduces the measured models.

Usage: export_expansion.py <log.xml> <tag> <outdir> [--max-depth=3] [--full]
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_probe import compose, qualified_maps  # noqa: E402

COLOR = {
    "items": "#e8590c", "employees": "#c2255c", "orders": "#1971c2",
    "packages": "#8a5a2b", "products": "#2f9e44", "customers": "#862e9c",
    "Application": "#1971c2", "Workflow": "#e8590c", "Offer": "#2f9e44",
    "Case_R": "#868e96",
}
PALETTE = ["#1971c2", "#e8590c", "#2f9e44", "#862e9c", "#c2255c", "#8a5a2b"]


def dfg(log, e2o):
    """Coloured directly-follows arcs, and the activities each type touches.

    Events of one object sharing a timestamp are CONCURRENT, not ordered. An
    earlier version sorted by timestamp alone and let the iteration order of a
    set break the ties, which made the arc counts depend on which process ran
    it: BPIC2017's resource type came out anywhere between 438 and 452 arcs
    across runs. Here the trace is a sequence of timestamp *groups*, no arc is
    drawn inside a group, and consecutive groups are connected pairwise. That is
    deterministic and it is the standard reading of simultaneous events.
    """
    evs: dict[str, list[str]] = defaultdict(list)
    for e, o in e2o:
        evs[o].append(e)
    arcs: dict[tuple[str, str, str], int] = defaultdict(int)
    acts: dict[str, set[str]] = defaultdict(set)
    for o, es in evs.items():
        t = log.obj_type[o]
        groups: dict[str, set[str]] = defaultdict(set)
        for e in es:
            groups[log.time.get(e, "")].add(log.act[e])
        stamps = sorted(groups)
        acts[t] |= {a for g in groups.values() for a in g}
        for s1, s2 in zip(stamps, stamps[1:]):
            for x in groups[s1]:
                for y in groups[s2]:
                    if x != y:
                        arcs[(t, x, y)] += 1
    return arcs, acts


def write_dot(path, arcs, acts, title):
    types = sorted(acts)
    col = {t: COLOR.get(t, PALETTE[i % len(PALETTE)]) for i, t in enumerate(types)}
    with open(path, "w") as f:
        f.write("digraph G {\n  rankdir=LR;\n  bgcolor=transparent;\n")
        f.write('  node [shape=box, style="rounded,filled", fillcolor=white, '
                'fontname="Helvetica", fontsize=10];\n')
        f.write('  edge [fontname="Helvetica", fontsize=8];\n')
        f.write(f'  labelloc="t"; label="{title}";\n')
        for a in sorted({a for v in acts.values() for a in v}):
            f.write(f'  "{a}";\n')
        for (t, x, y), n in sorted(arcs.items()):
            f.write(f'  "{x}" -> "{y}" [color="{col[t]}", penwidth=1.2, '
                    f'tooltip="{t} x{n}"];\n')
        f.write("}\n")


def main() -> None:
    depth, want_full = 3, "--full" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--max-depth="):
            depth = int(a.split("=", 1)[1])
    path, tag, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    log = parse(path)

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

    added: set[tuple[str, str]] = set()
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
                        added.add((e, img))

    if not want_full:
        per_oa: dict[tuple[str, str], list[str]] = defaultdict(list)
        for e, o in added:
            per_oa[(o, log.act[e])].append(e)
        keep = set()
        for (o, _a), evs in per_oa.items():
            evs.sort(key=lambda e: log.time.get(e, ""))
            keep.add((evs[0], o))
            keep.add((evs[-1], o))
        added = keep

    level = "full" if want_full else "firstlast"
    delta = f"{outdir}/{tag}-expansion-{level}.csv"
    with open(delta, "w") as f:
        f.write("event,object,activity,object_type\n")
        for e, o in sorted(added):
            f.write(f"{e},{o},{log.act[e]},{log.obj_type[o]}\n")
    print(f"{delta}: {len(added)} added participations "
          f"({100.0 * len(added) / len(log.e2o):.1f}% of recorded)")

    for name, rel in (("recorded", list(log.e2o)),
                      ("expanded", list(log.e2o) + sorted(added))):
        arcs, acts = dfg(log, rel)
        p = f"{outdir}/{tag}-ocdfg-{name}.dot"
        write_dot(p, arcs, acts, f"{tag} {name}")
        print(f"{p}: {len(arcs)} arcs, {len(acts)} types, "
              f"{len({a for v in acts.values() for a in v})} activities")


if __name__ == "__main__":
    main()
