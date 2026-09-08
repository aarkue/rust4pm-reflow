#!/usr/bin/env python3
"""What would structural expansion add to a log, and where?

Reduction and expansion are the two directions of one operator over the same
structural schema. Reduction removes a participation the schema already
determines; expansion materialises one the schema determines but the extraction
never recorded. Neither changes what the log knows, only where it is written.

For every total map f : S -> T the log carries, and every activity a whose
events name S-objects but no T-object, expansion adds the participation of
f(o) at each such event. This reports, per (activity, target type), how many
tuples that is, so the claim "reduction does nothing on this log but expansion
does a lot" is a measurement rather than an argument.

Usage: expansion_probe.py <log.xml|.sqlite|...> [--max-depth=2]
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402


def qualified_maps(log) -> dict[tuple[str, str, str], dict[str, str]]:
    """Total functions read off the O2O relation, per qualifier, both directions.

    A pair (source type, target type, qualifier) is a map only when every source
    object has exactly one image; anything else is a relation and cannot be used
    to reconstruct a participation.
    """
    fwd: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for a, b, q in log.o2o:
        ta, tb = log.obj_type.get(a), log.obj_type.get(b)
        if ta is None or tb is None:
            continue
        fwd[(ta, tb, q)][a].add(b)
        fwd[(tb, ta, f"~{q}")][b].add(a)

    out: dict[tuple[str, str, str], dict[str, str]] = {}
    by_type: dict[str, set[str]] = defaultdict(set)
    for oid, ot in log.obj_type.items():
        by_type[ot].add(oid)
    for key, images in fwd.items():
        st = key[0]
        if len(images) != len(by_type[st]):
            continue  # not total: some source object has no image
        if any(len(v) != 1 for v in images.values()):
            continue  # not functional
        out[key] = {k: next(iter(v)) for k, v in images.items()}
    return out


def compose(
    maps: dict[tuple[str, str, str], dict[str, str]],
    by_type: dict[str, set[str]],
    max_depth: int,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Close the map set under composition up to `max_depth` generators.

    A composite is kept only when it is again a TOTAL function on its source
    type, which is what licenses using it to reconstruct a participation. The
    qualifier of a composite is the dotted path, so two routes from S to U stay
    distinct and neither is silently preferred.
    """
    out = dict(maps)
    frontier = dict(maps)
    for _ in range(max_depth - 1):
        grown: dict[tuple[str, str, str], dict[str, str]] = {}
        for (s, t, q), f in frontier.items():
            for (t2, u, q2), g in maps.items():
                if t2 != t or u == s:
                    continue
                h = {o: g[i] for o, i in f.items() if i in g}
                if len(h) != len(by_type[s]):
                    continue
                key = (s, u, f"{q}.{q2}")
                if key not in out:
                    grown[key] = h
        if not grown:
            break
        out.update(grown)
        frontier = grown
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_depth = 1
    for a in sys.argv[1:]:
        if a.startswith("--max-depth="):
            max_depth = int(a.split("=", 1)[1])
    sys.argv = [sys.argv[0]] + args
    log = parse(sys.argv[1])

    by_type: dict[str, set[str]] = defaultdict(set)
    for oid, ot in log.obj_type.items():
        by_type[ot].add(oid)
    print(f"events {len(log.act)}  objects {len(log.obj_type)}  "
          f"types {len(by_type)}  E2O {len(log.e2o)}  O2O {len(log.o2o)}")
    for ot in sorted(by_type):
        print(f"    {ot:<24} {len(by_type[ot]):>8}")

    maps = qualified_maps(log)
    if max_depth > 1:
        before = len(maps)
        maps = compose(maps, by_type, max_depth)
        print(f"\ncomposition closure to depth {max_depth}: "
              f"{before} generators -> {len(maps)} maps")
    print(f"\ntotal maps (per qualifier, both directions): {len(maps)}")
    for (s, t, q), m in sorted(maps.items()):
        inj = len(set(m.values())) == len(m)
        print(f"    {s} -> {t}  [{q}]  {len(m)} objects"
              f"{'  (bijection)' if inj and len(m) == len(by_type[t]) else ''}")

    ev_objs: dict[str, set[str]] = defaultdict(set)
    for e, o in log.e2o:
        ev_objs[e].add(o)
    events_of: dict[str, list[str]] = defaultdict(list)
    for e, a in log.act.items():
        events_of[a].append(e)

    cells = log.cells()
    print(f"\ncells present: {len(cells)} of "
          f"{len(events_of)} activities x {len(by_type)} types")

    # Counted as a set of (event, object) pairs, not as a sum over maps: two
    # routes to the same object at the same event are one participation, and an
    # image the event already carries is not an addition. Summing per map
    # double-counts as soon as the closure has more than one route.
    print("\nexpansion, per (activity, target type) the log does not record:")
    per_cell: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    per_cell_via: dict[tuple[str, str], set[str]] = defaultdict(set)
    added_pairs: set[tuple[str, str]] = set()
    per_event: dict[str, set[str]] = defaultdict(set)
    for (s, t, q), m in maps.items():
        for e in events_of_type(events_of, cells, t):
            a = log.act[e]
            for o in ev_objs[e]:
                if log.obj_type.get(o) != s:
                    continue
                img = m.get(o)
                if img is None or img in ev_objs[e]:
                    continue
                per_cell[(a, t)].add((e, img))
                per_cell_via[(a, t)].add(f"{s} [{q}]")
                added_pairs.add((e, img))
                per_event[e].add(img)
    for (a, t), pairs in sorted(per_cell.items(), key=lambda kv: -len(kv[1])):
        via = sorted(per_cell_via[(a, t)])
        shown = ", ".join(via[:2]) + (f", +{len(via) - 2} more" if len(via) > 2 else "")
        print(f"    {a:<28} +{t:<20} {len(pairs)} tuples over "
              f"{len({e for e, _ in pairs})}/{len(events_of[a])} events  via {shown}")
    hist: dict[int, int] = defaultdict(int)
    for e in log.act:
        hist[len(per_event.get(e, ()))] += 1
    print("\n  added participations per event: " +
          ", ".join(f"{k}->{v}" for k, v in sorted(hist.items())))
    total_added = len(added_pairs)
    print(f"\ntuples expansion would add: {total_added} "
          f"against {len(log.e2o)} recorded ({100.0 * total_added / max(1, len(log.e2o)):.1f}%)")

    # E8, the direction rule. Two activities are relatable by a discovery
    # algorithm that reads participations when some object participates in
    # events of both. Components of that graph say whether the log's activity
    # families are already joined; if they are not, only expansion can join
    # them, and if they are, expansion duplicates a path that exists.
    def components(pairs: set[tuple[str, str]]) -> tuple[int, dict[str, int]]:
        parent: dict[str, str] = {a: a for a in events_of}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        obj_acts: dict[str, set[str]] = defaultdict(set)
        for e, o in pairs:
            obj_acts[o].add(log.act[e])
        for acts in obj_acts.values():
            it = iter(acts)
            first = find(next(it))
            for a in it:
                r = find(a)
                if r != first:
                    parent[r] = first
        sizes: dict[str, int] = defaultdict(int)
        for a in events_of:
            sizes[find(a)] += 1
        return len(sizes), dict(sizes)

    # Does the schema license INVOLVEMENT, or only IDENTITY? Determinacy says
    # which object would be named if one were named; it does not say the object
    # took part. The test: an added tuple that antedates its own object is not a
    # participation the log forgot, it is one that could not have happened.
    first_seen: dict[str, str] = {}
    for e, o in log.e2o:
        t = log.time.get(e, "")
        if o not in first_seen or t < first_seen[o]:
            first_seen[o] = t
    antedated = [(e, o) for e, o in added_pairs
                 if o in first_seen and log.time.get(e, "") < first_seen[o]]
    print(f"\nadded tuples that precede their own object's first recorded event: "
          f"{len(antedated)} of {len(added_pairs)} "
          f"({100.0 * len(antedated) / max(1, len(added_pairs)):.1f}%)")
    by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for e, o in antedated:
        by_cell[(log.act[e], log.obj_type[o])] += 1
    for (a, t), n in sorted(by_cell.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {a:<28} +{t:<20} {n}")

    before, _ = components(set(log.e2o))
    after, _ = components(set(log.e2o) | added_pairs)
    print(f"\nE8 direction rule, all types: {before} components recorded -> "
          f"{after} after expansion, over {len(events_of)} activities")

    # The all-types version is not the criterion: a free resource joins every
    # activity while relating none of them usefully. BPIC2017's `Case_R` is 149
    # objects spread over 1,202,267 events, so it connects the graph and says
    # nothing. Restrict to types the schema actually mentions, which is also the
    # only part of the log either direction of the operator can touch.
    sch_types = {t for k in maps for t in (k[0], k[1])}
    sch = {(e, o) for e, o in log.e2o if log.obj_type.get(o) in sch_types}
    sch_add = {(e, o) for e, o in added_pairs if log.obj_type.get(o) in sch_types}
    b2, sizes_b = components(sch)
    a2, _ = components(sch | sch_add)
    print(f"  schema types only ({', '.join(sorted(sch_types))}): "
          f"{b2} components recorded -> {a2} after expansion")
    if b2 > 1:
        print(f"    component sizes recorded: {sorted(sizes_b.values(), reverse=True)}")
    print("  verdict: " + (
        "EXPAND -- schema-relevant families are disconnected as recorded"
        if b2 > a2 else
        "REDUCE -- already joined, so expansion only duplicates existing paths"))


def events_of_type(events_of, cells, t):
    """Events at activities where type `t` is not already recorded."""
    for a, evs in events_of.items():
        if (a, t) in cells:
            continue
        yield from evs


if __name__ == "__main__":
    main()
