#!/usr/bin/env python3
"""Is the chained closure of the drawn relation contained in the ordering relation?

`chained.transitive` closes the drawn pairs under composition and the searches score
coverage on the result. That is the *completeness* direction: every pair some type orders
must be delivered. This asks the other one, which nothing in the operator checks -- what
does the closure deliver that no type orders? -- and it does it three ways:

  counterexamples  five synthetic OCEL 2.0 logs, written out so the Rust implementation
                   can be run on the same files (examples/chaining_soundness.rs);
  theorem check    the same-type claim, brute-forced over random small logs with
                   repetitions and ties, which is where the hand proof is easiest to fool;
  corpus           per log, the pairs the closure invents, classified.

Usage: chaining_soundness.py [--out DIR] [log.xml ...]
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_handoff_export import parse  # noqa: E402
from admissible_expansion import closure  # noqa: E402
from chained import drawn, transitive  # noqa: E402
from optimum import saturate  # noqa: E402

T0 = "2020-01-01T00:00:"


def ts(k: int) -> str:
    return f"{T0}{k:02d}"


# --------------------------------------------------------------------------
# Synthetic logs
# --------------------------------------------------------------------------

def write_log(path: str, objects, events, o2o=()) -> None:
    """objects: [(id, type)]; events: [(id, activity, time, [object ids])]."""
    otypes = sorted({t for _o, t in objects})
    acts = sorted({a for _e, a, _t, _os in events})
    L = ["<?xml version='1.0' encoding='UTF-8'?>", "<log>", "  <object-types>"]
    for t in otypes:
        L += [f'    <object-type name="{t}">', "      <attributes/>", "    </object-type>"]
    L += ["  </object-types>", "  <event-types>"]
    for a in acts:
        L += [f'    <event-type name="{a}">', "      <attributes/>", "    </event-type>"]
    L += ["  </event-types>", "  <objects>"]
    rel = defaultdict(list)
    for s, t, q in o2o:
        rel[s].append((t, q))
    for o, t in objects:
        L += [f'    <object id="{o}" type="{t}">', "      <attributes/>"]
        if rel[o]:
            L.append("      <objects>")
            for t2, q in rel[o]:
                L.append(f'        <relationship object-id="{t2}" qualifier="{q}"/>')
            L.append("      </objects>")
        L.append("    </object>")
    L += ["  </objects>", "  <events>"]
    for e, a, t, os in events:
        L += [f'    <event id="{e}" type="{a}" time="{t}">', "      <attributes/>",
              "      <objects>"]
        for o in os:
            L.append(f'        <relationship object-id="{o}" qualifier="p"/>')
        L += ["      </objects>", "    </event>"]
    L += ["  </events>", "</log>", ""]
    open(path, "w").write("\n".join(L))


def counterexamples(out: str, copies: int = 5) -> list[tuple[str, str]]:
    """The five shapes, each `copies` objects wide so map discovery has a domain."""
    made = []

    def build(name, plan, o2o_pairs=()):
        objects, events, o2o = [], [], []
        n = 0
        for k in range(copies):
            for oi, (otype, trace) in enumerate(plan):
                oid = f"{otype}{oi}_{k}"
                objects.append((oid, otype))
                for act, at in trace:
                    n += 1
                    events.append((f"e{n}", act, ts(at), [oid]))
            for i, j in o2o_pairs:
                o2o.append((f"{plan[i][0]}{i}_{k}", f"{plan[j][0]}{j}_{k}", "rel"))
        path = f"{out}/{name}.xml"
        write_log(path, objects, events, o2o)
        made.append((name, path))

    # 1. Same type, no object carries both endpoints: the chain asserts an order between
    #    two activities that share no object at all.
    build("ce1-same-type-vacuous",
          [("T", [("x", 1), ("m", 2)]), ("T", [("m", 3), ("y", 4)])])

    # 2. Same type, and a third object does y before x without m. The condition
    #    "every object carrying x and y also carries m" is exactly what this breaks.
    build("ce2-same-type-reversal",
          [("T", [("x", 1), ("m", 2)]), ("T", [("m", 3), ("y", 4)]),
           ("T", [("y", 5), ("x", 6)])])

    # 3. One type per hop, and a third type that orders the composed pair backwards. Each
    #    type on its own satisfies the same-type condition (vacuously), and the chain still
    #    reverses a fact.
    build("ce3-cross-type-reversal",
          [("A", [("x", 1), ("m", 2)]), ("B", [("m", 3), ("y", 4)]),
           ("C", [("y", 5), ("x", 6)])])

    # 4. The same with a recorded map A -> B, to see whether the schema rescues it.
    build("ce4-cross-type-mapped",
          [("A", [("x", 1), ("m", 2)]), ("B", [("m", 3), ("y", 4)]),
           ("C", [("y", 5), ("x", 6)])], o2o_pairs=[(0, 1)])

    # 5. Same type, condition satisfied, every three at one instant: the chain asserts an
    #    ordering the log ties.
    build("ce5-same-type-tie",
          [("T", [("x", 1), ("m", 2)]), ("T", [("m", 3), ("y", 4)]),
           ("T", [("x", 9), ("m", 9), ("y", 9)])])

    # 6. The positive case: the same cross-type chain, but B's objects are alive at `x`, so
    #    saturation writes (x, B) along the map and B orders (x, y) itself. The pair stops
    #    being a chain and becomes a drawn fact -- which is the point: where the schema
    #    makes the chain sound it also makes it unnecessary.
    build("ce6-cross-type-transported",
          [("A", [("x", 3), ("m", 4)]), ("B", [("w", 1), ("m", 5), ("y", 6)])],
          o2o_pairs=[(0, 1)])

    # 7 and 8. ce3's twins: the same types, the same cells, the same maps, the same hop
    # orderings, and only the third type's *timestamps* moved. Anything a condition on the
    # chain can read is identical across the three; what the chain asserts is a
    # contradiction in ce3, a restatement in ce7 and an over-claim in ce8.
    build("ce7-cross-type-twin-forward",
          [("A", [("x", 1), ("m", 2)]), ("B", [("m", 3), ("y", 4)]),
           ("C", [("x", 5), ("y", 6)])])
    build("ce8-cross-type-twin-tie",
          [("A", [("x", 1), ("m", 2)]), ("B", [("m", 3), ("y", 4)]),
           ("C", [("x", 5), ("y", 5)])])
    return made


# --------------------------------------------------------------------------
# The relations, straight off the definitions in admissible_expansion.closure
# --------------------------------------------------------------------------

def relations(log, rel):
    """(ef, ord) as unions over types, and the per-type ordering."""
    per = closure(log, rel)
    bd = defaultdict(dict)
    for e, o in rel:
        a, t = log.act[e], log.time.get(e, "")
        cur = bd[o].get(a)
        bd[o][a] = (t, t) if cur is None else (min(cur[0], t), max(cur[1], t))
    ef, co = set(), set()
    for o, p in bd.items():
        for x, (xmin, _) in p.items():
            for y, (_, ymax) in p.items():
                if x != y:
                    co.add((x, y))
                    if xmin < ymax:
                        ef.add((x, y))
    ordv = set()
    for v in per.values():
        ordv |= v
    return ef, co, ordv, per


def classify(delivered, ordv, ef, co):
    out = defaultdict(list)
    for x, y in sorted(delivered):
        if (x, y) in ordv:
            continue
        if x == y:
            out["self"].append((x, y))
        elif (y, x) in ordv:
            out["reversal"].append((x, y))
        elif (x, y) in ef:
            out["witnessed"].append((x, y))
        elif (x, y) in co:
            out["tie"].append((x, y))
        else:
            out["unwitnessed"].append((x, y))
    return out


CLASSES = ("self", "reversal", "witnessed", "tie", "unwitnessed")


def report(label, delivered, ordv, ef, co, show=0):
    c = classify(delivered, ordv, ef, co)
    tot = sum(len(v) for v in c.values())
    print(f"  {label:<28}|ord| {len(ordv):>5}  chained {len(delivered):>5}  "
          f"invented {tot:>5}  " +
          "  ".join(f"{k} {len(c.get(k, ())):>4}" for k in CLASSES))
    for k in CLASSES:
        for p in c.get(k, ())[:show]:
            print(f"      {k:<12} {p[0]} -> {p[1]}")
    return c


# --------------------------------------------------------------------------
# Same-type theorem, brute-forced
# --------------------------------------------------------------------------

def outcome(per):
    """(ef, co, ord) of a one-type log given as a list of activity -> times."""
    ef, co = set(), set()
    for p in per:
        for a, va in p.items():
            for b, vb in p.items():
                if a != b:
                    co.add((a, b))
                    if min(va) < max(vb):
                        ef.add((a, b))
    ordv = {(a, b) for (a, b) in ef if (b, a) not in ef}
    if ("x", "m") not in ordv or ("m", "y") not in ordv:
        return None
    cond = all("m" in p for p in per if "x" in p and "y" in p)
    return cond, ("ordered" if ("x", "y") in ordv else
                  "reversed" if ("y", "x") in ordv else
                  "tie" if ("x", "y") in co else "unwitnessed")


def exhaustive_same_type():
    """Every one-type log over three activities, three objects, times drawn from
    {absent, [0], [1], [2], [0,2]} per activity: 125 object shapes, 333,375 logs.

    Small, but it contains repetition and simultaneity, which is where the same-type
    argument is easiest to fool. The claim under test is that the condition rules out a
    reversal -- if a single log in this space reverses one, the condition is not sufficient.
    """
    vals = [None, [0], [1], [2], [0, 2]]
    shapes = [dict(p for p in zip("xmy", c) if p[1] is not None)
              for c in itertools.product(vals, repeat=3)]
    seen = defaultdict(int)
    bad = []
    for combo in itertools.combinations_with_replacement(shapes, 3):
        per = [p for p in combo if p]
        if not per:
            continue
        r = outcome(per)
        if r is None:
            continue
        seen[r] += 1
        if r == (True, "reversed") and len(bad) < 3:
            bad.append(per)
    print(f"\nsame-type exhaustive over {len(shapes)} shapes / "
          f"{sum(seen.values())} logs meeting the premises:")
    for k in sorted(seen):
        print(f"    {str(k):<28} {seen[k]}")
    print(f"  condition holds and the chain reverses a fact: {len(bad)} logs")
    return bad


def brute_force_chain(k=3, trials=200000, seed=13):
    """The same claim for a chain of `k` hops, one type: `a0 < a1 < ... < ak` all ordered,
    and every object carrying `a0` and `ak` carrying every intermediate.

    The multi-hop case does not follow from the two-hop one by induction -- the two-hop
    conclusion can be a tie, which cannot be chained further -- so it is tested directly.
    """
    rnd = random.Random(seed)
    acts = [f"a{i}" for i in range(k + 1)]
    seen = defaultdict(int)
    bad = []
    for _ in range(trials):
        per = []
        for _o in range(rnd.randint(1, 4)):
            has = [a for a in acts if rnd.random() < 0.6]
            if has:
                per.append({a: sorted(rnd.randint(0, 5) for _ in range(rnd.randint(1, 2)))
                            for a in has})
        if not per:
            continue
        ef, co = set(), set()
        for p in per:
            for a, va in p.items():
                for b, vb in p.items():
                    if a != b:
                        co.add((a, b))
                        if min(va) < max(vb):
                            ef.add((a, b))
        ordv = {(a, b) for (a, b) in ef if (b, a) not in ef}
        if any((acts[i], acts[i + 1]) not in ordv for i in range(k)):
            continue
        x, y = acts[0], acts[-1]
        cond = all(all(m in p for m in acts[1:-1])
                   for p in per if x in p and y in p)
        out = ("ordered" if (x, y) in ordv else
               "reversed" if (y, x) in ordv else
               "tie" if (x, y) in co else "unwitnessed")
        seen[(cond, out)] += 1
        if cond and out == "reversed" and len(bad) < 3:
            bad.append(per)
    print(f"\nsame-type {k}-hop brute force: (condition holds, outcome) -> count")
    for kk in sorted(seen):
        print(f"    {str(kk):<28} {seen[kk]}")
    print(f"  condition holds and the chain reverses a fact: {len(bad)}")
    return bad


def brute_force_same_type(trials=20000, seed=7):
    """Over random one-type logs: when x < m and m < y are ordered and every object
    carrying x and y also carries m, what can (x, y) be?

    Reports the outcome distribution and any reversal, which would refute the claim.
    """
    rnd = random.Random(seed)
    acts = ["x", "m", "y"]
    seen = defaultdict(int)
    witness_no_cond = []
    for _ in range(trials):
        nobj = rnd.randint(1, 4)
        per = []
        for _o in range(nobj):
            has = [a for a in acts if rnd.random() < 0.75]
            if not has:
                continue
            per.append({a: sorted(rnd.randint(0, 4) for _ in range(rnd.randint(1, 2)))
                        for a in has})
        if not per:
            continue
        ef, co = set(), set()
        for p in per:
            for a, va in p.items():
                for b, vb in p.items():
                    if a != b:
                        co.add((a, b))
                        if min(va) < max(vb):
                            ef.add((a, b))
        ordv = {(a, b) for (a, b) in ef if (b, a) not in ef}
        if ("x", "m") not in ordv or ("m", "y") not in ordv:
            continue
        cond = all("m" in p for p in per if "x" in p and "y" in p)
        out = ("ordered" if ("x", "y") in ordv else
               "reversed" if ("y", "x") in ordv else
               "tie" if ("x", "y") in co else "unwitnessed")
        seen[(cond, out)] += 1
        if not cond and out == "reversed" and len(witness_no_cond) < 3:
            witness_no_cond.append(per)
    print("\nsame-type brute force: (condition holds, outcome for (x,y)) -> count")
    for k in sorted(seen):
        print(f"    {str(k):<28} {seen[k]}")
    if witness_no_cond:
        print("  a log where the condition fails and the chain reverses a fact:")
        for p in witness_no_cond[0]:
            print(f"    object {p}")
    return seen


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def keepsets(log, mx, target):
    """The keep-sets compare_all.py builds, so the numbers are the paper's numbers."""
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        from compare_all import handoff, hoisted_drawn  # noqa: PLC0415
    finally:
        sys.argv = argv
    dr = hoisted_drawn(log, mx)
    rec = log.cells()
    nobj = defaultdict(int)
    for _o, t in log.obj_type.items():
        nobj[t] += 1
    clo = closure(log, mx)
    acts_of = defaultdict(set)
    for e, o in mx:
        acts_of[log.obj_type[o]].add(log.act[e])
    flow_ok = {t for t, a in acts_of.items()
               if len(clo.get(t, ())) / max(1, len(a) * (len(a) - 1) // 2) >= 0.05}
    cells_mx = {(log.act[e], log.obj_type[o]) for e, o in mx}
    cost = defaultdict(int)
    for e, o in mx:
        cost[(log.act[e], log.obj_type[o])] += 1
    return {"handoff": handoff(dr, target, rec, flow_ok, nobj, cells_mx, cost)}, dr


def run_log(path, do_saturate=True, searches=False):
    log = parse(path)
    print(f"\n=== {path.rsplit('/', 1)[-1]} ===")
    bases = [("recorded", list(log.e2o), log.cells())]
    mx = None
    if do_saturate:
        mx = sorted(saturate(log))
        bases.append(("max", mx, {(log.act[e], log.obj_type[o]) for e, o in mx}))
    for name, rel, cells in bases:
        ef, co, ordv, per = relations(log, rel)
        d = drawn(log, rel, cells)
        delivered = transitive(d)
        assert d <= ordv, "drawn must be inside ord"
        report(f"{name} (all cells)", delivered, ordv, ef, co, show=3)
        cyc = {(x, y) for (x, y) in ordv if (y, x) in ordv}
        if cyc:
            print(f"      {len(cyc)} pairs two types order both ways: {sorted(cyc)[:4]}")
    if not (searches and mx):
        return
    ef, co, ordv, _per = relations(log, mx)
    target = ordv
    sets, dr = keepsets(log, mx, target)
    for name, keep in sets.items():
        dd = dr(keep)
        delivered = transitive(dd)
        chain_only = (delivered & target) - dd
        rec_ord = relations(log, list(log.e2o))[2]
        print(f"  {name}: {len(keep)} cells, drawn {len(dd & target)}/{len(target)}, "
              f"chained {len(delivered & target)}/{len(target)}, "
              f"chain-only {len(chain_only)} "
              f"({sum(1 for p in chain_only if p not in rec_ord)} ordered only in max)")
        report(f"  {name}", delivered, target, ef, co, show=0)


def main():
    args = sys.argv[1:]
    out = None
    if args and args[0] == "--out":
        out, args = args[1], args[2:]
    if out:
        made = counterexamples(out)
        print("counterexample logs:")
        for name, path in made:
            print(f"  {name:<26} {path}")
        for _name, path in made:
            run_log(path, do_saturate=True)
    exhaustive_same_type()
    brute_force_same_type()
    brute_force_chain(3)
    for p in args:
        run_log(p, searches=True)


if __name__ == "__main__":
    main()
