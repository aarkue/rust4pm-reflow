#!/usr/bin/env python3
"""Search the keep-set lattice for the cut a given downstream analysis wants.

The named keep-sets (canonical, maximum, handoff, ...) are seven points chosen
by hand out of a space this enumerates. Two facts make the search tractable and
they are the reason it is worth stating as a procedure rather than a menu:

  FEASIBILITY FACTORISES PER ACTIVITY. A cell (a,T) may be cut only against a
  cell (a,S) that is kept and that determines it, and determination is checked
  over the events of `a` alone. So the feasible keep-sets are the product, over
  activities, of the feasible type-sets at that activity, and each factor is
  enumerable exactly.

  THE OBJECTIVE DOES NOT. Model size couples activities: an OC-DFG arc is a
  pair of activities that one surviving type visits in order. So the product
  cannot be optimised factor by factor, and the search is a real one.

Objectives are downstream-analysis specific and that is the point: `dfg` counts
coloured directly-follows arcs, `types` counts surviving object types, which is
what an OCPN's place groups track, and `cells` is size for its own sake.

Feasible means lossless by construction. `--exact` additionally rejects any
keep-set whose projection asserts a directly-follows arc no full trace supports.

Usage:
  search_keepset.py <log.xml> [--objective=dfg|types|cells] [--exact]
                    [--connected] [--compare=keepsets.json] [--budget=200000]
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402
from expansion_probe import compose, qualified_maps  # noqa: E402


def routes(maps, fibres):
    """Every one-step derivation: (source type, target type, callable)."""
    out = []
    for (s, t, q), f in maps.items():
        if s == t:
            continue
        out.append((s, t, f"{q}", lambda objs, f=f: {f[o] for o in objs if o in f}))
        inv = fibres[(s, t, q)]
        out.append((t, s, f"~{q}",
                    lambda objs, inv=inv: set().union(*(inv.get(x, set()) for x in objs))
                    if objs else set()))
    return out


def closure(present, ev_by_type, evs, K, rts, max_union=2, residual=0.0):
    """As `closure_cost`, keeping only the reconstructible set."""
    return closure_cost(present, ev_by_type, evs, K, rts, max_union, residual)[0]


def residual_cells(log, keep, acts_events, ev_objs, depth=3, residual=1.0):
    """Per cut cell, how many events its best reconstruction gets wrong.

    Zero means the cell is exactly determined and deleting it loses nothing.
    Non-zero means the deletion is only lossless because an exception list
    ships with it, which is a different claim and a figure should not draw the
    two the same way.
    """
    from collections import defaultdict as _dd
    by_type = _dd(set)
    for oid, ot in log.obj_type.items():
        by_type[ot].add(oid)
    maps = compose(qualified_maps(log), by_type, depth)
    fibres = {}
    for key, f in maps.items():
        inv = _dd(set)
        for o, img in f.items():
            inv[img].add(o)
        fibres[key] = inv
    rts = routes(maps, fibres)
    ev_by_type = {}
    for e, objs in ev_objs.items():
        d = _dd(set)
        for o in objs:
            d[log.obj_type[o]].add(o)
        ev_by_type[e] = d

    ch = _dd(set)
    for a, t in keep:
        ch[a].add(t)
    out = {}
    for a, evs in acts_events.items():
        present = {t for (aa, t) in log.cells() if aa == a}
        _, _, per = closure_cost(present, ev_by_type, evs, set(ch[a]), rts,
                                 residual=residual)
        for t, miss in per.items():
            if miss:
                out[(a, t)] = miss
    return out


def closure_cost(present, ev_by_type, evs, K, rts, max_union=2, residual=0.0):
    """Types reconstructible at one activity from the kept set `K`.

    A cell is reconstructible when one route, or a union of at most
    `max_union` routes, reproduces its objects at *every* event of the
    activity. The union is what \\autoref{sec:schema} calls a finite union of
    qualified maps and it is not optional: at \\texttt{confirm order} the
    employee is the customer's primary or secondary sales rep and no single
    qualifier covers both.

    Iterated to a fixpoint, so a type established from `K` can then witness
    another. That chaining is what lets a coarse type reach a fine one's
    attribute: fibre out to the fine type, then apply its function.
    """
    budget = int(residual * len(evs))
    known, cost = set(K), 0
    per_cell: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for t in present - known:
            want = [ev_by_type[e].get(t, set()) for e in evs]
            cands = []
            for s, tt, _q, fn in rts:
                if tt != t or s not in known:
                    continue
                cands.append([fn(ev_by_type[e].get(s, set())) for e in evs])
            if not cands:
                continue
            # Fewest events the best route (or union of routes) gets wrong.
            # With residual=0 this must be zero, which is the exact operator
            # the paper claims; above zero it is the coverage threshold the
            # implementation actually ships, and the difference is a number
            # rather than a matter of opinion.
            best = None
            for r in range(1, min(max_union, len(cands)) + 1):
                for combo in itertools.combinations(cands, r):
                    miss = sum(
                        1 for i in range(len(evs))
                        if set().union(*(c[i] for c in combo)) != want[i]
                    )
                    if best is None or miss < best:
                        best = miss
                    if best == 0:
                        break
                if best == 0:
                    break
            if best is not None and best <= budget:
                known.add(t)
                cost += best
                per_cell[t] = best
                changed = True
    return known, cost, per_cell


def determines(log, maps, acts_events, ev_objs):
    """`(a,S) |- (a,T)`: at every event of `a`, the T-objects are exactly what
    one total map says they are. Two rules, and both are needed:

      R-fun  f : S -> T   and  obj^T(e) = f[obj^S(e)]
      R-fib  f : T -> S   and  obj^T(e) = f^-1[obj^S(e)]

    R-fun folds out the coarser type, R-fib the finer one. Implementing only
    R-fun makes every keep-set that carries a coarse type look infeasible,
    which is most of them: `handoff` cuts \\otItem{} against \\otOrder{} and
    needs the fibre.

    Ambiguity rejects: a witness holding at all but one event reconstructs a
    log that is not the log."""
    fibres: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for key, f in maps.items():
        inv: dict[str, set[str]] = defaultdict(set)
        for o, img in f.items():
            inv[img].add(o)
        fibres[key] = inv

    ev_by_type: dict[str, dict[str, set[str]]] = {}
    for e, objs in ev_objs.items():
        d: dict[str, set[str]] = defaultdict(set)
        for o in objs:
            d[log.obj_type[o]].add(o)
        ev_by_type[e] = d

    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for a, evs in acts_events.items():
        present = {t for e in evs for t in ev_by_type[e]}
        for key, f in maps.items():
            s, t, _ = key
            if s == t:
                continue
            # R-fun: witness s, target t.
            if s in present and t in present and (s, t) not in out[a]:
                if all(
                    {f[o] for o in ev_by_type[e].get(s, ()) if o in f}
                    == ev_by_type[e].get(t, set())
                    for e in evs
                ):
                    out[a].add((s, t))
            # R-fib: the same map read backwards makes t the witness and s the
            # target, since f : s -> t means the s-objects are the fibre of t.
            if s in present and t in present and (t, s) not in out[a]:
                inv = fibres[key]
                ok = True
                for e in evs:
                    seen_t = ev_by_type[e].get(t, set())
                    pred: set[str] = set()
                    for x in seen_t:
                        pred |= inv.get(x, set())
                    if pred != ev_by_type[e].get(s, set()):
                        ok = False
                        break
                if ok:
                    out[a].add((t, s))
    return out


def feasible_sets(present, ev_by_type, evs, rts, residual=0.0):
    """Every K whose closure covers the activity, with its residual cost.

    Exact enumeration: activities carry few types, so the 2^|types| factor is
    cheap and there is no reason to approximate it when only the product needs
    searching."""
    ps = set(present)
    out = []
    for r in range(len(present) + 1):
        for keep in itertools.combinations(sorted(present), r):
            got, cost, _ = closure_cost(ps, ev_by_type, evs, set(keep), rts,
                                        residual=residual)
            if got >= ps:
                out.append((frozenset(keep), cost))
    return out


def traces(log, ev_objs, acts_of_obj):
    """Per object, its activity sequence, sorted by (timestamp, activity, id).

    Timestamp alone leaves simultaneous events in whatever order the parse
    produced, so the arc count varies between runs; the activity name and then
    the event id break the tie, both intrinsic to the log. Same key as the Rust
    implementation, which breaks on the interned activity index and gets the
    same order because that index follows the sorted activity names.
    """
    return {
        o: [log.act[e] for e in sorted(
            evs, key=lambda e: (log.time.get(e, ""), log.act[e], e))]
        for o, evs in acts_of_obj.items()
    }


def dfg_arcs(log, tr, choice) -> set[tuple[str, str, str]]:
    """Coloured DF arcs under a keep-set given as activity -> set of types.

    Adjacent entries of the projected sequence, so a tie is drawn in the
    direction the sort key puts it. An arc is one linearisation of the log and
    not a claim that every object orders the pair that way -- the eventually
    follows facts are where that claim lives, and there a tie asserts nothing.
    """
    arcs = set()
    for o, seq in tr.items():
        t = log.obj_type[o]
        proj = [a for a in seq if t in choice.get(a, ())]
        for x, y in zip(proj, proj[1:]):
            arcs.add((t, x, y))
    return arcs


def main() -> None:
    opt = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True)
           for a in sys.argv[2:] if a.startswith("--")}
    objective = opt.get("--objective", "dfg")
    budget = int(opt.get("--budget", 200000))

    log = parse(sys.argv[1])
    ev_objs = defaultdict(set)
    acts_of_obj = defaultdict(list)
    for e, o in log.e2o:
        ev_objs[e].add(o)
        acts_of_obj[o].append(e)
    acts_events = defaultdict(list)
    for e, a in log.act.items():
        acts_events[a].append(e)

    by_type = defaultdict(set)
    for oid, ot in log.obj_type.items():
        by_type[ot].add(oid)
    maps = compose(qualified_maps(log), by_type, 3)
    fibres = {}
    for key, f in maps.items():
        inv = defaultdict(set)
        for o, img in f.items():
            inv[img].add(o)
        fibres[key] = inv
    rts = routes(maps, fibres)
    tr = traces(log, ev_objs, acts_of_obj)

    ev_by_type = {}
    for e, objs in ev_objs.items():
        d = defaultdict(set)
        for o in objs:
            d[log.obj_type[o]].add(o)
        ev_by_type[e] = d

    cells = log.cells()
    acts = sorted(acts_events)
    present = {a: sorted({t for (aa, t) in cells if aa == a}) for a in acts}

    resid = float(opt.get("--residual", 0.0))
    factored = {a: feasible_sets(present[a], ev_by_type, acts_events[a], rts,
                                 residual=resid) for a in acts}
    factors = {a: [k for k, _ in factored[a]] for a in acts}
    rescost = {a: dict(factored[a]) for a in acts}
    total = 1
    for a in acts:
        total *= len(factors[a])
    print(f"activities {len(acts)}  cells {len(cells)}  "
          f"unconstrained space 2^{len(cells)}")
    print("feasible sets per activity: " +
          ", ".join(f"{a}={len(factors[a])}" for a in acts))
    print(f"feasible keep-sets: {total:,}"
          f"{'  (enumerated exactly)' if total <= budget else '  (searched)'}")

    full_arcs = dfg_arcs(log, tr, {a: set(present[a]) for a in acts})

    def score(choice):
        if objective == "cells":
            return sum(len(v) for v in choice.values())
        if objective == "types":
            return len({t for v in choice.values() for t in v})
        return len(dfg_arcs(log, tr, choice))

    def exact(choice):
        red = dfg_arcs(log, tr, choice)
        sup = {(t, x, y) for (t, x, y) in full_arcs
               if t in choice.get(x, ()) and t in choice.get(y, ())}
        return not (red - sup)

    want_exact = "--exact" in opt
    best, best_s = None, None
    if total <= budget:
        for combo in itertools.product(*(factors[a] for a in acts)):
            choice = dict(zip(acts, combo))
            if want_exact and not exact(choice):
                continue
            s = score(choice)
            if best_s is None or s < best_s:
                best, best_s = choice, s
    else:
        # Coordinate descent from the largest feasible set at each activity,
        # restarted; the objective is not separable so this is a heuristic and
        # the paper must not call it optimal.
        cur = {a: max(factors[a], key=len) for a in acts}
        best, best_s = dict(cur), score(cur)
        improved = True
        while improved:
            improved = False
            for a in acts:
                for cand in factors[a]:
                    trial = dict(best)
                    trial[a] = cand
                    if want_exact and not exact(trial):
                        continue
                    s = score(trial)
                    if s < best_s:
                        best, best_s = trial, s
                        improved = True

    kept = sorted((a, t) for a, ts in best.items() for t in ts)
    print(f"\nbest under objective={objective}"
          f"{' +exact' if want_exact else ''}: {objective}={best_s}, "
          f"{len(kept)} cells")
    print(f"  DF arcs {len(dfg_arcs(log, tr, best))}  "
          f"types {len({t for _, t in kept})}  "
          f"exact={exact(best)}")
    for a in acts:
        print(f"    {a:<22} {sorted(best[a])}")

    if "--compare" in opt:
        named = json.load(open(opt["--compare"]))
        print(f"\n{'keep-set':<26}{'cells':>6}{'DF arcs':>9}{'types':>7}"
              f"{'exact':>7}{'feasible':>10}{'residual':>10}")
        rows = [("SEARCH", kept)] + [(k, [tuple(c) for c in v])
                                     for k, v in sorted(named.items())]
        for name, ks in rows:
            ch = defaultdict(set)
            for a, t in ks:
                ch[a].add(t)
            feas = all(frozenset(ch.get(a, ())) in factors[a] for a in acts)
            res = (sum(rescost[a].get(frozenset(ch.get(a, ())), 0) for a in acts)
                   if feas else None)
            print(f"{name:<26}{len(ks):>6}{len(dfg_arcs(log, tr, ch)):>9}"
                  f"{len({t for _, t in ks}):>7}"
                  f"{str(exact(ch)):>7}{str(feas):>10}"
                  f"{'-' if res is None else res:>10}")


if __name__ == "__main__":
    main()
