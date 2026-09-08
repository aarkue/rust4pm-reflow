#!/usr/bin/env python3
"""DF2's graph against the reduction, in ordered activity pairs.

Arc counts are not comparable: DF2 accumulates one graph over all types where
the reduction keeps a per-type model. Ordered activity pairs are the unit both
answer, and the one Sec. 6 already uses for the expansion direction.

Columns:
  per-type    strict orderings the recorded per-type models deliver, under
              the paper's own drawn() and transitive() (fastdrawn, chained)
  DF2         pairs strictly ordered by reachability in the DF2 graph
  accumulated the same over the unfiltered accumulated DFG, i.e. what DF2's
              filtering is there to repair

Caveat: DF2's output is a process tree mined from its graph, not the graph.
This compares graphs, so it bounds rather than reports what their miner emits.

Usage: df2_orderings.py log.xml [log.xml ...]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_handoff_export import parse
from fastdrawn import Bounds
from chained import transitive
from df2_baseline import df2_edges, recorded_edges

def strict_reach(edges):
    nodes={x for e in edges for x in e}
    succ={n:set() for n in nodes}
    for a,b in edges: succ[a].add(b)
    reach={}
    for s in nodes:
        seen=set(); st=[s]
        while st:
            n=st.pop()
            for m in succ[n]:
                if m not in seen: seen.add(m); st.append(m)
        reach[s]=seen
    return {(a,b) for a in nodes for b in reach[a] if a!=b and a not in reach.get(b,())}

def supported(log):
    """Pairs (a,b) some single object's own trace witnesses: first(a) < last(b)."""
    from collections import defaultdict
    per=defaultdict(list)
    for e,o in log.e2o: per[o].append(e)
    out=set()
    for evs in per.values():
        evs.sort(key=lambda e:(log.time.get(e,''),e))
        tr=[log.act[e] for e in evs]
        first,last={},{}
        for i,a in enumerate(tr):
            first.setdefault(a,i); last[a]=i
        for a in first:
            for b in first:
                if a!=b and first[a]<last[b]: out.add((a,b))
    return out

out=[]
for p in sys.argv[1:]:
    log=parse(p)
    b=Bounds(log,[(e,o) for e,o in log.e2o])
    rec=transitive(b.drawn(log.cells()))
    d2=strict_reach(df2_edges(log))
    acc=strict_reach(recorded_edges(log))
    sup=supported(log)
    d2_only=d2-rec
    unsup=d2_only-sup
    rec_unsup=rec-sup
    n=p.rsplit('/',1)[-1]
    out.append({"log":n,"recorded_per_type":len(rec),"df2_graph":len(d2),
                "accumulated_dfg":len(acc),"df2_only":len(d2_only),
                "df2_only_unsupported":len(unsup),
                "per_type_unsupported":len(rec_unsup),
                "df2_only_unsupported_examples":sorted(unsup)[:6]})
    print(f"{n:<26} per-type {len(rec):>4} ({len(rec_unsup)} unsupported)   "
          f"DF2 {len(d2):>4}   DF2-only {len(d2_only):>4} of which "
          f"{len(unsup):>4} UNSUPPORTED by any object's trace")

with open(Path(__file__).resolve().parents[1] / 'results' / 'stats' / 'df2_orderings.json', 'w') as fh:
    json.dump(out,fh,indent=1)
