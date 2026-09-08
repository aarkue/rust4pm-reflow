#!/usr/bin/env python3
"""Draw the before/after expansion figure from two exported OC-DFGs.

Three things this gets right that a hand-tuned pair of graphviz calls did not:

  ONE FILTER, BOTH PANELS. Arcs are kept in frequency order until they account
  for `--cover` of the total mass, and the same rule runs on each panel. A
  fixed top-N is not comparable across two graphs of different size.

  ONE NODE SET, BOTH PANELS. Every activity either panel mentions is drawn in
  both, so an activity whose arcs fall below the filter appears isolated rather
  than absent. Filtering the nodes away made the expanded panel look as though
  expansion had removed six activities.

  THE FREE RESOURCE IS OMITTED, AND SAYING SO IS NOT OPTIONAL. On BPIC2017 the
  resource type spans all 26 activities and contributes 445 arcs to both
  panels, so with it in, both graphs are connected and the figure shows
  nothing. It is the same effect that made the unrestricted direction rule fail
  in Sec. 7.

Usage: make_expansion_figure.py <before.dot> <after.dot> <outstem>
                                [--cover=0.9] [--drop=Case_R]
"""

from __future__ import annotations

import subprocess
import sys

COL = {"Application": "#1971c2", "Workflow": "#e8590c", "Offer": "#2f9e44",
       "items": "#e8590c", "orders": "#1971c2", "packages": "#8a5a2b",
       "employees": "#c2255c", "products": "#2f9e44", "customers": "#862e9c"}


def load(src, drop):
    out = []
    for line in open(src):
        if "->" not in line:
            continue
        x = line.split('"')[1]
        y = line.split('"')[3]
        t = line.split('tooltip="')[1].split(" x")[0]
        n = int(line.split(" x")[-1].split('"')[0])
        if t != drop:
            out.append((t, x, y, n))
    return out


def cover(arcs, frac):
    arcs = sorted(arcs, key=lambda a: -a[3])
    tot = sum(a[3] for a in arcs)
    acc, keep = 0, []
    for a in arcs:
        keep.append(a)
        acc += a[3]
        if acc >= frac * tot:
            break
    return keep


def components(arcs, nodes):
    par = {a: a for a in nodes}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for _t, x, y, _n in arcs:
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry
    touched = {a for _t, x, y, _n in arcs for a in (x, y)}
    return len({find(a) for a in touched}), len(nodes - touched)


def famof(a):
    return a.split("_", 1)[0] if "_" in a[:2] else None


def dump(arcs, nodes, dst):
    fams = sorted({f for f in (famof(a) for a in nodes) if f})
    names = {"A": "Application", "W": "Workflow", "O": "Offer"}
    mx = max((n for *_, n in arcs), default=1)
    with open(dst, "w") as f:
        f.write("digraph G {\n  rankdir=LR; bgcolor=transparent; "
                "nodesep=0.12; ranksep=0.35;\n")
        f.write('  node [shape=box, style="rounded,filled", '
                'fillcolor="#f8f9fa", color="#ced4da", fontname="Helvetica", '
                'fontsize=8, margin="0.05,0.02", height=0.22];\n')
        f.write("  edge [arrowsize=0.5];\n")
        for p in fams:
            grp = sorted(a for a in nodes if famof(a) == p)
            name = names.get(p, p)
            f.write(f'  subgraph cluster_{p} {{ label="{name}"; '
                    f'fontname="Helvetica"; fontsize=9; '
                    f'color="{COL.get(name, "#868e96")}"; style=rounded; '
                    f"penwidth=0.8;\n")
            for a in grp:
                f.write(f'    "{a}" [label="{a.split("_", 1)[1]}"];\n')
            f.write("  }\n")
        for a in sorted(nodes):
            if not famof(a):
                f.write(f'  "{a}";\n')
        for t, x, y, n in arcs:
            w = 0.4 + 1.5 * (n / mx) ** 0.4
            f.write(f'  "{x}" -> "{y}" [color="{COL.get(t, "#868e96")}", '
                    f"penwidth={w:.2f}];\n")
        f.write("}\n")


def main() -> None:
    frac, drop = 0.9, "Case_R"
    for a in sys.argv[1:]:
        if a.startswith("--cover="):
            frac = float(a.split("=", 1)[1])
        if a.startswith("--drop="):
            drop = a.split("=", 1)[1]
    before, after, stem = sys.argv[1], sys.argv[2], sys.argv[3]

    b, a2 = load(before, drop), load(after, drop)
    nodes = {n for arcs in (b, a2) for _t, x, y, _c in arcs for n in (x, y)}
    for label, arcs in (("before", b), ("after", a2)):
        kept = cover(arcs, frac)
        c, iso = components(kept, nodes)
        dst = f"{stem}-{label}.dot"
        dump(kept, nodes, dst)
        subprocess.run(["dot", "-Tpdf", "-Gsize=5.9,2.6!", dst,
                        "-o", f"{stem}-{label}.pdf"], check=True)
        print(f"{label}: {len(kept)} of {len(arcs)} arcs at cover={frac}, "
              f"{c} component{'s' if c != 1 else ''}, {iso} isolated, "
              f"{len(nodes)} activities drawn")


if __name__ == "__main__":
    main()
