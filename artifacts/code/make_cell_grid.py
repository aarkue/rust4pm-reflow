#!/usr/bin/env python3
"""Emit the cell-grid figure as TikZ, from the log and a keep-set.

The grid is the paper's unit made visible: one square per (activity, object
type) pair, filled when the log records it and the operator keeps it, hollow
when the operator cuts it, blank when the log has no such participation. The
\\texttt{employees} row is the argument for per-cell granularity: eight cells
recorded, four kept, so no per-type rule can express the same decision.

A second keep-set can be overlaid as a ring in each cell it keeps. That is the
point the single grid cannot make on its own: two defensible levels disagree
about which type carries which activity, so the grid a reader is looking at is
a choice and not a consequence.

Generated rather than drawn, so it cannot drift from the measured keep-set.

Usage: make_cell_grid.py <log.xml> <keepsets.json> <variant> [overlay] > grid.tex
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verify_handoff_export import parse  # noqa: E402

# Defined in paper/main.tex; hex lives there so the text, both generated
# figures and the drawn OCPN share one palette.
COLOR = {
    "items": "otItems",
    "employees": "otEmployees",
    "orders": "otOrders",
    "packages": "otPackages",
    "products": "otProducts",
    "customers": "otCustomers",
}
# Types are rows and activities are columns, not the other way round: type
# names are short enough to sit in a left margin, activity names are not, and
# the resulting wide-and-short block sits beside the schema graph.
W, H = 0.48, 0.34


def cell_tikz(x, y, c, recorded, in_a, in_b, residual=0):
    """One cell, split along its diagonal when two keep-sets are compared.

    Upper-left half is the primary keep-set, lower-right the overlay. Marking
    each keep-set with its own half rather than marking only the overlay means
    an empty column reads as `this keep-set drops the type` instead of as a
    missing annotation, which is what a ring in the middle of the cell failed
    to say.
    """
    x0, y0 = x + 0.05, y + 0.04
    x1, y1 = x + W - 0.05, y + H - 0.04
    if not recorded:
        return [rf"  \path[draw=black!10, dash pattern=on 0.7pt off 1.2pt] "
                rf"({x0:.2f},{y0:.2f}) rectangle ({x1:.2f},{y1:.2f});"]
    if in_b is None:
        fill = f"{c}!58" if in_a else f"{c}!7"
        out = [rf"  \path[draw={c}{'' if in_a else '!30'}, fill={fill}, "
               rf"rounded corners=1pt] ({x0:.2f},{y0:.2f}) rectangle "
               rf"({x1:.2f},{y1:.2f});"]
        if residual:
            # Cut, but not exactly determined. Half-filled rather than crossed:
            # the cell is between the two states the other shades stand for,
            # and a cross reads as an error mark. Half, not the true fraction,
            # because a 12% wedge in a 0.48cm cell is invisible; the fraction
            # goes in the caption where it can be read.
            xm = (x0 + x1) / 2
            out.append(
                rf"  \path[fill={c}!58] ({x0:.2f},{y0:.2f}) rectangle "
                rf"({xm:.2f},{y1:.2f});")
            out.append(
                rf"  \draw[{c}!70, line width=0.4pt] ({xm:.2f},{y0:.2f}) -- "
                rf"({xm:.2f},{y1:.2f});")
            out.append(
                rf"  \draw[{c}, line width=0.4pt, rounded corners=1pt] "
                rf"({x0:.2f},{y0:.2f}) rectangle ({x1:.2f},{y1:.2f});")
        return out
    ul = f"{c}!58" if in_a else f"{c}!6"
    lr = f"{c}!58" if in_b else f"{c}!6"
    edge = c if (in_a or in_b) else f"{c}!30"
    return [
        rf"  \path[fill={ul}] ({x0:.2f},{y0:.2f}) -- ({x0:.2f},{y1:.2f}) -- "
        rf"({x1:.2f},{y1:.2f}) -- cycle;",
        rf"  \path[fill={lr}] ({x0:.2f},{y0:.2f}) -- ({x1:.2f},{y0:.2f}) -- "
        rf"({x1:.2f},{y1:.2f}) -- cycle;",
        rf"  \draw[{c}!35, line width=0.25pt] ({x0:.2f},{y0:.2f}) -- "
        rf"({x1:.2f},{y1:.2f});",
        rf"  \draw[{edge}, line width=0.4pt] ({x0:.2f},{y0:.2f}) rectangle "
        rf"({x1:.2f},{y1:.2f});",
    ]


def key_tikz(x, y, name_a, name_b):
    """A three-cell key, so the split does not depend on reading the caption."""
    out = []
    for i, (a, b, lab) in enumerate([
        (True, False, name_a), (False, True, name_b), (True, True, "both"),
    ]):
        cx = x + i * 2.75
        out += cell_tikz(cx, y, "black", True, a, b)
        out.append(rf"  \node[anchor=west, font=\scriptsize\ttfamily] "
                   rf"at ({cx + W + 0.03:.2f},{y + H / 2:.2f}) {{{lab}}};")
    return out


def main() -> None:
    log = parse(sys.argv[1])
    keepsets = json.load(open(sys.argv[2]))
    keep = {tuple(c) for c in keepsets[sys.argv[3]]}
    over = ({tuple(c) for c in keepsets[sys.argv[4]]}
            if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else None)
    cells = log.cells()

    # Cut cells are not all alike: most are determined exactly, a few only
    # against a stored exception list. Drawing them the same way claims the
    # schema determines something it does not.
    resid: dict[tuple[str, str], int] = {}
    if "--residuals" in sys.argv:
        from search_keepset import residual_cells
        ev_objs: dict[str, set[str]] = defaultdict(set)
        for e, o in log.e2o:
            ev_objs[e].add(o)
        acts_events: dict[str, list[str]] = defaultdict(list)
        for e, a in log.act.items():
            acts_events[a].append(e)
        resid = residual_cells(log, keep, acts_events, ev_objs)

    acts = sorted({a for a, _ in cells})
    # Types ordered by how much of them survives, so the eliminated block is
    # contiguous and the split is visible without reading the labels.
    types = sorted(
        {t for _, t in cells},
        key=lambda t: (-sum((a, t) in keep for a in acts), t),
    )

    out = [r"\begin{tikzpicture}[font=\footnotesize]"]
    for i, a in enumerate(acts):
        out.append(
            rf"  \node[anchor=west, rotate=40, black!70, "
            rf"font=\scriptsize\ttfamily] at ({i * W + W / 2:.2f}, 0.08) "
            rf"{{{a}}};"
        )
    for j, t in enumerate(types):
        c = COLOR.get(t, "black")
        y = -j * H - H / 2
        out.append(
            rf"  \node[anchor=east, text={c}, font=\scriptsize\ttfamily] "
            rf"at (-0.10, {y:.2f}) {{{t.replace('_', '-')}}};"
        )
        out.append(
            rf"  \node[anchor=west, font=\scriptsize, black!55] "
            rf"at ({len(acts) * W + 0.10:.2f}, {y:.2f}) "
            rf"{{{sum((a, t) in keep for a in acts)}"
            rf"/{sum((a, t) in cells for a in acts)}}};"
        )
    for j, t in enumerate(types):
        for i, a in enumerate(acts):
            out.extend(cell_tikz(i * W, -j * H - H, COLOR.get(t, "black"),
                                 (a, t) in cells,
                                 (a, t) in keep,
                                 None if over is None else (a, t) in over,
                                 resid.get((a, t), 0)))
    if over is not None:
        out += key_tikz(0.0, -len(types) * H - 0.60, sys.argv[3], sys.argv[4])

    # Where the surviving types stop.
    split = sum(1 for t in types if any((a, t) in keep for a in acts))
    if 0 < split < len(types):
        out.append(
            rf"  \draw[black!45] (-0.05, {-split * H:.2f}) -- "
            rf"({len(acts) * W + 0.05:.2f}, {-split * H:.2f});"
        )
    out.append(r"\end{tikzpicture}")

    n_keep, n_cell = len(keep & cells), len(cells)
    print("%% generated by code/make_cell_grid.py -- do not edit")
    print(f"%% {sys.argv[3]}: {n_keep} of {n_cell} cells kept, {len(acts)} "
          f"activities x {len(types)} types")
    if over is not None:
        print(f"%% overlay {sys.argv[4]}: {len(over & cells)} cells, "
              f"{len((over ^ keep) & cells)} disagreements")
    print("\n".join(out))


if __name__ == "__main__":
    main()
