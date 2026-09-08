# Code

The paper's numbers come from the Rust implementation: the examples in
`../../process_mining/examples/` and the `ocpn-quality` tool in
`../../process_mining/tools/ocpn-quality/`. What is here is the Python
reference oracle that the Rust run is diffed against, plus the figure
generators and the baselines. Logs go in `artifacts/logs/` (see
`artifacts/README.md`).

Every script here imports `verify_handoff_export.parse`, the OCEL 2.0 parser.

## Schema and reduction

| script | what it does |
|---|---|
| `verify_handoff_export.py` | the parser everything else uses, plus the keep-set verifier |
| `schema_discovery.py` | the reference oracle for schema discovery |
| `schema_maps.py` | the discovered maps for one log |
| `run_census.py` | census over a corpus, and the differential against the Rust implementation |
| `search_keepset.py` | feasible keep-sets, per-activity closure, residual cost per cell |
| `optimum.py`, `chained.py`, `fastdrawn.py` | the delivery functions: strict orderings a keep-set draws, and their closure under composition |
| `compare_all.py` | all search strategies under identical settings |
| `chaining_soundness.py` | the empirical check behind Prop. carried: does a chained ordering ever reverse a fact the log records |
| `redundancy.py` | orderings and co-occurrence facts a model asserts, per type |
| `df_arcs.py` | directly-follows arc counts per type and keep-set |
| `normalize_ocel_sqlite.py` | SQLite to OCEL 2.0 XML, used for the converted logs |

## Expansion

| script | what it does |
|---|---|
| `expansion_probe.py` | total maps, composition closure, candidate participations |
| `admissible_expansion.py` | Def. expansion's admissibility test |
| `expansion_facts.py` | orderings and co-occurrence facts before and after expansion |
| `expansion_levels.py` | candidate cells scored individually; the guided level |
| `expansion_boundary.py` | the boundary level |
| `novelty.py` | admissible cells that assert an ordering the model lacks; source of Sec. 6's counts |
| `expand_theta.py` | the feasible theta window per log |
| `minimal_expansion.py` | how few tuples buy the facts, and why sub-cell truncation is unsound |
| `write_expanded_ocel.py` | writes an expanded log as OCEL 2.0 XML at a chosen level |
| `export_expansion.py` | difference CSV and OC-DFG exports, before and after |

Known gap: `expand_theta.py` and `novelty.py` gate on event coverage where
Def. expansion is stated per participation. The published band 0.445-0.513 is
the per-participation one; `expand_theta.py`'s own band runs to 0.5545. The
two readings agree on the cells and tuples the paper reports: 19 cells /
73,861 tuples on Order Management, 8 cells on BPIC2017, either way.

## Figures

`make_schema_graph.py`, `make_cell_grid.py` and `make_expansion_figure.py` emit
TikZ or Graphviz from the logs. Generated, not drawn: regenerate rather than edit.

The cell grid takes its keep-set from `results/stats/<log>.keepsets.json`, which
`tools/ocpn-quality` writes from the assignment it scores, so the figure and the
table cannot disagree:

    (cd ../../process_mining/tools/ocpn-quality && cargo run --release -- ../../../artifacts/logs/<log>)
    python3 make_cell_grid.py <log> ../results/stats/<log>.keepsets.json measured > om-cellgrid.tex

## Baselines and per-type statistics

| script | what it does |
|---|---|
| `df2_baseline.py` | the divergence-free directly-follows graph of van Detten et al. (ICPM 2024) Sec. V-A, from their own definitions |
| `df2_orderings.py` | DF2's graph against the recorded per-type models in ordered activity pairs, the only unit the two share |
| `silence_vs_divergence.py` | the 2x2 of silent against divergent over every corpus cell; writes `results/stats/silence_vs_divergence.json` |
| `corpus_stats.py` | one JSON per log under `results/stats/`: per-type objects, share, participations, activities, DF pairs; per-cell multiplicity; divergence; ordered pairs |
| `model_silence.py` | the model-based behavioral abstraction read as orderings: mines each type's sublog with IMf and reports the pairs the process tree enforces, against the log-side verdicts of Def. orders |

## Traps, all of them hit at least once

- **Read O2O in both directions.** A recorded `A -[q]-> B` may be functional in
  either direction. Reading only the recorded one finds half of Order
  Management's schema.
- **Simultaneous events are concurrent.** Sorting by `(time, id)` invents an
  order. 66 of 7,659 Order Management items have two events at one instant, and
  it was enough to flip a 7,593-to-0 ordering into a parallel pair.
- **Never truncate inside a cell.** Keeping an object's first and last event per
  activity preserves eventually-follows and leaves involvement ragged; the
  minimum objects per event falls from 1 to 0 at 26 of 66 cells, which is what
  discovery reads to decide which constraints to propose.
- **Covering pairs are not information.** The transitive reduction of an ordering
  relation can *shrink* when you add a fact. Use the closure as an objective and
  the covering set as a readability measure.
