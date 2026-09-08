# How this implementation differs from `ProM` and `PM4Py`

Nothing here comes from their code. Both were run as black boxes and only their output compared.
The reference is Leemans, *Robust Process Mining with Guarantees*. Where the three disagree, the
section of the thesis that settles it is named below.

`InductiveMinerOptions::{prom, pm4py}` and `InductiveMinerDfgOptions::{prom, pm4py}` switch to the
other tools' behaviour, so a comparison run measures the algorithm instead of these choices.

All numbers come from running the three on the same 300 random logs of 2 to 7 traces over 2 to 4
activities. Fall throughs fire far more often on logs that short than on real ones, so read every
disagreement rate as an upper bound.

## Inductive Miner

| difference | thesis and us | `ProM` | `PM4Py` |
|---|---|---|---|
| minimum self distance, ∧↔.1 (§5.5) | applied | not applied | not applied |
| single-activity filtering (§6.2.2.3) | applied | applied | not applied |
| empty traces, `\|ε\| >= \|L\| * f` (§6.2.2.4) | `>=` | `>=` | `>` |

Neither tool restricts a concurrent cut by the minimum-self-distance relation. Both separate `a`
from an `f` that loops around it. For `⟨a,c,f⟩ ⟨a,f,a⟩ ⟨c,a,f⟩`:

```text
ours   ∧(f, ↻(a, τ), ×(τ, c))
both   ∧(↻(a, τ), →(×(τ, c), f))
```

`PM4Py` drops the base case that keeps a leaf when single executions dominate. For
`[⟨a⟩¹⁰⁰, ⟨a,a⟩]` at `f = 0.2`, where `p = 101/203`:

```text
ours, ProM   a
PM4Py        ↻(a, τ)
```

`PM4Py` models the optionality of empty traces only strictly above the threshold. With 10 traces at
`f = 0.2` the boundary is exactly 2:

```text
1 of 10 empty   ours  a          ProM  a          PM4Py  a
2 of 10 empty   ours  ×(τ, a)    ProM  ×(τ, a)    PM4Py  a
3 of 10 empty   ours  ×(τ, a)    ProM  ×(τ, a)    PM4Py  ×(τ, a)
```

Logs of 2 to 7 traces hit that boundary often. It dominates the `PM4Py` gap at `f > 0` and vanishes
at `f = 0`, where the threshold is 0 and both sides keep the optionality.

Agreement over the 300 logs, as identical trees and as identical languages up to length 7:

| | `imf()` | preset | same language |
|---|---|---|---|
| `ProM`, f=0 | 53% | 73% | 78% |
| `ProM`, f=0.2 | 65% | 77% | 78% |
| `PM4Py`, f=0 | 59% | 87% | 89% |
| `PM4Py`, f=0.2 | 10% | 74% | 74% |

What is left, by cause. "Fall through involved" means disabling the non-flower fall throughs
changes our answer. That shows they mattered without proving they caused the divergence, so the
column is an upper bound:

| | optionality only | fall through involved | cut, split or flower |
|---|---|---|---|
| `ProM`, f=0 | 0% | 79% | 21% |
| `ProM`, f=0.2 | 1% | 90% | 9% |
| `PM4Py`, f=0 | 0% | 92% | 8% |
| `PM4Py`, f=0.2 | 53% | 42% | 5% |

Most of it is which activity a fall through takes out, a choice §6.1.2.4 leaves open.

## Inductive Miner - directly follows

| difference | thesis and us | `ProM` | `PM4Py` |
|---|---|---|---|
| lone activity with a self edge (§6.6.3.3) | `strictDfgTauLoop`, `a⁺` | flower, `a*` | flower, `a*` |
| flower model (§6.1.2.4) | `↺(×(…), τ)` | `↺(τ, ×(…))` | `↺(τ, ×(…))` |
| tau-loop fall throughs (§6.6.3.4) | both | both | none |
| `IMfd` base-case filtering (§6.6.4.3) | applied | not applied | no `IMfd` at all |

The empty-traces step runs long before the recursion gets here. Every trace of the sub-log holds at
least one `a`, which makes `a⁺` both fitting and the more precise answer:

```text
⟨a,a⟩        ours  ↻(a, τ)          ProM, PM4Py  ↻(τ, a)
```

`PM4Py` has only empty traces and the flower as fall throughs, so it flowers where a tau loop
applies. On the `{f, g, h}` subgraph of the thesis' own worked example L113:

```text
ours, ProM   ↻(×(f, g, h), τ)
PM4Py        ↻(τ, ×(f, g, h))
```

`PM4Py`'s `IMd` also ignores `noise_threshold`. Its output is identical at `f = 0.0` and `f = 0.4`
on every one of the 298 logs it survives; it raises `IndexError` on the other 2.

| | default | preset | same language |
|---|---|---|---|
| `ProM`, f=0 | 40% | 54% | 94% |
| `ProM`, f=0.2 | 21% | 53% | 93% |
| `PM4Py`, f=0 | 4% | 88% | 93% |
| `PM4Py`, f=0.2 | 4% | 87% | 92% |

The gap between the tree and the language column is almost entirely one redundant `τ`. We emit
`×(τ, X)` from the empty-traces step even when `X` already accepts the empty trace, as the thesis
does. Same language, one node more, and it accounts for 86 of the 300 logs against `ProM`.

## Two side findings

`guard_empty_cut_parts` never changed an outcome, over all 64 option combinations at both
thresholds.

`ProM` and `PM4Py` agree with each other on 216 of the 300 logs at `f = 0` but only 22 at
`f = 0.2`. Their `IMf` implementations diverge from each other more than either diverges from us,
so above `f = 0` there is no one behaviour to match.
