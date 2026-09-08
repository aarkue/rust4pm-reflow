"""Run schema discovery over a corpus and print the census.

Also runs the differential check the pitch note demands: strict conflict
handling versus the defective variant that absorbs conflicts into the
residual slack.
"""

import sys
import time

from schema_discovery import (
    coparticipation_maps,
    generators,
    load,
    recorded_maps,
)


def census(name: str, path: str, min_coverage: float = 0.95) -> None:
    t0 = time.time()
    log = load(path)
    t_load = time.time() - t0

    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(
        f"{len(log.event_activity)} events / {len(log.obj_type)} objects / "
        f"{len(log.types)} types / {len(set(log.event_activity.values()))} "
        f"activities / {len(log.o2o)} O2O  (load {t_load:.2f}s)"
    )

    t0 = time.time()
    rec = recorded_maps(log, min_coverage)
    t_rec = time.time() - t0

    t0 = time.time()
    cop, rejected = coparticipation_maps(log, min_coverage, strict_conflicts=True)
    t_cop = time.time() - t0

    lax, _ = coparticipation_maps(log, min_coverage, strict_conflicts=False)

    print(f"\nrecorded O2O maps: {len(rec)}   ({t_rec:.2f}s)")
    for m in sorted(rec, key=str):
        print(f"  {m}")

    print(f"\nco-participation maps: {len(cop)}   ({t_cop:.2f}s)")
    for m in sorted(cop, key=str):
        print(f"  {m}")

    print(f"\nDIFFERENTIAL  strict={len(cop)}  lax(defective)={len(lax)}")
    strict_pairs = {(m.source, m.target) for m in cop}
    lax_pairs = {(m.source, m.target) for m in lax}
    spurious = lax_pairs - strict_pairs
    if spurious:
        for s, t in sorted(spurious):
            print(f"  ONLY UNDER LAX: {s} -> {t}   [{rejected.get((s, t), '?')}]")
    else:
        print("  no difference on this log")

    pairs = {(m.source, m.target) for m in rec} | strict_pairs
    gens, depth = generators(pairs)
    print(
        f"\ntype pairs covered: {len(pairs)}  generators: {len(gens)}  "
        f"max derivation depth: {depth}"
    )
    for s, t in sorted(gens):
        print(f"  gen: {s} -> {t}")

    only_rec = {(m.source, m.target) for m in rec} - strict_pairs
    only_cop = strict_pairs - {(m.source, m.target) for m in rec}
    print(f"\nrecorded only: {sorted(only_rec)}")
    print(f"co-participation only: {sorted(only_cop)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <log.sqlite> [<log.sqlite> ...]")
    corpus = [(p, p) for p in sys.argv[1:]]
    for name, path in corpus:
        census(name, path)
