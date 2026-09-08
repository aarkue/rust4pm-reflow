#!/usr/bin/env python3
"""The object-type schema every search in this directory should read.

`expansion_probe.qualified_maps` reads recorded O2O only and demands exact
totality. `schema_discovery` is the reference oracle: recorded O2O AND
co-participation, 0.95 coverage, residuals allowed. The gap is not marginal --
Container Logistics 1 map against 12, Hinge 15 against 35 -- so a search on the
former is a search on a different log.

Lives in its own module rather than in `optimum` because
`admissible_expansion` needs it and `optimum` imports `admissible_expansion`.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expansion_probe import compose  # noqa: E402
import schema_discovery as sd  # noqa: E402


def full_maps(log, depth: int = 3):
    """Recorded O2O AND co-participation, composed to `depth`.

    `log` is a `verify_handoff_export.Ocel`; it is adapted to
    `schema_discovery.Log` here so both parsers stay untouched.
    """
    objects_of = defaultdict(list)
    for oid, ot in log.obj_type.items():
        objects_of[ot].append(oid)
    event_objects = defaultdict(list)
    for e, o in log.e2o:
        event_objects[e].append(o)
    o2o = [(a, b, q) for a, b, q in log.o2o
           if a in log.obj_type and b in log.obj_type]
    slog = sd.Log(obj_type=dict(log.obj_type), objects_of=objects_of,
                  event_objects=event_objects,
                  event_activity=dict(log.act), o2o=o2o)
    ms = sd.recorded_maps(slog) + sd.coparticipation_maps_singlepass(slog)[0]
    base = {}
    for m in ms:
        key = (m.source, m.target, m.qualifier or m.origin)
        base[key] = dict(m.f)
    by_type = {t: set(v) for t, v in objects_of.items()}
    return compose(base, by_type, depth)
