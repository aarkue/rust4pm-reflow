"""Bring a variant OCEL 2.0 SQLite export into the layout the importer expects.

Several published logs (Hinge, Age of Empires) ship `object_<T>` and `event_<T>`
tables without the `ocel_time` / `ocel_changed_field` columns that the OCEL 2.0
relational format uses to carry time-varying object attributes. Objects in those
logs simply have no attribute history, so the columns are added as NULL rather
than being inferred.

Usage: python3 normalize_ocel_sqlite.py <in.sqlite> <out.sqlite>
"""

import shutil
import sqlite3
import sys


def normalize(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    con = sqlite3.connect(dst)
    tables = [
        r[0]
        for r in con.execute(
            "select name from sqlite_master where type='table' and name like 'object\\_%' escape '\\'"
        )
    ]
    patched = 0
    for t in tables:
        if t in ("object_map_type", "object_object"):
            continue
        cols = {r[1] for r in con.execute(f'pragma table_info("{t}")')}
        for col, decl in (("ocel_time", "TIMESTAMP"), ("ocel_changed_field", "TEXT")):
            if col not in cols:
                con.execute(f'alter table "{t}" add column {col} {decl}')
                patched += 1
    con.commit()
    con.close()
    print(f"{src} -> {dst}: {len(tables)} object tables, {patched} columns added")


if __name__ == "__main__":
    normalize(sys.argv[1], sys.argv[2])
