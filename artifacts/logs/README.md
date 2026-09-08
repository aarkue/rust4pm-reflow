# Logs

The nine-log corpus. The examples read the logs from this directory
(`REFLOW_LOGS` overrides it) under exactly these filenames, the list in
`process_mining/examples/corpus/mod.rs`:

| file | log | source |
|---|---|---|
| `01_o2c.xml` | LRMS order-to-cash | https://www.ocel-standard.org/ |
| `02_p2p.xml` | LRMS procure-to-pay | https://www.ocel-standard.org/ |
| `03_hiring.xml` | LRMS hiring | https://www.ocel-standard.org/ |
| `04_hospital.xml` | LRMS hospital | https://www.ocel-standard.org/ |
| `ocel2-p2p-updated.xml` | Procure-to-Pay | https://www.ocel-standard.org/ |
| `ContainerLogistics.xml` | Container Logistics | https://www.ocel-standard.org/ |
| `socel2_hinge.xml` | Hinge | https://www.ocel-standard.org/ |
| `order-management.xml` | Order Management | https://www.ocel-standard.org/ |
| `bpic2017-no-W.xml.gz` | BPIC2017 | https://data.4tu.nl/articles/_/12696884/1 |

The first eight are the OCEL 2.0 XML files from the OCEL 2.0 standard site;
rename the download where its filename differs from the table.

`bpic2017-no-W.xml.gz` is an OCEL 2.0 conversion of the case-centric BPIC2017
XES log with the object types `Application`, `Offer` and `Case_R` (the resource),
plus a declared but empty `Workflow` type; the `W_` workflow events are dropped,
leaving the 18 `A_` and `O_` activities. The conversion script is not part of
these artifacts. `code/normalize_ocel_sqlite.py` is unrelated to it: it adds the
`ocel_time` / `ocel_changed_field` columns that some SQLite exports (Hinge among
them) lack, so the importer accepts them; the corpus itself is read from XML.
