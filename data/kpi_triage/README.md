# KPI triage data

`online_retail_daily.csv` is derived from **Online Retail II** by Daqing Chen, UCI Machine Learning Repository, 2019, https://doi.org/10.24432/C5CG6D, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: transactions are summed to one row per trading day and segment (all countries, and eight of the larger countries): net revenue in GBP (cancellations subtracted), distinct non-cancelled invoices, and distinct customers. Rebuild it from the original with `python -c "from kpi_triage.backgrounds import derive_online_retail; derive_online_retail()"` (downloads 45 MB, checks its sha256).

The other backgrounds are the histories of the Context is Key test tasks (ServiceNow, Apache-2.0), downloaded at a pinned commit and checked by sha256 on first use.

`generated/` (not committed) holds the examples: `python -m kpi_triage generate` rebuilds them identically from the seed.
