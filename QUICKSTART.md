# Peaky — Quickstart (assign a Mascope batch on your machine)

A 5-minute path from a fresh clone to a peak-assignment report on **your own**
Mascope data. For depth see [SKILL.md](SKILL.md); for dev/iteration see
[README.md](README.md).

## 1. Install + set up the workspace

```bash
git clone https://github.com/ultra-trace-systems/peaky.git && cd peaky
python3 -m pip install -e .   # pulls mascope-sdk + pandas/numpy/scipy/matplotlib/openpyxl; registers `peaky`
peaky setup                   # creates .env + output/, verifies, prints what to do next
```
Needs Python ≥ 3.12. Everything (incl. `mascope-sdk`) installs from public PyPI —
no private index, no Mascope account needed just to install. `peaky setup` is
idempotent — re-run it any time to re-check the workspace. After it runs you have
**one self-contained folder**:

```
peaky/                 ← the workspace (= your clone)
  .env                 your Mascope creds (URL + token)   ← edit this (step 2)
  output/              every run's results land here  (PEAKY_OUTPUT_DIR)
  peaky/  scripts/     the package + helper scripts
  SKILL.md             Claude Code skill instructions (`peaky install-skill`)
  docs/                ARCHITECTURE / ASSIGNMENT / OUTPUTS / ROADMAP
```

(Driving Peaky with Claude Code? Just point it at the clone and ask it to "install
and set up Peaky" — it runs the two commands above and reports the layout.)

## 2. Credentials

`peaky setup` created a git-ignored `.env` in the repo root. Edit it with your
Mascope server URL + API token (from the Mascope web app → account / API settings):

```bash
#  .env  ->  MASCOPE_URL=...   MASCOPE_ACCESS_TOKEN=...
```
Prefer a shared home location instead? Use `~/.mascope/.env`, `export MASCOPE_URL=…
MASCOPE_ACCESS_TOKEN=…`, or pass `--env /path/to/.env` to any command. Search
order: `--env` / `$MASCOPE_ENV` > repo-root `.env` (or cwd) > `~/.mascope/.env`.

## 3. Find your data

```bash
peaky list datasets
peaky list batches  --dataset "<your workspace>"
peaky list samples  --batch "<your batch>" --dataset "<your workspace>"
```
`list samples` prints each `sample_item_id`, time and TIC — copy an id for step 4.

## 4. Assign one sample

```bash
peaky assign --sample-id <ID> --reagent <Br|Ur|NO3|NO3_15N|auto> \
    --height-cutoff 100 --output-dir ~/peaky-output/<name>
```
`--reagent` forces the analyte channels (a positive/sparse sample otherwise
mis-detects as negative). Writes `<ID>_<UTC>_{ledger.csv, assignments.xlsx,
summary.md, manifest.json, gka.html}`. A ~1000-peak sample takes ≈5 min.

## 5. Assign a whole batch (recommended)

A single averaged file misses analytes present only part of a run, so the batch
flow assigns a representative subset (5 time-spaced + max-TIC) and merges, then
builds cluster figures, a Van Krevelen and a PDF report:

```bash
peaky batch --batch "<your batch>" --dataset "<your workspace>" \
    --reagent <Br|Ur|NO3|NO3_15N|auto> --out-dir ~/peaky-output
```
Creates a timestamped run folder `~/peaky-output/<batch-slug>_<UTC>/` with the
merged ledger, per-file ledgers, cluster/VK figures, and `report_<run-id>.pdf`.
A full batch is ≈40 min (mostly the live `match_compounds` calls).

Regenerate the figures + report later **offline** (no re-assignment) with:

```bash
peaky report --run-dir ~/peaky-output/<run-folder> \
    --reagent <Br|Ur|NO3|NO3_15N> --ts ~/peaky-output/<run-folder>/<tag>_ts.parquet
```

## Adding your reagent

`Br` (bromide⁻), `Ur` (urea⁺), `NO3` (¹⁴N nitrate⁻) and `NO3_15N` (**¹⁵N-labelled
nitrate⁻**, aliases `15no3`/`^no3-`/`nitrate-15n`) are built in. ¹⁵N nitrate is the
`+^NO3-` server mechanism: its adduct adds ¹⁵NO₃ (+62.985), and the pipeline
re-anchors the assignment onto the ¹⁵N isotopologue line (the server tags the
real peak as a non-base `[15N]` line). Negative mode also carries known-species
for **PFCAs** (perfluoro-acids, e.g. TFA), **organophosphates** and **chlorinated
paraffins** (³⁷Cl-confirmed) — isotope-confirmable halogens (Cl/Br/S) are opened
and tiered on their envelope, while monoisotopic F/P stay off the grid except
these known families. To add another reagent **without editing the package**,
write a small JSON (or TOML) file and pass `--reagent-config`:

```json
[{"name": "Ac", "label": "Acetate⁻", "polarity": "-",
  "adducts": ["[M+CH3COO]-", "[M-H]-"], "normaliser": "reagent",
  "reagent_ion_re": "C2H3O2-?$", "ranges": "C0-30 H0-50 O0-15",
  "detect_adduct": "[M+CH3COO]-", "aliases": ["acetate"]}]
```
```bash
peaky batch --batch "<batch>" --reagent Ac --reagent-config myreagents.json ...
```

## Troubleshooting

- **`403 / Attention Required`** — Mascope's Cloudflare WAF is rate-limiting you
  after a burst of calls. Wait 15–30 min with no traffic (polling extends it).
- **`401` / token errors** — refresh `MASCOPE_ACCESS_TOKEN` in `~/.mascope/.env`.
- **sample/batch "not found"** — ids go stale when a server copy is renamed;
  re-fetch fresh names with `peaky list`.
- **`ModuleNotFoundError`** — re-run `pip install -e .` (or, for the exact pinned
  versions, `uv sync` from the repo root).

The CLI catches these at the boundary and prints an actionable hint rather than a
raw traceback.
