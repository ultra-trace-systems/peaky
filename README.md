# Peaky

**AI-native analysis toolbox for Mascope.** Describe what you want in plain
language and get reproducible peak assignments, figures, and reports from
high-resolution CIMS mass-spec data — without writing notebook code.

Peaky sits on top of **Mascope** (the data platform / database) and turns
post-processing into a conversation: you ask, [Claude Code](https://claude.com/claude-code)
drives Peaky's deterministic pipeline, and the numbers come out the same every time.

```
Mascope   →  data platform   (the app + database; system of record)
Peaky     →  analysis layer   (this toolbox; assignment, clustering, reports)
Claude    →  the interface    (you drive Peaky by asking in plain language)
```

## What it does

- **Chemical-formula assignment** — multi-pass, isotope-pattern-aware peak → formula
  annotation. Produces a tiered Excel (Assigned / Candidate / below-assignability)
  with commentary, close alternatives, per-isotopologue scores, and a peak-ownership
  audit, plus an interactive rotating-GKA widget.
- **Batch pipeline** — assigns a representative subset (5 time-spaced + max-TIC
  samples), merges them, then builds time-series correlation clusters, a full
  Van Krevelen, and an iterable PDF report.
- **Reagent-aware** — bromide (Br⁻), urea/uronium (Ur⁺), and nitrate (¹⁴N / ¹⁵N)
  CIMS reagents are built in; add your own with a small JSON/TOML file, no code changes.
- **Literature-aware** — curated, provenance-tagged reference peaklists (α-pinene
  OH-oxidation HOM, common MS contaminants) corroborate near-ties and rescue
  mass-matched unexplained peaks, always re-scored by Mascope before commit — they
  never override the isotope evidence.
- **Data curation** — the write side of the Mascope API (create / rename / copy / move
  workspaces, datasets, batches, samples), so you can *organise* a campaign, not just
  analyse it: split a run by time window, pull background/zero samples into their own
  batch, copy a subset into a new dataset. Name-driven, dry-run-safe, delete-gated.
  See **[docs/DATA_CURATION.md](docs/DATA_CURATION.md)**.
- **Reproducible by construction** — same inputs → same bytes out (determinism tests,
  byte-stable workbooks/reports), so any figure or assignment can be regenerated exactly.

## What it is (and isn't)

Peaky is a deterministic Python toolbox with two faces:

- a **natural-language interface** — installed as a Claude Code skill, you drive it by asking; and
- a **CLI** (`peaky`) for scripted / headless runs.

Mascope's scoring maths is the only scorer — run in-process by default via
`mascope_tools` (same IsoSpec + `score_pattern`), with the network `match_compounds`
as an opt-in fallback — and the chemistry gates (integer DBE, Senior, O-cap,
evidence-gated heteroatoms/halogens) are structural. **No LLM is in the assignment
loop** — the AI orchestrates, it never does the chemistry — which is
why results are reproducible and auditable. It is not an autonomous agent; you stay
in the loop and it asks when a choice (reagent, cutoff) actually matters.

## Install

Needs Python ≥ 3.12. Everything (including `mascope-sdk`) installs from public PyPI —
no private index, and **no Mascope account is needed just to install or to run the
offline tests.** (On PyPI the distribution is **`mascope-peaky`** — `peaky` was
taken; the import package and CLI are still `peaky`.)

```bash
git clone https://github.com/ultra-trace-systems/peaky.git
cd peaky
python3 -m pip install -e .          # registers the `peaky` command (and `mascope-assign` alias)
peaky setup                          # creates .env + output/, verifies, prints what to do next
```

`peaky setup` turns the clone into a **self-contained workspace** and tells you
what to do — re-run it any time:

```
peaky/                 ← the workspace (= your clone)
  .env                 your Mascope creds (URL + token)   ← edit this
  output/              every run's results land here  (PEAKY_OUTPUT_DIR)
  peaky/  scripts/     the package + helper scripts
  SKILL.md             Claude Code skill instructions
  docs/                ARCHITECTURE / ASSIGNMENT / OUTPUTS / ROADMAP
```

`pip install -e .` resolves the conservative version ranges in `pyproject.toml`.
For the **exact pinned versions** validated in CI, use the lockfile instead:
`uv sync`. `uv.lock` is the single source of truth for pins — there is no separate
`requirements.txt`. A no-network smoke check: `python3 tests/test_smoke.py`.

### Credentials

`peaky setup` created a git-ignored `.env` in the repo root — edit it with your
Mascope server URL + API token (from the Mascope web app's account / API settings):

```
.env  ->  MASCOPE_URL=...   MASCOPE_ACCESS_TOKEN=...
```

Prefer a shared location? Use `~/.mascope/.env` instead, or just `export MASCOPE_URL=…
MASCOPE_ACCESS_TOKEN=…`. Search order: `--env` / `$MASCOPE_ENV` > repo-root `.env`
(or cwd) > `~/.mascope/.env`.

## Run it with Claude Code (natural language)

Peaky ships as a Claude Code **skill**. Install [Claude Code](https://claude.com/claude-code),
register the skill (cross-platform — copies `SKILL.md` into your skills dir), restart
Claude Code, and just ask:

```bash
peaky install-skill      # -> ~/.claude/skills/peaky/SKILL.md   (re-run after editing SKILL.md)
```

Then, in Claude Code, ask in plain language — the skill triggers automatically:

> - "List my Mascope datasets."
> - "Assign formulas for `<your batch>` with the bromide reagent."
> - "Run the batch pipeline on `<your batch>` and build the Van Krevelen + PDF report."
> - "Why are these peaks unassigned?"

Claude reads `SKILL.md`, picks the right reagent and parameters, runs the
deterministic pipeline locally, and shows you the assignments / figures / report.

**The mental model** (paste this if a fresh Claude needs orienting): *Mascope =
data + scorer. Peaky = analysis. Claude = interface.* Mascope's `match_compounds`
is the **only** scorer — Peaky enumerates candidate formulas, hands them to
Mascope, and arbitrates; the chemistry gates are structural; **no LLM is in the
assignment loop**, so every run is reproducible and auditable. Claude orchestrates
(picks the reagent, runs the pipeline, reads results back) — it never does the
chemistry. Heavy work runs on the **host Python** (which has `mascope-sdk`) via the
Bash tool / `peaky` CLI — never transport peak tables through an MCP into context.

## Run it from any MCP client (ChatGPT, Cursor, Claude Desktop)

Claude Code is the most direct interface (it runs the CLI for you), but any
[MCP](https://modelcontextprotocol.io) client can drive Peaky too:

```bash
pip install 'mascope-peaky[mcp]'
peaky mcp                       # streamable-HTTP on 127.0.0.1:8765
peaky mcp --transport stdio     # for a local stdio client (Claude Desktop)
```

It exposes the pipeline as tools — `list_*`, `certify_neutrals` (offline),
`assign_sample` / `run_batch` (background jobs). The server runs **host-side**:
`io_mascope` stays a direct in-process HTTP client, so peak tables never cross
the MCP boundary and your credentials never leave the host. For **ChatGPT**
(Developer Mode connectors speak SSE / streamable-HTTP over a URL, not
localhost), expose it through a tunnel — full setup, incl. the ngrok +
connector steps, in **[docs/MCP.md](docs/MCP.md)**.

## Run it as a CLI (scripted)

```bash
peaky list datasets
peaky list batches  --dataset "<your workspace>"
peaky list samples  --batch "<your batch>" --dataset "<your workspace>"

# one sample
peaky assign --sample-id <ID> --reagent <Br|Ur|NO3|NO3_15N|auto> \
    --height-cutoff 100 --output-dir ~/peaky-output/<name>

# a whole batch (representative subset -> merge -> clusters -> Van Krevelen -> PDF)
peaky batch  --batch "<your batch>" --dataset "<your workspace>" \
    --reagent <Br|Ur|NO3|NO3_15N|auto> --out-dir ~/peaky-output --jobs 6

# organise data — the write API (--dry-run previews, sends nothing; deletes need --yes)
peaky curate tree --deep                                  # read the workspace/dataset/batch layout
peaky curate new-dataset  --workspace "<ws>" --name "<name>"
peaky curate copy-samples --sample-ids <ID...> \
    --to-workspace "<ws>" --to-dataset "<ds>" --to-batch "<batch>"
```

`--reagent` forces the analyte channels (a positive/sparse sample otherwise
mis-detects as negative). `--jobs/-j N` (or `PEAKY_JOBS`) assigns the selected
samples across `N` worker processes — ~3.5× faster on multicore, output identical
to a serial run; default is your physical-core count, `--jobs 1` is the serial
path. `mascope-assign` is kept as an alias of `peaky`.
Step-by-step walkthrough: **[QUICKSTART.md](QUICKSTART.md)**. Reagent depth, the
module map, and chemistry rules: **[SKILL.md](SKILL.md)**.

## Validation

Peaky is validated end-to-end on a representative two-reagent CIMS oxidation
experiment (representative-sample assign → merge → clustering → Van Krevelen →
PDF report):

- **Br⁻ CIMS run** — 80 samples / ~96 min → merged **502 M0**
  (402 Assigned / 100 Candidate), ~4× the per-file coverage.
- **Ur⁺ CIMS run** — 81 samples / ~97 min → merged **1319 M0**
  (1065 Assigned / 254 Candidate); the positive-mode NH₄→amine co-variation gate
  is applied at merge.

## Development

One ledger DataFrame (one row per peak; passes only fill/annotate), Mascope's scoring
maths is the only scorer (in-process by default), chemistry gates are structural. Every
change ships with a test, and the offline suite (no network) must stay green:

```bash
pytest tests/                        # or run any tests/test_*.py as a standalone script
```

CI runs the suite on Python 3.12–3.13 with no credentials.

- **How Peaky works** — the ledger model, pass sequence, data flow, module map:
  **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (start here).
- **Assignment explained** (for a scientist): **[docs/ASSIGNMENT.md](docs/ASSIGNMENT.md)**.
- **Outputs** — every artifact, where it's stored, what it is: **[docs/OUTPUTS.md](docs/OUTPUTS.md)**.
- **Data curation** — the reverse-engineered Mascope write API + the curation engine:
  **[docs/DATA_CURATION.md](docs/DATA_CURATION.md)**.
- **Claude-Code operating instructions** — reagents, flags, chemistry rules:
  **[SKILL.md](SKILL.md)**.
- **Development history + open items**: **[docs/ROADMAP.md](docs/ROADMAP.md)**.
- **Release notes**: **[CHANGELOG.md](CHANGELOG.md)**.

## License

Peaky is released under the **[Apache License 2.0](LICENSE)** — see [`LICENSE`](LICENSE)
for the full text and [`NOTICE`](NOTICE) for attribution. `mascope-sdk` and the Mascope /
Ultra Trace Systems platform are separately licensed and owned by their respective owners.

## Citation

If you use **peaky** in your research, please cite the software using the
metadata in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository"
button) or the archived Zenodo release:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21058928.svg)](https://doi.org/10.5281/zenodo.21058928)

**Mascope** has its own DOI:

> Mascope at Zenodo: https://zenodo.org/records/21037635
