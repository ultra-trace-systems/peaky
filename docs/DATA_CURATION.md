# Data curation — the Mascope WRITE API + `peaky.io.curate`

`mascope-sdk` is **read-only** (peaks, matches, listings) plus `match_compounds`.
It exposes **no** way to create/rename/copy/move/delete workspaces, datasets,
batches or samples. Those endpoints exist on the server — the web app uses them —
but are undocumented. This doc is the reverse-engineered spec and the engine
(`peaky/io/curate.py`) built on top of it, so a campaign can be *organised* (not
just analysed) from Python / Claude.

## How the spec was derived (so it can be re-verified)

1. **SDK source** (`mascope_sdk/_http.py`, `resources/*.py`): confirmed the read
   surface + that all traffic is `{MASCOPE_URL}/api/{path}`, bearer auth, envelope
   `{"data": …, "message": …}`. No write helpers beyond POST + a file-upload agent.
2. **SPA bundle** (`/assets/index-*.js`, identical across the servers checked): the single
   `so.http` axios client's call sites give every `(method, path, body)` the UI
   uses. **Body is the 2nd arg; the 3rd `{use,type,params}` is client meta/query.**
3. **Live probes** (non-mutating): `GET` on a write-only path returns `405 Allow:
   POST` (route exists) vs `404` (absent); `GET /workspaces/{wid}/datasets` → 200
   confirmed the dataset path. Validated 2026-07-15 against a live Mascope server.

Re-derive after a server/UI upgrade: re-fetch the JS bundle and re-grep for
`http.<verb>(` call sites; re-run the 405/200 probes.

## Endpoint map (base `{url}/api`, envelope `{data}`)

| Resource | Verb + path | Body |
|---|---|---|
| **workspace** | `GET /workspaces` · `POST /workspaces` | `{workspace_name, workspace_description}` |
| | `PATCH/DELETE /workspaces/{id}` | patch: `{workspace_id, workspace_name?, workspace_description?}` |
| | members: `GET/POST /workspaces/{id}/members` · `PATCH/DELETE …/members/{mid}` | |
| **dataset** | `GET/POST /workspaces/{wid}/datasets` | `{dataset_name, dataset_description}` |
| | `PATCH/DELETE /workspaces/{wid}/datasets/{did}` | patch: `{dataset_id, dataset_name?, …, dataset_type?}` |
| | `POST /workspaces/{wid}/datasets/{did}/move` | `{source_workspace_id, target_workspace_id}` |
| **batch** | `GET /sample/batches?dataset_id=` · `POST /sample/batches` | `{sample_batch_name, sample_batch_description, sample_batch_polarity, dataset_id, target_collection_ids:[]}` |
| | `GET/PATCH/DELETE /sample/batches/{id}` | patch: `{sample_batch_id, sample_batch_name?, …}` |
| | `POST /sample/batches/{id}/copy` | `{dataset_id, sample_batch_name, sample_batch_description}` |
| | `POST /sample/batches/{id}/import` | `{sample_items:[…]}` |
| **sample item** | `POST /sample/items` · `PATCH /sample/items/{id}` | patch: `{sample_item:{…fields}}` |
| | `POST /sample/items/copy` · `/move` | `{sample_batch_id, sample_item_ids:[…]}` |
| | `POST /sample/items/delete` | `{sample_item_ids:[…]}` |
| **raw file** | `POST /sample/files/upload` (multipart `files`) · `/upload/tus` (resumable) | — |
| | `POST /sample/files/delete` · `/reprocess` | `{sample_file_ids:[…]}` |
| **match** | `POST /match/rematch/batch/{id}?full_remove=&force=` · `/match/rematch/batches` | — |

**Gotchas.** Batch-create polarity field is `sample_batch_polarity` (∈ `"+" "-"
"+-"`), *not* the `polarity` seen in list responses. PATCH bodies echo the id. A
batch PATCH **requires** `target_collection_ids` and **replaces** the batch's
collections with them: `update_batch` reads the current ones
(`target/associations/target_collections_in_sample_batch`) and sends them back
unless you pass `target_collection_ids=`, so a rename keeps the batch's target lists.
`sample/items/copy`/`move` target a batch by `sample_batch_id`. Copy/move/delete of
sample items take id **lists**.

## Authorization — the token for reads and the opened writes, a cookie for the rest

Mascope 1.13 opened a set of write routes to API tokens, one route at a time (its
`security/pentest/token_writes.json` is the list), and mascope-sdk 2026.10.20 wraps
them. `curate.TOKEN_WRITES` holds the ones this client sends:

- **With the API token alone** (Mascope >= 1.13, with the role each route needs):
  dataset create / rename / move / delete; batch rename / copy / delete; sample copy /
  move / update; raw-file delete. Merging daily batches into one campaign batch is a
  `copy_batch` of the first followed by `copy_samples` of the others into it.
- **Still behind a logged-in session cookie**: workspace create / rename / delete,
  batch create, batch import, batch rematch, sample delete, file reprocess. Pass
  `CurationClient.from_env(cookie=…)`, `$MASCOPE_SESSION_COOKIE` or
  `peaky curate … --cookie …`: the full `Cookie:` header copied from a logged-in
  browser request (DevTools → Network → any `/api/…` request → Request Headers →
  Cookie). Without it such a write **fails fast with a clear message** (it never
  sends a doomed request); `dry_run=True` needs no cookie.
- **Reads / listings / name-resolution** → the bearer token is enough.

A server before 1.13 refuses every token write with
`401 {"error":"Authorization failed. Please sign in to the Mascope."}` (validated
2026-07-15, when every write was session-only); the client's error then says to pass
a cookie. A token is presented under the service it was minted for
(`X-Service-Name`): a Jupyter Notebooks token is `mascope_sdk`, an AI Agent token
(Settings → API Access Tokens, Mascope >= 1.13) is `ai-agent` — set
`$MASCOPE_SERVICE_NAME` beside it, or every request answers 401.

## The engine — `peaky.io.curate.CurationClient`

```python
from peaky.io import curate
c = curate.CurationClient.from_env()                     # reads + the token writes (Mascope >= 1.13)
c = curate.CurationClient.from_env(cookie="…")           # + the session-only writes
c = curate.CurationClient.from_env(dry_run=True)          # PLAN mutations, send nothing (no cookie needed)
```

- **One HTTP boundary** (`_request`) — same `{url}/api` shape, bearer auth,
  Cloudflare-WAF retry (`io_mascope._with_waf_retry`) and `{data}` unwrap as reads.
- **Safe by default.** Every mutation is recorded in `c.actions` (`c.plan()` /
  `c.summary()`). `dry_run=True` records + returns the request it *would* send and
  sends nothing — preview a whole reorganisation, show the user, re-run for real.
  `delete_*` additionally refuse without `confirm=True` (or dry-run).
- **Name-driven.** `resolve_workspace_id` / `resolve_dataset_id` / `resolve_batch_id`
  (exact-id > exact-name > substring, raise on 0/ambiguous) and idempotent
  `ensure_workspace` / `ensure_dataset` / `ensure_batch`.

Primitives: `create/update/delete_workspace`, `create/update/delete/move_dataset`,
`create/update/delete/copy_batch`, `import_samples`, `rematch_batch`,
`update_sample`, `copy_samples`, `move_samples`, `delete_samples`,
`delete_files`, `reprocess_files`, plus `list_*`.

### CLI

```
peaky curate tree [--workspace W] [--deep]      # read-only hierarchy overview
peaky curate new-workspace NAME [--desc D]
peaky curate new-dataset  --workspace W --name N [--desc D]
peaky curate new-batch    --workspace W --dataset D --name N --polarity +|-|+-
peaky curate copy-batch   --workspace W --dataset D --batch B --to-dataset D2 [--to-workspace W2] --name N
peaky curate copy-samples --sample-ids ID… --to-workspace W --to-dataset D --to-batch B
peaky curate move-samples --sample-ids ID… --to-workspace W --to-dataset D --to-batch B
peaky curate rename {workspace|dataset|batch} TARGET [--workspace W] [--dataset D] --name N [--desc D]
peaky curate delete-batch --workspace W --dataset D --batch B --yes
```

Every mutating verb accepts `--dry-run` (preview, sends nothing); `delete-*` needs
`--yes`.

### "Sort the data with Claude's help" — the intended loop

1. `peaky curate tree --deep` (or `c.list_*`) → read the current organisation.
2. Decide the policy in-session (which samples belong where — by time window, wind
   zone, zero-air vs ambient, polarity, …), selecting `sample_item_id`s from the
   peak/TS tables the read side already provides.
3. Preview with a `dry_run=True` client → show `c.summary()`.
4. Apply with a live client (`ensure_dataset` / `create_batch` / `move_samples` /
   `update_sample` for metadata).

The engine owns the *primitives*; the sorting *policy* lives in the session.

## Safety / scope

These are **destructive, production writes**. `delete_*` are guarded
(`confirm=`/`--yes`); prefer **copy** over **move**, and validate on a scratch
workspace first. Membership/permission endpoints exist but are intentionally not
wrapped (access-control changes are out of scope for automated curation).
