# Peaky as an MCP server

Drive the peaky pipeline from any MCP client — **ChatGPT Developer Mode**,
Claude Desktop, Cursor — without a shell. `peaky mcp` starts a server that
exposes the pipeline as MCP tools.

*Module:* `peaky/mcp_server.py` · *CLI:* `peaky mcp` · *Extra:* `pip install
'mascope-peaky[mcp]'` · *Since:* 2026-07.

## Why this shape (and why `io_mascope` is not MCP)

There are two boundaries; keep them straight:

```
 client  ⟷  [MCP transport: small tool calls + results]  ⟷  peaky MCP server (host)
                                                                    │
                                                        io_mascope (mascope-sdk)
                                                                    │
                                            [direct HTTP + .env token]  ⟷  Mascope REST API
```

- **MCP is the OUTER boundary** (client ↔ peaky): only small things cross it —
  a batch name, a summary, a file path.
- **`io_mascope` is the INNER boundary** (peaky ↔ Mascope): a direct in-process
  HTTP client. The thousands-of-rows peak tables flow here and **never touch the
  MCP boundary or the model's context.** This is the same data hygiene peaky
  already enforces by *avoiding* the generic `mascope__*` MCP.

Consequence: the server **must run host-side** (it needs network + `.env`
credentials + `mascope-sdk`); it cannot run inside a client's sandbox. The
credentials stay on your machine and are never sent to the client.

## Install & run

```bash
pip install 'mascope-peaky[mcp]'
peaky setup                     # once: writes .env (MASCOPE_URL + token) + output/
peaky mcp                       # serve on 127.0.0.1:8765, streamable-HTTP
peaky mcp --host 0.0.0.0 --port 8765 --transport streamable-http
peaky mcp --transport stdio     # for a local stdio client (Claude Desktop)
```

## Connecting ChatGPT (Developer Mode)

ChatGPT's MCP connectors speak **SSE / streamable-HTTP over a URL** — they do
**not** reach `localhost` directly, so expose the local server through a tunnel:

1. `peaky mcp` (defaults to streamable-HTTP on `127.0.0.1:8765`).
2. Tunnel it: `ngrok http 8765` → gives a public `https://…` URL.
3. ChatGPT → Settings → Apps → Advanced → **Developer mode** → Connectors →
   add a custom MCP server with the tunnel URL, **No authentication** (local dev).
4. Ask ChatGPT to call the peaky tools (see below).

For a persistent setup, host the server where it can reach Mascope and put OAuth
in front. Note: the Mascope origin sits behind Cloudflare and rate-limits bursts
(HTTP 403) — an always-on server benefits from the bulk-loader WAF retry that
ships in `io_mascope.fetch_batch_peaks`.

## Tools

| tool | kind | what |
|---|---|---|
| `health` | quick | credentials present + Mascope reachable (workspace count) |
| `list_workspaces` | quick | workspaces the token can see |
| `list_datasets(workspace?)` | quick | datasets in a workspace |
| `list_batches(dataset, workspace?)` | quick | batches (name / polarity / status) |
| `list_samples(dataset, batch, limit=50)` | quick | samples; count exact, rows capped |
| `certify_neutrals(ledger_csv, reagent, ts_parquet?)` | quick | **offline** certified-neutral table over a ledger CSV (no server) |
| `assign_sample(sample_id, reagent, …)` | **job** | one-sample multi-pass assign → job_id |
| `run_batch(batch, dataset, reagent, k_max, …)` | **job** | whole-batch pipeline → job_id |
| `job_status(job_id)` | quick | status + recent log + result/paths |
| `list_jobs()` | quick | recent jobs |

### Long runs are background jobs

`assign_sample` (~minutes) and `run_batch` (~many minutes) return a `job_id`
immediately and run on a background thread, so an MCP request never blocks or
times out. Poll `job_status(job_id)`; when `status == "done"` the result carries
the run folder, the report PDF, the merged-ledger path, and key counts. The job
registry is **in-memory** — a server restart forgets running jobs (fine for
interactive use; artifacts on disk survive).

### Results are paths, not payloads

Heavy artifacts (PDF, xlsx, figures, merged ledger) are written to the server's
output dir (`$PEAKY_OUTPUT_DIR` or `~/peaky-output`) and returned as **paths**,
not streamed through the client. Retrieve them from the host filesystem.

## Testability

The tool *functions* in `mcp_server.py` are plain Python and do not import the
`mcp` package — only `build_server()` / `serve()` do. So the offline test suite
(`tests/test_mcp_server.py`) exercises every tool (IO monkeypatched, no network)
and the job manager without the optional dependency installed.
