"""Console entry point for `peaky` (alias: `mascope-assign`).

Subcommands:
  setup          one-command workspace bootstrap (.env + output/ + verify)
  install-skill  register SKILL.md as a Claude Code skill (~/.claude/skills/peaky/)
  list           discover data on the server (workspaces / datasets / batches / samples)
  assign         single-sample multi-pass assignment -> ledger/xlsx/md/json/gka.html
  batch          whole-batch pipeline: assign subset -> merge -> cluster -> Van Krevelen -> PDF
  pool           pool many same-chemistry batches (regex) into ONE unified ledger + per-group reports
  report         regenerate figures + PDF from an existing run folder (offline)
  gka            build the interactive rotating-GKA HTML from a ledger CSV (offline)
  curate         organise data (write API): create/rename/copy/move workspaces,
                 datasets, batches, samples (--dry-run previews; deletes need --yes)
  mcp            serve the pipeline over MCP (ChatGPT / Claude Desktop / Cursor)

Run `peaky <cmd> --help` for each. Heavy work runs on the host Python
(this package + mascope-sdk). A Mascope account/token is read from ~/.mascope/.env
(or --env / $MASCOPE_ENV / the process environment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _require_creds() -> None:
    """Fail fast with an actionable message if no Mascope creds are resolvable
    (before any expensive work). Process env vars satisfy this without a file."""
    from peaky.io import io_mascope as IO

    if os.environ.get("MASCOPE_URL") and os.environ.get("MASCOPE_ACCESS_TOKEN"):
        return
    path = IO._find_env()
    if not os.path.exists(path):
        sys.exit(
            "No Mascope credentials found.\n"
            f"  Looked for: {path}\n"
            "  Fix: copy .env.example to ~/.mascope/.env and fill in MASCOPE_URL +\n"
            "       MASCOPE_ACCESS_TOKEN, pass --env PATH, or export the two vars.")


def _friendly_server_error(e: Exception) -> str | None:
    """Map a raw SDK/HTTP exception to an actionable hint, or None if unrecognised."""
    s = f"{type(e).__name__}: {e}".lower()
    if "403" in s or "attention required" in s or "cloudflare" in s:
        return ("Mascope is rate-limiting you (Cloudflare WAF 403). Wait 15-30 min "
                "with NO traffic, then retry — polling extends the block.")
    if "401" in s or "unauthorized" in s or "forbidden token" in s:
        return ("Authorization failed (401). MASCOPE_ACCESS_TOKEN is likely expired "
                "— refresh it in ~/.mascope/.env.")
    if "no peaks" in s or "no samples" in s or "no batches" in s or "404" in s \
            or "not found" in s:
        return ("Not found on the server. IDs can go stale when a server copy is "
                "renamed — re-fetch fresh names/ids with `peaky list`.")
    return None


def _run_guarded(fn) -> int:
    try:
        fn()
        return 0
    except SystemExit:
        raise
    except Exception as e:                       # noqa: BLE001 — CLI boundary
        hint = _friendly_server_error(e)
        msg = f"\nERROR: {hint}\n  (raw: {type(e).__name__}: {e})" if hint \
            else f"\nERROR: {type(e).__name__}: {e}"
        print(msg, file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_list(args) -> None:
    _require_creds()
    from peaky.io import io_mascope as IO

    if args.what == "workspaces":
        ws = IO.list_workspaces()
        col = "workspace_name" if "workspace_name" in ws.columns else ws.columns[0]
        print(f"{len(ws)} workspaces:")
        for v in ws[col].tolist():
            print("  ", v)
        return
    client = IO.connect()
    if args.what == "datasets":
        ds = IO.list_datasets(client)
        col = "dataset_name" if "dataset_name" in ds.columns else ds.columns[0]
        print(f"{len(ds)} datasets:")
        for v in ds[col].tolist():
            print("  ", v)
    elif args.what == "batches":
        if not args.dataset:
            sys.exit("`list batches` needs --dataset NAME "
                     "(see `peaky list datasets`)")
        bs = IO.list_batches(client, args.dataset)
        cols = [c for c in ("sample_batch_name", "polarity", "status") if c in bs.columns]
        print(f"{len(bs)} batches in {args.dataset!r}:")
        print(bs[cols].to_string(index=False) if cols else bs.to_string(index=False))
    elif args.what == "samples":
        if not (args.batch and args.dataset):
            sys.exit("`list samples` needs --batch NAME --dataset NAME")
        sl = IO.fetch_batch_samples(client, args.batch, dataset=args.dataset)
        cols = [c for c in ("sample_item_id", "sample_item_name", "datetime_utc",
                            "tic", "polarity") if c in sl.columns]
        print(f"{len(sl)} samples in {args.batch!r}:")
        print(sl[cols].to_string(index=False) if cols else sl.to_string(index=False))


def _resolve_reagent(args):
    """Return (adducts, context, note). Forces the analyte channels so a positive
    or sparse-match sample never silently falls back to [M-H]- (wrong polarity).
    adducts=None means 'let assign.run auto-detect from the sample'."""
    from peaky.chem import profiles

    config = getattr(args, "reagent_config", None)
    if args.adducts:
        return list(args.adducts), (args.context or "ambient-air"), \
            f"forced adducts={list(args.adducts)}"
    if args.reagent and args.reagent.lower() != "auto":
        prof = profiles.resolve(args.reagent, config=config)   # name/alias, no peaks needed
        return list(prof.adducts), (args.context or prof.context), \
            f"{prof.name} ({prof.label})"
    # auto: detect from the sample's own peaks (cached, so assign.run reuses it)
    from peaky.io import io_mascope as IO

    client = IO.connect()
    raw = IO.fetch_peaks(client, args.sample_id, use_cache=not args.no_cache)
    try:
        prof = profiles.resolve("auto", raw, config=config)
        return list(prof.adducts), (args.context or prof.context), \
            f"auto-detected {prof.name} ({prof.label})"
    except Exception as e:                           # noqa: BLE001
        return None, (args.context or "ambient-air"), \
            (f"auto-detect found no known profile ({e}); using per-sample adduct "
             "detection — pass --reagent explicitly for a positive/sparse sample")


def cmd_assign(args) -> None:
    _require_creds()
    from peaky.assignment import assign
    from peaky.reporting import gka_widget
    from peaky.io import io_mascope
    from peaky.assignment import passes
    from peaky.reporting import report

    cfg = passes.PassConfig(ppm=args.ppm, search_ppm=args.search_ppm,
                            height_cutoff=args.height_cutoff)
    od = Path(args.output_dir).expanduser()
    od.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    base = od / f"{args.sample_id}_{stamp}"

    adducts, context, note = _resolve_reagent(args)
    print(f"[reagent] {note}; adducts={adducts}; context={context}")

    ts_peaks = None
    if args.ts_batch:
        client = io_mascope.connect()
        ts_peaks = client.load_peaks(dataset=args.ts_dataset, batches=args.ts_batch,
                                     matches=False, areas=False, heights=True,
                                     average=False, confirm_above=None)
        print(f"[ts] loaded {len(ts_peaks)} peaks across "
              f"{ts_peaks['sample_item_id'].nunique()} samples")

    out = assign.run(args.sample_id, context, cfg=cfg, use_cache=not args.no_cache,
                     do_pass2=not args.no_pass2, do_pass3=not args.no_pass3,
                     do_pass4=not args.no_pass4, do_pass5=not args.no_pass5,
                     adducts=adducts, ts_peaks=ts_peaks,
                     checkpoint_dir=str(od / "checkpoints"))
    led = out["ledger"]

    led.to_csv(f"{base}_ledger.csv", index=False)
    report.write_excel(led, f"{base}_assignments.xlsx", out["context"],
                       sample_id=args.sample_id)
    report.write_markdown(out, f"{base}_summary.md")
    manifest = {k: v for k, v in out.items() if k != "ledger"}
    Path(f"{base}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    Path(f"{base}_gka.html").write_text(
        gka_widget.render_html(gka_widget.build_points(led), args.sample_id, args.ppm),
        encoding="utf-8")  # HTML carries non-ASCII glyphs (▶/⏸); force UTF-8 on Windows
    # second widget over the UNEXPLAINED residual only — the honest place to hunt
    # for missed homologous structure
    un = led[led["role"] == "unexplained"]
    Path(f"{base}_gka_unexplained.html").write_text(
        gka_widget.render_html(
            gka_widget.build_points(un),
            f"{args.sample_id} — UNEXPLAINED residual ({len(un)} peaks)", args.ppm),
        encoding="utf-8")

    st = out["stats"]
    expl = 100 * (st["signal_by_role"]["M0"] + st["signal_by_role"]["iso_child"]
                  + st["signal_by_role"]["reagent"])
    cf = st.get("count_frac_by_role", {})
    print(f"\nwrote {base}_*.{{csv,xlsx,md,json,html}} (+ _gka_unexplained.html)")
    print(f"assigned {st['by_role']['M0']} | iso {st['by_role']['iso_child']} | "
          f"reagent {st['by_role']['reagent']} | unexplained {st['by_role']['unexplained']}")
    head = (f"peaks explained {100*(1-cf['unexplained']):.1f}% | " if cf else "")
    print(head + f"signal explained {expl:.1f}%  | "
          f"ledger problems: {out['problems'] or 'none'}")


def cmd_batch(args) -> None:
    _require_creds()
    from peaky import pipeline as PL

    res = PL.run_batch(batch=args.batch, dataset=args.dataset, reagent=args.reagent,
                       base_out=resolve_out_dir(args.out_dir), ts=args.ts,
                       subject=args.subject, do_report=not args.no_report,
                       config=args.reagent_config, select=args.select,
                       coverage_target=args.coverage_target, k_max=args.k_max,
                       height_floor=args.height_floor, n_jobs=args.jobs)
    ctx = res["ctx"]
    print(f"\n[batch] done -> {ctx.out_dir}")
    if res.get("report_pdf"):
        print(f"  report: {res['report_pdf']}")
    if res.get("report_pdf_small"):
        print(f"  report (small): {res['report_pdf_small']}")


def cmd_pool(args) -> None:
    _require_creds()
    from peaky import pipeline as PL

    res = PL.run_pooled_batches(
        batches=args.batches, dataset=args.dataset, reagent=args.reagent,
        base_out=resolve_out_dir(args.out_dir), out_name=args.out_name,
        group_by=args.group_by, ts=args.ts, subject=args.subject,
        do_report=not args.no_report,
        per_group_reports=not args.no_group_reports, config=args.reagent_config,
        coverage_target=args.coverage_target, k_max=args.k_max,
        height_floor=args.height_floor, n_jobs=args.jobs)
    ctx = res["ctx"]
    print(f"\n[pool] unified ledger -> {ctx.out_dir}")
    if res.get("report_pdf"):
        print(f"  report: {res['report_pdf']}")
    if res.get("group_runs"):
        print(f"  {len(res['group_runs'])} per-group reports:")
        for gr in res["group_runs"]:
            print(f"    {gr}")


def cmd_report(args) -> None:
    # offline: regenerate cluster figures + Van Krevelen + the PDF report from an
    # existing run folder's ledgers (no assignment, no network).
    from peaky import pipeline as PL
    from peaky.chem import profiles as P

    prof = P.resolve(args.reagent)
    run_dir = os.path.expanduser(args.run_dir)
    ctx = PL.RunContext(
        out_dir=run_dir, batch_name=(args.batch or prof.label),
        tag=(args.tag or prof.name), label=prof.label, when=None,
        run_id=(args.run_id or os.path.basename(run_dir.rstrip("/"))),
        generated=(args.generated or ""), profile=prof)
    out = PL.generate_report(ctx, os.path.expanduser(args.ts), subject=args.subject)
    print("wrote", out.get("report_pdf"))
    if out.get("report_pdf_small"):
        print("wrote", out.get("report_pdf_small"), "(compressed)")


def cmd_gka(args) -> None:
    import pandas as pd

    from peaky.reporting import gka_widget

    led = pd.read_csv(args.ledger_csv)
    pts = gka_widget.build_points(led)
    out = args.out or (Path(args.ledger_csv).with_suffix("").as_posix() + "_gka.html")
    Path(out).write_text(
        gka_widget.render_html(pts, Path(args.ledger_csv).stem, args.ppm),
        encoding="utf-8")  # HTML carries non-ASCII glyphs (▶/⏸); force UTF-8 on Windows
    print(f"wrote {out}  ({len(pts)} points)")


# --------------------------------------------------------------------------- #
# parser + entry point
# --------------------------------------------------------------------------- #
def cmd_curate(args) -> None:
    """Data-curation verbs (create/rename/copy/move workspaces, datasets, batches,
    samples). Every mutation honors --dry-run (preview, sends nothing); deletes
    additionally need --yes. Backed by peaky.io.curate.CurationClient."""
    _require_creds()
    from peaky.io import curate as CU

    c = CU.CurationClient.from_env(dry_run=args.dry_run,
                                   cookie=getattr(args, "cookie", None))
    verb = args.verb

    if verb == "tree":
        wsdf = c.list_workspaces()
        if args.workspace:
            wsdf = wsdf[wsdf["workspace_id"] == c.resolve_workspace_id(args.workspace)]
        for _, w in wsdf.iterrows():
            print(f"▸ {w['workspace_name']}  [{w['workspace_id']}]")
            for _, d in c.list_datasets(w["workspace_id"]).iterrows():
                print(f"    · {d['dataset_name']}  [{d['dataset_id']}]")
                if args.deep:
                    for _, b in c.list_batches(d["dataset_id"]).iterrows():
                        pol = b.get("polarity", "")
                        print(f"        - {b['sample_batch_name']} ({pol})  "
                              f"[{b['sample_batch_id']}]")
        return

    if verb == "new-workspace":
        r = c.create_workspace(args.name, args.desc or "")
    elif verb == "new-dataset":
        wid = c.resolve_workspace_id(args.workspace)
        r = c.create_dataset(wid, args.name, args.desc or "")
    elif verb == "new-batch":
        did = c.resolve_dataset_id(args.workspace, args.dataset)
        r = c.create_batch(did, args.name, args.polarity, args.desc or "")
    elif verb == "copy-batch":
        did = c.resolve_dataset_id(args.workspace, args.dataset)
        bid = c.resolve_batch_id(did, args.batch)
        to_did = c.resolve_dataset_id(args.to_workspace or args.workspace,
                                      args.to_dataset)
        r = c.copy_batch(bid, to_did, name=args.name, description=args.desc or "")
    elif verb in ("copy-samples", "move-samples"):
        to_did = c.resolve_dataset_id(args.to_workspace, args.to_dataset)
        to_bid = c.resolve_batch_id(to_did, args.to_batch)
        fn = c.copy_samples if verb == "copy-samples" else c.move_samples
        r = fn(args.sample_ids, to_bid)
    elif verb == "rename":
        r = _curate_rename(c, args)
    elif verb == "delete-batch":
        did = c.resolve_dataset_id(args.workspace, args.dataset)
        bid = c.resolve_batch_id(did, args.batch)
        r = c.delete_batch(bid, confirm=args.yes)
    else:
        sys.exit(f"unknown curate verb {verb!r}")

    print(c.summary())
    if not args.dry_run and isinstance(r, dict) and not r.get("_planned"):
        print(f"server -> {r}")
    if args.dry_run:
        print("\n(--dry-run: nothing was sent. Re-run without --dry-run to apply.)")


def _curate_rename(c, args):
    if args.kind == "workspace":
        return c.update_workspace(c.resolve_workspace_id(args.target),
                                  name=args.name, description=args.desc)
    if args.kind == "dataset":
        wid = c.resolve_workspace_id(args.workspace)
        did = c._resolve(c.list_datasets(wid), args.target, id_col="dataset_id",
                         name_col="dataset_name", kind="dataset")
        return c.update_dataset(wid, did, name=args.name, description=args.desc)
    did = c.resolve_dataset_id(args.workspace, args.dataset)
    bid = c.resolve_batch_id(did, args.target)
    return c.update_batch(bid, name=args.name, description=args.desc)


def cmd_mcp(args) -> None:
    """Launch the peaky MCP server (drive the pipeline from an MCP client:
    ChatGPT Developer Mode, Claude Desktop, Cursor, ...). Credentials stay
    server-side; only small tool results cross the MCP boundary."""
    _require_creds()
    from peaky import mcp_server
    print(f"[mcp] peaky MCP server on {args.transport} at "
          f"{args.host}:{args.port} (Ctrl-C to stop)")
    print("[mcp] tools: health, list_workspaces/datasets/batches/samples, "
          "certify_neutrals, assign_sample, run_batch, job_status, list_jobs")
    mcp_server.serve(host=args.host, port=args.port, transport=args.transport)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="peaky",
        description="Peaky — reproducible multi-pass formula assignment for Mascope peaks.")
    ap.add_argument("--env", default=None,
                    help="path to a Mascope .env (else ~/.mascope/.env or $MASCOPE_ENV)")
    ap.add_argument("--workspace", default=None,
                    help="Mascope workspace name/substring/id (else $MASCOPE_WORKSPACE; "
                         "auto-selected only when the token sees exactly one). "
                         "Discover with `peaky list workspaces`. Goes BEFORE the command.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="discover workspaces / datasets / batches / samples")
    pl.add_argument("what", choices=["workspaces", "datasets", "batches", "samples"])
    pl.add_argument("--dataset", default=None, help="dataset name (see `peaky list datasets`)")
    pl.add_argument("--batch", default=None, help="sample-batch name (for `samples`)")
    pl.set_defaults(func=cmd_list)

    pa = sub.add_parser("assign", help="assign one sample")
    pa.add_argument("--sample-id", required=True)
    pa.add_argument("--reagent", default="auto",
                    help="reagent profile: auto | Br | Ur | ... — forces the analyte "
                         "channels + default context ('auto' detects from the sample)")
    pa.add_argument("--adducts", nargs="+", default=None,
                    help="explicit analyte adduct channels (overrides --reagent)")
    pa.add_argument("--context", default=None,
                    help="plausibility context (default = the reagent profile's context)")
    pa.add_argument("--reagent-config", default=None,
                    help="JSON/TOML file registering extra reagent profiles")
    pa.add_argument("--ppm", type=float, default=1.0)
    pa.add_argument("--search-ppm", type=float, default=3.0)
    pa.add_argument("--height-cutoff", type=float, default=100.0)
    pa.add_argument("--no-cache", action="store_true")
    pa.add_argument("--no-pass2", action="store_true")
    pa.add_argument("--no-pass3", action="store_true")
    pa.add_argument("--no-pass4", action="store_true")
    pa.add_argument("--no-pass5", action="store_true")
    pa.add_argument("--output-dir", default=".")
    pa.add_argument("--ts-batch", default=None,
                    help="batch name to load as the time series (optional TS step)")
    pa.add_argument("--ts-dataset", default=None, help="dataset for --ts-batch")
    pa.set_defaults(func=cmd_assign)

    pb = sub.add_parser("batch", help="assign + cluster + Van Krevelen + report for a whole batch")
    pb.add_argument("--batch", required=True, help="sample-batch name")
    pb.add_argument("--dataset", default=None, help="dataset (workspace) name")
    pb.add_argument("--reagent", default="auto", help="auto | Br | Ur | NO3 | ...")
    pb.add_argument("--reagent-config", default=None,
                    help="JSON/TOML file registering extra reagent profiles")
    pb.add_argument("--out-dir", default=None,
                    help="base output dir for the versioned run folder. Default: "
                         "$PEAKY_OUTPUT_DIR (set by `peaky setup`, = the workspace's "
                         "output/) else ~/peaky-output")
    pb.add_argument("--ts", default=None,
                    help="cached full-batch TS parquet (else fetched live from the server)")
    pb.add_argument("--subject", default=None, help="optional subject phrase for the VK title")
    pb.add_argument("--no-report", action="store_true", help="skip the PDF report")
    pb.add_argument("--select", choices=["representative", "brightest"],
                    default="representative",
                    help="sample-selection strategy: 'representative' (5 time-spaced + "
                         "max-TIC) or 'brightest' (bin all peaks, assign each significant "
                         "m/z bin's brightest sample — better analyte coverage)")
    pb.add_argument("--coverage-target", type=float, default=0.85,
                    help="brightest: fraction of significant m/z bins to cover (default 0.85)")
    pb.add_argument("--k-max", type=int, default=10,
                    help="brightest: max number of winner samples to assign (default 10)")
    pb.add_argument("--height-floor", type=float, default=1000.0,
                    help="brightest: a bin is significant if its max height >= this (cps)")
    pb.add_argument("--jobs", "-j", type=int, default=None,
                    help="assign samples in parallel across N worker processes "
                         "(default: physical cores, capped at the sample count; "
                         "1 = the serial path; env PEAKY_JOBS also honored). Output "
                         "is byte-identical to a serial run.")
    pb.set_defaults(func=cmd_batch)

    pp = sub.add_parser("pool",
                        help="pool many same-chemistry batches (regex) into ONE "
                             "unified ledger + whole-pool report + per-group reports")
    pp.add_argument("--batches", required=True,
                    help="REGEX over batch names to pool, e.g. "
                         "'HR-CIMS 100-500.*zone' (matches the per-zone batches "
                         "of one mode x range). Passed to the server UNescaped.")
    pp.add_argument("--dataset", default=None, help="dataset (workspace) name")
    pp.add_argument("--reagent", default="auto", help="auto | Br | Ur | NO3 | NO3_15N | ...")
    pp.add_argument("--reagent-config", default=None,
                    help="JSON/TOML file registering extra reagent profiles")
    pp.add_argument("--out-name", default=None,
                    help="run-folder label for the unified pool (default: derived "
                         "from the regex)")
    pp.add_argument("--group-by", default="sample_batch_name",
                    help="pooled-peaks column that splits groups for the per-group "
                         "brightest union + per-group reports (default sample_batch_name)")
    pp.add_argument("--out-dir", default=None,
                    help="base output dir for the versioned run folders. Default: "
                         "$PEAKY_OUTPUT_DIR else ~/peaky-output")
    pp.add_argument("--ts", default=None,
                    help="cached pooled TS parquet (must carry --group-by; else "
                         "fetched live for the whole regex)")
    pp.add_argument("--subject", default=None, help="optional subject phrase for the VK title")
    pp.add_argument("--no-report", action="store_true",
                    help="skip every PDF report (assignment + merge only)")
    pp.add_argument("--no-group-reports", action="store_true",
                    help="only the whole-pool report; skip the per-group ones")
    pp.add_argument("--coverage-target", type=float, default=0.90,
                    help="per group: fraction of significant m/z bins to cover (default 0.90)")
    pp.add_argument("--k-max", type=int, default=6,
                    help="per group: max winner samples PER GROUP (default 6; the "
                         "union across groups is what gets assigned)")
    pp.add_argument("--height-floor", type=float, default=1000.0,
                    help="a bin is significant if its max height >= this (cps)")
    pp.add_argument("--jobs", "-j", type=int, default=None,
                    help="assign the union in parallel across N worker processes "
                         "(default: physical cores; env PEAKY_JOBS honored)")
    pp.set_defaults(func=cmd_pool)

    pr = sub.add_parser("report",
                        help="regenerate figures + PDF report from an existing run folder (offline)")
    pr.add_argument("--run-dir", required=True,
                    help="run folder holding merged_ledger.csv + per_file/")
    pr.add_argument("--reagent", required=True, help="Br | Ur | NO3 | ...")
    pr.add_argument("--ts", required=True, help="full-batch TS parquet")
    pr.add_argument("--batch", default=None, help="batch name for the report title")
    pr.add_argument("--tag", default=None, help="filename token (default: reagent name)")
    pr.add_argument("--run-id", default=None, help="Report ID (default: run-dir basename)")
    pr.add_argument("--generated", default=None, help="generated stamp for the cover")
    pr.add_argument("--subject", default=None)
    pr.set_defaults(func=cmd_report)

    pg = sub.add_parser("gka", help="interactive rotating-GKA HTML from a ledger CSV")
    pg.add_argument("ledger_csv")
    pg.add_argument("-o", "--out", default=None)
    pg.add_argument("--ppm", type=float, default=2.0, help="mass accuracy for band width")
    pg.set_defaults(func=cmd_gka)

    ps = sub.add_parser("setup", help="one-command workspace bootstrap "
                                      "(.env + output/ + verify; run once after install)")
    ps.set_defaults(func=cmd_setup)

    pis = sub.add_parser("install-skill",
                         help="register SKILL.md as a Claude Code skill (~/.claude/skills/peaky/)")
    pis.add_argument("--name", default="peaky",
                     help="skill folder name under the skills dir (default: peaky)")
    pis.add_argument("--dir", default=None, help="skills dir (default: ~/.claude/skills)")
    pis.set_defaults(func=cmd_install_skill)

    pc = sub.add_parser("curate", help="create/rename/copy/move workspaces, datasets, "
                                       "batches, samples (write API; --dry-run to preview)")
    pcs = pc.add_subparsers(dest="verb", required=True)

    def _add_dry(p):
        p.add_argument("--dry-run", action="store_true",
                       help="preview only — record the plan, send nothing")
        p.add_argument("--cookie", default=None,
                       help="session cookie for WRITE endpoints (the server gates "
                            "mutations behind a logged-in session; the bearer token "
                            "only authorizes reads). Else $MASCOPE_SESSION_COOKIE. "
                            "Copy the Cookie header from a logged-in browser request.")

    pct = pcs.add_parser("tree", help="show the workspace/dataset[/batch] hierarchy (read-only)")
    pct.add_argument("--workspace", default=None, help="limit to one workspace")
    pct.add_argument("--deep", action="store_true", help="also list batches under each dataset")
    _add_dry(pct)

    pcw = pcs.add_parser("new-workspace", help="create a workspace")
    pcw.add_argument("name"); pcw.add_argument("--desc", default=None); _add_dry(pcw)

    pcd = pcs.add_parser("new-dataset", help="create a dataset in a workspace")
    pcd.add_argument("--workspace", required=True)
    pcd.add_argument("--name", required=True); pcd.add_argument("--desc", default=None)
    _add_dry(pcd)

    pcb = pcs.add_parser("new-batch", help="create an empty batch in a dataset")
    pcb.add_argument("--workspace", required=True); pcb.add_argument("--dataset", required=True)
    pcb.add_argument("--name", required=True)
    pcb.add_argument("--polarity", required=True, choices=["+", "-", "+-"])
    pcb.add_argument("--desc", default=None); _add_dry(pcb)

    pcc = pcs.add_parser("copy-batch", help="copy a batch (with its samples) into a dataset")
    pcc.add_argument("--workspace", required=True); pcc.add_argument("--dataset", required=True)
    pcc.add_argument("--batch", required=True)
    pcc.add_argument("--to-workspace", default=None, help="target workspace (default: same)")
    pcc.add_argument("--to-dataset", required=True); pcc.add_argument("--name", required=True)
    pcc.add_argument("--desc", default=None); _add_dry(pcc)

    for v in ("copy-samples", "move-samples"):
        pcm = pcs.add_parser(v, help=f"{v.split('-')[0]} sample items into another batch")
        pcm.add_argument("--sample-ids", nargs="+", required=True)
        pcm.add_argument("--to-workspace", required=True)
        pcm.add_argument("--to-dataset", required=True); pcm.add_argument("--to-batch", required=True)
        _add_dry(pcm)

    pcr = pcs.add_parser("rename", help="rename / re-describe a workspace, dataset or batch")
    pcr.add_argument("kind", choices=["workspace", "dataset", "batch"])
    pcr.add_argument("target", help="name or id to rename")
    pcr.add_argument("--workspace", default=None, help="owning workspace (dataset/batch)")
    pcr.add_argument("--dataset", default=None, help="owning dataset (batch)")
    pcr.add_argument("--name", default=None); pcr.add_argument("--desc", default=None)
    _add_dry(pcr)

    pcx = pcs.add_parser("delete-batch", help="delete a batch (needs --yes)")
    pcx.add_argument("--workspace", required=True); pcx.add_argument("--dataset", required=True)
    pcx.add_argument("--batch", required=True)
    pcx.add_argument("--yes", action="store_true", help="confirm the deletion")
    _add_dry(pcx)

    pc.set_defaults(func=cmd_curate)

    pm = sub.add_parser("mcp", help="run peaky as an MCP server (ChatGPT Developer "
                                    "Mode / Claude Desktop / Cursor)")
    pm.add_argument("--host", default="127.0.0.1",
                    help="bind host (default 127.0.0.1; a tunnel/ngrok exposes it to ChatGPT)")
    pm.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    pm.add_argument("--transport", default="streamable-http",
                    choices=("streamable-http", "sse", "stdio"),
                    help="MCP transport (default streamable-http = ChatGPT connectors; "
                         "stdio for local Claude Desktop)")
    pm.set_defaults(func=cmd_mcp)

    return ap


def _workspace_root() -> str:
    """The clone/workspace root (holds .env.example + the package). Falls back to
    the cwd for a non-editable install where the package isn't next to the repo."""
    from peaky import paths
    repo = os.path.dirname(paths.PKG_ROOT)  # parent of peaky/ == the clone root
    if not os.path.exists(os.path.join(repo, ".env.example")) \
            and os.path.exists(os.path.join(os.getcwd(), ".env.example")):
        return os.getcwd()
    return repo


def resolve_out_dir(explicit: str | None) -> str:
    """Output base for a batch run. Precedence: --out-dir > $PEAKY_OUTPUT_DIR (set in
    .env by `peaky setup`, pointing at the workspace's output/) > ~/peaky-output."""
    if explicit:
        return os.path.expanduser(explicit)
    from peaky.io import io_mascope as IO
    try:                                  # make PEAKY_OUTPUT_DIR from .env visible
        from dotenv import load_dotenv
        load_dotenv(IO._find_env())
    except Exception:
        pass
    return os.path.expanduser(os.environ.get("PEAKY_OUTPUT_DIR") or "~/peaky-output")


def cmd_setup(args) -> None:
    """One-command workspace bootstrap: create .env, point outputs at output/, verify
    the install (+ connection if creds are set), and print where everything is."""
    import shutil

    import peaky
    from peaky.io import io_mascope as IO

    repo = _workspace_root()
    env_path = os.path.join(repo, ".env")
    example = os.path.join(repo, ".env.example")
    out_dir = os.path.join(repo, "output")

    created = False
    if not os.path.exists(env_path):                       # 1. .env from the template
        if os.path.exists(example):
            shutil.copyfile(example, env_path); created = True
        else:
            open(env_path, "a").close()
    body = open(env_path).read()
    if "PEAKY_OUTPUT_DIR" not in body:                     # 2. outputs -> workspace output/
        with open(env_path, "a") as fh:
            fh.write(f"\n# Where run outputs go (this workspace's output/ folder).\n"
                     f"PEAKY_OUTPUT_DIR={out_dir}\n")
    os.makedirs(out_dir, exist_ok=True)

    print(f"peaky {peaky.__version__} — import OK.")       # 3. verify install
    try:                                                   # 4. connection check (if creds)
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except Exception:
        pass
    conn = "not set yet — edit .env (step 1)"
    if os.environ.get("MASCOPE_URL") and os.environ.get("MASCOPE_ACCESS_TOKEN"):
        try:
            ws = IO.list_workspaces()           # no workspace binding needed
            conn = f"connected — {len(ws)} workspace(s) visible"
        except Exception as e:
            conn = f"creds present but connection failed ({type(e).__name__})"

    print(f"""
Peaky workspace ready:  {repo}/
    .env              your Mascope creds (URL + token){'   <- CREATED, edit it' if created else ''}
    output/           all run outputs land here  (PEAKY_OUTPUT_DIR)
    peaky/  scripts/  the package + helper scripts
    SKILL.md          Claude Code skill instructions (run `peaky install-skill`)
    docs/             ARCHITECTURE / ASSIGNMENT / OUTPUTS / ROADMAP

Credentials: {conn}

Next steps:
  1. edit .env   -> MASCOPE_URL + MASCOPE_ACCESS_TOKEN (from the Mascope web app)
  2. peaky list workspaces
     peaky --workspace "<ws>" list datasets
     peaky --workspace "<ws>" list batches --dataset "<dataset>"
  3. peaky --workspace "<ws>" batch --batch "<name>" --dataset "<dataset>" --reagent <Br|Ur|NO3|...>
     -> a versioned run folder under output/ (ledger + figures + PDF report)
     (one sample only? peaky assign --sample-id <ID> --reagent <Br|Ur|...>)
  4. driving with Claude Code? run `peaky install-skill` once, then just ask:
     "assign formulas for batch <name> with the bromide reagent\"""")


def cmd_install_skill(args) -> None:
    # offline: register SKILL.md as a Claude Code skill at ~/.claude/skills/<name>/.
    # Copy (not symlink) so it works cross-platform with no admin / Developer Mode,
    # and stays lean (just SKILL.md, not the whole repo like a clone-symlink).
    import shutil

    src = Path(_workspace_root()) / "SKILL.md"
    if not src.is_file():
        sys.exit(f"SKILL.md not found at {src}. Run `peaky install-skill` from a "
                 "source checkout (git clone + pip install -e .).")
    skills_dir = Path(args.dir).expanduser() if args.dir \
        else Path.home() / ".claude" / "skills"
    dest = skills_dir / args.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest / "SKILL.md")
    print(f"Installed skill {args.name!r} -> {dest / 'SKILL.md'}")
    print("Restart Claude Code to pick it up. Re-run after editing SKILL.md.")


def _force_utf8_io() -> None:
    """Make stdout/stderr emit UTF-8 regardless of the host console codepage, so a
    non-ASCII character in any output never crashes the CLI with UnicodeEncodeError
    on a legacy Windows console (cp1252). Defensive: reagent labels are ASCII, but
    figure titles / formulas printed elsewhere may still carry unicode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):       # detached / non-text stream
                pass


def main(argv=None) -> int:
    _force_utf8_io()
    args = build_parser().parse_args(argv)
    if args.env:
        os.environ["MASCOPE_ENV"] = os.path.expanduser(args.env)
    if args.workspace:
        os.environ["MASCOPE_WORKSPACE"] = args.workspace
    return _run_guarded(lambda: args.func(args))


if __name__ == "__main__":
    sys.exit(main())
