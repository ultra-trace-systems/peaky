"""Certified-neutral discovery over an existing ledger CSV — standalone.

Finds groups of UNEXPLAINED peaks whose back-calculated neutral masses converge
across >=2 ion channels (adducts and/or reagent-cluster ladder rungs), then
enumerates the expanded element box (P/S/Cl open) for each certified core.
Ground truth this reproduces: the NBBS urea ladder (214.0896/274.1220/334.1544
-> core 213.0823 = C10H15NO2S) and cross-channel malathion (C10H19O6PS2).

OFFLINE (default): pure mass-domain — no server, no credentials. Prints/writes
the certificate table with candidate formulas.

    python3 scripts/certify_neutrals.py RUN_DIR/merged_ledger.csv --reagent Ur
    python3 scripts/certify_neutrals.py LEDGER.csv --reagent Ur \
        --ts RUN_DIR/data/Ur_ts.parquet -o certificates.csv

FULL (--sample-id): additionally scores the candidates through the Mascope
oracle and commits winners into a copy of the ledger (the in-pipeline pass-7,
run standalone).

    python3 scripts/certify_neutrals.py LEDGER.csv --reagent Ur \
        --sample-id <SID> -o certified_ledger.csv
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peaky.assignment import certified_neutral as CN          # noqa: E402
from peaky.chem import contexts as XC                          # noqa: E402
from peaky.chem import profiles as PR                          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledger", help="ledger CSV (single-sample or merged)")
    ap.add_argument("--reagent", default="auto",
                    help="Br | Ur | NO3 | ... | auto (default)")
    ap.add_argument("--ts", default=None,
                    help="optional batch TS parquet for co-variation corroboration")
    ap.add_argument("--sample-id", default=None,
                    help="score + commit via the live oracle (FULL mode)")
    ap.add_argument("--tol-mda", type=float, default=CN.CORE_TOL_MDA,
                    help="core convergence tolerance, mDa (default %(default)s)")
    ap.add_argument("--min-channels", type=int, default=2)
    ap.add_argument("-o", "--out", default=None, help="output CSV path")
    args = ap.parse_args(argv)

    led = pd.read_csv(args.ledger, low_memory=False)
    # unexplained rows only (merged ledgers lack `role`; treat formula-less as such)
    if "role" in led.columns:
        un = led[led["role"] == "unexplained"]
    else:
        fcol = "neutral_formula" if "neutral_formula" in led.columns else None
        un = led[led[fcol].isna()] if fcol else led
    if "peak_id" not in un.columns:
        un = un.assign(peak_id=[f"p{i}" for i in range(len(un))])
    if "height" not in un.columns:
        un = un.assign(height=1.0)
    print(f"[certify] {len(un)} unexplained peaks of {len(led)} ledger rows")

    # resolve the reagent -> adducts come from the ReagentProfile, grid caps
    # from its ContextProfile (they are different objects — see docs)
    rp = PR.resolve(args.reagent,
                    peaks=un if args.reagent == "auto" else None)
    profile = XC.get_context(rp.context)
    cluster_reagent = "urea" if str(rp.polarity) in ("+", "positive") else None
    offsets = CN.channel_offsets(list(rp.adducts), cluster_reagent)
    print(f"[certify] profile {rp.label}: {len(offsets)} channels "
          f"({', '.join(a for a, n, _ in offsets[:6])}...)")

    certs = CN.find_certificates(
        un[["peak_id", "mz", "height"]].reset_index(drop=True), offsets,
        tol_mda=args.tol_mda, min_channels=args.min_channels)
    print(f"[certify] {len(certs)} certified cores")

    ts = pd.read_parquet(args.ts) if args.ts else None
    rows = []
    for c in certs:
        forms = CN.enumerate_certified(c.core_mass, profile, force=True)
        forms = [f for f in forms if any(el in f for el in ("P", "S", "Cl"))]
        cov = (CN.ts_covariation(ts, [h.mz for h in c.hits])
               if ts is not None else None)
        rows.append({
            "core_mass": round(c.core_mass, 5),
            "n_channels": c.n_channels,
            "spread_mDa": round(c.spread_mda, 3),
            "member_mzs": ";".join(f"{h.mz:.4f}" for h in c.hits),
            "channels": ";".join(f"{h.adduct}+{h.cluster_order}R" if h.cluster_order
                                 else h.adduct for h in c.hits),
            "ts_covary_rmin": None if cov is None else round(cov, 2),
            "n_offgrid_candidates": len(forms),
            "offgrid_candidates": ";".join(forms[:8]),
        })
    table = pd.DataFrame(rows).sort_values(
        ["n_channels", "core_mass"], ascending=[False, True]) if rows else pd.DataFrame()
    if len(table):
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(table.head(30).to_string(index=False))

    if args.sample_id:
        # FULL mode: run the in-pipeline pass on a live-scored copy of the ledger
        from peaky.assignment import ledger as L
        from peaky.assignment import passes as P
        from peaky.io import io_mascope as IO
        client = IO.connect()
        led_run = L.new_ledger(un[["peak_id", "mz", "height"]].reset_index(drop=True))
        cfg = P.PassConfig()
        # Pass 7 is not height-gated (it reads the residual's peaks, never
        # cfg.height_cutoff), so this resolution changes nothing today. It is
        # here so that this entry point -- which resolves a profile and builds a
        # PassConfig, like every other one -- is not the single exception if the
        # pass ever starts gating. One rule, one place (REAGENTS.md 3a). No
        # `log=`: a "[gate] ..." line here would announce a gate that isn't used.
        PR.apply_height_cutoff_x_edge(cfg, rp)
        cfg.mechanism_ids = IO.resolve_mechanism_ids(client, list(rp.adducts))
        s = P.run_pass_certified(client, args.sample_id, led_run, profile, cfg,
                                 list(rp.adducts), reagent=cluster_reagent,
                                 ts_peaks=ts)
        print(f"[certify] pass-7 summary: {s}")
        committed = led_run[led_run["role"] == "M0"]
        if len(committed):
            print(committed[["peak_id", "mz", "neutral_formula", "adduct",
                             "confidence", "method"]].to_string(index=False))
        if args.out:
            led_run.to_csv(args.out, index=False)
            print(f"[certify] wrote {args.out}")
            return
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[certify] wrote {args.out}")


if __name__ == "__main__":
    main()
