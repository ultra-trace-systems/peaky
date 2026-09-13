"""Assignment-level determinism (#8): the deep-series tie-break must not depend
on PYTHONHASHSEED.

`ARCHITECTURE.md` promises a run's scientific content is a pure function of its
inputs. It was not: residual stage B collected its anchors into a set and
`propose_for_peak` walked that set, so when two anchors reached the same
candidate with identical support and ppm (C4H6O4 + CH2 and C6H10O4 - CH2
both give C5H8O4) the anchor named in the Pass-4 commentary was whichever the
set yielded first -- a different one per process. The formula, score and tier
never varied; the ledger bytes did. test_determinism.py covers only the report
layer, so nothing caught it.

This drives stage B in SUBPROCESSES under six different hash seeds (an in-process
check cannot vary the seed) and requires byte-identical ledgers, and it pins the
tie-break itself: the fewest steps, then the anchor text. Run: python3 tests/test_determinism_assign.py
"""
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from peaky.assignment import series_gka as G  # noqa: E402
from peaky.chem import chemistry as C  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# The scenario, as a script: three High anchors on one CH2 ladder (C4 / C6 / C7
# dicarboxylic acids), one unexplained peak at the C5 rung, a stub scorer that
# returns the C5 reading. Every anchor reaches C5H8O4 (+1, -1, -2 steps) with the
# same support (2: the C4 and C6 anchors sit one CH2 either side) and the same
# ppm (identical candidate), so the recorded anchor is pure tie-break.
SCRIPT = textwrap.dedent("""
    import hashlib, sys
    import pandas as pd
    from peaky.assignment import residual as RD
    from peaky.assignment import ledger as L
    from peaky.chem import contexts as X
    from peaky.chem import chemistry as C
    from peaky.assignment.passes import PassConfig

    CFG = PassConfig(height_cutoff_cps=100)
    PROF = X.get_context("ambient-air")
    T = "C5H8O4"; AD = "[M-H]-"; MZ_T = C.ion_mz(T, AD)

    def scorer(client, sample_id, formulas, *, mechanism_ids=None, **kw):
        rows = []
        if T in formulas:
            ion = "C5H7O4-"
            rows.append(dict(compound_formula=T, compound_score=0.85, compound_category=2,
                             ion_formula=ion, ion_score=0.85, ion_category=2, mechanism_id="m",
                             isotope_formula=ion, iso_label="M0", is_base=True, theo_mz=MZ_T,
                             rel_abundance=1.0, iso_score=0.9, iso_category=2,
                             sample_peak_id="T1", sample_peak_mz=MZ_T, sample_peak_intensity=2e3,
                             ppm_error=0.4, abundance_error=0.0))
        return pd.DataFrame(rows)

    anchors = ["C4H6O4", "C6H10O4", "C7H12O4"]
    pk = pd.DataFrame({"peak_id": [f"A{i}" for i in range(3)] + ["T1"],
                       "mz": [C.ion_mz(a, AD) for a in anchors] + [MZ_T],
                       "height": [1e5, 1e5, 1e5, 2e3]})
    led = L.new_ledger(pk)
    for i, a in enumerate(anchors):
        L.commit_assignment(led, f"A{i}", neutral_formula=a, adduct=AD, ion_score=0.97,
                            pass_no=1, method="x", confidence="High", commentary="anchor")
    res = RD.stage_b_series(None, "SID", led, PROF, CFG, [AD], reagent="Br",
                            score_fn=scorer, log=lambda *_: None)
    row = led.loc[led.peak_id == "T1"].iloc[0]
    print("committed", res["committed"], "|", row["neutral_formula"], "|", row["commentary"])
    print("sha1", hashlib.sha1(led.to_csv(index=False).encode()).hexdigest())
""")

outs = []
for seed in range(6):
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=str(ROOT), capture_output=True,
                       text=True, env={**os.environ, "PYTHONHASHSEED": str(seed)}, timeout=120)
    check(f"seed {seed}: stage B runs", r.returncode == 0, r.stderr[-400:])
    outs.append(r.stdout.strip())
first = outs[0].splitlines() if outs and outs[0] else [""]
check("the deep-series peak is committed with the C5 reading under every seed",
      all(o.startswith("committed 1 | C5H8O4 |") for o in outs), [o[:90] for o in outs])
check("the ledger is byte-identical under all six hash seeds",
      len(set(outs)) == 1, [o.splitlines()[0][:90] if o else "" for o in outs])
check("the tie-break names a one-step anchor, the lower one (+1 x CH2 from C4H6O4), never whichever came first",
      "+1xCH2 from anchor C4H6O4 (2 supporting anchors)" in first[0], first[:1])

# the proposal order itself is a pure function of the inputs
MZ = C.ion_mz("C5H8O4", "[M-H]-")
a = G.propose_for_peak(MZ, {"C4H6O4", "C6H10O4", "C7H12O4"}, ["[M-H]-"], ppm=3.0, max_steps=3)
b = G.propose_for_peak(MZ, ["C7H12O4", "C6H10O4", "C4H6O4"], ["[M-H]-"], ppm=3.0, max_steps=3)
check("propose_for_peak: the same proposals in the same order whatever the anchor container / order",
      [(p.anchor_formula, p.n_steps) for p in a] == [(p.anchor_formula, p.n_steps) for p in b]
      == [("C4H6O4", 1), ("C6H10O4", -1), ("C7H12O4", -2)],
      [(p.anchor_formula, p.n_steps) for p in a])
check("propose_for_peak: support and ppm still rank first (a 2-anchor candidate beats a 1-anchor one)",
      [p.n_supporting_anchors for p in G.propose_for_peak(
          C.ion_mz("C10H16O5", "[M-H]-"), {"C10H16O4", "C10H16O6", "C8H12O5"}, ["[M-H]-"],
          ppm=3.0, max_steps=1)][:1] == [2])


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
