"""Offline tests for profiles.py: built-in reagents (Br / Ur / NO3 / NO3_15N), alias
resolution, register(), and config-driven loading (JSON + TOML). The global
registry is snapshotted and restored so this file does not pollute other tests.
Run: python3 tests/test_profiles.py"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peaky import profiles as P  # noqa: E402

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}  {detail}")


_SAVED = (dict(P.PROFILES), dict(P._BY_ALIAS))   # snapshot the registry

# ---- built-in reagents + alias resolution -----------------------------------
check("resolve('Br')", P.resolve("Br").name == "Br")
check("resolve('uronium' alias) -> Ur", P.resolve("uronium").name == "Ur")
check("resolve('NO3') built-in", P.resolve("NO3").name == "NO3")
check("NO3 is negative mode", P.resolve("nitrate").polarity == "-")
check("NO3 analyte channel is [M+NO3]-", "[M+NO3]-" in P.resolve("NO3").adducts)

# ---- 15N-labelled nitrate (distinct from the 14N NO3 profile) ----------------
check("resolve('NO3_15N') built-in", P.resolve("NO3_15N").name == "NO3_15N")
for _a in ("15no3", "^no3-", "nitrate-15n", "15n-nitrate"):
    check(f"alias {_a!r} -> NO3_15N", P.resolve(_a).name == "NO3_15N")
check("15N profile is negative mode", P.resolve("NO3_15N").polarity == "-")
check("15N analyte channel is [M+^NO3]-", "[M+^NO3]-" in P.resolve("NO3_15N").adducts)
check("15N keeps deprotonation channel", "[M-H]-" in P.resolve("NO3_15N").adducts)
check("15N detect_adduct distinguishes it from 14N",
      P.resolve("NO3_15N").detect_adduct == "[M+^NO3]-"
      and P.resolve("NO3").detect_adduct == "[M+NO3]-")
check("plain 'no3' still resolves to 14N NO3, not the 15N profile",
      P.resolve("no3").name == "NO3")
check("15N profile normalises on TIC (reagent ions out of window)",
      P.resolve("NO3_15N").normaliser == "tic")
try:
    P.resolve("xenon"); check("unknown reagent raises", False, "no raise")
except KeyError:
    check("unknown reagent raises KeyError", True)

# ---- iodide (I⁻ CIMS) built-in ----------------------------------------------
check("resolve('I') built-in", P.resolve("I").name == "I")
for _a in ("i", "iodide", "iodine", "i-", "i-cims", "iodide-cims"):
    check(f"alias {_a!r} -> I", P.resolve(_a).name == "I")
check("iodide is negative mode", P.resolve("iodide").polarity == "-")
check("iodide primary channel is [M+I]-", "[M+I]-" in P.resolve("I").adducts)
check("iodide keeps deprotonation channel [M-H]-", "[M-H]-" in P.resolve("I").adducts)
check("iodide keeps the poly-iodide [M+I2]- channel", "[M+I2]-" in P.resolve("I").adducts)
check("iodide detect_adduct is [M+I]-", P.resolve("I").detect_adduct == "[M+I]-")
check("iodide normalises on the (in-window) reagent ion",
      P.resolve("I").normaliser == "reagent")
check("covalent iodine is OFF the neutral grid (monoisotopic)",
      "I" not in P.resolve("I").ranges)

# ---- every built-in profile adduct must be resolvable where it matters -------
# Hard invariants: (1) every profile adduct needs an ADDUCT_SHIFTS entry (ion_mz
# raises otherwise); (2) every detect_adduct needs an ADDUCT_TO_MECH mapping
# (auto-detect reads MECH_TO_ADDUCT -- unmapped means detection can NEVER fire).
# NB: not every adduct needs a mechanism -- [M+HBr+Br]- is a cluster-DECOMPOSITION
# adduct (same ion as a covalent reading, relabel-only, deliberately unmapped).
# assign.py filters channels with `if a in ADDUCT_TO_MECH`, so for the iodide
# profile (all three channels server-scored) pin each mapping explicitly: a
# dropped mapping silently disables the channel in live runs.
from peaky import io_mascope as IOM   # noqa: E402
from peaky import chemistry as CHEM   # noqa: E402
for _p in P.PROFILES.values():
    check(f"{_p.name}: all adducts in ADDUCT_SHIFTS",
          all(a in CHEM.ADDUCT_SHIFTS for a in _p.adducts),
          [a for a in _p.adducts if a not in CHEM.ADDUCT_SHIFTS])
    check(f"{_p.name}: detect_adduct is mechanism-mapped (auto-detect works)",
          _p.detect_adduct in IOM.ADDUCT_TO_MECH, _p.detect_adduct)
for _a, _m in (("[M+I]-", "+I-"), ("[M-H]-", "-H+"), ("[M+I2]-", "+I2-"),
               ("[M+I3]-", "+I3-")):
    check(f"iodide channel {_a} maps to server mechanism {_m}",
          IOM.ADDUCT_TO_MECH.get(_a) == _m)

# ---- register a new profile in code -----------------------------------------
acet = P.ReagentProfile(
    name="Ac", label="Acetate⁻", polarity="-", adducts=["[M+CH3COO]-", "[M-H]-"],
    normaliser="reagent", reagent_ion_re=r"C2H3O2-?$", ranges="C0-30 H0-50 O0-15",
    detect_adduct="[M+CH3COO]-", aliases=("acetate", "ac-"))
P.register(acet)
check("register() -> resolve by name", P.resolve("Ac").name == "Ac")
check("register() -> resolve by alias", P.resolve("acetate").name == "Ac")

# ---- config-driven loading (JSON + TOML) ------------------------------------
with tempfile.TemporaryDirectory() as d:
    cfgj = os.path.join(d, "r.json")
    json.dump([{"name": "Cust", "label": "custom⁻", "polarity": "-",
                "adducts": ["[M+X]-", "[M-H]-"], "normaliser": "reagent",
                "reagent_ion_re": "X-?$", "ranges": "C0-30 H0-50 O0-12",
                "detect_adduct": "[M+X]-", "aliases": ["custom-reagent"]}], open(cfgj, "w"))
    P.load_config(cfgj)
    check("load_config(JSON list) registers", P.resolve("custom-reagent").name == "Cust")
    check("loaded aliases are a tuple", isinstance(P.resolve("Cust").aliases, tuple))

    cfgw = os.path.join(d, "r2.json")
    json.dump({"reagents": [{"name": "Qx", "label": "Qx", "polarity": "-",
               "adducts": ["[M+Qx]-"], "normaliser": "tic", "reagent_ion_re": None,
               "ranges": "C0-10 H0-20", "detect_adduct": "[M+Qx]-"}]}, open(cfgw, "w"))
    check("resolve(config=) loads then resolves", P.resolve("Qx", config=cfgw).name == "Qx")

    cfgt = os.path.join(d, "r.toml")
    open(cfgt, "w").write(
        '[[reagents]]\nname="Tz"\nlabel="Tz"\npolarity="-"\nadducts=["[M+Tz]-"]\n'
        'normaliser="tic"\nreagent_ion_re=""\nranges="C0-5 H0-10"\ndetect_adduct="[M+Tz]-"\n')
    P.load_config(cfgt)
    check("load_config(TOML) registers", P.resolve("Tz").name == "Tz")

    # a user config may deliberately SHADOW a built-in alias (register overwrite
    # semantics): 'iodide' -> the custom profile until the registry is restored.
    cfgs = os.path.join(d, "shadow.json")
    json.dump([{"name": "MyI", "label": "my iodide", "polarity": "-",
                "adducts": ["[M+I]-"], "normaliser": "tic", "reagent_ion_re": None,
                "ranges": "C0-10 H0-20", "detect_adduct": "[M+I]-",
                "aliases": ["iodide"]}], open(cfgs, "w"))
    P.load_config(cfgs)
    check("user config SHADOWS the built-in 'iodide' alias",
          P.resolve("iodide").name == "MyI")
    check("built-in name 'I' itself is untouched by the alias shadow",
          P.resolve("I").name == "I")

# ---- restore the registry (no cross-test pollution) -------------------------
P.PROFILES.clear(); P.PROFILES.update(_SAVED[0])
P._BY_ALIAS.clear(); P._BY_ALIAS.update(_SAVED[1])
check("registry restored (Ac gone after cleanup)", "Ac" not in P.PROFILES and "ac-" not in P._BY_ALIAS)
check("built-ins intact after restore", {"Br", "Ur", "NO3", "NO3_15N", "I"} <= set(P.PROFILES))


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
