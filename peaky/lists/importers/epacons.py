"""NORMAN-SLE S22 EPACONS: the US EPA consumer-product suspect list.

Source: Phillips, K. A. et al. (2018), "Suspect Screening Analysis of Chemicals
in Consumer Products", Environ. Sci. Technol. 52, 3125; distributed by the NORMAN
Suspect List Exchange as list S22 (Zenodo record 2648759, concept DOI
10.5281/zenodo.2648758, licence CC-BY-4.0). 1024 substances exported from the
EPA CompTox dashboard: name, CASRN, InChIKey, SMILES, formula, masses, DTXSID.

Why this list first: it names the consumer-product chemistry a urea/ammonium
CIMS sees in a city (glycol ethers, Texanol, DEET, siloxanes, plasticizers) AND
the organophosphate pesticides (malathion) the CHON grid cannot reach -- the
2026-11 Texas case -- from a source that is not peaky's own campaign.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone

from peaky.chem import chemistry as C

__version__ = "0.1.0"

SOURCE_ID = "norman_sle_s22_epacons"
LIST_ID = SOURCE_ID

RECORD_ID = 2648759
CONCEPT_DOI = "10.5281/zenodo.2648758"
DOI = "10.5281/zenodo.2648759"
VERSION = "NORMAN-SLE-S22.0.1.0"
FILE = "EPA_ConsumerProductSuspects_01032018.csv"
BYTES = 482416
MD5 = "b44bc1224bf080ead559fc2f5ba7ca33"
LICENCE = "CC-BY-4.0"
ZENODO_API = "https://zenodo.org/api/records/{id}"
URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/{FILE}/content"

#: the exact header the parser was written for; any difference is a HeaderMismatch
HEADER = ("IDENTIFIER", "DTXSID", "PREFERRED NAME", "CASRN", "INCHI KEY", "IUPAC NAME", "SMILES",
          "INCHI STRING", "MOL FORMULA", "AVERAGE MASS", "MONOISOTOPIC MASS", "QC LEVEL",
          "DATA SOURCES", "TOXCAST % ACTIVE", "TOXCAST # ASSAYS/TOTAL")

ATTRIBUTION = ("NORMAN Suspect List Exchange, S22 | EPACONS | US EPA Consumer Product Suspect List "
               "(Phillips et al. 2018), https://doi.org/10.5281/zenodo.2648758, CC-BY-4.0. "
               "Formulas re-derived in Hill notation; masses recomputed by peaky; entries that are not "
               "a neutral molecule over peaky's mass table are listed under provenance.excluded.")

REFERENCES = [
    {"authors": "Phillips, K. A.; Yau, A.; Favela, K. A.; Isaacs, K. K.; McEachran, A.; Grulke, C.; "
                "Richard, A. M.; Williams, A. J.; Sobus, J. R.; Thomas, R. S.; Wambaugh, J. F.",
     "title": "Suspect Screening Analysis of Chemicals in Consumer Products",
     "publisher": "Environ. Sci. Technol. 52, 3125-3135", "year": 2018,
     "doi": "10.1021/acs.est.7b04781"},
    {"authors": "NORMAN Suspect List Exchange", "title": "S22 | EPACONS | US EPA Consumer Product Suspect List",
     "publisher": "Zenodo", "year": 2018, "doi": CONCEPT_DOI, "version": VERSION},
]

#: the mass-check tolerance: the source's monoisotopic mass vs peaky's own; a
#: larger gap means the formula column and the structure disagree (recorded, not fatal)
MASS_WARN_DA = 0.001


class ImporterError(RuntimeError):
    pass


class HeaderMismatch(ImporterError):
    pass


class ChecksumMismatch(ImporterError):
    pass


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ fetch (network; importer only)
def fetch(dest_dir: str, *, log=print) -> tuple[str, dict]:
    """Download the record's file into `dest_dir` and verify it against BOTH the
    constants above and the record's own file entry (size + md5). Returns
    (path, record_json). The only network code in the list layer."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API.format(id=RECORD_ID), timeout=60) as r:
        record = json.loads(r.read().decode("utf-8"))
    files = {f.get("key"): f for f in record.get("files", [])}
    if FILE not in files:
        raise ImporterError(f"record {RECORD_ID} has no file {FILE!r}; files: {sorted(files)}")
    entry = files[FILE]
    rec_md5 = str(entry.get("checksum", "")).split(":")[-1]
    if rec_md5 != MD5 or int(entry.get("size", -1)) != BYTES:
        raise ChecksumMismatch(f"record {RECORD_ID}/{FILE}: Zenodo says md5 {rec_md5} size {entry.get('size')}, "
                               f"this importer was written for md5 {MD5} size {BYTES} -- a new version? "
                               "update the constants deliberately")
    path = os.path.join(dest_dir, FILE)
    log(f"[lists import] fetching {URL} -> {path}")
    with urllib.request.urlopen(URL, timeout=120) as r, open(path, "wb") as fh:
        fh.write(r.read())
    verify_file(path)
    return path, record


def verify_file(path: str) -> None:
    size = os.path.getsize(path)
    got = md5_of(path)
    if size != BYTES or got != MD5:
        raise ChecksumMismatch(f"{path}: size {size} md5 {got}; expected size {BYTES} md5 {MD5}")


# ------------------------------------------------------------------ parse (pure)
_PPM_HAZARD = re.compile(r",\s*ppm\b", re.IGNORECASE)


def _clean_name(name: str) -> str:
    """A display name the batch's commentary parser cannot misread: the known-species
    label regex (`assign_batch._KNOWN_LABEL_RE`) ends a label at ', ppm '."""
    return _PPM_HAZARD.sub(" ppm", str(name or "").strip())


def parse_rows(rows, *, header) -> tuple[list, list, list]:
    """(species, excluded, mass_warnings) from an iterable of CSV rows whose first
    row is the header. The header must equal `HEADER` exactly."""
    from peaky.assignment import reflists as RL
    header = [h.strip() for h in header]
    if tuple(header) != HEADER:
        missing = [h for h in HEADER if h not in header]
        extra = [h for h in header if h not in HEADER]
        raise HeaderMismatch(f"{FILE}: header differs from the one this importer was written for; "
                             f"missing {missing}, unexpected {extra}, order {header}")
    species, excluded, warn = [], [], []
    for raw in rows:
        if not raw or all(not str(x).strip() for x in raw):
            continue
        row = dict(zip(HEADER, raw))
        fraw = str(row.get("MOL FORMULA") or "").strip()
        name = _clean_name(row.get("PREFERRED NAME"))
        dtxsid = str(row.get("DTXSID") or "").strip() or None
        if not fraw or fraw == "-":
            # the dashboard writes '-' for a substance with no single structure
            # (a mixture / UVCB: 'Alkanes, C4-12', 'Polysorbate 80', isomer blends)
            excluded.append({"formula_raw": fraw, "name": name, "dtxsid": dtxsid,
                             "reason": "no formula (mixture / UVCB, no single structure)"})
            continue
        cnt = C.parse_formula(fraw)
        formula = C.format_formula(cnt)
        reason = RL._not_a_neutral(formula) if formula else "unparseable formula"
        if reason:
            excluded.append({"formula_raw": fraw, "name": name, "dtxsid": dtxsid, "reason": reason})
            continue
        try:
            src_mass = float(row.get("MONOISOTOPIC MASS") or "nan")
        except ValueError:
            src_mass = float("nan")
        own = C.neutral_mass(formula)
        if src_mass == src_mass and abs(src_mass - own) > MASS_WARN_DA:
            warn.append({"formula": formula, "name": name, "source_mass": src_mass, "peaky_mass": round(own, 6)})
        ik = str(row.get("INCHI KEY") or "").strip() or None
        species.append({
            "formula": formula,
            "name": name,
            "inchikey": ik,
            "inchikey_block": ik.split("-")[0] if ik else None,
            "cas": str(row.get("CASRN") or "").strip() or None,
            "ids": {"dtxsid": dtxsid},
            "radical": bool(C.odd_electron(formula)),
        })
    return species, excluded, warn


#: the file is a Windows (CompTox dashboard) export: cp1252, not UTF-8 (a 0xA0
#: no-break space in a name breaks a UTF-8 read); recorded in source.encoding
ENCODING = "cp1252"


def parse_csv(path: str) -> tuple[list, list, list]:
    with open(path, encoding=ENCODING, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return parse_rows(reader, header=header)


def parse_text(text: str) -> tuple[list, list, list]:
    reader = csv.reader(io.StringIO(text))
    return parse_rows(reader, header=next(reader))


# ------------------------------------------------------------------ build (pure)
def build_snapshot(species, excluded, mass_warnings, *, retrieved_utc: str | None = None,
                   record: dict | None = None, source_path: str | None = None) -> dict:
    from peaky.lists import snapshot as SN
    retrieved_utc = retrieved_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d = {
        "schema_version": SN.SCHEMA_V2,
        "id": LIST_ID,
        "system": "consumer_products",
        "label": "US EPA consumer-product suspects (NORMAN-SLE S22 EPACONS, Phillips et al. 2018)",
        "data_version": VERSION,
        "polarity": "both",
        "native_detection": "GCxGC-TOFMS / LC-HRMS suspect screening of consumer products (not CIMS)",
        "always_active": False,
        "applies_to_contexts": ["consumer_products", "indoor_air", "urban_air"],
        "roles": ["name"],
        "family_scope": "compound",
        "source": {
            "kind": "norman-sle",
            "doi": DOI,
            "concept_doi": CONCEPT_DOI,
            "record_id": RECORD_ID,
            "version": VERSION,
            "file": FILE,
            "bytes": BYTES,
            "md5": MD5,
            "retrieved_utc": retrieved_utc,
            "licence": LICENCE,
            "url": URL,
            "encoding": ENCODING,
            "publication_date": (record or {}).get("metadata", {}).get("publication_date"),
        },
        "importer": {"name": f"peaky.lists.importers.epacons", "version": __version__},
        "references": REFERENCES,
        "provenance": {
            "attribution": ATTRIBUTION,
            "extracted_from": source_path or FILE,
            "verification": "Zenodo record file entry (size + md5) and the importer's constants; exact header asserted",
            "excluded": excluded,
            "n_excluded": len(excluded),
            "mass_warnings": mass_warnings,
            "n_mass_warnings": len(mass_warnings),
            "caveat": "a suspect list of consumer-product ingredients: a name is a context-list entry "
                      "(level 3c at most), never evidence by itself; isomers are open",
        },
        "n_species": len(species),
        "species": species,
    }
    d["sha256"] = SN.species_sha256(species)
    return d


# ------------------------------------------------------------------ run (the CLI verb)
def run(*, dest_dir: str, from_file: str | None = None, raw_dir: str | None = None, log=print) -> str:
    """Import into `dest_dir` (snapshot JSON + lock entry). `from_file` skips the
    download (an already-fetched CSV; still checksum-verified). Returns the
    snapshot path."""
    from peaky.lists import lock as LK
    from peaky.lists import snapshot as SN
    record = None
    if from_file:
        path = from_file
        verify_file(path)
    else:
        path, record = fetch(raw_dir or os.path.join(dest_dir, "_raw"), log=log)
    species, excluded, warn = parse_csv(path)
    d = build_snapshot(species, excluded, warn, record=record, source_path=os.path.basename(path))
    errs = SN.validate_snapshot(d)
    if errs:
        raise ImporterError("built snapshot is invalid: " + "; ".join(errs))
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"{LIST_ID}.json")
    SN.write_snapshot(out, d)
    LK.write_lock(dest_dir, LK.update_lock(LK.read_lock(dest_dir), d, out))
    log(f"[lists import] {LIST_ID}: {len(species)} species ({len(excluded)} excluded, "
        f"{len(warn)} mass warnings) -> {out}; sha256 {d['sha256'][:12]}; lock updated")
    return out
