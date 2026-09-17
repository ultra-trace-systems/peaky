#!/usr/bin/env python3
"""Refuse to publish internal identifiers.

peaky is a PUBLIC repository, and the things a run touches -- servers, workspaces,
datasets, batches, samples -- are NOT. The rule (c8beb8f, d09029a) has been applied
by hand three times now and missed once (a stacked PR raced the scrub), so it lives
here instead: shape-based rules, enforced by tests/test_privacy.py over every
tracked file and by a CI step over the pull request's own title and body.

WHY SHAPES AND NOT NAMES: a denylist of the actual server and workspace names would
have to live in this public file to work, which publishes exactly what it is meant
to hide. So the rules here match the SHAPE of an internal identifier -- a service
hostname, a 16-character Mascope id, a private address, a date-ranged batch name --
and never a literal secret. A denylist of real names belongs in a private checkout
hook, not in the repository.

Escape hatch: put `privacy-ok: <reason>` in a comment on the offending line. Use it
for a stand-in that has to keep the shape (a fixture batch name with a date range),
never to silence a real identifier.

    python scripts/privacy_scan.py                 # every tracked text file
    python scripts/privacy_scan.py FILE [FILE ...]
    python scripts/privacy_scan.py --stdin         # a PR title/body on stdin

Exit status 1 if anything is found.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PRAGMA = re.compile(r"privacy-ok\s*:", re.I)

# A 16-character Mascope id (dataset, batch, sample, target): base62, no
# separators, and -- being random -- ALTERNATING in case where a word is not.
# 'Dibutylphthalate' and 'devicePixelRatio' are also 16 mixed-case characters, so
# the length alone cannot decide; an id either carries a digit beside both cases
# or breaks case at least CASE_RUNS times ('devicePixelRatio' breaks 5). A placeholder ('DSxxxxxxxxxxxxxx',
# 'SIB11xxxxxxxxxxx') is the accepted stand-in and is exempt by its xxxx run.
_ID_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9]{16}(?![A-Za-z0-9_])")
CASE_RUNS = 6


def _case_runs(tok: str) -> int:
    """How many times the token switches upper/lower (digits keep the run)."""
    runs, prev = 0, None
    for ch in tok:
        if not ch.isalpha():
            continue
        cur = ch.isupper()
        if cur != prev:
            runs += 1
            prev = cur
    return runs


def looks_like_id(tok: str) -> bool:
    """True for a Mascope id, False for a 16-letter word or a camelCase name."""
    if "xxxx" in tok.lower():                                   # the placeholder
        return False
    has_digit = any(c.isdigit() for c in tok)
    has_upper = any(c.isupper() for c in tok)
    has_lower = any(c.islower() for c in tok)
    return (has_digit and has_upper and has_lower) or _case_runs(tok) >= CASE_RUNS


# A workspace or dataset named after a PERSON ("Ada's Workspace"). privacy-ok: invented. The possessor
# is a name unless it is a public product or vendor -- that list is an allowlist of
# things already published here, never a denylist of the names being hidden.
_PERSONAL = re.compile(r"\b([A-Z][a-z]{2,})['\u2019]s\s+"
                       r"(?:[Ww]orkspace|[Dd]ataset|[Bb]atch|[Ff]older|[Cc]ampaign|[Ss]erver)\b")
ALLOWED_POSSESSORS = frozenset({"mascope", "peaky", "orbion", "python", "github",
                                "pandas", "numpy", "argparse", "today"})


def looks_personal(found: str) -> bool:
    return found.split("'")[0].split("\u2019")[0].lower() not in ALLOWED_POSSESSORS


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    says: str
    keep: object = None          # match -> bool; None keeps every match


RULES = (
    Rule("host",
         re.compile(r"\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.(?:mascope|karsa)\.[a-z]{2,6}\b", re.I),
         "an internal service hostname -- describe the server's ROLE instead"),
    Rule("private-address",
         re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
         "a private network address"),
    Rule("mascope-id",
         _ID_TOKEN,
         "a Mascope id -- use a same-shape placeholder ('DSxxxxxxxxxxxxxx')",
         keep=looks_like_id),
    Rule("personal-name",
         _PERSONAL,
         "a workspace named after a person -- name the ROLE ('Team A\'s Workspace')",
         keep=looks_personal),
    Rule("dated-batch-name",
         re.compile(r"\b\d{2}-\d{2}\s+-\s+\d{2}-\d{2}\b"),
         "a date-ranged batch name -- name the SHAPE, not the campaign"),
)

# Captured server payloads whose ids are library rows, not run identifiers: the
# compound/ion/isotope ids of the match tree name no server, workspace or campaign.
ALLOWED_PATHS = frozenset({"tests/fixtures/match_tree.json"})

SKIP_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".xlsx", ".parquet", ".ico",
    ".woff", ".woff2", ".zip", ".gz", ".pyc", ".lock",
})


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    rule: str
    says: str
    found: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.found!r} is {self.says}"


def scan_text(text: str, path: str = "<text>") -> list[Hit]:
    """Every rule hit in `text`, minus placeholders and pragma'd lines."""
    hits: list[Hit] = []
    for n, line in enumerate(text.splitlines(), 1):
        if PRAGMA.search(line):
            continue
        for rule in RULES:
            for m in rule.pattern.finditer(line):
                if rule.keep and not rule.keep(m.group(0)):
                    continue
                hits.append(Hit(path, n, rule.name, rule.says, m.group(0)))
                break
    return hits


def scan_file(path: Path, root: Path | None = None) -> list[Hit]:
    rel = str(path.relative_to(root)) if root else str(path)
    if rel in ALLOWED_PATHS or path.suffix.lower() in SKIP_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []                      # binary or unreadable: nothing to publish
    return scan_text(text, rel)


def repo_files(root: Path) -> list[Path]:
    """Tracked files PLUS untracked, non-ignored ones -- a file that is about to be
    added has had no review at all, so it is the one most worth scanning."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z",
                          "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout
    return sorted({root / p for p in out.split("\0") if p})


def scan_repo(root: Path) -> list[Hit]:
    hits: list[Hit] = []
    for f in repo_files(root):
        if f.is_file():
            hits.extend(scan_file(f, root))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--stdin", action="store_true",
                    help="scan text on stdin (a PR title/body) instead of files")
    ap.add_argument("--label", default="<stdin>", help="what the stdin text is, for the report")
    a = ap.parse_args(argv)

    if a.stdin:
        hits = scan_text(sys.stdin.read(), a.label)
    elif a.files:
        hits = [h for f in a.files for h in scan_file(f)]
    else:
        hits = scan_repo(Path(__file__).resolve().parents[1])

    for h in hits:
        print(h, file=sys.stderr)
    if hits:
        print(f"\n{len(hits)} internal identifier(s) found. peaky is PUBLIC: neutralize "
              f"them, or mark a deliberate stand-in with a `privacy-ok: <reason>` comment.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
