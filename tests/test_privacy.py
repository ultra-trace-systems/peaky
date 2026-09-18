"""No internal identifier reaches the public repository.

peaky is public; the servers, workspaces, datasets and batches a run touches are
not. The rule was applied by hand three times (c8beb8f, d09029a, ...) and missed
once, when a stacked PR raced the scrub, so it is a test now: every tracked file
goes through scripts/privacy_scan.py, which is a required check on main. The same
scanner reads the pull request's own title and body in CI -- a body is not a file,
so nothing else can see it.

Section 1 pins the RULES against invented examples (a rule that stops matching is
worse than no rule). Section 2 scans the tree.

The examples below are fabricated and carry `privacy-ok:` so the scanner does not
flag its own test -- that pragma is the documented escape hatch for a stand-in
that must keep the shape.

Run: python3 tests/test_privacy.py   (or pytest)
"""
import subprocess
import sys
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import privacy_scan as PS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def rules(text):
    """The rule names that fire on `text`."""
    return sorted({h.rule for h in PS.scan_text(text)})


# ---------------------------------------------------------------------------
# 1. the rules. Every example is invented; none names a real server or batch.
# ---------------------------------------------------------------------------
print("-- rules --")
check("an internal service hostname is refused",
      rules("see northwind.mascope.app for the run") == ["host"])            # privacy-ok: invented
check("... and so is one under the vendor domain",
      rules("http://staging.karsa.fi/x") == ["host"])                        # privacy-ok: invented
check("a public host is not a finding",
      rules("cloned from github.com/ultra-trace-systems/peaky") == [])
check("a private network address is refused",
      rules("the box at 192.168.1.40") == ["private-address"])              # privacy-ok: invented
check("a Mascope id is refused (mixed case, 16 chars)",
      rules('BATCH = "QrmZ4tLpV8nKdW2s"') == ["mascope-id"])                 # privacy-ok: invented
check("... including one with no digits at all",
      rules('"WxYzAbCdEfGhIjKl"') == ["mascope-id"])                         # privacy-ok: invented
check("a same-shape placeholder is the accepted stand-in",
      rules('BATCH_ID = "BATCHxxxxxxxxxxx"') == [])
check("a 16-letter word is not an id",
      rules('"Dibutylphthalate" and Characterisation') == [])
check("a camelCase name is not an id",
      rules("const d = devicePixelRatio || 1") == [])
check("an underscore-joined identifier is not an id",
      rules("isinstance(a, _SubParsersAction)") == [])
check("a git sha is not an id",
      rules("see 4183ff3d117912116f9330b73de0dc536cd3dc9a") == [])
check("a date-ranged batch name is refused",
      rules('batch "Site B Ur 122-600 05-30 - 06-02"') == ["dated-batch-name"])  # privacy-ok: the range IS the shape
check("a plain ISO date is not a finding",
      rules("measured on 2026-08-10, reported 2026-08-14") == [])
check("a workspace named after a person is refused",
      rules('DATASET = "Ada\'s Workspace"') == ["personal-name"])            # privacy-ok: invented
check("... and a role-named one is not",
      rules('DATASET = "Team A\'s Workspace"') == [])
check("... nor is the product itself",
      rules("Mascope's batch listing") == [])
check("the pragma exempts the line it is on",
      rules("host mysite.mascope.app  # privacy-ok: invented example") == [])
check("several classes on one line are all reported",
      len(PS.scan_text('at hub.mascope.app ran "VqTr7xLmNp3wZk9d"')) == 2)   # privacy-ok: invented

# a PR title/body is text, not a file -- the CI step feeds it through this path
_hits = PS.scan_text("Validated against the run on hub.mascope.app", "PR body")  # privacy-ok: invented
check("PR text scans the same way, and says where the finding is",
      len(_hits) == 1 and _hits[0].path == "PR body" and _hits[0].line == 1, _hits)


# a Windows checkout hands scan_file '\'-separated paths, and ALLOWED_PATHS is spelled
# with '/'. A pure Windows path behaves the same on every OS, so Linux CI pins this
# too; each such "file" holds an invented id, so only the allowlist keeps it quiet.
class _WinFile(PureWindowsPath):
    def read_text(self, encoding=None):
        return 'BATCH = "QrmZ4tLpV8nKdW2s"'                                  # privacy-ok: invented


_win = _WinFile(r"C:\src\peaky")
check("an allowlisted file is skipped whatever the path separator",
      PS.scan_file(_win / r"tests\fixtures\match_tree.json", _win) == [])
_win_hits = PS.scan_file(_win / r"tests\fixtures\other.json", _win)
check("... and any other file is scanned, and reported by its '/' path",
      [h.path for h in _win_hits] == ["tests/fixtures/other.json"], _win_hits)

# ---------------------------------------------------------------------------
# 2. the tree itself
# ---------------------------------------------------------------------------
print("-- tracked files --")
try:
    _tracked = PS.repo_files(ROOT)
except (subprocess.CalledProcessError, FileNotFoundError):     # not a checkout
    print("  --  skipped: not a git checkout")
else:
    check("git ls-files found the tree (tracked + about-to-be-added)", len(_tracked) > 50, len(_tracked))
    _found = PS.scan_repo(ROOT)
    check("no tracked file carries an internal identifier",
          not _found, "\n" + "\n".join(str(h) for h in _found[:20]))


def test_all():
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
