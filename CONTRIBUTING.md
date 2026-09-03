# Contributing to Peaky

Thank you for your interest in contributing! This document covers how to get a
development setup, how to run the tests, our conventions, and the one-time
Contributor License Agreement.

## Development setup

```bash
git clone https://github.com/ultra-trace-systems/peaky.git && cd peaky
python3 -m pip install -e ".[dev]"   # or: uv sync --extra dev
peaky setup                          # creates .env + output/, checks the workspace
```

Needs Python 3.12 or later. Everything installs from public PyPI; a Mascope
account is only needed to run against real data (see
[QUICKSTART.md](QUICKSTART.md) for the credentials step). Start with
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the ledger model, the pass
sequence, and the module map.

## Running tests

```bash
pytest tests/
```

The suite is offline - no network and no Mascope credentials - and CI
(`.github/workflows/test.yml`) runs it on Python 3.12 and 3.13, once against the
`pyproject.toml` ranges and once against the frozen `uv.lock`. Every change
ships with a test, and the suite must stay green.

## Branches and pull requests

- Open pull requests against `main`.
- Keep pull requests focused; separate unrelated changes into separate pull
  requests.
- Add an entry to the Unreleased section of [CHANGELOG.md](CHANGELOG.md).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): description`, for example `fix(io): ...` or `docs(readme): ...`.

## Contributor License Agreement

Before we can merge your first pull request, we ask you to accept the
[Ultra Trace Systems Individual Contributor License Agreement](https://github.com/ultra-trace-systems/cla/blob/main/ICLA.md)
(ICLA). It is a one-time step, and one acceptance covers Peaky,
[Mascope](https://github.com/ultra-trace-systems/mascope), and our other
open-source projects.

In short: you keep the copyright to your work and may use it however you like;
you license it to Ultra Trace Systems so we can distribute Peaky under
Apache-2.0 and, where we need to, under other terms; and you confirm that you
are entitled to contribute it. Peaky itself stays Apache-2.0. The agreement is
based on the [Harmony](https://www.harmonyagreements.org) individual agreement.

**How to accept.** When you open your first pull request, the CLA assistant posts
a comment asking you to sign. Reply on the pull request with exactly

    I have read the CLA Document and I hereby sign the CLA

and the check passes on its next run (comment `recheck` if it has not updated
by itself). Your GitHub username, the pull request, and the time are recorded
in the public [signature register](https://github.com/ultra-trace-systems/cla).
Every commit author on the pull request has to have accepted, and commits must
be authored with an email address linked to a GitHub account so the assistant
can tell who wrote them. Ultra Trace's own developers and the bots are exempt:
their work is the company's already, and they are in the workflow's allowlist.

**If your contribution includes work that is not yours** - code adapted from
another project, a vendored file, data, images - keep it in its own commit or
file, say in the pull request where it comes from and under which licence, add
the attribution to [NOTICE](NOTICE), and make sure the licence is compatible
with Apache-2.0. The agreement asks you to confirm you have done this.

**If you contribute as part of your job**, make sure your employer agrees; the
agreement asks you to confirm that too. If your employer needs a corporate
agreement instead, or for any other question about the agreement, write to
support@ultratrace.eu.

## Reporting bugs and proposing features

Open a [GitHub issue](https://github.com/ultra-trace-systems/peaky/issues). For
questions and open-ended discussion, the Mascope
[Discord community](https://discord.gg/R5kEKJcKe8) is the place.
