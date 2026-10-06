"""chemistry.candidates_for_peaks on the memoised grid: the mass column is
cached with the grid (built once per element box, not once per call) and
candidates_for_peaks reads it from the cache, and the candidate sets are
exactly what a linear scan of the enumerated grid returns -- cold cache, warm
cache and after an eviction. Synthetic, offline."""
import random

import pytest

from peaky.chem import chemistry as C

BOX = "C0-14 H0-30 O0-10 N0-2"
ADDUCTS = ["[M-H]-", "[M+NO3]-", "[M+^NO3]-", "[M+Br]-", "[M+H]+", "[M+NH4]+"]


def _reference(peak_mzs, ranges, adducts, ppm, mass_min=30.0, mass_max=900.0):
    """Every grid formula whose neutral mass lies within the ion's ppm window of
    a peak under an adduct: a linear scan, no bisect, no cache."""
    grid = C.enumerate_grid(ranges, mass_min, mass_max)
    out = set()
    for mz in peak_mzs:
        tol = mz * ppm * 1e-6
        for a in adducts:
            if a not in C.ADDUCT_SHIFTS:
                continue
            m_neu = mz - C.ADDUCT_SHIFTS[a]
            if m_neu < mass_min or m_neu > mass_max:
                continue
            out.update(f for m, f in grid if m_neu - tol <= m <= m_neu + tol)
    return out


def _peaks(ranges, n=60, seed=7):
    """Ions of grid formulas (on centre and a few ppm off) plus random m/z."""
    rng = random.Random(seed)
    grid = C.enumerate_grid(ranges, 30.0, 900.0)
    out = []
    for _ in range(n):
        m, _f = grid[rng.randrange(len(grid))]
        a = rng.choice(ADDUCTS)
        out.append((m + C.ADDUCT_SHIFTS[a]) * (1 + rng.uniform(-4, 4) * 1e-6))
    out += [rng.uniform(40.0, 600.0) for _ in range(n)]
    return out


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(C, "_GRID_CACHE", {})
    return C._GRID_CACHE


def test_candidate_sets_equal_a_linear_scan_cold_warm_and_evicted(fresh_cache):
    box = C.parse_ranges(BOX)
    peaks = _peaks(box)
    for ppm in (1.0, 3.0, 10.0):
        want = [_reference([mz], box, ADDUCTS, ppm) for mz in peaks]
        cold = [C.candidates_for_peaks([mz], box, ADDUCTS, ppm_tolerance=ppm) for mz in peaks]
        warm = [C.candidates_for_peaks([mz], box, ADDUCTS, ppm_tolerance=ppm) for mz in peaks]
        assert cold == want and warm == want
        assert any(want)                                          # the probe set does find formulas
        if ppm >= 5.0:                                            # each ion of a grid formula (<= 4 ppm off) is found
            assert all(want[:60])
        fresh_cache.clear()                                       # an eviction rebuilds the same entry
        assert [C.candidates_for_peaks([mz], box, ADDUCTS, ppm_tolerance=ppm) for mz in peaks] == want
    # many peaks in one call: the union of the single-peak sets
    assert C.candidates_for_peaks(peaks, box, ADDUCTS, ppm_tolerance=3.0) == \
        _reference(peaks, box, ADDUCTS, 3.0)


def test_candidate_set_respects_the_mass_bounds_of_the_call(fresh_cache):
    box = C.parse_ranges(BOX)
    peaks = _peaks(box, seed=11)
    for lo, hi in ((30.0, 900.0), (100.0, 250.0)):
        got = [C.candidates_for_peaks([mz], box, ADDUCTS, ppm_tolerance=5.0, mass_min=lo, mass_max=hi)
               for mz in peaks]
        assert got == [_reference([mz], box, ADDUCTS, 5.0, lo, hi) for mz in peaks]
    assert len(fresh_cache) == 2                                  # one entry per (box, mass bounds)


def test_mass_column_is_built_once_with_the_grid(fresh_cache, monkeypatch):
    box = C.parse_ranges(BOX)
    built = []
    real = C.enumerate_grid
    monkeypatch.setattr(C, "enumerate_grid", lambda *a, **k: built.append(1) or real(*a, **k))
    grid, masses = C._grid_and_masses(box, 30.0, 900.0)
    assert masses == [m for m, _ in grid] and masses == sorted(masses)
    for mz in (150.0, 201.08, 333.3):
        C.candidates_for_peaks([mz], box, ADDUCTS, ppm_tolerance=3.0)
    g2, m2 = C._grid_and_masses(box, 30.0, 900.0)
    assert g2 is grid and m2 is masses and len(built) == 1        # reused, never rebuilt per call
    assert C._grid_cached(box, 30.0, 900.0) is grid


def test_candidates_for_peaks_reads_the_cached_mass_column_never_walks_the_grid(fresh_cache):
    """The call site itself: on a warm cache candidates_for_peaks bisects the
    cached mass column and indexes the grid; walking the whole grid (rebuilding
    the mass column per call) fails."""
    box = C.parse_ranges(BOX)
    peaks = _peaks(box, n=20, seed=3)
    want = C.candidates_for_peaks(peaks, box, ADDUCTS, ppm_tolerance=3.0)
    assert want                                                   # the probe indexes real grid entries

    class NoWalk(list):                                           # indexing allowed, iteration is not
        def __iter__(self):
            raise AssertionError("candidates_for_peaks walked the whole grid")

    (key, (grid, masses)), = fresh_cache.items()
    fresh_cache[key] = (NoWalk(grid), masses)
    assert C.candidates_for_peaks(peaks, box, ADDUCTS, ppm_tolerance=3.0) == want
