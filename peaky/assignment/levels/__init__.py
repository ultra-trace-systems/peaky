"""The evidence scale's machinery (internal). The public entry points stay in
``peaky.assignment.evidence``.

Modules: ``scale`` (levels, buckets, claims, the output columns), ``space``
(the enumeration space, decompositions, adduct deltas), ``context`` (the run
context: windows, per-file arrays, efficiency, carbon counts, indexes),
``lines`` (the isotope-line model, per-file probes and tests (a), (b), (c),
(k)), ``competitors`` (step 1: competitors, pass A, pass B), ``split`` (step 2:
the split, the side-channel lock, the amine gate), ``lists`` (step 3: the
context lists, named vs class, mode flags, the context source), ``routes``
(routes, ladders, series exclusion, other-source partners: tags and the step-1
anchors), ``decide`` (step 0, the internal pass, the level, the per-pair
record) and ``source`` (a source and the pipeline that levels it).
"""
