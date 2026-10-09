"""The assignment-pass director, split into cohesive submodules.

Public surface is unchanged: every name the pipeline used as `passes.X`
is re-exported here (config -> core -> postprocess -> directors)."""

from .config import *  # noqa: F401,F403
from .core import *  # noqa: F401,F403
from .postprocess import *  # noqa: F401,F403
from .directors import *  # noqa: F401,F403

__version__ = "0.13.1"  # + a GKA-opened pass-3 family commits as contaminant:<family>:gka; 0.13.0 pass 7's element gate (the element-evidence predicate), the twin
                        # tie-break on a skeleton-only reading, the siloxane channel restriction,
                        # the exact-offset twin window once the file is calibrated, a cleared
                        # peak only an envelope line; curated_formulas
                        # (0.12.0: fail-closed edge-relative height gate; RUNTIME_FIELDS on PassConfig;
                        # pass-1 candidates drawn from the admission gate (persistence OR brightness)
