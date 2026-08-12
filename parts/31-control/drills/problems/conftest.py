"""
Makes the "reference passes every test" guarantee
checkable from the tree instead of only from a stale compiled artifact.

Reader flow (default, `python3 -m pytest -q`): this file does nothing. The
stub `drills.py` raises NotImplementedError, and `attempt()` in
test_drills.py turns that into a skip, exactly as §8 rule 1 requires — the
suite must stay green-by-skipping until the reader has actually implemented
something. Monkey-patching the reference in unconditionally here would
silently overwrite the reader's own in-progress implementations on every
run, which defeats the exercise; that is why this is opt-in.

Verification flow (`CI_DRILLS_REFERENCE=1 python3 -m pytest -q`): the
reference implementations in ../solutions/reference.py are imported before
collection, which monkey-patches them onto `drills`, and the full suite runs
against a known-correct implementation.
"""
import os
import sys

if os.environ.get("CI_DRILLS_REFERENCE"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solutions"))
    import reference  # noqa: F401  (import side effect: patches drills.*)
