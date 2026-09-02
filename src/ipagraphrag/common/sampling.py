"""Which PMIDs a --n of N selects, shared by extraction and evaluation.

Phase 1 extracts a sample of the corpus and phase 4 evaluates a sample of the
questions. `run_pipeline.py --corpus pqal --n 50` relies on both landing on the
SAME 50 PMIDs -- otherwise the graph is built from abstracts that answer none
of the questions being asked, and the run looks merely bad instead of broken.
The three call sites used to repeat sorted -> shuffle -> slice with a literal
seed 42 each; sharing the function makes the invariant hold by construction.

Sorted before shuffling because dict order depends on insertion order, and a
fixed seed only reproduces the sample if the input order is fixed too.
"""
import random

# Fixed so a given --n always means the same abstracts, and so a larger --n is
# a superset of a smaller one (which is what makes a resumed extraction valid).
SEED = 42


def sample_pmids(data, n):
    """The first n PMIDs of `data` in the shared shuffled order; n <= 0 = all."""
    pmids = sorted(data)
    random.Random(SEED).shuffle(pmids)
    return pmids[:n] if n > 0 else pmids