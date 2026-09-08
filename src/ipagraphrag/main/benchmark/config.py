"""Every tunable of phase 4/5, for both halves: retrieval (retrieve.py) and
answer generation + scoring (eval_qa.py).

Almost all of it is env-overridable because this is the part that gets swept:
the shipping defaults below are the ones the reported results were produced
with, and run_pipeline.py strips these variables from the inherited environment
so a leftover shell value from a manual sweep cannot silently change a full
pipeline run. Nothing here invalidates an artifact -- retrieval and evaluation
only read the graph, so any change just means re-running the eval.

Linking thresholds are NOT here: retrieve.py has to link a question mention
exactly as ingest linked it, so it takes them from phase2_normalization/config.
The prompt text is in prompts.py, the endpoint in common/llm_config.py.
"""
import os

from dotenv import load_dotenv

from ipagraphrag.common import paths


# Read before the getenv calls below, so a knob set in .env takes effect no
# matter which module imports this config first. Idempotent, and it never
# overrides a variable the shell (or run_pipeline.py) already set.
load_dotenv()

# ------------------------------------------------------------------- retrieval

# The graph retrieval reads. Defaults to the IS_A-enriched one; point it at
# paths.GRAPH to measure what the taxonomy actually contributes.
GRAPH_FILE = os.getenv("GRAPH_FILE", paths.GRAPH_ISA)

# Bidirectional traversal depth around the question's anchor nodes. 3+ pulls in
# most of the graph on a dense build and drowns the relevant facts.
HOPS = int(os.getenv("RETRIEVE_HOPS", "2"))

K = 30            # UMLS candidates per question mention

# String fallback, used when no candidate CUI is a node in the graph. A
# high-degree raw-string node is a generic catch-all ("survival", "patients")
# shared across abstracts, not a real concept -- anchoring there would return a
# subgraph unrelated to the question.
FALLBACK_MAX_DEGREE = 15
FUZZY_CUTOFF = 0.8   # difflib ratio for the last-resort paraphrase match

# ------------------------------------------------------------------ evaluation

# Facts shown to the reader per question, selected by question-similarity rerank
# (facts_for). Env-overridable for sweeps.
MAX_FACTS = int(os.getenv("MAX_FACTS", "40"))
WORKERS = int(os.getenv("EVAL_WORKERS", "8"))

MAX_QUOTES_PER_FACT = 2
# Length floor for shown quotes. 1 = keep all (the shipping default): on the
# noisy main config the short finding quotes net-help, and dropping them costs
# ~6 QA. A 3-word floor removes bare noun phrases and helps on the clean
# pqal-only corpus, but that is not the config we ship -- same density-favouring
# call as the linking options in phase2. Env-overridable for the sweep.
MIN_QUOTE_WORDS = int(os.getenv("MIN_QUOTE_WORDS", "1"))
# Quotes only for the K facts most similar to the question (0 = all selected
# facts). Keeps distractor edges that made the cut on anchor-tier from
# contributing a misleading source sentence on a noisy corpus.
QUOTE_SIM_TOPK = int(os.getenv("QUOTE_SIM_TOPK", "0"))

# Selection key: default ranks anchor-tier before similarity, so on a noisy graph
# tier-0 distractor edges can crowd a relevant tier-1 fact out of the MAX_FACTS
# cut. =1 ranks similarity first (tier as tiebreak).
SELECT_SIM_FIRST = os.getenv("SELECT_SIM_FIRST") == "1"
# Rank facts by the similarity of their finding-quote text, not just the terse
# "node rel node" triple: the source sentence overlaps the question far more, so
# a relevant fact whose triple shares little vocabulary still ranks.
SIM_USE_QUOTES = os.getenv("SIM_USE_QUOTES") == "1"

# --textrag baseline: abstracts retrieved per question by TF-IDF.
TEXTRAG_K = int(os.getenv("TEXTRAG_K", "3"))
# Retrieval pool. Unset = the --n questions being evaluated (original behavior);
# run_pipeline points it at the full merged corpus so text-RAG faces the same
# noisy 62k pool the graph was built from.
TEXTRAG_CORPUS = os.getenv("TEXTRAG_CORPUS")

# Questions are always the labelled PQA-L ones, in both pipeline configs.
QUESTION_FILE = paths.CORPUS_PQAL
DEFAULT_N = 25   # --n default for a manual run; run_pipeline always passes one