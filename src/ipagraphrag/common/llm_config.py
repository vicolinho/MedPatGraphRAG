"""LLM endpoint and generation settings -- one place for every phase.

The pipeline talks to a single vLLM deployment on the ScaDS.AI Kiara cluster
through an OpenAI-compatible API (VPN required). Credentials come from .env
(KIARA_API_KEY / KIARA_BASE_URL) and never live in the repo; everything that
is a validated *setting* rather than a secret is pinned here.

Used by:
  phase1_extraction.extract_neo4j          endpoint + EXTRACTION_MODEL_PARAMS
  phase2_normalization.normalize_entities  MODEL only -- see below
  phase3_graph.build_graph                 MODEL only -- see below
  phase4_qa.eval_qa                        endpoint + TEMPERATURE

Phases 2 and 3 make no API call: they read the extraction JSON, which phase 1
writes under the model id as its top-level key, so all three have to agree on
that one string or the graph silently comes out empty. Everything else in the
pipeline (prep, phase3_isa, retrieve) is fully local and imports nothing here.

Imports no provider SDK on purpose: phase 1 runs in .venv-neo4j
(neo4j-graphrag) and everything else in .venv-scispacy (openai), so this module
has to import cleanly in both. Each caller builds its own client from
connection_kwargs().
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Deliberately not env-overridable: MODEL is also the key the extraction JSON is
# written under and that build_graph.py reads back, so a per-run override would
# desync the two and produce an empty graph rather than an error.
MODEL = 'vllm-nvidia-llama-3-3-70b-instruct-fp8'#os.getenv("LLM_MODEL")

# Greedy decoding everywhere -- extraction and evaluation both need runs to be
# reproducible, not creative.
TEMPERATURE = 0

# Phase 1 only. max_tokens is a safety ceiling, not a budget: real outputs top
# out around 1500 tokens and never hit it. The repetition penalties MUST stay at
# 0 - anything above breaks this model's JSON output.
EXTRACTION_MODEL_PARAMS = {
    "temperature": TEMPERATURE,
    "max_tokens": 4096,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


def connection_kwargs():
    """{"api_key": ..., "base_url": ...} for an OpenAI-compatible client.

    Fails immediately with a readable message instead of letting an unset
    variable surface later as a 401 from inside the client, or - worse for
    KIARA_BASE_URL - as a call to the real OpenAI endpoint.
    """
    missing = [v for v in ("API_KEY", "BASE_URL") if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing in .env: {', '.join(missing)}. Both are required - see the "
            f"README prerequisites."
        )
    return {"api_key": os.getenv("API_KEY"),
            "base_url": os.getenv("BASE_URL")}