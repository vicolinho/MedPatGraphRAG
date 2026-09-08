"""Every tunable of phase 1 in one place: what the extractor may put in the
graph (the schema) and how the run behaves (concurrency, retries, file names).

Nothing here opens a connection or runs anything -- extract_neo4j.py imports
this and does the work. The rest of phase 1's configuration lives next door:
prompt text in prompts.py, endpoint and generation settings in
common/llm_config.py, and the corpus sample (shared with phase 4, which has to
draw the same PMIDs) in common/sampling.py.

SCHEMA CHANGES INVALIDATE AN EXISTING EXTRACTION. Adding, renaming or removing
a node/relation type changes what the model is asked for and what STRICT
enforcement keeps, so an extraction JSON produced before the change is no
longer comparable to one produced after -- rebuild it rather than resuming
into it (extract_neo4j.py resumes per PMID and cannot detect this).
RUN section changes are safe to make mid-run: they only affect throughput and
retry behaviour, not extraction results.
"""
import os

from dotenv import load_dotenv
from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, RelationshipType, SchemaBuilder
from neo4j_graphrag.experimental.pipeline import Pipeline

from ipagraphrag.common import paths
from ipagraphrag.common.relations import EXTRACTION_RELATIONS

from ipagraphrag.kg_construction.extraction.llm import prompt

# Read before the getenv call below, so EXTRACT_WORKERS can be set in .env
# alongside the credentials. Idempotent, never overrides an already-set variable.
load_dotenv()

# ---------------------------------------------------------------- graph schema

# Entity types the extractor may emit. Relation types come from
# common/relations.py (build_graph.py needs the same list).
ALLOWED_NODES = [
    "Disease", "Drug", "Procedure", "Intervention", "AnatomicalStructure",
    "Symptom", "ClinicalOutcome", "BiologicalProcess", "CellularComponent",
    "MedicalSpecialty", "GeneticVariant", "Pathogen",
]

# The type definitions the model is given live in prompts.py, so check the two
# still line up rather than let a half-renamed type reach the model undefined
# (or be extracted and then silently dropped by STRICT enforcement).
_undefined = [t for t in ALLOWED_NODES + EXTRACTION_RELATIONS
              if f"\n- {t}:" not in prompt.EXTRA_INSTRUCTIONS]
if _undefined:
    raise RuntimeError(
        f"No definition in prompts.EXTRA_INSTRUCTIONS for: {', '.join(_undefined)}. "
        f"Add a '- <TYPE>: ...' line there, or drop the type from the schema."
    )

# Property descriptions are prompt text -- see prompts.py for both, and for why
# STRICT mode makes the "name" property mandatory.
NAME_PROPERTY = [{
    "name": "name", "type": "STRING",
    "description": prompt.NAME_PROPERTY_DESCRIPTION,
}]

QUOTE_PROPERTY = [{
    "name": "quote", "type": "STRING",
    "description": prompt.QUOTE_PROPERTY_DESCRIPTION,
}]


def _schema_entry(label, properties=None, is_node= True):
    if is_node:
        return NodeType(label=label, description = "", properties = properties or [])
    else:
        return RelationshipType(label=label, description = "", properties = properties or [])


node_types = [_schema_entry(t, NAME_PROPERTY, True) for t in ALLOWED_NODES]
relationship_types = [ _schema_entry(t, QUOTE_PROPERTY, False) for t in EXTRACTION_RELATIONS]
GRAPH_SCHEMA = SchemaBuilder.create_schema_model(node_types, relationship_types)

# STRICT: neo4j-graphrag's own hard schema enforcement. A post-hoc filter, not a
# constraint on generation - the model still emits whatever it wants and
# off-schema types are dropped afterwards.

# The lexical graph (Document/Chunk nodes and their edges) is neo4j-graphrag
# bookkeeping for writing into Neo4j. This pipeline builds its own graph from
# the returned triples and never uses it.
CREATE_LEXICAL_GRAPH = False

# ------------------------------------------------------------------------- run

# Concurrent in-flight abstracts. The ceiling is the cluster's, not ours.
WORKERS = int(os.getenv("EXTRACT_WORKERS", "8"))

# Output beyond this many nodes for a single abstract is model degeneration
# (repetition loops), not a rich abstract - the record is discarded as an error
# and retried on the next run.
MAX_NODES_PER_ABSTRACT = 50 #TODO: sind das nicht viele Nodes per abstract? ich hatte bei meinem durchlauf max 20 Nodes


# Per-abstract retries on API/parse failure, exponentially backed off: 2s, 4s.
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# ----------------------------------------------------------------------- files

# Overridable per run with --data / --out; these are only the fallbacks. The
# extraction file name is config-dependent (it encodes --n), so unlike the
# fixed downstream artifacts in common/paths.py it is built here.
DEFAULT_DATA_FILE = paths.CORPUS_PQAL
OUTPUT_FILE_TEMPLATE = "extraction_neo4j_{n}.json"   # --n > 0
OUTPUT_FILE_ALL = "extraction_neo4j_all.json"        # --n 0 (whole corpus)