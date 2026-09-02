"""Every artifact file name the pipeline reads or writes, in one place.

These are the fixed hand-off points between phases: each name is written by one
step and read by the next, so the two sides have to agree. They used to be
repeated as string literals in the producing module, in the consuming module's
os.getenv() default, and again in run_pipeline.py's env_extra -- up to six
copies of "graph.pickle" alone.

Names, not paths: every module resolves them relative to the CWD (the repo
root, see medgraphrag/__init__.py), and each stays overridable per run through
the environment variable the consuming module already documents.

Imports nothing on purpose. run_pipeline.py runs under the system interpreter,
which has none of the pipeline's dependencies installed, so this module must
stay importable there -- unlike common/llm_config.py, which needs dotenv.
"""

# ------------------------------------------------------------------- corpora
# Provided by the user, not built by any script (except the merge).
CORPUS_PQAL = "data/med_qa/ori_pqal.json"          # 1000 labelled PQA-L abstracts + questions
CORPUS_PQAU = "data/med_qa/ori_pqau.json"          # 61,249 unlabelled PQA-U distractors
CORPUS_MERGED = "data/med_qa/ori_pqal_pqau.json"   # prep.merge_pqal_pqau output

# -------------------------------------------------------------- UMLS resources
# One-time builds (prep/), reused by every graph rebuild.
UMLS_KB = "umls_custom_kb.jsonl"           # prep.build_umls_kb_jsonl
UMLS_LINKER_DIR = "umls_linker_custom"     # prep.build_umls_ann_index (4 files)
MRREL_PARCHD = "mrrel_parchd_sabfiltered.tsv"   # prep.extract_mrrel_parchd

# ------------------------------------------------------------ pipeline output
# Rebuilt per run. Extraction file names are config-dependent and live in
# phase1_extraction/config.py; everything downstream uses these fixed names,
# which is why run_pipeline.py keeps a stamp file to detect a config change.
NORMALIZATION = "phase2_linked.json"   # phase2_normalization.normalize_entities
GRAPH = "graph.pickle"                 # phase3_graph.build_graph
GRAPH_ISA = "graph_isa.pickle"         # phase3_isa.build_isa -- what retrieval reads

# The three IS_A stages, each narrowing the one before.
ISA_RELATIONS = "umls_isa_relations.tsv"        # stage 1: rows touching our CUIs
ISA_PAIRS = "umls_isa_pairs.tsv"                # stage 2: deduped child->parent
ISA_PAIRS_DIRECT = "umls_isa_pairs_direct.tsv"  # stage 3: our CUIs as the child