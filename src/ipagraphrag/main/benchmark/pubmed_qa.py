r"""Phase 4/5: answer generation + QA evaluation (GraphRAG end-to-end).

Per PubMedQA question: get context -> LLM prompt (ScaDS.AI/KIARA) -> parse
yes/no/maybe -> compare to final_decision. Four modes:
  (default)    graph, no quotes  : k-hop subgraph facts as context
  --quotes     graph + quotes    : facts + real per-triple supporting quotes
  --no-graph   baseline          : no context, LLM knowledge only
  --textrag    text-RAG          : TF-IDF-retrieved abstracts as context

Runs in .venv-scispacy. Tunables (fact budget, quote gating, sweep toggles) are
in config.py, the system prompt per mode in prompts.py.

Usage:
  python -m medgraphrag.phase4_qa.eval_qa --n 1000
  python -m medgraphrag.phase4_qa.eval_qa --n 1000 --quotes
  python -m medgraphrag.phase4_qa.eval_qa --n 1000 --no-graph
  python -m medgraphrag.phase4_qa.eval_qa --n 1000 --textrag
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

import prompts
from config import (
    WORKERS, MAX_QUOTES_PER_FACT, MIN_QUOTE_WORDS, SIM_USE_QUOTES, TEXTRAG_K, TEXTRAG_CORPUS,
    QUESTION_FILE, )
from ipagraphrag.common.llm_config import MODEL, TEMPERATURE, connection_kwargs
from ipagraphrag.common.sampling import sample_pmids
from ipagraphrag.kg_construction.extraction.llm import neo4j_llm_extractor
from ipagraphrag.search.context.json_context_generator import JSONContextGenerator
from ipagraphrag.search.retrieval.multi_hop_retriever import MultiHopNodeRetriever
from ipagraphrag.search.retrieval import util
from ipagraphrag.search.context.fact_context_generator import FactContextGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

parser = argparse.ArgumentParser(description='rl generation')

parser.add_argument('--vector_index', '-vi', type=str, default='mention_vector', help='vector index name')
parser.add_argument('--node_label', '-nl', type=str, default='mention',
                    help='nodel label for querying')
parser.add_argument('--embedding_property', '-ep', type=str, default='embedding',
                    help='embedding property')
parser.add_argument('--sim_threshold', '-t', type=float, default=0.5,
                    help='similarity threshold for node embedding and query mention embedding')
parser.add_argument('--top_k', '-top_k', type=int, default=2,
                    help='top k for query mention and node embedding similarity ranking')
parser.add_argument('--concept_label', '-cl', type=str, default='Concept',
                    help='patient to analyse')
parser.add_argument('--n', '-n', type=int, default=-1,
                    help='number of dcouments')
args = parser.parse_args()

client = OpenAI(**connection_kwargs())

if "--textrag" in sys.argv:
    MODE = "textrag"
elif "--no-graph" in sys.argv:
    MODE = "baseline"
else:
    MODE = "graph"
USE_QUOTES = "--quotes" in sys.argv

SYS = prompts.system_prompt(MODE)
LLM_MODEL = os.getenv("LLM_MODEL", None)
if LLM_MODEL is None:
    print("LLM model is not specified")
    exit(1)

N = args.n
QFILE = QUESTION_FILE
data = json.load(open(QFILE, encoding="utf-8"))
# Same sample as phase 1's --n, so in the pqal config the graph was built from
# exactly the abstracts these questions come from.
pmids = sample_pmids(data, N)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

EMBEDDING_PROPERTY = args.embedding_property

# Search parameters
TOP_K = args.top_k  # number of results
SIMILARITY_THRESHOLD = args.sim_threshold
CONCEPT_LABEL = args.concept_label

EMBEDDING_PROVIDER = os.getenv("provider", "huggingface")
LLM_MODEL = os.getenv("LLM_MODEL", None)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
retriever = MultiHopNodeRetriever(driver)
extractor = neo4j_llm_extractor.build_extractor()
embedder = util.get_embedding_model(EMBEDDING_PROVIDER)
if MODE == "textrag":
    # Default retrieval pool is the same n questions being evaluated (original
    # behavior, unchanged). TEXTRAG_CORPUS lets the pool be a superset (e.g. the
    # full pqal+pqau corpus) while --n still only controls how many QUESTIONS
    # get asked.
    if TEXTRAG_CORPUS:
        _pool = json.load(open(TEXTRAG_CORPUS, encoding="utf-8"))
        _pool_ids = sorted(_pool)
    else:
        _pool, _pool_ids = data, pmids
    _corpus = [" ".join(_pool[p]["CONTEXTS"]) for p in _pool_ids]
    _vec = TfidfVectorizer(stop_words="english")
    _M = _vec.fit_transform(_corpus)


def _sentence_line(G, u, v, rel):
    """Per-triple quotes from the graph (quotes/evidence are index-parallel to
    pmids, see build_graph.py). Only finding-tagged quotes are shown: those are
    the study's own stated results, not background or method."""
    d = G.get_edge_data(u, v, key=rel)
    if not d:
        return ""
    quotes_, evidence_ = d.get("quotes", []), d.get("evidence", [])
    items = [q for i, q in enumerate(quotes_)
             if q and len(q.split()) >= MIN_QUOTE_WORDS
             and i < len(evidence_) and evidence_[i] == "finding"]
    seen, lines_ = set(), []
    for q in items:
        if q in seen:
            continue
        seen.add(q)
        lines_.append(f'    source sentence: "{q.strip()}"')
        if len(lines_) >= MAX_QUOTES_PER_FACT:
            break
    return ("\n" + "\n".join(lines_)) if lines_ else ""


def _fact_sim_text(G, u, v, rel):
    """Text a fact is ranked by. With SIM_USE_QUOTES, append its finding quotes
    to the triple; else just the triple."""
    base = f"{G.nodes[u]['label']} {rel} {G.nodes[v]['label']}"
    if not SIM_USE_QUOTES:
        return base
    d = G.get_edge_data(u, v, key=rel) or {}
    q_, e_ = d.get("quotes", []), d.get("evidence", [])
    finding = {q for i, q in enumerate(q_) if q and i < len(e_) and e_[i] == "finding"}
    return (base + " " + " ".join(sorted(finding))) if finding else base


async def facts_for(q, pmid):
    """Selects the facts shown to the reader, in two stages with different keys.
    Selection: candidates are ranked (IS_A last, anchor tier, question-similarity)
    and cut at MAX_FACTS -- similarity discriminates relevance to THIS question,
    which corpus-wide weight cannot once an anchor has more edges than fit the
    window. Presentation: the survivors are re-sorted by weight."""

    searched_label_index = {"mention": "mention_vector", "Concept": "concept_vector"}
    results = await retriever.retrieve_subgraphs_with_neo4j_extractor(q, data_source=pmid,
                                           searched_label_index=searched_label_index,
                                           embedding_property=EMBEDDING_PROPERTY, top_k=TOP_K, hops=2,
                                           concept_label=CONCEPT_LABEL,
                                           threshold=SIMILARITY_THRESHOLD, embedder=embedder,
                                           extractor=extractor)
    basic_context = retriever.get_basic_context(pmid)

    #generator = JSONContextGenerator()
    generator = FactContextGenerator()
    context_list = generator.generate_context(results, {'text', 'name', 'definition', 'FSN', 'term'}, {'key'})
    return context_list[0], basic_context


def parse(ans):
    # Word boundaries, not substring: "no" must not match inside "not"/
    # "another"/"cannot". First real word wins.
    low = ans.lower()
    first = low.split("\n", 1)[0]
    m = re.findall(r"\b(yes|no|maybe)\b", first) or re.findall(r"\b(yes|no|maybe)\b", low)
    return m[0] if m else "?"


async def build_prompt(q, pmid):
    """Builds context + prompt. In graph mode, uses retrieve/scispacy ->
    called SEQUENTIALLY (scispacy is not thread-safe)."""
    user = f"Question: {q}\n"
    nf = 0
    if MODE == "graph":
        lines, basic_context = await facts_for(q, pmid)
        if lines:  # no facts -> no graph block at all, model answers from priors
            nf = len(lines)
            user += "\nKnowledge graph facts:\n" + "\n" + str(lines) + "\n"
            user += "\n Basic context:\n" + "\n" + str(basic_context) +"\n"
    elif MODE == "textrag":
        qv = _vec.transform([q])
        sims = linear_kernel(qv, _M).ravel()
        topk = sims.argsort()[::-1][:TEXTRAG_K]
        nf = len(topk)
        ctx = "\n\n".join(f"[Retrieved abstract {r + 1}]\n{_corpus[i]}"
                          for r, i in enumerate(topk))
        user += f"\nRetrieved abstracts:\n{ctx}\n"

    user += "\nAnswer (yes/no/maybe):"
    return user, nf


def call_llm(user):
    r = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


correct, rows, conf = 0, [], Counter()
print(f"Mode: {MODE}{' +quotes' if USE_QUOTES else ''}  (workers={WORKERS})  qfile={QFILE}\n")

print(f"Building prompts ({len(pmids)}, sequential"
      f"{' incl. retrieval/SNOMED index load' if MODE == 'graph' else ''}) ...", flush=True)
items = []  # (p, gold, nf, user)

async def run_all():
    todo = []
    for _i, p in tqdm(enumerate(pmids, 1)):
        q = data[p]["QUESTION"]
        todo.append((q, p))
    counter = {"n": 0}
    total = len(todo)
    sem = asyncio.Semaphore(WORKERS)
    lock = asyncio.Lock()
    async def worker(query, p):
        async with sem:
            user, nf = await build_prompt(query, p)
            gold = data[p]["final_decision"].strip().lower()
        async with lock:
            items.append((p, gold, nf, user))
            counter["n"] += 1
            print(f"  [{counter['n']}/{total}] {p}", flush=True)
    await asyncio.gather(*(worker(query, p) for query, p in todo))

asyncio.run(run_all())

print("number of item: {}".format(len(items)))

def _run(item):
    p, gold, nf, user = item
    try:
        resp = call_llm(user)
        return (p, gold, nf, parse(resp), resp, None)
    except Exception as e:
        return (p, gold, nf, None, None, str(e)[:80])


print(f"LLM calls (parallel, workers={2}) ...\n", flush=True)
with ThreadPoolExecutor(max_workers=1) as ex:
    for i, (p, gold, nf, pred, resp, err) in enumerate(ex.map(_run, items), 1):
        if err:
            print(f"  [{i}/{len(pmids)}] {p} -- ERROR: {err}")
            continue
        ok = pred == gold
        correct += ok
        conf[(gold, pred)] += 1
        rows.append((p, gold, pred, nf, ok))
        fstr = f"{nf:2}"
        print(f"  [{i}/{len(pmids)}] {p} facts={fstr} gold={gold:5} pred={pred:5} {'OK' if ok else 'X'}")

n = len(rows)
print(f"\n=== {MODE}{' +quotes' if USE_QUOTES else ''} ===")
print(f"Accuracy: {correct}/{n} = {100*correct/n:.0f}%")

gold_tot = Counter(r[1] for r in rows)
corr_cls = Counter(r[1] for r in rows if r[4])
classes = [c for c in ("yes", "no", "maybe") if gold_tot[c]]
recalls = {c: corr_cls[c] / gold_tot[c] for c in classes}
macro = sum(recalls.values()) / len(recalls)
maj = max(gold_tot.values()) / n
print("Per-class recall: " + " | ".join(
    f"{c} {corr_cls[c]}/{gold_tot[c]} ({100*recalls[c]:.0f}%)" for c in classes))
print(f"Macro-recall (balanced accuracy): {100*macro:.0f}%  | majority-class baseline: {100*maj:.0f}%")
yn = [c for c in ("yes", "no") if gold_tot[c]]
if len(yn) == 2:
    ynbal = sum(corr_cls[c] / gold_tot[c] for c in yn) / 2
    print(f"Yes/No balanced (excl. maybe): {100*ynbal:.0f}%")

if MODE == "graph":
    hit = [r for r in rows if r[3] > 0]
    if hit:
        hc = sum(1 for r in hit if r[4])
        print(f"Questions WITH a graph hit ({len(hit)}): {hc}/{len(hit)} = {100*hc/len(hit):.0f}%")

print("Confusion (gold -> pred):")
for (g, pr), c in sorted(conf.items()):
    print(f"  {g:6} -> {pr:6} : {c}")

LOG_FILE = os.getenv("LOG_FILE")
if LOG_FILE:
    log = {p: {"gold": gold, "pred": pred, "nf": nf, "ok": ok}
           for p, gold, pred, nf, ok in rows}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\nPer-question log saved: {LOG_FILE} ({len(log)} questions)")