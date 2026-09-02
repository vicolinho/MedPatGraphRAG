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
import os, sys, json, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ipagraphrag.common.llm_config import MODEL, TEMPERATURE, connection_kwargs
from ipagraphrag.common.sampling import sample_pmids
import prompts
from config import (
    MAX_FACTS, WORKERS, MAX_QUOTES_PER_FACT, MIN_QUOTE_WORDS, QUOTE_SIM_TOPK,
    SELECT_SIM_FIRST, SIM_USE_QUOTES, TEXTRAG_K, TEXTRAG_CORPUS,
    QUESTION_FILE, DEFAULT_N,
)
from medgraphrag.phase4_qa import retrieve   # loads the graph + scispacy linker on import

client = OpenAI(**connection_kwargs())

if "--textrag" in sys.argv:
    MODE = "textrag"
elif "--no-graph" in sys.argv:
    MODE = "baseline"
else:
    MODE = "graph"
USE_QUOTES = "--quotes" in sys.argv

SYS = prompts.system_prompt(MODE)


def _argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


N = int(_argval("--n", str(DEFAULT_N)))
QFILE = QUESTION_FILE
data = json.load(open(QFILE, encoding="utf-8"))
# Same sample as phase 1's --n, so in the pqal config the graph was built from
# exactly the abstracts these questions come from.
pmids = sample_pmids(data, N)

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


def facts_for(q):
    """Selects the facts shown to the reader, in two stages with different keys.
    Selection: candidates are ranked (IS_A last, anchor tier, question-similarity)
    and cut at MAX_FACTS -- similarity discriminates relevance to THIS question,
    which corpus-wide weight cannot once an anchor has more edges than fit the
    window. Presentation: the survivors are re-sorted by weight."""
    G = retrieve.G
    _, anchors, _, _, triples = retrieve.retrieve(q)
    anchor_keys = {key for _, key in anchors}

    def _anchor_tier(u, v):
        return 2 - ((u in anchor_keys) + (v in anchor_keys))

    if triples:
        fact_texts = [_fact_sim_text(G, u, v, rel) for u, v, rel, w, src, ev in triples]
        # Fit locally per question (question + its own candidates only): pools
        # are small (dozens-hundreds), a corpus-wide vectorizer isn't needed.
        vec = TfidfVectorizer(stop_words="english")
        M = vec.fit_transform([q] + fact_texts)
        sims = linear_kernel(M[0:1], M[1:]).ravel()
    else:
        sims = []

    def _sel_key(pair):
        isa, tier, sim = pair[0][2] == "IS_A", _anchor_tier(pair[0][0], pair[0][1]), -pair[1]
        return (isa, sim, tier) if SELECT_SIM_FIRST else (isa, tier, sim)

    ranked = sorted(zip(triples, sims), key=_sel_key)
    top = ranked[:MAX_FACTS]   # (triple, similarity) pairs

    # Quote gate: with QUOTE_SIM_TOPK>0, only the K facts most similar to the
    # question keep their source sentence; the rest still show as bare triples.
    if USE_QUOTES and QUOTE_SIM_TOPK and len(top) > QUOTE_SIM_TOPK:
        sim_cut = sorted((s for _, s in top), reverse=True)[QUOTE_SIM_TOPK - 1]
    else:
        sim_cut = float("-inf")

    # Re-sort the survivors for presentation: IS_A last, then weight (= number
    # of supporting abstracts) first. Similarity decides only which facts make
    # the MAX_FACTS cut; the reader sees them ordered by corroboration.
    top = sorted(top, key=lambda pair: (pair[0][2] == "IS_A", -pair[0][3]))

    lines = [
        f"{G.nodes[u]['label']} --{rel}--> {G.nodes[v]['label']} (support: {w})"
        + (_sentence_line(G, u, v, rel) if USE_QUOTES and sim >= sim_cut else "")
        for (u, v, rel, w, src, ev), sim in top
    ]
    return lines


def parse(ans):
    # Word boundaries, not substring: "no" must not match inside "not"/
    # "another"/"cannot". First real word wins.
    low = ans.lower()
    first = low.split("\n", 1)[0]
    m = re.findall(r"\b(yes|no|maybe)\b", first) or re.findall(r"\b(yes|no|maybe)\b", low)
    return m[0] if m else "?"


def build_prompt(q, p):
    """Builds context + prompt. In graph mode, uses retrieve/scispacy ->
    called SEQUENTIALLY (scispacy is not thread-safe)."""
    user = f"Question: {q}\n"
    nf = 0

    if MODE == "graph":
        lines = facts_for(q)
        nf = len(lines)
        if lines:  # no facts -> no graph block at all, model answers from priors
            user += "\nKnowledge graph facts:\n" + "\n".join(lines) + "\n"

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
      f"{' incl. retrieval/UMLS index load' if MODE == 'graph' else ''}) ...", flush=True)
items = []  # (p, gold, nf, user)
for _i, p in enumerate(pmids, 1):
    q = data[p]["QUESTION"]
    gold = data[p]["final_decision"].strip().lower()
    user, nf = build_prompt(q, p)
    items.append((p, gold, nf, user))
    if _i % 25 == 0 or _i == len(pmids):
        print(f"  prompt {_i}/{len(pmids)} built", flush=True)


def _run(item):
    p, gold, nf, user = item
    try:
        resp = call_llm(user)
        return (p, gold, nf, parse(resp), resp, None)
    except Exception as e:
        return (p, gold, nf, None, None, str(e)[:80])


print(f"LLM calls (parallel, workers={WORKERS}) ...\n", flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
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