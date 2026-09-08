"""Phase 1: entity/relation extraction via neo4j-graphrag (LLMEntityRelationExtractor).

Runs in a separate venv (neo4j-graphrag<1.8 needs numpy<2), on the ScaDS.AI
Kiara cluster (VPN required). Checkpoints (atomically) after every abstract;
safe to interrupt and resume - already-succeeded PMIDs are skipped, errored
ones retried.

What lives elsewhere, all of it tunable without touching this file: the graph
schema and the run parameters in config.py, every prompt text in prompts.py
(both same package), endpoint and generation settings in common/llm_config.py,
the relation types in common/relations.py. This file is only the mechanics:
sampling, concurrency, retries, checkpointing, and the output record format.

Usage:
    python -m medgraphrag.phase1_extraction.extract_neo4j --n 1000 --out extraction_neo4j_1000.json   # PQA-L only
    python -m medgraphrag.phase1_extraction.extract_neo4j --n 0 --data ori_pqal_pqau.json \
        --out extraction_neo4j_pqal_pqau.json                           # full corpus (--n 0 = all)
"""
import os, sys, json, time, asyncio, argparse
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from ipagraphrag.common.llm_config import (
    MODEL, EXTRACTION_MODEL_PARAMS, connection_kwargs,
)
from ipagraphrag.common.sampling import SEED, sample_pmids
from ipagraphrag.kg_construction.extraction.llm.config import (
     CREATE_LEXICAL_GRAPH,GRAPH_SCHEMA,
    WORKERS, MAX_NODES_PER_ABSTRACT, MAX_RETRIES, RETRY_BACKOFF,
    DEFAULT_DATA_FILE, OUTPUT_FILE_TEMPLATE, OUTPUT_FILE_ALL,
)
from ipagraphrag.kg_construction.extraction.llm import prompt

# Windows: SelectorEventLoop avoids "Event loop is closed" on async-client
# cleanup (httpx inside OpenAILLM).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def build_extractor():
    llm = OpenAILLM(model_name=MODEL, model_params=EXTRACTION_MODEL_PARAMS,
                    **connection_kwargs())
    return LLMEntityRelationExtractor(
        llm=llm, create_lexical_graph=CREATE_LEXICAL_GRAPH,
    )



async def extract_graph(extractor, text, attempts=MAX_RETRIES):
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            chunks = TextChunks(chunks=[TextChunk(text=text, index=0)])
            graph = await extractor.run(chunks=chunks, schema=GRAPH_SCHEMA,
                          examples=prompt.EXTRA_INSTRUCTIONS)
            return graph
        except Exception as e:
            last_err = e
            if attempt < attempts:
                wait = RETRY_BACKOFF ** attempt
                print(f"    attempt {attempt}/{attempts} failed "
                      f"({type(e).__name__}: {e}) - retry in {wait:.0f}s ...")
                await asyncio.sleep(wait)
    raise last_err


def name_of(n):
    # Falsy (None or empty string) when there's no usable name; to_record drops those nodes.
    return n.properties.get("name")


def to_record(graph, pmid, abstract, latency):
    """Neo4jGraph -> our nested JSON format. Drops nameless nodes and any edge
    with a nameless endpoint."""
    triples = []
    # id2node = {n.id: n for n in graph.nodes}
    valid = {n.id for n in graph.nodes if name_of(n)}
    nodes = [{"id": n.id, "text": n.properties['name'], "type": [n.label, "mention"], "data_source": pmid}
             for n in graph.nodes if n.id in valid]
    nodes.append({"id":pmid, "type": ["abstract", "chunk"], "text": abstract, "data_source": pmid})#
    for n_id in valid:
        triples.append({"source": pmid, "type": "has_mention", "target": n_id
                    })

    for r in graph.relationships:
        if r.start_node_id not in valid or r.end_node_id not in valid:
            continue
        #h, t = id2node[r.start_node_id], id2node[r.end_node_id]
        triples.append({"source": r.start_node_id, "type": r.type, "target": r.end_node_id,
                        "quote": r.properties.get("quote")})

    return {
        "n_nodes": len(nodes), "n_rels": len(triples),
        "latency": round(latency, 2), "error": None,
        "nodes": nodes, "triples": triples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="number of abstracts (seed 42); <=0 = all")
    ap.add_argument("--out", default=None, help="output file")
    ap.add_argument("--data", default=DEFAULT_DATA_FILE, help="input dataset")
    args = ap.parse_args()

    out_file = args.out or (OUTPUT_FILE_TEMPLATE.format(n=args.n) if args.n > 0
                            else OUTPUT_FILE_ALL)

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    pmids = sample_pmids(data, args.n)   # <=0 means the whole corpus

    if os.path.exists(out_file):
        results = json.load(open(out_file, encoding="utf-8"))
    else:
        results = {MODEL: {}}
    results.setdefault(MODEL, {})

    def save():
        # Atomic write (temp file + os.replace) so a crash mid-write leaves
        # out_file as the last complete version, never a truncated one.
        tmp = out_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out_file)

    # A pmid counts as done only if it succeeded; on resume, retry anything
    # missing OR errored (error records are saved, but must not block a retry).
    done = sum(1 for p in pmids if p in results[MODEL] and not results[MODEL][p].get("error"))
    print(f"Model: {MODEL} | Framework: neo4j-graphrag (enforce_schema=STRICT)")
    print(f"Sample: {len(pmids)} abstracts (seed {SEED}) | already done: {done} "
          f"| parallel: {WORKERS} workers")
    print(f"Output: {out_file}\n")
    async def run_all():
        extractor = build_extractor()
        todo = [p for p in pmids if p not in results[MODEL] or results[MODEL][p].get("error")]
        total = len(todo)
        sem = asyncio.Semaphore(WORKERS)
        lock = asyncio.Lock()
        counter = {"n": 0}

        async def worker(pmid):
            async with sem:
                t0 = time.time()
                text = " ".join(data[pmid]["CONTEXTS"])
                try:
                    graph = await extract_graph(extractor, text)
                    latency = time.time() - t0
                    if len(graph.nodes) > MAX_NODES_PER_ABSTRACT:
                        rec = {"error": f"degeneration: {len(graph.nodes)} nodes"}
                        msg = f"DISCARDED ({len(graph.nodes)} nodes)"
                    else:
                        rec = to_record(graph, pmid, text, latency)
                        msg = f"{rec['n_nodes']} nodes, {rec['n_rels']} rels, {rec['latency']}s"
                except Exception as e:
                    rec = {"error": str(e)[:200]}
                    msg = f"ERROR: {str(e)[:80]}"
            async with lock:
                results[MODEL][pmid] = rec
                save()
                counter["n"] += 1
                print(f"  [{counter['n']}/{total}] {pmid} - {msg}", flush=True)

        await asyncio.gather(*(worker(p) for p in todo))

    asyncio.run(run_all())

    ok = sum(1 for p in pmids if not results[MODEL][p].get("error"))
    print(f"\nDone. Successful: {ok}/{len(pmids)} | saved to {out_file}")


if __name__ == "__main__":
    main()