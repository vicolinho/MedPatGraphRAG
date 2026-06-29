import argparse
import json
import os
import sys
from os.path import isfile, join

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.append(os.getcwd())
from ipagraphrag.kg_construction.extraction.llm.llm_extractor import LLMExtractor
from ipagraphrag.search.query_expansion.query_expander import QueryExpander
from ipagraphrag.search.context.json_context_generator import JSONContextGenerator
from ipagraphrag.search.retrieval.multi_hop_retriever import MultiHopNodeRetriever


def print_results(results: list[dict]) -> None:
    """Pretty-print the list of matched nodes and their similarity scores."""
    if not results:
        print("\n[RESULT] No nodes found above the similarity threshold.")
        return

    print(f"\n[RESULT] Found {len(results)} similar node(s):\n")
    separator = "-" * 60
    for rank, item in enumerate(results, start=1):
        print(f"  Rank {rank}  |  Score: {item['score']:.4f}")
        print(separator)
        for prop, value in item["node"].items():
            # Truncate long values (e.g. raw embedding arrays) for readability
            display_value = (
                f"[vector, {len(value)} dims]"
                if isinstance(value, (list, tuple)) and len(value) > 10
                else value
            )
            print(f"    {prop}: {display_value}")
        print()

def main() -> None:
    load_dotenv()
    # ---------------------------------------------------------------------------
    # CONFIG – update these values or override them via environment variables
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='rl generation')

    parser.add_argument('--vector_index', '-vi', type=str, default='vector', help='vector index name')
    parser.add_argument('--node_label', '-nl', type=str, default='mention',
                        help='nodel label for querying')
    parser.add_argument('--embedding_property', '-ep', type=str, default='embedding',
                        help='embedding property')
    parser.add_argument('--sim_threshold', '-t', type=float, default=0.5,
                        help='similarity threshold for node embedding and query mention embedding')
    parser.add_argument('--top_k', '-top_k', type=int, default=5,
                        help='top k for query mention and node embedding similarity ranking')
    args = parser.parse_args()
    # ---------------------------------------------------------------------------
    # CONFIG – update these values or override them via environment variables
    # ---------------------------------------------------------------------------
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    retriever = MultiHopNodeRetriever(driver)
    # Name of the vector index created in Neo4j (PRIMARY approach)
    # Create it once with:
    #   CREATE VECTOR INDEX `node-embeddings`
    #   FOR (n:Document) ON (n.embedding)
    #   OPTIONS {indexConfig: {`vector.dimensions`: 1536,
    #                          `vector.similarity_function`: 'cosine'}}
    VECTOR_INDEX_NAME = args.vector_index

    # Node label and property used for embeddings (FALLBACK approach)
    NODE_LABEL = args.node_label

    EMBEDDING_PROPERTY = args.embedding_property

    # Search parameters
    TOP_K = args.top_k  # number of results
    SIMILARITY_THRESHOLD = args.sim_threshold


    EMBEDDING_PROVIDER = os.getenv("provider", "huggingface")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-vl-235b-a22b-instruct-fp8")
    """Interactive entry point – prompts the user for a search query."""
    """Interactive entry point – prompts the user for a search query."""
    EVALUATION_QUERY_PATH = os.getenv("eval_query_path", "data/graSSCo/evaluation")
    query_files = [f for f in os.listdir(EVALUATION_QUERY_PATH) if isfile(join(EVALUATION_QUERY_PATH, f))]
    # Accept query from CLI argument or interactive prompt
    for file_name in query_files:
        with open(join(EVALUATION_QUERY_PATH, file_name)) as f:
            patient = file_name.split("_")[0]
            result_list = []
            for query_text in f:
                query_text = query_text.split(",")
                extractor = LLMExtractor(os.getenv('LLM_STUB_URL'),os.getenv('API_KEY'), LLM_MODEL)
                results = retriever.retrieve_subgraphs(query_text, index_name=VECTOR_INDEX_NAME,
                                        node_label=NODE_LABEL, patient_name=patient, embedding_property=EMBEDDING_PROPERTY, top_k=TOP_K, hops=1,
                                                       threshold=SIMILARITY_THRESHOLD, provider=EMBEDDING_PROVIDER,
                                                      extractor=extractor)
                generator = JSONContextGenerator()
                context_list = generator.generate_context(results, {'text', 'name', 'definition', 'key'}, {'key'})
                expander = QueryExpander(os.getenv('LLM_STUB_URL'),os.getenv('API_KEY'), LLM_MODEL)

                for context in context_list:
                   json_result = expander.expand_query(query_text, context)
                   result_list.append(json_result)
            json.dump(result_list, open(join(EVALUATION_QUERY_PATH, "results/" + file_name), "w"), indent=4,
                      ensure_ascii=False)



if __name__ == "__main__":
    main()