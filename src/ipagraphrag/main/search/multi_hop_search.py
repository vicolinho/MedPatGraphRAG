import argparse
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ipagraphrag.search.retrieval import util

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
    parser.add_argument('--patient', '-p', type=str, default='Albers',
                        help='patient to analyse')
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
    CONCEPT_LABEL = args.concept_label

    EMBEDDING_PROVIDER = os.getenv("provider", "huggingface")
    LLM_MODEL = os.getenv("LLM_MODEL", None)
    if LLM_MODEL is None:
        print("LLM model is not specified")
        exit(1)
    PATIENT_NAME = args.patient
    """Interactive entry point – prompts the user for a search query."""

    print("=" * 60)
    print("  Neo4j Vector Search")
    print("=" * 60)

    # Accept query from CLI argument or interactive prompt
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        print(f"[INFO] Query (from CLI args): {query_text}")
    else:
        query_text = input("\nEnter your search query: ").strip()
        if not query_text:
            print("[ERROR] Query cannot be empty.")
            sys.exit(1)
    query_text = query_text.split(",")
    extractor = LLMExtractor(os.getenv('BASE_URL'),os.getenv('API_KEY'), LLM_MODEL)
    searched_label_index = {"mention": "mention_vector", "Concept": "concept_vector"}
    embedder = util.get_embedding_model(EMBEDDING_PROVIDER)
    results = retriever.retrieve_subgraphs(query_text, data_source=PATIENT_NAME,
                                           searched_label_index=searched_label_index,
                                           embedding_property=EMBEDDING_PROPERTY, top_k=TOP_K, hops=2,
                                           concept_label=CONCEPT_LABEL,
                                           threshold=SIMILARITY_THRESHOLD, embedder=embedder,
                                          extractor=extractor)
    generator = JSONContextGenerator()
    context_list = generator.generate_context(results, {'text', 'name', 'definition', 'key', 'FSN', 'term'}, {'key'})
    expander = QueryExpander(os.getenv('BASE_URL'),os.getenv('API_KEY'), LLM_MODEL)
    for context in context_list:
       expander.expand_query(query_text, context)



if __name__ == "__main__":
    main()