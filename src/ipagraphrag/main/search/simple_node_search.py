import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

from search.retrieval import util

sys.path.append(os.getcwd())

from ipagraphrag.search.retrieval.simple_node_retriever import SimpleNodeRetriever


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
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    retriever = SimpleNodeRetriever(driver)
    # Name of the vector index created in Neo4j (PRIMARY approach)
    # Create it once with:
    #   CREATE VECTOR INDEX `node-embeddings`
    #   FOR (n:Document) ON (n.embedding)
    #   OPTIONS {indexConfig: {`vector.dimensions`: 1536,
    #                          `vector.similarity_function`: 'cosine'}}
    VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector")

    # Node label and property used for embeddings (FALLBACK approach)
    NODE_LABEL = os.getenv("NODE_LABEL", "mention")
    EMBEDDING_PROPERTY = os.getenv("EMBEDDING_PROP", "embedding")

    # Search parameters
    TOP_K = int(os.getenv("TOP_K", "5"))  # number of results
    SIMILARITY_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.5"))  # fallback min score


    EMBEDDING_PROVIDER = os.getenv("provider", "huggingface")
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
    searched_label_index = {NODE_LABEL: ""}
    embedder = util.get_embedding_model(EMBEDDING_PROVIDER)
    results = retriever.retrieve_subgraphs(query_text, searched_label_index=searched_label_index,
                             embedding_property=EMBEDDING_PROPERTY,
                                           top_k=TOP_K, threshold=SIMILARITY_THRESHOLD,
                                           embedder=embedder)
    print_results(results)



if __name__ == "__main__":
    main()