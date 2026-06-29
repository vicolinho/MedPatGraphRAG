import argparse
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
sys.path.append(os.getcwd())
from ipagraphrag.kg_construction.linking.semantic_type_linker import NEO4J_AVAILABLE, find_unlinked_mentions, get_semantic_type, \
    link_mentions, save_links


def main():
    parser = argparse.ArgumentParser(description="Link a medical mention to UMLS semantic types using scispaCy (and "
                                                 "optionally Neo4j).")
    load_dotenv()

    args = parser.parse_args()

    # Load scispaCy model and linker
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    # Optional: Neo4j enrichment
    neo4j_driver = None
    if neo4j_uri and neo4j_user and neo4j_password and NEO4J_AVAILABLE:
        print("Connecting to Neo4j for semantic type enrichment...")
        neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    # Process the chunk
    unlinked_mentions = find_unlinked_mentions(neo4j_driver)
    print("unlinked: {}".format(len(unlinked_mentions)))
    semantic_types = get_semantic_type(neo4j_driver)
    print(len(semantic_types))
    links = link_mentions(unlinked_mentions, semantic_types)
    save_links(neo4j_driver, links)


if __name__ == "__main__":
    main()
