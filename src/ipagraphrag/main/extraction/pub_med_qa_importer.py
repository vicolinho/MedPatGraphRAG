import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ipagraphrag.kg_construction.io.neo4j import json_importer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import JSON graph data into a Neo4j database."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to the input JSON file (required)",
    )
    return parser.parse_args()

def main() -> None:
    load_dotenv()
    args = parse_args()

    # 1. Load and parse the JSON file
    data = json_importer.load_json(args.file)

    # 2. Extract nodes and triples from the nested structure
    nodes, triples = json_importer.extract_nodes_and_triples(data)

    if not nodes and not triples:
        logger.warning("No nodes or triples found in the JSON. Exiting.")
        return

    # 3. Connect to Neo4j and run imports
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"),
                                  auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", os.getenv("NEO4J_URI"))

        with driver.session() as session:
            # Optional: add constraint to speed up MERGE on `id`
            json_importer.ensure_constraints(session)

            # Import nodes
           #nodes_created = json_importer.create_nodes(session, nodes)

            # Import relationships
            json_importer.create_relationships(session, triples, batch_size=500)

        # 4. Summary report
        print("\n" + "=" * 50)
        print(f"  Import complete")
        # print(f"  Nodes created        : {nodes_created}")
       # print(f"  Relationships created: {rels_created}")
        print("=" * 50)

    finally:
        driver.close()
        logger.info("Neo4j driver closed.")


if __name__ == "__main__":
    main()