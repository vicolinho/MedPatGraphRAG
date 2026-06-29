import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ipagraphrag.kg_construction.io.neo4j.semanatic_network_importer import parse_srstre1, create_indexes, import_sty_nodes, \
    import_relationships, parse_srdef

#── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Import UMLS Semantic Network into Neo4j.")
    parser.add_argument("--srdef_path", default='./data/sn_current/2023AA/SRDEF', help="Path to the SRDEF file.")
    parser.add_argument("--srstre1_path", default='./data/sn_current/2023AA/SRSTRE1',  help="Path to the SRSTRE1 file.")
    args = parser.parse_args()
    srdef_path = args.srdef_path
    srstre1_path = args.srstre1_path
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


    # ── 1. Validate file paths ──────────────────────────────────────────────
    for fp in (srdef_path, srstre1_path):
        if not Path(fp).exists():
            raise FileNotFoundError(
                f"Required file not found: {os.path.abspath(fp)}\n"
                "Please update srdef_path / srstre1_path in the CONFIG block."
            )

    # ── 2. Parse source files ───────────────────────────────────────────────
    print(f"Parsing {srdef_path} ...")
    sty_nodes, rl_nodes, rl_by_name = parse_srdef(srdef_path)
    print(f"  Found {len(sty_nodes)} SemanticType rows and {len(rl_nodes)} Relation rows.")

    print(f"Parsing {srstre1_path} ...")
    relationships = parse_srstre1(srstre1_path)
    print(f"  Found {len(relationships)} directed edges.")

    # ── 3. Write to Neo4j ───────────────────────────────────────────────────
    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        print("Creating indexes ...")
        create_indexes(driver)

        print("Importing SemanticType nodes ...")
        import_sty_nodes(driver, sty_nodes)

        print("Importing relationships ...")
        import_relationships(driver, relationships, rl_by_name)

    finally:
        driver.close()

    # ── 4. Summary ──────────────────────────────────────────────────────────
    print("\n✓ Import complete.")
    print(f"  SemanticType nodes : {len(sty_nodes)}")
    print(f"  Relation nodes     : {len(rl_nodes)}")
    print(f"  RELATED_TO edges   : {len(relationships)}")
