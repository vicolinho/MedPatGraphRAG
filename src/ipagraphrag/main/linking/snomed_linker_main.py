import argparse
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
import pickle

sys.path.append(os.getcwd())
from ipagraphrag.kg_construction.linking import  snomed_ger_linker
from ipagraphrag.kg_construction.linking import util



def main():
    parser = argparse.ArgumentParser(description="Link a medical mention to SNOMED concepts.")
    parser.add_argument("--ontology", "-ont", default='ObjectConcept', help="ontology name for linking")
    load_dotenv()

    args = parser.parse_args()

    # Load scispaCy model and linker
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    # Optional: Neo4j enrichment
    neo4j_driver = None
    if neo4j_uri and neo4j_user and neo4j_password:
        print("Connecting to Neo4j for SNOMED enrichment...")
        neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    # Process the chunk
    # TODO better selection of unlinked mentions harmonize schema
    unlinked_mentions = util.find_unlinked_mentions_pub_med(neo4j_driver, args.ontology)
    if os.path.exists("links.tmp"):
        with open("links.tmp", "rb") as f:
            links = pickle.load(f)
        for m_id, _ in links:
            del unlinked_mentions[m_id]
    session = neo4j_driver.session()
    print("unlinked: {}".format(len(unlinked_mentions)))
    if len(unlinked_mentions)> 0:
        # semantic_types = get_semantic_type(neo4j_driver, args.ontology)
        links = snomed_ger_linker.link_mentions(session, unlinked_mentions)
        snomed_ger_linker.save_links(neo4j_driver, links)
    else:
        print(f"""No unlinked mentions available for ontology {args.ontology}""")



if __name__ == "__main__":
    main()
