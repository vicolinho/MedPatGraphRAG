import argparse
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
sys.path.append(os.getcwd())
from ipagraphrag.kg_construction.extraction.llm import prompt
from ipagraphrag.kg_construction.extraction.llm.llm_extractor import LLMExtractor
from ipagraphrag.kg_construction.io.neo4j.importer import Neo4jImport



if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extract entities for a specified patient document or document from a directory")
    parser.add_argument("--input", "-i", default='data/graSSCo/Albers.txt', help="Path to patient file or directory")
    args = parser.parse_args()
    extractor = LLMExtractor(os.getenv("LLM_STUB_URL"), os.getenv("API_KEY"), model=os.getenv("LLM_MODEL"))
    mentions, chunk_list = extractor.extract(args.input, prompt=prompt.EXTRACT_MENTIONS)
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    neo4j_driver = GraphDatabase.driver(neo4j_uri,
                                        auth=(neo4j_user, neo4j_password))
    neo4j_importer = Neo4jImport(neo4j_driver)
    neo4j_importer.import_chunks(chunk_list)