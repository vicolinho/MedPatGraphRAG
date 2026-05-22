import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from kg_construction.extraction.llm import prompt
from kg_construction.extraction.llm.llm_extractor import LLMExtractor
from kg_construction.io.neo4j.importer import Neo4jImport



if __name__ == '__main__':
    load_dotenv()
    extractor = LLMExtractor(os.getenv("LLM_STUB_URL"), os.getenv("API_KEY"), model=os.getenv("LLM_MODEL"))
    mentions, chunk_list = extractor.extract("data/graSSCo/Albers.txt", prompt=prompt.EXTRACT_MENTIONS)
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    neo4j_driver = GraphDatabase.driver(neo4j_uri,
                                        auth=(neo4j_user, neo4j_password))
    neo4j_importer = Neo4jImport(neo4j_driver)
    neo4j_importer.import_chunks(chunk_list)
    for c in chunk_list:
        print(c)
        print("=" * 80)
