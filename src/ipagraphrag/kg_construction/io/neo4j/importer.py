from lib2to3.pgen2 import driver

from ipagraphrag.kg_construction.data.chunk import Chunk
from ipagraphrag.kg_construction.data.mention import Mention
from dotenv import load_dotenv
from ipagraphrag.kg_construction.io.neo4j import query

# Load environment variables from .env file
load_dotenv()


class Neo4jImport:

    def __init__(self, neo4j_driver):
       self.driver = neo4j_driver


    def import_chunks(self, chunks: list[Chunk]):
        node_queries = []
        edges_queries = []
        for chunk in chunks:
            node_queries.extend(query.create_chunk_node_queries(chunk))
        for i in range(1, len(chunks)):
            previous_chunk = chunks[i - 1]
            chunk = chunks[i]
            edges_queries.extend(query.create_chunk_edge_queries(previous_chunk, chunk))
        node_queries.extend(edges_queries)
        with self.driver.session() as session:
            self.execute_queries(session, node_queries)

    def import_onology(self, ontology):
        pass

    def import_mention_entity(self, entity_links: dict[Mention, str]):
        pass

    def execute_queries(self, session, queries):
        for query in queries:
            session.run(query)
