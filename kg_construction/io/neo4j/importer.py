import os

from kg_construction.data.chunk import Chunk
from kg_construction.data.mention import Mention
import os
from dotenv import load_dotenv
from kg_construction.io.neo4j import query
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase

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
        self.execute_queries(self.driver, node_queries)

    def import_onology(self, ontology):
        pass

    def import_mention_entity(self, entity_links: dict[Mention, str]):
        pass

    def execute_queries(self, driver, queries):
        with driver.session() as session:
            for query in queries:
                session.run(query)
