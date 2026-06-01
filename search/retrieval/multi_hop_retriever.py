


import os
import sys

import networkx
import networkx as nx
from kg_construction.extraction.llm import prompt

from kg_construction.data.mention import Mention
from kg_construction.extraction.extractor import Extractor
from kg_construction.extraction.llm.llm_extractor import LLMExtractor
from search.retrieval.retriever import Retriever



class MultiHopNodeRetriever(Retriever):

    def __init__(self, neo4j_driver):
        super().__init__(neo4j_driver)

    def retrieve_subgraphs(self, query:str|list[str], **kwargs) -> list[networkx.Graph]:
        embedder = self.get_embedding_model(kwargs["provider"])

        node_label = kwargs.get("node_label","")
        index_name = kwargs.get("index_name","")
        embedding_property = kwargs.get("embedding_property", "embedding")

        extractor = kwargs.get("extractor")
        mentions:list[Mention] = extractor.extract_from_text(query[0], prompt=prompt.EXTRACT_MENTIONS_CONCEPT)
        top_k = kwargs.get("top_k", 5)
        hops = kwargs.get("hops", 2)
        threshold = kwargs.get("threshold", 0.5)
        result = []
        for m in mentions:
            result.extend(self.vector_search(m.term, index_name, node_label, embedding_property, top_k, threshold, embedder))
        graph_result_list = []
        for n in result:
            with self.neo4j_driver.session() as session:
                query = f'''
                    MATCH p = (start) - [x] - {{1,{hops} }}(end)
                    WHERE
                    NONE(i
                    IN
                    range(0, size(nodes(p)) - 2)
                    WHERE
                    'SemanticType'
                    IN
                    labels(nodes(p)[i])
                    AND
                    'SemanticType'
                    IN
                    labels(nodes(p)[i + 1])) AND
                    elementId(start) = '{n['node']['id']}'
                    RETURN
                    p'''
                result = session.run(query)
                g:nx.Graph = self.create_networkx_graph(result)
                graph_result_list.append(g)
        for g in graph_result_list:
            print("|V| = {} |E| = {}".format(g.number_of_nodes(), g.number_of_edges()))

        combined_graph = nx.compose_all(graph_result_list)
        print("|V| = {} |E| = {}".format(combined_graph.number_of_nodes(), combined_graph.number_of_edges()))
        return [combined_graph]

    def create_networkx_graph(self, result):
        g = nx.DiGraph()  # Create a directed graph (use nx.Graph() for undirected)
        # Step 3: Process the query result
        for record in result:
            path = record["p"]  # Extract the path from the result

            # Extract nodes and relationships from the path
            nodes = path.nodes
            relationships = path.relationships

            # Add nodes to the graph
            for node in nodes:
               g.add_node(node.id, key=node.labels, **node._properties)  # Add node with its properties

            # Add relationships as edges
            for rel in relationships:
                g.add_edge(rel.start_node.id, rel.end_node.id, key=rel.type, **rel._properties)
        return g