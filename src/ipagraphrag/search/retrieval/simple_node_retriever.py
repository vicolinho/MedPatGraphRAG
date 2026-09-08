
"""
neo4j_vector_search.py
======================
Production-ready script for finding similar nodes in a Neo4j graph database
using vector (semantic) search powered by LangChain embeddings.

Workflow
--------
1. Accept a plain-text query string.
2. Convert it to a dense vector via LangChain OpenAIEmbeddings.
3. Connect to Neo4j with the official Python driver.
4. PRIMARY  – query a pre-built vector index with db.index.vector.queryNodes
              (available in Neo4j ≥ 5.15; fast approximate nearest-neighbour).
5. FALLBACK – compute cosine similarity directly in Cypher with
              vector.similarity.cosine (exact, works on any version that has
              the function; slower on large graphs).
6. Print the top-k nodes with their similarity scores.

Requirements (install before running)
--------------------------------------
    pip install neo4j langchain-openai openai

Configuration
-------------
Edit the constants in the CONFIG section below or export the corresponding
environment variables before running the script.

Usage
-----
    python neo4j_vector_search.py

You will be prompted to enter a search query at runtime, or you can call
vector_search() directly from another module.
"""

import networkx

from ipagraphrag.search.retrieval.retriever import Retriever



class SimpleNodeRetriever(Retriever):

    def __init__(self, neo4j_driver):
        super().__init__(neo4j_driver)

    def retrieve_subgraphs(self, query:str|list[str], **kwargs) -> networkx.Graph:
        embedder = kwargs["embedder"]
        searched_label_index =  kwargs.get("searched_label_index",{"":""})
        embedding_property = kwargs.get("embedding_property", "embedding")
        top_k = kwargs.get("top_k", 5)
        threshold = kwargs.get("threshold", 0.5)
        result = []
        if type(query) == list:
            for q in query:
                result.extend(self.vector_search(q, searched_label_index, embedding_property, top_k, threshold, embedder))
        else:
            result = self.vector_search(query, searched_label_index, embedding_property, top_k, threshold, embedder)
        # graph = nx.DiGraph()
        # for n in result:
        #     graph.add_node()
        return result






