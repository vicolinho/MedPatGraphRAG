import sys

import networkx
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from neo4j import Driver
import os

class Retriever(object):

    def __init__(self, neo4j_driver:Driver):
        self.neo4j_driver = neo4j_driver

    def get_embedding_model(self, provider="openai"):
        """
        Returns an embedding model instance based on the provider.
        """
        if provider == "openai":
            # You can customize the model name as needed
            return OpenAIEmbeddings(model="text-embedding-3-large")
        elif provider == "huggingface":
            # You can customize the model name as needed
            return HuggingFaceEmbeddings(
                model_name=f"{os.getenv('LM_MODEL')}",#paraphrase-multilingual-MiniLM-L12-v2, all-mpnet-base-v2"
                model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU
                encode_kwargs={"normalize_embeddings": False}
            )
        else:
            print(f"Unknown embedding provider: {provider}", file=sys.stderr)
            sys.exit(1)

    def generate_embedding(self, query_text: str, embedder) -> list[float]:
        """
        Transform a plain-text query into a dense float vector.

        Parameters
        ----------
        query_text : str
            The natural-language search query.
        embedder : Embeddings
            A LangChain embeddings instance.

        Returns
        -------
        list[float]
            The embedding vector (length == vector.dimensions in the index).
        """
        try:
            embedding = embedder.embed_query(query_text)
            print(f"[INFO] Embedding generated – dimensions: {len(embedding)}")
            return embedding
        except Exception as exc:
            print(f"[ERROR] Failed to generate embedding: {exc}")
            raise

    def vector_search(self,
            query_text: str,
            data_source: str,
            index_name: str,
            node_label: str,
            embedding_property: str,
            top_k: int,
            threshold: float,
            embedding_model: Embeddings
    ) -> list[dict]:
        """
        End-to-end vector search: embed a query and retrieve similar Neo4j nodes.

        Automatically selects the PRIMARY (vector index) or FALLBACK (brute-force
        cosine similarity) approach depending on whether the index exists.

        Parameters
        ----------
        query_text : str
            Natural-language search query.
        driver : GraphDatabase
            Neo4j connection
        index_name : str
            Name of the Neo4j vector index (PRIMARY approach).
        node_label : str
            Label of nodes to scan (FALLBACK approach).
        embedding_property : str
            Node property that holds the embedding (FALLBACK approach).
        top_k : int
            Number of results to return.
        threshold : float
            Minimum cosine similarity score for the FALLBACK approach (0–1).
        embedding_model : str
            OpenAI embedding model identifier.
        api_key : str
            OpenAI API key.

        Returns
        -------
        list[dict]
            Sorted list of {"node": {...}, "score": float} dicts.
        """

        # ------------------------------------------------------------------
        # 1. Generate the query embedding
        # ------------------------------------------------------------------
        print(f"\n[INFO] Generating embedding for query: '{query_text}'")
        embedding = self.generate_embedding(query_text, embedding_model)

        # ------------------------------------------------------------------
        # 2. Connect to Neo4j and run the appropriate search
        # ------------------------------------------------------------------

        results: list[dict] = []

        try:
            with self.neo4j_driver.session() as session:
                # Choose approach based on index availability
                use_index = self.check_vector_index_exists(session, index_name)

                if use_index:
                    print(
                        f"[INFO] Vector index '{index_name}' found → "
                        "using PRIMARY approach (db.index.vector.queryNodes)."
                    )
                    results = self.search_with_vector_index(
                        session, index_name, embedding, top_k
                    )
                else:
                    print(
                        f"[WARN] Vector index '{index_name}' NOT found → "
                        "using FALLBACK approach (vector.similarity.cosine)."
                    )
                    results = self.search_with_cosine_similarity(
                        session,
                        node_label,
                        embedding_property,
                        embedding,
                        threshold,
                        top_k,
                        data_source
                    )
        except Exception as exc:
            print(f"[ERROR] Neo4j query failed: {exc}")
            raise
        return results

    def check_vector_index_exists(self, session, index_name: str) -> bool:
        """
        Return True if a vector index with the given name exists in the database.

        Uses SHOW INDEXES which is available in Neo4j 4.x and 5.x.
        """
        try:
            result = session.run(
                "SHOW INDEXES YIELD name, type "
                "WHERE type = 'VECTOR' AND name = $name "
                "RETURN count(*) AS cnt",
                name=index_name,
            )
            record = result.single()
            return record["cnt"] > 0 if record else False
        except Exception as exc:
            # If SHOW INDEXES fails (very old Neo4j), assume no index
            print(f"[WARN] Could not check for vector index: {exc}. Using fallback.")
            return False

    def search_with_vector_index(self,
            session,
            index_name: str,
            embedding: list[float],
            top_k: int,
    ) -> list[dict]:
        """
        PRIMARY approach – query a pre-built Neo4j vector index.

        Uses the APOC-style procedure available in Neo4j ≥ 5.15:
            CALL db.index.vector.queryNodes(indexName, k, embedding)
            YIELD node, score

        Parameters
        ----------
        session      : neo4j.Session
        index_name   : str   Name of the vector index.
        embedding    : list[float]   Query embedding vector.
        top_k        : int   Number of nearest neighbours to retrieve.

        Returns
        -------
        list[dict]  Each dict has keys 'node' (node properties) and 'score'.
        """
        cypher = """
            CALL db.index.vector.queryNodes($index, $k, $embedding)
            YIELD node, score
            RETURN node, elementId(node) as id, score
            ORDER BY score DESC
        """
        params = {"index": index_name, "k": top_k, "embedding": embedding}

        results = []
        for record in session.run(cypher, **params):
            props = dict(record["node"])
            props['id'] = record["id"]
            results.append({
                "node": props,  # convert Node to plain dict
                "score": record["score"],
            })
        return results

    def search_with_cosine_similarity(self,
            session,
            node_label: str,
            embedding_property: str,
            embedding: list[float],
            threshold: float,
            top_k: int,
            data_source: str = ""
    ) -> list[dict]:
        """
        FALLBACK approach – brute-force cosine similarity in Cypher.

        Uses the built-in function:
            vector.similarity.cosine(n.embedding, $embedding)

        This is exact but O(N) – suitable for small/medium graphs or when
        a vector index is not available.

        Parameters
        ----------
        session            : neo4j.Session
        node_label         : str    Label of nodes to scan.
        embedding_property : str    Name of the embedding property on nodes.
        embedding          : list[float]   Query embedding vector.
        threshold          : float  Minimum similarity score (0–1).
        top_k              : int    Maximum number of results to return.

        Returns
        -------
        list[dict]  Each dict has keys 'node' (node properties) and 'score'.
        :param session:
        :param node_label:
        :param embedding_property:
        :param embedding:
        :param top_k:
        :param threshold:
        :param data_source:
        """
        if node_label != "":
            node_label = ":"+node_label
        else:
            node_label = " "
        cypher = f"""
            MATCH (n{node_label})
            WHERE n.{embedding_property} IS NOT NULL
            WITH n, vector.similarity.cosine(n.{embedding_property}, $embedding) AS score
            WHERE score > $threshold AND 
               ('semantic type' IN LABELS(n) 
               OR ('chunk' IN LABELS(n) AND n.source='{data_source}')
               OR ('mention' IN LABELS(n) AND n.source='{data_source}')
               )
            RETURN n as node, elementId(n) as id, score
            ORDER BY score DESC
            LIMIT $limit
        """
        params = {
            "embedding": embedding,
            "threshold": threshold,
            "limit": top_k,
        }

        results = []
        for record in session.run(cypher, **params):
            props = dict(record["node"])
            props['id'] = record["id"]
            results.append({
                "node": props,  # convert Node to plain dict
                "score": record["score"],
            })
        return results

    def retrieve_subgraphs(self, query:str|list[str], **kwargs)-> networkx.Graph:
        pass