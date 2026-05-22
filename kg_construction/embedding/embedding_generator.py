


import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables from a .env file if present
load_dotenv()

# Import embedding providers
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Import Neo4j vector store
from langchain_neo4j import Neo4jVector



def get_embedding_model(provider="openai", model='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Returns an embedding model instance based on the provider.
    """
    if provider == "openai":
        # You can customize the model name as needed
        return OpenAIEmbeddings(model="text-embedding-3-large")
    elif provider == "huggingface":
        # You can customize the model name as needed
        return HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{os.getenv('LM_MODEL')}",# paraphrase-multilingual-MiniLM-L12-v2
            model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU
            encode_kwargs={"normalize_embeddings": False}
        )
    else:
        print(f"Unknown embedding provider: {provider}", file=sys.stderr)
        sys.exit(1)

def process_label(
    embedding,
    url,
    username,
    password,
    database,
    node_label,
    text_node_properties,
    embedding_node_property,
    index_name
):
    """
    Generates and stores embeddings for all nodes of a given label in Neo4j.
    """
    print(f"Processing label '{node_label}' with properties {text_node_properties}...")
    try:
        driver = GraphDatabase.driver(url, auth=(username, password))
        query = f"DROP INDEX {index_name} IF EXISTS"
        embedding_null_query = f"""
            MATCH (n:{node_label})
            SET n.{embedding_node_property} = null"""
        with driver.session() as session:
            try:
                session.run(query)
                session.run(embedding_null_query)
            except Exception as e:
                print(f"Error dropping index '{index_name}': {e}")
            finally:
                driver.close()
        Neo4jVector.from_existing_graph(
            embedding=embedding,
            url=url,
            username=username,
            password=password,
            database=database,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
            index_name=index_name
        )
        print(f"✅ Embeddings stored for label '{node_label}'.")
    except Exception as e:
        print(f"❌ Error processing label '{node_label}': {e}", file=sys.stderr)
