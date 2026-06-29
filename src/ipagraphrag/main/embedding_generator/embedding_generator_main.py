import os
import sys

sys.path.append(os.getcwd())
from ipagraphrag.kg_construction.embedding.embedding_generator import process_label, get_embedding_model

def get_env_var(name, required=True):
    value = os.getenv(name)
    if required and not value:
        print(f"Error: Environment variable '{name}' is required.", file=sys.stderr)
        sys.exit(1)
    return value

def main():
    # --- Read Neo4j credentials from environment variables ---
    url = get_env_var("NEO4J_URI")
    username = get_env_var("NEO4J_USERNAME")
    password = get_env_var("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")  # Optional, defaults to 'neo4j'
    # --- Choose embedding provider: 'openai' or 'huggingface' ---
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    embedding = get_embedding_model(provider, os.getenv('LM_MODEL'))

    # --- Define node label configurations ---
    configs = [
        {
            "node_label": "mention",
            "text_node_properties": ["text"],
            "embedding_node_property": "embedding",
            "index_name": "mention_vector"
        },
        {
            "node_label": "chunk",
            "text_node_properties": ["text"],
            "embedding_node_property": "embedding",
            "index_name": "chunk_vector"
        },
        {
            "node_label": "SemanticType",
            "text_node_properties": ["name", "definition"],
            "embedding_node_property": "embedding",
            "index_name": "semantictype_vector"
        }
    ]

    # --- Process each node label ---
    for cfg in configs:
        process_label(
            embedding=embedding,
            url=url,
            username=username,
            password=password,
            database=database,
            node_label=cfg["node_label"],
            text_node_properties=cfg["text_node_properties"],
            embedding_node_property=cfg["embedding_node_property"],
            index_name=cfg["index_name"]
        )
    print("🎉 All embeddings generated and stored in Neo4j.")


if __name__ == "__main__":
    main()
