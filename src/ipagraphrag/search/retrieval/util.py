import os
import sys

from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


def get_embedding_model(provider="openai"):
    """
    Returns an embedding model instance based on the provider.
    """
    if provider == "openai":
        # You can customize the model name as needed
        return OpenAIEmbeddings(model="text-embedding-3-large")
    elif provider == "huggingface":
        # You can customize the model name as needed
        return HuggingFaceEmbeddings(
            model_name=f"{os.getenv('LM_MODEL')}",  # paraphrase-multilingual-MiniLM-L12-v2, all-mpnet-base-v2"
            model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU
            encode_kwargs={"normalize_embeddings": False}
        )
    else:
        print(f"Unknown embedding provider: {provider}", file=sys.stderr)
        sys.exit(1)