"""Turning file system text into vectors using OpenAI's embedding model"""
from openai import OpenAI
from src.config import EMBEDDING_MODEL

_client = OpenAI()

def embed_texts(texts:list[str]) -> list[list[float]]:
    """Embed many chunks of text in one API call. Returns one vector per input"""

    response = _client.embeddings.create(model = EMBEDDING_MODEL,input=texts)

    return [item.embedding for item in response.data]

