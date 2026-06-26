"""Turning file system text into vectors using OpenAI's embedding model"""
import chromadb
from openai import OpenAI
from rag_build.config import EMBEDDING_MODEL

PERSIST_DIR = ".chroma"
COLLECTION_NAME = "vault_chunks"


_client = OpenAI()

def embed_texts(texts:list[str]) -> list[list[float]]:
    """Embed many chunks of text in one API call. Returns one vector per input"""

    response = _client.embeddings.create(model = EMBEDDING_MODEL,input=texts)

    return [item.embedding for item in response.data]

def _flatten_metadata_lists(metadata:dict) -> dict:

    flat_metadata = {}

    for key,value in metadata.items():
        flat_metadata[key] = ','.join(value) if isinstance(value, list) else value 
    
    return flat_metadata

def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={'hnsw:space':"cosine"}
    )


def index_chunks(chunks:list[str]) -> None:

    collection = get_collection()

    # Building deterministic ids for chromadb
    ids = [f'{chunk.metadata['source']}_{chunk.metadata['index']}' for chunk in chunks]
    
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(chunk_texts)
    metadatas = [_flatten_metadata_lists(chunk.metadata) for chunk in chunks]

    collection.upsert(
        ids = ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas
    )