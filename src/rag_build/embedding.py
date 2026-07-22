"""Turning file system text into vectors using OpenAI's embedding model"""
from pathlib import Path

import chromadb
from openai import OpenAI
from rag_build.config import EMBEDDING_MODEL

# Anchored to the project root so the store resolves to the same place no matter
# which directory the caller runs from
def _find_root(marker = 'pyproject.toml'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent/marker).exists():
            return parent
    raise FileNotFoundError(f'No file: {marker} found above {__file__}')


PERSIST_DIR = _find_root() / ".chroma"
#PERSIST_DIR = str(Path(__file__).resolve().parents[2] / ".chroma")
COLLECTION_NAME = "vault_chunks"

_client = OpenAI()


def _flatten_metadata_lists(metadata:dict) -> dict:
    """Takes metadata dictionary and flatten any values that are lists to string values to pass to chromadb"""
    flat_metadata = {}

    for key,value in metadata.items():
        flat_metadata[key] = ','.join(value) if isinstance(value, list) else value 
    
    return flat_metadata

def embed_texts(texts:list[str]) -> list[list[float]]:
    """Embed many chunks of text in one API call. Returns one vector per input"""

    response = _client.embeddings.create(model = EMBEDDING_MODEL,input=texts)

    return [item.embedding for item in response.data]


def get_collection():
    """calls the chromadb collection"""
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={'hnsw:space':"cosine"}
    )


def index_chunks(chunks:list[str]) -> None:
    """upsert the chunks to chromadb, using document path and the chunk index within the document"""
    collection = get_collection()

    # Building deterministic ids for chromadb
    ids = [f'{chunk.metadata['source']}_{chunk.metadata['index']}' for chunk in chunks]
    
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(chunk_texts)
    metadata = [_flatten_metadata_lists(chunk.metadata) for chunk in chunks]

    collection.upsert(
        ids         = ids,
        embeddings  =embeddings,
        documents   =chunk_texts,
        metadatas    = metadata
    )