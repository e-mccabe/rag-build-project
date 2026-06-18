"""Building a local vector store using Chroma DB"""
import chromadb

from src.chunking import Chunk
from src.embedding import embed_texts

PERSIST_DIR = ".chroma"
COLLECTION_NAME = "vault_chunks"


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
