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


def search(query:str,top_k:int = 5) ->list[dict]:

    collection = get_collection()
    query_vector = embed_texts([query])[0]

    results = collection.query(query_embeddings=[query_vector],n_results = top_k)

    query_chunks = []

    for text, metadata, distance in zip(results['documents'][0],results['metadatas'][0],results['distances'][0]):

        query_chunks.append(
            {
                'text':text,
                'metadata':metadata,
                'distance':distance
            }
        )
 
    return query_chunks


