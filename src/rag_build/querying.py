"""Building the logic to use a query to retrieve best matching chunks from chromadb"""
import json
import re
import chromadb
from openai import OpenAI

from rag_build.utils import _generate_numbered_context_strings
from rag_build.embedding import embed_texts, get_collection
from rag_build.config import RESPONSE_MODEL, RERANK_PROMPT

_client = OpenAI()

def _extract_json_block(response:str) -> dict:

    match = re.search(r'\{.*?\}',response, re.DOTALL)

    if match:
        json_patterns = match.group()

    else:
        raise ValueError(f'No JSON block found in model response: {repr(response)}')

    try:
        scores = json.loads(json_patterns)
    except json.JSONDecodeError as e:
        raise ValueError(f'Extracted block is not valid JSON: {repr(json_patterns)}') from e
    
    if not isinstance(scores,dict):
        raise TypeError(f"Expected JSON object, got {type(scores).__name__}: {repr(scores)}")

    return {int(key):value for key,value in scores.items()}

def _inspect_collection(collection:chromadb.Collection)-> None:
    
    if collection.count() == 0:
        raise ValueError (f'Collection {collection.name} is empty')


def search(query: str, top_k: int = 15, where: dict | None = None,contains: dict | None = None,max_distance: float = 0.5) ->list[dict]:

    collection = get_collection()
    _inspect_collection(collection)

    query_vector = embed_texts([query])[0]

    if where:
        results = collection.query(query_embeddings=[query_vector],n_results = top_k,where=where)

    else:
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


def rerank(question:str,hits:list[dict],top_n:int = 3) -> list[dict]:
    """Takes the top k resulting from searching the vector store for cosine similarity and asks LLM to rerank based on relevance"""
    if not hits:
        return hits

    context_string = _generate_numbered_context_strings(hits)

    input = f'User Question: {question}. **Retrieved Context: {context_string}'

    reranked = _client.chat.completions.create(
        model=RESPONSE_MODEL,
        max_tokens = 100,
        messages= [
            {"role":"system","content":RERANK_PROMPT},
            {"role":"user","content":input}
        ]
    )


    # Include defensive element if there are issues returning a valid JSON   
    ranked_dictionary = _extract_json_block(reranked.choices[0].message.content)

    seen = set()
    ranked = []
    for source_index in ranked_dictionary.keys():
        if 1 <= source_index <= len(hits) and source_index not in seen:
            seen.add(source_index)
            ranked.append(hits[source_index-1])
        if len(ranked) == top_n:
            break
    
    return ranked if ranked else hits[:top_n]




