"""Use a user's query/prompt to retrieve the best matching chunks from the vector store (chromadb)"""
import chromadb
from pydantic import BaseModel

from rag_build.config import MODELS, get_openai_client
from rag_build.embedding import embed_texts, get_collection
from rag_build.prompts import PROMPTS
from rag_build.utils import generate_numbered_context_strings


# Define response structure for OpenAI response
class Response(BaseModel):
    chunk_ids: list[int]
    chunk_ranks: list[int]

def _inspect_collection(collection:chromadb.Collection)-> None:
    """run a check if the collection has chunks loaded, raise error if not"""
    if collection.count() == 0:
        raise ValueError (f'Collection {collection.name} is empty')

def search(query: str, top_k: int = 15, where: dict | None = None,contains: dict | None = None,max_distance: float = 0.8) ->list[dict]:
    """
    Using user query, search vector database to find the top k results by semantic similarity
    
    > top_k         : how many results to draw from the vector database
    > where         : a chromadb parameter to find chunks using metadata or distance filters
    > contains      : 
    > max_distance  :

    """
    collection = get_collection()
    _inspect_collection(collection)

    # Convert the query to a semantic vector
    query_vector = embed_texts([query])[0]

    # Apply any 'where' filters
    if where:
        results = collection.query(query_embeddings=[query_vector],n_results = top_k,where=where)

    else:
        results = collection.query(query_embeddings=[query_vector],n_results = top_k)
 
    query_chunks = []

    for text, metadata, distance,id in zip(results['documents'][0],results['metadatas'][0],results['distances'][0],results['ids'][0]):

        if distance >= max_distance:
            continue

        query_chunks.append(
            {
                'text':text,
                'metadata':metadata,
                'distance':distance,
                'id':id
            }
        )
 
    return query_chunks

def rerank(question:str,hits:list[dict],top_n:int = 5) -> list[dict]:
    """
    > Takes the top k resulting from search of the vector store by cosine similarity,
    > Asks LLM to rerank based on relevance
    """
    # Case where no relevant chunks we're found
    if not hits:
        return hits
    
    context_string = generate_numbered_context_strings(hits)

    prompt = f'User Question: {question}. **Retrieved Context: {context_string}'

    response = get_openai_client().chat.completions.parse(
        model = MODELS.reranking,
        max_tokens= 500,
        messages= [
            {'role':'system','content':PROMPTS.rerank},
            {'role':'user','content':prompt}
        ],
        response_format= Response
    )
    case = response.choices[0].message.parsed

    # Build a dictionary ordered by rank score and using original id keys
    reranked = {id:score for id,score in sorted(zip(case.chunk_ids,case.chunk_ranks),
                                                key = lambda pair: pair[1],
                                                reverse=True)}

    
    seen = set() # Initialise set to ensure no duplicates
    ranked = []
    for source_index in reranked:
        if 1 <= source_index <= len(hits) and source_index not in seen:
            seen.add(source_index)
            ranked.append(hits[source_index-1])
        if len(ranked) == top_n:
            break

    
    return ranked if ranked else hits[:top_n]

    


