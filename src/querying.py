"""Building the logic to use a query to retrieve best matching chunks from chromadb"""
import json
import re
from openai import OpenAI
from src.vector_store import search
from src.config import SYSTEM_PROMPT, RESPONSE_MODEL, RERANK_PROMPT

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

def _generate_numbered_context_strings(hits:list[dict]) -> str:

    context_strings = []

    for i, hit in enumerate(hits,1):

        file = hit['metadata']['file']
        headings = hit['metadata']['headings'].split(',')
        breadcrumb = f'<{i} {file}: {' > '.join(headings)}>'
        full_text = f'{breadcrumb}\n\n{hit['text']}' 
        context_strings.append(full_text)

    return '\n\n'.join(context_strings)


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



    return reranked.choices[0].message.content

def ask(question: str,**search_kwargs) -> str:

    hits = search(query=question,**search_kwargs)

    if not hits:
        return {'answer':"I can't have anything in the notes about that",
                'sources':[]}
    

    context_string = _generate_numbered_context_strings(hits)

    input = f'User Question: {question}. **Retrieved Context: {context_string}'

    reranked = rerank(question,hits)

    response = _client.chat.completions.create(
        model=RESPONSE_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input}
        ],
        )
    
    sources = [f'{hit['metadata']['file']} > {hit['metadata']['headings']}' for hit in hits]

    return {
        'answer':response.choices[0].message.content,
        'sources':sources
    }




