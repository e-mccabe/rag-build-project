from openai import OpenAI

from rag_build.querying import search,rerank
from rag_build.utils import generate_numbered_context_strings
from rag_build.config import RESPONSE_MODEL,SYSTEM_PROMPT

_client = OpenAI()

def retrieval(question:str, **search_kwargs) -> list[dict]:
    """Based of user query generate the relevant chunks for the vector database"""

    hits = search(query=question,**search_kwargs)

    if not hits:
        return []
    
    return rerank(question,hits)

def ask(question: str,**search_kwargs) -> dict:

    hits = retrieval(question,**search_kwargs)

    if not hits:
        return {
            'answer':'No Answer Found in Corpus',
            'sources':['N/A'],
            'ids':['N/A']
        }

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'


    response = _client.chat.completions.create(
        model=RESPONSE_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        )
    
    sources = [f'{hit['metadata']['source']}/{'/'.join(hit['metadata']['headings'].split(','))}' for hit in hits]
    ids = [hit['id'] for hit in hits]
    return {
        'answer':response.choices[0].message.content,
        'sources':sources,
        'ids':ids
    }

def generate_stream(question: str,hits:list[dict]):

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'

    stream = _client.chat.completions.create(
         model=RESPONSE_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        stream=True
        )
    
    for piece in stream:
        yield piece
