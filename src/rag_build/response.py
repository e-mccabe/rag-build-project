"""Generate the response output from a users query"""

from rag_build.config import MODELS, get_openai_client
from rag_build.prompts import PROMPTS
from rag_build.querying import rerank, search
from rag_build.utils import generate_numbered_context_strings


def retrieval(question:str, **search_kwargs) -> list[dict]:
    """Based of user query generate the relevant chunks for the vector database"""

    hits = search(query=question,**search_kwargs)

    if not hits:
        return []
    
    return hits

def ask(question: str,**search_kwargs) -> dict:

    full_hits = retrieval(question, **search_kwargs)

    hits = rerank(question,full_hits,**search_kwargs)

    if not hits:
        return {
            'answer':'No Answer Found in Corpus',
            'sources':['N/A'],
            'ids':['N/A']
        }

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'


    response = get_openai_client().chat.completions.create(
        model=MODELS.response,
        max_tokens=500,
        messages=[
            {"role": "system", "content": PROMPTS.system},
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

    stream = get_openai_client().chat.completions.create(
        model=MODELS.response,
        max_tokens=500,
        messages=[
            {"role": "system", "content": PROMPTS.system},
            {"role": "user", "content": prompt}
        ],
        stream=True
        )
    
    for piece in stream:
        yield piece
