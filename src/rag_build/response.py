"""Generate the response output from a users query"""

from rag_build.llm import AI
from rag_build.prompts import PROMPTS
from rag_build.querying import rerank, search
from rag_build.utils import generate_numbered_context_strings


def retrieval(question:str, **search_kwargs) -> list[dict]:
    """Based of user query generate the relevant chunks for the vector database"""

    hits = search(query=question,**search_kwargs)

    if not hits:
        return []
    
    return hits

def ask(question: str,top_n:int = 5,**search_kwargs) -> dict:

    full_hits = retrieval(question, **search_kwargs)

    hits = rerank(question,full_hits,top_n=top_n)

    if not hits:
        return {
            'answer':'No Answer Found in Corpus',
            'sources':['N/A'],
            'ids':['N/A']
        }

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'

    answer = AI.generate_text(prompt,PROMPTS.system)
    
    sources = [f'{hit['metadata']['source']}/{'/'.join(hit['metadata']['headings'].split(','))}' for hit in hits]
    ids = [hit['id'] for hit in hits]
    return {
        'answer':answer,
        'sources':sources,
        'ids':ids
    }

def generate_stream(question: str,hits:list[dict]):

    context_string = generate_numbered_context_strings(hits)

    prompt =  f'User Question: {question}. **Retrieved Context: {context_string}'

    yield from AI.generate_stream(prompt, PROMPTS.system)
