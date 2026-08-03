"""Generate the response output from a users query"""

from rag_build.llm import AI
from rag_build.prompts import PROMPTS
from rag_build.querying import rerank, search
from rag_build.utils import generate_numbered_context_strings

NO_ANSWER = 'No answer found in corpus'

def _build_prompt(question:str, hits: list[dict]) -> str:

    context_string = generate_numbered_context_strings(hits)
    return f'User Question: {question}. **Retrieved Context: {context_string}'

def format_sources(hits: list[dict]) -> list[str]:
    return [f'{hit['metadata']['source']}/{'/'.join(hit['metadata']['headings'].split(','))}' for hit in hits]

def retrieval(question:str, rerank_top_n: int | None = None,**search_kwargs) -> list[dict]:
    """Based of user query generate the relevant chunks for the vector database"""
    hits = search(query=question,**search_kwargs)

    if not hits:
        return []   

    # If no rerank parameter is passed
    if rerank_top_n is None:
        return hits
    
    return rerank(question, hits, top_n=rerank_top_n)

def ask(question: str, rerank_top_n: int | None = None, **search_kwargs) -> dict:
    """
    Answer a user query from the corpus.

    > rerank_top_n : if set, LLM-rerank the search hits and keep this many.
    """
    hits = retrieval(question, rerank_top_n=rerank_top_n, **search_kwargs)

    if not hits:
        return {
            'answer':NO_ANSWER,
            'sources':[],
            'ids':[]
        }

    prompt =  _build_prompt(question,hits)

    answer = AI.generate_text(prompt,PROMPTS.system)
    
    return {
        'answer':answer,
        'sources':format_sources(hits),
        'ids': [hit['id'] for hit in hits]
    }

def generate_stream(question: str,hits:list[dict]):

    prompt =  _build_prompt(question,hits)

    yield from AI.generate_stream(prompt, PROMPTS.system)
