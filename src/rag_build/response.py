from openai import OpenAI

from rag_build.querying import search,rerank
from rag_build.utils import _generate_numbered_context_strings
from rag_build.config import RESPONSE_MODEL,SYSTEM_PROMPT


_client = OpenAI()

def ask(question: str,**search_kwargs) -> str:

    hits = search(query=question,**search_kwargs)

    if not hits:
        return {'answer':"I can't have anything in the notes about that",
                'sources':[]}
    
    hits = rerank(question,hits)

    context_string = _generate_numbered_context_strings(hits)

    input = f'User Question: {question}. **Retrieved Context: {context_string}'
    

    response = _client.chat.completions.create(
        model=RESPONSE_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input}
        ],
        )
    
    sources = [f'{hit['metadata']['source']}/{'/'.join(hit['metadata']['headings'].split(','))}' for hit in hits]

    return {
        'answer':response.choices[0].message.content,
        'sources':sources
    }
