"""
Need to build valuable ground-truth QA pairs

Develop the set as a hybrid of 2 methods
1. Hand-Authored: to reflect real human questions with no LLM bias
2. Sythetic Generation: use LLM ro read chunks and generate question
""" 
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, NamedTuple

from chromadb import Collection
from pydantic import BaseModel

from rag_build.config import PATHS
from rag_build.embedding import get_collection
from rag_build.llm import AI
from rag_build.prompts import PROMPTS

SEED = 10
SAMPLE_SIZE = 80

# OpenAI calls in flight at once
MAX_WORKERS = 8 

# Specify format for OpenAI response
class QACase(BaseModel):
    question : str  # Question Generated
    answer   : str  # Reference Answer

# Define dataclass for each sampled chunk
class SampledChunk(NamedTuple):
    id       : str
    text     : str
    metadata : dict[str,str]

def write_evaluation_dataset(path:Path = PATHS.eval_dir)-> None:
    """Writes the generated evaluation questions to a json file"""
    path.mkdir(exist_ok=True, parents=True)

    all_evals = generate_eval_set(SAMPLE_SIZE,get_collection())
    with open (path / 'evaluation_set.json','w',encoding = 'utf-8') as f:
        json.dump(all_evals,f,indent=2, ensure_ascii=False)

def _sample_chunk_ids(sample_size:int,ids:list[str],rng:random.Random)-> list[str]:
    """Randomly samples chunk ids from the database"""
    return rng.sample(ids,min(sample_size,len(ids)))

def _get_related_chunks(collection:Collection,
                        metadata:dict[str,str],
                        rng:random.Random) -> tuple[str,str]| None:
    """For multi hop questions generates a related chunk using `related topic` metadata"""

    # Get the list of related topics for the seed chunk
    related_topics = metadata.get('related_topics')
    if not related_topics:
        return

    # Choose random topic from list of topics and find the chunks in related file 
    topic = rng.choice(related_topics.split(','))
    related_chunks = collection.get(where = {'file':topic.strip()})
    if not related_chunks['ids']:
        return

    # Randomly selecting one chunk from the related topic page
    bridge_position = rng.randrange(len(related_chunks['ids']))
    bridge_id = related_chunks['ids'][bridge_position]
    bridge_text = related_chunks['documents'][bridge_position]

    return (bridge_id, bridge_text)

def _generate_case_triple(case_type:Literal['single-hop','multi-hop'],
                          answer_mode:str,
                          prompt:str,
                          collection:Collection,
                          chunk:SampledChunk,
                          rng:random.Random)->dict[str,str|list[str]] | None:
    """Uses OpenAI and relevant chunks to generate a QA question, answer and ground truth triple"""

    # Define multi-hop case
    if case_type == 'multi-hop':
        result = _get_related_chunks(collection,chunk.metadata,rng)
        if result is None:
            return None

        related_id, related_chunk = result

        # Use Chunk A and B to match instructions in the multi-hop prompts
        chunk_text = f'Chunk A:\n{chunk.text}\n\nChunk B:\n{related_chunk}'
        sources = [chunk.id,related_id]

    else:
        chunk_text = chunk.text
        sources = [chunk.id]

    # Generate Question and Answer using OpenAI and defined response format
    case = AI.generate_eval_case(chunk_text,prompt,QACase)

    if case is None:
        return None

    return {
        'question'  : case.question,
        'answer'    : case.answer,
        'sources'   : sources,
        'type'      : case_type,
        'answer_mode' : answer_mode
    }

def generate_eval_set(sample_size:int,collection:Collection,seed:int = SEED) -> list[dict[str,str|list[str]]]:
    """Generate a full evaluation set of QA triple of size n, splitting 50/50 between single and multi hop questions"""

    rng = random.Random(seed)

    sampled_ids = _sample_chunk_ids(sample_size,collection.get()['ids'],rng)
    data = collection.get(ids=sampled_ids)

    split_point = len(sampled_ids) // 2
    
    # Building work items
    tasks = []

    for ix, (chunk_id, chunk_text, metadata) in enumerate(
        zip(data['ids'],data['documents'],data['metadatas'])
    ):
        chunk = SampledChunk(id = chunk_id, text = chunk_text,metadata=metadata)
        case_type  = 'single-hop' if ix < split_point else 'multi-hop'
        prompt = PROMPTS.single_hop if case_type == 'single-hop' else PROMPTS.multi_hop    

        # Each task gets individual rng so results can be reproduced despite parralel execution order
        task_rng = random.Random(seed + ix)
        tasks.append((case_type ,prompt ,chunk ,task_rng))

    # Nested function to bundle one task and make API call
    def run(task):
        case_type, prompt, chunk, task_rng = task
        return _generate_case_triple(case_type,'answer',prompt,collection,chunk,task_rng)


    # Run all the task calls concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(run,tasks)

    return [entry for entry in results if entry is not None]

if __name__ == '__main__':
    write_evaluation_dataset()