"""
Need to build valuable ground-truth QA pairs

Develop the set as a hybrid of 2 methods
1. Hand-Authored: to reflect real human questions with no LLM bias
2. Sythetic Generation: use LLM ro read chunks and generate question
""" 
from pydantic import BaseModel
from openai import OpenAI
import random 
import json
from rag_build.embedding import get_collection
from rag_build.config import RESPONSE_MODEL, EVAL_PROMPT, DATA_PATH, MULTI_HOP_PROMPT

SEED = 10
ONE_HOP_COUNT = 40

collection = get_collection()
_client = OpenAI()


# Specify format for OpenAI response
class QACase(BaseModel):
    question : str  # Question Generated
    answer   : str  # Reference Answer

def write_evaluation_dataset():

    DATA_PATH.eval.mkdir(exist_ok=True, parents=True)

    single_hop_evals = generate_single_hop_questions(SEED)

    multi_hop_evals = generate_multi_hop_questions(SEED)

    all_evals = single_hop_evals + multi_hop_evals
    
    #eval_set = json.dump(sythetic_single_hops)

    with open (DATA_PATH.eval / 'evaluation_set.json','w',encoding = 'utf-8') as f:
        json.dump(all_evals,f,indent=2, ensure_ascii=False)

def generate_single_hop_questions(seed:int = SEED):
    """
    Randomly samples chunks and uses LLM to generate evaluation a dictionary of triples
    in the format (Question, Answer, Chunk)  
    """
    # Get all the ids in the database
    ids = collection.get()['ids']

    # Randomly sample ids
    sampled_ids = _sample_chunk_ids(ids,seed)

    ids = collection.get(ids=sampled_ids)['ids']
    chunks = collection.get(ids=sampled_ids)['documents']
    metadatas = collection.get(ids=sampled_ids)['metadatas']
    evaluation_set = []

    index = 1

    for id,chunk,metadata in zip(ids,chunks,metadatas):

        response = _client.chat.completions.parse(
            model=RESPONSE_MODEL,
            max_completion_tokens= 500,
            messages=[
                {'role':'system','content':EVAL_PROMPT},
                {'role':'user','content':chunk}
            ],
            response_format=QACase
        )
        # Response from GPT
        case = response.choices[0].message.parsed

        # If no response is given surface it
        if case is None:
            print(f'skipped {id}: {response.choices[0].finish_reason}')
            continue            

        index_string = f'sh_{index}'
        index += 1
        
        entry = {
            'id':index_string,
            'question': case.question,
            'answer': case.answer,
            'sources': [id],
            'type':'single_hop',
            'answer_mode':'answer'
            }

        evaluation_set.append(entry)

    return evaluation_set

def generate_multi_hop_questions(seed:int = SEED):

    ids = _sample_chunk_ids(collection.get()['ids'], seed=SEED+1)
    data = collection.get(ids=ids)
    docs_by_id = dict(zip(data['ids'],data['documents']))
    meta_by_id = dict(zip(data['ids'],data['metadatas']))

    evaluation_set = []

    index = 1

    for chunk_id in sorted(ids):

        chunk = docs_by_id[chunk_id]
        metadata = meta_by_id[chunk_id]

        if not chunk or not metadata:
             continue
            
        related_topics = metadata.get('related_topics',None)
        if not related_topics:
            continue

        related_topics = related_topics.split(',')
        rng = random.Random(seed+1)
        topic = rng.choice(related_topics)
        
        related = collection.get(where = {'file':topic})

        related_chunks = related['documents']
        if not related_chunks:
            print(f'No related chunks under topic = {topic}')
            continue

        full_chunks = '\n'.join([chunk] + related_chunks)
        response = _client.chat.completions.parse(
            model=RESPONSE_MODEL,
            max_completion_tokens= 500,
            messages=[
                {'role':'system','content':MULTI_HOP_PROMPT},
                {'role':'user','content':full_chunks}
            ],
            response_format=QACase
        )

        # Response from GPT
        case = response.choices[0].message.parsed

        # If no response is given surface it
        if case is None:
            print(f'skipped {chunk_id}: {response.choices[0].finish_reason}')
            continue

        
        index_string = f'mh_{index}'
        index += 1
        
        entry = {
             'id':index_string,
             'question':case.question,
             'answer':case.answer,
             'sources': related['ids'],
             'type': 'multi-hop',
             'answer_mode':'answer'
        }

        evaluation_set.append(entry)

    return evaluation_set

def _sample_chunk_ids(ids:list,seed:int):
    """Randomly samples chunk ids from the database"""
    rng = random.Random(seed)
    return rng.sample(ids,min(ONE_HOP_COUNT,len(ids)))

if __name__ == '__main__':
    write_evaluation_dataset()
    #print(generate_multi_hop_questions())