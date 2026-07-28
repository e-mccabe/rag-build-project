import pandas as pd
import json
from rag_build.config import DATA_PATH
from rag_build.response import ask
from pathlib import Path

def _load_evaluation_set(path:Path = DATA_PATH):

    with open(DATA_PATH.eval / 'evaluation_set.json') as f:
        eval_set = json.load(f)
    return pd.DataFrame(eval_set)

def case_recall(eval_row:pd.Series,response:dict):

    retrieved_chunks = response['ids']

    print(f'\nRetrieved Chunks: {retrieved_chunks}')
    # The relevant chunks for the questions
    ground_truth = set(eval_row.get('sources',[]))
    print(f'\nRelevant Chunks: {ground_truth}\n')

#    if not ground_truth:
#        return 0.0
    
    retrieved_set = set(retrieved_chunks)
    relevant_retrieved = ground_truth.intersection(retrieved_set)

    recall = len(relevant_retrieved)/len(ground_truth) if ground_truth else 0.0

    precision = len(relevant_retrieved) / len(retrieved_chunks) if retrieved_chunks else 0.0

    return round(recall,5), round(precision,5)

def evaluate(eval_set:pd.DataFrame):

    results = []

    for _,row in eval_set.iterrows():
        response = ask(row.question)

        recall,precision = case_recall(row,response)
        result = {'id':row.id,'recall':recall,'precision':precision}
        results.append(result)

    return pd.DataFrame(results)

if __name__ == '__main__':
    df = _load_evaluation_set()
    df = df.iloc[:4].copy()
    print(evaluate(df))

