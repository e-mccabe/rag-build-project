"""
Generates evaluation score for the RAG system using the below metrics
> Recall
> Precision
"""
import json
from pathlib import Path

import pandas as pd

from rag_build.config import PATHS
from rag_build.response import ask


def _load_evaluation_set(path:Path = PATHS.eval_dir) -> pd.DataFrame:
    """Load the evaluation set from JSON to a DataFrame.
    
    Defaults to the project's eval_set/evaluation_set.json
    """
    with open(path / 'evaluation_set.json', encoding= 'utf-8') as f:
        return pd.DataFrame(json.load(f))

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

