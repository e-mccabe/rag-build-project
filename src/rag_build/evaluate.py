"""
Generates evaluation score for the RAG system using the below metrics
> Recall
> Precision
"""
import json
from pathlib import Path

import pandas as pd

from rag_build.config import PATHS
from rag_build.metrics import precision_at_k, recall_at_k, reciprocal_rank
from rag_build.response import retrieval


def _load_evaluation_set(path:Path = PATHS.eval_dir) -> pd.DataFrame:
    """Load the evaluation set from JSON to a DataFrame.
    
    Defaults to the project's eval_set/evaluation_set.json
    """
    with open(path / 'evaluation_set.json', encoding= 'utf-8') as f:
        df = pd.DataFrame(json.load(f))

    df['sources'] = df['sources'].apply(set)
    return df

def run_evaluation(k:int):

    eval_df = _load_evaluation_set()

    eval_results = []

    for _, row in eval_df.iterrows():
        retrieved_chunks = retrieval(row.question)
        retrieved = [chunk['id'] for chunk in retrieved_chunks]
        recall = recall_at_k(retrieved,row.sources,k)
        precision = precision_at_k(retrieved,row.sources,k) 
        rr = reciprocal_rank(retrieved,row.sources)

        result = {
            'id':row.type,
            'question':row.question,
            'sources':row.sources,
            'retrieved':retrieved,
            f'recall@{k}':recall,
            f'precision@{k}':precision,
            'reciprocal-rank':rr
        }
        eval_results.append(result)

    return pd.DataFrame(eval_results)


if __name__ == '__main__':
    df = run_evaluation(10)
    df.to_csv(PATHS.eval_dir / 'test.csv')