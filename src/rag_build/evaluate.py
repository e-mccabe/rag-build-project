"""
Generates evaluation score for the RAG system using the below metrics
> Recall
> Precision
"""
import json
from pathlib import Path

import pandas as pd

from rag_build.config import PATHS
from rag_build.metrics import score_ranking
from rag_build.response import rerank, retrieval


def _load_evaluation_set(path:Path = PATHS.eval_dir) -> pd.DataFrame:
    """Load the evaluation set from JSON to a DataFrame.
    
    Defaults to the project's eval_set/evaluation_set.json
    """
    with open(path / 'evaluation_set.json', encoding= 'utf-8') as f:
        df = pd.DataFrame(json.load(f))

    df['sources'] = df['sources'].apply(set)
    return df


def run_retrieval_evaluation(k_values:tuple[int],rank:bool = False)-> list[dict[str,float]]:

    eval_df = _load_evaluation_set()

    eval_results = []

    for _, row in eval_df.iterrows():
        retrieved_chunks = retrieval(row.question)
        if rank:
            retrieved_chunks = rerank(retrieved_chunks)

        retrieved = [chunk['id'] for chunk in retrieved_chunks]
        scores = score_ranking(retrieved,row.sources,k_values)

        scores['type'] = row.type
        scores['answer_mode'] = row.answer_mode
        
        eval_results.append(scores)

    return pd.DataFrame(eval_results)


def summarise_results(df:pd.DataFrame)->pd.DataFrame:

    metric_cols = [c for c in df.columns if c.startswith(('recall', 'precision','mrr'))]
    overall = df[metric_cols].mean().to_frame('overall')
    by_type = df.groupby('type')[metric_cols].mean().T

    return overall.join(by_type)

    