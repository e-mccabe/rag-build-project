
def recall_at_k(retrieved:list[str],ground_truth:set[str],k:int)-> float:
    """Takes a ranked list & ground truth and produces recall value"""

    if not ground_truth:
        return 0.0

    return len(ground_truth & set(retrieved[:k]))/len(ground_truth)

def precision_at_k(retrieved:list[str],ground_truth:set[str],k:int)-> float:
    """Takes a ranked list & ground truth and produces precision value"""

    top = retrieved[:k]

    if not top :
        return 0.0

    return len(ground_truth & set(top))/len(top)

def reciprocal_rank(retrieved: list[str],ground_truth: set[str]):
    """1 / (rank of the first ground truth chunk retrieved)"""

    hit_count = 0
    for position, chunk_id in enumerate(retrieved,start=1):
        if chunk_id in ground_truth:
            hit_count += 1
            return 1.0 / position
    return 0.0


def score_ranking(retrieved:list[str], ground_truth:set[str], k_values:tuple[int])-> dict[str,float]:

    scores = {'mrr': reciprocal_rank(retrieved,ground_truth)}
    for k in k_values:
        scores[f'recall@{k}'] = recall_at_k(retrieved,ground_truth,k)
        scores[f'precision@{k}'] = precision_at_k(retrieved,ground_truth,k)

    return scores


    