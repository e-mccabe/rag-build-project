
def recall_at_k(retrieved:list[str],ground_truth:set[str],k:int)-> float:
    """Takes a ranked list & ground truth and produces recall value"""

    if ground_truth is None:
        return 0.0

    return len(ground_truth & set(retrieved[:k]))/len(ground_truth)

def precision_at_k(retrieved:list[str],ground_truth:set[str],k:int)-> float:
    """Takes a ranked list & ground truth and produces precision value"""

    top = retrieved[:k]

    if top is None:
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




    