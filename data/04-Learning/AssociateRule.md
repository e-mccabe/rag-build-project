---
section: "04-Learning"
topic: "Association Rule Learning"
ml_task: "unsupervised"
related_topics: ["RecommendationEngines", "K-Means", "TypesOfData"]
summary: "An unsupervised technique that discovers frequent co-occurrence patterns and if-then rules between items in transactional data."
keywords: ["association rules", "apriori", "support", "confidence", "lift", "market basket"]
---

# Association Rule Learning

Association rule learning finds relationships between items that frequently occur together. It is best known as the engine behind market basket analysis, which uncovers rules such as customers who buy one product often buy another.

## How It Works

The method scans transactions to find itemsets that appear together often, then turns these into rules of the form if a customer buys A then they are likely to buy B. The strength of each rule is judged by a few key measures.

### Support

Support is the fraction of all transactions that contain an itemset. It measures how common the combination is and filters out rare, unreliable patterns.

### Confidence

Confidence is the probability that the consequent appears given the antecedent. A rule with high confidence holds true in most of the transactions where the antecedent is present.

### Lift

Lift compares the confidence of a rule to what would be expected if the items were independent. A lift greater than one means the items appear together more often than chance, indicating a genuine association.

## The Apriori Algorithm

The Apriori algorithm makes the search efficient using a simple insight: if an itemset is infrequent, any larger set containing it must also be infrequent. This lets it prune the search space and avoid examining unpromising combinations. The FP-Growth algorithm achieves similar results faster by compressing the data into a tree.

## Strengths and Limitations

Association rules are intuitive and easy to interpret. Their limitations are the explosion of candidate combinations on large item catalogues and the risk of surfacing many trivial or coincidental rules.

## Use Cases

Beyond retail recommendations, it is applied to web usage analysis, fraud pattern detection, and discovering co-occurring symptoms or events in records.
