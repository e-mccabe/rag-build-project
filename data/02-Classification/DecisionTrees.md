---
section: "02-Classification"
topic: "Classification"
ml_task: "supervised"
related_topics: ["RandomForest", "ModelSelectionAndBoosting", "LogisticRegression"]
summary: "A model that splits data into branches using simple feature thresholds to reach a classification or regression decision."
keywords: ["decision tree", "splits", "entropy", "gini", "interpretable", "overfitting"]
---

# Decision Trees

A decision tree makes predictions by asking a series of yes-or-no questions about the features, following branches until it reaches a leaf that holds the answer. Its flowchart structure makes it one of the most interpretable models.

## How It Works

Starting from the root, the tree repeatedly splits the data on the feature and threshold that best separate the classes. Each split sends samples down one of two branches, and the process continues until the branches are pure or a stopping rule is met. A new sample is classified by following the questions from root to leaf.

### Choosing the Best Split

Split quality is measured by impurity. Gini impurity and entropy both quantify how mixed the classes are in a node, and the algorithm picks the split that reduces impurity the most. For regression trees, the criterion is the reduction in variance.

## Overfitting and Pruning

Left unchecked, a tree can grow until every leaf holds a single sample, memorising the training data and generalising poorly. Pruning removes branches that add little value, and limiting depth or requiring a minimum number of samples per leaf restrains growth.

## Strengths and Limitations

Trees are easy to interpret and visualise, require little data preparation, and handle both numerical and categorical features and non-linear relationships. Their weakness is instability: small changes in the data can produce very different trees, and single trees tend to overfit.

## From Trees to Ensembles

Because individual trees are unstable, they are most powerful when combined into ensembles such as random forests and gradient boosting, which average many trees to produce robust predictions.

## Use Cases

Decision trees suit problems where interpretability is essential, such as credit approval, medical triage, and rule extraction for business decisions.
