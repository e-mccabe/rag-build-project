---
section: "02-Classification"
topic: "Classification"
ml_task: "supervised"
related_topics: ["DecisionTrees", "ModelSelectionAndBoosting", "SupportVectorMachines"]
summary: "An ensemble method that combines many decorrelated decision trees through bagging to produce accurate, stable predictions."
keywords: ["random forest", "ensemble", "bagging", "decision trees", "feature importance", "robust"]
---

# Random Forest

A random forest is an ensemble of decision trees whose predictions are combined by voting or averaging. By aggregating many diverse trees, it overcomes the instability and overfitting of a single tree.

## How It Works

The forest trains many trees, each on a different random sample of the data drawn with replacement, a technique called bagging. At each split, each tree also considers only a random subset of features. These two sources of randomness make the trees different from one another, so their errors tend to cancel out when averaged.

### Why Decorrelation Helps

If every tree made the same mistakes, averaging would not help. By forcing trees to see different data and features, the forest reduces the correlation between them, which lowers the variance of the combined prediction without greatly increasing bias.

## Out-of-Bag Evaluation

Because each tree is trained on a bootstrap sample, the samples it did not see, the out-of-bag set, can be used to estimate accuracy without a separate validation set.

## Feature Importance

The forest can rank features by how much they reduce impurity across all trees, providing a useful, if approximate, measure of which inputs drive predictions.

## Strengths and Limitations

Random forests are accurate, robust to noise and outliers, resistant to overfitting, and require little tuning. Their drawbacks are reduced interpretability compared with a single tree, larger memory use, and slower prediction.

## Random Forest vs Boosting

Random forests build trees independently in parallel and reduce variance, while boosting builds trees sequentially to reduce bias. Forests are easier to tune, whereas boosting can reach higher accuracy with careful configuration.

## Use Cases

They are a reliable default for tabular data in tasks such as fraud detection, churn prediction, and feature ranking.
