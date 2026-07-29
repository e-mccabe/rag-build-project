---
section: "09-ModelSelection"
topic: "Model Selection and Boosting"
ml_task: "training"
related_topics: ["RandomForest", "DecisionTrees", "PolynomialRegression"]
summary: "Methods for choosing, tuning, and evaluating models, including cross-validation, the bias-variance trade-off, and boosting ensembles."
keywords: ["model selection", "cross-validation", "bias-variance", "boosting", "hyperparameters", "gradient boosting"]
---

# Model Selection and Boosting

Model selection is the process of choosing the best model and settings for a problem and verifying that it will generalise to new data. Boosting is a powerful ensemble strategy that often produces the strongest models on tabular data.

## The Bias-Variance Trade-off

Every model balances two sources of error. Bias is error from overly simple assumptions that miss real patterns, causing underfitting. Variance is error from excessive sensitivity to the training data, causing overfitting. The goal is the sweet spot that minimises total error on unseen data.

## Cross-Validation

To estimate how a model will perform on new data, cross-validation splits the data into several folds, trains on most of them, and tests on the held-out fold, rotating until every fold has been used for testing. Averaging the results gives a more reliable estimate than a single split and uses the data efficiently.

## Hyperparameter Tuning

Hyperparameters, such as tree depth or regularisation strength, are set before training and strongly affect performance. Grid search tries every combination from a defined set, random search samples combinations at random, and more advanced methods like Bayesian optimisation search intelligently for good settings.

## Ensemble Methods

Ensembles combine many models to outperform any single one. Bagging trains models in parallel on bootstrap samples and averages them to reduce variance, as in random forests. Boosting takes a different route.

### Boosting

Boosting builds models sequentially, with each new model focusing on the mistakes of the previous ones. By combining many weak learners, usually shallow trees, it gradually reduces bias and builds a strong predictor. Gradient boosting fits each new tree to the residual errors of the current ensemble, and popular implementations are widely used in competitions and industry.

## Avoiding Data Leakage

A reliable evaluation keeps the test data completely separate from training, including during preprocessing and tuning. Leaking test information produces optimistic scores that do not hold up in production.

## Why It Matters

Sound model selection ensures reported performance reflects real-world behaviour, while boosting provides a dependable path to high accuracy on structured data.
