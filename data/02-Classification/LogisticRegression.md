---
section: "02-Classification"
topic: "Classification"
ml_task: "supervised"
related_topics: ["LinearRegression", "SupportVectorMachines", "DeepLearning"]
summary: "A supervised classifier that estimates the probability of a class using a sigmoid-transformed linear combination of features."
keywords: ["classification", "sigmoid", "probability", "log loss", "decision boundary", "binary"]
---

# Logistic Regression

Despite its name, logistic regression is a classification algorithm. It predicts the probability that a sample belongs to a class and is a strong, interpretable baseline for binary and multiclass problems.

## How It Works

The model computes a weighted sum of the features, just like linear regression, and then passes the result through a sigmoid function. The sigmoid squashes any number into a value between zero and one, which is interpreted as a probability. A threshold, usually 0.5, converts the probability into a class label.

### The Decision Boundary

Because the underlying combination is linear, logistic regression draws a straight decision boundary between classes. Samples on one side are assigned to one class and those on the other side to the other class.

## The Cost Function

Training minimises log loss, also called cross-entropy, which heavily penalises confident but wrong predictions. This cost function is convex, so gradient descent reliably finds the best coefficients.

## Multiclass Classification

For more than two classes, the model is extended using a one-versus-rest scheme, training one classifier per class, or with the softmax function, which produces a probability distribution across all classes at once.

## Regularisation

Ridge and lasso penalties can be added to prevent overfitting when there are many features, shrinking coefficients and improving generalisation.

## Strengths and Limitations

Logistic regression is fast, interpretable, and outputs calibrated probabilities. Its limitation is that it can only separate classes with a linear boundary unless features are transformed, so it struggles with complex, non-linear patterns.

## Use Cases

It is widely used for spam detection, credit default prediction, and medical diagnosis, where probability estimates and the ability to interpret feature influence are important.
