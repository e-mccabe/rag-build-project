---
section: "02-Classification"
topic: "Classification"
ml_task: "supervised"
related_topics: ["SupportVectorMachines", "K-Means", "TypesOfData"]
summary: "A simple instance-based classifier that labels a sample according to the majority class among its nearest neighbours."
keywords: ["knn", "distance", "neighbours", "instance-based", "lazy learning", "classification"]
---

# K-Nearest Neighbours

K-nearest neighbours (KNN) classifies a new sample by looking at the labelled examples closest to it. It is an intuitive, non-parametric method that makes no assumptions about the shape of the data.

## How It Works

To classify a point, the algorithm measures the distance from it to every training point, selects the k closest, and assigns the class that is most common among them. For regression, it averages the neighbours' values instead of voting.

### Choosing K

The value of k controls the smoothness of the decision boundary. A small k makes the model sensitive to noise and individual points, while a large k smooths the boundary but can blur the distinction between classes. An odd k avoids ties in binary problems.

### Distance Metrics

Euclidean distance is the most common choice, but Manhattan or cosine distance may suit particular data. Because distance depends on scale, features should be normalised so that no single feature dominates.

## Lazy Learning

KNN is called a lazy learner because it does no work during training; it simply stores the data. All computation happens at prediction time, when distances to every stored point must be calculated. This makes prediction slow on large datasets.

## The Curse of Dimensionality

In high-dimensional spaces, distances between points become similar and lose meaning, which degrades KNN's performance. Dimensionality reduction is often applied first to keep the method effective.

## Strengths and Limitations

KNN is simple, has no training phase, and naturally handles multiclass problems and complex boundaries. Its weaknesses are slow prediction, high memory use, sensitivity to feature scaling, and poor behaviour in high dimensions.

## Use Cases

It suits small to medium datasets with meaningful distance relationships, such as recommendation by similarity, handwriting recognition, and basic anomaly detection.
