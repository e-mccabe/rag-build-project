---
section: "07-DimensionalityReduction"
topic: "Dimensionality Reduction"
ml_task: "pre-processing"
related_topics: ["PCA", "LinearAlgebra", "knn"]
summary: "Techniques that reduce the number of features in data while preserving its important structure to improve efficiency and visualisation."
keywords: ["dimensionality reduction", "curse of dimensionality", "feature selection", "projection", "visualisation", "tsne"]
---

# Dimensionality Reduction

Dimensionality reduction compresses data with many features into fewer dimensions while keeping as much meaningful information as possible. It addresses problems that arise when datasets have too many features relative to samples.

## The Curse of Dimensionality

As the number of features grows, data becomes sparse and points spread far apart, so distance-based methods lose meaning and models need exponentially more data to learn reliably. Reducing dimensions counteracts this and often improves both speed and accuracy.

## Feature Selection vs Feature Extraction

There are two broad strategies. Feature selection keeps a subset of the original features, discarding redundant or irrelevant ones, which preserves interpretability. Feature extraction creates new features that are combinations of the originals, capturing the same information in fewer dimensions but losing the direct meaning of each feature.

## Linear Methods

Principal component analysis is the most common linear technique, projecting data onto the directions of greatest variance. Linear discriminant analysis is a supervised alternative that finds directions which best separate known classes.

## Non-Linear Methods

When structure is curved, non-linear methods help. t-SNE and UMAP are popular for visualisation, mapping high-dimensional data into two or three dimensions so that similar points stay close together. These are used to explore data rather than to feed downstream models.

## Benefits

Reducing dimensions speeds up training, lowers memory use, reduces overfitting by removing noise, and enables visualisation of otherwise unviewable high-dimensional data.

## Trade-offs

Compression always discards some information, and extracted features can be hard to interpret. The goal is to remove redundancy and noise while retaining the signal that matters for the task.

## Use Cases

It is applied before clustering and classification, for data visualisation and exploration, for compressing images and signals, and for removing noise from sensor data.
