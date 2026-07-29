---
section: "07-DimensionalityReduction"
topic: "Dimensionality Reduction"
ml_task: "pre-processing"
related_topics: ["DimensionalityReduction", "LinearAlgebra", "K-Means"]
summary: "A linear technique that projects data onto the directions of greatest variance to reduce dimensions while retaining most information."
keywords: ["pca", "principal components", "variance", "eigenvectors", "projection", "covariance"]
---

# Principal Component Analysis

Principal component analysis (PCA) is the most widely used dimensionality reduction technique. It finds new axes, called principal components, that capture the maximum variance in the data and re-expresses the data along them.

## How It Works

PCA identifies the directions in which the data varies most. The first principal component is the single direction of greatest variance, the second is the direction of next greatest variance that is perpendicular to the first, and so on. Projecting the data onto the first few components retains most of its structure in fewer dimensions.

### The Role of Variance

Variance is treated as information. Directions with high variance are assumed to carry the meaningful signal, while low-variance directions are treated as noise and discarded. This is why PCA preserves the components that explain the most variance.

### The Mathematics

PCA is computed from the covariance matrix of the features. Its eigenvectors give the principal component directions and its eigenvalues give the amount of variance each captures. The eigenvectors with the largest eigenvalues are kept.

## Preprocessing Requirements

Features must be centred by subtracting their mean, and usually scaled to unit variance, because PCA is sensitive to the scale of features. Without scaling, features with large numeric ranges would dominate the components.

## Choosing the Number of Components

A common approach is to keep enough components to explain a target share of total variance, such as ninety-five percent. A scree plot of explained variance helps identify where adding components stops helping.

## Strengths and Limitations

PCA is fast, removes correlated and redundant features, and aids visualisation. Its limitations are that components are linear combinations and hard to interpret, it only captures linear structure, and it is sensitive to outliers.

## Use Cases

PCA is used to compress data, speed up downstream models, remove noise, and visualise high-dimensional datasets in two or three dimensions.
