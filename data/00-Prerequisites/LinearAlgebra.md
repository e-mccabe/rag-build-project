---
section: "00-Prerequisites"
topic: "Linear Algebra"
ml_task: "theory"
related_topics: ["Statistics", "PCA", "DeepLearning"]
summary: "Core linear algebra objects and operations that underpin how machine learning models represent and transform data."
keywords: ["vectors", "matrices", "dot product", "eigenvectors", "norms", "linear transformation"]
---

# Linear Algebra

Linear algebra is the language machine learning uses to represent data and computation. A dataset is stored as a matrix where rows are samples and columns are features, and most model operations reduce to multiplying and decomposing these matrices.

## Vectors

A vector is an ordered list of numbers that represents a single sample or a direction in space. A row of features such as `[height, weight, age]` is a vector in three-dimensional space. Vectors can be added together and scaled by a number (a scalar), which is the basis for combining and weighting features.

### Norms

A norm measures the length or magnitude of a vector. The L2 (Euclidean) norm is the straight-line distance from the origin, while the L1 norm sums absolute values. Norms are used in distance calculations and in regularisation, where they penalise large model weights.

## Matrices

A matrix is a rectangular grid of numbers. In machine learning it usually holds an entire dataset, with one row per sample and one column per feature. Model parameters and the transformations applied to data are also expressed as matrices.

### Matrix Multiplication

Matrix multiplication combines two matrices by taking dot products of rows and columns. It lets a model apply a learned set of weights to every sample at once, which is why it is the central operation in linear models and neural networks.

## The Dot Product

The dot product multiplies two vectors element by element and sums the result. It measures how aligned two vectors are: a large positive value means they point in a similar direction. Predictions in linear models are dot products between a feature vector and a weight vector.

## Eigenvectors and Eigenvalues

An eigenvector of a matrix is a direction that is only stretched, not rotated, when the matrix is applied to it, and its eigenvalue is the amount of stretching. These describe the dominant directions of variation in data and are the foundation of dimensionality reduction techniques such as principal component analysis.

## Why It Matters

Understanding linear algebra clarifies what models actually do: they project, rotate, and scale data in high-dimensional space to separate or fit it. Efficient matrix operations are also what make training on large datasets computationally feasible.
