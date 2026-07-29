---
section: "03-Clustering"
topic: "Clustering"
ml_task: "unsupervised"
related_topics: ["HierarchicalClustering", "GaussianMixtureModels", "knn"]
summary: "An unsupervised algorithm that partitions data into k groups by iteratively assigning points to the nearest cluster centre."
keywords: ["k-means", "centroids", "clustering", "unsupervised", "inertia", "elbow method"]
---

# K-Means

K-means is the most widely used clustering algorithm. It divides data into a chosen number of groups by finding cluster centres that minimise the distance between points and their assigned centre.

## How It Works

The algorithm starts by placing k cluster centres, called centroids, at random. It then repeats two steps until the assignments stop changing. First, each point is assigned to its nearest centroid. Second, each centroid is moved to the average position of the points assigned to it.

### The Objective

K-means minimises inertia, the total squared distance from each point to its centroid. Lower inertia means tighter, more compact clusters.

## Choosing the Number of Clusters

The number k must be chosen in advance. The elbow method plots inertia against k and looks for the point where adding clusters stops helping much. The silhouette score measures how well separated the clusters are and offers another guide.

## Initialisation Matters

Because the starting centroids are random, the algorithm can converge to a poor solution. The k-means++ initialisation spreads the initial centroids apart, leading to better and more consistent results.

## Strengths and Limitations

K-means is fast, simple, and scales to large datasets. Its limitations are that k must be specified, it assumes clusters are roughly spherical and similar in size, it is sensitive to outliers and scaling, and results depend on initialisation.

## Use Cases

It is used for customer segmentation, image colour compression, document grouping, and as a preprocessing step to summarise data before further analysis.
