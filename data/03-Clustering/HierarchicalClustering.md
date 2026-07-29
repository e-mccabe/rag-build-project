---
section: "03-Clustering"
topic: "Clustering"
ml_task: "unsupervised"
related_topics: ["K-Means", "GaussianMixtureModels", "TypesOfData"]
summary: "An unsupervised method that builds a tree of nested clusters by repeatedly merging or splitting groups of points."
keywords: ["hierarchical", "dendrogram", "agglomerative", "linkage", "clustering", "unsupervised"]
---

# Hierarchical Clustering

Hierarchical clustering organises data into a tree of nested groups, revealing structure at many levels of granularity at once. Unlike k-means, it does not require the number of clusters to be fixed in advance.

## How It Works

The most common form is agglomerative, a bottom-up approach. Every point starts as its own cluster, and the two closest clusters are repeatedly merged until a single cluster remains. The divisive approach works top-down, starting with one cluster and splitting it.

### The Dendrogram

The sequence of merges is drawn as a dendrogram, a tree diagram where the height of each join shows how far apart the merged clusters were. Cutting the dendrogram at a chosen height produces a particular number of clusters, so the structure can be explored after the fact.

## Linkage Criteria

How the distance between clusters is measured shapes the result. Single linkage uses the closest pair of points and can produce long, chained clusters. Complete linkage uses the farthest pair and favours compact clusters. Average linkage and Ward's method, which minimises within-cluster variance, are common balanced choices.

## Strengths and Limitations

Hierarchical clustering produces an interpretable hierarchy, does not require k in advance, and can capture nested structure. Its main drawbacks are computational cost, which grows quickly with dataset size, and sensitivity to noise and the choice of linkage and distance.

## Use Cases

It is used in biology to build taxonomies and gene-expression trees, in document organisation, and in any setting where understanding relationships between groups is as valuable as the groups themselves.
