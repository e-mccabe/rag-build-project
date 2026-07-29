---
section: "02-Classification"
topic: "Classification"
ml_task: "supervised"
related_topics: ["SupportVectorRegression", "LogisticRegression", "knn"]
summary: "A classifier that finds the boundary with the widest margin between classes and uses kernels to handle non-linear data."
keywords: ["svm", "margin", "hyperplane", "kernel", "support vectors", "classification"]
---

# Support Vector Machines

Support vector machines (SVMs) classify data by finding the boundary that separates classes with the largest possible gap. This focus on the margin tends to produce models that generalise well.

## How It Works

An SVM searches for the hyperplane, a line in two dimensions or a flat surface in higher dimensions, that divides the classes. Among all possible separating hyperplanes, it chooses the one that maximises the distance to the nearest points of each class.

### Support Vectors and the Margin

The closest points to the boundary are the support vectors, and they alone define the margin. Points far from the boundary have no influence, which makes the model compact and resistant to irrelevant data.

### Soft Margins

Real data is rarely perfectly separable, so a soft margin allows some points to fall inside or across the boundary. A regularisation parameter controls the trade-off between a wide margin and the number of misclassifications tolerated.

## The Kernel Trick

When classes cannot be separated by a straight boundary, kernels map the data into a higher-dimensional space where separation becomes possible, without ever computing the new coordinates directly. The radial basis function kernel handles smooth non-linear boundaries, while polynomial kernels capture interactions.

## Strengths and Limitations

SVMs are effective in high-dimensional spaces, memory-efficient because they rely only on support vectors, and flexible through kernels. Their drawbacks are slow training on large datasets, sensitivity to parameter choices, and the lack of natural probability outputs.

## Use Cases

They perform well in text classification, image recognition, and bioinformatics, particularly when there are many features relative to the number of samples.
