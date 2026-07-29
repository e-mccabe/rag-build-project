---
section: "01-Regression"
topic: "Regression"
ml_task: "supervised"
related_topics: ["SupportVectorMachines", "LinearRegression", "PolynomialRegression"]
summary: "A regression method that fits a margin of tolerance around the data and uses kernels to model non-linear relationships."
keywords: ["svr", "epsilon margin", "kernel", "support vectors", "regression", "robust"]
---

# Support Vector Regression

Support vector regression (SVR) adapts the support vector machine framework to predict continuous values. Instead of fitting a line that minimises every error, it fits a band of tolerance and only penalises predictions that fall outside it.

## How It Works

SVR defines a margin of width epsilon around the predicted function. Points that lie within this tube are considered close enough and contribute no error. Only points outside the tube, called support vectors, influence the fit, which makes the model focus on the hardest cases.

### The Epsilon Tube

The epsilon parameter sets how much deviation is tolerated. A wider tube produces a simpler, flatter function that ignores small fluctuations, while a narrower tube forces the model to track the data more closely.

### The Regularisation Parameter

A parameter often called C balances flatness of the function against tolerance for points outside the tube. A large C penalises deviations heavily and risks overfitting, while a small C produces a smoother, more general fit.

## The Kernel Trick

Like classification SVMs, SVR can use kernels to handle non-linear relationships. A kernel implicitly maps the data into a higher-dimensional space where a straight fit becomes possible, letting the radial basis function or polynomial kernels capture complex curves without explicitly computing the transformation.

## Strengths and Limitations

SVR is robust to outliers because points inside the tube are ignored, and kernels give it flexibility for non-linear problems. Its drawbacks are sensitivity to the choice of kernel and parameters, and poor scalability to very large datasets.

## Use Cases

It works well for smaller datasets with non-linear patterns and a need for robustness, such as financial forecasting, energy load prediction, and engineering measurements where outliers should not dominate the fit.
