---
section: "01-Regression"
topic: "Regression"
ml_task: "supervised"
related_topics: ["LinearRegression", "SupportVectorRegression", "ModelSelectionAndBoosting"]
summary: "An extension of linear regression that models curved relationships by adding powers of the input features."
keywords: ["polynomial", "non-linear", "curve fitting", "overfitting", "degree", "feature expansion"]
---

# Polynomial Regression

Polynomial regression captures curved relationships that a straight line cannot. It keeps the simplicity of linear regression but expands the feature set with powers of the inputs, allowing the model to bend to fit the data.

## How It Works

The technique adds new features that are powers of the originals, such as the square and cube of a feature, and then fits an ordinary linear model to this expanded set. Because the model is still linear in its coefficients, the same training methods apply even though the resulting curve is non-linear in the original feature.

### Choosing the Degree

The degree controls how flexible the curve is. A degree of two produces a single bend, while higher degrees allow more wiggles. The degree is a key hyperparameter that must be tuned, usually with cross-validation.

## The Overfitting Risk

Higher degrees fit the training data more closely but tend to capture noise rather than true patterns, producing wild swings between data points. This is a classic example of the bias-variance trade-off: too low a degree underfits, while too high a degree overfits.

## Regularisation

As with linear regression, ridge or lasso penalties can be applied to keep the coefficients small. This restrains the curve and helps a high-degree model generalise without collapsing to a straight line.

## Strengths and Limitations

Polynomial regression is easy to implement and interpret for low degrees, and it extends a familiar model to non-linear problems. Its weakness is instability at high degrees and poor behaviour when extrapolating beyond the range of the training data.

## Use Cases

It suits problems where the relationship is smooth and curved, such as modelling growth rates, physical trajectories, or dose-response curves, when the underlying shape is known to be gradual.
