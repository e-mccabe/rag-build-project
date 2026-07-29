---
section: "01-Regression"
topic: "Regression"
ml_task: "supervised"
related_topics: ["PolynomialRegression", "SupportVectorRegression", "LinearAlgebra"]
summary: "A supervised method that fits a straight-line relationship between input features and a continuous target value."
keywords: ["regression", "least squares", "coefficients", "gradient descent", "prediction", "linear model"]
---

# Linear Regression

Linear regression predicts a continuous outcome by fitting a straight line, or a flat plane in higher dimensions, through the data. It is the simplest and most interpretable regression model and a baseline against which more complex models are compared.

## How It Works

The model assumes the target is a weighted sum of the input features plus an intercept. Each feature receives a coefficient that represents how much the prediction changes when that feature increases by one unit, holding others constant. Training finds the coefficients that best fit the observed data.

### The Cost Function

Fit quality is measured by the mean squared error, the average of the squared differences between predicted and actual values. Squaring penalises large errors more heavily and produces a smooth surface that is easy to optimise.

### Finding the Coefficients

The coefficients can be solved directly with the normal equation for small datasets, or learned iteratively with gradient descent, which repeatedly nudges the coefficients in the direction that reduces error. Gradient descent scales better to large datasets and many features.

## Assumptions

Linear regression assumes a roughly linear relationship between features and target, that errors are independent and have constant variance, and that features are not strongly correlated with one another. Violating these assumptions can produce misleading coefficients.

## Regularisation

When there are many features, regularised variants constrain the coefficients to prevent overfitting. Ridge regression penalises the squared size of coefficients, while Lasso penalises their absolute size and can drive some to zero, effectively selecting features.

## Strengths and Limitations

The model is fast, interpretable, and works well when relationships truly are linear. Its main limitation is that it cannot capture curved or interactive relationships without manual feature engineering, and it is sensitive to outliers.

## Use Cases

Common applications include forecasting prices, estimating demand, and quantifying how individual factors influence an outcome, where the transparency of the coefficients is valued.
