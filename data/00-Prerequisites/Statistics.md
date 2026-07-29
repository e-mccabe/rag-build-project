---
section: "00-Prerequisites"
topic: "Statistics"
ml_task: "theory"
related_topics: ["LinearAlgebra", "NaiveBayes", "GaussianMixtureModels"]
summary: "Foundational statistical concepts of distributions, probability, and inference that machine learning models rely on to learn from data."
keywords: ["probability", "distribution", "mean", "variance", "bayes", "hypothesis testing"]
---

# Statistics

Statistics provides the tools for reasoning about uncertainty and drawing conclusions from data. Machine learning is in large part applied statistics: models estimate patterns from samples and quantify how confident those estimates are.

## Descriptive Statistics

Descriptive statistics summarise a dataset. The mean describes the central value, while the median is the middle value and is more robust to outliers. Variance and standard deviation measure how spread out values are around the mean, which informs feature scaling and outlier detection.

## Probability

Probability quantifies the likelihood of events on a scale from zero to one. Joint probability is the chance of two events occurring together, while conditional probability is the chance of one event given that another has occurred. These ideas drive probabilistic classifiers and generative models.

### Bayes' Theorem

Bayes' theorem updates the probability of a hypothesis after observing evidence. It combines a prior belief with the likelihood of the data to produce a posterior probability. This is the engine behind Naive Bayes classifiers and Bayesian inference.

## Distributions

A probability distribution describes how likely each possible value of a variable is. The normal (Gaussian) distribution is the familiar bell curve and appears throughout machine learning because many natural quantities and model errors approximate it. The Bernoulli and binomial distributions describe binary outcomes, and the uniform distribution assigns equal likelihood to all values.

## Sampling and Estimation

Models are trained on a sample drawn from a larger population. A statistic computed on the sample, such as the mean, is an estimate of the true population value. Larger and more representative samples produce estimates with less variance, which is why data quantity and quality matter so much.

## Hypothesis Testing

Hypothesis testing decides whether an observed effect is likely real or due to chance. A p-value measures how surprising the data would be if there were no real effect. In machine learning this thinking supports comparing models and validating that an improvement is statistically meaningful rather than noise.

## Why It Matters

Statistical literacy helps practitioners choose appropriate models, interpret their outputs honestly, and avoid being fooled by random patterns. It also underlies evaluation, where metrics are themselves statistical estimates with uncertainty.
