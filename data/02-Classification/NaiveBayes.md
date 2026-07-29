---
section: "02-Classification"
topic: "Classification"
ml_task: "supervised"
related_topics: ["Statistics", "NaturalLanguageProcessing", "LogisticRegression"]
summary: "A probabilistic classifier built on Bayes' theorem that assumes features are conditionally independent given the class."
keywords: ["naive bayes", "probability", "bayes theorem", "conditional independence", "text classification", "prior"]
---

# Naive Bayes

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem. Despite a strong simplifying assumption, it is fast, scalable, and surprisingly effective, especially for text.

## How It Works

The classifier estimates the probability of each class given the observed features and predicts the class with the highest probability. Using Bayes' theorem, it combines the prior probability of each class with the likelihood of the features under that class.

### The Naive Assumption

The method assumes that all features are independent of one another given the class. This is rarely true in practice, but the simplification makes the maths tractable and often works well because the model only needs to rank classes correctly, not estimate exact probabilities.

## Variants

The Gaussian variant assumes continuous features follow a normal distribution. The multinomial variant models counts, such as word frequencies, and is the standard choice for text. The Bernoulli variant handles binary features indicating presence or absence.

## Handling Zero Probabilities

If a feature value never appears with a class in training, its likelihood is zero and would cancel the whole prediction. Smoothing, such as adding a small count to every possibility, prevents this and keeps the model stable.

## Strengths and Limitations

Naive Bayes trains and predicts extremely quickly, needs little data, and handles many features gracefully. Its main weakness is the independence assumption, which limits accuracy when features are strongly correlated, and its probability estimates are often poorly calibrated.

## Use Cases

It excels at spam filtering, sentiment analysis, and document categorisation, where features are word occurrences and speed matters.
