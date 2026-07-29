---
section: "00-Prerequisites"
topic: "Data Types"
ml_task: "pre-processing"
related_topics: ["Statistics", "LinearRegression", "NaturalLanguageProcessing"]
summary: "An overview of the categories of data and how each type is encoded and prepared for machine learning models."
keywords: ["numerical", "categorical", "ordinal", "encoding", "structured", "unstructured"]
---

# Types of Data

The kind of data you have determines which preprocessing steps and models are appropriate. Recognising data types early prevents mistakes such as treating categories as numbers or feeding raw text directly into a numeric model.

## Numerical Data

Numerical data consists of measurable quantities. Continuous values, such as temperature or price, can take any value in a range, while discrete values, such as a count of items, take whole numbers. Numerical features are often scaled or normalised so that no single feature dominates distance and gradient calculations.

## Categorical Data

Categorical data represents groups or labels. Nominal categories, such as colour or country, have no inherent order. Ordinal categories, such as small, medium, and large, do have a meaningful order. Models require these to be converted to numbers, commonly through one-hot encoding for nominal data and ordinal encoding for ranked categories.

## Structured vs Unstructured Data

Structured data fits neatly into rows and columns, like a spreadsheet or database table, and is ready for tabular models. Unstructured data, such as text, images, and audio, has no fixed schema and must be transformed into numerical representations, for example through embeddings, before a model can use it.

## Time Series Data

Time series data is a sequence of observations ordered in time, such as daily sales. The order carries information, so it cannot be shuffled freely, and special care is needed to avoid using future information to predict the past.

## Preprocessing Considerations

Different types need different handling. Missing values may be imputed with a mean or a placeholder category. Numerical features are scaled, categorical features are encoded, and text is tokenised. Identifying the data type is the first decision in any preprocessing pipeline.

## Why It Matters

Choosing encodings and transformations that respect each data type ensures that models receive meaningful inputs. Many real-world failures come not from the model but from data that was misinterpreted or prepared incorrectly.
