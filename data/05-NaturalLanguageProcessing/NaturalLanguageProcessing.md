---
section: "05-NaturalLanguageProcessing"
topic: "Natural Language Processing"
ml_task: "pre-processing"
related_topics: ["NaiveBayes", "DeepLearning", "HiddenMarkovModels"]
summary: "An overview of how text is cleaned, represented numerically, and modelled so machines can understand and generate language."
keywords: ["nlp", "tokenisation", "embeddings", "tf-idf", "transformers", "text"]
---

# Natural Language Processing

Natural language processing (NLP) enables machines to work with human language. Because models operate on numbers, the central challenge is turning messy text into meaningful numerical representations.

## Text Preprocessing

Raw text is first cleaned and standardised. Tokenisation splits text into words or subword units. Lowercasing, removing punctuation, and stripping common stop words reduce noise. Stemming and lemmatisation collapse words to their root form so that variants are treated as one.

## Classic Representations

Early methods represent documents as collections of words. The bag-of-words model counts how often each word appears, ignoring order. TF-IDF improves on this by weighting words according to how distinctive they are, downweighting terms that appear everywhere and highlighting rare, informative ones.

## Word Embeddings

Embeddings map words to dense vectors so that words used in similar contexts sit close together in space. This captures meaning and relationships, allowing the model to see that two different words can be related. Embeddings transformed NLP by giving models a sense of semantic similarity.

## Sequence Models

Because language is sequential, models that respect word order perform better. Recurrent networks process text one token at a time while carrying a memory of earlier words, which helps with context but struggles with long passages.

## Transformers

Transformers use an attention mechanism that lets every word directly weigh the relevance of every other word, regardless of distance. This captures long-range context efficiently and in parallel, and underpins modern large language models used for translation, summarisation, and question answering.

## Common Tasks

NLP covers classification such as sentiment analysis, named entity recognition, machine translation, summarisation, and text generation. Retrieval-augmented generation combines search over a document corpus with a language model to ground answers in source material.

## Why It Matters

Effective text representation is the foundation of search, chatbots, and knowledge systems, making NLP central to extracting value from the vast amount of unstructured text organisations hold.
