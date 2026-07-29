---
section: "06-DeepLearning"
topic: "Deep Learning"
ml_task: "training"
related_topics: ["LinearAlgebra", "NaturalLanguageProcessing", "ReenforcementLearning"]
summary: "An introduction to neural networks, how they learn through backpropagation, and the major architectures used in practice."
keywords: ["neural networks", "backpropagation", "activation", "cnn", "rnn", "deep learning"]
---

# Deep Learning

Deep learning uses neural networks with many layers to learn rich representations directly from raw data. By stacking layers, these models discover increasingly abstract features without manual feature engineering.

## The Neuron

A neuron computes a weighted sum of its inputs, adds a bias, and passes the result through an activation function. The activation introduces non-linearity, allowing the network to model complex relationships that a linear model cannot.

### Activation Functions

The ReLU function, which outputs zero for negative inputs and the value itself for positive ones, is the common default because it trains efficiently. Sigmoid and tanh squash values into fixed ranges and are used in specific places such as output gates and probabilities.

## Network Structure

Neurons are arranged in layers: an input layer receives the data, hidden layers transform it, and an output layer produces the prediction. Depth, the number of hidden layers, lets the network build a hierarchy of features, from simple edges to complex concepts.

## How It Learns

Training uses backpropagation. The network makes a prediction, a loss function measures the error, and the error is propagated backwards to compute how each weight contributed. Gradient descent then adjusts the weights to reduce the loss, repeating over many passes through the data.

## Key Architectures

Convolutional neural networks apply filters that scan across an image, making them ideal for vision. Recurrent networks process sequences by carrying state across steps and suit time series and text. Transformers use attention to relate all parts of an input at once and dominate modern language and multimodal tasks.

## Regularisation

Deep networks can overfit, so techniques like dropout, which randomly disables neurons during training, and early stopping help them generalise.

## Strengths and Limitations

Deep learning achieves state-of-the-art results on unstructured data such as images, audio, and text. Its costs are large data and compute requirements, long training times, and limited interpretability.

## Use Cases

It powers image recognition, speech systems, machine translation, recommendation, and generative models for text and images.
