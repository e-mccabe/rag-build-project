---
section: "04-Learning"
topic: "Reinforcement Learning"
ml_task: "training"
related_topics: ["DeepLearning", "Statistics", "ModelSelectionAndBoosting"]
summary: "A learning paradigm where an agent learns to act by trial and error, maximising cumulative reward through interaction with an environment."
keywords: ["reinforcement learning", "agent", "reward", "policy", "q-learning", "exploration"]
---

# Reinforcement Learning

Reinforcement learning (RL) teaches an agent to make good decisions by letting it act in an environment and learn from the rewards and penalties it receives. Unlike supervised learning, there are no labelled answers, only feedback signals.

## The Core Loop

At each step the agent observes the current state, chooses an action, and receives a reward along with a new state. Over many such interactions it learns which actions lead to the most reward in the long run.

### Key Components

The policy is the agent's strategy for choosing actions. The reward signal defines the goal by scoring each action. The value function estimates the expected long-term reward from a state, and the environment model, if available, predicts how the world responds.

## Exploration vs Exploitation

The agent faces a dilemma: it can exploit actions known to give good rewards or explore new actions that might be even better. Balancing the two is essential, as too much exploitation can trap the agent in a mediocre strategy.

## Value-Based Methods

Q-learning is a foundational approach that learns the value of taking each action in each state. The agent gradually updates these estimates and acts greedily with respect to them, converging toward an optimal policy.

## Deep Reinforcement Learning

When states are too numerous to tabulate, neural networks approximate the value function or policy. This combination, deep reinforcement learning, has mastered complex video games and board games.

## Strengths and Limitations

RL can learn sophisticated behaviour without explicit instructions and handles sequential decision problems naturally. Its challenges are sample inefficiency, instability during training, and the difficulty of designing reward signals that produce the intended behaviour.

## Use Cases

It powers game-playing agents, robotics control, recommendation sequencing, resource scheduling, and autonomous navigation.
