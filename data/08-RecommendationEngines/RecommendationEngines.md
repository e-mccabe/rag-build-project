---
section: "08-RecommendationEngines"
topic: "Recommendation Systems"
ml_task: "unsupervised"
related_topics: ["AssociateRule", "knn", "DimensionalityReduction"]
summary: "Systems that predict user preferences and suggest relevant items using collaborative filtering, content-based, and hybrid approaches."
keywords: ["recommendation", "collaborative filtering", "content-based", "matrix factorisation", "cold start", "personalisation"]
---

# Recommendation Engines

Recommendation engines predict what a user will like and surface relevant items from a large catalogue. They are central to e-commerce, streaming, and content platforms, where personalisation drives engagement.

## Content-Based Filtering

Content-based methods recommend items similar to those a user has liked before, using item features. If a user enjoys a film, the system suggests others with similar genres, actors, or themes. It works from each user's own history and needs no data about other users, but it tends to recommend more of the same and struggles to surprise.

## Collaborative Filtering

Collaborative filtering uses the behaviour of many users to make recommendations. The idea is that people who agreed in the past will agree in the future.

### User-Based and Item-Based

User-based filtering finds users with similar tastes and recommends what they liked. Item-based filtering finds items that tend to be liked by the same users and recommends those, which is often more stable as catalogues grow.

### Matrix Factorisation

Large user-item rating tables are sparse and high-dimensional. Matrix factorisation decomposes the table into latent factors that describe users and items in a shared low-dimensional space, revealing hidden preferences and filling in missing ratings.

## The Cold Start Problem

New users and new items have little or no history, making recommendations hard. Systems address this with content-based fallbacks, popularity-based suggestions, or by prompting new users for initial preferences.

## Hybrid Systems

Most production systems combine approaches, blending collaborative and content-based signals to balance accuracy, coverage, and the ability to recommend novel items.

## Evaluation

Recommenders are judged on accuracy of predicted ratings, on ranking quality measures such as precision at the top results, and on broader goals like diversity and novelty, since useful recommendations should not all be obvious.

## Use Cases

They drive product suggestions in retail, content discovery in streaming and news, and connection suggestions in social networks.
