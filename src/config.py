"""Configuration of project global variables and required key"""
import os
from dotenv import load_dotenv


load_dotenv() # read .env file into the environment

def _require(name:str) -> str:
    """Runs check if environment variable is available. Fails with clear message if not available"""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. {name} needs to be added to the .env file"
        )
    
    return value


OPENAI_API_KEY = _require("OPENAI_API_KEY")

# Project Model Choice 
EMBEDDING_MODEL = 'text-embedding-3-small'

RESPONSE_MODEL = 'gpt-4o-mini'

SYSTEM_PROMPT =  """
**Role & Persona** You are a helpful and highly accurate AI Assistant

**Task** Answer user queries based strictly on context provided from the knowledge base

# 1. Context Rules
- You must base your answer ONLY on the provided context blocks.
- Do not use any prior, pre-trained or external knowledge not found in the context.
- If there is any uncertainty in the provided from the context output it as if it were the absolute truth and include a confidence rating [Low, Medium, High] alongside the answer.  
  

# 2. Hallucination and Fallback
- If the answer cannot be found in the context blocks, state explicitly: "I cannot find the answer to your questions in the provided context"
- Never make up information, guess or assume details not explicitly supported by the text.

# 3. Citation and Verifiability
- You must include inline citations for every major claim you make.
- Reference the exact document title and ID, i.e [1], corresponding to the context you used. 
- Format citations as [ID: 1, Source: Document_Name].

# 4. Formatting
- Present your response in a clear, professional and easy-to-read format.
- Be direct, concise, and avoid repetitive language.
- Adhere to the Minto principles, leading with the conclusion/answer first followed by the supporting arguements and ending with underlying data.
"""

RERANK_PROMPT = """
**Role & Persona** You are an expert relevance evaluator.

**Task** Evaluate context passages and score their relevance for a given user query in a retrieval augmented generation (RAG) system.

# 1. Evaluation Process
- Analyse the user query to identify both explicit needs and implicit context including underlying goals
- Assess each context chunk on how directly it resolves the query or provides substantive supporting information with actionable guidance
- Score based on how effectively the passage addresses the query's core intent while considering potential interpretations

# 2. Grading Criteria
<grading_scale>
10: EXCEPTIONAL match. Contains exact step-by-step instructions that perfectly match the query's specific scenario. Includes all required parameters and context. Resolves the issue completely without ambiguity. Requires no interpretation.

9: NEAR-PERFECT solution. Contains all critical steps for resolution but may lack one minor non-essential detail. Directly applicable without adaptation or assumptions.

8: STRONG MATCH. Provides complete resolution through specific instructions but may require simple logical inferences for full application. Covers all essential components with minor contextualisation needed.

7: GOOD MATCH. Addresses core aspects of the query with substantial relevant detail but lacks one important element for complete resolution. Requires some user interpretation.

6: PARTIAL MATCH. On-topic but lacks specifics for direct application. Resolves only a subset of the request.

5: LIMITED RELEVANCE. Related context or approach but indirect. Requires substantial effort to adapt to the user's exact need.

1-4: LOW RELEVANCE. Tangential mentions, keyword overlap, or general domain information with no actionable connection to the query. Score lower as relevance decreases.

0: UNRELATED. No thematic or contextual connection to the query.
</grading_scale>

# 3. Output Format
<output_format>
Return ONLY valid minified JSON with no additional text, preamble, or formatting:
{{"[i]": score, "[i]": score}}

Rules:
- Keys must be passage IDs in the format [i]
- Scores must be integers between 0 and 10, no decimals
- Maintain original passage ID order
</output_format>
"""


# Each dict maps a natural-language question to the source document that should
# be retrieved to answer it. The value matches the `source` field in chunk
# metadata 

EVALUATION_SET = [
    # 00-Prerequisites
    {"What does the dot product measure between two vectors?": "00-Prerequisites/LinearAlgebra.md"},
    {"What are eigenvectors and eigenvalues?": "00-Prerequisites/LinearAlgebra.md"},
    {"What is the difference between variance and standard deviation?": "00-Prerequisites/Statistics.md"},
    {"What does a p-value represent in hypothesis testing?": "00-Prerequisites/Statistics.md"},
    {"What is the difference between nominal and ordinal categorical data?": "00-Prerequisites/TypesOfData.md"},
    {"How is unstructured data different from structured data?": "00-Prerequisites/TypesOfData.md"},

    # 01-Regression
    {"What cost function does linear regression minimise?": "01-Regression/LinearRegression.md"},
    {"How does linear regression use coefficients to make predictions?": "01-Regression/LinearRegression.md"},
    {"How does polynomial regression model curved relationships?": "01-Regression/PolynomialRegression.md"},
    {"Why does a high polynomial degree lead to overfitting?": "01-Regression/PolynomialRegression.md"},
    {"What is the epsilon tube in support vector regression?": "01-Regression/SupportVectorRegression.md"},
    {"How does support vector regression stay robust to outliers?": "01-Regression/SupportVectorRegression.md"},

    # 02-Classification
    {"What function does logistic regression use to output a probability?": "02-Classification/LogisticRegression.md"},
    {"Why is logistic regression a classification algorithm despite its name?": "02-Classification/LogisticRegression.md"},
    {"Why is k-nearest neighbours called a lazy learner?": "02-Classification/knn.md"},
    {"How does the choice of k affect the KNN decision boundary?": "02-Classification/knn.md"},
    {"How does a support vector machine choose its decision boundary?": "02-Classification/SupportVectorMachines.md"},
    {"What are support vectors in an SVM?": "02-Classification/SupportVectorMachines.md"},
    {"What independence assumption does the Naive Bayes classifier make?": "02-Classification/NaiveBayes.md"},
    {"Why is smoothing needed in Naive Bayes to handle zero probabilities?": "02-Classification/NaiveBayes.md"},
    {"How does a decision tree decide where to split the data?": "02-Classification/DecisionTrees.md"},
    {"What is pruning in a decision tree and why is it used?": "02-Classification/DecisionTrees.md"},
    {"How does bagging work in a random forest?": "02-Classification/RandomForest.md"},
    {"What is out-of-bag evaluation in a random forest?": "02-Classification/RandomForest.md"},
    {"What is the Markov assumption in a hidden Markov model?": "02-Classification/HiddenMarkovModels.md"},
    {"What does the Viterbi algorithm do in a hidden Markov model?": "02-Classification/HiddenMarkovModels.md"},

    # 03-Clustering
    {"How does the k-means algorithm assign points to clusters?": "03-Clustering/K-Means.md"},
    {"What is the elbow method used for in k-means?": "03-Clustering/K-Means.md"},
    {"What is a dendrogram in hierarchical clustering?": "03-Clustering/HierarchicalClustering.md"},
    {"What is the difference between single and complete linkage?": "03-Clustering/HierarchicalClustering.md"},
    {"How do Gaussian mixture models produce soft cluster assignments?": "03-Clustering/GaussianMixtureModels.md"},
    {"What is the expectation-maximisation algorithm used for in a GMM?": "03-Clustering/GaussianMixtureModels.md"},

    # 04-Learning
    {"What do support, confidence, and lift measure in association rule learning?": "04-Learning/AssociateRule.md"},
    {"How does the Apriori algorithm prune its search space?": "04-Learning/AssociateRule.md"},
    {"What is the exploration versus exploitation trade-off in reinforcement learning?": "04-Learning/ReenforcementLearning.md"},
    {"How does Q-learning learn the value of actions?": "04-Learning/ReenforcementLearning.md"},

    # 05-NaturalLanguageProcessing
    {"What is TF-IDF and how does it weight words?": "05-NaturalLanguageProcessing/NaturalLanguageProcessing.md"},
    {"What is tokenisation in natural language processing?": "05-NaturalLanguageProcessing/NaturalLanguageProcessing.md"},

    # 06-DeepLearning
    {"What does backpropagation do when training a neural network?": "06-DeepLearning/DeepLearning.md"},
    {"What is the role of an activation function in a neural network?": "06-DeepLearning/DeepLearning.md"},

    # 07-DimensionalityReduction
    {"What is the curse of dimensionality?": "07-DimensionalityReduction/DimensionalityReduction.md"},
    {"What is the difference between feature selection and feature extraction?": "07-DimensionalityReduction/DimensionalityReduction.md"},
    {"How does PCA use variance to reduce dimensions?": "07-DimensionalityReduction/PCA.md"},
    {"Why must features be scaled before applying PCA?": "07-DimensionalityReduction/PCA.md"},

    # 08-RecommendationEngines
    {"What is the cold start problem in recommendation systems?": "08-RecommendationEngines/RecommendationEngines.md"},
    {"How does collaborative filtering differ from content-based filtering?": "08-RecommendationEngines/RecommendationEngines.md"},

    # 09-ModelSelection
    {"What is the bias-variance trade-off?": "09-ModelSelection/ModelSelectionAndBoosting.md"},
    {"How does boosting build models sequentially?": "09-ModelSelection/ModelSelectionAndBoosting.md"},
]