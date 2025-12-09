# 6.4610 Project: Reference-Free Text Summary Evaluation Enhancement

This project focuses on creating a new evaluation metric for reference-free summary evaluation.

We will be focusing on the following evaluation metrics:
* Distributional or Embedding Similarities
    * Shannon Score
    * KG-BERT Score (Ultimately dropped due to computation constraints)
    * SUPERT
* Grammatical/Syntactical Correctness
    * Grammatical/spelling error rate per word count (Python package language_check)
    * Flesch Reading Ease
* Cohesiveness
    * Perplexity
* Conciseness
    * Self-similarity
    * Semantic Distribution Correlation

We linearly combined these metrics and compared them against the following metrics using pairwise correlation:
* LLM Evaluation
* Rogue
* Rogue-SS
* BERTScore
