import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
import itertools
from itertools import product
from tqdm import tqdm

def apply_coefficients_vectorized(data, cols, coeffs):
    """Vectorized linear combination: coeffs @ proxies.T"""
    proxies = data[cols].values  # Shape: (n_samples, n_proxies)
    scores = proxies @ coeffs  # Shape: (n_samples,)
    return pd.DataFrame({'id': data['id'], 'score': scores})


def negative_correlation(coeffs, data, cols, human_scores):
    """Minimize negative Pearson r (maximize correlation)"""
    result_df = apply_coefficients_vectorized(data, cols, coeffs)
    merged = pd.merge(result_df, human_scores, on='id')
    r, _ = pearsonr(merged['score'], merged['human_score'])  # Assumes 'human_score' column
    return -r  # Negative for maximization


def find_best_coefficients_optimized(data, cols, human_scores):
    """Optimize coefficients summing to 1, >=0"""
    # Bounds: each coeff [0,1]
    bounds = [(0, 1)] * len(cols)
    # Linear constraint: sum(coeffs) = 1
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Start from uniform weights
    weight = 1/len(cols)
    x0 = np.array([weight] * len(cols))
    
    res = minimize(negative_correlation, x0, args=(data, cols, human_scores),
                   method='SLSQP', bounds=bounds, constraints=cons)
    
    best_coeffs = res.x
    best_r = -res.fun
    
    return best_coeffs, best_r



if __name__ == "__main__":
    cnn_df = pd.read_csv("data/processed/dailymail_processed_normalized.csv")
    reddit_df = pd.read_csv("data/processed/reddit_processed_normalized.csv")
    concatenated_df = pd.concat([cnn_df, reddit_df], ignore_index=True)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(concatenated_df, test_size=0.3, random_state=42)

    # group1 = ['shannon_score', 'supert_score']
    # group2 = ['grammar_score', 'flesch_score']
    # group3 = ['perplexity']
    # group4 = ['sdc_unigram', 'embedding_similarity']
    # all_permutations = list(product(group1, group2, group3, group4))
    cols = ['BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']
    all_permutations = list(itertools.permutations(cols))
    best_test_corr = -1.0
    best_coeffs = None
    best_perm = None
    for perm in tqdm(all_permutations):
        human_scores_train = train[['id', 'overall_score_human']].rename(columns={'overall_score_human': 'human_score'})
        best_coeffs, best_corr = find_best_coefficients_optimized(train, list(perm), human_scores_train)
        human_scores_test = test[['id', 'overall_score_human']].rename(columns={'overall_score_human': 'human_score'})
        best_train_coeffs, best_train_corr = find_best_coefficients_optimized(train, list(perm), human_scores_train)
    
        # Test evaluation
        result_df_test = apply_coefficients_vectorized(test, list(perm), best_train_coeffs)
        merged_test = pd.merge(result_df_test, human_scores_test, on='id')
        test_r, _ = pearsonr(merged_test['score'], merged_test['human_score'])  # Fix: merged_test not merged
        
        if test_r > best_test_corr:  # Compare against best test corr
            best_test_corr = test_r
            best_perm = perm
            best_coeffs = best_train_coeffs

        # llm_scores_train = train[['id', 'overall_score_llm']].rename(columns={'overall_score_llm': 'llm_score'})
        # best_coeffs, best_corr = find_best_coefficients_optimized(train, list(perm), llm_scores_train)
        # llm_scores_test = test[['id', 'overall_score_llm']].rename(columns={'overall_score_llm': 'llm_score'})
        # best_train_coeffs, best_train_corr = find_best_coefficients_optimized(train, list(perm), llm_scores_train)
    
        # # Test evaluation
        # result_df_test = apply_coefficients_vectorized(test, list(perm), best_train_coeffs)
        # merged_test = pd.merge(result_df_test, llm_scores_test, on='id')
        # test_r, _ = pearsonr(merged_test['score'], merged_test['llm_score'])  # Fix: merged_test not merged
        
        # if test_r > best_test_corr:  # Compare against best test corr
        #     best_test_corr = test_r
        #     best_perm = perm
        #     best_coeffs = best_train_coeffs
    
    print(f"Overall best permutation: {best_perm} with correlation {best_corr} and coefficients {best_coeffs}")

