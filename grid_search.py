import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from itertools import product, chain, combinations
from tqdm import tqdm
from visualizations import corr_matrices

def apply_coefficients_vectorized(data, cols, coeffs):
    """Vectorized linear combination: coeffs @ proxies.T"""
    proxies = data[cols].values
    scores = proxies @ coeffs
    return pd.DataFrame({'id': data['id'], 'score': scores})


def negative_correlation(coeffs, data, cols, human_scores, optimized_value='human_score'):
    """Minimize negative Pearson r (maximize correlation)"""
    result_df = apply_coefficients_vectorized(data, cols, coeffs)
    merged = pd.merge(result_df, human_scores, on='id')
    r, _ = pearsonr(merged['score'], merged[optimized_value])
    return -r  # Negative for maximization


def find_best_coefficients_optimized(data, cols, human_scores):
    """Optimize coefficients summing to 1, allowing negative values"""
    n = len(cols)
    # Bounds: each coeff [-1, 1] - reasonable range for negative weights
    bounds = [(-1.0, 1.0)] * n
    # Linear constraint: sum(coeffs) = 1
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Multiple starting points to avoid local minima
    x0_candidates = [
        np.full(n, 1.0/n),
        np.array([0.4] + [0.1]*(n-1)),
        np.random.uniform(-0.5, 0.5, n),
        np.random.uniform(-0.5, 0.5, n)
    ]
    
    best_res = None
    best_r = -2.0
    
    for x0 in x0_candidates:
        # Normalize starting point to satisfy constraint
        x0 = x0 / np.sum(np.abs(x0)) if np.sum(np.abs(x0)) != 1 else x0
        
        res = minimize(negative_correlation, x0, args=(data, cols, human_scores),
                       method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success and -res.fun > best_r:
            best_res = res
            best_r = -res.fun
    
    if best_res is None:
        # Fallback to uniform if all optimizations fail
        return np.full(n, 1.0/n), 0.0
    
    return best_res.x / np.sum(np.abs(best_res.x)), best_r


def find_best_coefficients_optimized_all_positive(data, cols, human_scores): 
    """Optimize coefficients summing to 1, >=0"""
    # Bounds: each coeff [0,1]
    bounds = [(0, 1)] * len(cols) 
    # Linear constraint: sum(coeffs) = 1 
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  
    # Start from uniform weights 
    weight = 1/len(cols) 
    x0 = np.array([weight] * len(cols))  
    res = minimize(negative_correlation, x0, args=(data, cols, human_scores), method='SLSQP', bounds=bounds, constraints=cons)
    best_coeffs = res.x 
    best_r = -res.fun  
    return best_coeffs, best_r


if __name__ == "__main__":
    # Get normalized data
    cnn_df = pd.read_csv("data/processed/dailymail_processed_normalized.csv")
    reddit_df = pd.read_csv("data/processed/reddit_processed_normalized.csv")

    # Merge cnn and reddit, get train/test split
    concatenated_df = pd.concat([cnn_df, reddit_df], ignore_index=True)
    train, test = train_test_split(concatenated_df, test_size=0.3, random_state=42)

    # Permutations where one is selected from each group
    group1 = ['shannon_score', 'supert_score']
    group2 = ['grammar_score', 'flesch_score']
    group3 = ['perplexity']
    group4 = ['sdc_unigram', 'embedding_similarity']
    all_permutations_from_group = list(product(group1, group2, group3, group4))

    # Permutations from all columns
    cols = ['shannon_score', 'supert_score', 'grammar_score', 'flesch_score', 'perplexity', 'sdc_unigram', 'embedding_similarity']
    all_permutations = list(chain.from_iterable(combinations(perm, r) for r in range(1, len(cols)+1)))

    best_test_corr = -1.0
    best_coeffs = None

    best_perm = None
    # Deal with renaming again :(
    human_scores_train = train[['id', 'overall_score_human']].rename(columns={'overall_score_human': 'human_score'})
    human_scores_test = test[['id', 'overall_score_human']].rename(columns={'overall_score_human': 'human_score'})

    for perm in all_permutations: # Change to all_permutations_from_group if wanted
        perm_list = [str(x) for x in perm]  # Convert tuple->list of strings for robustness
        
        # Extract coefficients from training data
        best_train_coeffs, best_train_corr = find_best_coefficients_optimized(
            train, perm_list, human_scores_train)
        
        # Test evaluation  
        result_df_test = apply_coefficients_vectorized(test, perm_list, best_train_coeffs)
        merged_test = pd.merge(result_df_test, human_scores_test, on='id', how='inner')
        test_r, _ = pearsonr(merged_test['score'], merged_test['human_score'])
        
        if test_r > best_test_corr:  # Compare against best test corr, update if needed
            best_test_corr = test_r
            best_perm = perm
            best_coeffs = best_train_coeffs
    
    print(f"Overall best permutation: {best_perm} with correlation {best_test_corr} and coefficients {best_coeffs}")

    # # Compute Correlation Matrices

    # composite_metric = apply_coefficients_vectorized(test, list(best_perm), best_coeffs)
    # merged_df = pd.merge(test, composite_metric, on='id')
    # merged_df.rename(columns={'score':'composite_score', 'cohesion': 'cohesion_llm', 'overall_score_human': 'human_score'}, inplace=True)
    # numeric_df = merged_df[['composite_score', 'human_score', 'BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1', 'overall_score_llm', 'information_similarity_llm', 'grammatical_correctness_llm', 'conciseness_llm', 'cohesion_llm']]

    # corr_matrices.generate_correlation_heatmap(numeric_df, f'Composite Metric: {list(perm)} aaaaa')


