# import numpy as np
# import pandas as pd

# def grid_search_linear_combination(data, step=0.1): # There's gotta be a better way to do this
#     """
#     Performs a grid search over coefficients e, f, g, h in [0, 1] with e+f+g+h = 1.
#     Returns a list of (e,f,g,h, result).
#     """
#     results = []

#     values = np.arange(0, 1 + step/2, step)  # ensure 1 is included

#     for e in values:
#         for f in values:
#             for g in values:
#                 h = 1 - (e + f + g)
#                 if h < 0 or h > 1:
#                     continue  # skip invalid combinations

#                 # Apply linear combination
#                 result = apply_coefficients(data, e, f, g, h)

#                 results.append((e, f, g, h, result))

#     return results

# def apply_coefficients(data, e, f, g, h):
#     """
#     Apply the linear combination with coefficients e, f, g, h to the data.
#     Returns a DataFrame with 'id' and the computed score.
#     """
#     results = []

#     # Column names for the proxies of the subjective metrics
#     information_similarity = 'a'
#     grammatical_correctness = 'b'
#     conciseness = 'c'
#     cohesion = 'd'

#     for _, row in data.iterrows():
#         id = row['id']
#         a = row[information_similarity]
#         b = row[grammatical_correctness]
#         c = row[conciseness]
#         d = row[cohesion]
#         results.append(id, a*e + b*f + c*g + d*h)

#     return pd.DataFrame(results)

# def find_best_coefficients(grid_results, human_scores):
#     best = None
#     best_correlation = 0

#     for grid_result in grid_results:
#         e, f, g, h, result = grid_result
#         df = pd.merge(result, human_scores, on='id')
#         correlation = df.corr().iloc[0,1]  # correlation between computed score and human score
#         if correlation > best_correlation:
#             best = grid_result
#             best_correlation = correlation

#     return best, best_correlation


# if __name__ == "__main__":
#     cnn_df = pd.read_csv("data/processed/dailymail_processed.csv")
#     reddit_df = pd.read_csv("data/processed/reddit_processed.csv")
#     concatenated_df = pd.concat([cnn_df, reddit_df], ignore_index=True)

#     from sklearn.model_selection import train_test_split
#     train, test = train_test_split(concatenated_df, test_size=0.3, random_state=42)

#     grid_results = grid_search_linear_combination(train)
#     best_coeffs, best_corr = find_best_coefficients(grid_results, train[['id','overall_score_human']])
#     print(f"Best coefficients: e={best_coeffs[0]}, f={best_coeffs[1]}, g={best_coeffs[2]}, h={best_coeffs[3]} with correlation {best_corr}")


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

def apply_coefficients_vectorized(data, cols, coeffs):
    """Vectorized linear combination: coeffs @ proxies.T"""
    proxies = data[cols].values  # Shape: (n_samples, 4)
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
    bounds = [(0, 1)] * 4
    # Linear constraint: sum(coeffs) = 1
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Start from uniform weights
    x0 = np.array([0.25, 0.25, 0.25, 0.25])
    
    res = minimize(negative_correlation, x0, args=(data, cols, human_scores),
                   method='SLSQP', bounds=bounds, constraints=cons)
    
    best_coeffs = res.x
    best_r = -res.fun
    
    return best_coeffs, best_r


if __name__ == "__main__":
    cnn_df = pd.read_csv("data/processed/dailymail_processed.csv")
    reddit_df = pd.read_csv("data/processed/reddit_processed.csv")
    concatenated_df = pd.concat([cnn_df, reddit_df], ignore_index=True)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(concatenated_df, test_size=0.3, random_state=42)

    # Replace with actual column names for the proxies
    cols = ['information_similarity_llm', 'grammatical_correctness_llm', 'conciseness_llm', 'cohesion_llm']
    human_scores_train = train[['id', 'overall_score_human']].rename(columns={'overall_score_human': 'human_score'})
    best_coeffs, best_corr = find_best_coefficients_optimized(train, cols, human_scores_train)
    print(f"Best coefficients: e={best_coeffs[0]}, f={best_coeffs[1]}, g={best_coeffs[2]}, h={best_coeffs[3]} with correlation {best_corr} over training set")
    print("Evaluating on test set...")
    human_scores_test = test[['id', 'overall_score_human']].rename(columns={'overall_score_human': 'human_score'})
    result_df_test = apply_coefficients_vectorized(test, cols, best_coeffs)
    merged_test = pd.merge(result_df_test, human_scores_test, on='id')
    test_r, _ = pearsonr(merged_test['score'], merged_test['human_score'])
    print(f"Correlation on test set: {test_r}")