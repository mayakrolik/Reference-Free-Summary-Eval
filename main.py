import pandas as pd
import matplotlib.pyplot as plt
from eval_metrics import embedding_similarity, grammar_and_ease, perplexity, semantic_distribution_criterion, shannon_score, SUPERT_score
from base_metrics import compute_base_scores
from visualizations import corr_matrices, histograms
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, FunctionTransformer
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def load_and_process_data():
    """Load and process all datasets."""
    # Load non-human labeled data
    non_human_labeled = load_data("data/labeled_data/cnn_dailymail_test_labeled.csv") # No human labels
    
    # Rename to distinguish LLM scores from human score
    column_mappings = {'overall_score': 'overall_score_llm', 'information_similarity': 'information_similarity_llm',
                       'grammatical_correctness': 'grammatical_correctness_llm', 'conciseness': 'conciseness_llm',
                       'cohesion:': 'cohesion_llm'}
    non_human_labeled.rename(columns=column_mappings, inplace=True)

    # Dailymail Data, have both LLM and human labels so merge together
    dailymail_llm_labeled = load_data("data/labeled_data/dailymail_labeled.csv")
    dailymail_llm_labeled.drop(columns=['text', 'summary'], inplace=True)
    dailymail_human_labeled = load_data("data/labeled_data/dailymail_summaries.csv")
    dailymail_df = pd.merge(dailymail_llm_labeled, dailymail_human_labeled, on='id')

    # Rename to distinguish LLM scores from human score
    column_mappings = {'article': 'text', 'highlights': 'summary', 'note_human': 'note', 'overall': 'overall_score_human',
                       'overall_score': 'overall_score_llm', 'information_similarity': 'information_similarity_llm',
                       'grammatical_correctness': 'grammatical_correctness_llm', 'conciseness': 'conciseness_llm',
                       'cohesion:': 'cohesion_llm', 'accuracy': 'accuracy_human', 'coverage': 'coverage_human', 'coherence': 'coherence_human'}
    dailymail_df.rename(columns=column_mappings, inplace=True)

    # Reddit Data, have both LLM and human labels so merge together
    reddit_llm_labeled = load_data("data/labeled_data/reddit_labeled.csv")
    reddit_llm_labeled.drop(columns=['text', 'summary'], inplace=True)
    reddit_human_labeled = load_data("data/labeled_data/reddit_summaries.csv")
    reddit_df = pd.merge(reddit_llm_labeled, reddit_human_labeled, on='id')

    # Rename to distinguish LLM scores from human score
    reddit_df.rename(columns=column_mappings, inplace=True)

    return non_human_labeled, dailymail_df, reddit_df

def pipeline(df):
    """Main processing pipeline."""
    df = embedding_similarity.compute_embedding_similarity(df, verbose=False)
    df = grammar_and_ease.compute_grammar_and_ease_wrapper(df, verbose=False)
    df = perplexity.compute_perplexity(df, verbose=False)
    df = semantic_distribution_criterion.compute_sdc_unigram_wrapper(df, verbose=False)
    df = shannon_score.compute_shannon_score_wrapper(df, verbose=False)
    df = SUPERT_score.compute_supert_wrapper(df, verbose=False)
    df = compute_base_scores.compute_base_scores_wrapper(df, verbose=False)

    return df

def get_normalization_pipeline(skew):
    """Normalize distribution based on skew."""
    q01, q99 = df[col].quantile([0.01, 0.99])  # Per column
    clipper = {'a_min': q01, 'a_max': q99}
    
    if skew > 0.5:  # Right skew
        return Pipeline([('clip', FunctionTransformer(np.clip, kw_args=clipper)),
                        ('log', FunctionTransformer(np.log1p)), ('scale', MinMaxScaler())])
    elif skew < -0.5:  # Left skew
        return Pipeline([('clip', FunctionTransformer(np.clip, kw_args=clipper)),
                        ('power', PowerTransformer('yeo-johnson')), ('scale', MinMaxScaler())])
    else:  # Near-normal
        return Pipeline([('clip', FunctionTransformer(np.clip, kw_args=clipper)),
                        ('scale', MinMaxScaler())])

if __name__ == "__main__":
    # Can load in data if needed, change to False
    precomputed = True
    if not precomputed:
        non_human_labeled, dailymail_df, reddit_df = load_and_process_data()

        # Process datasets through the pipeline
        non_human_labeled_copy = non_human_labeled[:2000].copy() # Limit to first 2000 rows for speed, can change this value
        non_human_labeled_processed = pipeline(non_human_labeled_copy)
        dailymail_processed = pipeline(dailymail_df)
        reddit_processed = pipeline(reddit_df)
        
        # Save values
        non_human_labeled_processed.to_csv("data/processed/non_human_labeled_processed.csv", index=False)
        dailymail_processed.to_csv("data/processed/dailymail_processed.csv", index=False)
        reddit_processed.to_csv("data/processed/reddit_processed.csv", index=False)
    else:
        non_human_labeled_processed = pd.read_csv("data/processed/non_human_labeled_processed.csv")
        dailymail_processed = pd.read_csv("data/processed/dailymail_processed.csv")
        reddit_processed = pd.read_csv("data/processed/reddit_processed.csv")

        non_human_labeled_processed_normalized = pd.read_csv("data/processed/non_human_labeled_processed_normalized.csv")
        dailymail_processed_normalized = pd.read_csv("data/processed/dailymail_processed_normalized.csv")
        reddit_processed_normalized = pd.read_csv("data/processed/reddit_processed_normalized.csv")

    # # Generate histograms to visualize distributions
    for col in ['overall_score', 'information_similarity', 'grammatical_correctness', 'conciseness', 'cohesion', 'embedding_similarity',
                'flesch_score', 'grammar_score', 'perplexity', 'sdc_unigram', 'shannon_score', 'supert_score', 'BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']:
        histograms.generate_histogram(non_human_labeled_processed, column=col, title=f'Non-Human Labeled: {col.replace("_", " ").title()}', xlabel=col.replace("_", " ").title(), ylabel='Frequency')
    for col in ['overall_score_llm', 'information_similarity_llm', 'grammatical_correctness_llm', 'conciseness_llm', 'cohesion', 
                'accuracy_human', 'coverage_human', 'coherence_human', 'overall_score_human', 'embedding_similarity', 'flesch_score', 
                'grammar_score', 'perplexity', 'sdc_unigram', 'shannon_score', 'supert_score', 'BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']:
        histograms.generate_histogram(dailymail_processed, column=col, title=f'Dailymail: {col.replace("_", " ").title()}', xlabel=col.replace("_", " ").title(), ylabel='Frequency')
        histograms.generate_histogram(reddit_processed, column=col, title=f'Reddit: {col.replace("_", " ").title()}', xlabel=col.replace("_", " ").title(), ylabel='Frequency')
    
    # # Normalization step, ensure values are between 0 and 1

    # for df in [non_human_labeled_processed, dailymail_processed, reddit_processed]:
    #     for col in df.columns:
    #         if col == 'id' or not np.issubdtype(df[col].dtype, np.number):
    #             continue  # Skip non-numeric columns

    #         skewness = df[col].skew()
    #         pipeline = get_normalization_pipeline(skewness)
    #         df[col] = pipeline.fit_transform(df[[col]])
        
    # # Save normalized dataframes
    # non_human_labeled_processed.to_csv("data/processed/non_human_labeled_processed_normalized.csv", index=False)
    # dailymail_processed.to_csv("data/processed/dailymail_processed_normalized.csv", index=False)
    # reddit_processed.to_csv("data/processed/reddit_processed_normalized.csv", index=False)

    # Generate histograms after normalization
    for col in ['overall_score', 'information_similarity', 'grammatical_correctness', 'conciseness', 'cohesion', 'embedding_similarity',
                'flesch_score', 'grammar_score', 'perplexity', 'sdc_unigram', 'shannon_score', 'supert_score', 'BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']:
        histograms.generate_histogram(non_human_labeled_processed_normalized, column=col, title=f'Non-Human Labeled (Normalized): {col.replace("_", " ").title()}', xlabel=col.replace("_", " ").title(), ylabel='Frequency')
    for col in ['overall_score_llm', 'information_similarity_llm', 'grammatical_correctness_llm', 'conciseness_llm', 'cohesion', 
                'accuracy_human', 'coverage_human', 'coherence_human', 'overall_score_human', 'embedding_similarity', 'flesch_score', 
                'grammar_score', 'perplexity', 'sdc_unigram', 'shannon_score', 'supert_score', 'BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']:
        histograms.generate_histogram(dailymail_processed_normalized, column=col, title=f'Dailymail (Normalized): {col.replace("_", " ").title()}', xlabel=col.replace("_", " ").title(), ylabel='Frequency')
        histograms.generate_histogram(reddit_processed_normalized, column=col, title=f'Reddit (Normalized): {col.replace("_", " ").title()}', xlabel=col.replace("_", " ").title(), ylabel='Frequency')
                
    # # Compute correlations and generate plots as needed, ensure only numeric columns are used
    # non_human_numeric = non_human_labeled_processed.select_dtypes(include=[np.number])
    # dailymail_numeric = dailymail_processed.select_dtypes(include=[np.number])
    # reddit_numeric = reddit_processed.select_dtypes(include=[np.number])

    # corr_matrices.generate_correlation_heatmap(non_human_numeric, title='Non-Human Labeled Data Correlation Heatmap')
    # corr_matrices.generate_correlation_heatmap(dailymail_numeric, title='Dailymail Data Correlation Heatmap')
    # corr_matrices.generate_correlation_heatmap(reddit_numeric, title='Reddit Data Correlation Heatmap')

    # non_human_labeled_processed_normalized_numeric = non_human_labeled_processed_normalized.select_dtypes(include=[np.number])
    # dailymail_processed_normalized_numeric = dailymail_processed_normalized.select_dtypes(include=[np.number])
    # reddit_processed_normalized_numeric = reddit_processed_normalized.select_dtypes(include=[np.number])

    # corr_matrices.generate_correlation_heatmap(non_human_labeled_processed_normalized_numeric, title='Non-Human Labeled Data Correlation Heatmap (Normalized)')
    # corr_matrices.generate_correlation_heatmap(dailymail_processed_normalized_numeric, title='Dailymail Data Correlation Heatmap (Normalized)')
    # corr_matrices.generate_correlation_heatmap(reddit_processed_normalized_numeric, title='Reddit Data Correlation Heatmap (Normalized)')

    # concatenated_df = pd.concat([dailymail_processed_normalized, reddit_processed_normalized], ignore_index=True)
    # concatenated_numeric = concatenated_df.select_dtypes(include=[np.number])
    # # corr_matrices.generate_correlation_heatmap(concatenated_numeric, title='Concatenated Data Correlation Heatmap, All Metrics')

    # concatenated_numeric.rename(columns={'cohesion': 'cohesion_llm'}, inplace=True)
    # specific_metrics = concatenated_numeric[['embedding_similarity', 'flesch_score', 'grammar_score', 'perplexity', 'sdc_unigram', 'shannon_score', 'supert_score', 'BERT F1', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']]
    # corr_matrices.generate_correlation_heatmap(specific_metrics, title='Concatenated Data Correlation Heatmap, Specific Metrics 6')

