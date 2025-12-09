import pandas as pd
from .compute_bert_scores import compute_bert_score
from .rouge_scores import compute_rouge_scores
from tqdm import tqdm


def compute_base_scores(candidate, reference):
    """Computes all base scores for a text summary pair."""
    bert_scores = compute_bert_score(candidate, reference)
    rouge_scores = compute_rouge_scores(candidate, reference)
    return {
        'BERT Precision': bert_scores[0].mean().item(),
        'BERT Recall': bert_scores[1].mean().item(),
        'BERT F1': bert_scores[2].mean().item(),
        **rouge_scores
    }

def compute_base_scores_wrapper(df, verbose=False):
    """Computes all base scores for a dataframe containing text summary pairs."""
    # Initialize to zeroes, should these be None?
    df["BERT Precision"] = 0.0
    df["BERT Recall"] = 0.0
    df["BERT F1"] = 0.0
    df["ROUGE-1 F1"] = 0.0
    df["ROUGE-1 Recall"] = 0.0
    df["ROUGE-1 Precision"] = 0.0
    df["ROUGE-2 F1"] = 0.0
    df["ROUGE-2 Recall"] = 0.0
    df["ROUGE-2 Precision"] = 0.0
    df["ROUGE-L F1"] = 0.0
    df["ROUGE-L Recall"] = 0.0
    df["ROUGE-L Precision"] = 0.0

    # Iterate through all rows
    for i, row in tqdm(df.iterrows()):
        original_text = row["text"]
        highlight_text = row["summary"]
        scores = compute_base_scores(highlight_text, original_text)
        if verbose:
            print(f"ID: {row['id']}, Scores: {scores}")
        df.at[i, "BERT Precision"] = scores['BERT Precision']
        df.at[i, "BERT Recall"] = scores['BERT Recall']
        df.at[i, "BERT F1"] = scores['BERT F1']
        df.at[i, "ROUGE-1 F1"] = scores['ROUGE-1 F1']
        df.at[i, "ROUGE-1 Recall"] = scores['ROUGE-1 Recall']
        df.at[i, "ROUGE-1 Precision"] = scores['ROUGE-1 Precision']
        df.at[i, "ROUGE-2 F1"] = scores['ROUGE-2 F1']
        df.at[i, "ROUGE-2 Recall"] = scores['ROUGE-2 Recall']
        df.at[i, "ROUGE-2 Precision"] = scores['ROUGE-2 Precision']
        df.at[i, "ROUGE-L F1"] = scores['ROUGE-L F1']
        df.at[i, "ROUGE-L Recall"] = scores['ROUGE-L Recall']
        df.at[i, "ROUGE-L Precision"] = scores['ROUGE-L Precision']
    return df