import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy.stats import pearsonr
import numpy as np

nltk.download('punkt', quiet=True)

def get_document_unigram_probs(text):
    """Stable unigram probs from document itself."""
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    total_words = len(words)
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    return word_probs

def get_token_probs_unigram(text, word_probs):
    """Compute P(text) using document unigram distribution."""
    tokens = word_tokenize(text.lower())
    probs = []
    for token in tokens:
        prob = word_probs.get(token, 1e-6)  # Smooth OOV
        probs.append(prob)
    return np.array(probs)

def compute_sdc_unigram(D, S):
    """SDC using document unigram distribution (stable, no training needed)."""
    
    # Get unigram probs from D
    word_probs_D = get_document_unigram_probs(D)
    
    # P(D): token probs under D's own distribution
    P_D = get_token_probs_unigram(D, word_probs_D)
    
    # P(D|S): token probs under D's distribution (context ignored for unigram)
    # Summary doesn't change unigram probs, but we simulate context shift
    full_text = S + ' ' + D
    word_probs_full = get_document_unigram_probs(full_text)  # Train on S+D
    
    P_DS = get_token_probs_unigram(D, word_probs_full)
    
    # Align lengths
    min_len = min(len(P_D), len(P_DS))
    P_D_short = P_D[:min_len]
    P_DS_short = P_DS[:min_len]
    
    # Add small noise to avoid constant arrays
    P_D_noisy = P_D_short + np.random.uniform(1e-8, 1e-6, len(P_D_short))
    P_DS_noisy = P_DS_short + np.random.uniform(1e-8, 1e-6, len(P_DS_short))
    
    C = pearsonr(P_D_noisy, P_DS_noisy)[0]
    C_norm = (C + 1) / 2
    W = np.sum(np.abs(P_DS_short - P_D_short))
    sdc = W * C_norm
    
    return sdc

def compute_sdc_unigram_wrapper(df, verbose=False):
    df["sdc_unigram"] = 0.0
    for i, row in df.iterrows():
        original_text = row["text"]
        highlight_text = row["summary"]
        sdc = compute_sdc_unigram(original_text, highlight_text)
        if verbose:
            print(f"ID: {row['id']}, SDC Unigram: {sdc}")
        df.at[i, "sdc_unigram"] = sdc
    return df

