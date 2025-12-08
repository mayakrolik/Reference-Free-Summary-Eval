import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from rouge_score import rouge_scorer

def word_tokenize(text):
    # Simple whitespace tokenizer; replace with more sophisticated one if needed
    return text.split()

def find_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms

def compute_rouge_ss_score(candidate, reference): # Requires synonym finding
    # Tokenize by words since we need to compute synonym sets
    candidate_tokens = word_tokenize(candidate)
    reference_tokens = word_tokenize(reference)
    Nr = len(reference_tokens)
    Ng = len(candidate_tokens)
    L = set()
    for r in reference_tokens:
        syns = find_synonyms(r)  # Implement this function to find synonyms
        L.update(syns)
        L.add(r)
    count = len(set(candidate_tokens) & L)
    RecallSS = count / Nr if Nr > 0 else 0
    PrecisionSS = count / Ng if Ng > 0 else 0
    if (RecallSS + PrecisionSS) == 0:
        f1_measure = 0
    else:
        f1_measure = 2 * (RecallSS * PrecisionSS) / (RecallSS + PrecisionSS)
    return {'ROUGE-SS Precision': PrecisionSS, 'ROUGE-SS Recall': RecallSS, 'ROUGE-SS F1': f1_measure}

def compute_rouge_scores(candidate, reference):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Compute standard ROUGE scores with raw inputs
    scores = scorer.score(reference, candidate)
    # Compute ROUGE-SS scores
    ss_scores = compute_rouge_ss_score(candidate, reference)
    # Combine all scores into a single dictionary and return
    scores.update(ss_scores)
    return {
        'ROUGE-1 Precision': scores['rouge1'].precision,
        'ROUGE-1 Recall': scores['rouge1'].recall,
        'ROUGE-1 F1': scores['rouge1'].fmeasure,
        'ROUGE-2 Precision': scores['rouge2'].precision,
        'ROUGE-2 Recall': scores['rouge2'].recall,
        'ROUGE-2 F1': scores['rouge2'].fmeasure,
        'ROUGE-L Precision': scores['rougeL'].precision,
        'ROUGE-L Recall': scores['rougeL'].recall,
        'ROUGE-L F1': scores['rougeL'].fmeasure,
        'ROUGE-SS Precision': scores['ROUGE-SS Precision'],
        'ROUGE-SS Recall': scores['ROUGE-SS Recall'],
        'ROUGE-SS F1': scores['ROUGE-SS F1']
    }