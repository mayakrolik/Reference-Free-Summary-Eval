import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('perluniprops', quiet=True)
nltk.download('universal_tagset', quiet=True)

def train_ngram_model(sentences, n=4):
    tokenized_sents = [word_tokenize(sent.lower()) for sent in sentences]
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_sents)
    all_tokens = [token for sent in tokenized_sents for token in sent]
    vocab = Vocabulary(all_tokens, unk_label='<unk>')
    from nltk.lm import KneserNeyInterpolated
    lm = KneserNeyInterpolated(n)
    lm.fit(train_data, vocab)
    return lm

def compute_ngram_surprisal(lm, text_or_list, context_sentences=None):
    if isinstance(text_or_list, str):
        target_sents = sent_tokenize(text_or_list)
    else:
        target_sents = text_or_list
    
    total_surprisal = 0.0
    if context_sentences:
        context_words = [w for sent in context_sentences for w in word_tokenize(sent.lower())]
    else:
        context_words = []
    
    for target_sent in target_sents:
        target_words = word_tokenize(target_sent.lower())
        for i, word in enumerate(target_words):
            full_context = context_words + target_words[:i]
            context_ngram = tuple(full_context[-lm.order+1:])
            
            if len(context_ngram) < lm.order - 1:  # Backoff for short context
                context_ngram = tuple(['<unk>'] * (lm.order - 1 - len(context_ngram))) + context_ngram
            
            logprob = lm.logscore(word, context_ngram)
            surprisal = -logprob  # Explicitly positive
            total_surprisal += max(0, surprisal)  # Clamp individual surprisals

    
    return total_surprisal

def shannon_score_ngram(document_text, summary_text, n=4):
    """Full Shannon score using NLTK 4-gram LM."""
    doc_sents = sent_tokenize(document_text)
    sum_sents = sent_tokenize(summary_text)
    
    # Train LM on document
    lm = train_ngram_model(doc_sents, n)
    
    # I(D): standalone document surprisal
    I_D = compute_ngram_surprisal(lm, doc_sents)
    
    # I(D|S): document conditional on summary context
    I_D_given_S = compute_ngram_surprisal(lm, doc_sents, sum_sents)
    
    # I(D|D): self-conditioned (last sentence or average prior context)
    if len(doc_sents) > 1:
        prior_sents = doc_sents[:-1]
        I_D_given_D = compute_ngram_surprisal(lm, [doc_sents[-1]], prior_sents)
    else:
        I_D_given_D = 0.0
    
    numerator = max(0, I_D - I_D_given_S)  # Clamp numerator
    denominator = I_D - I_D_given_D
    score = numerator / denominator if denominator > 0 else 0.0
    return score

def compute_shannon_score_wrapper(df, verbose=False):
    df["shannon_score"] = 0.0
    for i, row in tqdm(df.iterrows()):
        original_text = row["text"]
        highlight_text = row["summary"]
        shannon = shannon_score_ngram(original_text, highlight_text, n=2)
        if shannon > 0:
            print(shannon)
        if verbose:
            print(f"ID: {row['id']}, Shannon Score: {shannon}")
        df.at[i, "shannon_score"] = shannon
    return df