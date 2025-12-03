import math
from collections import defaultdict, Counter
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import csv

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, corpus_text):
        tokens = word_tokenize(corpus_text.lower())
        self.vocab = set(tokens)

        # Pad with start tokens
        padded_tokens = ["<s>"] * (self.n - 1) + tokens

        # Count ngrams and contexts
        for ng in ngrams(padded_tokens, self.n):
            self.ngram_counts[ng] += 1
            self.context_counts[ng[:-1]] += 1

    def ngram_prob(self, ngram):
        """
        Laplace-smoothed probability.
        P(w_n | w_1 ... w_{n-1})
        """
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[ngram[:-1]]
        V = len(self.vocab)

        return (count_ngram + 1) / (count_context + V)


def perplexity(lm, text):
    tokens = word_tokenize(text.lower())
    padded_tokens = ["<s>"] * (lm.n - 1) + tokens

    log_prob_sum = 0
    N = 0

    for ng in ngrams(padded_tokens, lm.n):
        prob = lm.ngram_prob(ng)
        log_prob_sum += math.log(prob)
        N += 1

    # Perplexity = exp( -1/N * Σ log p(token) )
    return math.exp(-log_prob_sum / N)

if __name__ == "__main__":
    # input_file = "Data/reddit_summaries.csv"
    # output_file = "Data/reddit_perplexity.csv"

    input_file = "Data/dailymail_summaries.csv"
    output_file = "Data/reddit_perplexity.csv"

    # input_file = "Data/reddit_summaries.csv"
    # output_file = "Data/reddit_perplexity.csv"

    unlabeled = False
    with open(input_file, mode="r", newline="", encoding="utf-8") as infile, open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["id", "perplexity"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            print(f"attempting to read row #{i}")
            original_text = ""
            highlight_text = ""
            if unlabeled:
                highlight_text = row["summary"]
                original_text = row["text"]
            else:
                highlight_text = row["highlights"]
                original_text = row["article"]

            language_model = NGramLanguageModel(3)
            language_model.train(original_text)

            w_row = {
                "id": row["id"],
                "perplexity": perplexity(language_model, highlight_text),
            }

            writer.writerow(w_row)