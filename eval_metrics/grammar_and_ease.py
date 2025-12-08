import nltk
import re
import csv
import harper_py
from tqdm import tqdm
import sys
from contextlib import redirect_stdout, redirect_stderr
import io

nltk.download('cmudict')
from nltk.corpus import cmudict

def flesch_reading_ease(text: str):
    """
    Calculates the Flesch reading ease score
    https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """

    d = cmudict.dict()

    def count_syllables_cmu(word):
        word_lower = word.lower()
        if word_lower in d:
            # Get all possible pronunciations for the word
            pronunciations = d[word_lower]
            # For each pronunciation, count the number of phonemes with a digit (stress marker)
            syllable_counts = [len([p for p in pron if p[-1].isdigit()]) for pron in pronunciations]
            # You might choose to return the minimum, maximum, or a specific pronunciation's count
            return min(syllable_counts) if syllable_counts else 0
        return 0 # Word not found in CMU dict

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_words = len(words)

    if num_words == 0:
        return 0

    num_syllables = 0

    for word in words:
        num_syllables += count_syllables_cmu(word)

    if num_syllables == 0:
        raise ValueError

    flesh_score = 206.835 - 1.015*(num_words/num_syllables) - 84.6*(num_syllables/num_words)

    return flesh_score

def grammar_score(lint_group, text: str):
    fnull = io.StringIO()
    doc = harper_py.create_english_document(text)
    with redirect_stdout(fnull):  # Captures DEBUG: prints
        lints = doc.get_lints(lint_group)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(lints) / (len(words) or 1)

import os
class SuppressAll:
    def __enter__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds: os.close(fd)

def silent_grammar_score(lint_group, text: str):
    doc = harper_py.create_english_document(text)
    with SuppressAll():
        lints = doc.get_lints(lint_group)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(lints) / (len(words) or 1)

# In your wrapper:
def compute_grammar_and_ease_wrapper(df, verbose=False):
    lint_group = harper_py.create_curated_lint_group()
    for i, row in tqdm(df.iterrows()):
        highlight_text = row["summary"]
        df.at[i, "flesch_score"] = flesch_reading_ease(highlight_text)
        df.at[i, "grammar_score"] = silent_grammar_score(lint_group, highlight_text)
        if verbose:
            print(f"ID: {row['id']}")
    return df


if __name__ == "__main__":
    # input_file = "Data/reddit_summaries.csv"
    # output_file = "Data/reddit_grammar_and_ease.csv"

    # input_file = "Data/dailymail_summaries.csv"
    # output_file = "Data/dailymail_grammar_and_ease.csv"

    input_file = "Data/cnn_dailymail_test_labeled.csv"
    output_file = "Data/cnn_dailymail_grammar_and_ease.csv"

    lint_group = harper_py.create_curated_lint_group()


    unlabeled = True
    with open(input_file, mode="r", newline="", encoding="utf-8") as infile, open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["id", "flesch_score", "grammar_score"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            print(f"attempting to read row #{i}")
            highlight_text = ""
            if unlabeled:
                highlight_text = row["summary"]
            else:
                highlight_text = row["highlights"]

            w_row = {
                "id": row["id"],
                "flesch_score": flesch_reading_ease(highlight_text),
                "grammar_score": grammar_score(lint_group, highlight_text),
            }

            writer.writerow(w_row)



