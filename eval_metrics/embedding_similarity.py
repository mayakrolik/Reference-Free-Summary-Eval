import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load local embedding model (runs completely offline)
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embedding_similarity(s1, s2):
    emb1 = model.encode(s1)
    emb2 = model.encode(s2)
    return cosine_similarity(emb1, emb2)

def compute_embedding_similarity(df, verbose=False):
    df["embedding_similarity"] = 0.0
    for i, row in tqdm(df.iterrows()):
        original_text = row["text"]
        highlight_text = row["summary"]
        sim = embedding_similarity(original_text, highlight_text)
        if verbose:
            print(f"ID: {row['id']}, Embedding Similarity: {sim}")
        df.at[i, "embedding_similarity"] = sim
    return df

if __name__ == "__main__":

    # input_file = "Data/reddit_summaries.csv"
    # output_file = "Data/reddit_embedding_similarity.csv"

    # input_file = "Data/dailymail_summaries.csv"
    # output_file = "Data/dailymail_embedding_similarity.csv"

    input_file = "Data/cnn_dailymail_test_labeled.csv"
    output_file = "Data/cnn_dailymail_embedding_similarity.csv"

    unlabeled = True # SET THIS TO TRUE FOR THE CNN_DAILYMAIL DATASET
    with open(input_file, mode="r", newline="", encoding="utf-8") as infile, open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["id", "embedding_similarity"]
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

            w_row = {
                "id": row["id"],
                "embedding_similarity": embedding_similarity(original_text, highlight_text),
            }

            writer.writerow(w_row)
