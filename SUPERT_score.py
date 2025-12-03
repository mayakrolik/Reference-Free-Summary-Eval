import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download('punkt')


# --------------------------------------------------
# STEP 1: Embed sentences
# --------------------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")

def embed_sentences(sent_list):
    return model.encode(sent_list, convert_to_tensor=True)


# --------------------------------------------------
# Utility: sentence splitting
# --------------------------------------------------
def sent_split(text):
    return nltk.sent_tokenize(text)


# --------------------------------------------------
# STEP 2: Select pseudo-references (SUPERT)
# --------------------------------------------------
def get_pseudo_references(doc_sents, summary_sents):
    """
    Perform k-means on document embeddings.
    Choose 1 pseudo-reference per cluster.
    """
    k = len(summary_sents)
    if k == 0:
        return []

    doc_emb = embed_sentences(doc_sents)
    doc_emb_np = doc_emb.cpu().numpy()

    # k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(doc_emb_np)
    centroids = kmeans.cluster_centers_

    pseudo_refs = []
    for i in range(k):
        # extract all indices in cluster i
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        # pick the closest-to-centroid sentence
        cluster_vectors = doc_emb_np[cluster_indices]
        centroid = centroids[i]

        dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
        best_idx = cluster_indices[np.argmin(dists)]
        pseudo_refs.append(doc_sents[best_idx])

    return pseudo_refs


# --------------------------------------------------
# STEP 3: Compute SUPERT score
# --------------------------------------------------
def supert(document, summary):
    doc_sents = sent_split(document)
    sum_sents = sent_split(summary)

    if len(sum_sents) == 0:
        return 0.0

    # Create pseudo-references
    pseudo_refs = get_pseudo_references(doc_sents, sum_sents)

    # Embed summary + pseudo-reference sentences
    sum_emb = embed_sentences(sum_sents)
    pseudo_emb = embed_sentences(pseudo_refs)

    # Compute best similarity for each summary sentence
    sim_matrix = util.cos_sim(sum_emb, pseudo_emb)  # shape (m, k)

    best_sims = sim_matrix.max(dim=1).values
    score = best_sims.mean().item()

    return score


# --------------------------------------------------
# Example
# --------------------------------------------------
document = """
The Eiffel Tower is one of the most famous landmarks in the world.
Located in Paris, it attracts millions of visitors each year.
Originally constructed as a temporary exhibit for the 1889 World’s Fair,
it has since become a global cultural icon of France.
"""

summary = "The Eiffel Tower in Paris is visited by millions annually."

print("SUPERT score:", supert(document, summary))