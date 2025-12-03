import requests
from bert_score import score
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


# --------------------------
# 1. BERTSCORE
# --------------------------
def compute_bertscore(src, hyp, model="xlm-roberta-large"):
    P, R, F1 = score([hyp], [src], lang="en", model_type=model)
    return F1.item()


# --------------------------
# 2. Multilingual NER
# --------------------------
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",     # replace with multilingual model
    aggregation_strategy="simple"
)

def extract_entities(text):
    entities = ner(text)
    return [e["word"] for e in entities if e["entity_group"] in ["PER","ORG","LOC"]]


# --------------------------
# 3. Wikidata entity linking (simplified)
# --------------------------
def wikidata_search(entity_name):
    """Return Wikidata QID for an entity string."""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_name
    }
    response = requests.get(url, params=params).json()
    if "search" in response and len(response["search"]) > 0:
        return response["search"][0]["id"]
    return None


def link_entities(entity_list):
    linked = set()
    for ent in entity_list:
        qid = wikidata_search(ent)
        if qid:
            linked.add(qid)
    return linked


# --------------------------
# 4. KG Entity Match Score
# --------------------------
def compute_kg_score(src_text, hyp_text):
    src_ents = extract_entities(src_text)
    hyp_ents = extract_entities(hyp_text)

    src_links = link_entities(src_ents)
    hyp_links = link_entities(hyp_ents)

    if len(src_links) == 0:
        return 0.0

    intersection = len(src_links.intersection(hyp_links))
    return intersection / len(src_links)


# --------------------------
# 5. Combine into KG-BERTScore
# --------------------------
def kg_bertscore(src, hyp, lam=0.7):
    bert = compute_bertscore(src, hyp)
    kg = compute_kg_score(src, hyp)
    return lam * bert + (1 - lam) * kg


# --------------------------
# Example usage
# --------------------------
src = "Angela Merkel met with the president of France in Berlin."
hyp = "The German chancellor met the French president in the capital."

print("BERTScore:", compute_bertscore(src, hyp))
print("KG Score:", compute_kg_score(src, hyp))
print("KG-BERTScore:", kg_bertscore(src, hyp, lam=0.7))

if __name__ == "__main__":
    pass