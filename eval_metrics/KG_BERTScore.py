import requests
import sqlite3
import time
from pathlib import Path
from functools import lru_cache
from bert_score import score
from transformers import pipeline

# --------------------------
# CACHE SETUP
# --------------------------
CACHE_DIR = Path.home() / ".cache" / "summary_eval"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "wikidata_cache.db"

def init_cache():
    """Initialize SQLite cache for persistent Wikidata lookups."""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wikidata_cache (
            entity TEXT PRIMARY KEY,
            qid TEXT,
            last_updated INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Initialize on import
init_cache()


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
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)


def extract_entities(text):
    """Extract person, organization, and location entities from text."""
    entities = ner(text)
    return [e["word"] for e in entities if e["entity_group"] in ["PER", "ORG", "LOC"]]


# --------------------------
# 3. Wikidata entity linking with persistent cache
# --------------------------
@lru_cache(maxsize=5000)  # In-memory cache for recent lookups
def wikidata_search(entity_name):
    """Return Wikidata QID for an entity string with persistent caching."""
    # Check SQLite cache first
    conn = sqlite3.connect(CACHE_DB, timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT qid FROM wikidata_cache WHERE entity=?", (entity_name,))
    cached = cursor.fetchone()

    if cached is not None:
        qid = cached[0]
        conn.close()
        return qid

    # API call with error handling
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_name
    }

    qid = None
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "search" in data and len(data["search"]) > 0:
            qid = data["search"][0]["id"]

        # Cache result (successful or None)
        cursor.execute(
            "INSERT OR REPLACE INTO wikidata_cache (entity, qid, last_updated) VALUES (?, ?, ?)",
            (entity_name, qid, int(time.time()))
        )
        conn.commit()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for '{entity_name}': {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for '{entity_name}': {e}")
    except ValueError as e:
        print(f"JSON decode error for '{entity_name}': {e}")
    finally:
        conn.close()

    return qid


def link_entities_cached(entity_list):
    """Link entities using shared cache - no redundant API calls."""
    linked = set()
    for ent in entity_list:
        ent_clean = ent.strip()
        if not ent_clean:
            continue
        qid = wikidata_search(ent_clean)
        if qid:
            linked.add(qid)
        time.sleep(0.05)  # Respect Wikidata rate limits
    return linked


# --------------------------
# 4. KG Entity Match Score
# --------------------------
def compute_kg_score(src_text, hyp_text):
    """
    Compute KG-based entity match score between source and hypothesis.
    Returns ratio of matched entities to source entities.
    """
    src_ents = extract_entities(src_text)
    hyp_ents = extract_entities(hyp_text)

    src_links = link_entities_cached(src_ents)
    hyp_links = link_entities_cached(hyp_ents)

    overlap = len(src_links.intersection(hyp_links))
    return overlap / max(len(src_links), 1.0)


# --------------------------
# 5. Combine into KG-BERTScore
# --------------------------
def kg_bertscore(src, hyp, lam=0.7):
    """
    Hybrid metric combining BERTScore and KG entity matching.

    Args:
        src: Source/reference text
        hyp: Hypothesis/summary text
        lam: Weight for BERTScore (1-lam weights KG score)

    Returns:
        Weighted combination of BERTScore F1 and KG entity match
    """
    bert = compute_bertscore(src, hyp)
    kg = compute_kg_score(src, hyp)
    return lam * bert + (1 - lam) * kg


def compute_kg_score_wrapper(df, verbose=False):
    """
    Apply KG scoring to all rows in DataFrame.
    Uses persistent cache across all rows.

    Args:
        df: DataFrame with 'text' (original) and 'summary' columns
        verbose: Print per-row scores

    Returns:
        DataFrame with new 'kg_score' column
    """
    df["kg_score"] = 0.0
    for i, row in df.iterrows():
        original_text = row["text"]
        summary_text = row["summary"]
        kg = compute_kg_score(original_text, summary_text)
        if verbose:
            print(f"ID: {row.get('id', i)}, KG Score: {kg:.4f}")
        df.at[i, "kg_score"] = kg
    return df


def get_cache_stats():
    """Print cache statistics."""
    try:
        conn = sqlite3.connect(CACHE_DB)
        count = conn.execute("SELECT COUNT(*) FROM wikidata_cache").fetchone()[0]
        hits = conn.execute("SELECT COUNT(*) FROM wikidata_cache WHERE qid IS NOT NULL").fetchone()[0]
        conn.close()
        print(f"Cache Stats: {count} total entities, {hits} successful matches")
        return {"total": count, "matches": hits}
    except Exception as e:
        print(f"Error reading cache: {e}")
        return None


def clear_old_cache(days=30):
    """Delete cache entries older than specified days."""
    try:
        conn = sqlite3.connect(CACHE_DB)
        cutoff = int(time.time()) - (days * 86400)
        deleted = conn.execute("DELETE FROM wikidata_cache WHERE last_updated < ?", (cutoff,))
        conn.commit()
        conn.close()
        print(f"Deleted {deleted.rowcount} cache entries older than {days} days")
    except Exception as e:
        print(f"Error clearing cache: {e}")


if __name__ == "__main__":
    # Test with sample data only when run directly
    src = "Apple Inc. was founded by Steve Jobs in California."
    hyp = "Apple was started by Jobs."

    print("Testing KG_BERTScore module...")
    print(f"Source: {src}")
    print(f"Hypothesis: {hyp}")
    print(f"KG Score: {compute_kg_score(src, hyp):.4f}")
    print(f"KG-BERTScore: {kg_bertscore(src, hyp):.4f}")
    print()
    get_cache_stats()
