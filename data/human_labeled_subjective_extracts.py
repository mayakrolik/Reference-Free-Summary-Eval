import pandas as pd
from subjective_extracts import extract_subjective
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

dailymail_df = pd.read_csv("labeled_data/dailymail_summaries.csv")
reddit_df = pd.read_csv("labeled_data/reddit_summaries.csv")

# Extract subjective values for test, train, and validation sets
def extract_row(row):
    text = row['article']
    summary = row['highlights']
    id = row['id']
    # Add in human labeled scores
    subjective_scores = extract_subjective(text, summary)
    data = {'id': id, 'text': text, 'summary': summary}
    data.update(subjective_scores)
    return data

def extract_for_dataset(dataset, num_workers=3):
    subjective_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all extraction tasks
        futures = [executor.submit(extract_row, row) for _, row in dataset.iterrows()]
        # Collect results as they complete
        for f in tqdm(as_completed(futures), total=len(futures)):
            subjective_data.append(f.result())
    return pd.DataFrame(subjective_data)

dailymail_subjective = extract_for_dataset(dailymail_df)
dailymail_subjective.to_csv("labeled_data/dailymail_labeled.csv", index=False)

reddit_subjective = extract_for_dataset(reddit_df)
reddit_subjective.to_csv("labeled_data/reddit_labeled.csv", index=False)