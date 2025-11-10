import pandas as pd
from subjective_extracts import extract_subjective
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

test = pd.read_csv("../cnn_dailymail/test.csv")
train = pd.read_csv("../cnn_dailymail/train.csv")
validation = pd.read_csv("../cnn_dailymail/validation.csv")

# Extract subjective values for test, train, and validation sets
def extract_row(row):
    text = row['article']
    summary = row['highlights']
    subjective_scores = extract_subjective(text, summary)
    data = {'text': text, 'summary': summary}
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

test_subjective = extract_for_dataset(test)
test_subjective.to_csv("labeled_data/cnn_dailymail_test_labeled.csv", index=False)
train_subjective = extract_for_dataset(train[:5000])
train_subjective.to_csv("labeled_data/cnn_dailymail_train_labeled.csv", index=False)