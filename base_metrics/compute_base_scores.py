import pandas as pd
from bert_scores import compute_bert_score
from rouge_scores import compute_rouge_scores
from tqdm import tqdm

cnn_test_df = pd.read_csv("../data/cnn_dailymail/test.csv")
reddit_df = pd.read_csv("../data/labeled_data/reddit_summaries.csv")
dailymail_df = pd.read_csv("../data/labeled_data/dailymail_summaries.csv")

def compute_base_scores(candidate, reference):
    bert_scores = compute_bert_score(candidate, reference)
    rouge_scores = compute_rouge_scores(candidate, reference)
    return {
        'BERT Precision': bert_scores[0].mean().item(),
        'BERT Recall': bert_scores[1].mean().item(),
        'BERT F1': bert_scores[2].mean().item(),
        **rouge_scores
    }

# Apply to CNN/DailyMail test set
cnn_test_scores = []
for _, row in tqdm(cnn_test_df.iterrows()):
    scores = compute_base_scores(row['highlights'], row['article'])
    scores['id'] = row['id']
    cnn_test_scores.append(scores)

cnn_test_scores_df = pd.DataFrame(cnn_test_scores)
cnn_test_scores_df.to_csv("../data/labeled_data/cnn_dailymail_test_base_scores.csv", index=False)  

# Apply to Reddit dataset
reddit_scores = []
for _, row in tqdm(reddit_df.iterrows()): 
    scores = compute_base_scores(row['highlights'], row['article'])
    scores['id'] = row['id']
    reddit_scores.append(scores)

reddit_scores_df = pd.DataFrame(reddit_scores)
reddit_scores_df.to_csv("../data/labeled_data/reddit_base_scores.csv", index=False)

# Apply to DailyMail dataset
dailymail_scores = []
for _, row in tqdm(dailymail_df.iterrows()): 
    scores = compute_base_scores(row['highlights'], row['article'])
    scores['id'] = row['id']
    dailymail_scores.append(scores)

dailymail_scores_df = pd.DataFrame(dailymail_scores)
dailymail_scores_df.to_csv("../data/labeled_data/dailymail_base_scores.csv", index=False)