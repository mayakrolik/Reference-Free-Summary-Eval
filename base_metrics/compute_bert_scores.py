import bert_score
from bert_score import score

from transformers import logging
logging.set_verbosity_error()

def compute_bert_score(cands, refs):
    cands = [cands] if isinstance(cands, str) else cands
    refs = [refs] if isinstance(refs, str) else refs
    return score(cands, refs, lang='en')