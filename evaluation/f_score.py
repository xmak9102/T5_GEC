from collections import Counter
from nltk.util import everygrams, ngrams
import nltk



def f_beta_score(grt, pred, beta=1):
    precision, recall = pr_score(grt, pred)
    
#     print("f1 score is will be processed as default if no beta provided ")
    f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    return f_score


def pr_score(grt, pred):
    pred = nltk.word_tokenize(pred)
    grt = [nltk.word_tokenize(ref) for ref in grt]
    
    return pre([grt], [pred], 1, 4)


def pre(grt, pred, start=1, end=4):
    assert len(grt) == len(pred)

    for x, y in zip(grt, pred):
        prn = Counter(everygrams(y, start, end))
        tpfp = sum(prn.values()) # true positives + false positives
        
        for i in x:
            refn = Counter(everygrams(i, start, end))
            tpfn = sum(refn.values())  # True positives + False negatives.
            
            overlap = refn & prn
            tp = sum(overlap.values())  # True positives.
                        
    precision = tp / tpfp
    recall = tp / tpfn
    
    return precision, recall

