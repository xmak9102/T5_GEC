import nltk
from collections import Counter
from nltk.util import everygrams, ngrams

def sentence_gleu(references, hypothesis, min_len=1, max_len=4):
    hypothesis = nltk.word_tokenize(hypothesis)
    references = [nltk.word_tokenize(ref) for ref in references]
    
    return corpus_gleu([references], [hypothesis], min_len=min_len, max_len=max_len)


def corpus_gleu(list_of_references, hypotheses, min_len=1, max_len=4):
    
    assert len(list_of_references) == len(hypotheses)

    corpus_n_match = 0
    corpus_n_all = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(everygrams(hypothesis, min_len, max_len))
        tpfp = sum(hyp_ngrams.values())

        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(everygrams(reference, min_len, max_len))
            tpfn = sum(ref_ngrams.values())

            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())
            
            n_all = max(tpfp, tpfn)

            if n_all > 0:
                hyp_counts.append((tp, n_all))

        if hyp_counts:
            n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all

    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all

    return gleu_score
