from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import pandas as pd
import pickle

with open('tfidf.pkl', 'rb') as f:
    tf = pickle.load(f)

with open('word_vec_reduced.pkl', 'rb') as f:
    word_vec_reduced = pickle.load(f)

with open('lsa.pkl', 'rb') as f:
    lsa = pickle.load(f)

def top_n_similar(n, X, sample_str, vectorizer):
    Y_tf = vectorizer.transform([sample_str])
    sims = cosine_similarity(X.components_, Y_tf)
    sort_order = np.argsort(sims.reshape(sims.shape[0]))[::-1]
    sims_and_idx = zip(sims[sort_order], sort_order)
    # return cosine_similarity(X.components_, Y_tf)
    return list(sims_and_idx)[:n]


def make_result_frame(sims_and_idx, model):
    return pd.DataFrame([np.concatenate((x[0], lsa.components_[x[1]]), axis=0) for x in sims],
                 columns=['cosine_similarity'] + tf.get_feature_names())

def make_ordered_result_frame(sims_and_idx, model):
    return pd.DataFrame([np.concatenate((x[0], lsa.components_[x[1]]), axis=0) for x in sims],
                 columns=['cosine_similarity'] + tf.get_feature_names())

sims = top_n_similar(3, lsa, 'baked chicken with asparagus spears', tf)
sims

make_result_frame(sims, lsa)


terms = tf.get_feature_names()
with open('notes.txt', 'w') as notes:
    for idx, comp in enumerate(lsa.components_):
        terms_in_components = zip(terms, comp)
        sorted_terms = sorted(terms_in_components, key=lambda x: x[1], reverse=True)
        notes.write('Topic {}\n:'.format(idx))
        for i, term in enumerate(sorted_terms[:10]):
            notes.write('{} '.format(term[0]))
        notes.write('\n')


np.array([4,2,3,1]).argsort()[::-1]


p = {'a': 1, (1,2): [[1,2,3], [4,5,6]]}
pd.DataFrame(p)
lsa.components_.shape
