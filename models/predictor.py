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

# def make_result_frame(similarities, model):
#     indexes = []
#     sims = []
#     for item in similarities:
#         indexes.append(item[1])
#         sims.append(item[0][0])
#     frame = pd.DataFrame(sims, index=indexes, columns=['cosine_similarity'])
#     return frame

def make_result_frame(sims_and_idx, model):
    return pd.DataFrame([np.concatenate((x[0], lsa.components_[x[1]]), axis=0) for x in sims],
                 columns=['cosine_similarity'] + tf.get_feature_names())


sims = top_n_similar(3, lsa, 'ckicken pot pie with asparugus spears', tf)
sims

make_result_frame(sims, lsa)

lsa.components_
