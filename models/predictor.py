from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

from gensim.parsing.preprocessing import preprocess_string, strip_tags
from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text, strip_numeric, strip_short
from gensim.corpora import Dictionary

import numpy as np
import pandas as pd
import pickle

def clean_str(string):
    """Cleans and tokenizes a string with some functions from gensim.

    Input: A string
    Output: A lists of tokenized words"""
    rtn = preprocess_string(string, filters=[strip_tags, strip_punctuation,
                                    strip_numeric, strip_multiple_whitespaces,
                                    strip_short])
    # print('Clean_str:', rtn)
    return rtn

def tokenize(string):
    """Cleans and tokenizes a string and returns a list of strings that are in
    VOCABULARY"""
    words_in_string = clean_str(string)
    rtn = [word for word in words_in_string if word in VOCABULARY]
    # return [word for word in words_in_string if word in VOCABULARY]
    # print('tokenize:', rtn)
    return rtn

def top_n_similar(n, X, sample_str, vectorizer):
    Y_tf = vectorizer.transform([sample_str])
    print(Y_tf.shape)
    sims = cosine_similarity(X.components_, Y_tf)
    sort_order = np.argsort(sims.reshape(sims.shape[0]))[::-1]
    sims_and_idx = zip(sims[sort_order], sort_order)
    # return cosine_similarity(X.components_, Y_tf)
    return list(sims_and_idx)[:n]


def make_result_frame(sims_and_idx, model):
    return pd.DataFrame([np.concatenate((x[0], model.components_[x[1]]), axis=0) for x in sims],
                 columns=['cosine_similarity'] + tf.get_feature_names())

def make_ordered_result_frame(sims_and_idx, model):
    return pd.DataFrame([np.concatenate((x[0], model.components_[x[1]]), axis=0) for x in sims],
                 columns=['cosine_similarity'] + tf.get_feature_names())

with open('/Users/scott/p4/pickles/vocabulary.pkl', 'rb') as f:
    VOCABULARY = pickle.load(f)

with open('/Users/scott/p4/pickles/vectorizer_tfidf_NMF_20.pkl', 'rb') as f:
    tf = pickle.load(f)

with open('/Users/scott/p4/pickles/doc_topic_tfidf_NMF_20.pkl', 'rb') as f:
    word_vec_reduced = pickle.load(f)
word_vec_reduced.shape

with open('/Users/scott/p4/pickles/model_tfidf_NMF_20.pkl', 'rb') as f:
    nmf = pickle.load(f)


word_vec_reduced.shape

sims = top_n_similar(3, nmf, 'baked chicken with asparagus spears', tf)
sims

make_result_frame(sims, nmf)


terms = tf.get_feature_names()
with open('notes.txt', 'w') as notes:
    for idx, comp in enumerate(lsa.components_):
        terms_in_components = zip(terms, comp)
        sorted_terms = sorted(terms_in_components, key=lambda x: x[1], reverse=True)
        notes.write('Topic {}\n:'.format(idx))
        for i, term in enumerate(sorted_terms[:10]):
            notes.write('{} '.format(term[0]))
        notes.write('\n')


preprocess_string('chicken with rice and dumplings', filters=[strip_tags, strip_punctuation,
                                strip_numeric, strip_multiple_whitespaces,
                                strip_short])


p = {'a': 1, (1,2): [[1,2,3], [4,5,6]]}
pd.DataFrame(p)
nmf.components_.shape

step1 = tf.transform(['Baked chicken with broccoli'])
step1.shape
step2 = nmf.transform(step1)
step2.shape
step2
word_vec_reduced.shape
