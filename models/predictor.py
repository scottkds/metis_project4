from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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
    return preprocess_string(string, filters=[strip_tags, strip_punctuation,
                                    strip_numeric, strip_multiple_whitespaces,
                                    strip_short])

def tokenize(string):
    """Cleans and tokenizes a string and returns a list of strings that are in
    VOCABULARY"""
    words_in_string = clean_str(string)
    return [word for word in words_in_string if word in VOCABULARY]

def top_topic(vectorizer, model, new_document):
    """Returns the topic most associated with the input string"""
    word_vec = tf.transform([new_document])
    topics = nmf.transform(word_vec)
    return np.argmax(topics)

# def top_n_similar(sample_str, n, topic_vector, model, vectorizer):
#     """Returns the top n similar posts to the subreddit forum
#     Inputs:
#         n -> number of posts to return
#         topic_vector -> the topic vector to compare cosine cosine_similarity
#         sample_str -> the input string
#         vectorizer -> the vectorizer used to vectorize the input string
#     Output:
#         numpy array of cosine similarities"""
#     Y_tf = vectorizer.transform([sample_str])
#     Y_nmf = nmf.transform(Y_tf)
#     print(Y_nmf.shape)
#     sims = cosine_similarity(topic_vector, Y_nmf)
#     sort_order = np.argsort(sims.reshape(sims.shape[0]))[::-1]
#     sims_and_idx = zip(sims[sort_order], sort_order)
#     # return cosine_similarity(X.components_, Y_tf)
#     return list(sims_and_idx)[:n]

def top_n_similar(sample_str, n, topic_vector, model, vectorizer):
    """Returns the top n similar posts to the subreddit forum
    Inputs:
        n -> number of posts to return
        topic_vector -> the topic vector to compare cosine cosine_similarity
        sample_str -> the input string
        vectorizer -> the vectorizer used to vectorize the input string
    Output:
        numpy array of cosine similarities"""
    Y_tf = vectorizer.transform([sample_str])
    Y_nmf = nmf.transform(Y_tf)
    print(Y_nmf.shape)
    sims = cosine_similarity(topic_vector, Y_nmf)
    sort_order = np.argsort(sims.reshape(sims.shape[0]))[::-1]
    sims_and_idx = zip(sims[sort_order], sort_order)
    # return cosine_similarity(X.components_, Y_tf)
    # return list(sims_and_idx)[:n]
    return [x for x in sort_order[:n]]

def load_pickles():
    """Loads pickled objects from previous text analisys."""

    with open('/Users/scott/p4/pickles/vocabulary.pkl', 'rb') as f:
        VOCABULARY = pickle.load(f)

    with open('/Users/scott/p4/pickles/vectorizer_tfidf_NMF_20.pkl', 'rb') as f:
        tf = pickle.load(f)

    with open('/Users/scott/p4/pickles/doc_topic_tfidf_NMF_20.pkl', 'rb') as f:
        word_vec_reduced = pickle.load(f)

    with open('/Users/scott/p4/pickles/model_tfidf_NMF_20.pkl', 'rb') as f:
        nmf = pickle.load(f)

    with open('/Users/scott/p4/pickles/km.pkl', 'rb') as f:
        km = pickle.load(f)

    return (VOCABULARY, tf, word_vec_reduced, nmf, km)

if __name__ == '__main__':

    VOCABULARY, tf, word_vec_reduced, nmf, km = load_pickles()

    test_strings = ['bbq brisket and coleslaw',
                    'sausage eggs and hashbrowns',
                    'Check out this christmas spread',
                    'A nice variety for my son\'s birthday',
                    'What would good with this steak?']

    print(top_topic(tf, nmf, 'bbq brisket and coleslaw'))
    print(top_topic(tf, nmf, 'sausage eggs and hashbrowns'))
    print(top_topic(tf, nmf, 'Check out this christmas spread'))
    print(top_topic(tf, nmf, 'A nice variety for my son\'s birthday'))
    print(top_topic(tf, nmf, 'What would good with this steak?'))
    print(top_topic(tf, nmf, ''))

    docs = pd.read_csv('/Users/scott/p4/data/interim/fp_posts.csv')

    top_n_similar(test_strings[1], 5, word_vec_reduced, nmf, tf)

docs.loc[18180]
