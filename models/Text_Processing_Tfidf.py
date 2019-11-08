#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import pickle

tf = TfidfVectorizer(stop_words='english')
fp = pd.read_csv('~/p4/data/interim/fp_posts.csv')
fp.head()

fp['title'] = fp.title.str.replace(r'\d+', '')
fp['title'] = fp.title.str.replace(r'\[.*\]', '')
print(fp['title'].head())


wv = tf.fit_transform(fp['title'])

del fp

tf.get_feature_names()[:20]

lsa = TruncatedSVD(n_components=50)
word_vec_reduced = lsa.fit_transform(wv.toarray())

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tf, f)

with open('word_vec_reduced.pkl', 'wb') as f:
    pickle.dump(word_vec_reduced, f)

with open('lsa.pkl', 'wb') as f:
    pickle.dump(lsa, f)


with open('tfidf.pkl', 'rb') as f:
    tf = pickle.load(f)

with open('word_vec_reduced.pkl', 'rb') as f:
    word_vec_reduced = pickle.load(f)

with open('lsa.pkl', 'rb') as f:
    lsa = pickle.load(f)

terms = tf.get_feature_names()


len(lsa.components_)

with open('notes.txt', 'w') as notes:
    for idx, comp in enumerate(lsa.components_):
        terms_in_components = zip(terms, comp)
        sorted_terms = sorted(terms_in_components, key=lambda x: x[1], reverse=True)
        notes.write('Concept {}\n:'.format(idx))
        for i, term in enumerate(sorted_terms):
            notes.write('\t{} :: {}\n'.format(i, term[0]))
