#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import words
import nltk
import pandas as pd
import pickle

# nltk.download('words')

tf = TfidfVectorizer(stop_words='english')
fp = pd.read_csv('~/p4/data/interim/fp_posts.csv')
fp.head()

# def remove_non_words(input_string):
#     list_of_words = input_string.split()
#     for idx, val in enumerate(list_of_words):
#         if not val in words.words():
#             list_of_words.pop(idx)
#     return ' '.join(list_of_words)
# remove_non_words('This is asdfj a 439ufr srtio string of 334jherjfgh aaaahhh word'.lower())
stemmer = LancasterStemmer()

fp['title'] = fp.title.str.replace(r'\d+', '')
fp['title'] = fp.title.str.replace(r'\[.*\]', '')
fp['title'] = fp.title.str.replace(r'[^A-Za-z\s]', '')
fp['title'] = fp.apply(lambda row: stemmer.stem(row['title']), axis=1)
# fp['title'] = fp.apply(lambda row: remove_non_words(row['title']), axis=1)
print(fp['title'].head())
fp.shape

# fp.to_csv('~/p4/data/interim/fp_posts.csv')
# exit()

wv = tf.fit_transform(fp['title'])

# cnt = 0
# for val in tf.get_feature_names():
#     if val not in words.words()

del fp

tf.get_feature_names()[:20]

print('Entering LSA step...')
lsa = TruncatedSVD(n_components=50)
word_vec_reduced = lsa.fit_transform(wv.toarray())

print('Pickling TfidfVectorizer...')
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tf, f)

print('Pickling word_vec_reduced...')
with open('word_vec_reduced.pkl', 'wb') as f:
    pickle.dump(word_vec_reduced, f)

print('Pickling lsa object...')
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
        notes.write('Topic {}\n:'.format(idx))
        for i, term in enumerate(sorted_terms):
            notes.write('{} '.format(term[0]))
        notes.write('\n')
