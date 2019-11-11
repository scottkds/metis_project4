#!/usr/bin/env python

import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import words
import nltk
import pandas as pd
import pickle

# nltk.download('words')
fp = pd.read_csv('~/p4/data/interim/fp_posts.csv')
fp.head()


stemmer = LancasterStemmer()

fp['title'] = fp.title.str.replace(r'\d+', '')
fp['title'] = fp.title.str.replace(r'\[.*\]', '')
fp['title'] = fp.title.str.replace(r'[^A-Za-z\s]', '')
fp['title'] = fp.apply(lambda row: stemmer.stem(row['title']), axis=1)
fp['title'] = fp['title'].str.split()
# fp['title'] = fp.apply(lambda row: remove_non_words(row['title']), axis=1)
print(fp['title'].head())
fp.shape

fp.title.values[:10]

# fp.to_csv('~/p4/data/interim/fp_posts.csv')
# exit()

model = gensim.models.Word2Vec(fp.title.values, min_count=1, workers=2, sg=0)
list(model.wv.vocab.items())[:10]

model.most_similar('pork butt'.lower().split())
# cnt = 0
# for val in tf.get_feature_names():
#     if val not in words.words()

del fp

tf.get_feature_names()[:20]

print('Entering LSA step...')
lsa = TruncatedSVD(n_components=100)
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
with open('notes.txt', 'w') as notes:
    for idx, comp in enumerate(lsa.components_):
        terms_in_components = zip(terms, comp)
        sorted_terms = sorted(terms_in_components, key=lambda x: x[1], reverse=True)
        notes.write('Topic {}\n:'.format(idx))
        for i, term in enumerate(sorted_terms[:10]):
            notes.write('{} '.format(term[0]))
        notes.write('\n')
