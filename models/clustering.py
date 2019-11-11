from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def find_best_num_clusters(matrix, num_clusters):
    clusters_errors = []
    for n in range(1, num_clusters):
        km = KMeans(n_clusters=n)
        km.fit(matrix)
        clusters_errors.append((n, km.inertia_))
    return clusters_errors

with open('tfidf.pkl', 'rb') as f:
    tf = pickle.load(f)

with open('word_vec_reduced.pkl', 'rb') as f:
    word_vec_reduced = pickle.load(f)

with open('lsa.pkl', 'rb') as f:
    lsa = pickle.load(f)

vec = lsa.components_

errors = find_best_num_clusters(vec, 100)

e1 = [x[1] for x in errors]


plt.title('No elbow here!');
plt.xlabel('Number of Clusters');
plt.ylabel('Interia');
plt.plot(range(1,100), e1);
plt.savefig('images/no_elbow_here.png')
