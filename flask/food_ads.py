# Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from gensim.parsing.preprocessing import preprocess_string, strip_tags
from gensim.parsing.preprocessing import strip_punctuation, strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text, strip_numeric, strip_short
from gensim.corpora import Dictionary

import re
import pickle
import numpy as np
from flask import Flask, render_template, request, url_for

with open('/Users/scott/p4/pickles/vocabulary.pkl', 'rb') as f:
    VOCABULARY = pickle.load(f)

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

def top_n_similar(sample_str, n, topic_vector, model, vectorizer):
    """Returns the top n similar posts to the subreddit forum
    Inputs:
        n -> number of posts to return
        topic_vector -> the topic vector to compare cosine cosine_similarity
        sample_str -> the input string
        vectorizer -> the vectorizer used to vectorize the input string
    Output:
        list of indexes to URLs"""
    Y_tf = vectorizer.transform([sample_str])
    Y_nmf = model.transform(Y_tf)
    print(Y_nmf.shape)
    sims = cosine_similarity(topic_vector, Y_nmf)
    sort_order = np.argsort(sims.reshape(sims.shape[0]))[::-1]
    indexes = [x for x in sort_order[:n]]
    return [app.extensions['threads'][x] for x in indexes]

app = Flask(__name__) #creating the Flask class object
if not hasattr(app, 'extensions'):
    app.extensions = {}
with open('/Users/scott/p4/pickles/model_tfidf_NMF_20.pkl', 'rb') as f:
    app.extensions['nmf'] = pickle.load(f)

with open('/Users/scott/p4/pickles/vectorizer_tfidf_NMF_20.pkl', 'rb') as f:
    app.extensions['tf'] = pickle.load(f)

with open('/Users/scott/p4/pickles/idx2thread.pkl', 'rb') as f:
    app.extensions['threads'] = pickle.load(f)

with open('/Users/scott/p4/pickles/doc_topic_tfidf_NMF_20.pkl', 'rb') as f:
    app.extensions['doc_topic'] = pickle.load(f)

@app.route('/') #decorator drfines the
def index():
    return render_template('index.html')

@app.route('/query', methods = ['POST'])
def query():
    # from transform_data import transform_data

    query = request.form['query']
    word_vec = app.extensions['tf'].transform([query])
    topics = app.extensions['nmf'].transform(word_vec)
    topic_number = np.argmax(topics)
    return render_template('query.html', query=topic_number)

@app.route('/api', methods = ['GET'])
def api():
    api_query = request.args.get('api_query')
    word_vec = app.extensions['tf'].transform([api_query])
    topics = app.extensions['nmf'].transform(word_vec)
    topic_number = np.argmax(topics)
    return render_template('api.html', query=topic_number)

@app.route('/suggested_topics', methods = ['GET'])
def suggested_topics():
    suggested_topics = request.args.get('suggested_topics')
    num_topics=request.args.get('num_topics')
    try:
        num_topics = int(float(num_topics) // 1)
    except:
        num_topics = 5
    similar_posts = top_n_similar(suggested_topics, num_topics, app.extensions['doc_topic'],
                                    app.extensions['nmf'], app.extensions['tf'])
    return render_template('suggested_topics.html', threads=similar_posts, num_topics=len(similar_posts))


@app.route('/upload', methods = ['POST'])
def upload():
   return render_template('upload.html')

@app.route('/history')
def history():
   return render_template('history.html')

def about():
    return "This is the about page"

def clean_input(input):
    pass

app.add_url_rule("/about","about",about)

if __name__ =='__main__':
    app.run(debug = True)
