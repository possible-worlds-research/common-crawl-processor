"""Common Crawl hashing - Querying clusters

Usage:
  query.py
  query.py (-h | --help)
  query.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.

"""

import os
import pickle
import numpy as np
from docopt import docopt
import sentencepiece as spm
from scipy.sparse import coo_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer


# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('spmcc.model')

def read_kmeans_log():
    clusters = {}
    with open("kmeans.log") as f:
        for l in f:
            l = l.rstrip('\n')
            fields = l.split()
            clusters[int(fields[0])] = l
    return clusters

def read_vocab():
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open("spmcc.vocab") as f:
        for l in f:
            l = l.rstrip('\n')
            wp = l.split('\t')[0]
            logprob = -(float(l.split('\t')[1]))
            #logprob = log(lp + 1.1)
            if wp in vocab or wp == '':
                continue
            vocab[wp] = c
            reverse_vocab[c] = wp
            logprobs.append(logprob**3)
            c+=1
    return vocab, reverse_vocab, logprobs


def read_projections():
    c = 0
    projections = {}
    with open("spmcc.projs") as f:
        for l in f:
            l=l.rstrip('\n')
            p = np.array([int(n) for n in l.split()])
            projections[c]=p
            c+=1
    return projections


def projection(projection_layer):
    kenyon_layer = np.zeros(KC_size)
    for cell in range(KC_size):
        activated_pns = projection_functions[cell]
        for pn in activated_pns:
            kenyon_layer[cell]+=projection_layer[pn]
    return kenyon_layer

def wta(kenyon_layer):
    #print(kenyon_layer[:100])
    kenyon_activations = np.zeros(KC_size)
    top = int(percent_hash * KC_size / 100)
    activated_kcs = np.argpartition(kenyon_layer, -top)[-top:]
    for cell in activated_kcs:
        if kenyon_layer[cell] != 0:
            kenyon_activations[cell] = 1
    return kenyon_activations

def hash_input(vec,reverse_vocab):
    kenyon_layer = projection(vec)
    hashed_kenyon = wta(kenyon_layer)
    return hashed_kenyon
 
def return_keywords(vec):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-10:]:
        if vec[i] != 0:
            keywords.append(i)
    return keywords


if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Querying 0.1')

    clusters = read_kmeans_log()
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    kmeans = pickle.load(open("kmeans.pkl", "rb"))

    # Setting up the fly
    PN_size = len(vocab)
    KC_size = len(vocab) * 2
    proj_size = 3
    percent_hash = 0.1

    projection_layer = np.zeros(PN_size)
    kenyon_layer = np.zeros(KC_size)
    projection_functions = read_projections()

    query = input("Query: ")
    while query != 'q':
        qe = sp.encode_as_pieces(query)
        doc=' '.join(wp for wp in qe)

        X = vectorizer.fit_transform([doc])
        X = X.toarray()[0]
        vec = logprobs * X
        hs = hash_input(vec,reverse_vocab)
        hs = coo_matrix(hs)
        print([reverse_vocab[w] for w in return_keywords(vec)])
        cluster = kmeans.predict(hs)[0]
        print(clusters[cluster])
        query = input("\nQuery: ")
