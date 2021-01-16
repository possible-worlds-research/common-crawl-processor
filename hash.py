"""Common Crawl hashing - creating hashes for documents in .wet files

Usage:
  hash.py --file=<filename> [--mkprojections]
  hash.py (-h | --help)
  hash.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --file=<filename>               Name of file with .wet file paths
  --mkprojections                 Make projections if they don't exist

"""

import os
import re
import gzip
import random
import pickle
import numpy as np
from docopt import docopt
import sentencepiece as spm
from itertools import combinations
from scipy.sparse import coo_matrix
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer


# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('spmcc.model')


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


def create_projections(in_file, PN_size, KC_size, proj_size):
    print("Creating",KC_size,"projections...")
    projection_functions = {}
    doc = ""
    c=0

    potential_projections = []
    with gzip.open(in_file,'r') as f:
        for l in f:
            l = l.decode("utf-8").rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*url=([^ ]*) ",l)
                print(c,m.group(1))
            elif l[:5] == "</doc":
                X = vectorizer.fit_transform([doc])
                X = X.toarray()[0]
                vec = logprobs * X
                keywords = return_keywords(vec)
                keywords = [w for w in keywords if len(reverse_vocab[w]) > 3]
                print([reverse_vocab[w] for w in keywords])
                potential_projections.extend(combinations(keywords, proj_size))
                doc=""
                c+=1
            else:
                ll = sp.encode_as_pieces(l)
                doc+=' '.join(wp for wp in ll)+' '

    potential_projections = list(set(potential_projections))
    f=open("spmcc.projs",'w')
    for cell in range(KC_size):
        p = random.choice(potential_projections)
        projection_functions[cell] = np.array(random.choice(potential_projections))
        f.write(str(p[0])+' '+str(p[1])+' '+str(p[2])+'\n')
    f.close()
    return projection_functions


def show_projections(hashed_kenyon,reverse_vocab):
    important_words = {}
    for i in range(len(hashed_kenyon)):
        if hashed_kenyon[i] == 1:
            activated_pns = projection_functions[i]
            print([reverse_vocab[pn] for pn in activated_pns])
            for pn in activated_pns:
                w = reverse_vocab[pn]
                if w in important_words:
                    important_words[w]+=1
                else:
                    important_words[w]=1
    print("BEST PNS", sorted(important_words, key=important_words.get, reverse=True)[:proj_size])


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
        kenyon_activations[cell] = 1
    return kenyon_activations

def hash_input(vec,reverse_vocab):
    kenyon_layer = projection(vec)
    hashed_kenyon = wta(kenyon_layer)
    #show_projections(hashed_kenyon,reverse_vocab)
    return hashed_kenyon
 
def return_keywords(vec):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-10:]:
        keywords.append(i)
    return keywords


if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Hashing 0.1')

    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    # Setting up the fly
    PN_size = len(vocab)
    KC_size = len(vocab) * 2
    proj_size = 3
    percent_hash = 0.1
    #print("SIZES PN LAYER:",PN_size,"KC LAYER:",KC_size)
    #print("SIZE OF PROJECTIONS:",proj_size)
    #print("SIZE OF FINAL HASH:",percent_hash,"%")

    projection_layer = np.zeros(PN_size)
    kenyon_layer = np.zeros(KC_size)

    #Reading through documents
    n_doc = 0
    doc = ""

    M_data = []
    M_col = []
    M_row = []
    urls = []
    keywords = {}

    in_file = args["--file"]
    hs_file = in_file.replace('.gz','.hs')
    url_file = in_file.replace('.gz','.urls')
    keyword_file = in_file.replace('.gz','.kwords')

    if args["--mkprojections"]:
        projection_functions = create_projections(in_file,PN_size, KC_size, proj_size)
    else:
        projection_functions = read_projections()

    with gzip.open(in_file,'r') as f:
        for l in f:        
            l = l.decode("utf-8").rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*url=([^ ]*) ",l)
                urls.append(m.group(1))
            elif l[:5] == "</doc":
                X = vectorizer.fit_transform([doc])
                X = X.toarray()[0]
                vec = logprobs * X
                hs = hash_input(vec,reverse_vocab)
                hs = coo_matrix(hs)
                #print(urls[-1],' '.join([str(i) for i in hs.col]))
                keywords[urls[-1]] = [reverse_vocab[w] for w in return_keywords(vec)]
                for i in hs.col:
                    M_row.append(n_doc)
                    M_col.append(i)
                    M_data.append(1)
                doc = ""
                n_doc+=1
            else:
                ll = sp.encode_as_pieces(l)
                doc+=' '.join(wp for wp in ll)+' '
    M = coo_matrix((M_data, (M_row, M_col)), shape=(n_doc, KC_size))

with open(hs_file,"wb") as hsf:
    pickle.dump(M,hsf)
with open(url_file,"wb") as urlf:
    pickle.dump(urls,urlf)
with open(keyword_file,"wb") as kf:
    pickle.dump(keywords,kf)
