import sys
import pickle
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

num_clusters = 200

def read_vocab():
    c = 0
    vocab = {}
    reverse_vocab = {}
    with open("spmcc.vocab") as f:
        for l in f:
            l = l.rstrip('\n')
            wp = l.split('\t')[0]
            if wp in vocab or wp == '':
                continue
            vocab[wp] = c
            reverse_vocab[c] = wp
            c+=1
    return vocab, reverse_vocab


def read_projections():
    c = 0
    projections = {}
    with open("spmcc.projs") as f:
        for l in f:
            l=l.rstrip('\n')
            p = [int(n) for n in l.split()]
            projections[c]=p
            c+=1
    return projections

def read_keywords():
    keywords = {}
    with open("tmp") as f:
        for l in f:
            fields=l.rstrip('\n').split()
            url = fields[0]
            words = fields[1:]
            keywords[url] = words
    return keywords


vocab, reverse_vocab = read_vocab()
projections = read_projections()
keywords = read_keywords()

M = pickle.load(open(sys.argv[1],'rb'))
urls = list(pickle.load(open(sys.argv[2],'rb')))
kmeans = KMeans(n_clusters=num_clusters, random_state=143).fit(M)

url_proj_words = {}
labels = {}
clusters = {}
for i in range(len(urls)):
    label = kmeans.labels_[i]
    labels[urls[i]] = label
    
    if label in clusters:
        clusters[label].append(urls[i])
    else:
        clusters[label] = [urls[i]]
    #rows = np.where(M.row == i)
    #ps = M.col[rows]
    #words = list(set([reverse_vocab[n] for p in ps for n in projections[p]]))
    #url_proj_words[urls[i]] = words
    #print(urls[i],words)
#print(Counter(kmeans.labels_))




for i in range(1,num_clusters):
    ks = []
    print('***')
    for url in clusters[i]:
        #keywords.extend(url_proj_words[url])
        ks.extend(keywords[url])
    print(i,len(clusters[i]),Counter(ks).most_common(10))
    print('***')
