"""Common Crawl hashing - creating clusters from document hashes

Usage:
  clustr.py --file=<filename>
  cluster.py (-h | --help)
  cluster.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --file=<filename>               Name of file with .wet file paths

"""


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

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Hashing 0.1')

    hash_file = args['--file']
    url_file = hash_file.replace("hs","urls")
    keyword_file = hash_file.replace("hs","kwords")

    vocab, reverse_vocab = read_vocab()
    projections = read_projections()

    M = pickle.load(open(hash_file,'rb'))
    urls = list(pickle.load(open(url_file,'rb')))
    keywords = pickle.load(open(keyword_file,'rb'))

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


    for i in range(1,num_clusters):
        ks = []
        print('***')
        for url in clusters[i]:
            ks.extend(keywords[url])
        print(i,len(clusters[i]),Counter(ks).most_common(10))
        print('***')
