"""Common Crawl hashing - creating clusters from document hashes

Usage:
  cluster.py --dir=<dirname> --nclusters=<n>
  cluster.py (-h | --help)
  cluster.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --dir=<dirname>                 Name of directory with .hs file paths
  --nclusters=<n>                 Number of clusters for kmeans

"""


import pickle
import numpy as np
from docopt import docopt
import tldextract
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from os import listdir
from os.path import join

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
    args = docopt(__doc__, version='Common Crawl Clustering 0.1')

    hsdir = args['--dir']
    num_clusters = int(args["--nclusters"])
    vocab, reverse_vocab = read_vocab()
    projections = read_projections()
    log_file = "./kmeans.log"

    url_proj_words = {}
    labels = {}
    clusters = {}
    keywords = {}
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=143, batch_size=1024, verbose=True)

    hsfiles = [join(hsdir,f) for f in listdir(hsdir) if join(hsdir, f)[-3:] == ".hs"]

    for hash_file in hsfiles:
        M = pickle.load(open(hash_file,'rb'))
        kmeans = kmeans.partial_fit(M)
        print(hash_file)

    for hash_file in hsfiles:
        url_file = hash_file.replace("hs","urls")
        M = pickle.load(open(hash_file,'rb'))
        urls = list(pickle.load(open(url_file,'rb')))
        keyword_file = hash_file.replace("hs","kwords")
        keywords.update(pickle.load(open(keyword_file,'rb')))

        predictions = kmeans.predict(M)
        print("Predicting",hash_file)

        for i in range(len(urls)):
            label = predictions[i]
            labels[urls[i]] = label
    
            if label in clusters:
                clusters[label].append(urls[i])
            else:
                clusters[label] = [urls[i]]

    log = open(log_file,'w')
    for i in range(num_clusters):
        ks = []
        domains = []
        for url in clusters[i]:
            ks.extend(keywords[url])
            ext = tldextract.extract(url)
            domains.append('.'.join(ext[1:]))
        counter = dict(Counter(ks).most_common(10))
        counter_str = ' '.join([str(k)+'('+str(v)+')' for k,v in counter.items()])

        counter_urls = dict(Counter(domains).most_common(10))
        counter_urls_str = ' '.join([str(k)+'('+str(v)+')' for k,v in counter_urls.items()])
        log.write(str(i)+' '+str(len(clusters[i]))+' '+counter_str+' '+counter_urls_str+'\n')
    log.close()

    pickle.dump(kmeans, open("kmeans.pkl", "wb"))
