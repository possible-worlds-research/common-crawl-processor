"""Common Crawl hashing - creating hashes for documents in .wet files

Usage:
  hash.py --file=<filename>
  hash.py (-h | --help)
  hash.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --file=<filename>        Name of file with .wet file paths

"""

# Reusing some code from http://ethen8181.github.io/machine-learning/recsys/content_based/lsh_text.html

import os
import re
import gzip
import shutil
import numpy as np
from docopt import docopt
import sentencepiece as spm
from itertools import combinations
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


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)


def lsh(X, n_vectors, seed):    
    if seed is not None:
        np.random.seed(seed)

    dim = X.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)  

    # partition data points into bins,
    # and encode bin index bits into integers
    # https://wiki.python.org/moin/BitwiseOperators
     # x << y is the same as multiplying x by 2 ** y
    bin_indices_bits = X.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)
    
    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model


def retain_keywords(vec,reverse_vocab):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-20:]:
        keywords.append(reverse_vocab[i])
    return vec, keywords
    

def process_batch(M,urls,keywords,out_file):
    n_vectors = 16
    M = np.array(M)
    f = open(out_file,'w')

    model = lsh(M, n_vectors, seed=143)
    for i in range(len(urls)):
        s = str(model['bin_indices'][i])+' '+urls[i]+' '+' '.join([k for k in keywords[i]])
        f.write(s+'\n')
    f.close()


if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Hashing 0.1')

 
vocab, reverse_vocab, logprobs = read_vocab()
vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

# encode: text => id
n_doc = 0
doc = ""

M = []
urls = []
keywords = []
in_file = args["--file"]
out_file = in_file.replace('.gz','.hs')

with gzip.open(in_file,'r') as f:
    for l in f:        
        l = l.decode("utf-8").rstrip('\n')
        if l[:4] == "<doc":
            m = re.search(".*url=([^ ]*) ",l)
            urls.append(m.group(1))
        elif l[:5] == "</doc":
            X = vectorizer.fit_transform([doc])
            hs = X.toarray()[0]
            vec = logprobs * hs
            vec, ks = retain_keywords(vec,reverse_vocab)
            keywords.append(ks)
            M.append(vec)

            doc = ""
            n_doc+=1
            if n_doc % 1000 == 0:
                process_batch(M,urls,keywords,out_file)
                M.clear()
                urls.clear()
                keywords.clear()
                print("Processed",n_doc,"hashes...")
        else:
            ll = sp.encode_as_pieces(l)
            doc+=' '.join(wp for wp in ll)+' '
    process_batch(M,urls,keywords,out_file)

    with open(out_file, 'rb') as f_in:
        with gzip.open(out_file+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)   

    os.unlink(out_file) 
