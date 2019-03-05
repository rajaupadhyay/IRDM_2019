# from BTrees.OOBTree import OOSet
# from BTrees.OOBTree import OOBTree
import sys
from sqlitedict import SqliteDict
import pandas as pd
import zlib, pickle, sqlite3
import json
from collections import Counter
from nltk import word_tokenize
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import os
import pickle
import redis
import ast
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords
import time
import numpy as np


high_doc_freq_words = pd.read_csv('high_freq_words_stash.csv', sep=',')
high_doc_freq_words = list(high_doc_freq_words['word'].values)

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())
# print(conn.hget('doc_name_index', 'Fox_Broadcasting_Company'))

print(conn.dbsize())

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))


# with SqliteDict('high_frequency_stash.sqlite', decode=decompress_set) as mydict:
#     res = mydict['ividx_born']
#     r3 = mydict['ividx_leonidovitch']
#     rr = res.union(r3)
#
#     print(len(res), len(r3), len(rr))






print('Loading the idf vector')
idf_vec_f = open('idf_vector.pickle', 'rb')
idf_vec = pickle.load(idf_vec_f)
idf_vec_f.close()
print('IDF Vector loaded')


print('Loading the tfidf matrix')
tfidf_comp_f = open('tf_idf_matrix_compressed.pickle', 'rb')
tfidf_comp = pickle.load(tfidf_comp_f)
print('Converting to CSR Format')
tfidf_comp = tfidf_comp.tocsr()
tfidf_comp_f.close()
print('TFIDF Matrix loaded')

def retrieve_top_five_docs(tokens):
    print('Retrieving Top 5 Documents')
    start_time = time.time()
    term_freq = Counter(tokens)
    total_length = len(tokens)
    term_freq_vector = np.array([term_freq[tkn]/total_length for tkn in tokens])

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]
    idf_vector = np.array([idf_vec[tkn_idx] for tkn_idx in tkn_indices])

    tfidf = term_freq_vector * idf_vector

    set_of_documents_to_search = set()
    hf_words = []
    with SqliteDict('high_frequency_stash.sqlite', decode=decompress_set) as mydict:

        for tkn in tokens:
            if tkn in high_doc_freq_words:
                hf_words.append(tkn)
            else:
                doc_ids = mydict['ividx_{}'.format(tkn)]
                set_of_documents_to_search = set_of_documents_to_search.union(doc_ids)

        for hf_tkn in hf_words:
            doc_ids = mydict['ividx_{}'.format(hf_tkn)]
            set_of_documents_to_search = set_of_documents_to_search.intersection(doc_ids)



    rows = list(set_of_documents_to_search)
    cols = tkn_indices
    print('total documents', len(rows))
    sub_matrix = tfidf_comp[rows, :][:, cols]

    dot_prod = sub_matrix.dot(tfidf)

    print(len(dot_prod))
    top_5_document_indices = dot_prod.argsort()[-5:][::-1]
    # The actual indices of the top 5 documents need to be retrieved from the rows list
    top_5_document_indices = [rows[in_idx] for in_idx in top_5_document_indices]
    end_time = time.time()
    total_time = end_time - start_time
    print('Total Time', total_time)

    return top_5_document_indices




translator = str.maketrans('', '', string.punctuation)

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['-lrb-', '-rrb-'])


sample_query = 'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.'

porter_stemmer = PorterStemmer()

tokens = word_tokenize(sample_query)

# Lowercase
tokens = list(map(lambda x: x.lower(), tokens))
# Remove stopwords
tokens = list(filter(lambda l_ph: l_ph not in stop_words, tokens))
# Remove punctuation and stem
tokens = [porter_stemmer.stem((val.translate(translator))) for val in tokens]
# tokens = list(map(lambda val: PorterStemmer().stem(val), tokens))

# tokens = set(tokens)
if '' in tokens:
    tokens.remove('')

tokens = list(tokens)
print(tokens)

top_five = retrieve_top_five_docs(tokens)
print(top_five)
