'''
FINAL SCRIPT USED FOR INVERTED INDEX (The inverted index created
in this script INCLUDES the frequencies of the terms in the relevant
documents.)
'''

import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix
import time
import sys
from sqlitedict import SqliteDict
import pandas as pd
import zlib
import sqlite3
from tqdm import tqdm

# ['vocab_set', 'doc_name_index', 'vocab_dictionary']
conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())
# print(conn.hget('vocab_dictionary', 'hp18c'))

print(conn.dbsize())


def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

ividx_with_freq = SqliteDict('ividx_with_freq.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)


def add_key_to_ividx_with_freq(key, value_set):
    ividx_with_freq[key] = value_set


print('Loading the document term matrix')
dtm_mat = open('../doc_term_matrix.pickle', 'rb')
test_dtm = pickle.load(dtm_mat)
dtm_mat.close()
print('DOC-TERM Matrix loaded')

# Transposing DOC-TERM Matrix
print('Transposing DOC-TERM Matrix')
test_dtm_transposed = test_dtm.transpose()
test_dtm = None

# Convert to CSR
print('Converting to CSR Format')
test_dtm_transposed_csr = test_dtm_transposed.tocsr()
test_dtm_transposed = None




############################## INVERTED INDEX LOGIC ############################

# Load the vocabulary index
vocab_pickled = open('../vocab_index.pickle', 'rb')
vocab_dict_pkl = pickle.load(vocab_pickled)
vocab_pickled.close()

# Load the normalised_term_freqs
print('Loading normalised term freqs matrix')
ntf_f = open('../normalised_term_freqs.pickle', 'rb')
ntf = pickle.load(ntf_f)
ntf = ntf.transpose()
ntf = ntf.tocsr()
ntf_f.close()


# Iterate over the vocabulary (building the index for each word)
cnt = 0
print('Beginning indexing')

for vocab_word in tqdm(vocab_dict_pkl):
    # if cnt < 4:
    #     print(vocab_word)
    cnt += 1
    vocab_word_idx = conn.hget('vocab_dictionary', vocab_word)
    idx_for_word = int(vocab_word_idx)
    set_key_name = 'ividx_{}'.format(vocab_word)

    if cnt % 100000 == 0:
        print('checkpoint', cnt)
        time.sleep(5)
        print('total commited', len(ividx_with_freq))

    # if cnt % 50000 == 0:
    #     time.sleep(5)

    # set_of_relevant_docs = None
    # with SqliteDict('../ividx_no_freq.sqlite', decode=decompress_set) as ividx_no_freq_dict:
    #     set_of_relevant_docs = list(ividx_no_freq_dict[set_key_name])

    transposed_row_from_matrix = test_dtm_transposed_csr[idx_for_word]
    rows, cols = transposed_row_from_matrix.nonzero()
    set_of_relevant_docs = list(map(int, cols))


    array_with_ntf_vals = ntf[idx_for_word].data
    array_with_ntf_vals = list(array_with_ntf_vals)

    tuples_list_with_doc_and_freq = list(zip(set_of_relevant_docs, array_with_ntf_vals))


    try:
        add_key_to_ividx_with_freq(set_key_name, tuples_list_with_doc_and_freq)
    except Exception as e:
        print(e)
        print('current count', cnt)
        print('current word', vocab_word)
        break


ividx_with_freq.close()
