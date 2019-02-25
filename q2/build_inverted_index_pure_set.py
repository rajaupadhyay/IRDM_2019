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


# ['vocab_set', 'doc_name_index', 'vocab_dictionary']
conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())
# print(conn.hget('vocab_dictionary', 'hp18c'))

print(conn.dbsize())


def delete_keys():
    print('DELETING KEYS')
    # time.sleep(10)
    for k_val in conn.scan_iter(match='ividx_*'):
        conn.delete(k_val)

# delete_keys()

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

hf_stash = SqliteDict('high_frequency_stash.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)

def add_key_to_high_frequency_stash(key, value_set):
    hf_stash[key] = value_set
    # print('successfully added to hf_stash')


############################## INVERTED INDEX LOGIC ############################

# Load the vocabulary index
vocab_pickled = open('vocab_index.pickle', 'rb')
vocab_dict_pkl = pickle.load(vocab_pickled)
vocab_pickled.close()


print('Loading the document term matrix')
dtm_mat = open('doc_term_matrix.pickle', 'rb')
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

# Iterate over the vocabulary (building the index for each word)
cnt = 0
print('Beginning indexing')

# pipe = conn.pipeline()

for vocab_word in vocab_dict_pkl:
    if cnt < 4:
        print(vocab_word)
    cnt += 1

    vocab_word_idx = conn.hget('vocab_dictionary', vocab_word)
    idx_for_word = int(vocab_word_idx)
    set_key_name = 'ividx_{}'.format(vocab_word)


    # if cnt%10 == 0:
    #     try:
    #         pipe.execute()
    #         pipe = conn.pipeline()
    #     except Exception as e:
    #         print(cnt)
    #         print(e)
    #         print(vocab_word)
    #         break

    if cnt % 100000 == 0:
        print('index', cnt)

    transposed_row_from_matrix = test_dtm_transposed_csr[idx_for_word]
    rows, cols = transposed_row_from_matrix.nonzero()
    cols = set(map(int, cols))
    # dta = transposed_row_from_matrix.data
    # dta = list(map(int, dta))
    # inner_values_for_set = dict(list(zip(cols, dta)))

    # dlen_val = len(cols)

    # if dlen_val > 100000:
    #     add_key_to_high_frequency_stash(set_key_name, cols)
    # else:
    #     pipe.sadd(set_key_name, *cols)

    try:
        add_key_to_high_frequency_stash(set_key_name, cols)
    except Exception as e:
        print(e)
        print('current count', cnt)
        print('current word', vocab_word)
        break


hf_stash.close()
