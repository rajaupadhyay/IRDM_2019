'''
Retrieves top 5 documents for the claims (without MultiProcessing)
'''
import pickle
import numpy as np
import redis
from scipy.sparse import coo_matrix
import time
from collections import defaultdict
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from nltk import word_tokenize
from sqlitedict import SqliteDict
import zlib
from numpy import linalg as LA
import pandas as pd
import json
from multiprocessing import Pool
import sys
from scipy.sparse import csr_matrix, csc_matrix

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print('Loading the vocab')
# vocab_index_f = open('tf_idf_matrix_compressed.pickle', 'rb')
# vocab_index = pickle.load(vocab_index_f)
# vocab_index_f.close()



print('Loading the tfidf matrix')
tfidf_matrix_f = open('tfidf_csr.pickle', 'rb')
tfidf_matrix = pickle.load(tfidf_matrix_f)
tfidf_matrix_f.close()
print('TFIDF Matrix loaded')

print('Loading the enl vector')
enl_vec_f = open('enlv_for_documents.pickle', 'rb')
enlv = pickle.load(enl_vec_f)
enl_vec_f.close()
print('enl Vector loaded')


print('Loading the idf vector')
idf_vec_f = open('idf_vector.pickle', 'rb')
idf_vec = pickle.load(idf_vec_f)
idf_vec_f.close()
print('IDF Vector loaded')


def retrieve_top_five_docs(tokens):
    # print('Retrieving Top 5 Documents')
    start_time = time.time()
    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]
    # tokens = [tkn for tkn in tokens if tkn in vocab_index]
    term_freq = Counter(tokens)
    total_length = len(tokens)

    tokens = list(set(tokens))

    term_freq_vector = np.array([term_freq[tkn]/total_length for tkn in tokens])

    # tkn_indices = [int(vocab_index[tkn]) for tkn in tokens]
    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]

    idf_vector = np.array([idf_vec[tkn_idx] for tkn_idx in tkn_indices])

    tfidf_for_query = term_freq_vector * idf_vector
    enlv_for_query = LA.norm(tfidf_for_query)
    print('enlv', enlv_for_query, tfidf_for_query)


    # Key: Doc number/ID, value: cosine similarity
    result_dictionary = defaultdict(float)
    enlv_dictionary = {}


    rows = tkn_indices
    cols = [0]*len(tkn_indices)
    data = list(tfidf_for_query)


    tfidf_q = csc_matrix( (data,(rows,cols)), shape=(3922275,1) )

    unique_docs = set()
    with SqliteDict('ividx_no_freq.sqlite', decode=decompress_set) as ividx_dct:
        for tkn_idx in range(len(tokens)):
            tkn = tokens[tkn_idx]

            docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]
            unique_docs = unique_docs.union(docs_from_ividx)


    rows_lhs = list(unique_docs)

    start_time_mul = time.time()
    res = tfidf_matrix[rows_lhs] * tfidf_q
    end_time_mul = time.time()
    print(end_time_mul-start_time_mul)




    end_time = time.time()
    print('total_time', end_time - start_time)


translator = str.maketrans('', '', string.punctuation)

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['-lrb-', '-rrb-'])

porter_stemmer = PorterStemmer()


def tokenise_claim(claim):

    tokens = word_tokenize(claim)

    # Lowercase
    tokens = list(map(lambda x: x.lower(), tokens))
    # Remove stopwords
    tokens = list(filter(lambda l_ph: l_ph not in stop_words, tokens))
    # Remove punctuation and stem
    tokens = [porter_stemmer.stem((val.translate(translator))) for val in tokens]
    # tokens = list(map(lambda val: PorterStemmer().stem(val), tokens))

    # tokens = set(tokens)
    if '' in tokens:
        while '' in tokens:
            tokens.remove('')

    tokens = list(tokens)
    return tokens



def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


total_itrs = 0
start_time = time.time()
with open('data/train.jsonl', 'r') as openfile:
    for line in file_reader_generator(openfile):
        if total_itrs == 1:
            break

        if total_itrs%50 == 0:
            end_time = time.time()
            total_time = end_time - start_time
            print(total_itrs, total_time)
            start_time = time.time()

        total_itrs += 1

        json_dict = json.loads(line)
        curr_claim = json_dict['claim']
        claim_id = json_dict['id']
        tokenised_claim = tokenise_claim(curr_claim)

        res = retrieve_top_five_docs(tokenised_claim)
        # print(claim_id, res)
