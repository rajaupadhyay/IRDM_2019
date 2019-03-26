'''
Query Likelihood Unigram Language Model (WITH DIRICHLET SMOOTHING)
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
import pandas as pd
import json
from multiprocessing import Pool
import sys
from functools import reduce
from math import log

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)


def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))



wordFreqInColl_f = open('wordFrequencyInCollection.pickle', 'rb')
wordFreqInColl = pickle.load(wordFreqInColl_f)
wordFreqInColl_f.close()

doc_length_vec_f = open('doc_length_vector.pickle', 'rb')
doc_length_vec = pickle.load(doc_length_vec_f)
doc_length_vec_f.close()

doc_length_vec = np.array(doc_length_vec)


VOCABSIZE = 3922275
totalWordsInCollection = sum(wordFreqInColl)
print('Total words in collection ', totalWordsInCollection)

mu = totalWordsInCollection/5396041
print('µ value', mu)

wordFreqInColl = wordFreqInColl/totalWordsInCollection


denom_vec = doc_length_vec + mu
lhs_of_prod = doc_length_vec/denom_vec
mu_div_denom = mu/denom_vec

def retrieve_top_five_docs(tokens, ividx_dct):
    l1_st = time.time()


    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]

    length_of_claim = len(tokens)

    result_dictionary = defaultdict(float)

    for tkn_idx in range(len(tokens)):
        tkn = tokens[tkn_idx]

        tknIdxInVocab = tkn_indices[tkn_idx]

        docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]

        normalisedWordCountInColl = wordFreqInColl[tknIdxInVocab]
        # normalisedWordCountInColl = word_fre_in_coll/totalWordsInCollection

        for doc_tuple in docs_from_ividx:
            curr_idx = int(doc_tuple[0])
            # curr_doc_length = doc_length_vec[int(doc_tuple[0])]
            # denom = curr_doc_length+mu
            inner_calc = (((lhs_of_prod[curr_idx])*(doc_tuple[1])) / ((mu_div_denom[curr_idx])*(normalisedWordCountInColl))) + 1
            result_dictionary[doc_tuple[0]] += log(inner_calc)
            # words_added[doc_tuple[0]].append(tknIdxInVocab)


    top_5_document_indices = sorted(result_dictionary, key=result_dictionary.get, reverse=True)[:5]


    return top_5_document_indices




'''
Iterate over the training file containing the claims
Retrieve top 5 docs for each claim
'''
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

ividx_dct = SqliteDict('ividx_with_freq.sqlite', decode=decompress_set)


with open('data/train.jsonl', 'r') as openfile:
    for line in file_reader_generator(openfile):
        if total_itrs == 13:
            end_time = time.time()
            total_time = end_time - start_time
            print(total_itrs, total_time)
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

        res = retrieve_top_five_docs(tokenised_claim, ividx_dct)
        print(claim_id, res)
