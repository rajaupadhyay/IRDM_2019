'''
Query Likelihood Unigram Language Model (WITH JELINEK-MERCER SMOOTHING)
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

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)


def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))



wordFreqInColl_f = open('wordFrequencyInCollection.pickle', 'rb')
wordFreqInColl = pickle.load(wordFreqInColl_f)
wordFreqInColl_f.close()

VOCABSIZE = 3922275
totalWordsInCollection = sum(wordFreqInColl)
print('Total words in collection ', totalWordsInCollection)


mul = lambda x, y: x*y

'''
Iterate over the terms in the query
For each term, obtain relevant docs form inverted index

Maintain 2 dicts
Dict1: Stores the product of results
Dict2: Stores the indices of the terms in the claim that have been used in Dict1

Iterate over the relevant docs
Multiply the normalised frequency by lambda and add the result to (lambda * normalised frequency over
the entire collection)
Now multiply the respective values in Dict1 by this new result
When iterating over the relevant docs add the indices of the terms that have been added to Dict2

Iterate over Dict2
For each of the term indices that are not in the list
Get the normalised collection frequency for that term
Multiply the value in Dict1 by this value


Required:
Inverted index with frequency
Collection frequency for each term

'''

def retrieve_top_five_docs(tokens, lambdaParam=0.5):
    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]

    length_of_claim = len(tokens)

    # Key: Doc number/ID, value: cosine similarity
    # Set defaultdict to start with 1 (because we're multplying the probs)
    result_dictionary = defaultdict(lambda: 1.0)

    words_added = defaultdict(list)

    with SqliteDict('ividx_with_freq.sqlite', decode=decompress_set) as ividx_dct:
        for tkn_idx in range(len(tokens)):
            tkn = tokens[tkn_idx]

            tknIdxInVocab = tkn_indices[tkn_idx]

            docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]

            for doc_tuple in docs_from_ividx:

                result_dictionary[doc_tuple[0]] *= ((lambdaParam*doc_tuple[1]) + ((1-lambdaParam)*(wordFreqInColl[tknIdxInVocab]/totalWordsInCollection)))
                words_added[doc_tuple[0]].append(tknIdxInVocab)

    formatted_result_dictionary = {}

    for k, v in result_dictionary.items():
        new_value = v
        if len(words_added[k]) != length_of_claim:
            mulProd = [(1-lambdaParam)*(wordFreqInColl[tknIdxInVocab]/totalWordsInCollection) for tknIdxInVocab in tkn_indices if tknIdxInVocab not in words_added[k]]
            mulRes = reduce(mul, mulProd)
            new_value *= mulRes

        formatted_result_dictionary[k] = new_value


    top_5_document_indices = sorted(formatted_result_dictionary, key=formatted_result_dictionary.get, reverse=True)[:5]

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
with open('data/train.jsonl', 'r') as openfile:
    for line in file_reader_generator(openfile):
        if total_itrs == 13:
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

        res = retrieve_top_five_docs(tokenised_claim, lambdaParam=0.5)
        print(claim_id, res)
