'''
Vanilla Query Likelihood Unigram Language Model (No smoothing)
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
from math import floor

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))


'''
Retrieve top 5 documents using Query Likelihood Unigram Language Model
Try set intersection method to only include documents that contain ALL the query terms
Since a lack of a term results in 0 anyway
'''
def retrieve_top_five_docs(tokens):
    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

    length_of_claim = len(tokens)
    # Key: Doc number/ID, value: cosine similarity
    # Set defaultdict to start with 1 (because we're multplying the probs)
    result_dictionary = defaultdict(lambda: 1.0)

    # Lets maintain a dictionary to keep count of how many query terms have been added to the doc
    # This is important because if a query term does not exist in a doc then the product should result in 0
    words_added = defaultdict(int)

    with SqliteDict('ividx_with_freq.sqlite', decode=decompress_set) as ividx_dct:
        for tkn_idx in range(len(tokens)):
            tkn = tokens[tkn_idx]

            docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]

            for doc_tuple in docs_from_ividx:
                result_dictionary[doc_tuple[0]] *= doc_tuple[1]
                words_added[doc_tuple[0]] += 1

    # We only want to look at the documents that have all query terms
    # If the document did not have all query terms then it would be zeroed out when multplying
    formatted_result_dictionary = {k:v for k,v in result_dictionary.items() if floor(words_added[k]/length_of_claim) == 1}

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
        if total_itrs == 10:
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
        print(claim_id, res)
