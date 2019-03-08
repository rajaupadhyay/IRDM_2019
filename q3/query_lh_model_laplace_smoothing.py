'''
Query Likelihood Unigram Language Model (WITH LAPLACE SMOOTHING)
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

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)


def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))


'''
Retrieve top 5 documents using Query Likelihood Unigram Language Model
WITH LAPLACE SMOOTHING

Iterate over the tokens in the claim
Get the relevant documents from inverted index
Create 2 dictionaries as before

Dict1 will maintain how many query terms have been added to a doc in Dict2
Dict2 will maintain the product of the different query terms

For each relevant document per query term:
Get the normalised frequency and multiply it by the length of the document
A vector needs to be preloaded which contains the length of each document
Dict2['docName'] *= ((normalisedTermFrequency * docLength) + 1)/(docLength+VOCABSIZE)
VOCABSIZE = 3922275

Finally iterate over each key in Dict2:
If the claim/query had e.g. 5 terms but a particular document only has 3 of the terms
then the value of that key in Dict2 needs to be further multiplied as follows
The missingQueryTerms in this case would be 2 (since this particular document is missing
2 query terms)
Dict2['docName'] *= ((1/docLength+VOCABSIZE)^(missingQueryTerms))


The following items are required:
Inverted index with normalised frequency: ividx_with_freq.sqlite
A vector containing the length of each document: CREATE

'''

def retrieve_top_five_docs(tokens):
    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]

    # Key: Doc number/ID, value: cosine similarity
    # Set defaultdict to start with 1 (because we're multplying the probs)
    result_dictionary = defaultdict(lambda: 1.0)

    with SqliteDict('ividx_with_freq.sqlite', decode=decompress_set) as ividx_dct:
        for tkn_idx in range(len(tokens)):
            tkn = tokens[tkn_idx]

            docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]

            for doc_tuple in docs_from_ividx:
                result_dictionary[doc_tuple[0]] *= doc_tuple[1]


    top_5_document_indices = sorted(result_dictionary, key=result_dictionary.get, reverse=True)[:5]

    return top_5_document_indices
