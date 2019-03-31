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
A vector containing the length of each document: doc_length_vector.pickle

'''

doc_length_vec_f = open('doc_length_vector.pickle', 'rb')
doc_length_vec = pickle.load(doc_length_vec_f)
doc_length_vec_f.close()

VOCABSIZE = 3922275


def retrieve_top_five_docs(tokens):
    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]

    length_of_claim = len(tokens)

    # Key: Doc number/ID, value: cosine similarity
    # Set defaultdict to start with 1 (because we're multplying the probs)
    result_dictionary = defaultdict(lambda: 1.0)

    # Maintain this dict to see how many query terms have been added per doc
    # If a doc does not have a particular query then the final value must be multiplied
    # by ((1/docLength+VOCABSIZE)^(missingQueryTerms)) to account for the missing values
    # This dict will help us calculate missingQueryTerms
    words_added = defaultdict(int)

    with SqliteDict('ividx_with_freq.sqlite', decode=decompress_set) as ividx_dct:
        for tkn_idx in range(len(tokens)):
            tkn = tokens[tkn_idx]

            docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]

            for doc_tuple in docs_from_ividx:
                curr_doc_length = doc_length_vec[int(doc_tuple[0])]
                # Apply Laplace smoothing
                # Add 1 and divide by the (document length + Vocab size)
                result_dictionary[doc_tuple[0]] *= ((doc_tuple[1] * curr_doc_length) + 1)/(curr_doc_length + VOCABSIZE)
                words_added[doc_tuple[0]] += 1

    formatted_result_dictionary = {}

    for k, v in result_dictionary.items():
        new_value = v
        if words_added[k] != length_of_claim:
            missingQueryTerms = length_of_claim - words_added[k]
            assert(missingQueryTerms>=1)
            curr_doc_length = doc_length_vec[int(k)]
            new_value = v * ((1/(curr_doc_length+VOCABSIZE))**(missingQueryTerms))

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

        res = retrieve_top_five_docs(tokenised_claim)
        print(claim_id, res)
