'''
Retrieves top 5 documents for the claims (with MultiProcessing)
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
import sqlite3

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
# print(conn.keys())

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

high_doc_freq_words = pd.read_csv('high_freq_words_stash.csv', sep=',')
high_doc_freq_words = list(high_doc_freq_words['word'].values)

print('Loading the idf vector')
idf_vec_f = open('idf_vector.pickle', 'rb')
idf_vec = pickle.load(idf_vec_f)
idf_vec_f.close()
print('IDF Vector loaded')

print('Loading the enl vector')
enl_vec_f = open('enlv_for_documents.pickle', 'rb')
enlv = pickle.load(enl_vec_f)
enl_vec_f.close()
print('enl Vector loaded')


def retrieve_top_five_docs(tokens, ividx_dct):
    # Where the magic happens

    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]
    term_freq = Counter(tokens)
    total_length = len(tokens)
    term_freq_vector = np.array([term_freq[tkn]/total_length for tkn in tokens])

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]
    idf_vector = np.array([idf_vec[tkn_idx] for tkn_idx in tkn_indices])

    tfidf_for_query = term_freq_vector * idf_vector
    enlv_for_query = LA.norm(tfidf_for_query)

    # Key: Doc number/ID, value: cosine similarity
    '''
    Iteratively add the cosine similarity values to the result_dictionary.
    Maybe a more optimal method is required to reduce the rate below 1 claim/sec
    '''

    result_dictionary = defaultdict(float)
    enlv_dictionary = {}

    for tkn_idx in range(len(tokens)):
        tkn = tokens[tkn_idx]
        # if tkn in high_doc_freq_words:
        #     continue

        idf_for_word = idf_vector[tkn_idx]
        tfidf_query = tfidf_for_query[tkn_idx]
        temp_buf = tfidf_query * enlv_for_query

        docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]


        for doc_tuple in docs_from_ividx:
            # result_dictionary[doc_tuple[0]] += (doc_tuple[1] * idf_for_word * tfidf_query)/(enlv[doc_tuple[0]]*enlv_for_query)
            result_dictionary[doc_tuple[0]] += ((doc_tuple[1] * idf_for_word)/(enlv[doc_tuple[0]])) * temp_buf


    top_5_document_indices = sorted(result_dictionary, key=result_dictionary.get, reverse=True)[:5]

    return top_5_document_indices


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


def helper_fn(batch_fn):
    # Use this to iterate over the file
    # For each claim, first tokenise it and then retrieve_top_five_docs
    total_itrs = 0
    start_time = time.time()
    top_5_docs_dict = {}
    claim_id = None

    # Get a local connection to the db although global would work too
    ividx_dct = SqliteDict('ividx_with_freq.sqlite', decode=decompress_set)

    print('ENTERED PROCESS')
    sys.stdout.flush()

    with open(batch_fn, 'r') as openfile:
        for line in file_reader_generator(openfile):
            if total_itrs == 10:
                break

            if total_itrs%50 == 0:
                end_time = time.time()
                total_time = end_time - start_time
                print(total_itrs, total_time)
                sys.stdout.flush()
                start_time = time.time()

            total_itrs += 1

            json_dict = json.loads(line)
            curr_claim = json_dict['claim']
            claim_id = json_dict['id']
            tokenised_claim = tokenise_claim(curr_claim)

            res = retrieve_top_five_docs(tokenised_claim, ividx_dct)
            top_5_docs_dict[claim_id] = res

    ividx_dct.close()

    return(top_5_docs_dict, claim_id)



# Main for multiprocessing
def main():
    # Create a new db which will store the 5 top docs for each claim
    top_5_docs_sq_db = SqliteDict('top_5_documents_tfidf.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)

    # Split the training file of claims into 4 batches to distribute load
    batch_1 = 'data/train_batches/train_batch1.jsonl'
    batch_2 = 'data/train_batches/train_batch2.jsonl'
    batch_3 = 'data/train_batches/train_batch3.jsonl'
    batch_4 = 'data/train_batches/train_batch4.jsonl'

    start_time_parent = time.time()

    pool = Pool(processes=4)
    first_batch_res = pool.apply_async(helper_fn, [batch_1])
    second_batch_res = pool.apply_async(helper_fn, [batch_2])
    third_batch_res = pool.apply_async(helper_fn, [batch_3])
    fourth_batch_res = pool.apply_async(helper_fn, [batch_4])

    pool.close()
    pool.join()


    first_batch_res, claim_id_1 = first_batch_res.get()
    second_batch_res, claim_id_2 = second_batch_res.get()
    third_batch_res, claim_id_3 = third_batch_res.get()
    fourth_batch_res, claim_id_4 = fourth_batch_res.get()

    print('batch_1 last claim:', claim_id_1)
    print('batch_2 last claim:', claim_id_2)
    print('batch_3 last claim:', claim_id_3)
    print('batch_4 last claim:', claim_id_4)

    end_time_parent = time.time()
    print('Processing time:', end_time_parent-start_time_parent)

    # Combine results from all 4 processes
    final_dict = {**first_batch_res, **second_batch_res, **third_batch_res, **fourth_batch_res}

    # Insert the results into the database
    for k,v in final_dict.items():
        top_5_docs_sq_db[k] = v


    top_5_docs_sq_db.close()


if __name__ == '__main__':
    main()
