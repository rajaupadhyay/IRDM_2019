'''
Query Likelihood Unigram Language Model (WITH DIRICHLET SMOOTHING)
RETURNS 5 DOCS PER CLAIM
MultiProcessing
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

'''
First tokenise the claims using tokenise_claims.py
Read the claimToDocsDict.pickle dict
Add to this dict new claims and docs
pickle this dict back to disk
edit the train batches and reset cursor
'''

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


    # tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

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
inverted_doc_name_dict_f = open('inverted_doc_name_dict.pickle', 'rb')
inverted_doc_name_dict = pickle.load(inverted_doc_name_dict_f)
inverted_doc_name_dict_f.close()


def helper_fn(batch_fn):
    # Use this to iterate over the file
    # For each claim, first tokenise it and then retrieve_top_five_docs
    total_itrs = 0
    start_time = time.time()
    top_5_docs_dict = {}
    cid = None

    # Get a local connection to the db although global would work too
    ividx_dct = SqliteDict('ividx_with_freq.sqlite', decode=decompress_set)

    print('ENTERED PROCESS')
    sys.stdout.flush()

    train_btch_f = open(batch_fn, 'rb')
    train_btch = pickle.load(train_btch_f)
    train_btch_f.close()



    for cid, tokens in train_btch.items():
        if total_itrs == 1500:
            break

        if total_itrs%50 == 0:
            end_time = time.time()
            total_time = end_time - start_time
            print(total_itrs, total_time)
            sys.stdout.flush()
            start_time = time.time()

        total_itrs += 1

        res = retrieve_top_five_docs(tokens, ividx_dct)
        top_5_doc_items = [inverted_doc_name_dict[str(itx)] for itx in res]
        top_5_docs_dict[cid] = top_5_doc_items

    ividx_dct.close()

    return(top_5_docs_dict, cid)





# Main for multiprocessing
def main():
    # Create a new db which will store the 5 top docs for each claim
    # top_5_docs_sq_db = SqliteDict('top_5_documents_tfidf.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)

    # claimToDocsDict_f = open('claimToDocsDict.pickle', 'rb')
    # claimToDocsDict = pickle.load(claimToDocsDict_f)
    # claimToDocsDict_f.close()


    carryOutTrainData = 0

    if carryOutTrainData == 1:
        batch_1 = 'data/tokenised_train_batches/train_batch1.pickle'
        batch_2 = 'data/tokenised_train_batches/train_batch2.pickle'
        batch_3 = 'data/tokenised_train_batches/train_batch3.pickle'
        batch_4 = 'data/tokenised_train_batches/train_batch4.pickle'

        claimToDocsDict_f = open('claimToDocsDict_train.pickle', 'rb')
        claimToDocsDict = pickle.load(claimToDocsDict_f)
        claimToDocsDict_f.close()
        cTd_outname = 'claimToDocsDict_train_V1'

    else:
        batch_1 = 'data/tokenised_test_batches/test_batch1.pickle'
        batch_2 = 'data/tokenised_test_batches/test_batch2.pickle'
        batch_3 = 'data/tokenised_test_batches/test_batch3.pickle'
        batch_4 = 'data/tokenised_test_batches/test_batch4.pickle'

        claimToDocsDict_f = open('claimToDocsDict_test.pickle', 'rb')
        claimToDocsDict = pickle.load(claimToDocsDict_f)
        claimToDocsDict_f.close()
        cTd_outname = 'claimToDocsDict_test_V1'


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

    print('Pickling dict')
    # Update claimToDocsDict

    claimToDocsDict.update(final_dict)
    pickle_object(claimToDocsDict, cTd_outname)


    # # Insert the results into the database
    # for k,v in final_dict.items():
    #     top_5_docs_sq_db[k] = v
    #
    #
    # top_5_docs_sq_db.close()


if __name__ == '__main__':
    main()
