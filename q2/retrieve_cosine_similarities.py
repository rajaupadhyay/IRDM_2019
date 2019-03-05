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

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())
print(conn.hget('doc_name_index', 'The_Ten_Commandments_-LRB-1956_film-RRB-'))
check_idf = conn.hget('vocab_dictionary', 'costerwaldau')

# [3510191, 5149384, 291223, 3541706, 3508468]

# [4730014, 4696498, 4700169, 4706623, 4773870]


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


print('idf_val', idf_vec[int(check_idf)])

def retrieve_top_five_docs(tokens):
    # print('Retrieving Top 5 Documents')
    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]
    term_freq = Counter(tokens)
    total_length = len(tokens)
    term_freq_vector = np.array([term_freq[tkn]/total_length for tkn in tokens])

    tkn_indices = [int(conn.hget('vocab_dictionary', tkn)) for tkn in tokens]
    idf_vector = np.array([idf_vec[tkn_idx] for tkn_idx in tkn_indices])

    tfidf_for_query = term_freq_vector * idf_vector
    enlv_for_query = LA.norm(tfidf_for_query)



    # Key: Doc number/ID, value: cosine similarity
    result_dictionary = defaultdict(float)
    enlv_dictionary = {}

    with SqliteDict('ividx_with_freq.sqlite', decode=decompress_set) as ividx_dct:
        for tkn_idx in range(len(tokens)):
            tkn = tokens[tkn_idx]
            # if tkn in high_doc_freq_words:
            #     continue

            idf_for_word = idf_vector[tkn_idx]
            tfidf_query = tfidf_for_query[tkn_idx]
            temp_buf = tfidf_query * enlv_for_query

            docs_from_ividx = ividx_dct['ividx_{}'.format(tkn)]
            # sorted_docs = sorted(docs_from_ividx, key=lambda x: x[1], reverse=True)
            # print(tkn)
            # print(sorted_docs[:10])

            for doc_tuple in docs_from_ividx:
                # result_dictionary[doc_tuple[0]] += (doc_tuple[1] * idf_for_word * tfidf_query)/(enlv[doc_tuple[0]]*enlv_for_query)
                result_dictionary[doc_tuple[0]] += ((doc_tuple[1] * idf_for_word)/(enlv[doc_tuple[0]])) * temp_buf

    # print(len(result_dictionary))
    # with SqliteDict('enlv_for_docs.sqlite', decode=decompress_set) as enlv_dct:
    #     result_dictionary = {k: result_dictionary[k]/(enlv_dct[k]*enlv_for_query) for k in result_dictionary.keys()}


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
