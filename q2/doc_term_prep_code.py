import json
from collections import Counter
from nltk import word_tokenize
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import os
import pickle
import redis
import ast
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from scipy.sparse import coo_matrix
import numpy as np
import time
import string

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())
# print(conn.hget('doc_name_index', '2010_World_Junior_Ice_Hockey_Championships_rosters'))


def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


'''
We need to get the count of unique words in each document
We then need to get the sum of all these counts to find out the number of
non-zero entries in our sparse matrix 211,785,807
'''

translator = str.maketrans('', '', string.punctuation)
wiki_files_path = 'data/wiki-pages/wiki-pages/'
list_of_wiki_files = os.listdir(wiki_files_path)
list_of_wiki_files.sort()

# stop_words = set(stopwords.words('english'))
# stop_words = stop_words.union(['-lrb-', '-rrb-'])

def get_non_zero_count():
    non_zero_count = 0
    porter_stemmer = PorterStemmer()
    for fn in list_of_wiki_files:
        print(fn, non_zero_count)
        start_time = time.time()
        with open(wiki_files_path+fn, 'r') as openfile:
            for line in file_reader_generator(openfile):

                json_dict = json.loads(line)
                file_key = json_dict['id']
                if file_key:
                    text_data = json_dict['text']

                    # Tokenise
                    tokens = word_tokenize(text_data)
                    if tokens:
                        # Lowercase
                        tokens = list(map(lambda x: x.lower(), tokens))
                        # Remove stopwords
                        tokens = list(filter(lambda l_ph: l_ph not in stop_words, tokens))
                        # Remove punctuation and stem
                        tokens = [porter_stemmer.stem((val.translate(translator))) for val in tokens]
                        # tokens = list(map(lambda val: PorterStemmer().stem(val), tokens))

                        tokens = set(tokens)
                        if '' in tokens:
                            tokens.remove('')

                        non_zero_count += len(tokens)



        end_time = time.time()
        print('time taken', end_time - start_time)

    return non_zero_count


# res = get_non_zero_count()
# print(res)


'''
Build document-term matrix: This will help with TFIDF
'''
def build_sparse_term_document_matrix():

    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(['-lrb-', '-rrb-'])
    porter_stemmer = PorterStemmer()

    '''
    Load the docnames
    '''
    ndocs = conn.hlen('doc_name_index')

    '''
    Load the vocab
    '''
    nvocab = conn.hlen('vocab_dictionary')


    n_nonzero = 211785807
    # Dimensions of our Document-Term matrix will be len(docnames) X len(vocab)
    # Create single data, row and cols array - this will contain all the required data
    print('creating empty data, row and cols arrays')
    data = np.empty(n_nonzero, dtype=np.intc)
    rows = np.empty(n_nonzero, dtype=np.intc)
    cols = np.empty(n_nonzero, dtype=np.intc)
    print('created arrays')

    # Current index in data array
    ind = 0

    for fn in list_of_wiki_files:
        print(fn)
        start_time = time.time()
        with open(wiki_files_path+fn, 'r') as openfile:
            for line in file_reader_generator(openfile):
                json_dict = json.loads(line)
                file_key = json_dict['id']

                if file_key:
                    tokens = word_tokenize(json_dict['text'])
                    if tokens:
                        pipe = conn.pipeline()

                        # Lowercase
                        tokens = list(map(lambda x: x.lower(), tokens))
                        # Remove stopwords
                        tokens = list(filter(lambda l_ph: l_ph not in stop_words, tokens))
                        # Remove punctuation and stem
                        tokens = [porter_stemmer.stem((val.translate(translator))) for val in tokens]
                        # Remove any empty strings from the tokens list
                        tokens = list(filter(lambda itx: itx, tokens))


                        if ind == 0:
                            print(tokens)

                        if tokens:
                            term_indices = []
                            for tkn in tokens:
                                idx_from_redis = pipe.hget('vocab_dictionary', tkn)

                            doc_idx = pipe.hget('doc_name_index', file_key)
                            res = pipe.execute()
                            res = list(map(int, res))

                            term_indices = res[:-1]
                            doc_idx = res[-1]

                            term_indices = np.array(term_indices)

                            uniq_indices, counts = np.unique(term_indices, return_counts=True)
                            n_vals = len(uniq_indices)
                            ind_end = ind + n_vals

                            data[ind:ind_end] = counts
                            cols[ind:ind_end] = uniq_indices
                            rows[ind:ind_end] = np.repeat(doc_idx, n_vals)

                            ind = ind_end


        end_time = time.time()
        print('time taken', end_time - start_time)

    print('building sparse doc-term matrix')

    dtm = coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)

    print('successfully built doc-term matrix!')

    print('pickling doc-term matrix')
    pickle_object(dtm, 'doc_term_matrix')
    print('DOC-TERM MATRIX READY!')

build_sparse_term_document_matrix()
