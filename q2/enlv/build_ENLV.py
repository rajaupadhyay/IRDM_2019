import sys
from sqlitedict import SqliteDict
import pandas as pd
import zlib, pickle, sqlite3
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
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords
import time
import numpy as np

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

# enlv = SqliteDict('enlv_for_docs.sqlite', autocommit=True, encode=compress_set, decode=decompress_set)
#
# def add_enlv_for_doc(doc, enlv_value):
#     enlv[doc] = enlv_value
#
# tfidf_f = open('tf_idf_matrix_compressed.pickle', 'rb')
# tfidf_matrix = pickle.load(tfidf_f)
# tfidf_matrix = tfidf_matrix.tocsr()
# tfidf_f.close()
#
# t_shape = tfidf_matrix.shape
# print(t_shape)
#
# for doc_idx in range(t_shape[0]):
#     if doc_idx%100000 == 0:
#         print('checkpoint', doc_idx)
#         print('total commited', len(enlv))
#         time.sleep(5)
#
#     if doc_idx < 10:
#         print(doc_idx)
#
#     curr_row = tfidf_matrix[doc_idx].data
#     enlv_val = np.linalg.norm(curr_row).round(4)
#     add_enlv_for_doc(doc_idx, enlv_val)
#
#
# print('Completed')
# enlv.close()
