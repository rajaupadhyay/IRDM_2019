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

def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())

# print(conn.hget('vocab_dictionary', 'newdelhi'))

# print(conn.sismember('vocab_set', 'maría'))
# print(conn.scard('vocab_set'))

# vocabulary_list = list(conn.smembers('vocab_set'))
# print('pickling vocab')
# pickle_object(vocabulary_list, 'vocab_index')


# with open('vocab_index.pickle', 'rb') as input_file:
#     vocab = pickle.load(input_file)
#     print(len(vocab))
#     print(type(vocab))
#     # print('nairobi' in vocab)
#
#     for idx, word in enumerate(vocab):
#         conn.hset("vocab_dictionary", word, idx)
