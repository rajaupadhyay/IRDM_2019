'''
81.53% accuracy!
'''

import pickle
import numpy as np
import redis
import time
import sys
from sqlitedict import SqliteDict
import pandas as pd
import zlib
import sqlite3
import json
import re
from gensim.models import KeyedVectors
import string
import random
from nltk import word_tokenize
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

prepare_ = 'train'


def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compress_set(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def decompress_set(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

def add_key_to_db(key, val_list):
    claims_db[key] = val_list


def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data

print('Loading Claims')
if prepare_ == 'train':
    training_db = SqliteDict('training_db.sqlite', decode=decompress_set)
else:
    training_db = SqliteDict('testing_db.sqlite', decode=decompress_set)

print('Loading wiki corpus')
conn = sqlite3.connect('wiki_corpus.db')
c = conn.cursor()


def flatten_list(lst):
    flattened = [item for nstd in lst for item in nstd]
    return flattened

translator = str.maketrans('', '', string.punctuation)

def tokenise_line(line):
    line = line.replace('\n', '')
    line = line.replace('\t', ' ')
    tokens = word_tokenize(line)

    # Lowercase
    tokens = list(map(lambda x: x.lower(), tokens))
    # Remove punctuation and stem
    tokens = [val.translate(translator) for val in tokens]

    # tokens = set(tokens)
    if '' in tokens:
        while '' in tokens:
            tokens.remove('')

    tokens = list(tokens)
    return tokens



claimLst = []
evidenceLst = []
targetLabelLst = []
ctr_breaker = 0

for claimId, val in training_db.items():
    supportsOrRefutes = val[1]

    if ctr_breaker % 500 == 0:
        print(ctr_breaker)


    if ctr_breaker == 150000:
        break

    ctr_breaker += 1

    bufferEvidenceCombination = []

    if supportsOrRefutes != 'NOT ENOUGH INFO':
        claim = val[2]
        claimTokens = tokenise_line(claim)
        claim = ' '.join(claimTokens)

        positiveExamples = val[-1]
        positiveExamples = flatten_list(positiveExamples)

        for itx in positiveExamples:
            docname = itx[-2]
            lineNumber = itx[-1]

            c.execute('SELECT lines FROM wiki WHERE id = ?', (docname, ))

            try:
                lines = c.fetchone()[0]
            except:
                # print('could not get lines for doc {} '.format(docname))
                continue

            lines = list(filter(lambda x: x, re.split('\d+\\t', lines)))
            line = lines[lineNumber]

            lineTokens = tokenise_line(line)

            bufferEvidenceCombination.extend(lineTokens)


        bufferEvidenceCombination = ' '.join(bufferEvidenceCombination)


        claimLst.append(claim)
        evidenceLst.append(bufferEvidenceCombination)
        if supportsOrRefutes == 'SUPPORTS':
            targetLabelLst.append(1)
        else:
            targetLabelLst.append(0)

trainingData = (claimLst, evidenceLst, targetLabelLst)

pickle_object(trainingData, '{}ing_data'.format(prepare_))
