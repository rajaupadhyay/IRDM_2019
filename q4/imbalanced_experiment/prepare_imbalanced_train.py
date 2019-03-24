'''
Build the training dataset without sampling for logit
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
from collections import defaultdict

print('Loading word2vec')
w2v_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('Loaded Google pretrained embeddings')


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

print('Loading claimToDocsDict')
claimToDocsDict_f = open('claimToDocsDict_train.pickle', 'rb')
claimToDocsDict = pickle.load(claimToDocsDict_f)
claimToDocsDict_f.close()

print('Loading Claims')
training_db = SqliteDict('training_db.sqlite', decode=decompress_set)

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



X_train = []
y_train = []

'''
Build training dataset
'''
print('Building training dataset')
ctr_breaker = 0
start_time = time.time()
for claimId, docList in claimToDocsDict.items():
    # if ctr_breaker == 50:
    #     break

    if ctr_breaker % 500 == 0:
        print(ctr_breaker)

    ctr_breaker += 1

    supportsOrRefutes = training_db[claimId][1]
    if supportsOrRefutes != 'NOT ENOUGH INFO':
        claim = training_db[claimId][2]

        claimTokens = tokenise_line(claim)

        claimVec = []

        for token in claimTokens:
            if token in w2v_model:
                claimVec.append(w2v_model[token])


        if len(claimVec) > 0:
            lenClaimVec = len(claimVec)
            claimVec = sum(claimVec)/lenClaimVec

            positiveExamples = training_db[claimId][-1]
            positiveExamples = flatten_list(positiveExamples)
            addedLines = []


            addedPosExamples = defaultdict(list)


            # Add positive examples
            for itx in positiveExamples:
                docname = itx[-2]
                lineNumber = itx[-1]

                c.execute('SELECT lines FROM wiki WHERE id = ?', (docname, ))

                try:
                    lines = c.fetchone()[0]
                except:
                    print('could not get lines for doc {} '.format(docname))
                    continue

                lines = list(filter(lambda x: x, re.split('\d+\\t', lines)))
                line = lines[lineNumber]
                addedLines.append(lineNumber)

                lineTokens = tokenise_line(line)

                lineVec = []

                for token in lineTokens:
                    if token in w2v_model:
                        lineVec.append(w2v_model[token])


                if len(lineVec) > 0:
                    lenLineVec = len(lineVec)
                    lineVec = sum(lineVec)/lenLineVec

                    trainingVector = np.concatenate([claimVec, lineVec])
                    X_train.append(trainingVector)
                    y_train.append(1)

                addedPosExamples[docname].append(lineNumber)


            # Add negative examples

            for docItx in docList:
                c.execute('SELECT lines FROM wiki WHERE id = ?', (docname, ))

                try:
                    lines = c.fetchone()[0]
                except:
                    print('could not get lines for doc {} '.format(docname))
                    continue

                lines = list(filter(lambda x: x, re.split('\d+\\t', lines)))

                for lineNumber in range(len(lines)):
                    if lineNumber in addedPosExamples[docItx]:
                        continue

                    line = lines[lineNumber]

                    lineTokens = tokenise_line(line)

                    lineVec = []

                    for token in lineTokens:
                        if token in w2v_model:
                            lineVec.append(w2v_model[token])


                    if len(lineVec) > 0:
                        lenLineVec = len(lineVec)
                        lineVec = sum(lineVec)/lenLineVec

                        trainingVector = np.concatenate([claimVec, lineVec])
                        X_train.append(trainingVector)
                        y_train.append(0)


end_time = time.time()
print('Time to create ds ', end_time - start_time)
print('Pickling training ds')

X_train = np.array(X_train)
y_train = np.array(y_train)

pickle_object(X_train, 'X_train_imbalanced')
pickle_object(y_train, 'y_train_imbalanced')


conn.close()
training_db.close()
