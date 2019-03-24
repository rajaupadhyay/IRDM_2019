'''
Build the test dataset for logit (Development FEVER dataset)
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
claimToDocsDict_f = open('claimToDocsDict_test.pickle', 'rb')
claimToDocsDict = pickle.load(claimToDocsDict_f)
claimToDocsDict_f.close()

print('Loading Claims')
testing_db = SqliteDict('testing_db.sqlite', decode=decompress_set)

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



X_test = []
y_test = []

'''
Build testing dataset
'''
print('Building testing dataset')
ctr_breaker = 0
start_time = time.time()
for claimId, docList in claimToDocsDict.items():
    # if ctr_breaker == 50:
    #     break

    if ctr_breaker % 500 == 0:
        print(ctr_breaker)

    ctr_breaker += 1

    supportsOrRefutes = testing_db[claimId][1]
    if supportsOrRefutes != 'NOT ENOUGH INFO':
        claim = testing_db[claimId][2]

        claimTokens = tokenise_line(claim)

        claimVec = []

        for token in claimTokens:
            if token in w2v_model:
                claimVec.append(w2v_model[token])


        if len(claimVec) > 0:
            lenClaimVec = len(claimVec)
            claimVec = sum(claimVec)/lenClaimVec

            positiveExamples = testing_db[claimId][-1]
            positiveExamples = flatten_list(positiveExamples)
            addedLines = []

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
                    X_test.append(trainingVector)
                    y_test.append(1)


            # Add negative examples
            for _ in range(len(addedLines)+1):
                randomDoc = random.choice(docList)
                c.execute('SELECT lines FROM wiki WHERE id = ?', (randomDoc, ))

                lines = c.fetchone()[0]
                lines = list(filter(lambda x: x, re.split('\d+\\t', lines)))
                totalLines = len(lines)

                lenMask = [lnmb for lnmb in range(totalLines) if lnmb not in addedLines]

                if len(lenMask) == 0:
                    continue

                randLineNumber = random.choice(lenMask)

                line = lines[randLineNumber]

                lineTokens = tokenise_line(line)

                lineVec = []

                for token in lineTokens:
                    if token in w2v_model:
                        lineVec.append(w2v_model[token])

                if len(lineVec) > 0:
                    lenLineVec = len(lineVec)
                    lineVec = sum(lineVec)/lenLineVec

                    trainingVector = np.concatenate([claimVec, lineVec])
                    X_test.append(trainingVector)
                    y_test.append(0)



end_time = time.time()
print('Time to create ds ', end_time - start_time)
print('Pickling testing ds')

X_test = np.array(X_test)
y_test = np.array(y_test)

pickle_object(X_test, 'X_test')
pickle_object(y_test, 'y_test')


conn.close()
testing_db.close()
