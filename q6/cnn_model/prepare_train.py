'''
Build the training dataset for Q6 without using pretrined embeddings
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

for claimId, val in training_db.items():
    supportsOrRefutes = val[1]

    if ctr_breaker % 500 == 0:
        print(ctr_breaker)


    if ctr_breaker == 50000:
        break

    ctr_breaker += 1

    if supportsOrRefutes != 'NOT ENOUGH INFO':
        claim = val[2]

        claimTokens = tokenise_line(claim)

        combinedClaimAndSentence = []

        combinedClaimAndSentence.extend(claimTokens)

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

            combinedClaimAndSentence.extend(lineTokens)


        combinedClaimAndSentence = ' '.join(combinedClaimAndSentence)
        if ctr_breaker < 5:
            print(combinedClaimAndSentence)

        X_train.append(combinedClaimAndSentence)
        if supportsOrRefutes == 'SUPPORTS':
            y_train.append(1)
        else:
            y_train.append(0)


end_time = time.time()
print('Time to create ds ', end_time - start_time)

X_train = np.array(X_train)
y_train = np.array(y_train)




def build_vocab(questions):
    # Build vocab and word index
    vocab_set = set()
    for qstn in questions:
        vocab_set = vocab_set.union(set(qstn.split()))
    vocab = list(vocab_set)

    # 0 is used for padding
    word_to_idx = OrderedDict([(w,i) for i,w in enumerate(vocab,1)])

    return word_to_idx

def retrieve_one_hot_embeddings(questions, word_to_idx):
    embeddings = []
    for qstn in questions:
        embeddings.append([word_to_idx[word] for word in str(qstn).split() if word in word_to_idx])
    return embeddings


print('Building vocab')
vocab_dict = build_vocab(X_train)

print('pickling vocab')
pickle_object(vocab_dict, 'vocab_dict_q6')

vocabulary_size = len(vocab_dict)+1


X_train_embedding = retrieve_one_hot_embeddings(X_train, vocab_dict)
# X_test_embedding = retrieve_one_hot_embeddings(X_test,vocab_dict)

# Number of labels would be just 4: +,-,*,/
num_of_labels = len(set(y_train))
print('num of labels', num_of_labels)
labels = list(set(y_train))
print(labels)

# # Create dict for labels to index
# label_to_index = {o:i for i,o in enumerate(labels)}
# index_to_label = {i:o for i,o in enumerate(labels)}
#
#
# # Convert labels in the training and test datset to numeric format
# y_train_label_numeric_rep = [label_to_index[label] for label in y_train]
# y_test_label_numeric_rep = [label_to_index[label] for label in y_test]

# Just creates the actual one hot encoded vectors
# e.g. 0 : [0 0 0 0]
# 1: [0 1 0 0]
# y_train_distribution = np_utils.to_categorical(y_train_label_numeric_rep, num_of_labels)
# y_test_distribution = np_utils.to_categorical(y_test_label_numeric_rep, num_of_labels)


# pad (post) questions to max length
max_length = 500
print('Padding training sequences')
X_train_embedding_padded = pad_sequences(X_train_embedding, maxlen=max_length, padding='post')
# X_test_embedding_padded = pad_sequences(X_test_embedding, maxlen=max_length, padding='post')


pickle_object(X_train_embedding_padded, 'X_train_q6')

pickle_object(y_train, 'y_train_q6')







conn.close()
training_db.close()
