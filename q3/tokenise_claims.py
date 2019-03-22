import pickle
import numpy as np
import redis
import time
import string
from nltk import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from nltk import word_tokenize
import zlib
import pandas as pd
import json

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)

translator = str.maketrans('', '', string.punctuation)

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['-lrb-', '-rrb-'])

porter_stemmer = PorterStemmer()


def pickle_object(obj, filename):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)



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

    tokens = [tkn for tkn in tokens if conn.hget('vocab_dictionary', tkn)]

    return tokens

def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data



def helper_fn(filePath):
    total_itrs = 0
    claim_id = None
    train_batch = {}

    with open(filePath, 'r') as openfile:
        for line in file_reader_generator(openfile):
            if total_itrs == 3000:
                break

            total_itrs += 1

            json_dict = json.loads(line)
            curr_claim = json_dict['claim']
            claim_id = json_dict['id']
            tokenised_claim = tokenise_claim(curr_claim)
            train_batch[claim_id] = tokenised_claim

    return train_batch


batch_1 = 'data/train_batches/train_batch1.jsonl'
batch_2 = 'data/train_batches/train_batch2.jsonl'
batch_3 = 'data/train_batches/train_batch3.jsonl'
batch_4 = 'data/train_batches/train_batch4.jsonl'

train_batch1 = helper_fn(batch_1)
train_batch2 = helper_fn(batch_2)
train_batch3 = helper_fn(batch_3)
train_batch4 = helper_fn(batch_4)


pickle_object(train_batch1, 'train_batch1')
pickle_object(train_batch2, 'train_batch2')
pickle_object(train_batch3, 'train_batch3')
pickle_object(train_batch4, 'train_batch4')
