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

conn = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
print(conn.keys())
print(conn.scard('vocab_set'))

# conn.hset("test_vd", '', 5)

# print(conn.keys())
# x = list(conn.smembers('vocab_set'))
# print(x[0:30])
# print(conn.srem('test_sa', *set(['car', 'bull', 'type', 'f', 'f'])))
# print(conn.keys())

# print(conn.sismember('vocab_set', 'upadhyay'))


# translator = str.maketrans('', '', string.punctuation)
# wiki_files_path = 'data/wiki-pages/wiki-pages/'
# list_of_wiki_files = os.listdir(wiki_files_path)
# list_of_wiki_files.sort()
#
# stop_words = set(stopwords.words('english'))
# stop_words = stop_words.union(['-lrb-', '-rrb-'])


def file_reader_generator(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


def load_wiki_json():
    VOCAB_INDEX = 1
    DOC_INDEX = 0
    porter_stemmer = PorterStemmer()
    for fn in list_of_wiki_files:
        print(fn)
        start_time = time.time()
        with open(wiki_files_path+fn, 'r') as openfile:
            for line in file_reader_generator(openfile):
                # if DOC_INDEX >= 5:
                #     break

                json_dict = json.loads(line)
                file_key = json_dict['id']
                if file_key:
                    text_data = json_dict['text']
                    if DOC_INDEX < 3:
                        print(text_data)
                    # Remove punctuation
                    # text_data = text_data.translate(translator)

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

                        if DOC_INDEX < 3:
                            print(tokens)

                        if tokens:
                            conn.sadd('vocab_set', *tokens)
                        else:
                            print('error inserting set')
                            print(tokens, file_key, fn)

                        DOC_INDEX += 1

        end_time = time.time()
        print('time taken', end_time - start_time)

# load_wiki_json()
