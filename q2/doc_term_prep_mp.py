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
from multiprocessing import Pool
import sys

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

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(['-lrb-', '-rrb-'])

def get_non_zero_count(list_of_wiki_files):
    non_zero_count = 0
    porter_stemmer = PorterStemmer()
    for fn in list_of_wiki_files:
        print(fn, non_zero_count)
        sys.stdout.flush()
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
        sys.stdout.flush()

    return non_zero_count



def main():
    list_of_wiki_files = os.listdir(wiki_files_path)
    list_of_wiki_files.sort()

    batch_1 = list_of_wiki_files[:27]
    batch_2 = list_of_wiki_files[27:54]
    batch_3 = list_of_wiki_files[54:81]
    batch_4 = list_of_wiki_files[81:109]

    pool = Pool(processes=4)
    first_batch = pool.apply_async(get_non_zero_count, [batch_1])
    second_batch = pool.apply_async(get_non_zero_count, [batch_2])
    third_batch = pool.apply_async(get_non_zero_count, [batch_3])
    fourth_batch = pool.apply_async(get_non_zero_count, [batch_4])

    pool.close()
    pool.join()

    print(first_batch.get())
    print(second_batch.get())
    print(third_batch.get())
    print(fourth_batch.get())

    res = first_batch.get() + second_batch.get() + third_batch.get() + fourth_batch.get()
    print(res)

if __name__ == '__main__':
    main()
