import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed, concatenate, recurrent
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from keras.models import Sequential


def preprocessGlove():
    GLOVE_STORE = './precomputed_glove.weights'
    if USE_GLOVE:
      if not os.path.exists(GLOVE_STORE + '.npy'):
        print('Computing GloVe')

        embeddings_index = {}
        # f = open('../glove.840B.300d.txt')
        with zipfile.ZipFile('../input/glove-embeddings/glove.840B.300d') as z:
            with z.open('glove.840B.300d.txt', 'r') as f:
                for line in f:
                  line = line.decode("utf-8")
                  values = line.split(' ')
                  word = values[0]
                  coefs = np.asarray(values[1:], dtype='float32')
                  embeddings_index[word] = coefs
        # f.close()

        print('Preparing embedding matrix')
        # prepare embedding matrix
        embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
        for word, i in tokenizer.word_index.items():
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
          else:
            print('Missing from GloVe: {}'.format(word))

        np.save(GLOVE_STORE, embedding_matrix)

      print('Loading GloVe')
      embedding_matrix = np.load(GLOVE_STORE + '.npy')

      print('Total number of null word embeddings:')
      print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

      embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
    else:
      embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)

    return embed
