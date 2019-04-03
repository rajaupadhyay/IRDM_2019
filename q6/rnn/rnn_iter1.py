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
from glove_preprocessor import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pickle
import os
import zipfile
from collections import Counter

print(os.listdir("../input"))




training_f = open('../input/training-d/training_data.pickle', 'rb')
training = pickle.load(training_f)
training_f.close()
print('Loading')

testing_f = open('../input/testing/testing_data.pickle', 'rb')
test = pickle.load(testing_f)
testing_f.close()


tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])


training = (training[0], training[1], np_utils.to_categorical(training[2], 2))
test = (test[0], test[1], np_utils.to_categorical(test[2], 2))


# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1


USE_GLOVE = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512
MAX_EPOCHS = 10 # 10
MAX_LEN = 30 # 42
DP = 0.2 # 0.2
ACTIVATION = 'relu'
OPTIMIZER = 'adam'


# Pad the sequences to a max length of MAX_LEN
pad_sequence = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
# Apply sequence pad function to data
metaPadder = lambda data: (pad_sequence(data[0]), pad_sequence(data[1]), data[2])

training = metaPadder(training)
test = metaPadder(test)


print('Build model...')
print('Vocab size =', VOCAB)

embed = preprocessGlove()


translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

embSum = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

claim = Input(shape=(MAX_LEN,), dtype='int32')
clm = embed(claim)
clm = translate(clm)
clm = embSum(clm)
clm = BatchNormalization()(clm)

evidence = Input(shape=(MAX_LEN,), dtype='int32')
evdnce = embed(evidence)
evdnce = translate(evdnce)
evdnce = embSum(evdnce)
evdnce = BatchNormalization()(evdnce)

joint = concatenate([clm, evdnce])

joint = Dropout(DP)(joint)
for i in range(4):
  joint = Dense(3 * SENT_HIDDEN_SIZE, activation=ACTIVATION)(joint)
  joint = Dropout(DP)(joint)
  joint = BatchNormalization()(joint)

pred = Dense(2, activation='sigmoid')(joint)

model = Model(input=[claim, evidence], output=pred)
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')

model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, verbose=1)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
