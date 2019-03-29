import pandas as pd
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import np_utils
import random
from collections import OrderedDict
import numpy as np
import re
import pickle
from keras.models import load_model
from keras import layers
from keras.utils.np_utils import to_categorical

X_train_f = open('q6/data/embedding_train_test/X_train.pickle', 'rb')
X_train = pickle.load(X_train_f)
X_train_f.close()

y_train_f = open('q6/data/embedding_train_test/y_train.pickle', 'rb')
y_train = pickle.load(y_train_f)
y_train_f.close()

X_test_f = open('q6/data/embedding_train_test/X_test.pickle', 'rb')
X_test = pickle.load(X_test_f)
X_test_f.close()

y_test_f = open('q6/data/embedding_train_test/y_test.pickle', 'rb')
y_test = pickle.load(y_test_f)
y_test_f.close()


print(len(X_train))
print(len(X_test))


# model = Sequential()
# model.add(layers.Dense(10, input_dim=600, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(X_train, y_train,
#                     epochs=100,
#                     verbose=False,
#                     batch_size=128)
#
#
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
