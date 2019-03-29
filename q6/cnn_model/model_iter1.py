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
from keras.utils.np_utils import to_categorical


X_train_f = open('q6/data/train_test_learnt_embeddings/X_train.pickle', 'rb')
X_train = pickle.load(X_train_f)
X_train_f.close()

y_train_f = open('q6/data/train_test_learnt_embeddings/y_train.pickle', 'rb')
y_train = pickle.load(y_train_f)
y_train_f.close()

X_test_f = open('q6/data/train_test_learnt_embeddings/X_test.pickle', 'rb')
X_test = pickle.load(X_test_f)
X_test_f.close()

y_test_f = open('q6/data/train_test_learnt_embeddings/y_test.pickle', 'rb')
y_test = pickle.load(y_test_f)
y_test_f.close()


vocab_dict_f = open('q6/data/train_test_learnt_embeddings/vocab_dict.pickle', 'rb')
vocab_dict = pickle.load(vocab_dict_f)
vocab_dict_f.close()

print(len(X_test))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#
#
# embedding_dim = 256
# filter_sizes = [3, 4, 5]
# num_filters = 128
# std_drop = 0.5
#
# epochs = 15
# batch_size = 512
# vocabulary_size = len(vocab_dict)+1
# max_length = 500
#
# print("Creating Model...")
# inputs = Input(shape=(max_length,), dtype='int32')
# embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length)(inputs)
# reshape = Reshape((max_length, embedding_dim, 1))(embedding)
#
# # Kernel size specifies the size of the 2-D conv window
# # looking at 3 words at a time in the 1st layer, 4 in the 2nd ...
# # set padding to valid to ensure no padding
# conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
#             activation='relu')(reshape)
# conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
#             activation='relu')(reshape)
# conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
#             activation='relu')(reshape)
#
# # Pool size is the downscaling factor
# maxpool_0 = MaxPool2D(pool_size=(max_length-filter_sizes[0]+1, 1), strides=(2,2), padding='valid')(conv_0)
# maxpool_1 = MaxPool2D(pool_size=(max_length-filter_sizes[1]+1, 1), strides=(2,2), padding='valid')(conv_1)
# maxpool_2 = MaxPool2D(pool_size=(max_length-filter_sizes[2]+1, 1), strides=(2,2), padding='valid')(conv_2)
#
# concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
# flatten = Flatten()(concatenated_tensor)
# dropout = Dropout(std_drop)(flatten)
# output = Dense(units=2, activation='softmax')(dropout)
#
# model = Model(inputs=inputs, outputs=output)
#
# # checkpoint = ModelCheckpoint('model_flag_0.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
# # adam = Adam(lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# #print(model.summary())
# print("Training Model...")
# # model.fit(X_train_onehot, y_train_distribution, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[checkpoint],
# #      validation_data=(X_dev_onehot, y_dev_distribution))
#
# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
#
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
