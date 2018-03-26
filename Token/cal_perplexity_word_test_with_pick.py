from __future__ import print_function
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM,CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import LambdaCallback
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import numpy as np
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import random
import sys
import io
import h5py
import os
import codecs
import nltk
import math
import pickle

pickle_in = open("word_dic.pickle","rb")
word_index = pickle.load(pickle_in)
pickle_in.close()
"""
f = codecs.open("gut_dev_word.txt", encoding='latin')
raw_data =f.read()
f.close()

raw_data = raw_data.lower()

texts = nltk.word_tokenize(raw_data)
print("number of words in corpus:{}",(len(texts)) )
"""
index_word = dict((word_index[key],key) for key in word_index)

MAX_SEQUENCE_LENGTH = 20

EMBEDDING_DIM = 300

# cut the text in semi-redundant sequences of maxlen characters
maxlen = MAX_SEQUENCE_LENGTH
step = 3

pickle_in = open("emb_mat_test.pickle","rb")
embedding_matrix = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("per_word_x_test.pickle","rb")
x = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("per_word_y_test.pickle","rb")
y = pickle.load(pickle_in)
pickle_in.close()

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(CuDNNLSTM(256,input_shape=(maxlen,EMBEDDING_DIM ),return_sequences=True))
#return_sequences=True,
model.add(CuDNNLSTM(256))
model.add(Dense(len(word_index)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001)
filename = "49lstm_word_weights_256.hdf5"
model.load_weights(filename)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

#calculate perplexity
prob = 0
beta = 0.00001
length = x.shape[0] #len(sentences)
print("length is ",length)
for j in range(length):
    trainx = np.reshape( x[j],(1,x[j].shape[0],x[j].shape[1]) ) 
    #trainy = np.reshape(y[j],(1,y[j].shape[0]))
    predictions = model.predict(trainx)
    p=(predictions[0][y[j]])/(beta+1)
    p = p + beta/(beta*length)
    prob = prob + np.log(p)
    
pp = math.exp(-prob/length)
print("Perplexity is ", pp)


