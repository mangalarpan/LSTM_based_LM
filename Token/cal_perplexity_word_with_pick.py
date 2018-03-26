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

index_word = dict((word_index[key],key) for key in word_index)
#Merge all sequences, after each sentence add index of end token

MAX_SEQUENCE_LENGTH = 20

EMBEDDING_DIM = 300

# cut the text in semi-redundant sequences of maxlen characters
maxlen = MAX_SEQUENCE_LENGTH
step = 3

pickle_in = open("emb_mat.pickle","rb")
embedding_matrix = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("per_word_x.pickle","rb")
x = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("per_word_y.pickle","rb")
y = pickle.load(pickle_in)
pickle_in.close()

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(CuDNNLSTM(512,input_shape=(maxlen,EMBEDDING_DIM ),return_sequences=True))
#return_sequences=True,
model.add(CuDNNLSTM(512))
model.add(Dense(len(word_index)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001)
filename = "34lstm_word_weights_512.hdf5"
model.load_weights(filename)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

#calculate perplexity
prob = [0, 0, 0,0,0 ,0]
beta = [0.00001,0.0001,0.001,0.01,0.1,1]
length = x.shape[0] #len(sentences)
print("length is here ",length)
for j in range(length):
    trainx = np.reshape( x[j],(1,x[j].shape[0],x[j].shape[1]) ) 
    #trainy = np.reshape(y[j],(1,y[j].shape[0]))
    predictions = model.predict(trainx)
    #print(predictions,y[j])
    p = [0,0,0,0,0,0]
    for k in range(6):
        p[k]=(predictions[0][y[j]])/(beta[k]+1)
        p[k] = p[k] + beta[k]/(beta[k]*length)
        prob[k] = prob[k] + np.log(p[k])
    
    
for k in range(6):
    pp = math.exp(-prob[k]/length)
    print("Perplexity is ", pp,"when beta=",beta[k])


