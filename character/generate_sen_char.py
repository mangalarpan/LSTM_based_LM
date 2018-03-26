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
from nltk.tokenize import RegexpTokenizer
import numpy as np
import random
import sys
import io
import h5py
import os
import codecs
import math
import nltk
f = codecs.open("gut_train_char_half.txt", encoding='latin')
raw_data =f.read()
f.close()
text = raw_data.lower()

#print('corpus length:', len(text))

chars = sorted(list(set(text)))
#print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
#print('nb sequences:', len(sentences))

#print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
#print('Build model...')
model = Sequential()
model.add(CuDNNLSTM(256,input_shape=(maxlen, len(chars)),return_sequences=True))
#return_sequences=True,
model.add(CuDNNLSTM(256))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001)

# load the network weights

filename = "29lstm_char_weights_256.hdf5" #weight of model which is giving least loss
model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

start_index = random.randint(0, len(text) - maxlen - 1)
generated = ''
sentence = text[start_index: start_index + maxlen]
generated += sentence

for i in range(300):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

tokenizer = RegexpTokenizer(r'\w+')
sen = tokenizer.tokenize(generated)
sent = " ".join(sen[5:15])
print(sent)
