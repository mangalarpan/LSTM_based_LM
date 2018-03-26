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
from keras.callbacks import CSVLogger
import random
import sys
import io
import h5py
import os
import codecs
import nltk
import pickle

f = codecs.open("gut_train_word.txt", encoding='latin')
raw_data =f.read()
f.close()

raw_data = raw_data.lower()

texts = nltk.sent_tokenize(raw_data)
print("number of sentences in corpus:{}",(len(texts)) )

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) #list of sequences, each list contain index of words present in that sentence

word_index = tokenizer.word_index  #Dictionary of word map to their index
word_index['<unk>'] = len(word_index)+1 #add start token to the dictionary

#save dictionary
pickle_out = open("word_dic.pickle","wb")
pickle.dump(word_index, pickle_out)
pickle_out.close()

#word_index['<end>'] = len(word_index)+1 #add start token to the dictionary
print('Found %s unique tokens.',len(word_index))

index_word = dict((word_index[key],key) for key in word_index)
#Merge all sequences, after each sentence add index of end token
data = []
for seq in sequences:
    data = data + seq

MAX_SEQUENCE_LENGTH = 20

embeddings_index = {}
f = open(os.path.join("glove.6B", 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index)) #400000 words
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM)) #each word vector is of 100 dimension
count = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        count = count+1
#out of 5874 unique words, embedding of 5611 words are present
print(embedding_matrix[4],count)
        
# cut the text in semi-redundant sequences of maxlen characters
maxlen = MAX_SEQUENCE_LENGTH
step = 3
sentences = []
next_chars = []
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i: i + maxlen])
    next_chars.append(data[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, EMBEDDING_DIM), dtype='float32')
y = np.zeros(len(sentences), dtype='int')
#print(next_chars)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t] = embedding_matrix[word]
    y[i] = next_chars[i]-1


print('Build model...')
model = Sequential()
model.add(CuDNNLSTM(256,input_shape=(maxlen,EMBEDDING_DIM ),return_sequences=True))
#return_sequences=True,
model.add(CuDNNLSTM(256))
model.add(Dense(len(word_index)))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    path ="" + str(epoch)
    model.save_weights(path + 'lstm_word_weights_256.hdf5')
    

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
csv_logger = CSVLogger('training_word_256.log')
model.fit(x, y, batch_size=128, epochs=50,callbacks=[print_callback,csv_logger])
model.save('lstm_word_model_256.h5')
