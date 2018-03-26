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

f = codecs.open("gut_dev_word.txt", encoding='latin')
raw_data =f.read()
f.close()

raw_data = raw_data.lower()

texts = nltk.word_tokenize(raw_data)
print("number of words in corpus:{}",(len(texts)) )

index_word = dict((word_index[key],key) for key in word_index)
#Merge all sequences, after each sentence add index of end token
data = []
for word in texts:
    if word in word_index:
        data = data + [word_index[word]]
    else:
        data = data + [word_index['<unk>']]
    

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


pickle_out = open("emb_mat.pickle","wb")
pickle.dump(embedding_matrix, pickle_out)
pickle_out.close()
pickle_out = open("per_word_x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()
pickle_out = open("per_word_y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
print("done")

