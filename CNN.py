from gensim.models import Doc2Vec

from __future__ import absolute_import
from __future__ import print_function

import sys
import pdb


import numpy as np
np.random.seed(1337)  # for reproducibility
import scipy as sp

import pandas as pd  

'''
import sklearn 
from sklearn import manifold # for TSNE
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation as cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
'''

import gensim 
from gensim.models import Word2Vec

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

import nltk 

import matplotlib.pyplot as plt 
import seaborn as sns 

import parse_summerscales 

def load_trained_w2v_model(path="PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m 


def LSTM_exp(wv=None, wv_dim=200):
    if wv is None:
        wv = load_trained_w2v_model() 

    X_embedded, X_tokens, y, vectorizer, unknown_words_to_vecs = get_X_y(wv=wv, wv_dim=wv_dim)

    model = Sequential()

    v_size = len(vectorizer.vocabulary_)

    init_vectors = []
    for t in v.vocabulary_:
        try:
            init_vectors.append(wv[t])
        except:
            init_vectors.append(unknown_words_to_vecs[t])

    embedding_layer = Embedding(v_size, wv_dim, weights=[init_vectors])
    model.add(embedding_layer)

    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) 

#def build_model(vectorizer, wv, wv_dim=200):



def get_X_y(wv, wv_dim):

    pmids, sentences, lbls, vectorizer = parse_summerscales.get_tokens_and_lbls()


    # see: https://github.com/fchollet/keras/issues/233
    # num_sentences x 1 x max_token_len x wv_dim
    # number of sequences x 1 x max number of tokens (padded to max len) x word vector size
    num_sentences = len(sentences)
    #max_token_len = max([len(s) for s in sentences])

    #X_embedded = np.zeros((num_sentences, wv_dim))
    X_embedded, X_tokens = [], [] # here a sequence associated with each doc/abstract
    y = []

    #X_tokens = []
    cur_pmid = pmids[0]
    cur_x_embedded, cur_x_tokens, cur_y = [], [], []
 
    unknown_words_to_vecs = {}

    for idx, s in enumerate(sentences):
        if cur_pmid != pmids[idx]:
            X_embedded.append(np.vstack(cur_x_embedded))
            X_tokens.append(np.vstack(cur_x_tokens))
            y.append(np.array(cur_y))
            cur_x_embedded, cur_x_tokens, cur_y = [], [], []
            cur_pmid = pmids[idx]
        
        for j, t in enumerate(s): 
            try:
                v = wv[t]
            except:
                # or maybe use 0s???
                if not t in unknown_words_to_vecs:
                    v = np.random.uniform(-1,1,wv_dim)
                    unknown_words_to_vecs[t] = v 
                
                v = unknown_words_to_vecs[t]

            cur_x_embedded.append(v)
            cur_x_tokens.append(vectorizer.vocabulary_[t])
            

        cur_y.extend(lbls[idx])

            #cur_x_tokens.append(v)

    X_embedded.append(np.vstack(cur_x_embedded))
    X_tokens.append(np.vstack(cur_x_tokens))
    y.append(np.array(cur_y))

    return X_embedded, X_tokens, y, vectorizer, unknown_words_to_vecs





def preprocess_texts(texts, m, dim=200):

    for text in texts: 
        tokenized_text = nltk.word_tokenize(text)
        for t in tokenized_text: 
            try:
                v = m[t]
            except:
                # or maybe use 0s???
                v = np.random.uniform(-1,1,dim)


def setup_model(vocab, X):

    model = Sequential()



### ignore this for now.
def load_trained_d2v_model(path="Doc2Vec/400_pvdbow_doc2vec.d2v"):
    ''' @TODO swap in MEDLINE trained variant '''
    m = Doc2Vec.load(path)
    return m


### @TODO revisit this for loading up PMC + MEDLINE model!
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs




