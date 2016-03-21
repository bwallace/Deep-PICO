# -*- coding: utf-8 -*-

'''
Byron C. Wallace 

LSTM model for picking out groups (tokens) from abstracts. 
Annotated data is courtesy of Rodney Sumerscales. 

Sample use: 

    > import LSTM_extraction
    > LSTM_extraction.LSTM_exp() # assumes *-w2v.bin file exists!!!

Requires keras, sklearn and associated dependencies. 

notes to self:

    * run in python 2.x 
    * this is only groups at the moment (using LSTM)

@TODO 
    
    * implement BoW CRFs as a baseline 

'''





from __future__ import absolute_import
from __future__ import print_function

import sys
#import pdb
import copy 

import numpy as np
np.random.seed(1337)  # for reproducibility
import scipy as sp

import pandas as pd  

import gensim 
from gensim.models import Word2Vec

import keras
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding

from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

from keras.optimizers import Adam


from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

import nltk 

import matplotlib.pyplot as plt 
import seaborn as sns 

import sklearn
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.cross_validation import KFold

import parse_summerscales 
# @TODO Add this
#import distant_intervention_tag


#"PubMed-w2v.bin"
def load_trained_w2v_model(path="PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m 


def build_model(use_w2v, v_size, wv_dim, max_length, init_vectors=None):
    ''' build, compile and return model '''
    print("constructing model...")
    model = Sequential()
    # embedding layer; map token indices to vector representations

    if use_w2v:
        embedding_layer = Embedding(v_size, wv_diDm, weights=[init_vectors], input_length=max_length)
    else:
        print ("no initial embeddings!!")
        embedding_layer = Embedding(v_size, wv_dim)

    model.add(embedding_layer)
    model.add(LSTM(output_dim=128, return_sequences=False, inner_activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(max_length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    """
    # @TODO! tune
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")
    """
    print("model compiled.")
    return model 


def _get_init_vectors(vectorizer, wv, unknown_words_to_vecs):
    init_vectors = []

    for token_idx, t in enumerate(vectorizer.vocabulary):
        try:
            init_vectors.append(wv[t])
        except:
            init_vectors.append(unknown_words_to_vecs[t])
    init_vectors = np.vstack(init_vectors)

    return init_vectors

def _assemble_X_y_for_pmids(pmids_X_y_d, pmids, groups_map, max_size, vocab_map=None):
    X, y = [], []
    if not vocab_map:
        vocab_map = {}
    i = 0
    """
    for pmid in pmids:
        X_i_vec, X_i, y_i = pmids_X_y_d[pmid]

        for x_i in range(i, i + len(X_i_vec)):

            vocab_map[x_i] = groups_map[pmid]
            i+=1

        X.append(X_i)
        y.append(y_i)
    """

    for i, pmid in enumerate(pmids):
        X_i_vec, X_i, y_i = pmids_X_y_d[pmid]
        vocab_map[i] = pmid
        X.append(X_i)
        y.append(y_i)
    X = pad_sequences(X, maxlen=max_size)


   #     X.extend(X_i)
     #   y.extend(y_i)
   # X = np.vstack(X)
    y = np.vstack(y)
    return X, y, vocab_map
      
def _get_threshold_func(theta):
    def map_f(x):
        if x >= theta: 
            return 1
        else: 
            return -1

    vec_map_f = np.vectorize(map_f)
    return vec_map_f


def _get_tokens():
    pass 

def _lax_match(true_idx_star, true_tokens, pred_indices, pred_spans):
    # as per summerscales
    ignore_these = ["a", "an", "the", "of", "had", "group", "groups", "arm"]

    ### 
    # any overlap?
    overlapping_indices, overlapping_tokens = [], []
    for indices, tokens in zip(pred_indices, pred_spans): 
        overlapping_indices, overlapping_tokens = [], []
        for j, idx in enumerate(indices): 
            if (idx in true_idx_star) and (not tokens[j] in ignore_these):
                overlapping_indices.append(idx)
                overlapping_tokens = tokens[j]
                pred_span = indices 

        #overlapping_indices = [
        #    idx for j, idx in enumerate(indices) if idx in true_idx_star and 
        #        not tokens[j] in ignore_these]

        if len(overlapping_indices) > 0:
            break 

    if len(overlapping_indices) == 0:
        # no overlap
        return False 

    # here, overlapping_indices refers to the subset of indices
    # in the *predicted set* that matches the indices in the
    # specified true set
    return (overlapping_indices, overlapping_tokens, pred_span)

def _crf_evaluate_detection(ground_truth, predicted, X, vectorizer):
    '''
    Summerscales PhD thesis, 2013
    Page 100-101

    This is the approach used for evaluating detected mentions. A detected mention is
    considered a match for an annotated mention if they consist of the same set of words
    (ignoring “a”, “an”, “the”, “of”, “had”, “group(s)”, and “arm”) or if the detected
    mention overlaps the annotated one and the overlap is not a symbol or stop
    100 word. If a detected mention overlaps multiple annotated mentions, it is
    considered to be a false positive.
    '''
    #stop_words =
    # @TODO implement more forgiving metric.. plus let's
    # look at false negative/positives
    tps, fps = 0, 0

    for y_true, y_hat in zip(ground_truth, predicted):

        true_pos_seqs = _contiguous_pos_indices(y_true)
        print('true_pos_seqs: {}'.format(true_pos_seqs))
        pred_pos_seqs = _contiguous_pos_indices(y_hat)

        true_spans = _get_text_spans(X, true_pos_seqs, vectorizer)
        pred_spans = _get_text_spans(X, pred_pos_seqs, vectorizer)

        tp_overlapping_tokens = []
        fp_tokens = []
        # keep track of the indices already matched
        already_matched_indices = []
        for idx, true_pos_seq in enumerate(true_pos_seqs):
            matched = _lax_match(true_pos_seqs[idx], true_spans, pred_pos_seqs, pred_spans)
            if matched:
                # overlapping indices is the set of *target* indices that
                # match the predicted tokens

                overlapping_indices, overlapping_tokens, pred_span = matched

                if not pred_span in already_matched_indices:
                    already_matched_indices.append(pred_span)
                    #true_pos_overlapping.append((overlapping_indices, overlapping_tokens))
                    tps += 1
                    tp_overlapping_tokens.append(overlapping_tokens)
                else:
                    fp_tokens.append(overlapping_tokens)
                    fps += 1

    ###
    # now count up predictions that were not matched
    # with any true positives
    ###
    for idx, pred_pos_seq in enumerate(pred_pos_seqs):
        # then this sequence didn't match any of the
        # true_pos_seq entries!
        if not pred_pos_seq in already_matched_indices:
            fp_tokens.append(pred_spans[idx])
            fps += 1
    if true_pos_seqs is None:
        recall = 0
    elif len(true_pos_seqs) > 0:
        recall = float(tps) / float(len(true_pos_seqs))
    else:
        recall = 0
    precision = float(tps) / float(tps + fps) #float(len(pred_pos_seqs))
    #pdb.set_trace()
    '''
    # now precision!
    fps = 0
    pred_pos_overlapping = []
    for idx, pred_pos_seq in enumerate(pred_pos_seqs):
        matched = _lax_match(pred_pos_seqs[idx], pred_spans, true_pos_seqs, true_spans)
        if not matched:
            # because no match for this predicted positive!
            fps += 1
    precision = float(tps) / float()
    '''
   # pdb.set_trace()
    accuracy = accuracy_score(y_true, y_hat)
    return recall, precision, tp_overlapping_tokens, fp_tokens, accuracy


def _evaluate_detection(y_true, y_hat, X, vectorizer):
    print('Y true: {}'.format(y_true))
    print('Y hat: {}'.format(y_hat))
    '''
    Summerscales PhD thesis, 2013 
    Page 100-101

    This is the approach used for evaluating detected mentions. A detected mention is 
    considered a match for an annotated mention if they consist of the same set of words 
    (ignoring “a”, “an”, “the”, “of”, “had”, “group(s)”, and “arm”) or if the detected 
    mention overlaps the annotated one and the overlap is not a symbol or stop
    100 word. If a detected mention overlaps multiple annotated mentions, it is 
    considered to be a false positive.
    '''
    #stop_words = 
    # @TODO implement more forgiving metric.. plus let's 
    # look at false negative/positives

    true_pos_seqs = _contiguous_pos_indices(y_true)
    print('true_pos_seqs: {}'.format(true_pos_seqs))
    pred_pos_seqs = _contiguous_pos_indices(y_hat)

    true_spans = _get_text_spans(X, true_pos_seqs, vectorizer)
    pred_spans = _get_text_spans(X, pred_pos_seqs, vectorizer)

    tps, fps = 0, 0

    tp_overlapping_tokens = []
    fp_tokens = []
    # keep track of the indices already matched
    already_matched_indices = []
    for idx, true_pos_seq in enumerate(true_pos_seqs):
        matched = _lax_match(true_pos_seqs[idx], true_spans, pred_pos_seqs, pred_spans)
        if matched:
            # overlapping indices is the set of *target* indices that 
            # match the predicted tokens

            overlapping_indices, overlapping_tokens, pred_span = matched 

            if not pred_span in already_matched_indices:
                already_matched_indices.append(pred_span)
                #true_pos_overlapping.append((overlapping_indices, overlapping_tokens))
                tps += 1
                tp_overlapping_tokens.append(overlapping_tokens)
            else: 
                fp_tokens.append(overlapping_tokens)
                fps += 1
       
    ###
    # now count up predictions that were not matched
    # with any true positives
    ###
    for idx, pred_pos_seq in enumerate(pred_pos_seqs):
        # then this sequence didn't match any of the 
        # true_pos_seq entries!
        if not pred_pos_seq in already_matched_indices: 
            fp_tokens.append(pred_spans[idx])
            fps += 1
    if true_pos_seqs is None:
        recall = 0
    elif len(true_pos_seqs) > 0:
        recall = float(tps) / float(len(true_pos_seqs))
    else:
        recall = 0
    precision = float(tps) / float(tps + fps) #float(len(pred_pos_seqs))
    #pdb.set_trace()
    '''
    # now precision! 
    fps = 0
    pred_pos_overlapping = []
    for idx, pred_pos_seq in enumerate(pred_pos_seqs):
        matched = _lax_match(pred_pos_seqs[idx], pred_spans, true_pos_seqs, true_spans)
        if not matched:
            # because no match for this predicted positive!
            fps += 1
    precision = float(tps) / float()
    '''
   # pdb.set_trace()
    accuracy = accuracy_score(y_true, y_hat)
    return recall, precision, tp_overlapping_tokens, fp_tokens, accuracy
    


'''
        overlapping_indices = [idx for j, idx in enumerate(indices) if idx in true_pos_seqs[idx]]
'''
def _tune_theta(y, raw_preds, X, vectorizer):
    theta_vals = np.linspace(0,.25,200)
    best_theta = None 
    best_score = -np.inf
    for theta in theta_vals:
        vec_map_f = _get_threshold_func(theta)
        y_hat = vec_map_f(raw_preds)
        #cur_score = f1_score(y, y_hat)
        r, p, tp_overlapping_tokens, fp_tokens = _evaluate_detection(y, y_hat, X, vectorizer)

        cur_score = (2 * p * r) / (p + r)
        
        if cur_score >= best_score:
            best_score = cur_score
            best_theta = theta 

    return best_theta, best_score
    

def _contiguous_pos_indices(y):
    groups, cur_group = [], []
    last_y = None
    for idx, y_i in enumerate(list(y)):
        y_i = int(y_i)
        if y_i == last_y == 1:
            cur_group.append(idx)
        elif y_i == 1: 
            # then last_y was -1, but this is 1.
            cur_group = [idx]
        elif last_y == 1: 
            groups.append(cur_group)
            cur_group = []
        last_y = y_i
    
    if len(cur_group) > 0:
        groups.append(cur_group)
    return groups


def _get_text_spans(X, index_seqs, vectorizer):
    spans = []

    for idx_seq in index_seqs:
        cur_tokens = [vectorizer.vocabulary[X[idx]] for idx in idx_seq]
        spans.append(cur_tokens)
    return spans 

def _error_report(y_hat, y_true, vectorizer, X):
    true_pos_seqs = _contiguous_pos_indices(y_true) 
    pred_pos_seqs = _contiguous_pos_indices(y_hat)

    true_spans = _get_text_spans(X, true_pos_seqs, vectorizer)
    pred_spans = _get_text_spans(X, pred_pos_seqs, vectorizer)
    print("true spans: {}".format(true_spans))
    print('pred spans: {}'.format(pred_spans))

    return true_spans, pred_spans


def LSTM_exp2(wv=None, wv_dim=200, n_epochs=10, use_w2v=True,
                n_folds=5, distant_pretrain=False, model_name="LSTM_exp2.h5py"):
    

    if wv is None and use_w2v:
        print("loading embeddings...")
        wv = load_trained_w2v_model(path='wikipedia-pubmed-and-PMC-w2v.bin')
        print("ok!")
    max_l = 400

    # pmids_X_y_d maps pubmed identifiers to threeples of
    # tokens, embedded vectors and corresponding labels
    if distant_pretrain:
        pmids_X_y_d, vectorizer, unknown_words_to_vecs, X_DS_embedded, X_DS_tokens, y_DS = get_PMIDs_to_X_y(
                                    wv=wv, wv_dim=wv_dim, distant=True, n=5000)
        #X_DS_tokens = np.array([np.array(token_idx) for token_idx in X_DS_tokens])
        X_DS_tokens = np.vstack(X_DS_tokens)
        y_DS = np.array(y_DS)
    else:
        pmids_X_y_d, vectorizer, unknown_words_to_vecs, groups_map, pmids_dict = get_PMIDs_to_X_y(
                                    wv=wv, wv_dim=wv_dim, distant=False, max_length=max_l)

    init_vectors = None
    if use_w2v:
        init_vectors = _get_init_vectors(vectorizer, wv, unknown_words_to_vecs)

    v_size = len(vectorizer.vocabulary_)

    
    ''' train / test '''
    '''
        * CV on PMIDs 
        * 
    '''
    all_pmids = pmids_X_y_d.keys()
    n = len(all_pmids)

    kf = KFold(n, random_state=1337, shuffle=True, n_folds=n_folds)
    if distant_pretrain:
        DS_model = build_model(use_w2v, v_size, wv_dim, init_vectors)
        print("pretraining!")
        DS_model.fit(X_DS_tokens, y_DS, nb_epoch=n_epochs, verbose=2)
        DS_model_wts = DS_model.get_weights() 

    fold_metrics = []
    for fold_idx, (train, test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        train_pmids = [all_pmids[pmid_idx] for pmid_idx in train]
        test_pmids  = [all_pmids[pmid_idx] for pmid_idx in test]
        # sanity check
        assert(len(set(train_pmids).intersection(set(test_pmids)))) == 0

        train_X, train_y, _ = _assemble_X_y_for_pmids(pmids_X_y_d, train_pmids, groups_map, max_size=max_l)
        test_X, test_y, _vocab_map   = _assemble_X_y_for_pmids(pmids_X_y_d, test_pmids, groups_map, max_size=max_l)

        if distant_pretrain:
            model = keras.models.model_from_yaml(DS_model.to_yaml())
            model.set_weights(DS_model_wts)
        else:

            model = build_model(use_w2v, v_size, wv_dim, max_l, init_vectors)

        checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
        history = model.fit(train_X, train_y, batch_size=48, nb_epoch=n_epochs, callbacks=[checkpointer])
        train_preds = model.predict(train_X)
        #theta, score = _tune_theta(train_preds, train_y)

        '''
        pdb.set_trace()

        theta, score = _tune_theta(train_y, train_preds, train_X, vectorizer)
        # _evaluate_detection(test_y, binary_preds, test_X, vectorizer))
        print("best theta is: %s with score: %s" % (theta, score))

        preds = model.predict(test_X)
        vec_map_f = _get_threshold_func(theta)        
        binary_preds = [x[0] for x in vec_map_f(preds)]
        '''

        test_pred = model.predict_classes(test_X)
        predictions = model.predict(test_X)

        pred_list = list(test_pred)

        """

        pred_pos_indices = [idx for idx in range(len(binary_preds)) if binary_preds[idx]>0]
        intervention_preds = [vectorizer.vocabulary[test_X[j]] for j in pred_pos_indices]

        true_pos_indices = [idx for idx in range(test_y.shape[0]) if test_y[idx]>0]
        """
        g_i = []

        for i, pmid in enumerate(test_pmids):
            print(pmid)
            prob_predictions = predictions[i, :]

            idxs = np.argpartition(prob_predictions, -2)[-2:]
            sorted_idxs = idxs[np.argsort(prob_predictions[idxs])]

            vocab = groups_map[pmid]
            orig_text = pmids_dict[pmid][0]
            count = 0
            pred_words = []

            for i in range(sorted_idxs.shape[0]):
                x = sorted_idxs[i]

                if len(orig_text) < max_l and x - (max_l - len(orig_text)) > 0:
                     x = x - (max_l - len(orig_text))
                word = orig_text[x]
                pred_words.append(word)

                for v_word in vocab:


                    if word in v_word:
                        count += 1

            print(pred_words, vocab)
            """
            print(orig_text[x])

            count = 0
            for word in vocab:
                if x in word:
                    count += 1

            """
            if len(vocab) == 0:
                continue
            g_i1 = float(count)/float(len(vocab))
            g_i.append(g_i1)
        avg_g_i = float(0)

        for x in g_i:
            avg_g_i += x
        avg_g_i = avg_g_i/len(g_i)

        print('g_i: ', avg_g_i)

        """
        intervention_annotations = [vectorizer.vocabulary[test_X[j]] for j in true_pos_indices]

        p, r, tp_overlapping_tokens, fp_tokens, accuracy = _evaluate_detection(test_y, binary_preds, test_X, vectorizer)

        if p+r == 0:
            f1 = None 
        else:
            f1 = (2 * p * r) / (p + r)
        #fold_score = f1_score(test_y, binary_preds)
        print("fold %s. precision: %s; recall: %s; f1: %s; accuracy: %s" % (fold_idx, p, r, f1, accuracy))
        #pdb.set_trace()
        fold_metrics.append((p, r, f1))
        """


        sys.exit()
    return fold_metrics


def _get_distantly_lbled_tokens(n, wv, wv_dim):
    X_DS_embedded, y_DS = [], []
    # keep track of unique tokens in DS because 
    # we'll want to hand these off to our vectorizer
    tokens_DS = []
    unknown_words_to_vecs = {}
    tagged_pmids, tagged_abstracts, tokens_and_lbls, intervention_texts = \
                distant_intervention_tag.distantly_annotate(n=n)
    for abs_idx, abs_tokens_and_lbls in enumerate(tokens_and_lbls):
        for token_idx, token_and_lbl in enumerate(abs_tokens_and_lbls):
            t, lbl = token_and_lbl
            try:
                v = wv[t]
            except:
                if not t in unknown_words_to_vecs:
                    v = np.random.uniform(-1,1,wv_dim)
                    unknown_words_to_vecs[t] = v 
                v = unknown_words_to_vecs[t]
            X_DS_embedded.append(v)
            y_DS.append(lbl)
            tokens_DS.append(t)
            tokens_DS = list(set(tokens_DS))
    
    return tokens_and_lbls, X_DS_embedded, y_DS, tokens_DS, unknown_words_to_vecs


def get_PMIDs_to_X_y(wv, wv_dim, max_length=None, distant=False, n=200):
    
    unknown_words_to_vecs = {}
    tokens_DS = None 

    if distant:
        tokens_and_lbls, X_DS_embedded, y_DS, tokens_DS, unknown_words_to_vecs = \
                _get_distantly_lbled_tokens(n=n, wv=wv, wv_dim=wv_dim)

    # we pass tokens_DS -- the unique tokens in the DS
    # data -- to go into our vectorizer!
    """
    pmids_dict, pmids, sentences, lbls, vectorizer, groups_map = \
                parse_summerscales.get_tokens_and_lbls(
                        make_pmids_dict=True)
    """
    pmids_dict, pmids, abstracts, lbls, vectorizer, groups_map, one_hot, dicts = \
        parse_summerscales.get_tokens_and_lbls(
                make_pmids_dict=True, sen=True)
    ###
    # now loop through and get X_tokens representation!
    if distant:
        # really token_indices maybe more correct
        X_DS_tokens = []
        for abs_idx, abs_tokens_and_lbls in enumerate(tokens_and_lbls):
            for token_idx, token_and_lbl in enumerate(abs_tokens_and_lbls):
                t, lbl = token_and_lbl
                X_DS_tokens.append(vectorizer.vocabulary_[t])

    # see: https://github.com/fchollet/keras/issues/233
    # num_sentences x 1 x max_token_len x wv_dim
    # number of sequences x 1 x max number of tokens (padded to max len) x word vector size
   # num_sentences = len(sentences)
    #max_token_len = max([len(s) for s in sentences])

    #X_embedded = np.zeros((num_sentences, wv_dim))
    X_embedded, X_tokens = [], [] # here a sequence associated with each doc/abstract

    #unknown_words_to_vecs = {}
    pmids_to_X_y = {}

    for pmid in pmids_dict:
       # pmid_sentences, pmid_lbls = pmids_dict[pmid]
        abstract_tokens, abstract_output_labels, _ = pmids_dict[pmid]
        # for this sentence
        X_embedded = [] 
        X_tokens   = []
        y = []


        for w_i, word_token in enumerate(abstract_tokens):
            try:
                v = wv[word_token]
            except:
                # or maybe use 0s???
                if word_token not in unknown_words_to_vecs:
                    print("word '%s' not known!" % word_token)
                    v = np.random.uniform(-1,1,wv_dim)
                    unknown_words_to_vecs[word_token] = v

                v = unknown_words_to_vecs[word_token]
            X_embedded.append(v)
            X_tokens.append(vectorizer.vocabulary_[word_token])

        #pmids_to_X_y[pmid] = (np.vstack(X_embedded), np.vstack(X_tokens), np.hstack(y))

        if len(abstract_output_labels) > max_length:
            abstract_output_labels = abstract_output_labels[:max_length]
        elif len(abstract_output_labels) < max_length:
            padding = []
            for i in range(max_length - len(abstract_output_labels)):
                padding.append(0)
            abstract_output_labels = padding + abstract_output_labels
        assert len(abstract_output_labels) == max_length, 'Must be same size'

        pmids_to_X_y[pmid] = (X_embedded, X_tokens, abstract_output_labels)
        """
        for sent_idx, s in enumerate(pmid_sentences):
            for j, t in enumerate(s): 
                try:
                    v = wv[t]
                except:
                    # or maybe use 0s???
                    if not t in unknown_words_to_vecs:
                        print("word '%s' not known!" % t)
                        v = np.random.uniform(-1,1,wv_dim)
                        unknown_words_to_vecs[t] = v 
                    
                    v = unknown_words_to_vecs[t]

                X_embedded.append(v)
                X_tokens.append(vectorizer.vocabulary_[t])
            
            y.extend(pmid_lbls[sent_idx])

        pmids_to_X_y[pmid] = (np.vstack(X_embedded), np.vstack(X_tokens), np.hstack(y))
    """
    if distant:
        return pmids_to_X_y, vectorizer, unknown_words_to_vecs, X_DS_embedded, X_DS_tokens, y_DS

    return pmids_to_X_y, vectorizer, unknown_words_to_vecs, groups_map, pmids_dict
    


def get_X_y(wv, wv_dim, vectorizer=None, distant=False, n=None):

    pmids, sentences, lbls, vectorizer = [None]*4
    if distant: 
        #pmids, sentences, lbls, vectorizer = distant_intervention_tag.get_tokens_and_lbls(N=N)
        pmids, tagged_abstracts, tokens_and_lbls, intervention_texts = \
                distant_intervention_tag.distantly_annotate(n=n)
    else:
        pmids, sentences, lbls, vectorizer = parse_summerscales.get_tokens_and_lbls()

    #pdb.set_trace()

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
   
    cur_x_embedded, cur_x_tokens, cur_y, token_pmid_list = [], [], [], []
 
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
                print("%s not known!" % t)

                # or maybe use 0s???
                if not t in unknown_words_to_vecs:
                    v = np.random.uniform(-1,1,wv_dim)
                    unknown_words_to_vecs[t] = v 
                
                v = unknown_words_to_vecs[t]

            cur_x_embedded.append(v)
            cur_x_tokens.append(vectorizer.vocabulary_[t])
            token_pmid_list.append(cur_pmid)

        cur_y.extend(lbls[idx])


    X_embedded.append(np.vstack(cur_x_embedded))
    X_tokens.append(np.vstack(cur_x_tokens))
    y.append(np.array(cur_y))

    X_embedded = np.vstack(X_embedded)
    X_tokens   = np.vstack(X_tokens)
    y          = np.hstack(y)
    return X_embedded, X_tokens, y, vectorizer, unknown_words_to_vecs, token_pmid_list



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

'''
A detected mention is considered a match for an annotated mention if they 
consist of the same set of words (ignoring "a", "an", "the", "of", "group(s)",
and "arm") or if the detected mention overlaps the annotated 
one and the overlap is not a symbol or stop word. If a detected 
mention overlaps multiple annotated mentions, it is considered to be 
a false positive. If multiple detected mentions overlap the same annotated 
mention the detected mention with the most overlapping tokens (not counting 
symbols and stop words) is considered to be a true positive and the 
others are counted as false positives. Annotated mentions that do not 
match detected mentions are considered to be false negatives.
 -- summerscales, p100-101
'''




'''
*** antiquated -- use the extremely well-named LSTM_exp2, above. ***

note: this is for groups right now. the ranking performance is reasonable,
        if not great (~.7 AUC). 
'''
def LSTM_exp(wv=None, wv_dim=200, p_test=.25, n_epochs=10, use_w2v=True):
    if wv is None and use_w2v:
        print("loading embeddings...")
        wv = load_trained_w2v_model() 
        print("ok!")


    X_embedded, X_tokens, y, vectorizer, unknown_words_to_vecs, pmids = get_X_y(
            wv=wv, wv_dim=wv_dim)

    v_size = len(vectorizer.vocabulary_)

    init_vectors = []
    if use_w2v:
        #for t in vectorizer.vocabulary_:
        for token_idx, t in enumerate(vectorizer.vocabulary):
            try:
                init_vectors.append(wv[t])
            except:
                init_vectors.append(unknown_words_to_vecs[t])
        init_vectors = np.vstack(init_vectors)

   
    ''' build model; this should probably be factored out! '''
    print("constructing model...")
    model = Sequential()
    # embedding layer; map token indices to vector representations

    if use_w2v:
        embedding_layer = Embedding(v_size, wv_dim, weights=[init_vectors])
    else:
        print ("no initial embeddings!!")
        embedding_layer = Embedding(v_size, wv_dim)

    model.add(embedding_layer)

    model.add(LSTM(output_dim=128, 
        activation='sigmoid', inner_activation='hard_sigmoid'))
    
    # @TODO! tune
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) 

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")
    print("model compiled.")
  

    ''' train / test '''
    # @TODO! be sure to split at start of a pmid 
    #   (i.e., do not split midway through and abstract!)
    N = X_tokens.shape[0]
    test_n = int(p_test*N)

    X_tokens_train = X_tokens[:-test_n]
    X_tokens_test  = X_tokens[-test_n:]
    y_train  = y[:-test_n]
    y_test   = y[-test_n:]
    pmids_train = pmids[:-test_n]
    pmids_test  = pmids[-test_n:]

    print("training!")
    model.fit(X_tokens_train, y_train, nb_epoch=n_epochs)

    ''' evaluation '''
    print("ok. predicting...")
    preds = model.predict(X_tokens_test)

    fpr, tpr, thresholds = roc_curve(y_test, preds)
    cur_auc = auc(fpr, tpr)
    print("auc: %s" % cur_auc)

    ### note to self: to inspect features you can do something like:
    ### words = [vectorizer.vocabulary[j] for j in X_tokens_test[180:200]]
    return model, preds, y_test

