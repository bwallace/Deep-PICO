__author__ = 'zhangye'
"""
    Modified by Eric Rincon
"""
#classify sentences
dir = "abstracts2_sen/"
import os
import pickle

import numpy as np
from geniatagger import *
from pymedtermino import *
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score,recall_score
import BeautifulSoup as bs
import drugbank
import numpy

from normalize import normalize

np.random.seed(1337)  # for reproducibility

tagger = GeniaTagger('/home1/03186/ericr/code/geniatagger-3.0.1/geniatagger')
drug = drugbank.Drugbank()
sen_to_doc = {}
sen_counts = {}
from sklearn.cross_validation import train_test_split


def pos_to_string(sentence):    #convert a sentence into a sequence of POS tagger
    result = tagger.parse(sentence)
    return ' '.join([r[2] for r in result])


def preprocess_sentence(sentence):
    normalize(sentence, drug)

def tokenizer(input):
    return [t[0] for t in tagger.parse(input)]

def doc_to_sen(x,index_map):   #map document indices to sentence indices
    sentence_index = []


    for i in x:
        sentence_index += index_map[i]


    return sentence_index
def preprocess(to_train=False, abstracts=None, abstracts_labels=None, abstracts_file_names=None, count_vect=None, group_counts=None):
    #print tagger.parse('Twenty patients undergoing liver surgery were randomly assigned\
         #   to IPM with INTEGER MEASURE_UNIT (15IPM) or 30 (30IPM) minutes ischemic intervals.')
    file_dict = {}
    global sen_to_doc
    global sen_counts
    sen_counts = group_counts

    if to_train:
        for file in os.listdir(dir):
            if(not file.endswith('.txt')): continue
            name_key = file.split('.')[0]
            cur = open(dir+file,'rb')
            content_value = []
            label_value = []
            for c in cur:
                #normalize strings
                c = c.decode('unicode-escape')
                c = c.encode('ascii','ignore')
                content_value.append(c.strip()[:-1])
                label_value.append(int(c.split('\t')[-1].strip()))
            file_dict[name_key] = (content_value,label_value)
    else:
        for abstract, sentence_labels, abstracts_file_name in zip(abstracts, abstracts_labels, abstracts_file_names):
            content_value = []
            label_value = []

            for sentence, sentence_label in zip(abstract, sentence_labels):
                #
                # normalize strings
                #print(sentence, sentence_label)
                sentence = sentence.decode('unicode-escape')
                sentence = sentence.encode('ascii','ignore')
                content_value.append(sentence.strip()[:-1])
                label_value.append(sentence_label)
                sen_to_doc.update({})
            file_dict[abstracts_file_name] = (content_value,label_value)

    file_names = file_dict.keys()
    print str(len(file_names)) + " abstracts"
    all_sentences = []
    all_labels = []
    position_feature =[]
    len_sen = []
    #map document index to sentence index in feature matrix
    #key is the document index
    #value is the list of sentence indices
    #extract POS features
    index_map = {}
    fea_to_abs = {}   #key is the index in the feature matrix, and value is the (name_key, sentence index)
    start_index = 0


    pos = []
    #all_len = []
    prev_index = []   #index of previous sentence
    next_index = []   #index of next sentence
    num_sentence = 0
    for i, name in enumerate(file_names):
        print name

        all_sentences += [normalize(t,drug) for t in file_dict[name][0]]   #normalize each sentence
        all_labels += file_dict[name][1]
        if not to_train:
            sentence_counts = sen_counts[name]
        print len(file_dict[name][0])
        print '\n\n\n'
        for j in range(len(file_dict[name][0])):
            #print('position feature: ', j)
            position_feature.append(j)    #extract position features

        index_map[i] = range(start_index,len(file_dict[name][0])+start_index)

        for i, k in enumerate(range(start_index,len(file_dict[name][0])+start_index)):
            fea_to_abs[k] = (name,k-start_index)
            if not to_train:
                sen_counts[k] = sentence_counts[i]


        start_index += len(file_dict[name][0])
        num_sentence += len(file_dict[name][0])
    print sen_counts

    #k = 1
    #extract content features of previous k sentences and next k sentences
    #k = 2
    all_prev = []
    all_next = []
    all_len = []
    all_prev_prev = []
    all_next_next = []
    for i, name in enumerate(file_names):
        sentences = file_dict[name][0]
        num_sen = len(sentences)
        #print name
        for j in range(num_sen):
            if(j==0):
                #print name
                all_prev.append("none")
                all_next.append(sentences[j+1])
                all_prev_prev.append("none")
                all_next_next.append(sentences[j+2])
            elif(j==num_sen-1):
                all_next.append("none")
                all_prev.append(sentences[j-1])
                all_next_next.append("none")
                all_prev_prev.append(sentences[j-2])
            elif(j==1):
                all_prev.append(sentences[0])
                all_next.append(sentences[2])
                all_prev_prev.append("none")
                all_next_next.append(sentences[3])
            elif(j==num_sen-2):
                all_next.append(sentences[j+1])
                all_prev.append(sentences[j-1])
                all_prev_prev.append(sentences[j-2])
                all_next_next.append("none")
            else:
                all_next.append(sentences[j+1])
                all_prev.append(sentences[j-1])
                all_next_next.append(sentences[j+2])
                all_prev_prev.append(sentences[j-2])
            if(len(sentences[j])>500): print "sentence too long:" + name + ":" + str(len(sentences[j]))


    if to_train:
        count_vect = HashingVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b')
    all_labels = np.array(all_labels)
    print "count vectorizer..."

    if to_train:
        X_train_counts = count_vect.fit_transform(all_sentences)
        X_train_prev = count_vect.fit_transform(all_next)
        X_train_next = count_vect.fit_transform(all_prev)
    else:
        X_train_counts = count_vect.transform(all_sentences)
        X_train_prev = count_vect.transform(all_next)
        X_train_next = count_vect.transform(all_prev)
    #X_train_next_next = count_vect.fit_transform(all_next_next)
    #X_train_prev_prev = count_vect.fit_transform(all_prev_prev)
    print "count vectorizer finish..."

    #combine features
    features = hstack([X_train_counts,np.array(position_feature).reshape(-1,1)])
    features = hstack([features,X_train_next])
    features = hstack([features,X_train_prev])

    if to_train:
        return features, all_labels, file_names, index_map, file_dict, fea_to_abs, count_vect
    else:
        return features, all_labels, file_names, index_map, file_dict, fea_to_abs, sen_counts

def setup_train_test(features, all_labels, file_names, index_map, test_size=.2):


    indices = np.arange(len(file_names))
    idx_train, idx_test = train_test_split(indices, test_size=test_size)
    train_index = doc_to_sen(idx_train, index_map)
    test_index = doc_to_sen(idx_test, index_map)
    train_data = features.tocsr()[train_index, :]
    train_label = all_labels[train_index]
    test_data = features.tocsr()[test_index,:]
    test_label = all_labels[test_index]

    return train_data, train_label, test_data, test_label, idx_train, idx_test, test_index

def train():
    features, all_labels, file_names, index_map, file_dict, fea_to_abs, count_vec = preprocess(to_train=True)
    clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=250, alpha=.0001)
    train_data, train_label, test_data, test_label, idx_train, idx_test, test_index = setup_train_test(features, all_labels, file_names, index_map)


    print('Fitting classifier to data...')
    clf.fit(train_data,train_label)
    print('Fitted classifier to data...')

    predictions = test(clf, index_map, features, idx_test, test_data, file_dict, fea_to_abs, test_label, test_index)

    return clf, count_vec

def test(clf, index_map, features, idx_test, test_data, file_dict, fea_to_abs, test_label, test_index, groups_dict=None,
         counts=None):
    FP = open('fp.txt','wb')
    FN = open('fn.txt','wb')
    test_label[test_label == -1] = 0
    test_label = numpy.float32(test_label)
    test_score = clf.decision_function(test_data)
    g_i = []
    sorted_index_test = []
    prediction = []
    index_to_sen = {}
    sentence_list = []
    for t in idx_test:
        if groups_dict:
            n_groups_abstract = groups_dict[fea_to_abs[t][0]]
        print fea_to_abs[t]
        sen_index = doc_to_sen([t],index_map)
        if groups_dict:
            abs_name = fea_to_abs[t]
            print abs_name[0]
            count = groups_dict[abs_name[0]]
            sentence_counts = counts[abs_name[0]]

            print sentence_counts
        cur_test_score = clf.decision_function(features.tocsr()[sen_index,:])
        temp = [i[0] for i in sorted(enumerate(list(cur_test_score)),key=lambda x:x[1],reverse=True)]
        sorted_index = np.zeros(len(temp))
        for i, q in enumerate(temp): sorted_index[q] = i
        sorted_index_test += list(sorted_index)
        pred_abs = np.zeros(len(temp))
        pred_abs[temp[0]] = 1
        index_one = list(pred_abs).index(1)
        if groups_dict:
            print sentence_counts[index_one]
            print count
            if count == 0:
                g_i.append(0)
            else:
                print "sentence count: {}".format(sentence_counts[index_one])
                print 'abstract count: {}'.format(count)
                g_i.append(float(sentence_counts[index_one])/float(count))
        prediction += list(pred_abs)
        indices = [i for i, x in enumerate(pred_abs) if x == 1]

        for index in indices:
            sentence_list.append(sen_index[index])

    abs_sen = [fea_to_abs[t] for t in np.array(test_index)[np.logical_and(prediction==1,test_label==0)]]
    map(lambda x: FP.write(x[0]+"\t"+file_dict[x[0]][0][x[1]]+"\n"), abs_sen)
    abs_sen_1 = [fea_to_abs[t] for t in np.array(test_index)[np.logical_and(prediction==0,test_label==1)]]
    map(lambda x: FN.write(x[0]+"\t"+file_dict[x[0]][0][x[1]]+"\n"), abs_sen_1)
    f1 = f1_score(test_label, prediction)
    print('F1: {}'.format(f1))
    print("Precision: {}".format(precision_score(test_label,prediction)))
    print("Recall: {}".format(recall_score(test_label,prediction)))

    avg_gi = float(0)

    if groups_dict:
        for g in g_i:
            avg_gi+=g
        print float(avg_gi)/float(len(g_i))
    print g_i

    return prediction
    FP.close()
    FN.close()
