from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import parse_summerscales
from sklearn.cross_validation import KFold
from gensim.models import Word2Vec
import LSTM_extraction
import sys
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import getopt
import numpy as np

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM


def abstract2features(pmid_dict, pmids, vectorizer, one_hot, dicts,  w2v=None):

    X = []
    y = []
    pos_dict_len = len(dicts['pos'])
    chunk_dict_len = len(dicts['chunk'])
    ne_dict_len = len(dicts['ne'])

    for pmid in pmids:
        abstract, labels, tagged_abstract = pmid_dict[pmid]

        x = np.zeros((len(abstract) , 400 + pos_dict_len + chunk_dict_len + ne_dict_len))
        previous_w2v = None
        previous_pos_tags = None
        for i, (word, label, tags) in enumerate(zip(abstract, labels, tagged_abstract)):

            if previous_w2v:
                x[i, :200] = previous_w2v

            if previous_pos_tags:
                x[i, 400:pos_dict_len] = pos_tag
                x[i, 400 + pos_dict_len + 1:chunk_dict_len] = chunk_tag
                x[i, 400 + pos_dict_len + chunk_dict_len + 1:] = ne_tag

            pos_tag, chunk_tag, ne_tag = tags
            one_hot_pos = np.zeros(pos_dict_len)
            one_hot_chunk = np.zeros(chunk_dict_len)
            one_hot_ne = np.zeros(ne_dict_len)

            one_hot_pos[pos_tag] = 1
            one_hot_chunk[chunk_tag] = 1
            one_hot_ne[ne_tag] = 1

            previous_pos_tags = (one_hot_pos, one_hot_chunk, one_hot_ne)

            # Add the POS one hot vectors as features
            x[i, 400:pos_dict_len] = pos_tag
            x[i, 400 + pos_dict_len + 1:chunk_dict_len] = chunk_tag
            x[i, 400 + pos_dict_len + chunk_dict_len + 1:] = ne_tag


            if w2v:
                try:
                    w2v_word = w2v[word]
                except:
                    w2v_word = np.zeros(200)
                previous_w2v = w2v_word
                x[i, 200 + 1: 400] = w2v_word
                X.append(x)
                y.append(np.array(labels))


    return X, y



def pystruct_crf(w2v):

    pmids_dict, pmids, abstracts, lbls, vectorizer, groups_map, one_hot, dicts = \
        parse_summerscales.get_tokens_and_lbls(
                make_pmids_dict=True, sen=True)


    """
        Create model
    """
    model = ChainCRF(directed=False)
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=30)

    all_pmids = pmids_dict.keys()
    n = len(all_pmids)
    n_folds = 5
    kf = KFold(n, random_state=1337, shuffle=True, n_folds=n_folds)
    fold_gi = []


    for fold_idx, (train, test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        train_pmids = [all_pmids[pmid_idx] for pmid_idx in train]
        test_pmids  = [all_pmids[pmid_idx] for pmid_idx in test]
        print('loading data...')
        train_x, train_y = abstract2features(pmids_dict, train_pmids, vectorizer, one_hot, dicts,  w2v)
        test_x, test_y = abstract2features(pmids_dict, test_pmids, vectorizer, one_hot, dicts, w2v)

        print('loaded data...')
        print 'training...'
        ssvm.fit(train_x, train_y)
        print ssvm.score(test_x, test_y)

        for i, (pmid, x, y) in enumerate(zip(test_pmids, test_x, test_y)):
            abstract_words, _, _= pmids_dict[pmid]

            print(pmid)

            # predict() takes in a list returns another list
            prediction = ssvm.predict([x]).pop(0)

            predicted = ''
            output = ''

            if len(prediction) > 0:

                for p in prediction:
                    if p == 1:
                        print "word: {}".format(abstract_words[p])
                        if n == 0:
                            predicted += abstract_words[p]
                        else:
                            predicted += ' ' + abstract_words[p]

                if not predicted == '':
                    output = 'predicted: {}'.format(predicted)
                else:
                    output = 'Predicted nothing!'
            else:
                output = 'Predicted nothing!'
            print output


def run():
    w2v = True
    l1 = 1
    l2 = 1e-3
    iters = 200
    wiki = True

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'w:i:c:l:', ['w2v=', 'iters=', 'l1=', 'l2=', 'wiki=','pystruct='])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-w', '--w2v'):
            option = int(arg)

            if option == 1:
                w2v = True
        elif opt in ('-i', '--iters'):
            iters = int(arg)
        elif opt in ('-c', '--l1'):
            l1 = float(arg)
        elif opt in ('-l', '--l2'):
            l2 = float(l2)
        elif opt == '--wiki':
            option = int(arg)


            if option == 0:
                wiki = False
        elif opt == '--pystruct':
            option = int(arg)

            if option == 0:
                is_pystruct = False
        else:
            sys.exit(2)
    if w2v:
        print('Loading word2vec model...')

        if wiki:
            word2vec_model = 'wikipedia-pubmed-and-PMC-w2v.bin'
        else:
            word2vec_model = 'PubMed-w2v.bin'
        w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)
        print('Loaded word2vec model')
    else:
        w2v = None
    pystruct_crf(w2v)

def main():
    run()
if __name__ == '__main__':
    main()