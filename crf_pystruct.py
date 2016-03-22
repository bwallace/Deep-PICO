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


# i: postion
# word_i: index of where in abstract it is
def get_features(word_i, i, abstract, tagged_sentences, feature_dict, type, shallow_parse, w2v, w2v_size=128):
    postion = None
    if type == 'before':
        postion = '-'
    elif type == 'after':
        postion = '+'
    else:
        postion = ''

    if shallow_parse:
        word = abstract[word_i]

        pos, chunk, ne = tagged_sentences[word_i]

        feature_dict['word[{}{}]'.format(postion, i)] = abstract[word_i]
        feature_dict['pos[{}{}]'.format(postion, i)] = pos
        feature_dict['chunk[{}{}]'.format(postion, i)] = chunk
        feature_dict['ne[{}{}]'.format(postion, i)] = ne
        feature_dict['isupper[{}{}]'.format(postion, i)] = word.isupper()
        feature_dict['istitle[{}{}]'.format(postion, i)] = word.istitle()

        """
            R.L. Summerscales. Automatic summarization of clinical abstracts for evidence-based medicine.
            Ph.D. Thesis, Illinois Institute of Technology, 2013
            Numeric. Features that capture the characteristics of the token if it is a number.
            Is the number a percentage, integer, or floating point value?
            Is the number negative?
            Is it a small integer (< 10)?
        """

        is_digit = word.isdigit()

        if is_digit:
            int_from_string = int(word)
            int_is_neg = False
            int_is_small = False

            if int_from_string < 0:
                int_is_neg = True

            if int_from_string < 10:
                int_is_small = True

            feature_dict['isdigit[{}{}]'.format(postion, i)] = word.isdigit()
            feature_dict['isneg[{}{}]'.format(postion, i)] = int_is_neg
            feature_dict['issmall[{}{}]'.format(postion, i)] = int_is_small


        # @TODO Add closest parent verb and that third thing
        """
        Syntactic context. These features capture the syntactic context of the token in question.
        Is the token inside parentheses?
        The closest parent verb in the parse tree. Starting at the token in question travel up the parse tree until a verb
          phrase is reached. Then return the main verb (the first word) in the verb phrase.
        Dependency features. These features are based on a collapsed typed de- pendency parse of the sentence.
        They consist of the token and semantic
        66

        features for each governor and dependent token and the type of relation- ship. For the number classifiers,
        only features for the governor tokens are used; any dependent tokens of number are ignored.
        """

        is_token_paren = False

        if i > 0 and i < len(abstract):
            if abstract[word_i-1] == '(' and abstract[word_i + 1] == ')':
                is_token_paren = True

        feature_dict['paren'] = is_token_paren

    if w2v:
        if word_i > i:
            word = abstract[word_i]

            try:
                w2v_word = w2v[word]
                found_word = True
            except:
                w2v_word = None
                found_word = False

            for n in range(w2v_size):
                if found_word:
                    feature_dict["w2v[{}{}][{}]".format(postion, i, n)] = w2v_word[n]
                else:
                    feature_dict["w2v[{}{}][{}]".format(postion, i, n)] = 0
            if i > 0:
                try:
                    cosine_simil = w2v.similarity(abstract[word_i-1], abstract[word_i])
                except:
                    cosine_simil = 0
                feature_dict['cos'] = cosine_simil

    return feature_dict


def get_feature_window(n_before, n_after, abstract, tagged_sentences, feature_dict, word_i, shallow_parse, w2v):

    for i in range(1, n_before + 1):

        if word_i - i >= 0:
            feature_dict = get_features(word_i - i, i, abstract, tagged_sentences, feature_dict, 'before', shallow_parse, w2v)

    for i in range(1, n_after + 1):
        if word_i + i <= (len(abstract) - 1):
            feature_dict = get_features(word_i + i, i, abstract, tagged_sentences, feature_dict, 'after', shallow_parse, w2v)


    return feature_dict


def get_w2v(n_before, n_after, abstract, feature_dict, word_i, w2v, w2v_size):

    for i in range(n_before):
        found_word = True

        if word_i > i:
            word = abstract[word_i - i]

            try:
                w2v_word = w2v[word]
            except:
                w2v_word = None
                found_word = False

            for n in range(w2v_size):
                if found_word:
                    feature_dict["w2v[-{}][{}]".format(i, n)] = w2v_word[n]
                else:
                    feature_dict["w2v[-{}][{}]".format(i, n)] = 0

    for i in range(n_after):
        if word_i < (len(abstract) - i):
            found_word = True

            word = abstract[word_i - i]

            try:
                w2v_word = w2v[word]
            except:
                w2v_word = None
                found_word = False

            for n in range(w2v_size):
                if found_word:
                    feature_dict["w2v[+{}][{}]".format(i, n)] = w2v_word[n]
                else:
                    feature_dict["w2v[+{}][{}]".format(i, n)] = 0

    return feature_dict


#def get_cosine_features(abstract, feature_dict, word_i):

def abstract2features(pmid_dict, pmids, words_before, words_after, w2v, shallow_parse, w2v_size=128):
    x, y = [], []

    for pmid in pmids:
        abstract, labels, tagged_sentences, _, _ = pmid_dict[pmid]
        string_labels = []
        features = []

        for i, (word, label, tags) in enumerate(zip(abstract, labels, tagged_sentences)):

            feature_dict = {}
            # Get features for current word
            feature_dict = get_features(i, 0, abstract, tagged_sentences, feature_dict, '', shallow_parse, w2v)

            # Get features for words around the current word
            feature_dict = get_feature_window(words_before, words_after, abstract, tagged_sentences, feature_dict, i,
                                              shallow_parse, w2v)

            features.append(feature_dict)

            string_labels.append(str(label))
        x.append(features)
        y.append(string_labels)

    return x, y




def run_crf(w2v, words_before, words_after, shallow_parse):

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
        train_x, train_y = abstract2features(pmids_dict, words_before, w2v, shallow_parse)
        test_x, test_y = abstract2features(pmids_dict, words_after, w2v, shallow_parse)

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
    words_before = 4
    words_after = 4
    shallow_parse = True

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'w:i:c:l:', ['w2v=', 'iters=', 'l1=', 'l2=', 'wiki=',
                                                              'words_before=', 'words_after=', 'shallow_parse='])
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
        elif opt == '--words_before':
            words_before = int(arg)
        elif opt == '--words_after':
            words_after = int(arg)
        elif opt == 'shallow_parse':
            option = int(arg)

            if option == 0:
                shallow_parse = False


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
    run_crf(w2v, words_before, words_after, shallow_parse)


def main():
    run()

if __name__ == '__main__':
    main()