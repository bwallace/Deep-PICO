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
from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import GroupNN
import GroupCNNExperiment
import theano.tensor as T


def _transform_to_string(words, stop_words):
    words_string = ""

    for word in words:
        if word in stop_words:
            continue
        else:
            words_string += " " + word
    return words_string


def eveluate(predicted_mentions, true_mentions):
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    mentions = {}
    overlap = False

    for abs_pred, true_pred in zip(predicted_mentions, true_mentions):

        for mention in abs_pred:
            already_overlapped = False

            for true_mention in true_pred:
                intersection = list(set(mention) & set(true_mention))

                # Annotated mentions that do not match detected mentions are considered to be false negatives.
                if len(intersection) > 0:
                    # A detected mention is considered a match for an annotated mention if they consist of the same set
                    # of words or if the detected mention
                    #  overlaps the annotated one and the overlap is not a symbol or stop word
                    # If a detected mention overlaps multiple annotated mentions, it is considered to be a false positive

                    if already_overlapped:
                        false_positives += 1
                    else:

                        true_positives += 1

                    already_overlapped = True
                else:
                    false_negatives += 1


    #print "false negatives: {}".format(false_negatives)
    #print "true postitives: {}".format(true_positives)
    if not (true_positives + false_negatives) == 0:
        recall = float(true_positives)/float((true_positives + false_negatives))
    else:
        recall = 0
        print 'Error: divide by zero default to 0 for recall '
    if not true_positives + false_positives == 0:
        precision = float(true_positives) / float(true_positives + false_positives)
    else:
        precision = 0
        print 'Error: divide by zero default to 0 for precision'

    if not precision + recall == 0:
        f1_score = float(2 * precision * recall) / float(precision + recall)
    else:
        f1_score = 0
        print 'Error: divide by zero default to 0 for f1'


    return recall, precision, f1_score


def transform_features(nn_model, X):
    transformed_X = []

    for x in X:
        transformed_x = nn_model.transform(x)
        feature_dict = {}

        for i, value in enumerate(transformed_x):
            feature_dict['{}'.format(i)] = value

        transformed_X.append(feature_dict)

    return transformed_X


def _labels_to_str(labels):
    str_labels = []

    for abstract_labels in labels:
        str_abstract_label = []

        for label in abstract_labels:
            str_abstract_label.append(str(label))
        str_labels.append(str_abstract_label)

    return str_labels

def run_crf(w2v, l2, l1, iters, shallow_parse, words_before, words_after, grid_search,tacc, transfer_learning=False):

    pmids_dict, pmids, abstracts, lbls, vectorizer, groups_map, one_hot, dicts = \
        parse_summerscales.get_tokens_and_lbls(
                make_pmids_dict=True, sen=True, use_genia=shallow_parse, using_tacc=tacc)

    model = pycrfsuite.Trainer(verbose=False)
    all_pmids = pmids_dict.keys()
    n = len(all_pmids)
    n_folds = 5
    kf = KFold(n, random_state=1337, shuffle=True, n_folds=n_folds)
    fold_gi = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    model_type = 'nn'
    binary_cross_entropy = True

    for fold_idx, (train, test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        train_pmids = [all_pmids[pmid_idx] for pmid_idx in train]
        test_pmids  = [all_pmids[pmid_idx] for pmid_idx in test]
        print('loading data...')

        if transfer_learning:
            nn_model = GroupNN.load_model(model_path='NNModel.hdf5', model_info_path='NNModel.hdf5.p')
            window_size = nn_model.model_info['window_size']

            train_x, train_y = GroupCNNExperiment._prep_data(train_pmids, pmids_dict, w2v, window_size, model_type,
                                                             binary_ce=binary_cross_entropy, crf=True)
            test_x, test_y = GroupCNNExperiment._prep_data(test_pmids, pmids_dict, w2v, window_size, model_type,
                                                           binary_ce=binary_cross_entropy, crf=True)

            train_x = transform_features(nn_model, train_x)
            test_x = transform_features(nn_model, test_x)

            train_y = _labels_to_str(train_y)
            test_y = _labels_to_str(test_y)
        else:

            train_x, train_y = abstracts2features(pmids_dict, train_pmids, words_before, words_after, w2v, shallow_parse)
            test_x, test_y = abstracts2features(pmids_dict, test_pmids, words_before, words_after, w2v, shallow_parse)

        print('loaded data...')
        for x, y in zip(train_x, train_y):
            model.append(x, y)

        if grid_search:
            model.set_params({
                'c1': l1,   # coefficient for L1 penalty
                'c2': l2,  # coefficient for L2 penalty
                'max_iterations': iters,  # stop earlier

                # include transitions that are possible, but not observed
                'feature.possible_transitions': True
            })

            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=l1,
                c2=l2,
                max_iterations=iters,
                all_possible_transitions=False
            )

            params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
            }

            # use the same metric for evaluation
            f1_scorer = make_scorer(metrics.flat_f1_score,
                                    average='weighted', labels=test_y)


            # search
            rs = RandomizedSearchCV(crf, params_space,
                                    cv=3,
                                    verbose=1,
                                    n_jobs=-1,
                                    n_iter=50,
                                    scoring=f1_scorer)
            rs.fit(train_x, train_y)
            info = rs.best_estimator_.tagger_.info()
            tagger = rs.best_estimator_.tagger_
        else:
            model.set_params({
                'c1': l1,   # coefficient for L1 penalty
                'c2': l2,  # coefficient for L2 penalty
                'max_iterations': iters,  # stop earlier

                # include transitions that are possible, but not observed
                'feature.possible_transitions': True
            })
            model_name = 'model {}'.format(fold_idx)
            print('training model...')
            model.train(model_name)
            print('done...')
            tagger = pycrfsuite.Tagger()
            tagger.open(model_name)

            info = tagger.info()

        def print_transitions(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        print("Top likely transitions:")
        print_transitions(Counter(info.transitions).most_common(80))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(info.transitions).most_common()[-80:])

        def print_state_features(state_features):
            for (attr, label), weight in state_features:
                print("%0.6f %-6s %s" % (weight, label, attr))

        print("Top positive:")
        print_state_features(Counter(info.state_features).most_common(80))

        print("\nTop negative:")
        print_state_features(Counter(info.state_features).most_common()[-80:])

        g_i = []
        y_truths, predictions = [], []
        abstract_predicted_mentions, true_abstract_mentions = [], []

        for i, (pmid, x, y) in enumerate(zip(test_pmids, test_x, test_y)):
            print(pmid)
            abstract_words, abstract_labels, tagged_abstract, groups_dict, groups =  pmids_dict[pmid]

            vocab = groups_map[pmid]
            abstract_words, _, _, _, _ = pmids_dict[pmid]

            count = 0
            pred_labels = tagger.tag(x)
            pred_mentions = output2words(pred_labels, abstract_words)
            true_mentions = output2words(abstract_labels, abstract_words)

            print "Predicted: {}".format(pred_mentions)
            print "True: {}".format(true_mentions)
            print '\n'
            """
            abstract_predicted_words.append(pred_words)
          #  vocab2 = output2words(x, y, vectorizer, w2v, abstract_words)
            if len(test_y) > 0:
                y_truths.append(y)
                predictions.append(pred_labels)


            for pred_word in pred_words:
                for v_word in vocab:
                    if pred_word == v_word:
                        count += 1

            pred_words_string = ''
            true_words = " "
            for word in pred_words:
                pred_words_string = pred_words_string + " " + word
            for word in vocab:
                true_words = true_words + " " + word

            print("Predicted: {}".format(pred_words_string))
            print("True: {}".format(true_words))
            print ""
            if len(vocab) == 0:
                continue
            g_i1 = float(count)/float(len(vocab))

            g_i.append(g_i1)
            print LSTM_extraction._crf_evaluate_detection(y, pred_labels, )
            """
            abstract_predicted_mentions.append(pred_mentions)
            true_abstract_mentions.append(true_mentions)
        fold_recall, fold_precision, fold_f1_score = eveluate(abstract_predicted_mentions, true_abstract_mentions)
        recall_scores.append(fold_recall)
        precision_scores.append(fold_precision)
        f1_scores.append(fold_f1_score)

        fold_recall_results = "Fold recall: {}".format(fold_recall)
        fold_precision_results = "Fold precision: {}".format(fold_precision)
        fold_f1_results = "Fold F1 Score: {}".format(fold_f1_score)
        print fold_recall_results
        print fold_precision_results
        print fold_f1_results

        file = open('results.txt', 'w+')

        file.write(fold_recall_results + '\n')
        file.write(fold_precision_results + '\n')
        file.write(fold_f1_results + '\n')

       # avg_g_i = float(0)
        """
        for x in g_i:
            avg_g_i += x
        avg_g_i = avg_g_i/len(g_i)
        fold_gi.append(avg_g_i)
        print('g_i: {}'.format(avg_g_i))

    """
    """
        recall, precision, tp_overlapping_tokens, fp_tokens, accuracy = LSTM_extraction._crf_evaluate_detection(y_truths, predictions, abstract, vectorizer)

        print('Recall: {}'.format(recall))
        print('precision: {}'.format(precision))
        print('')

    avg_g_i = 0

    for g in fold_gi:
        avg_g_i+=g

    avg_g_i = float(avg_g_i)/float(len(fold_gi))

    avg_recall = 0

    for g in fold_gi:
        avg_recall+=g

    avg_recall = float(avg_recall)/float(len(recalls))

    avg_precision = 0

    for g in precisions:
        avg_precision+=g

    avg_precision = float(avg_precision)/float(len(precisions))
    "
    print("Fold avg g_i: {}".format(avg_g_i))
    """

    recall_average = _compute_average(recall_scores)
    precision_average = _compute_average(precision_scores)
    f1_scores = _compute_average(f1_scores)

    print "Recall Average: {}".format(recall_average)
    print "Precision Average: {}".format(precision_average)
    print "F1 Average: {}".format(f1_scores)

def _compute_average(scores):
    average = 0

    for score in scores:
        average += score
    average /= len(scores)

    return average


def output2words(labels, words):
    predicted_mention = []
    predicted_mentions = []
    stop_words = ['a', 'an', 'the', 'of', 'had', 'group', 'groups', 'arm', ',']

    mention = True
    for label, word in zip(labels, words):
        if label == '1' or label == 1:
            if word not in stop_words:
                predicted_mention.append(word)
                mention = True
        else:
            if mention:
                mention = False
                if len(predicted_mention) == 0:
                    continue
                predicted_mentions.append(predicted_mention)
                predicted_mention = []

    return predicted_mentions

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

def abstracts2features(pmid_dict, pmids, words_before, words_after, w2v, shallow_parse, w2v_size=128):
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



def run():
    w2v = False
    l1 = 1
    l2 = 1e-3
    iters = 200
    wiki = True
    shallow_parse = True
    words_before = 5
    words_after = 5
    grid_search = True
    tacc = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'w:i:c:l:',
                                   ['w2v=', 'iters=', 'l1=', 'l2=', 'wiki=', 'shallow_parse=', 'words_before=',
                                    'words_after=', 'grid_search=', 'tacc='])
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
        elif opt == '--shallow_parse':
            option = int(arg)

            if option == 0:
                shallow_parse = False
        elif opt == '--grid_search':
            option = int(arg)

            if option == 0:
                grid_search = False
        elif opt == '--words_before':
            words_before = int(arg)
        elif opt == '--words_after':
            words_after = int(arg)
        elif opt == '--tacc':
            if int(arg) == 1:
                tacc = True
        else:
            sys.exit(2)
    if w2v:
        print('Loading word2vec model...')

        if wiki:
            print 'Using wiki word2vec...'
            word2vec_model = 'wikipedia-pubmed-and-PMC-w2v.bin'
        else:
            print 'Using non-wiki word2vec...'
            word2vec_model = 'PubMed-w2v.bin'
        w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)
        print('Loaded word2vec model')
    else:
        w2v = False

    run_crf(w2v, l2, l1, iters, shallow_parse, words_before, words_after, grid_search, tacc)


def main():
    run()


if __name__ == '__main__':
    main()