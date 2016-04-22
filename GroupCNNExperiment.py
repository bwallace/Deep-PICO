import getopt
import sys
import parse_summerscales
import numpy
import crf
from GroupNN import GroupNN

from gensim.models import Word2Vec
from GroupCNN import GroupCNN

from sklearn.cross_validation import KFold

from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe


model = None

def main():
    n_folds = 5
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['window_size=', 'wiki=', 'n_feature_maps=', 'epochs=',
                                                      'undersample=', 'n_feature_maps=', 'criterion=',
                                                      'optimizer=', 'model=', 'genia=', 'tacc=', 'layers=',
                                                      'hyperopt='])
    except getopt.GetoptError as error:
        print error
        sys.exit(2)
    model_type = 'nn'
    window_size = 5
    wiki = True
    n_feature_maps = 100
    epochs = 20
    undersample = False
    binary_cross_entropy = False
    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    k = 2
    use_genia = False
    using_tacc = False
    layer_sizes = []
    hyperopt = False

    for opt, arg in opts:
        if opt == '--window_size':
            window_size = int(arg)
        elif opt == '--wiki':
            if arg == 0:
                wiki = False
        elif opt == '--epochs':
            epochs = int(arg)
        elif opt == '--layers':
            layer_sizes = arg.split(',')
        elif opt == '--n_feature_maps':
            n_feature_maps = int(arg)
        elif opt == '--undersample':
            option = int(arg)

            if option == 1:
                undersample = True

        elif opt == '--n_feature_maps':
            n_feature_maps = int(arg)
        elif opt == '--criterion':
            criterion = arg
        elif opt == '--optimizer':
            optimizer = arg
        elif opt == '--model':
            model_type = arg
        elif opt == '--genia':
            if int(arg) == 1:
                use_genia= True
        elif opt == '--tacc':
            if int(arg) == 1:
                using_tacc = True
        elif opt == '--hyperopt':
            if int(arg) == 1:
                hyperopt = True
        else:
            print "Option {} is not valid!".format(opt)

    if criterion == 'binary_crossentropy':
        binary_cross_entropy = True
        k = 1

    print('Loading word2vec model...')

    if wiki:
        print 'Using wiki word2vec...'
        word2vec_model = 'wikipedia-pubmed-and-PMC-w2v.bin'
    else:
        print 'Using non-wiki word2vec...'
        word2vec_model = 'PubMed-w2v.bin'
    w2v = Word2Vec.load_word2vec_format(word2vec_model, binary=True)
    print('Loaded word2vec model')

    pmids_dict, pmids, abstracts, lbls, vectorizer, groups_map, one_hot, dicts = \
        parse_summerscales.get_tokens_and_lbls(
                make_pmids_dict=True, sen=True, use_genia=use_genia, using_tacc=using_tacc)
    all_pmids = pmids_dict.keys()
    n = len(all_pmids)
    kf = KFold(n, random_state=1337, shuffle=True, n_folds=n_folds)

    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    aucs = []

    global model

    for fold_idx, (train, test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        train_pmids = [all_pmids[pmid_idx] for pmid_idx in train]
        test_pmids  = [all_pmids[pmid_idx] for pmid_idx in test]

        print train_pmids
        print('loading data...')

        if model_type == 'cnn':
            X_train, y_train = _prep_data(train_pmids, pmids_dict, w2v, window_size, model_type, binary_ce=binary_cross_entropy)
            X_test, y_test = _prep_data(test_pmids, pmids_dict, w2v, window_size, model_type,  binary_ce=binary_cross_entropy)
        elif model_type == 'nn':
            X_train, y_train = _prep_data(train_pmids, pmids_dict, w2v, window_size, model_type, binary_ce=binary_cross_entropy)
            X_test, y_test = _prep_data(test_pmids, pmids_dict, w2v, window_size, model_type, binary_ce=binary_cross_entropy)

        if undersample:
            # Undersample the non group tags at random....probably a bad idea...
            if binary_cross_entropy:
                idx_undersample = numpy.where(y_train == 0)[0]
                idx_postive = numpy.where(y_train == 1)[0]
            else:
                idx_undersample = numpy.where(y_train[:, 1] == 0)[0]
                idx_postive = numpy.where(y_train[:, 1] == 1)[0]
            random_negative_sample = numpy.random.choice(idx_undersample, idx_postive.shape[0])

            if model_type == 'nn':
                X_train_postive = X_train[idx_postive, :]
                X_train_negative = X_train[random_negative_sample, :]
            else:
                X_train_postive = X_train[idx_postive, :, :, :]

                X_train_negative = X_train[random_negative_sample, :, :, :]

            if binary_cross_entropy:
                y_train_postive = y_train[idx_postive]
                y_train_negative = y_train[random_negative_sample]
            else:
                y_train_postive = y_train[idx_postive, :]
                y_train_negative = y_train[random_negative_sample, :]


            X_train = numpy.vstack((X_train_postive, X_train_negative))

            if binary_cross_entropy:
                y_train = numpy.hstack((y_train_postive, y_train_negative))

            else:
                y_train = numpy.vstack((y_train_postive, y_train_negative))

        print('loaded data...')

        if model_type == 'cnn':
            model = GroupCNN(window_size=window_size, n_feature_maps=n_feature_maps, k_output=k)
        elif model_type == 'nn':
            model = GroupNN(window_size=window_size, k=k, hyperparameter_search=hyperopt)

        if hyperopt:
            best_run, best_model = optim.minimize(model=_model,
                                          data=_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
            model.model = best_model

        else:
            model.train(X_train, y_train, epochs, optim_algo=optimizer, criterion=criterion)

        words = []
        for pmid in test_pmids:
            words.extend(pmids_dict[pmid][0])

        predictions = model.predict_classes(X_test)

        predicted_words = crf.output2words(predictions, words)
        y_test_arg_max = numpy.argmax(y_test, axis=1)
        true_words = crf.output2words(y_test_arg_max, words)

        accuracy, f1_score, precision, auc, recall = model.test(X_test, y_test)
        recall, precision, f1_score = crf.eveluate(predicted_words, true_words)

        print "Accuracy: {}".format(accuracy)
        print "F1: {}".format(f1_score)
        print "Precision: {}".format(precision)
        print "AUC: {}".format(auc)
        print "Recall: {}".format(recall)

        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        precisions.append(precision)
        aucs.append(auc)
        recalls.append(recall)
    mean_accuracy = numpy.mean(accuracies)
    mean_f1_score = numpy.mean(f1_scores)
    mean_precision = numpy.mean(precisions)
    mean_auc_score = numpy.mean(aucs)
    mean_recall = numpy.mean(recalls)

    mean_accuracy_string = "Mean Accuracy: {}".format(mean_accuracy)
    mean_f1_score_string = "Mean F1: {}".format(mean_f1_score)
    mean_precision_string = "Mean Precision: {}".format(mean_precision)
    mean_auc_score_string = "Mean AUC: {}".format(mean_auc_score)
    mean_recall_string = "Mean Recall: {}".format(mean_recall)

    print mean_accuracy_string
    print mean_f1_score_string
    print mean_precision_string
    print mean_auc_score_string
    print mean_recall_string

    results = open('{}_fold_results'.format(model.model_name))
    results.write(mean_accuracy_string)
    results.write(mean_f1_score_string)
    results.write(mean_precision_string)
    results.write(mean_auc_score_string)
    results.write(mean_recall_string)




def _get_word_vector(word, word2vec, w2v_size=200):
    if word == "PADDING":
        word_vector = numpy.zeros((1, w2v_size))

    else:
        try:
            word_vector = word2vec[word]
        except:
            word_vector = numpy.zeros((1, w2v_size))

    return word_vector


def _prep_data(pmids, pmid_dict, word2vec, window_size, model_type, w2v_size=200, binary_ce=False, crf=False):
    n_examples = 0
    feature_size = (window_size * 2 + 1) * w2v_size

    # Get sizes and padding before procssing to make things fastert
    for pmid in pmids:
        abstract, labels, tagged_sentences, _, _ = pmid_dict[pmid]

        n = len(abstract)

        # You might wonder why check if the first word in the abstract is "padding"
        # well if I don't check then when going through the folds some abstracts will have twice the amount of padding
        # because the abstract will be modified twice.
        if not abstract[0] == "PADDING":
            padding = []

            for i in range(window_size):
                padding.append("PADDING")
            pmid_dict[pmid][0] = padding + abstract + padding

        n_examples += n
    if crf:
        X = []
        y = []
    else:
        if model_type == 'nn':
            X = numpy.zeros((n_examples, feature_size))
        elif model_type == 'cnn':
            X = numpy.zeros((n_examples, 1, window_size * 2 + 1, w2v_size))


        if binary_ce:
            y = numpy.zeros(n_examples)
        else:
            y = numpy.zeros((n_examples, 2))

    example_i = 0

    for abstract_i, pmid in enumerate(pmids):
        abstract, labels, tagged_sentences, _, _ = pmid_dict[pmid]

        n = len(abstract)
        abstract_features = []
        abstract_labels = []

        for i_abstract, i in enumerate(range(window_size, n - window_size)):

            if model_type == 'nn':
                example = numpy.zeros(feature_size)
            elif model_type == 'cnn':
                example = numpy.zeros((1, window_size * 2 + 1, w2v_size))

            for window_i, word_i in enumerate(range(i - window_size, i + window_size)):
                word = abstract[word_i]

                if model_type == 'nn':
                    example[window_i * w2v_size: (window_i+1) * w2v_size] = _get_word_vector(word, word2vec)
                elif model_type == 'cnn':
                    example[:, window_i, :] = _get_word_vector(word, word2vec)
            label = labels[i_abstract]

            if crf:
                abstract_features.append(example)
                abstract_labels.append(label)
            else:
                if model_type == 'nn':
                    X[example_i, :] = example
                elif model_type =='cnn':
                    X[example_i, :, :, :] = example


                if binary_ce:
                    y[example_i] = label
                else:
                    y[example_i, label] = 1
            example_i += 1
        if crf:
            X.append(abstract_features)
            y.append(abstract_labels)

    return X, y

def _cnn_prep_data(pmids, pmid_dict, word2vec, window_size, w2v_size=200, binary_ce=False):
    n_examples = 0

    # Get sizes and padding before procssing to make things fastert
    for pmid in pmids:
        abstract, labels, tagged_sentences, _, _ = pmid_dict[pmid]

        n = len(abstract)

        if n % window_size != 0:
            n_padding = (window_size - (n % window_size))

            n += n_padding

            for i in range(n_padding):
                abstract.append("PADDING")
                labels.append(0)
        n_examples += n

    X = numpy.zeros((n_examples * window_size, 1, window_size, w2v_size))

    if binary_ce:
        y = numpy.zeros((n_examples * window_size))
    else:
        y = numpy.zeros((n_examples * window_size, 2))

    for pmid in pmids:
        abstract, labels, tagged_sentences, _, _ = pmid_dict[pmid]

        word_counter = 0

        example = numpy.zeros((1, window_size, w2v_size))
        example_count = 0
        n = len(abstract)

        target_window = []

        print '\nNumber of Examples in abstract {}: {} | {} words per example'.format(pmid, n, window_size)

        for word, label in zip(abstract, labels):
            word_vector = _get_word_vector(word, word2vec)

            if word_counter < window_size:
                example[:, word_counter, :] = word_vector

                target_window.append(label)
                word_counter += 1
            else:
                for i, label in enumerate(target_window):
                    X[example_count + i, :, :, :] = example

                    if binary_ce:
                        y[example_count + i] = label
                    else:

                        target = numpy.zeros((1, 2))

                        if label == 1:
                            target[:, 1] = 1
                        else:
                            target[:, 0] = 1

                        y[example_count + i, :] = target

                example_count += 1
                word_counter = 0

                example = numpy.zeros((1, window_size, w2v_size))
                target_window = []
    print "X shape: {}".format(X.shape)
    print "Y.shape: {}".format(y.shape)

    return X, y

def _model(X_train, y_train, X_test, y_test):
    global model

    model.compile(loss='categorical_crossentropy', optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def _data(X_train, y_train, X_test, y_test):
    return





if __name__ == '__main__':
    main()
