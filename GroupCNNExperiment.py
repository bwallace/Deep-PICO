import getopt
import sys
import parse_summerscales
import itertools
import numpy

from gensim.models import Word2Vec
from GroupCNN import GroupCNN

from sklearn.cross_validation import KFold

def main():
    n_folds = 5
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['window_size=', 'wiki=', 'n_feature_maps=', 'epochs=',
                                                      'undersample=', 'n_feature_maps=', 'criterion=',
                                                      'optimizer='])
    except getopt.GetoptError as error:
        print error
        sys.exit(2)

    window_size = 5
    wiki = True
    n_feature_maps = 100
    epochs = 20
    undersample = False
    binary_cross_entropy = False
    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    k = 2


    for opt, arg in opts:
        if opt == '--window_size':
            window_size = int(arg)
        elif opt == '--wiki':
            if arg == 0:
                wiki = False
        elif opt == '--epochs':
            epochs = int(arg)
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
                make_pmids_dict=True, sen=True)
    all_pmids = pmids_dict.keys()
    n = len(all_pmids)
    kf = KFold(n, random_state=1337, shuffle=True, n_folds=n_folds)

    for fold_idx, (train, test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        train_pmids = [all_pmids[pmid_idx] for pmid_idx in train]
        test_pmids  = [all_pmids[pmid_idx] for pmid_idx in test]
        print train_pmids
        print('loading data...')
        X_train, y_train = prep_data(train_pmids, pmids_dict, w2v, window_size, binary_ce=binary_cross_entropy)
        X_test, y_test = prep_data(test_pmids, pmids_dict, w2v, window_size, binary_ce=binary_cross_entropy)


        if undersample:
            # Undersample the non group tags at random....probably a bad idea...
            idx_undersample = numpy.where(y_train[:, 1] == 0)[0]
            idx_postive = numpy.where(y_train[:, 1] == 1)[0]
            random_negative_sample = numpy.random.choice(idx_undersample, idx_postive.shape[0])
            X_train_postive = X_train[idx_postive, :, :, :]
            y_train_postive = y_train[idx_postive, :]

            X_train_negative = X_train[random_negative_sample, :, :, :]
            y_train_negative = y_train[random_negative_sample, :]

            X_train = numpy.vstack((X_train_postive, X_train_negative))
            y_train = numpy.vstack((y_train_postive, y_train_negative))

        print('loaded data...')
        model = GroupCNN(window_size=window_size, n_feature_maps=n_feature_maps, k_output=k)
        model.train(X_train, y_train, epochs, optim_algo=optimizer, criterion=criterion)

        accuracy = model.test(X_test, y_test)

        print "Accuracy: {}".format(accuracy)


        sys.exit()
def _get_word_vector(word, word2vec, w2v_size=200):
    if word == "PADDING":
        word_vector = numpy.zeros((1, w2v_size))

    else:
        try:
            word_vector = word2vec[word]
        except:
            word_vector = numpy.zeros((1, w2v_size))

    return word_vector


def prep_data(pmids, pmid_dict, word2vec, window_size, w2v_size=200, binary_ce=False):
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

                        y[example_count + i, :] = target

                example_count += 1
                word_counter = 0

                example = numpy.zeros((1, window_size, w2v_size))
                target_window = []
    print "X shape: {}".format(X.shape)
    print "Y.shape: {}".format(y.shape)




    return X, y

if __name__ == '__main__':
    main()