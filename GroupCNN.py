from keras.models import Sequential
from keras.models import Graph

from keras.layers import containers

from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten

from keras.layers.core import Activation

from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.layers.containers import Merge

from keras import backend as K

import numpy
import sklearn.metrics as metrics

# Taken from https://gist.github.com/jerheff/8cf06fe1df0695806456
# @TODO Maybe look into this more http://papers.nips.cc/paper/2518-auc-optimization-vs-error-rate-minimization.pdf
def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

    # next, build a rank loss

    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(y_pred_score * (y_true <1))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss


class GroupCNN:
    def __init__(self, window_size, n_feature_maps, k_output, word_vector_size=200, activation_function='relu',
                 filter_sizes=[2,3,5], dense_layer_sizes=[200], input_dropout_rate=.2,
                 hidden_dropout_rate=.5, dropout=True, conv_type=2, name='CNNModel'):
        self.model = self.build_model(window_size, n_feature_maps, word_vector_size, activation_function,
                                      filter_sizes, dense_layer_sizes, input_dropout_rate, hidden_dropout_rate,
                                      dropout, k_output)
        self.k_output = k_output
        self.model_name = name
        self.window_size = window_size

    def build_model(self, window_size, n_feature_maps, word_vector_size, activation_function,
        filter_sizes, dense_layer_sizes, input_dropout_rate, hidden_dropout_rate, dropout, k_output):

        print "Window size: {}".format(window_size)
        print "N Feature Maps: {}".format(n_feature_maps)
        print "Word vector size: {}".format(word_vector_size)

        model = Graph()

        model.add_input('data', input_shape=(1, window_size * 2 + 1, word_vector_size))

        for filter_size in filter_sizes:
            conv_layer = containers.Sequential()
            conv_layer.add(Convolution2D(n_feature_maps, filter_size, word_vector_size,
                                         input_shape=(1, window_size * 2 + 1, word_vector_size)))
            conv_layer.add(Activation(activation_function))
            conv_layer.add(MaxPooling2D(pool_size=(window_size * 2 + 1 - filter_size + 1, 1)))
            conv_layer.add(Flatten())

            model.add_node(conv_layer, name='filter_unit_' + str(filter_size), input='data')

        fully_connected_nn = containers.Sequential()

        fully_connected_nn.add(Dense(n_feature_maps * len(filter_sizes),
                                     input_dim=n_feature_maps * len(filter_sizes)))
        fully_connected_nn.add(Activation(activation_function))

        if dropout:
            fully_connected_nn.add(Dropout(hidden_dropout_rate))

        fully_connected_nn.add(Dense(k_output))
        fully_connected_nn.add(Activation('softmax'))

        model.add_node(fully_connected_nn, name='fully_connected_nn',
                       inputs=['filter_unit_' + str(n) for n in filter_sizes])
        model.add_output(name='nn_output', input='fully_connected_nn')

        return model

    def train(self, x, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy', save_model=True):

        if optim_algo == 'adam':
            optim_algo = Adam()
        else:
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        print criterion

        if criterion == 'binary_crossentropy':
            criterion = binary_crossentropy_with_ranking

        self.model.compile(loss={'nn_output': criterion}, optimizer=optim_algo)

        self.model.fit({'data': x, 'nn_output': y}, nb_epoch=n_epochs)

        if save_model:
            self.model.save_weights(self.model_name)

    def test(self, x, y):
        truth = []
#        if self.k_output == self.window_size:

        predictions = self.predict_classes(x)

        if self.k_output == 2:

            for i in range(y.shape[0]):
                if y[i, 1] == 1:
                    truth.append(1)
                else:
                    truth.append(0)
        else:
            truth = y

        print "Predictions: {}".format(predictions)
        print "True: {}".format(y)

        accuracy = metrics.accuracy_score(truth, predictions)
        f1_score = metrics.f1_score(truth, predictions)
        precision = metrics.precision_score(truth, predictions)
        auc = metrics.roc_auc_score(truth, predictions)
        recall = metrics.recall_score(truth, predictions)

        return accuracy, f1_score, precision, auc, recall

    # Graph model uses a dictionary as input so wrap the Keras function to makes it easier to
    # use!
    def predict_classes(self, x):
        predictions = self.model.predict({'data': x})

        if self.k_output > 1:
            predicted_classes = numpy.argmax(predictions['nn_output'], axis=1)
        else:
            predicted_classes = predictions['nn_output']

        return predicted_classes

