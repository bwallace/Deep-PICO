from keras.models import Sequential

from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Activation

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras import backend as K

import sklearn.metrics as metrics
import numpy
import pickle
import theano
import theano.tensor as T

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

# @TODO fix this crappy naming
def load_model(model_info_path, model_path):
    model_info = pickle.load(open(model_info_path))
    model = GroupNN(build_model=False)
    model.build_model(window_size=model_info['window_size'],
                      word_vector_size=200,
                      activation_function=model_info['activation_function'],
                      dense_layer_sizes=model_info['dense_layer_sizes'],
                      dropout=model_info['dropout'],
                      k_output=model_info['k_output'],
                      hidden_dropout_rate=model_info['hidden_dropout_rate'],
                      name='nn')
    model.model.compile(optimizer=Adam(), loss='categorical_crossentropy')

    model.model.load_weights(model_path)

    return model


class GroupNN:
    def __init__(self, window_size=5, word_vector_size=200, activation_function='relu',
                 dense_layer_sizes=[], hidden_dropout_rate=.5, dropout=True, k=2, name='NNModel.hdf5',
                 build_model=True):
        if build_model:
            self.model = self.build_model(window_size, word_vector_size, activation_function, dense_layer_sizes,
                                          hidden_dropout_rate, dropout, k_output=k, name=name)

    def build_model(self, window_size, word_vector_size, activation_function, dense_layer_sizes,
                    hidden_dropout_rate, dropout, k_output, name):

        self.model_info = {}

        self.model_info['window_size'] = window_size
        self.model_info['activation_function'] = activation_function
        self.model_info['k_output'] = k_output
        self.model_info['dropout'] = dropout
        self.model_info['hidden_dropout_rate'] = hidden_dropout_rate
        self.model_info['dense_layer_sizes'] = dense_layer_sizes
        self.model_info['name'] = name
        self.model_info['word_vector_size'] = word_vector_size

        model = Sequential()
        model.add(Dense(100, input_dim=(window_size * 2 + 1) * word_vector_size))
        model.add(Activation(activation_function))

        for layer_size in dense_layer_sizes:
            model.add(Dense(layer_size))

            if dropout:
                model.add(Dropout(hidden_dropout_rate))

        model.add(Dense(k_output))
        model.add(Activation('softmax'))

        self.model = model

        return model

    def train(self, x, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy', save=True):

        if optim_algo == 'adam':
            optim_algo = Adam()
        else:
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        if criterion == 'binary_crossentropy':
            criterion = binary_crossentropy_with_ranking

        self.model_info['criterion'] = criterion
        self.model_info['optimizer'] = optim_algo

        self.model.compile(loss=criterion, optimizer=optim_algo)
        self.model.fit(x, y, nb_epoch=n_epochs)

        if save:
            pickle.dump(self.model_info, open(self.model_info['name'] + '.p', 'wb'))
            self.model.save_weights(self.model_info['name'])

    def test(self, x, y):
        predictions = self.model.predict_classes(x)

        if not self.model_info['k_output'] == 1:
            truth = numpy.argmax(y, axis=1)
        else:
            truth = predictions

        print "Predictions: {}".format(predictions)
        print "True: {}".format(y)

        accuracy = metrics.accuracy_score(truth, predictions)
        f1_score = metrics.f1_score(truth, predictions)
        precision = metrics.precision_score(truth, predictions)
        auc = metrics.roc_auc_score(truth, predictions)
        recall = metrics.recall_score(truth, predictions)

        return accuracy, f1_score, precision, auc, recall

    def predict_classes(self, x):
        predicted_classes = self.model.predict_classes(x)

        return predicted_classes

    def transform(self, x):
        a_function = self.model_info['activation_function']

        if a_function == 'relu':
            a_function = K.relu
        elif a_function == 'tanh':
            a_function = K.tanh

        output = x

        weights = self.model.layers[0].get_weights()
        W = weights[0]
        b = weights[1]

        return a_function(K.dot(output, W) + b).eval()
        # @TODO Generalize this to more than one layer!
        """

        for layer_i, layer in enumerate(self.model.layers):
            if not type(layer) is Dense:
                continue

            weights = layer.get_weights()

            W = weights[0]
            b = weights[1]

            output = a_function(K.dot(output, W) + b).eval()
            break


        return output
        """
