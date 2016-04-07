from keras.models import Sequential


from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Activation


from keras.optimizers import SGD
from keras.optimizers import Adam


import sklearn.metrics as metrics
import numpy
from keras import backend as K


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


class GroupNN:
    def __init__(self, window_size, word_vector_size=200, activation_function='relu',
                 dense_layer_sizes=[], input_dropout_rate=.2, hidden_dropout_rate=.5, dropout=True, k=2, name='NNModel.hdf5'):
        self.model = self.build_model(window_size, word_vector_size, activation_function, dense_layer_sizes, input_dropout_rate, hidden_dropout_rate,
                                      dropout, k_output=k)
        self.window_size = window_size
        self.k_output = k
        self.model_name = name

    def build_model(self, window_size, word_vector_size, activation_function, dense_layer_sizes, input_dropout_rate,
                    hidden_dropout_rate, dropout, k_output):

        print "Window size: {}".format(window_size)
        print "K output: {}".format(k_output)
        print "Dropout: {}".format(dropout)

        model = Sequential()
        model.add(Dense(100, input_dim=(window_size * 2 + 1) * word_vector_size))
        model.add(Activation(activation_function))

        if dropout:
            model.add(Dropout(input_dropout_rate))

        for layer_size in dense_layer_sizes:
            model.add(Dense(layer_size))

            if dropout:
                model.add(Dropout(hidden_dropout_rate))

        model.add(Dense(k_output))
        model.add(Activation('softmax'))

        return model

    def train(self, x, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy', save=True):

        if optim_algo == 'adam':
            optim_algo = Adam()
        else:
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        if criterion == 'binary_crossentropy':
            criterion = binary_crossentropy_with_ranking

        self.model.compile(loss=criterion, optimizer=optim_algo)
        self.model.fit(x, y, nb_epoch=n_epochs)

        if save:
            self.model.save_weights(self.model_name)

    def test(self, x, y):
        predictions = self.model.predict_classes(x)

        if not self.k_output == 1:
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

