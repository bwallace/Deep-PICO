from keras.models import Sequential


from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Activation


from keras.optimizers import SGD
from keras.optimizers import Adam


import sklearn.metrics as metrics
import numpy

class GroupNN:
    def __init__(self, window_size, word_vector_size=200, activation_function='relu',
                 dense_layer_sizes=[], input_dropout_rate=.2, hidden_dropout_rate=.5, dropout=True, k=2):
        self.model = self.build_model(window_size, word_vector_size, activation_function, dense_layer_sizes, input_dropout_rate, hidden_dropout_rate,
                                      dropout, k_output=k)
        self.window_size = window_size

    def build_model(self, window_size, word_vector_size, activation_function, dense_layer_sizes, input_dropout_rate,
                    hidden_dropout_rate, dropout, k_output):

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

    def train(self, x, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy'):

        if optim_algo == 'adam':
            optim_algo = Adam()
        else:
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss=criterion, optimizer=optim_algo)
        self.model.fit(x, y, nb_epoch=n_epochs)

    def test(self, x, y):
        predictions = self.model.predict_classes(x)
        truth = numpy.argmax(y, axis=1)
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
