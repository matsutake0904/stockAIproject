# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import logging


##
class LSTM_model():
    def __init__(self):
        ##batch size
        self.batch_size = 10
        ## Number of epochs
        self.epochs = 1000
        ##number of Neuron at Input layer
        self.n_in = 1
        ##number of Neuron at Hidden layer
        self.n_mid = 20
        ##number of Neuron at Output layer
        self.n_out = 1
        ##length of timeseries
        self.n_rnn = 50

        ##Create Model
        self.model = Sequential()

    ##Create LSTM Model
    def config_model(self):
        self.model.add(LSTM(self.n_mid, input_shape=(self.n_rnn, self.n_in), return_sequences=True))
        self.model.add(Dense(self.n_out, activation="linear"))
        self.model.compile(loss="mean_squared_error", metrics=['mean_squared_error'], optimizer="sgd")

    def train_model(self, data):
        if (len(data) < self.n_rnn):
            return False, 0
        else:
            logging.info('Training!')
            self.config_model()

            logging.debug('make dataset!')
            x, t = self.make_dataset(data)

            self.history = self.model.fit(x, t, epochs=self.epochs, batch_size=self.batch_size)

            return True, self.history.history['loss']

    def predict(self, data):
        if (len(data) < self.n_rnn):
            return False, 0
        else:
            predicted = self.model.predict(data[-self.n_rnn:].reshape(1, self.n_rnn, 1))
            # predicted = self.model.predict(data[-self.n_rnn:])
            print(data[-self.n_rnn:])

            return True, predicted

    def make_dataset(self, data):

        logging.debug('Create dataset!')
        ##Number of sample
        n_sample = len(data) - self.n_rnn

        ##Normalize
        for i in range(len(data[0])):
            sigma =np.sqrt(np.average((np.average(data[:,i])-data[:,i])**2))
            data[:,i] = (np.average(data[:,i])-data[:,i])/sigma


        ##Input
        x = np.zeros((n_sample, self.n_rnn))
        ##answer
        t = np.zeros((n_sample, self.n_rnn))

        ##Create Input and Answer data
        for i in range(0, n_sample):
            x[i] = data[i: i + self.n_rnn, 3]
            t[i] = data[i + 1: i + self.n_rnn + 1, 3]

        ## Sample, n_rnn, number of Neuron at Input layer
        x = x.reshape(n_sample, self.n_rnn, self.n_in).astype(np.float32)
        t = t.reshape(n_sample, self.n_rnn, self.n_in).astype(np.float32)

        logging.debug('Succsess to create dataset!')
        return x, t

# x_data = np.linspace(-2*np.pi, 2*np.pi)  # -2πから2πまで
# sin_data = np.sin(x_data) + 0.1*np.random.randn(len(x_data))  #
# logging.info("length "+str(len(sin_data)))
# model = LSTM_model()
# # model.config_model()
# boo,test=  model.train_model(sin_data)
# logging.debug(test)
# success, predict0=model.predict(sin_data)
# print(predict0)
