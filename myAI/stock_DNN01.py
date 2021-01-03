# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import numpy as np
import logging
from myAI.AI_interface import MyAIModel

##
class DNN01_model(MyAIModel):
    def __init__(self):
        ##batch size
        self.batch_size = 10
        ## Number of epochs
        self.epochs = 1000
        ##number of Neuron at Input layer
        self.n_in = 5
        ##number of Neuron at Hidden layer
        self.n_mid = 50
        ##number of Neuron at Output layer
        self.n_out = 1
        ##Number of timeseries
        self.n_rnn = 10

        ##Create Model
        self.model = Sequential()

    ##Create LSTM Model
    def config_model(self):
        # self.model_lstm.add(LSTM(self.n_mid, input_shape=(self.n_rnn, self.n_in), return_sequences=True))
        self.model.add(Dense(self.n_mid,input_dim=self.n_in))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.n_mid, activation="relu"))
        self.model.add(Dense(self.n_out, activation="linear"))
        # self.model.add(Dense(self.n_out, activation="linear"))
        self.model.compile(loss="mean_squared_error", optimizer="sgd",
                           metrics=['mean_squared_error'])
        print(self.model.summary())

    def train_model(self, data):
        if len(data) < self.n_rnn:
            return False, 0
        else:
            logging.info('Training!')
            self.config_model()

            logging.debug('make dataset!')
            x, t = self.make_dataset(data)
            x_test = x[-10:]
            t_test = t[-10:]

            self.history = self.model.fit(x, t, epochs=self.epochs, batch_size=self.batch_size,
                                          )
        # verbose=0 ⇦これを入れると学習過程が出力されない。

            return True, self.history.history['loss']

    def predict(self, data, path):
        if len(data) < self.n_rnn:
            return False, 0
        else:
            x, t = self.make_dataset(data)
            predicted = self.model.predict(x)
            # predicted = self.model_lstm.predict(data[-self.n_rnn:])

            return True, predicted

    def make_dataset(self, data):

        logging.debug('Create dataset!')
        ##Number of sample
        n_sample = len(data) - 1

        ##Normalize
        for i in range(len(data[0])):
            sigma =np.sqrt(np.average((np.average(data[:,i])-data[:,i])**2))
            data[:,i] = (np.average(data[:,i])-data[:,i])/sigma

        ##Input
        x = np.zeros((n_sample, self.n_in))

        ##answer
        t = np.zeros((n_sample, self.n_out))

        ##Create Input and Answer data
        for i in range(n_sample):
             t[i] = data[i + 1, 3]
             for j in range(5):
                x[i, j] = data[i, j]

        ## Sample, n_rnn, number of Neuron at Input layer
        x = x.reshape(n_sample, self.n_in)
        t = t.reshape(n_sample, self.n_out)

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
