# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import numpy as np
import logging
from myAI.AI_interface import MyAIModel


##
## Categorize risingrate of nextday
## category: >.05, <.05 & >-.05, <-.05

class DNN03_model(MyAIModel):
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
        self.n_out = 3
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
        self.model.add(Dense(self.n_mid, activation="relu"))
        self.model.add(Dense(self.n_mid, activation="relu"))
        self.model.add(Dense(self.n_out, activation="softmax"))
        # self.model.add(Dense(self.n_out, activation="linear"))
        self.model.compile(loss="mean_squared_error", optimizer="sgd",
                           metrics=['accuracy','mean_squared_error'])
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
            predicted_x = np.zeros((len(x)))
            predicted_del = np.zeros((len(x)))
            for i in range(len(x)):
                print("predicted " + str(len(predicted[i]))+ " "+ str(predicted[i,0]) + str(predicted[i,1]) + str(predicted[i,2]))
                for j in range(len(predicted[i])):
                    if j == 0:
                        pre = 0
                    elif predicted[i, pre] < predicted[i, j]:
                        pre = j
                print("pre = " + str(pre))
                predicted_x[i] = x[i, 3] * (1 + .05*(pre-1))
                predicted_del[i] = .05*(pre-1)
            # predicted = self.model_lstm.predict(data[-self.n_rnn:])
            if not path == "":
                np.savetxt(path, predicted_del)
            else:
                print("predicted_del " ,predicted_del)

            return True, predicted_x

    def make_dataset(self, data):

        logging.debug('Create dataset!')
        ##Number of sample
        n_sample = len(data) - 1

        ##Input
        x = np.zeros((n_sample, self.n_in))

        ##answer
        t = np.zeros((n_sample, self.n_out))

        ##Onehot
        for i in range(len(data)-1):
            rising_rate=(data[i+1,3]-data[i+1,0])/data[i,3]
            if rising_rate > .05:
                t[i] = np.array((1, 0, 0))
            elif rising_rate > -.05:
                t[i] = np.array((0, 1, 0))
            else :
                t[i] = np.array((0, 0, 1))

        ##Normalize
        for i in range(len(data[0])):
            sigma =np.sqrt(np.average((np.average(data[:,i])-data[:,i])**2))
            data[:,i] = (np.average(data[:,i])-data[:,i])/sigma



        ##Create Input and Answer data
        for i in range(n_sample):
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
