# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import numpy as np
import logging
from my_util.util import movingAve
from myAI.AI_interface import MyAIModel
##
## Categorize risingrate of nextday
## category: >.05, >0, >-.05, <-.05

class DNN04_model(MyAIModel):
    def __init__(self):
        ##batch size
        self.batch_size = 10
        ## Number of epochs
        self.epochs = 1000
        ##number of Neuron at Hidden layer
        self.n_mid = 30
        ##number of Neuron at Output layer
        self.n_out = 4
        ##Number of timeseries
        self.n_rnn = 5
        ##number of Neuron at Input layer
        self.n_in = 7 * self.n_rnn

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
        self.model.add(Dense(self.n_mid, activation="relu"))
        self.model.add(Dense(self.n_out, activation="softmax"))
        # self.model.add(Dense(self.n_out, activation="linear"))
        self.model.compile(loss="categorical_crossentropy", optimizer="sgd",
                           metrics=['accuracy'])
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
            # self.loss, self.accuracy = self.model.evaluate(x, t)
            self.loss, self.accuracy = self.model.evaluate(x, t)
            print("loss: " + str(self.loss) + " accuracy: " + str(self.accuracy))
            # predicted = self.model_lstm.predict(data[-self.n_rnn:])
            predicted_x = np.zeros((len(x)))
            predicted_del = np.zeros((len(x)))
            for i in range(len(x)):
                print("predicted " + str(len(predicted[i]))+ " "+ str(predicted[i,0]) + str(predicted[i,1]) + str(predicted[i,2]) + str(predicted[i,3]))
                for j in range(len(predicted[i])):
                    if j == 0:
                        pre = 0
                    elif predicted[i, pre] < predicted[i, j]:
                        pre = j
                print("pre = " + str(pre))
                predicted_x[i] = x[i, 3] * (1 + .03*(pre-2))
                predicted_del[i] = .03*(pre-2)
            # predicted = self.model_lstm.predict(data[-self.n_rnn:])
            if not path == "":
                np.savetxt(path, predicted_del)
            else:
                print("predicted_del " ,predicted_del)

            return True, predicted_x

    def make_dataset(self, data):

        logging.debug('Create dataset!')
        ##Number of sample
        n_sample = len(data) - self.n_rnn

        ##Input
        x = np.zeros((n_sample, self.n_in))

        ##answer
        t = np.zeros((n_sample, self.n_out))

        ##Onehot
        for i in range(n_sample):
            rising_rate=(data[i+self.n_rnn,3]-data[i+self.n_rnn,0])/data[i+self.n_rnn-1,3]
            if rising_rate > .05:
                t[i] = np.array((1, 0, 0, 0))
            elif rising_rate > 0:
                t[i] = np.array((0, 1, 0, 0))
            elif rising_rate > -.05:
                t[i] = np.array((0, 0, 1, 0))
            else :
                t[i] = np.array((0, 0, 0, 1))


        ##Calculate movingAve
        data = np.hstack([data, movingAve(data[:,-2].reshape(len(data),1), 75), movingAve(data[:,-2].reshape(len(data),1), 200)])

        ##Normalize
        ## Standerd data == Close (data[:,3])
        for i in range(len(data[0])):
            if not i == 4:
                sigma =np.sqrt(np.average((data[-1,3]-data[:,i])**2))
                data[:,i] = (data[-1,3]-data[:,i])/sigma
            else:
                sigma = np.sqrt(np.average((data[-1, i] - data[:, i]) ** 2))
                data[:,i] = (data[-1,i]-data[:,i])/sigma
            # sigma =np.sqrt(np.average((np.average(data[:,i])-data[:,i])**2))
            # data[:,i] = (np.average(data[:,i])-data[:,i])/sigma


        ##Create Input and Answer data
        for i in range(n_sample):
            for k in range(self.n_rnn):
                for j in range(len(data[0])):
                    x[i, j+len(data[0])*k] = data[i+k, j]


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
