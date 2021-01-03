import abc

class MyAIModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def predict(self, data, path):
        pass

