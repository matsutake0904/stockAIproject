import json,codecs
import numpy as np
import tensorflow
def saveHist(path,history):

    new_hist = {}
    for key in list(history.history.keys()):
        # print("key = "+str(key))
        # print("key"+str(key)+"= "+ str(history.history[key]))
        # print("type of key")
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float:
               new_hist[key] = list(map(float, history.history[key]))

    print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)

def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n

def saveModel(path, model):
    path_model=path+"_model.json"
    path_weigth=path+"_weight.jdf5"
    ##save model structure
    with codecs.open(path_model, 'w', encoding='utf-8') as f:
        f.write(model.to_json())
    ##save model weight
    model.save_weights(path_weigth)


def movingAve(data, n_day):
    data_ma = np.zeros((len(data), len(data[0])))
    num_data = np.zeros((len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if j >= i - n_day and j <= i:
                for k in range(len(data[0])):
                    data_ma[i, k] += data[j, k]
                num_data[i] += 1

    for i in range(len(data)):
        for k in range(len(data[0])):
            data_ma[i,k] = data_ma[i,k]/num_data[i]

    return  data_ma